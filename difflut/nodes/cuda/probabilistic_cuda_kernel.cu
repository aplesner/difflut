#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

// CUDA kernel for Probabilistic forward pass with per-layer-node parameters
// Forward: Continuous probabilistic computation
// Input: (batch_size, layer_size, input_dim)
// Parameters: raw_weights (layer_size, 2^input_dim, output_dim)
// Output: (batch_size, layer_size, output_dim)
__global__ void probabilistic_cuda_forward_kernel(
    const float* __restrict__ input,        // (batch_size, layer_size, input_dim)
    const float* __restrict__ raw_weights,  // (layer_size, 2^input_dim, output_dim)
    const float temperature,
    float* __restrict__ output,             // (batch_size, layer_size, output_dim)
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {                   // lut_size = 2^input_dim
    
    // Each thread handles one (batch, layer, output_dim) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * output_dim;
    
    if (idx >= total_elements) return;
    
    const int batch_idx = idx / (layer_size * output_dim);
    const int remainder = idx % (layer_size * output_dim);
    const int layer_idx = remainder / output_dim;
    const int dim_idx = remainder % output_dim;
    
    if (batch_idx >= batch_size || layer_idx >= layer_size || dim_idx >= output_dim) return;
    
    const float temp = fmaxf(temperature, 1e-6f);
    float result = 0.0f;
    
    // Iterate over all 2^n binary combinations
    for (int addr = 0; addr < lut_size; ++addr) {
        // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
        float prob = 1.0f;
        
        for (int l = 0; l < input_dim; ++l) {
            // LSB-first bit ordering (consistent across all nodes)
            int a_l = (addr >> l) & 1;  // LSB first
            float x_l = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + l];
            
            // Clamp input to [0, 1] for numerical stability
            x_l = fmaxf(0.0f, fminf(1.0f, x_l));
            
            // prob *= x_l^a_l * (1-x_l)^(1-a_l)
            if (a_l == 1) {
                prob *= x_l;
            } else {
                prob *= (1.0f - x_l);
            }
        }
        
        // Apply sigmoid to LUT weight with temperature scaling (per-layer-node)
        float raw_weight = raw_weights[layer_idx * lut_size * output_dim + addr * output_dim + dim_idx];
        float lut_weight = 1.0f / (1.0f + expf(-raw_weight / temp));
        
        // Accumulate: result += lut_weight * prob
        result += lut_weight * prob;
    }
    
    output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx] = result;
}

torch::Tensor probabilistic_cuda_forward(
    torch::Tensor input_tensor,
    torch::Tensor raw_weights_tensor,
    torch::Tensor temperature_tensor) {
  
    // Input: (batch_size, layer_size, input_dim)
    // raw_weights: (layer_size, 2^input_dim, output_dim)
    const int batch_size = input_tensor.size(0);
    const int layer_size = input_tensor.size(1);
    const int input_dim = input_tensor.size(2);
    const int lut_size = raw_weights_tensor.size(1);  // 2^input_dim
    const int output_dim = raw_weights_tensor.size(2);
    
    float temperature = temperature_tensor.item<float>();

    auto output_tensor = torch::zeros({batch_size, layer_size, output_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_tensor.device()));

    const int threads = 256;
    const int total_elements = batch_size * layer_size * output_dim;
    const int blocks = (total_elements + threads - 1) / threads;

    probabilistic_cuda_forward_kernel<<<blocks, threads>>>(
        input_tensor.data_ptr<float>(),
        raw_weights_tensor.data_ptr<float>(),
        temperature,
        output_tensor.data_ptr<float>(),
        batch_size,
        layer_size,
        input_dim,
        output_dim,
        lut_size
    );

    cudaDeviceSynchronize();

    return output_tensor;
}

// CUDA kernel for Probabilistic backward pass - input gradients
// Input: (batch_size, layer_size, input_dim)
// Grad output: (batch_size, layer_size, output_dim)
__global__ void probabilistic_cuda_backward_input_kernel(
    const float* __restrict__ input,
    const float* __restrict__ raw_weights,
    const float temperature,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    // Each thread handles one (batch, layer, input) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * input_dim;
    
    if (idx >= total_elements) return;
    
    const int batch_idx = idx / (layer_size * input_dim);
    const int remainder = idx % (layer_size * input_dim);
    const int layer_idx = remainder / input_dim;
    const int input_idx = remainder % input_dim;
    
    if (batch_idx >= batch_size || layer_idx >= layer_size || input_idx >= input_dim) return;
    
    const float temp = fmaxf(temperature, 1e-6f);
    float grad_sum = 0.0f;
    
    // For each output dimension
    for (int dim_idx = 0; dim_idx < output_dim; dim_idx++) {
        const float grad_out = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx];
        
        // For each binary combination
        for (int addr = 0; addr < lut_size; ++addr) {
            // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
            float prob = 1.0f;
            for (int l = 0; l < input_dim; ++l) {
                int a_l = (addr >> l) & 1;
                float x_l = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + l];
                x_l = fmaxf(0.0f, fminf(1.0f, x_l));
                
                if (a_l == 1) {
                    prob *= x_l;
                } else {
                    prob *= (1.0f - x_l);
                }
            }
            
            // Get sigmoid value (per-layer-node)
            float raw_weight = raw_weights[layer_idx * lut_size * output_dim + addr * output_dim + dim_idx];
            float sigmoid_val = 1.0f / (1.0f + expf(-raw_weight / temp));
            
            // Gradient w.r.t. inputs: d(output)/d(x_l) = sigmoid(raw_weight) * d(prob)/d(x_l) * grad_out
            int a_l = (addr >> input_idx) & 1;
            float x_l = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + input_idx];
            x_l = fmaxf(0.0f, fminf(1.0f, x_l));
            
            // d(prob)/d(x_l):
            // If a_l == 1: prob/x_l (derivative of x_l term)
            // If a_l == 0: -prob/(1-x_l) (derivative of (1-x_l) term)
            float dprob_dx = 0.0f;
            if (a_l == 1) {
                if (x_l > 1e-6f) {
                    dprob_dx = prob / x_l;
                }
            } else {
                if ((1.0f - x_l) > 1e-6f) {
                    dprob_dx = -prob / (1.0f - x_l);
                }
            }
            
            grad_sum += sigmoid_val * dprob_dx * grad_out;
        }
    }
    
    grad_input[batch_idx * layer_size * input_dim + layer_idx * input_dim + input_idx] = grad_sum;
}

// CUDA kernel for Probabilistic backward pass - weights gradients
// Parameters: (layer_size, lut_size, output_dim)
// Grad output: (batch_size, layer_size, output_dim)
__global__ void probabilistic_cuda_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ raw_weights,
    const float temperature,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    // Each thread handles one (layer, lut_entry, output_dim) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = layer_size * lut_size * output_dim;
    
    if (idx >= total_elements) return;
    
    const int layer_idx = idx / (lut_size * output_dim);
    const int remainder = idx % (lut_size * output_dim);
    const int addr = remainder / output_dim;
    const int dim_idx = remainder % output_dim;
    
    if (layer_idx >= layer_size || addr >= lut_size || dim_idx >= output_dim) return;
    
    const float temp = fmaxf(temperature, 1e-6f);
    float grad_sum = 0.0f;
    
    // Sum over batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
        float prob = 1.0f;
        for (int l = 0; l < input_dim; ++l) {
            int a_l = (addr >> l) & 1;
            float x_l = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + l];
            x_l = fmaxf(0.0f, fminf(1.0f, x_l));
            
            if (a_l == 1) {
                prob *= x_l;
            } else {
                prob *= (1.0f - x_l);
            }
        }
        
        // Compute sigmoid derivative (per-layer-node)
        float raw_weight = raw_weights[layer_idx * lut_size * output_dim + addr * output_dim + dim_idx];
        float exp_neg = expf(-raw_weight / temp);
        float sigmoid_val = 1.0f / (1.0f + exp_neg);
        float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val) / temp;
        
        // Gradient w.r.t. raw_weights: d(output)/d(raw_weight) = prob * sigmoid'(raw_weight) * grad_out
        const float grad_out = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx];
        grad_sum += prob * sigmoid_grad * grad_out;
    }
    
    grad_weights[layer_idx * lut_size * output_dim + addr * output_dim + dim_idx] = grad_sum;
}

std::vector<torch::Tensor> probabilistic_cuda_backward(
    torch::Tensor input,
    torch::Tensor raw_weights,
    torch::Tensor temperature_tensor,
    torch::Tensor grad_output) {
  
    const int batch_size = input.size(0);
    const int layer_size = input.size(1);
    const int input_dim = input.size(2);
    const int output_dim = grad_output.size(2);
    const int lut_size = raw_weights.size(1);
    
    float temperature = temperature_tensor.item<float>();
    
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(raw_weights);

    const int threads = 256;
    
    // Launch input gradient kernel
    int total_elements_input = batch_size * layer_size * input_dim;
    int blocks_input = (total_elements_input + threads - 1) / threads;
    
    probabilistic_cuda_backward_input_kernel<<<blocks_input, threads>>>(
        input.data_ptr<float>(),
        raw_weights.data_ptr<float>(),
        temperature,
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch_size,
        layer_size,
        input_dim,
        output_dim,
        lut_size
    );
    
    // Launch weights gradient kernel
    int total_elements_weights = layer_size * lut_size * output_dim;
    int blocks_weights = (total_elements_weights + threads - 1) / threads;
    
    probabilistic_cuda_backward_weights_kernel<<<blocks_weights, threads>>>(
        input.data_ptr<float>(),
        raw_weights.data_ptr<float>(),
        temperature,
        grad_output.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        batch_size,
        layer_size,
        input_dim,
        output_dim,
        lut_size
    );

    cudaDeviceSynchronize();

    return {grad_input, grad_weights};
}
