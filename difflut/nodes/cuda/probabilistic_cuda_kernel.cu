#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Simplified forward kernel - one block per (batch, layer) pair
__global__ void
probabilistic_cuda_forward_kernel(
    const float* __restrict__ input,        // (batch_size, layer_size, input_dim)
    const float* __restrict__ raw_weights,  // (layer_size, 2^input_dim, output_dim)
    const float temperature,
    float* __restrict__ output,             // (batch_size, layer_size, output_dim)
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {                   // lut_size = 2^input_dim
    
    // Shared memory for inputs and probabilities
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_probs = &shared_mem[input_dim];
    
    // Each block handles one (batch, layer) pair
    const int batch_idx = blockIdx.x / layer_size;
    const int layer_idx = blockIdx.x % layer_size;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    
    // Load inputs (clamp to [0,1])
    if (tid < input_dim) {
        float val = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + tid];
        s_input[tid] = fmaxf(0.0f, fminf(1.0f, val));
    }
    
    __syncthreads();
    
    // Compute probabilities: Pr(a|x) = Π_j [x_j^a_j * (1-x_j)^(1-a_j)]
    for (int addr = tid; addr < lut_size; addr += block_size) {
        float prob = 1.0f;
        for (int l = 0; l < input_dim; ++l) {
            int a_l = (addr >> l) & 1;  // LSB-first bit extraction
            float x_l = s_input[l];
            prob *= (a_l == 1) ? x_l : (1.0f - x_l);
        }
        s_probs[addr] = prob;
    }
    
    __syncthreads();
    
    // Compute outputs: Σ_a σ(w_a / T) * Pr(a|x)
    const int weights_base = layer_idx * lut_size * output_dim;
    
    for (int dim_idx = tid; dim_idx < output_dim; dim_idx += block_size) {
        float result = 0.0f;
        for (int addr = 0; addr < lut_size; ++addr) {
            float raw_weight = raw_weights[weights_base + addr * output_dim + dim_idx];
            float lut_weight = 1.0f / (1.0f + expf(-raw_weight / temp));  // sigmoid
            result += lut_weight * s_probs[addr];
        }
        output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx] = result;
    }
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

    const size_t shared_mem_size = (input_dim + lut_size) * sizeof(float);
    const int threads = min(256, max(32, max(output_dim, lut_size)));
    const int blocks = batch_size * layer_size;
    
    probabilistic_cuda_forward_kernel<<<blocks, threads, shared_mem_size>>>(
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

    return output_tensor;
}


// Simplified backward input kernel
__global__ void
probabilistic_cuda_backward_input_kernel(
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
    
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_grad_output = &shared_mem[input_dim];
    float* s_probs = &s_grad_output[output_dim];
    
    const int batch_idx = blockIdx.x / layer_size;
    const int layer_idx = blockIdx.x % layer_size;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    
    // Load inputs (clamp to [0,1])
    if (tid < input_dim) {
        float val = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + tid];
        s_input[tid] = fmaxf(0.0f, fminf(1.0f, val));
    }
    
    // Load grad_output
    if (tid < output_dim) {
        s_grad_output[tid] = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + tid];
    }
    
    __syncthreads();
    
    // Compute and cache probabilities
    for (int addr = tid; addr < lut_size; addr += block_size) {
        float prob = 1.0f;
        for (int l = 0; l < input_dim; ++l) {
            int a_l = (addr >> l) & 1;
            float x_l = s_input[l];
            prob *= (a_l == 1) ? x_l : (1.0f - x_l);
        }
        s_probs[addr] = prob;
    }
    
    __syncthreads();
    
    const int weights_base = layer_idx * lut_size * output_dim;
    
    // Each thread computes gradient for one input dimension
    for (int input_idx = tid; input_idx < input_dim; input_idx += block_size) {
        float grad_sum = 0.0f;
        const float x_l = s_input[input_idx];
        
        for (int dim_idx = 0; dim_idx < output_dim; dim_idx++) {
            const float grad_out = s_grad_output[dim_idx];
            
            for (int addr = 0; addr < lut_size; ++addr) {
                float prob = s_probs[addr];
                
                // Sigmoid
                float raw_weight = raw_weights[weights_base + addr * output_dim + dim_idx];
                float sigmoid_val = 1.0f / (1.0f + expf(-raw_weight / temp));
                
                // Derivative of probability w.r.t. this input
                int a_l = (addr >> input_idx) & 1;
                
                float dprob_dx = 0.0f;
                if (a_l == 1) {
                    if (x_l > 1e-6f) dprob_dx = prob / x_l;
                } else {
                    if ((1.0f - x_l) > 1e-6f) dprob_dx = -prob / (1.0f - x_l);
                }
                
                grad_sum += sigmoid_val * dprob_dx * grad_out;
            }
        }
        
        grad_input[batch_idx * layer_size * input_dim + layer_idx * input_dim + input_idx] = grad_sum;
    }
}

// Simplified backward weights kernel
__global__ void
probabilistic_cuda_backward_weights_kernel(
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
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    
    // Each block handles one layer
    const int layer_idx = blockIdx.x;
    if (layer_idx >= layer_size) return;
    
    const int total_weights = lut_size * output_dim;
    const int weights_base = layer_idx * total_weights;
    
    // Each thread processes multiple weight elements
    for (int weight_idx = tid; weight_idx < total_weights; weight_idx += block_size) {
        const int addr = weight_idx / output_dim;
        const int dim_idx = weight_idx % output_dim;
        
        float grad_sum = 0.0f;
        
        // Compute sigmoid and its derivative
        float raw_weight = raw_weights[weights_base + weight_idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-raw_weight / temp));
        float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val) / temp;
        
        // Accumulate over batch
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // Compute probability for this batch element
            float prob = 1.0f;
            for (int l = 0; l < input_dim; ++l) {
                int a_l = (addr >> l) & 1;
                float x_l = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + l];
                x_l = fmaxf(0.0f, fminf(1.0f, x_l));
                prob *= (a_l == 1) ? x_l : (1.0f - x_l);
            }
            
            float grad_out = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx];
            grad_sum += prob * sigmoid_grad * grad_out;
        }
        
        grad_weights[weights_base + weight_idx] = grad_sum;
    }
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

    // Input gradients
    const size_t shared_mem_input = (input_dim + output_dim + lut_size) * sizeof(float);
    const int threads_input = min(256, max(32, input_dim));
    const int blocks_input = batch_size * layer_size;
    
    probabilistic_cuda_backward_input_kernel<<<blocks_input, threads_input, shared_mem_input>>>(
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
    
    // Weight gradients
    const int threads_weights = 256;
    const int blocks_weights = layer_size;
    
    probabilistic_cuda_backward_weights_kernel<<<blocks_weights, threads_weights>>>(
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

    return {grad_input, grad_weights};
}
