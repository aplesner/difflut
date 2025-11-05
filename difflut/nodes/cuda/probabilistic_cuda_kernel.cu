#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized forward kernel - one block per batch element
__global__ void
probabilistic_cuda_forward_kernel(
    const float* __restrict__ input,        // (batch_size, input_dim)
    const float* __restrict__ raw_weights,  // (2^input_dim, output_dim)
    const float temperature,
    float* __restrict__ output,             // (batch_size, output_dim)
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {                   // lut_size = 2^input_dim
    
    // Shared memory for inputs and probabilities
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_probs = &shared_mem[input_dim];
    
    // Each block handles one batch element
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp_inv = 1.0f / fmaxf(temperature, 1e-6f);  // Precompute inverse
    
    // Load inputs (clamp to [0,1]) - coalesced access
    if (tid < input_dim) {
        const int input_offset = batch_idx * input_dim + tid;
        float val = input[input_offset];
        s_input[tid] = fmaxf(0.0f, fminf(1.0f, val));
    }
    
    __syncthreads();
    
    // Compute probabilities: Pr(a|x) = Π_j [x_j^a_j * (1-x_j)^(1-a_j)]
    // Optimized with reduced branching
    for (int addr = tid; addr < lut_size; addr += block_size) {
        float prob = 1.0f;
        #pragma unroll 4
        for (int l = 0; l < input_dim; ++l) {
            const int a_l = (addr >> l) & 1;  // LSB-first bit extraction
            const float x_l = s_input[l];
            // Branchless: prob *= (a_l ? x_l : (1-x_l))
            prob *= x_l * a_l + (1.0f - x_l) * (1 - a_l);
        }
        s_probs[addr] = prob;
    }
    
    __syncthreads();
    
    // Compute outputs: Σ_a σ(w_a / T) * Pr(a|x)
    for (int dim_idx = tid; dim_idx < output_dim; dim_idx += block_size) {
        float result = 0.0f;
        // Sequential access pattern - better cache utilization
        #pragma unroll 4
        for (int addr = 0; addr < lut_size; ++addr) {
            const float raw_weight = raw_weights[addr * output_dim + dim_idx];
            const float scaled_weight = raw_weight * temp_inv;  // Use precomputed inverse
            const float lut_weight = 1.0f / (1.0f + expf(-scaled_weight));  // sigmoid
            result += lut_weight * s_probs[addr];
        }
        const int output_offset = batch_idx * output_dim + dim_idx;
        output[output_offset] = result;
    }
}

torch::Tensor probabilistic_cuda_forward(
    torch::Tensor input_tensor,
    torch::Tensor raw_weights_tensor,
    torch::Tensor temperature_tensor) {
  
    // Input: (batch_size, input_dim)
    // raw_weights: (2^input_dim, output_dim)
    const int batch_size = input_tensor.size(0);
    const int input_dim = input_tensor.size(1);
    const int lut_size = raw_weights_tensor.size(0);  // 2^input_dim
    const int output_dim = raw_weights_tensor.size(1);
    
    float temperature = temperature_tensor.item<float>();

    auto output_tensor = torch::zeros({batch_size, output_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_tensor.device()));

    const size_t shared_mem_size = (input_dim + lut_size) * sizeof(float);
    const int threads = min(256, max(32, max(output_dim, lut_size)));
    const int blocks = batch_size;
    
    probabilistic_cuda_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        input_tensor.data_ptr<float>(),
        raw_weights_tensor.data_ptr<float>(),
        temperature,
        output_tensor.data_ptr<float>(),
        batch_size,
        input_dim,
        output_dim,
        lut_size
    );

    return output_tensor;
}


// Optimized backward input kernel
__global__ void
probabilistic_cuda_backward_input_kernel(
    const float* __restrict__ input,
    const float* __restrict__ raw_weights,
    const float temperature,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_grad_output = &shared_mem[input_dim];
    float* s_probs = &s_grad_output[output_dim];
    
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp_inv = 1.0f / fmaxf(temperature, 1e-6f);  // Precompute inverse
    
    // Load inputs (clamp to [0,1]) - coalesced access
    if (tid < input_dim) {
        const int input_offset = batch_idx * input_dim + tid;
        float val = input[input_offset];
        s_input[tid] = fmaxf(0.0f, fminf(1.0f, val));
    }
    
    // Load grad_output - coalesced access
    if (tid < output_dim) {
        const int grad_offset = batch_idx * output_dim + tid;
        s_grad_output[tid] = grad_output[grad_offset];
    }
    
    __syncthreads();
    
    // Compute and cache probabilities - optimized with reduced branching
    for (int addr = tid; addr < lut_size; addr += block_size) {
        float prob = 1.0f;
        #pragma unroll 4
        for (int l = 0; l < input_dim; ++l) {
            const int a_l = (addr >> l) & 1;
            const float x_l = s_input[l];
            // Branchless computation
            prob *= x_l * a_l + (1.0f - x_l) * (1 - a_l);
        }
        s_probs[addr] = prob;
    }
    
    __syncthreads();
    
    // Each thread computes gradient for one input dimension
    for (int input_idx = tid; input_idx < input_dim; input_idx += block_size) {
        float grad_sum = 0.0f;
        const float x_l = s_input[input_idx];
        const float x_l_inv = (x_l > 1e-6f) ? (1.0f / x_l) : 0.0f;  // Precompute inverse
        const float one_minus_x_inv = ((1.0f - x_l) > 1e-6f) ? (1.0f / (1.0f - x_l)) : 0.0f;
        
        #pragma unroll 2
        for (int dim_idx = 0; dim_idx < output_dim; dim_idx++) {
            const float grad_out = s_grad_output[dim_idx];
            
            #pragma unroll 4
            for (int addr = 0; addr < lut_size; ++addr) {
                const float prob = s_probs[addr];
                
                // Sigmoid - use precomputed inverse temperature
                const float raw_weight = raw_weights[addr * output_dim + dim_idx];
                const float scaled_weight = raw_weight * temp_inv;
                const float sigmoid_val = 1.0f / (1.0f + expf(-scaled_weight));
                
                // Derivative of probability w.r.t. this input - branchless
                const int a_l = (addr >> input_idx) & 1;
                const float dprob_dx = prob * (a_l * x_l_inv - (1 - a_l) * one_minus_x_inv);
                
                grad_sum += sigmoid_val * dprob_dx * grad_out;
            }
        }
        
        const int grad_input_offset = batch_idx * input_dim + input_idx;
        grad_input[grad_input_offset] = grad_sum;
    }
}

// Optimized backward weights kernel  
__global__ void
probabilistic_cuda_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ raw_weights,
    const float temperature,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    const float temp_inv = 1.0f / temp;  // Precompute inverse
    
    const int total_weights = lut_size * output_dim;
    
    // Each thread processes multiple weight elements
    for (int weight_idx = tid; weight_idx < total_weights; weight_idx += block_size) {
        const int addr = weight_idx / output_dim;
        const int dim_idx = weight_idx % output_dim;
        
        // Compute sigmoid and its derivative once (hoisted out of loop)
        const float raw_weight = raw_weights[weight_idx];
        const float scaled_weight = raw_weight * temp_inv;
        const float exp_neg = expf(-scaled_weight);
        const float sigmoid_val = 1.0f / (1.0f + exp_neg);
        const float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val) * temp_inv;
        
        float grad_sum = 0.0f;
        
        // Accumulate over batch
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // Compute probability for this batch element - optimized
            float prob = 1.0f;
            const int input_base = batch_idx * input_dim;
            
            #pragma unroll 4
            for (int l = 0; l < input_dim; ++l) {
                const int a_l = (addr >> l) & 1;
                float x_l = input[input_base + l];
                x_l = fmaxf(0.0f, fminf(1.0f, x_l));
                // Branchless computation
                prob *= x_l * a_l + (1.0f - x_l) * (1 - a_l);
            }
            
            const int grad_output_offset = batch_idx * output_dim + dim_idx;
            const float grad_out = grad_output[grad_output_offset];
            grad_sum += prob * grad_out;  // Multiply by sigmoid_grad outside loop
        }
        
        // Apply sigmoid gradient once to accumulated sum
        grad_weights[weight_idx] = grad_sum * sigmoid_grad;
    }
}

std::vector<torch::Tensor> probabilistic_cuda_backward(
    torch::Tensor input,
    torch::Tensor raw_weights,
    torch::Tensor temperature_tensor,
    torch::Tensor grad_output) {
  
    const int batch_size = input.size(0);
    const int input_dim = input.size(1);
    const int output_dim = grad_output.size(1);
    const int lut_size = raw_weights.size(0);
    
    float temperature = temperature_tensor.item<float>();
    
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(raw_weights);

    // Input gradients
    const size_t shared_mem_input = (input_dim + output_dim + lut_size) * sizeof(float);
    const int threads_input = min(256, max(32, input_dim));
    const int blocks_input = batch_size;
    
    probabilistic_cuda_backward_input_kernel<<<blocks_input, threads_input, shared_mem_input>>>(
        input.data_ptr<float>(),
        raw_weights.data_ptr<float>(),
        temperature,
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch_size,
        input_dim,
        output_dim,
        lut_size
    );
    
    // Weight gradients - use single block since we're accumulating over batch
    const int threads_weights = 256;
    const int blocks_weights = 1;
    
    probabilistic_cuda_backward_weights_kernel<<<blocks_weights, threads_weights>>>(
        input.data_ptr<float>(),
        raw_weights.data_ptr<float>(),
        temperature,
        grad_output.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        batch_size,
        input_dim,
        output_dim,
        lut_size
    );

    return {grad_input, grad_weights};
}
