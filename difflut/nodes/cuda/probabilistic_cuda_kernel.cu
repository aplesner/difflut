#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

// Warp-level reduction for sum
__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized CUDA kernel for Probabilistic forward pass
// Uses shared memory and improved parallelization strategy
// Each block processes one (batch, layer) pair for all output dims
// Threads collaborate to compute probabilities once, then apply to all outputs
__global__ void probabilistic_cuda_forward_kernel_optimized(
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
    float* s_input = shared_mem;                           // input_dim elements
    float* s_probs = &shared_mem[input_dim];               // lut_size elements
    float* s_weights = &s_probs[lut_size];                 // lut_size * output_dim elements
    
    // Each block handles one (batch, layer) pair
    const int batch_idx = blockIdx.x / layer_size;
    const int layer_idx = blockIdx.x % layer_size;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    
    // Step 1: Cooperatively load inputs into shared memory
    if (tid < input_dim) {
        s_input[tid] = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + tid];
        // Clamp to [0, 1]
        s_input[tid] = fmaxf(0.0f, fminf(1.0f, s_input[tid]));
    }
    
    // Step 2: Cooperatively load weights into shared memory
    const int weights_offset = layer_idx * lut_size * output_dim;
    for (int i = tid; i < lut_size * output_dim; i += block_size) {
        s_weights[i] = raw_weights[weights_offset + i];
    }
    
    __syncthreads();
    
    // Step 3: Compute probabilities for all LUT entries (parallelized)
    // Each thread computes probabilities for multiple LUT entries
    for (int addr = tid; addr < lut_size; addr += block_size) {
        float prob = 1.0f;
        
        // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
        for (int l = 0; l < input_dim; ++l) {
            int a_l = (addr >> l) & 1;  // LSB first
            float x_l = s_input[l];
            
            prob *= (a_l == 1) ? x_l : (1.0f - x_l);
        }
        
        s_probs[addr] = prob;
    }
    
    __syncthreads();
    
    // Step 4: Compute outputs for all dimensions (parallelized)
    // Each thread computes one or more output dimensions
    for (int dim_idx = tid; dim_idx < output_dim; dim_idx += block_size) {
        float result = 0.0f;
        
        // Sum over all LUT entries
        for (int addr = 0; addr < lut_size; ++addr) {
            // Apply sigmoid to LUT weight with temperature scaling
            float raw_weight = s_weights[addr * output_dim + dim_idx];
            float lut_weight = 1.0f / (1.0f + expf(-raw_weight / temp));
            
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

    // Use optimized kernel with shared memory
    // Shared memory needed: input_dim + lut_size + (lut_size * output_dim) floats
    const size_t shared_mem_size = (input_dim + lut_size + lut_size * output_dim) * sizeof(float);
    const int threads = min(256, max(32, max(output_dim, lut_size)));
    const int blocks = batch_size * layer_size;
    
    probabilistic_cuda_forward_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
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

// Optimized CUDA kernel for input gradients using shared memory
// Each block processes one (batch, layer) pair
__global__ void probabilistic_cuda_backward_input_kernel_optimized(
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
    float* s_weights = &s_grad_output[output_dim];
    
    const int batch_idx = blockIdx.x / layer_size;
    const int layer_idx = blockIdx.x % layer_size;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    
    // Load inputs
    if (tid < input_dim) {
        s_input[tid] = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + tid];
        s_input[tid] = fmaxf(0.0f, fminf(1.0f, s_input[tid]));
    }
    
    // Load grad_output
    if (tid < output_dim) {
        s_grad_output[tid] = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + tid];
    }
    
    // Load weights
    const int weights_offset = layer_idx * lut_size * output_dim;
    for (int i = tid; i < lut_size * output_dim; i += block_size) {
        s_weights[i] = raw_weights[weights_offset + i];
    }
    
    __syncthreads();
    
    // Each thread computes gradient for one input dimension
    for (int input_idx = tid; input_idx < input_dim; input_idx += block_size) {
        float grad_sum = 0.0f;
        
        // For each output dimension
        for (int dim_idx = 0; dim_idx < output_dim; dim_idx++) {
            const float grad_out = s_grad_output[dim_idx];
            
            // For each binary combination
            for (int addr = 0; addr < lut_size; ++addr) {
                // Compute probability
                float prob = 1.0f;
                for (int l = 0; l < input_dim; ++l) {
                    int a_l = (addr >> l) & 1;
                    prob *= (a_l == 1) ? s_input[l] : (1.0f - s_input[l]);
                }
                
                // Get sigmoid value
                float raw_weight = s_weights[addr * output_dim + dim_idx];
                float sigmoid_val = 1.0f / (1.0f + expf(-raw_weight / temp));
                
                // Compute derivative
                int a_l = (addr >> input_idx) & 1;
                float x_l = s_input[input_idx];
                
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
}

// Optimized CUDA kernel for weights gradients
// Each block processes multiple weight elements, using shared memory for batch reduction
__global__ void probabilistic_cuda_backward_weights_kernel_optimized(
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
    
    extern __shared__ float shared_mem[];
    
    // Each block handles one (layer, addr, dim) combination
    const int layer_idx = blockIdx.x / (lut_size * output_dim);
    const int remainder = blockIdx.x % (lut_size * output_dim);
    const int addr = remainder / output_dim;
    const int dim_idx = remainder % output_dim;
    
    if (layer_idx >= layer_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float temp = fmaxf(temperature, 1e-6f);
    
    // Each thread processes a subset of the batch
    float local_grad_sum = 0.0f;
    
    for (int batch_idx = tid; batch_idx < batch_size; batch_idx += block_size) {
        // Compute probability for this address
        float prob = 1.0f;
        for (int l = 0; l < input_dim; ++l) {
            int a_l = (addr >> l) & 1;
            float x_l = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + l];
            x_l = fmaxf(0.0f, fminf(1.0f, x_l));
            
            prob *= (a_l == 1) ? x_l : (1.0f - x_l);
        }
        
        // Compute sigmoid derivative
        float raw_weight = raw_weights[layer_idx * lut_size * output_dim + addr * output_dim + dim_idx];
        float exp_neg = expf(-raw_weight / temp);
        float sigmoid_val = 1.0f / (1.0f + exp_neg);
        float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val) / temp;
        
        // Get gradient output
        const float grad_out = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx];
        
        local_grad_sum += prob * sigmoid_grad * grad_out;
    }
    
    // Store in shared memory for reduction
    shared_mem[tid] = local_grad_sum;
    __syncthreads();
    
    // Reduce within block using tree reduction
    for (int stride = block_size / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) {
        float val = shared_mem[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            grad_weights[layer_idx * lut_size * output_dim + addr * output_dim + dim_idx] = val;
        }
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

    // Use optimized kernels
    const size_t shared_mem_input = (input_dim + output_dim + lut_size * output_dim) * sizeof(float);
    const size_t shared_mem_weights = 256 * sizeof(float);  // For reduction
    
    // Launch input gradient kernel
    const int threads_input = min(256, max(32, input_dim));
    const int blocks_input = batch_size * layer_size;
    
    probabilistic_cuda_backward_input_kernel_optimized<<<blocks_input, threads_input, shared_mem_input>>>(
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
    const int threads_weights = min(256, max(32, batch_size));
    const int blocks_weights = layer_size * lut_size * output_dim;
    
    probabilistic_cuda_backward_weights_kernel_optimized<<<blocks_weights, threads_weights, shared_mem_weights>>>(
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
