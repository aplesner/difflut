#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

// CUDA kernel for EFD forward pass with 3D tensors
// Input: (batch_size, layer_size, input_dim)
// LUTs: (layer_size, output_dim, 2^input_dim)
// Output: (batch_size, layer_size, output_dim)
__global__ void efd_cuda_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ luts,
    float* __restrict__ output,
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    // Each thread handles one (batch, layer, output) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * output_dim;
    
    if (idx >= total_elements) return;
    
    const int batch_idx = idx / (layer_size * output_dim);
    const int remainder = idx % (layer_size * output_dim);
    const int layer_idx = remainder / output_dim;
    const int dim_idx = remainder % output_dim;
    
    // Compute LUT address from binary input
    uint addr = 0;
    for (int i = 0; i < input_dim; ++i) {
        float x_val = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + i];
        if (x_val >= 0.5f) {
            addr |= (1u << i);
        }
    }
    
    // Look up LUT value (per-layer-node)
    float lut_val = luts[layer_idx * output_dim * lut_size + dim_idx * lut_size + addr];
    
    output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx] = lut_val;
}

torch::Tensor efd_cuda_forward(
    torch::Tensor input,
    torch::Tensor luts) {
  
    const int batch_size = input.size(0);
    const int layer_size = input.size(1);
    const int input_dim = input.size(2);
    const int output_dim = luts.size(1);
    const int lut_size = luts.size(2);

    auto output = torch::zeros({batch_size, layer_size, output_dim}, input.options());
    
    const int threads = 256;
    const int total_elements = batch_size * layer_size * output_dim;
    const int blocks = (total_elements + threads - 1) / threads;

    efd_cuda_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        luts.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        layer_size,
        input_dim,
        output_dim,
        lut_size
    );

    cudaDeviceSynchronize();

    return output;
};

// CUDA kernel for EFD backward pass - input gradients
// Input: (batch_size, layer_size, input_dim)
// LUTs: (layer_size, output_dim, 2^input_dim)
// Grad output: (batch_size, layer_size, output_dim)
__global__ void efd_cuda_backward_input_kernel(
    const float* __restrict__ input,
    const float* __restrict__ luts,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const float alpha,
    const float beta,
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    // Each thread handles one (batch, layer, input_dim) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * input_dim;
    
    if (idx >= total_elements) return;
    
    const int batch_idx = idx / (layer_size * input_dim);
    const int remainder = idx % (layer_size * input_dim);
    const int layer_idx = remainder / input_dim;
    const int input_idx = remainder % input_dim;
    
    // Compute current address from binary input
    uint addr = 0;
    for (int i = 0; i < input_dim; ++i) {
        float x_val = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + i];
        if (x_val >= 0.5f) {
            addr |= (1u << i);
        }
    }
    
    float grad_sum = 0.0f;
    
    // For each output dimension
    for (int dim_idx = 0; dim_idx < output_dim; ++dim_idx) {
        const float grad_out = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx];
        
        // Extended Finite Difference (EFD) gradient
        // Create mask to exclude input_idx-th bit
        uint mask = ((1u << input_dim) - 1u) & ~(1u << input_idx);
        uint addr_masked = addr & mask;
        
        float total_gradient = 0.0f;
        
        // Iterate over all possible addresses k
        for (uint k = 0; k < lut_size; ++k) {
            // Calculate Hamming distance between addr and k, excluding input_idx-th bit
            uint k_masked = k & mask;
            int hamming_dist = __popc(addr_masked ^ k_masked);
            
            // Get k_l (input_idx-th bit of k)
            uint k_l = (k >> input_idx) & 1u;
            
            // Calculate sign factor: (-1)^(1-k_l)
            float sign_factor = (k_l == 0) ? -1.0f : 1.0f;
            
            // Get LUT value at position k (per-layer-node)
            float lut_value = luts[layer_idx * output_dim * lut_size + dim_idx * lut_size + k];
            
            // Add weighted contribution: alpha * sign * lut * beta^hamming_dist
            total_gradient += alpha * sign_factor * lut_value * powf(beta, hamming_dist);
        }
        
        grad_sum += total_gradient * grad_out;
    }
    
    grad_input[batch_idx * layer_size * input_dim + layer_idx * input_dim + input_idx] = grad_sum;
}

// CUDA kernel for EFD backward pass - LUT gradients
// Grad output: (batch_size, layer_size, output_dim)
// LUTs: (layer_size, output_dim, 2^input_dim)
__global__ void efd_cuda_backward_luts_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_luts,
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int output_dim,
    const int lut_size) {
    
    // Each thread handles one (layer, output, lut_entry) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = layer_size * output_dim * lut_size;
    
    if (idx >= total_elements) return;
    
    const int layer_idx = idx / (output_dim * lut_size);
    const int remainder = idx % (output_dim * lut_size);
    const int dim_idx = remainder / lut_size;
    const int lut_entry = remainder % lut_size;
    
    float grad_sum = 0.0f;
    
    // Sum over batch dimension
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Compute address from binary input
        uint addr = 0;
        for (int i = 0; i < input_dim; ++i) {
            float x_val = input[batch_idx * layer_size * input_dim + layer_idx * input_dim + i];
            if (x_val >= 0.5f) {
                addr |= (1u << i);
            }
        }
        
        // Only accumulate gradient if this LUT entry was accessed
        if (addr == lut_entry) {
            const float grad_out = grad_output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx];
            grad_sum += grad_out;
        }
    }
    
    grad_luts[layer_idx * output_dim * lut_size + dim_idx * lut_size + lut_entry] = grad_sum;
}

std::vector<torch::Tensor> efd_cuda_backward(
    torch::Tensor input,
    torch::Tensor luts,
    torch::Tensor grad_output,
    float alpha,
    float beta) {
  
    const int batch_size = input.size(0);
    const int layer_size = input.size(1);
    const int input_dim = input.size(2);
    const int output_dim = luts.size(1);
    const int lut_size = luts.size(2);

    auto grad_input = torch::zeros_like(input);
    auto grad_luts = torch::zeros_like(luts);

    const int threads = 256;
    
    // Launch input gradient kernel
    int total_elements_input = batch_size * layer_size * input_dim;
    int blocks_input = (total_elements_input + threads - 1) / threads;
    
    efd_cuda_backward_input_kernel<<<blocks_input, threads>>>(
        input.data_ptr<float>(),
        luts.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        alpha,
        beta,
        batch_size,
        layer_size,
        input_dim,
        output_dim,
        lut_size
    );
    
    // Launch LUT gradient kernel
    int total_elements_luts = layer_size * output_dim * lut_size;
    int blocks_luts = (total_elements_luts + threads - 1) / threads;
    
    efd_cuda_backward_luts_kernel<<<blocks_luts, threads>>>(
        input.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_luts.data_ptr<float>(),
        batch_size,
        layer_size,
        input_dim,
        output_dim,
        lut_size
    );

    cudaDeviceSynchronize();

    return {grad_input, grad_luts};
};