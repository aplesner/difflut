#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// CUDA kernel for hybrid forward pass
// Forward: Binary thresholding (like DWN)
// Updated for 2D tensors
__global__ void hybrid_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ luts,
    float* __restrict__ output,
    const int batch_size,
    const int input_dim,
    const int output_dim) {
    
    // Each thread handles one (batch, output) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * output_dim;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into batch and output indices
    const int output_idx = idx % output_dim;
    const int batch_idx = idx / output_dim;
    
    // Compute LUT address using binary thresholding at 0.5
    // Input: (batch_size, input_dim)
    const int input_offset = batch_idx * input_dim;
    
    int addr = 0;
    for (int i = 0; i < input_dim; i++) {
        float val = input[input_offset + i];
        // Threshold at 0.5: val >= 0.5 -> 1, val < 0.5 -> 0
        if (val >= 0.5f) {
            addr |= (1 << i);
        }
    }
    
    // Look up value from LUT
    // LUTs shape: (output_dim, 2^input_dim)
    const int lut_size = 1 << input_dim;
    const int lut_offset = output_idx * lut_size;
    
    output[idx] = luts[lut_offset + addr];
}

// CUDA kernel for hybrid backward pass - input gradients
// Backward: Probabilistic gradients with inputs in [0, 1]
// Updated for 2D tensors
__global__ void hybrid_backward_input_kernel(
    const float* __restrict__ input,
    const float* __restrict__ luts,
    const float* __restrict__ binary_combinations,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int batch_size,
    const int input_dim,
    const int output_dim) {
    
    // Each thread handles one (batch, input) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * input_dim;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index
    const int input_idx = idx % input_dim;
    const int batch_idx = idx / input_dim;
    
    const int lut_size = 1 << input_dim;
    const float eps = 1e-8f;
    
    // Get input value for this element
    const int input_offset = batch_idx * input_dim;
    float x_prob = input[input_offset + input_idx];
    
    float grad_sum = 0.0f;
    
    // For each output dimension
    for (int out_idx = 0; out_idx < output_dim; out_idx++) {
        // Compute probabilistic gradient
        // Sum over all binary combinations
        for (int addr = 0; addr < lut_size; addr++) {
            // Compute Pr(addr|x)
            float prob = 1.0f;
            for (int i = 0; i < input_dim; i++) {
                float xi = input[input_offset + i];
                float ai = binary_combinations[addr * input_dim + i];
                prob *= (xi * ai + (1.0f - xi) * (1.0f - ai));
            }
            
            // Derivative of probability w.r.t. x_prob[input_idx]
            float ai = binary_combinations[addr * input_dim + input_idx];
            float deriv_factor = (ai - x_prob) / (x_prob * (1.0f - x_prob) + eps);
            
            // Weight by LUT value
            const int lut_offset = out_idx * lut_size;
            float lut_val = luts[lut_offset + addr];
            
            // Get gradient from output
            const int grad_out_offset = batch_idx * output_dim;
            float grad_out = grad_output[grad_out_offset + out_idx];
            
            // Accumulate gradient
            grad_sum += prob * deriv_factor * lut_val * grad_out;
        }
    }
    
    grad_input[idx] = grad_sum;
}

// CUDA kernel for hybrid backward pass - LUT gradients
// Updated for 2D tensors - accumulate over batch dimension
__global__ void hybrid_backward_lut_kernel(
    const float* __restrict__ input,
    const float* __restrict__ binary_combinations,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_luts,
    const int batch_size,
    const int input_dim,
    const int output_dim) {
    
    // Each thread handles one (output, addr) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lut_size = 1 << input_dim;
    const int total_lut_elements = output_dim * lut_size;
    
    if (idx >= total_lut_elements) return;
    
    // Decompose linear index
    const int addr = idx % lut_size;
    const int out_idx = idx / lut_size;
    
    float grad_sum = 0.0f;
    
    // Sum over batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Compute Pr(addr|x) for this batch sample
        const int input_offset = batch_idx * input_dim;
        float prob = 1.0f;
        for (int i = 0; i < input_dim; i++) {
            float xi = input[input_offset + i];
            float ai = binary_combinations[addr * input_dim + i];
            prob *= (xi * ai + (1.0f - xi) * (1.0f - ai));
        }
        
        // Get gradient from output
        const int grad_out_offset = batch_idx * output_dim;
        float grad_out = grad_output[grad_out_offset + out_idx];
        
        // Accumulate gradient
        grad_sum += prob * grad_out;
    }
    
    grad_luts[idx] = grad_sum;
}

// C++ interface

torch::Tensor hybrid_cuda_forward(
    torch::Tensor input,
    torch::Tensor luts) {
    
    // Extract 2D dimensions
    const int batch_size = input.size(0);
    const int input_dim = input.size(1);
    const int output_dim = luts.size(0);  // luts: (output_dim, 2^input_dim)
    
    // Allocate output: (batch_size, output_dim)
    auto output = torch::zeros({batch_size, output_dim}, input.options());
    
    // Launch kernel with 1D thread distribution
    const int total_elements = batch_size * output_dim;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    hybrid_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        luts.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_dim,
        output_dim
    );
    
    return output;
}

std::vector<torch::Tensor> hybrid_cuda_backward(
    torch::Tensor input,
    torch::Tensor luts,
    torch::Tensor binary_combinations,
    torch::Tensor grad_output) {
    
    // Extract 2D dimensions
    const int batch_size = input.size(0);
    const int input_dim = input.size(1);
    const int output_dim = luts.size(0);
    const int lut_size = luts.size(1);
    
    // Allocate gradients
    auto grad_input = torch::zeros_like(input);  // (batch_size, input_dim)
    auto grad_luts = torch::zeros_like(luts);    // (output_dim, lut_size)
    
    const int threads = 256;
    
    // Compute input gradients
    {
        const int total_elements = batch_size * input_dim;
        const int blocks = (total_elements + threads - 1) / threads;
        
        hybrid_backward_input_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            luts.data_ptr<float>(),
            binary_combinations.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size,
            input_dim,
            output_dim
        );
    }
    
    // Compute LUT gradients
    {
        const int total_lut_elements = output_dim * lut_size;
        const int blocks = (total_lut_elements + threads - 1) / threads;
        
        hybrid_backward_lut_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            binary_combinations.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_luts.data_ptr<float>(),
            batch_size,
            input_dim,
            output_dim
        );
    }
    
    return {grad_input, grad_luts};
}
