#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// CUDA kernel for hybrid forward pass
// Forward: Binary thresholding (like DWN)
__global__ void hybrid_forward_kernel(
    const float* __restrict__ input,
    const int* __restrict__ mapping,
    const float* __restrict__ luts,
    float* __restrict__ output,
    const int batch_size,
    const int input_length,
    const int num_luts,
    const int n) {
    
    // Each thread handles one (batch, lut) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / num_luts;
    const int lut_idx = idx % num_luts;
    
    if (batch_idx >= batch_size || lut_idx >= num_luts) return;
    
    // Compute LUT address using binary thresholding at 0.5
    int addr = 0;
    for (int i = 0; i < n; i++) {
        int input_idx = mapping[lut_idx * n + i];
        float val = input[batch_idx * input_length + input_idx];
        // Threshold at 0.5: val >= 0.5 -> 1, val < 0.5 -> 0
        if (val >= 0.5f) {
            addr |= (1 << i);
        }
    }
    
    // Look up value
    const int lut_size = 1 << n;
    output[batch_idx * num_luts + lut_idx] = luts[lut_idx * lut_size + addr];
}

// CUDA kernel for hybrid backward pass - input gradients
// Backward: Probabilistic gradients with inputs in [0, 1]
__global__ void hybrid_backward_input_kernel(
    const float* __restrict__ input,
    const int* __restrict__ mapping,
    const float* __restrict__ luts,
    const float* __restrict__ binary_combinations,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int batch_size,
    const int input_length,
    const int num_luts,
    const int n) {
    
    // Each thread handles one (batch, input) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / input_length;
    const int input_idx = idx % input_length;
    
    if (batch_idx >= batch_size || input_idx >= input_length) return;
    
    const int lut_size = 1 << n;
    const float eps = 1e-8f;
    
    // Input is already in [0, 1], use directly for probabilistic computation
    float x_prob = input[batch_idx * input_length + input_idx];
    
    float grad_sum = 0.0f;
    
    // For each LUT that uses this input
    for (int lut_idx = 0; lut_idx < num_luts; lut_idx++) {
        // Check if this LUT uses this input
        int input_pos = -1;
        for (int i = 0; i < n; i++) {
            if (mapping[lut_idx * n + i] == input_idx) {
                input_pos = i;
                break;
            }
        }
        
        if (input_pos < 0) continue;
        
        // Compute probabilistic gradient
        // Sum over all binary combinations
        for (int addr = 0; addr < lut_size; addr++) {
            // Compute Pr(addr|x)
            float prob = 1.0f;
            for (int i = 0; i < n; i++) {
                int mapped_input = mapping[lut_idx * n + i];
                float xi = input[batch_idx * input_length + mapped_input];
                float ai = binary_combinations[addr * n + i];
                prob *= (xi * ai + (1.0f - xi) * (1.0f - ai));
            }
            
            // Derivative of probability w.r.t. x_prob[input_idx]
            float ai = binary_combinations[addr * n + input_pos];
            float deriv_factor = (ai - x_prob) / (x_prob * (1.0f - x_prob) + eps);
            
            // Weight by LUT value
            float lut_val = luts[lut_idx * lut_size + addr];
            
            // Accumulate gradient
            grad_sum += prob * deriv_factor * lut_val * grad_output[batch_idx * num_luts + lut_idx];
        }
    }
    
    // No sigmoid chain rule needed since input is already in [0, 1]
    grad_input[batch_idx * input_length + input_idx] = grad_sum;
}

// CUDA kernel for hybrid backward pass - LUT gradients
__global__ void hybrid_backward_lut_kernel(
    const float* __restrict__ input,
    const int* __restrict__ mapping,
    const float* __restrict__ binary_combinations,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_luts,
    const int batch_size,
    const int input_length,
    const int num_luts,
    const int n) {
    
    // Each thread handles one (lut, addr) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lut_size = 1 << n;
    const int lut_idx = idx / lut_size;
    const int addr = idx % lut_size;
    
    if (lut_idx >= num_luts || addr >= lut_size) return;
    
    float grad_sum = 0.0f;
    
    // Sum over batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Compute Pr(addr|x) for this batch sample
        float prob = 1.0f;
        for (int i = 0; i < n; i++) {
            int mapped_input = mapping[lut_idx * n + i];
            // Input is already in [0, 1], use directly
            float xi = input[batch_idx * input_length + mapped_input];
            float ai = binary_combinations[addr * n + i];
            prob *= (xi * ai + (1.0f - xi) * (1.0f - ai));
        }
        
        // Accumulate gradient
        grad_sum += prob * grad_output[batch_idx * num_luts + lut_idx];
    }
    
    grad_luts[lut_idx * lut_size + addr] = grad_sum;
}

// C++ interface

torch::Tensor hybrid_cuda_forward(
    torch::Tensor input,
    torch::Tensor mapping,
    torch::Tensor luts) {
    
    const int batch_size = input.size(0);
    const int input_length = input.size(1);
    const int num_luts = luts.size(0);
    const int lut_size = luts.size(1);
    const int n = mapping.size(1);
    
    auto output = torch::zeros({batch_size, num_luts}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * num_luts + threads - 1) / threads;
    
    hybrid_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mapping.data_ptr<int>(),
        luts.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_length,
        num_luts,
        n
    );
    
    return output;
}

std::vector<torch::Tensor> hybrid_cuda_backward(
    torch::Tensor input,
    torch::Tensor mapping,
    torch::Tensor luts,
    torch::Tensor binary_combinations,
    torch::Tensor grad_output) {
    
    const int batch_size = input.size(0);
    const int input_length = input.size(1);
    const int num_luts = luts.size(0);
    const int lut_size = luts.size(1);
    const int n = mapping.size(1);
    
    auto grad_input = torch::zeros_like(input);
    auto grad_luts = torch::zeros_like(luts);
    
    const int threads = 256;
    
    // Compute input gradients
    {
        const int blocks = (batch_size * input_length + threads - 1) / threads;
        hybrid_backward_input_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            mapping.data_ptr<int>(),
            luts.data_ptr<float>(),
            binary_combinations.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size,
            input_length,
            num_luts,
            n
        );
    }
    
    // Compute LUT gradients
    {
        const int blocks = (num_luts * lut_size + threads - 1) / threads;
        hybrid_backward_lut_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            mapping.data_ptr<int>(),
            binary_combinations.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_luts.data_ptr<float>(),
            batch_size,
            input_length,
            num_luts,
            n
        );
    }
    
    return {grad_input, grad_luts};
}
