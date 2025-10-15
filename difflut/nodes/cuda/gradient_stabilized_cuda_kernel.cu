#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

__global__ void gradient_stabilized_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,    // (batch_size, input_length)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,    // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,     // (num_luts, 2^n)
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {       // (batch_size, num_luts)
    
    const int batch_size = output.size(0);
    const int num_luts = output.size(1);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_luts; j += blockDim.y * gridDim.y) {
                
            // Binary thresholding at 0.5: input >= 0.5 -> 1, input < 0.5 -> 0
            uint addr = input[i][mapping[j][0]] >= 0.5f;
            for(int l = 1; l < mapping.size(1); ++l)
                addr |= (uint)(input[i][mapping[j][l]] >= 0.5f) << l;

            output[i][j] = luts[j][addr];
        };
    };
}

torch::Tensor gradient_stabilized_cuda_forward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor) {
  
    auto batch_size = input_tensor.size(0);
    auto output_size = luts_tensor.size(0);

    auto output_tensor = torch::empty({batch_size, output_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, input_tensor.device().index()));

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    gradient_stabilized_cuda_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        output_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return output_tensor;
}

__global__ void gradient_stabilized_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,          // (batch_size, input_length)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,          // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,           // (num_luts, 2^n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_grad,    // (batch_size, num_luts)
    const float gradient_scale,                                                               // scalar
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_grad,           // (batch_size, input_length) 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts_grad) {          // (num_luts, 2^n)
          
    const int batch_size = output_grad.size(0);
    const int num_luts = output_grad.size(1);
    const int n = mapping.size(1);
    const int lut_size = 1 << n;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_luts; j += blockDim.y * gridDim.y) {

            // Current address based on binary thresholding at 0.5
            uint addr = input[i][mapping[j][0]] >= 0.5f;
            for(int l = 1; l < n; ++l) {
                addr |= (uint)(input[i][mapping[j][l]] >= 0.5f) << l;
            }
            
            // Apply gradient scaling
            float scaled_grad_output = output_grad[i][j] * gradient_scale;
            
            // LUT gradient - standard accumulation
            atomicAdd(&luts_grad[j][addr], scaled_grad_output);

            // Input gradient with distance-based weighting
            // Compute LUT variation (max - min) for this LUT
            float lut_min = luts[j][0];
            float lut_max = luts[j][0];
            for(int k = 1; k < lut_size; ++k) {
                lut_min = fminf(lut_min, luts[j][k]);
                lut_max = fmaxf(lut_max, luts[j][k]);
            }
            float lut_variation = lut_max - lut_min;
            
            // For each input dimension
            for(int l = 0; l < n; ++l) {
                int input_idx = mapping[j][l];
                float x_val = input[i][input_idx];
                
                // Distance from threshold (0.5)
                // At 0.5: distance_from_threshold = 1.0
                // At 0.0 or 1.0: distance_from_threshold = 0.0
                float distance_from_threshold = 1.0f - 2.0f * fabsf(x_val - 0.5f);
                distance_from_threshold = fmaxf(0.0f, fminf(1.0f, distance_from_threshold));
                
                // Gradient magnitude based on LUT variation and distance
                float gradient_magnitude = distance_from_threshold * lut_variation;
                
                // Apply gradient with proper sign
                float grad_contrib;
                if (x_val >= 0.5f) {
                    grad_contrib = gradient_magnitude * scaled_grad_output;
                } else {
                    grad_contrib = -gradient_magnitude * scaled_grad_output;
                }
                
                atomicAdd(&input_grad[i][input_idx], grad_contrib);
            }
        }
    }
}

std::vector<torch::Tensor> gradient_stabilized_cuda_backward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor,
    torch::Tensor output_grad_tensor,
    float gradient_scale) {
  
    auto batch_size = output_grad_tensor.size(0);
    auto output_size = output_grad_tensor.size(1);

    auto input_grad_tensor = torch::zeros_like(input_tensor);
    auto luts_grad_tensor = torch::zeros_like(luts_tensor);

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    gradient_stabilized_cuda_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        output_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        gradient_scale,
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        luts_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return {input_grad_tensor, luts_grad_tensor};
}
