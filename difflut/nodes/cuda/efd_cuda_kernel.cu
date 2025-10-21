#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

__global__ void efd_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,    // (batch_size, input_length) in [0, 1]
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,    // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,     // (num_luts, 2^n) in [0, 1]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {       // (batch_size, num_luts) in [0, 1]
    
    const int batch_size = output.size(0);
    const int num_luts = output.size(1);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_luts; j += blockDim.y * gridDim.y) {

            // Binary threshold at 0.5 for inputs in [0, 1]
            uint addr = input[i][mapping[j][0]] >= 0.5f;
            for(int l = 1; l < mapping.size(1); ++l)
                addr |= (uint)(input[i][mapping[j][l]] >= 0.5f) << l;

            output[i][j] = luts[j][addr];
    
        };
    };

};

torch::Tensor efd_cuda_forward(
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

    efd_cuda_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        output_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return output_tensor;
};

__global__ void efd_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,          // (batch_size, input_length) in [0, 1]
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,          // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,           // (num_luts, 2^n) in [0, 1]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_grad,    // (batch_size, num_luts)
    const float alpha,                                                                        // gradient scaling factor
    const float beta,                                                                         // Hamming distance decay factor
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_grad,           // (batch_size, input_length) 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts_grad) {          // (num_luts, 2^n)
          

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_grad.size(0); i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < output_grad.size(1); j += blockDim.y * gridDim.y) {

            // LUT grad - threshold at 0.5 for inputs in [0, 1]
            uint addr = input[i][mapping[j][0]] >= 0.5f;
            for(int l = 1; l < mapping.size(1); ++l) {
                addr |= (uint)(input[i][mapping[j][l]] >= 0.5f) << l;
            };
            atomicAdd(&luts_grad[j][addr], output_grad[i][j]);

            // Input grad using Extended Finite Difference (EFD) with alpha/beta scaling
            // Iterate over ALL 2^n possible addresses with Hamming distance weighting
            const int n = mapping.size(1);
            const int lut_size = 1 << n;
            
            for(int l = 0; l < n; ++l) {
                float total_gradient = 0.0f;
                
                // Create mask to exclude l-th bit for Hamming distance calculation
                uint mask = ((1 << n) - 1) & ~(1 << l);
                uint addr_masked = addr & mask;
                
                // Iterate over all possible addresses k
                for(uint k = 0; k < lut_size; ++k) {
                    // Calculate Hamming distance between addr and k, excluding l-th bit
                    uint k_masked = k & mask;
                    int hamming_dist = __popc(addr_masked ^ k_masked);
                    
                    // Get k_l (l-th bit of k)
                    uint k_l = (k >> l) & 1;
                    
                    // Calculate sign factor: (-1)^(1-k_l)
                    float sign_factor = (k_l == 0) ? -1.0f : 1.0f;
                    
                    // Get LUT value at position k
                    float lut_value = luts[j][k];
                    
                    // Add weighted contribution: alpha * sign * lut * beta^hamming_dist
                    total_gradient += alpha * sign_factor * lut_value * powf(beta, hamming_dist);
                }
                
                atomicAdd(&input_grad[i][mapping[j][l]], total_gradient * output_grad[i][j]);
            };

        };
    };

};

std::vector<torch::Tensor> efd_cuda_backward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor,
    torch::Tensor output_grad_tensor,
    float alpha,
    float beta) {
  
    auto batch_size = output_grad_tensor.size(0);
    auto output_size = output_grad_tensor.size(1);

    auto input_grad_tensor = torch::zeros_like(input_tensor);
    auto luts_grad_tensor = torch::zeros_like(luts_tensor);

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    efd_cuda_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        output_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        alpha,
        beta,
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        luts_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return {input_grad_tensor, luts_grad_tensor};
};