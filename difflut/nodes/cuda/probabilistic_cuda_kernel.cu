#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

__global__ void probabilistic_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,    // (batch_size, input_length)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,    // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,     // (num_luts, 2^n)
    const float temperature,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {       // (batch_size, num_luts)
    
    const int batch_size = output.size(0);
    const int num_luts = output.size(1);
    const int n = mapping.size(1);
    const int lut_size = 1 << n;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_luts; j += blockDim.y * gridDim.y) {
            
            float result = 0.0f;
            
            // Iterate over all 2^n binary combinations
            for (uint addr = 0; addr < lut_size; ++addr) {
                // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
                float prob = 1.0f;
                
                for (int l = 0; l < n; ++l) {
                    // LSB-first bit ordering (consistent across all nodes)
                    uint a_l = (addr >> l) & 1;  // LSB first
                    float x_l = input[i][mapping[j][l]];
                    
                    // Clamp input to [0, 1] for numerical stability
                    x_l = fmaxf(0.0f, fminf(1.0f, x_l));
                    
                    // prob *= x_l^a_l * (1-x_l)^(1-a_l)
                    if (a_l == 1) {
                        prob *= x_l;
                    } else {
                        prob *= (1.0f - x_l);
                    }
                }
                
                // Apply sigmoid to LUT weight with temperature scaling
                float lut_weight = 1.0f / (1.0f + expf(-luts[j][addr] / fmaxf(temperature, 1e-6f)));
                
                // Accumulate: result += lut_weight * prob
                result += lut_weight * prob;
            }
            
            output[i][j] = result;
        }
    }
}

torch::Tensor probabilistic_cuda_forward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor,
    torch::Tensor temperature_tensor) {
  
    auto batch_size = input_tensor.size(0);
    auto output_size = luts_tensor.size(0);
    
    float temperature = temperature_tensor.item<float>();

    auto output_tensor = torch::empty({batch_size, output_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, input_tensor.device().index()));

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    probabilistic_cuda_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        temperature,
        output_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return output_tensor;
}

__global__ void probabilistic_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,          // (batch_size, input_length)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,          // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,           // (num_luts, 2^n)
    const float temperature,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_grad,    // (batch_size, num_luts)
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_grad,           // (batch_size, input_length) 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts_grad) {          // (num_luts, 2^n)
    
    const int n = mapping.size(1);
    const int lut_size = 1 << n;
    const float temp = fmaxf(temperature, 1e-6f);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_grad.size(0); i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < output_grad.size(1); j += blockDim.y * gridDim.y) {

            const float grad_out = output_grad[i][j];
            
            // Iterate over all 2^n binary combinations
            for (uint addr = 0; addr < lut_size; ++addr) {
                // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
                float prob = 1.0f;
                
                for (int l = 0; l < n; ++l) {
                    // LSB-first bit ordering (consistent across all nodes)
                    uint a_l = (addr >> l) & 1;  // LSB first
                    float x_l = input[i][mapping[j][l]];
                    x_l = fmaxf(0.0f, fminf(1.0f, x_l));
                    
                    if (a_l == 1) {
                        prob *= x_l;
                    } else {
                        prob *= (1.0f - x_l);
                    }
                }
                
                // Compute sigmoid and its derivative
                float raw_weight = luts[j][addr];
                float exp_neg = expf(-raw_weight / temp);
                float sigmoid_val = 1.0f / (1.0f + exp_neg);
                float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val) / temp;
                
                // Gradient w.r.t. LUT weights (raw_weights before sigmoid)
                // d(output)/d(raw_weight) = prob * sigmoid'(raw_weight/temp) * grad_out
                atomicAdd(&luts_grad[j][addr], prob * sigmoid_grad * grad_out);
                
                // Gradient w.r.t. inputs
                // d(output)/d(x_l) = sigmoid(raw_weight/temp) * d(prob)/d(x_l) * grad_out
                for (int l = 0; l < n; ++l) {
                    uint a_l = (addr >> l) & 1;
                    float x_l = input[i][mapping[j][l]];
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
                    
                    atomicAdd(&input_grad[i][mapping[j][l]], sigmoid_val * dprob_dx * grad_out);
                }
            }
        }
    }
}

std::vector<torch::Tensor> probabilistic_cuda_backward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor,
    torch::Tensor temperature_tensor,
    torch::Tensor output_grad_tensor) {
  
    auto batch_size = output_grad_tensor.size(0);
    auto output_size = output_grad_tensor.size(1);
    
    float temperature = temperature_tensor.item<float>();

    auto input_grad_tensor = torch::zeros_like(input_tensor);
    auto luts_grad_tensor = torch::zeros_like(luts_tensor);

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    probabilistic_cuda_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        temperature,
        output_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        luts_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return {input_grad_tensor, luts_grad_tensor};
}
