#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

__global__ void probabilistic_stable_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,    // (batch_size, input_length)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,    // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,     // (num_luts, 2^n)
    const float temperature,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {       // (batch_size, num_luts)
    
    const int batch_size = output.size(0);
    const int num_luts = output.size(1);
    const int n = mapping.size(1);
    const int lut_size = 1 << n;
    
    // Shared memory for caching inputs (up to 32 inputs per thread block)
    extern __shared__ float shared_inputs[];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_luts; j += blockDim.y * gridDim.y) {
            
            // Load inputs into shared memory (reused across all addr iterations)
            // Only load what we need for this LUT
            float local_inputs[32];  // Max 32 inputs per LUT (2^32 is huge anyway)
            for (int l = 0; l < n; ++l) {
                float x_l = input[i][mapping[j][l]];
                local_inputs[l] = fmaxf(0.0f, fminf(1.0f, x_l));  // Clamp once
            }
            
            float result = 0.0f;
            
            // Iterate over all 2^n binary combinations
            for (uint addr = 0; addr < lut_size; ++addr) {
                // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
                float prob = 1.0f;
                
                // Unroll small loops for common cases
                #pragma unroll
                for (int l = 0; l < n; ++l) {
                    // LSB-first bit ordering (consistent across all nodes)
                    uint a_l = (addr >> l) & 1;
                    float x_l = local_inputs[l];
                    
                    // prob *= x_l^a_l * (1-x_l)^(1-a_l)
                    // Branchless version for better performance
                    prob *= (a_l == 1) ? x_l : (1.0f - x_l);
                }
                
                // Apply sigmoid to LUT weight for smooth gradients
                // sigmoid(x) = 1 / (1 + exp(-x))
                float lut_weight = 1.0f / (1.0f + expf(-luts[j][addr]));
                
                // Accumulate: result += lut_weight * prob
                result += lut_weight * prob;
            }
            
            output[i][j] = result;
        }
    }
}

torch::Tensor probabilistic_stable_cuda_forward(
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

    probabilistic_stable_cuda_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        temperature,
        output_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    return output_tensor;
}

__global__ void probabilistic_stable_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,          // (batch_size, input_length)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mapping,          // (num_luts, n)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts,           // (num_luts, 2^n)
    const float temperature,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_grad,    // (batch_size, num_luts)
    const float alpha,
    const int num_inputs,
    const int num_outputs,
    const float scale_min,
    const float scale_max,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input_grad,           // (batch_size, input_length) 
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> luts_grad) {          // (num_luts, 2^n)
    
    const int n = mapping.size(1);
    const int lut_size = 1 << n;
    const float eps = 1e-6f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_grad.size(0); i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < output_grad.size(1); j += blockDim.y * gridDim.y) {

            const float grad_out = output_grad[i][j];
            
            // Load and clamp inputs once (reuse across all addresses)
            float local_inputs[32];
            for (int l = 0; l < n; ++l) {
                float x_l = input[i][mapping[j][l]];
                local_inputs[l] = fmaxf(0.0f, fminf(1.0f, x_l));
            }
            
            // Accumulate input gradients locally to reduce atomic operations
            float local_input_grad[32] = {0.0f};
            
            // Iterate over all 2^n binary combinations
            for (uint addr = 0; addr < lut_size; ++addr) {
                // Compute Pr(addr|x) = Π_l [x_l^a_l * (1-x_l)^(1-a_l)]
                float prob = 1.0f;
                
                #pragma unroll
                for (int l = 0; l < n; ++l) {
                    uint a_l = (addr >> l) & 1;
                    float x_l = local_inputs[l];
                    prob *= (a_l == 1) ? x_l : (1.0f - x_l);
                }
                
                // Get raw LUT weight
                float raw_weight = luts[j][addr];
                
                // Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
                float sigmoid_weight = 1.0f / (1.0f + expf(-raw_weight));
                
                // Gradient w.r.t. LUT weights
                // d(sigmoid(w))/dw = sigmoid(w) * (1 - sigmoid(w))
                float sigmoid_grad = sigmoid_weight * (1.0f - sigmoid_weight);
                atomicAdd(&luts_grad[j][addr], prob * grad_out * sigmoid_grad);
                
                // Gradient w.r.t. inputs (accumulate locally first)
                #pragma unroll
                for (int l = 0; l < n; ++l) {
                    uint a_l = (addr >> l) & 1;
                    float x_l = local_inputs[l];
                    
                    // d(prob)/d(x_l): prob/x_l if a_l==1, -prob/(1-x_l) if a_l==0
                    float dprob_dx = 0.0f;
                    if (a_l == 1) {
                        if (x_l > eps) {
                            dprob_dx = prob / x_l;
                        }
                    } else {
                        float one_minus_x = 1.0f - x_l;
                        if (one_minus_x > eps) {
                            dprob_dx = -prob / one_minus_x;
                        }
                    }
                    
                    local_input_grad[l] += sigmoid_weight * dprob_dx * grad_out;
                }
            }
            
            // Write accumulated input gradients with fewer atomic operations
            for (int l = 0; l < n; ++l) {
                if (local_input_grad[l] != 0.0f) {
                    atomicAdd(&input_grad[i][mapping[j][l]], local_input_grad[l]);
                }
            }
        }
    }
}

// Kernel to compute L1 norms for gradient stabilization
__global__ void compute_l1_norms_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_output,  // (batch_size, num_outputs)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_input,   // (batch_size, num_inputs)
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grad_output_l1,     // (batch_size,)
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grad_input_l1) {    // (batch_size,)
    
    const int batch_size = grad_output.size(0);
    const int num_outputs = grad_output.size(1);
    const int num_inputs = grad_input.size(1);
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        // Compute L1 norm of output gradient for sample i
        float sum_out = 0.0f;
        for (int j = 0; j < num_outputs; ++j) {
            sum_out += fabsf(grad_output[i][j]);
        }
        grad_output_l1[i] = sum_out;
        
        // Compute L1 norm of input gradient for sample i
        float sum_in = 0.0f;
        for (int j = 0; j < num_inputs; ++j) {
            sum_in += fabsf(grad_input[i][j]);
        }
        grad_input_l1[i] = sum_in;
    }
}

// Kernel to apply gradient stabilization
__global__ void apply_gradient_stabilization_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_input,           // (batch_size, num_inputs)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grad_output_l1, // (batch_size,)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> grad_input_l1,  // (batch_size,)
    const float alpha,
    const float log_ratio,
    const float scale_min,
    const float scale_max,
    const float eps) {
    
    const int batch_size = grad_input.size(0);
    const int num_inputs = grad_input.size(1);
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
        // Compute scaling factor for sample i
        // s_i = (alpha * log_ratio * ||g_out[i]||_1) / (||g_in[i]||_1 + eps)
        float scale = (alpha * log_ratio * grad_output_l1[i]) / (grad_input_l1[i] + eps);
        
        // Clamp scale to [scale_min, scale_max]
        scale = fmaxf(scale_min, fminf(scale_max, scale));
        
        // Apply scaling to all input gradients for sample i
        for (int j = 0; j < num_inputs; ++j) {
            grad_input[i][j] *= scale;
        }
    }
}

std::vector<torch::Tensor> probabilistic_stable_cuda_backward(
    torch::Tensor input_tensor,
    torch::Tensor mapping_tensor,
    torch::Tensor luts_tensor,
    torch::Tensor temperature_tensor,
    torch::Tensor output_grad_tensor,
    torch::Tensor alpha_tensor,
    int num_inputs,
    int num_outputs,
    torch::Tensor scale_min_tensor,
    torch::Tensor scale_max_tensor) {
  
    auto batch_size = output_grad_tensor.size(0);
    auto output_size = output_grad_tensor.size(1);
    
    float temperature = temperature_tensor.item<float>();
    float alpha = alpha_tensor.item<float>();
    float scale_min = scale_min_tensor.item<float>();
    float scale_max = scale_max_tensor.item<float>();

    auto input_grad_tensor = torch::zeros_like(input_tensor);
    auto luts_grad_tensor = torch::zeros_like(luts_tensor);

    dim3 threads_per_block(32, 32);

    dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(output_size, static_cast<int64_t>(threads_per_block.y)))
    );

    // Step 1: Compute raw gradients (without stabilization)
    probabilistic_stable_cuda_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        mapping_tensor.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        luts_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        temperature,
        output_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        alpha,
        num_inputs,
        num_outputs,
        scale_min,
        scale_max,
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        luts_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    // Step 2: Compute L1 norms for gradient stabilization
    auto grad_output_l1 = torch::empty({batch_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, input_tensor.device().index()));
    auto grad_input_l1 = torch::empty({batch_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, input_tensor.device().index()));

    dim3 threads_per_block_1d(256);
    dim3 blocks_per_grid_1d(min(static_cast<int64_t>(65535), 
                                ceil_div(batch_size, static_cast<int64_t>(threads_per_block_1d.x))));

    compute_l1_norms_kernel<<<blocks_per_grid_1d, threads_per_block_1d>>>(
        output_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_output_l1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        grad_input_l1.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();

    // Step 3: Apply gradient stabilization
    // Compute log ratio: log(1 + num_inputs/num_outputs)
    float log_ratio = logf(1.0f + static_cast<float>(num_inputs) / fmaxf(static_cast<float>(num_outputs), 1e-6f));
    float eps = 1e-6f;

    apply_gradient_stabilization_kernel<<<blocks_per_grid_1d, threads_per_block_1d>>>(
        input_grad_tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_output_l1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        grad_input_l1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        alpha,
        log_ratio,
        scale_min,
        scale_max,
        eps
    );

    cudaDeviceSynchronize();

    return {input_grad_tensor, luts_grad_tensor};
}
