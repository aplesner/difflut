#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

// Fused EFD forward kernel with on-the-fly indexing
// This eliminates the need to materialize (batch_size, layer_size, input_dim) mapped_inputs tensor
// Instead, indexing happens directly inside the kernel using the mapping indices
__global__ void efd_fused_forward_kernel(
    const float* __restrict__ input,         // (batch_size, input_size)
    const int64_t* __restrict__ mapping,     // (layer_size, input_dim) - indices into input_size
    const float* __restrict__ luts,          // (layer_size, output_dim, lut_size)
    float* __restrict__ output,              // (batch_size, layer_size, output_dim)
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int input_size,
    const int output_dim,
    const int lut_size) {

    // Each thread handles one (batch, layer, output_dim) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * output_dim;

    if (idx >= total_elements) return;

    // Decompose linear index
    const int batch_idx = idx / (layer_size * output_dim);
    const int remainder = idx % (layer_size * output_dim);
    const int layer_idx = remainder / output_dim;
    const int dim_idx = remainder % output_dim;

    // KEY OPTIMIZATION: Gather from input on-the-fly using mapping indices
    // Compute LUT address from binary input
    uint addr = 0;
    for (int i = 0; i < input_dim; ++i) {
        // Get the input index for this position from mapping
        int64_t input_idx = mapping[layer_idx * input_dim + i];

        // Read from raw input tensor (no materialized mapped_inputs!)
        float x_val = input[batch_idx * input_size + input_idx];

        if (x_val >= 0.5f) {
            addr |= (1u << i);
        }
    }

    // Look up LUT value (same as non-fused version)
    float lut_val = luts[layer_idx * output_dim * lut_size + dim_idx * lut_size + addr];
    output[batch_idx * layer_size * output_dim + layer_idx * output_dim + dim_idx] = lut_val;
}

torch::Tensor efd_fused_cuda_forward(
    torch::Tensor input,          // (batch_size, input_size)
    torch::Tensor mapping,        // (layer_size, input_dim)
    torch::Tensor luts) {         // (layer_size, output_dim, lut_size)

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int layer_size = mapping.size(0);
    const int input_dim = mapping.size(1);
    const int output_dim = luts.size(1);
    const int lut_size = luts.size(2);

    auto output = torch::zeros({batch_size, layer_size, output_dim}, input.options());

    const int threads = 256;
    const int total_elements = batch_size * layer_size * output_dim;
    const int blocks = (total_elements + threads - 1) / threads;

    efd_fused_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        mapping.data_ptr<int64_t>(),
        luts.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        layer_size,
        input_dim,
        input_size,
        output_dim,
        lut_size
    );

    cudaDeviceSynchronize();

    return output;
}

// Fused EFD backward kernel for input gradients
__global__ void efd_fused_backward_input_kernel(
    const float* __restrict__ input,         // (batch_size, input_size)
    const int64_t* __restrict__ mapping,     // (layer_size, input_dim)
    const float* __restrict__ luts,          // (layer_size, output_dim, lut_size)
    const float* __restrict__ grad_output,   // (batch_size, layer_size, output_dim)
    float* __restrict__ grad_input,          // (batch_size, input_size)
    const float alpha,
    const float beta,
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int input_size,
    const int output_dim,
    const int lut_size) {

    // Each thread handles one (batch, layer, input_dim) element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * input_dim;

    if (idx >= total_elements) return;

    const int batch_idx = idx / (layer_size * input_dim);
    const int remainder = idx % (layer_size * input_dim);
    const int layer_idx = remainder / input_dim;
    const int input_dim_idx = remainder % input_dim;

    // Get the actual input index from mapping
    int64_t actual_input_idx = mapping[layer_idx * input_dim + input_dim_idx];

    // Compute current address for this layer node
    uint addr = 0;
    for (int i = 0; i < input_dim; ++i) {
        int64_t idx_i = mapping[layer_idx * input_dim + i];
        float x_val = input[batch_idx * input_size + idx_i];
        if (x_val >= 0.5f) {
            addr |= (1u << i);
        }
    }

    // Extended Finite Difference gradient computation
    const uint mask = ((1u << input_dim) - 1) & ~(1u << input_dim_idx);
    const uint addr_masked = addr & mask;

    float total_gradient = 0.0f;

    for (uint k = 0; k < lut_size; ++k) {
        const uint k_masked = k & mask;
        const int hamming_dist = __popc(addr_masked ^ k_masked);
        const uint k_l = (k >> input_dim_idx) & 1u;
        const float sign_factor = (k_l == 0) ? -1.0f : 1.0f;

        float beta_power = powf(beta, static_cast<float>(hamming_dist));

        for (int dim_idx = 0; dim_idx < output_dim; ++dim_idx) {
            float lut_value = luts[layer_idx * output_dim * lut_size + dim_idx * lut_size + k];
            float grad_out = grad_output[batch_idx * layer_size * output_dim +
                                        layer_idx * output_dim + dim_idx];

            total_gradient += alpha * sign_factor * lut_value * beta_power * grad_out;
        }
    }

    // IMPORTANT: Scatter gradient to the correct input position
    // Use atomicAdd since multiple (layer, input_dim) pairs may map to same input position
    atomicAdd(&grad_input[batch_idx * input_size + actual_input_idx], total_gradient);
}

// Fused EFD backward kernel for LUT gradients
__global__ void efd_fused_backward_luts_kernel(
    const float* __restrict__ input,         // (batch_size, input_size)
    const int64_t* __restrict__ mapping,     // (layer_size, input_dim)
    const float* __restrict__ grad_output,   // (batch_size, layer_size, output_dim)
    float* __restrict__ grad_luts,           // (layer_size, output_dim, lut_size)
    const int batch_size,
    const int layer_size,
    const int input_dim,
    const int input_size,
    const int output_dim,
    const int lut_size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * layer_size * output_dim;

    if (idx >= total_elements) return;

    const int batch_idx = idx / (layer_size * output_dim);
    const int remainder = idx % (layer_size * output_dim);
    const int layer_idx = remainder / output_dim;
    const int dim_idx = remainder % output_dim;

    // Compute address using mapping
    uint addr = 0;
    for (int i = 0; i < input_dim; ++i) {
        int64_t input_idx = mapping[layer_idx * input_dim + i];
        float x_val = input[batch_idx * input_size + input_idx];
        if (x_val >= 0.5f) {
            addr |= (1u << i);
        }
    }

    // Accumulate gradient for this LUT entry
    float grad = grad_output[batch_idx * layer_size * output_dim +
                            layer_idx * output_dim + dim_idx];

    atomicAdd(&grad_luts[layer_idx * output_dim * lut_size + dim_idx * lut_size + addr], grad);
}

std::vector<torch::Tensor> efd_fused_cuda_backward(
    torch::Tensor input,          // (batch_size, input_size)
    torch::Tensor mapping,        // (layer_size, input_dim)
    torch::Tensor luts,           // (layer_size, output_dim, lut_size)
    torch::Tensor grad_output,    // (batch_size, layer_size, output_dim)
    float alpha,
    float beta) {

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int layer_size = mapping.size(0);
    const int input_dim = mapping.size(1);
    const int output_dim = luts.size(1);
    const int lut_size = luts.size(2);

    auto grad_input = torch::zeros_like(input);
    auto grad_luts = torch::zeros_like(luts);

    const int threads = 256;

    // Gradient for input
    int total_input = batch_size * layer_size * input_dim;
    int blocks_input = (total_input + threads - 1) / threads;
    efd_fused_backward_input_kernel<<<blocks_input, threads>>>(
        input.data_ptr<float>(),
        mapping.data_ptr<int64_t>(),
        luts.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        alpha, beta,
        batch_size, layer_size, input_dim, input_size, output_dim, lut_size
    );

    // Gradient for LUTs
    int total_luts = batch_size * layer_size * output_dim;
    int blocks_luts = (total_luts + threads - 1) / threads;
    efd_fused_backward_luts_kernel<<<blocks_luts, threads>>>(
        input.data_ptr<float>(),
        mapping.data_ptr<int64_t>(),
        grad_output.data_ptr<float>(),
        grad_luts.data_ptr<float>(),
        batch_size, layer_size, input_dim, input_size, output_dim, lut_size
    );

    cudaDeviceSynchronize();

    return {grad_input, grad_luts};
}
