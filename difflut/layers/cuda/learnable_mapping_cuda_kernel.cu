#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/**
 * CUDA kernel for learnable mapping hard selection (evaluation mode).
 * 
 * This kernel implements the einsum('bi,io->bo') operation used in
 * LearnableLayer's hard selection mode, but more efficiently.
 * 
 * Instead of creating a full binary mask and doing einsum, we directly
 * lookup the selected input index for each output position.
 * 
 * @param input: (batch_size, input_size) - input features
 * @param indices: (output_size,) - argmax indices from weight matrix
 * @param output: (batch_size, output_size) - selected outputs
 * @param batch_size: number of samples in batch
 * @param input_size: number of input features
 * @param output_size: number of output positions (layer_output_size * n)
 */
__global__ void learnable_hard_selection_kernel(
    const float* __restrict__ input,      // (batch_size, input_size)
    const int32_t* __restrict__ indices,  // (output_size,) - argmax results
    float* __restrict__ output,           // (batch_size, output_size)
    const int batch_size,
    const int input_size,
    const int output_size) {
    
    // Each thread processes one output element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * output_size;
    
    if (idx < total_elements) {
        // Decode linear index to (batch, output_pos)
        const int b = idx / output_size;
        const int o = idx % output_size;
        
        // Look up which input feature is selected for this output position
        const int32_t src_idx = indices[o];
        
        // Directly copy: output[b, o] = input[b, src_idx]
        output[idx] = input[b * input_size + src_idx];
    }
}

/**
 * Forward pass for learnable mapping in evaluation mode.
 * 
 * This replaces the einsum operation with a more efficient direct lookup.
 * Training mode (soft selection) should still use PyTorch's matmul as it's
 * already optimal and requires differentiable operations.
 */
torch::Tensor learnable_mapping_cuda_forward(
    torch::Tensor input,
    torch::Tensor indices) {
    
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = indices.size(0);
    
    // Allocate output tensor
    auto output = torch::zeros(
        {batch_size, output_size},
        torch::TensorOptions().dtype(input.dtype()).device(input.device())
    );
    
    // Launch configuration
    const int total_elements = batch_size * output_size;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel
    learnable_hard_selection_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size
    );
    
    return output;
}

/**
 * CUDA kernel for soft selection (training mode) with fused softmax + matmul.
 * 
 * This kernel computes: output = input @ softmax(W / tau).T
 * 
 * For large matrices, this can be faster than separate softmax + matmul operations
 * as it fuses memory accesses and avoids materializing the full softmax weights.
 * 
 * @param input: (batch_size, input_size)
 * @param weights: (output_size, input_size) - raw weight matrix W
 * @param output: (batch_size, output_size)
 * @param tau: temperature for softmax
 */
__global__ void learnable_soft_selection_kernel(
    const float* __restrict__ input,      // (batch_size, input_size)
    const float* __restrict__ weights,    // (output_size, input_size)
    float* __restrict__ output,           // (batch_size, output_size)
    const float tau,
    const int batch_size,
    const int input_size,
    const int output_size) {
    
    // Each block computes one output element (batch, output_pos)
    const int batch_idx = blockIdx.x;
    const int output_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || output_idx >= output_size) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for partial sums and max reduction
    extern __shared__ float shared_mem[];
    float* s_partial_sums = shared_mem;
    float* s_max_vals = &shared_mem[block_size];
    
    const float tau_inv = 1.0f / fmaxf(tau, 1e-6f);
    
    // Step 1: Find max value for numerical stability (reduce over input_size)
    float thread_max = -INFINITY;
    for (int i = tid; i < input_size; i += block_size) {
        const float w = weights[output_idx * input_size + i];
        thread_max = fmaxf(thread_max, w * tau_inv);
    }
    s_max_vals[tid] = thread_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_vals[tid] = fmaxf(s_max_vals[tid], s_max_vals[tid + stride]);
        }
        __syncthreads();
    }
    const float max_val = s_max_vals[0];
    __syncthreads();
    
    // Step 2: Compute exp sum for softmax normalization
    float thread_exp_sum = 0.0f;
    for (int i = tid; i < input_size; i += block_size) {
        const float w = weights[output_idx * input_size + i];
        thread_exp_sum += expf((w * tau_inv) - max_val);
    }
    s_partial_sums[tid] = thread_exp_sum;
    __syncthreads();
    
    // Reduce to get total exp sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial_sums[tid] += s_partial_sums[tid + stride];
        }
        __syncthreads();
    }
    const float exp_sum = s_partial_sums[0];
    const float norm_factor = 1.0f / (exp_sum + 1e-10f);
    __syncthreads();
    
    // Step 3: Compute weighted sum with normalized softmax weights
    float thread_sum = 0.0f;
    for (int i = tid; i < input_size; i += block_size) {
        const float w = weights[output_idx * input_size + i];
        const float softmax_w = expf((w * tau_inv) - max_val) * norm_factor;
        const float x = input[batch_idx * input_size + i];
        thread_sum += softmax_w * x;
    }
    s_partial_sums[tid] = thread_sum;
    __syncthreads();
    
    // Final reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial_sums[tid] += s_partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[batch_idx * output_size + output_idx] = s_partial_sums[0];
    }
}

/**
 * Fused soft selection implementation for training mode.
 * 
 * This can be faster than PyTorch's softmax + matmul for certain matrix sizes,
 * especially when memory bandwidth is the bottleneck.
 */
torch::Tensor learnable_mapping_cuda_soft_forward(
    torch::Tensor input,
    torch::Tensor weights,
    float tau) {
    
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weights.size(0);
    
    // Allocate output tensor
    auto output = torch::zeros(
        {batch_size, output_size},
        torch::TensorOptions().dtype(input.dtype()).device(input.device())
    );
    
    // Launch configuration - 2D grid (batch, output)
    const int threads = 256;
    dim3 blocks(batch_size, output_size);
    const size_t shared_mem = 2 * threads * sizeof(float);
    
    // Launch kernel
    learnable_soft_selection_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        tau,
        batch_size,
        input_size,
        output_size
    );
    
    return output;
}

/**
 * Backward pass for hard selection.
 * 
 * Since hard selection uses argmax (non-differentiable), we use straight-through
 * estimator: gradients flow back as if it were identity mapping.
 * 
 * grad_input[b, src_idx] += grad_output[b, o] for src_idx = indices[o]
 */
__global__ void learnable_hard_selection_backward_kernel(
    const float* __restrict__ grad_output,  // (batch_size, output_size)
    const int32_t* __restrict__ indices,    // (output_size,)
    float* __restrict__ grad_input,         // (batch_size, input_size)
    const int batch_size,
    const int input_size,
    const int output_size) {
    
    // Each thread processes one grad_output element
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * output_size;
    
    if (idx < total_elements) {
        const int b = idx / output_size;
        const int o = idx % output_size;
        
        const int32_t src_idx = indices[o];
        
        // Atomic add to accumulate gradients
        atomicAdd(&grad_input[b * input_size + src_idx], grad_output[idx]);
    }
}

torch::Tensor learnable_mapping_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int input_size) {
    
    const int batch_size = grad_output.size(0);
    const int output_size = grad_output.size(1);
    
    // Allocate gradient tensor
    auto grad_input = torch::zeros(
        {batch_size, input_size},
        torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device())
    );
    
    // Launch configuration
    const int total_elements = batch_size * output_size;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel
    learnable_hard_selection_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        grad_input.data_ptr<float>(),
        batch_size,
        input_size,
        output_size
    );
    
    return grad_input;
}
