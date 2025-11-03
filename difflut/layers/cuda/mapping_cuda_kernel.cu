#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

/**
 * CUDA kernel for optimized mapping operation.
 * 
 * Replaces the expand + gather pattern with a direct lookup kernel.
 * This eliminates the need for intermediate expanded tensors and reduces
 * memory traffic significantly.
 * 
 * @param input: (batch_size, input_size) - input features
 * @param indices: (output_size, n) - mapping indices (int16 or int32)
 * @param output: (batch_size, output_size, n) - mapped outputs
 * @param batch_size: number of samples in batch
 * @param input_size: number of input features
 * @param output_size: number of output nodes
 * @param n: number of inputs per node
 */
template <typename index_t>
__global__ void gather_mapping_kernel(
    const float* __restrict__ input,      // (batch_size, input_size)
    const index_t* __restrict__ indices,  // (output_size, n)
    float* __restrict__ output,           // (batch_size, output_size, n)
    const int batch_size,
    const int input_size,
    const int output_size,
    const int n) {
    
    // Each thread processes one element of the output tensor
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * output_size * n;
    
    if (idx < total_elements) {
        // Decode linear index to (batch, output, input_pos)
        const int b = idx / (output_size * n);
        const int o = (idx / n) % output_size;
        const int i = idx % n;
        
        // Look up which input feature to use
        const index_t src_idx = indices[o * n + i];
        
        // Gather from input: output[b, o, i] = input[b, src_idx]
        output[idx] = input[b * input_size + src_idx];
    }
}

/**
 * Optimized mapping forward pass using CUDA kernel.
 * 
 * This replaces the PyTorch operations:
 *   x_expanded = x.unsqueeze(1).expand(-1, output_size, -1)
 *   mapped_inputs = torch.gather(x_expanded, dim=2, index=indices_long)
 * 
 * With a single fused kernel that directly performs the lookup without
 * creating intermediate tensors.
 */
torch::Tensor mapping_cuda_forward(
    torch::Tensor input,
    torch::Tensor indices) {
    
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = indices.size(0);
    const int n = indices.size(1);
    
    // Allocate output tensor
    auto output = torch::zeros(
        {batch_size, output_size, n},
        torch::TensorOptions().dtype(input.dtype()).device(input.device())
    );
    
    // Launch configuration
    const int total_elements = batch_size * output_size * n;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel based on index type
    if (indices.scalar_type() == torch::kInt16 || indices.scalar_type() == torch::kShort) {
        gather_mapping_kernel<int16_t><<<blocks, threads>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int16_t>(),
            output.data_ptr<float>(),
            batch_size,
            input_size,
            output_size,
            n
        );
    } else if (indices.scalar_type() == torch::kInt32 || indices.scalar_type() == torch::kInt) {
        gather_mapping_kernel<int32_t><<<blocks, threads>>>(
            input.data_ptr<float>(),
            indices.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            batch_size,
            input_size,
            output_size,
            n
        );
    } else {
        AT_ERROR("Unsupported index type for mapping kernel. Use int16 or int32.");
    }
    
    return output;
}

/**
 * Backward pass for mapping operation.
 * 
 * Computes gradient with respect to input:
 *   grad_input[b, src_idx] += grad_output[b, o, i]
 * 
 * This is a scatter-add operation where gradients from multiple output
 * positions may accumulate to the same input position.
 */
template <typename index_t>
__global__ void gather_mapping_backward_kernel(
    const float* __restrict__ grad_output,  // (batch_size, output_size, n)
    const index_t* __restrict__ indices,    // (output_size, n)
    float* __restrict__ grad_input,         // (batch_size, input_size)
    const int batch_size,
    const int input_size,
    const int output_size,
    const int n) {
    
    // Each thread processes one element of grad_output
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * output_size * n;
    
    if (idx < total_elements) {
        // Decode linear index
        const int b = idx / (output_size * n);
        const int o = (idx / n) % output_size;
        const int i = idx % n;
        
        // Look up source index
        const index_t src_idx = indices[o * n + i];
        
        // Atomic add to grad_input (multiple threads may write to same location)
        atomicAdd(&grad_input[b * input_size + src_idx], grad_output[idx]);
    }
}

torch::Tensor mapping_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int input_size) {
    
    const int batch_size = grad_output.size(0);
    const int output_size = grad_output.size(1);
    const int n = grad_output.size(2);
    
    // Allocate gradient tensor (zero-initialized for accumulation)
    auto grad_input = torch::zeros(
        {batch_size, input_size},
        torch::TensorOptions().dtype(grad_output.dtype()).device(grad_output.device())
    );
    
    // Launch configuration
    const int total_elements = batch_size * output_size * n;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel based on index type
    if (indices.scalar_type() == torch::kInt16 || indices.scalar_type() == torch::kShort) {
        gather_mapping_backward_kernel<int16_t><<<blocks, threads>>>(
            grad_output.data_ptr<float>(),
            indices.data_ptr<int16_t>(),
            grad_input.data_ptr<float>(),
            batch_size,
            input_size,
            output_size,
            n
        );
    } else if (indices.scalar_type() == torch::kInt32 || indices.scalar_type() == torch::kInt) {
        gather_mapping_backward_kernel<int32_t><<<blocks, threads>>>(
            grad_output.data_ptr<float>(),
            indices.data_ptr<int32_t>(),
            grad_input.data_ptr<float>(),
            batch_size,
            input_size,
            output_size,
            n
        );
    } else {
        AT_ERROR("Unsupported index type for mapping backward kernel. Use int16 or int32.");
    }
    
    return grad_input;
}
