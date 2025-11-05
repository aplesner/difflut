#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA kernels
torch::Tensor mapping_cuda_forward(
    torch::Tensor input,
    torch::Tensor indices
);

torch::Tensor mapping_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int input_size
);

// Input validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Python-facing wrapper for mapping forward pass.
 * Validates inputs and calls CUDA kernel.
 */
torch::Tensor mapping_forward(
    torch::Tensor input,
    torch::Tensor indices) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(indices);
    
    // Validate shapes
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, input_size)");
    TORCH_CHECK(indices.dim() == 2, "Indices must be 2D (output_size, n)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(
        indices.scalar_type() == torch::kInt16 || 
        indices.scalar_type() == torch::kInt32 ||
        indices.scalar_type() == torch::kShort ||
        indices.scalar_type() == torch::kInt,
        "Indices must be int16 or int32"
    );
    
    return mapping_cuda_forward(input, indices);
}

/**
 * Python-facing wrapper for mapping backward pass.
 * Validates inputs and calls CUDA kernel.
 */
torch::Tensor mapping_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int input_size) {
    
    CHECK_INPUT(grad_output);
    CHECK_INPUT(indices);
    
    // Validate shapes
    TORCH_CHECK(grad_output.dim() == 3, "Grad output must be 3D (batch_size, output_size, n)");
    TORCH_CHECK(indices.dim() == 2, "Indices must be 2D (output_size, n)");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "Grad output must be float32");
    TORCH_CHECK(input_size > 0, "Input size must be positive");
    
    return mapping_cuda_backward(grad_output, indices, input_size);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mapping_forward, "Mapping CUDA forward",
          py::arg("input"), py::arg("indices"));
    m.def("backward", &mapping_backward, "Mapping CUDA backward",
          py::arg("grad_output"), py::arg("indices"), py::arg("input_size"));
}
