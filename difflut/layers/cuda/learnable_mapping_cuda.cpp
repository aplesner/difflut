#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA kernels
torch::Tensor learnable_mapping_cuda_forward(
    torch::Tensor input,
    torch::Tensor indices
);

torch::Tensor learnable_mapping_cuda_soft_forward(
    torch::Tensor input,
    torch::Tensor weights,
    float tau
);

torch::Tensor learnable_mapping_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int input_size
);

// Input validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * Python-facing wrapper for learnable mapping hard selection (eval mode).
 */
torch::Tensor learnable_mapping_forward(
    torch::Tensor input,
    torch::Tensor indices) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(indices);
    
    // Validate shapes
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, input_size)");
    TORCH_CHECK(indices.dim() == 1, "Indices must be 1D (output_size,)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(
        indices.scalar_type() == torch::kInt32 || indices.scalar_type() == torch::kInt,
        "Indices must be int32"
    );
    
    return learnable_mapping_cuda_forward(input, indices);
}

/**
 * Python-facing wrapper for learnable mapping soft selection (training mode).
 */
torch::Tensor learnable_mapping_soft_forward(
    torch::Tensor input,
    torch::Tensor weights,
    float tau) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    
    // Validate shapes
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, input_size)");
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D (output_size, input_size)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "Weights must be float32");
    TORCH_CHECK(input.size(1) == weights.size(1), "Input and weights must have same input_size");
    TORCH_CHECK(tau > 0, "Temperature must be positive");
    
    return learnable_mapping_cuda_soft_forward(input, weights, tau);
}

/**
 * Python-facing wrapper for learnable mapping backward pass.
 */
torch::Tensor learnable_mapping_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int input_size) {
    
    CHECK_INPUT(grad_output);
    CHECK_INPUT(indices);
    
    // Validate shapes
    TORCH_CHECK(grad_output.dim() == 2, "Grad output must be 2D (batch_size, output_size)");
    TORCH_CHECK(indices.dim() == 1, "Indices must be 1D (output_size,)");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "Grad output must be float32");
    TORCH_CHECK(input_size > 0, "Input size must be positive");
    
    return learnable_mapping_cuda_backward(grad_output, indices, input_size);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &learnable_mapping_forward, "Learnable mapping CUDA hard selection forward",
          py::arg("input"), py::arg("indices"));
    m.def("soft_forward", &learnable_mapping_soft_forward, "Learnable mapping CUDA soft selection forward",
          py::arg("input"), py::arg("weights"), py::arg("tau"));
    m.def("backward", &learnable_mapping_backward, "Learnable mapping CUDA backward",
          py::arg("grad_output"), py::arg("indices"), py::arg("input_size"));
}
