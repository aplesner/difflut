#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor efd_fused_cuda_forward(
    torch::Tensor input,
    torch::Tensor mapping,
    torch::Tensor luts);

std::vector<torch::Tensor> efd_fused_cuda_backward(
    torch::Tensor input,
    torch::Tensor mapping,
    torch::Tensor luts,
    torch::Tensor grad_output,
    float alpha,
    float beta);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efd_fused_cuda_forward,
          "EFD fused forward (CUDA) - performs mapping and LUT lookup in single kernel",
          py::arg("input"),
          py::arg("mapping"),
          py::arg("luts"));

    m.def("backward", &efd_fused_cuda_backward,
          "EFD fused backward (CUDA) - computes gradients with on-the-fly mapping",
          py::arg("input"),
          py::arg("mapping"),
          py::arg("luts"),
          py::arg("grad_output"),
          py::arg("alpha"),
          py::arg("beta"));
}
