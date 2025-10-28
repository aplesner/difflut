#include <torch/extension.h>

#include <vector>

torch::Tensor dwn_stable_cuda_forward(
  torch::Tensor input,
  torch::Tensor luts
);

std::vector<torch::Tensor> dwn_stable_cuda_backward(
  torch::Tensor input,
  torch::Tensor luts,
  torch::Tensor grad_output,
  float gradient_scale
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dwn_stable_forward(
  torch::Tensor input,
  torch::Tensor luts) {
    CHECK_INPUT(input);
    CHECK_INPUT(luts);
    return dwn_stable_cuda_forward(input, luts);
}

std::vector<torch::Tensor> dwn_stable_backward(
  torch::Tensor input,
  torch::Tensor luts,
  torch::Tensor grad_output,
  float gradient_scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(luts);
    CHECK_INPUT(grad_output);
    return dwn_stable_cuda_backward(input, luts, grad_output, gradient_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dwn_stable_forward, "Gradient Stabilized CUDA forward");
  m.def("backward", &dwn_stable_backward, "Gradient Stabilized CUDA backward");
}
