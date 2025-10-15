#include <torch/extension.h>

#include <vector>

torch::Tensor gradient_stabilized_cuda_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts
);

std::vector<torch::Tensor> gradient_stabilized_cuda_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  torch::Tensor output_grad,
  float gradient_scale
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gradient_stabilized_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts) {
    CHECK_INPUT(input);
    CHECK_INPUT(mapping);
    CHECK_INPUT(luts);
    return gradient_stabilized_cuda_forward(input, mapping, luts);
}

std::vector<torch::Tensor> gradient_stabilized_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  torch::Tensor output_grad,
  float gradient_scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(mapping);
    CHECK_INPUT(luts);
    CHECK_INPUT(output_grad);
    return gradient_stabilized_cuda_backward(input, mapping, luts, output_grad, gradient_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gradient_stabilized_forward, "Gradient Stabilized CUDA forward");
  m.def("backward", &gradient_stabilized_backward, "Gradient Stabilized CUDA backward");
}
