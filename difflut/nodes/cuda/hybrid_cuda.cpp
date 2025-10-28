#include <torch/extension.h>

#include <vector>

torch::Tensor hybrid_cuda_forward(
  torch::Tensor input,
  torch::Tensor luts
);

std::vector<torch::Tensor> hybrid_cuda_backward(
  torch::Tensor input,
  torch::Tensor luts,
  torch::Tensor binary_combinations,
  torch::Tensor grad_output
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor hybrid_forward(
  torch::Tensor input,
  torch::Tensor luts) {
    CHECK_INPUT(input);
    CHECK_INPUT(luts);
    return hybrid_cuda_forward(input, luts);
};

std::vector<torch::Tensor> hybrid_backward(
  torch::Tensor input,
  torch::Tensor luts,
  torch::Tensor binary_combinations,
  torch::Tensor grad_output) {
    CHECK_INPUT(input);
    CHECK_INPUT(luts);
    CHECK_INPUT(binary_combinations);
    CHECK_INPUT(grad_output);
    return hybrid_cuda_backward(input, luts, binary_combinations, grad_output);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hybrid_forward, "Hybrid CUDA forward");
  m.def("backward", &hybrid_backward, "Hybrid CUDA backward");
}
