#include <torch/extension.h>

#include <vector>

torch::Tensor probabilistic_stable_cuda_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  torch::Tensor temperature
);

std::vector<torch::Tensor> probabilistic_stable_cuda_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  torch::Tensor temperature,
  torch::Tensor output_grad,
  torch::Tensor alpha,
  int num_inputs,
  int num_outputs,
  torch::Tensor scale_min,
  torch::Tensor scale_max
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor probabilistic_stable_forward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  torch::Tensor temperature) {
    CHECK_INPUT(input);
    CHECK_INPUT(mapping);
    CHECK_INPUT(luts);
    return probabilistic_stable_cuda_forward(input, mapping, luts, temperature);
};

std::vector<torch::Tensor> probabilistic_stable_backward(
  torch::Tensor input,
  torch::Tensor mapping,
  torch::Tensor luts,
  torch::Tensor temperature,
  torch::Tensor output_grad,
  torch::Tensor alpha,
  int num_inputs,
  int num_outputs,
  torch::Tensor scale_min,
  torch::Tensor scale_max) {
    CHECK_INPUT(input);
    CHECK_INPUT(mapping);
    CHECK_INPUT(luts);
    CHECK_INPUT(output_grad);
    return probabilistic_stable_cuda_backward(input, mapping, luts, temperature, output_grad,
                                             alpha, num_inputs, num_outputs, scale_min, scale_max);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &probabilistic_stable_forward, "Probabilistic Stable CUDA forward");
  m.def("backward", &probabilistic_stable_backward, "Probabilistic Stable CUDA backward with gradient stabilization");
}
