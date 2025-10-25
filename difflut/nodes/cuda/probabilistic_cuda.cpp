#include <torch/extension.h>

#include <vector>

torch::Tensor probabilistic_cuda_forward(
  torch::Tensor input,
  torch::Tensor raw_weights,
  float temperature
);

std::vector<torch::Tensor> probabilistic_cuda_backward(
  torch::Tensor input,
  torch::Tensor raw_weights,
  float temperature,
  torch::Tensor grad_output
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor probabilistic_forward(
  torch::Tensor input,
  torch::Tensor raw_weights,
  float temperature) {
    CHECK_INPUT(input);
    CHECK_INPUT(raw_weights);
    return probabilistic_cuda_forward(input, raw_weights, temperature);
};

std::vector<torch::Tensor> probabilistic_backward(
  torch::Tensor input,
  torch::Tensor raw_weights,
  float temperature,
  torch::Tensor grad_output) {
    CHECK_INPUT(input);
    CHECK_INPUT(raw_weights);
    CHECK_INPUT(grad_output);
    return probabilistic_cuda_backward(input, raw_weights, temperature, grad_output);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &probabilistic_forward, "Probabilistic CUDA forward");
  m.def("backward", &probabilistic_backward, "Probabilistic CUDA backward");
}
}
