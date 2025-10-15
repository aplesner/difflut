#include <torch/extension.h>

#include <vector>

torch::Tensor fourier_cuda_forward(
  torch::Tensor input,
  torch::Tensor frequencies,
  torch::Tensor amplitudes,
  torch::Tensor phases,
  torch::Tensor bias,
  float max_amplitude
);

torch::Tensor fourier_cuda_forward_eval(
  torch::Tensor input,
  torch::Tensor frequencies,
  torch::Tensor amplitudes,
  torch::Tensor phases,
  torch::Tensor bias,
  float max_amplitude
);

std::vector<torch::Tensor> fourier_cuda_backward(
  torch::Tensor input,
  torch::Tensor frequencies,
  torch::Tensor amplitudes,
  torch::Tensor phases,
  torch::Tensor grad_output,
  float max_amplitude
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fourier_forward(
  torch::Tensor input,
  torch::Tensor frequencies,
  torch::Tensor amplitudes,
  torch::Tensor phases,
  torch::Tensor bias,
  float max_amplitude) {
    CHECK_INPUT(input);
    CHECK_INPUT(frequencies);
    CHECK_INPUT(amplitudes);
    CHECK_INPUT(phases);
    CHECK_INPUT(bias);
    return fourier_cuda_forward(input, frequencies, amplitudes, phases, bias, max_amplitude);
};

torch::Tensor fourier_forward_eval(
  torch::Tensor input,
  torch::Tensor frequencies,
  torch::Tensor amplitudes,
  torch::Tensor phases,
  torch::Tensor bias,
  float max_amplitude) {
    CHECK_INPUT(input);
    CHECK_INPUT(frequencies);
    CHECK_INPUT(amplitudes);
    CHECK_INPUT(phases);
    CHECK_INPUT(bias);
    return fourier_cuda_forward_eval(input, frequencies, amplitudes, phases, bias, max_amplitude);
};

std::vector<torch::Tensor> fourier_backward(
  torch::Tensor input,
  torch::Tensor frequencies,
  torch::Tensor amplitudes,
  torch::Tensor phases,
  torch::Tensor grad_output,
  float max_amplitude) {
    CHECK_INPUT(input);
    CHECK_INPUT(frequencies);
    CHECK_INPUT(amplitudes);
    CHECK_INPUT(phases);
    CHECK_INPUT(grad_output);
    return fourier_cuda_backward(input, frequencies, amplitudes, phases, grad_output, max_amplitude);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fourier_forward, "Fourier CUDA forward");
  m.def("forward_eval", &fourier_forward_eval, "Fourier CUDA forward eval with Heaviside");
  m.def("backward", &fourier_backward, "Fourier CUDA backward");
}
