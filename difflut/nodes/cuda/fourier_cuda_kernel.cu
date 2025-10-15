#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// CUDA kernel for Fourier forward pass
// Forward: Continuous Fourier computation
__global__ void fourier_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ frequencies,
    const float* __restrict__ amplitudes,
    const float* __restrict__ phases,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int num_inputs,
    const int num_frequencies,
    const int output_dim,
    const float max_amplitude) {
    
    // Each thread handles one (batch, output_dim) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / output_dim;
    const int dim_idx = idx % output_dim;
    
    if (batch_idx >= batch_size || dim_idx >= output_dim) return;
    
    const float PI = 3.14159265358979323846f;
    const float eps = 1e-8f;
    
    // Apply sigmoid to input to ensure [0, 1] range
    float x_sigmoid[32];  // Assuming max num_inputs <= 32
    for (int i = 0; i < num_inputs; i++) {
        float x_val = input[batch_idx * num_inputs + i];
        x_sigmoid[i] = 1.0f / (1.0f + expf(-x_val));
    }
    
    // Compute amplitude normalization
    float amplitude_sum = 0.0f;
    for (int k = 0; k < num_frequencies; k++) {
        amplitude_sum += amplitudes[k * output_dim + dim_idx];
    }
    float amplitude_scale = max_amplitude / (amplitude_sum + eps);
    
    // Compute Fourier sum: Σ_k amplitude_k * cos(2π * <k, x> + phase_k)
    float fourier_sum = 0.0f;
    for (int k = 0; k < num_frequencies; k++) {
        // Compute dot product <k, x>
        float dot_product = 0.0f;
        for (int i = 0; i < num_inputs; i++) {
            dot_product += frequencies[k * num_inputs + i] * x_sigmoid[i];
        }
        
        // Compute angle: 2π * <k, x> + phase_k
        float angle = 2.0f * PI * dot_product + phases[k * output_dim + dim_idx];
        
        // Get normalized amplitude
        float amp = amplitudes[k * output_dim + dim_idx] * amplitude_scale;
        
        // Add contribution
        fourier_sum += amp * cosf(angle);
    }
    
    // Add bias and clamp to [0, 1]
    float result = fourier_sum + bias[dim_idx];
    result = fmaxf(0.0f, fminf(1.0f, result));
    
    output[batch_idx * output_dim + dim_idx] = result;
}

// CUDA kernel for Fourier forward pass during evaluation using Heaviside
// Forward: Uses Heaviside thresholding on input
__global__ void fourier_forward_eval_kernel(
    const float* __restrict__ input,
    const float* __restrict__ frequencies,
    const float* __restrict__ amplitudes,
    const float* __restrict__ phases,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int num_inputs,
    const int num_frequencies,
    const int output_dim,
    const float max_amplitude) {
    
    // Each thread handles one (batch, output_dim) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / output_dim;
    const int dim_idx = idx % output_dim;
    
    if (batch_idx >= batch_size || dim_idx >= output_dim) return;
    
    const float PI = 3.14159265358979323846f;
    const float eps = 1e-8f;
    
    // Apply Heaviside step function to input (binary threshold at 0.5)
    float x_heaviside[32];  // Assuming max num_inputs <= 32
    for (int i = 0; i < num_inputs; i++) {
        float x_val = input[batch_idx * num_inputs + i];
        // Heaviside: 1 if x > 0.5, else 0
        x_heaviside[i] = (x_val > 0.5f) ? 1.0f : 0.0f;
    }
    
    // Compute amplitude normalization
    float amplitude_sum = 0.0f;
    for (int k = 0; k < num_frequencies; k++) {
        amplitude_sum += amplitudes[k * output_dim + dim_idx];
    }
    float amplitude_scale = max_amplitude / (amplitude_sum + eps);
    
    // Compute Fourier sum: Σ_k amplitude_k * cos(2π * <k, x> + phase_k)
    float fourier_sum = 0.0f;
    for (int k = 0; k < num_frequencies; k++) {
        // Compute dot product <k, x>
        float dot_product = 0.0f;
        for (int i = 0; i < num_inputs; i++) {
            dot_product += frequencies[k * num_inputs + i] * x_heaviside[i];
        }
        
        // Compute angle: 2π * <k, x> + phase_k
        float angle = 2.0f * PI * dot_product + phases[k * output_dim + dim_idx];
        
        // Get normalized amplitude
        float amp = amplitudes[k * output_dim + dim_idx] * amplitude_scale;
        
        // Add contribution
        fourier_sum += amp * cosf(angle);
    }
    
    // Add bias and clamp to [0, 1]
    float result = fourier_sum + bias[dim_idx];
    result = fmaxf(0.0f, fminf(1.0f, result));
    
    output[batch_idx * output_dim + dim_idx] = result;
}

// CUDA kernel for Fourier backward pass - input gradients
__global__ void fourier_backward_input_kernel(
    const float* __restrict__ input,
    const float* __restrict__ frequencies,
    const float* __restrict__ amplitudes,
    const float* __restrict__ phases,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int batch_size,
    const int num_inputs,
    const int num_frequencies,
    const int output_dim,
    const float max_amplitude) {
    
    // Each thread handles one (batch, input) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / num_inputs;
    const int input_idx = idx % num_inputs;
    
    if (batch_idx >= batch_size || input_idx >= num_inputs) return;
    
    const float PI = 3.14159265358979323846f;
    const float eps = 1e-8f;
    
    // Apply sigmoid to input
    float x_sigmoid[32];
    for (int i = 0; i < num_inputs; i++) {
        float x_val = input[batch_idx * num_inputs + i];
        x_sigmoid[i] = 1.0f / (1.0f + expf(-x_val));
    }
    
    float grad_sum = 0.0f;
    
    // For each output dimension
    for (int dim_idx = 0; dim_idx < output_dim; dim_idx++) {
        // Compute amplitude normalization
        float amplitude_sum = 0.0f;
        for (int k = 0; k < num_frequencies; k++) {
            amplitude_sum += amplitudes[k * output_dim + dim_idx];
        }
        float amplitude_scale = max_amplitude / (amplitude_sum + eps);
        
        // For each frequency that uses this input
        for (int k = 0; k < num_frequencies; k++) {
            float freq_val = frequencies[k * num_inputs + input_idx];
            
            if (fabsf(freq_val) < eps) continue;  // Skip if frequency doesn't use this input
            
            // Compute dot product <k, x>
            float dot_product = 0.0f;
            for (int i = 0; i < num_inputs; i++) {
                dot_product += frequencies[k * num_inputs + i] * x_sigmoid[i];
            }
            
            // Compute angle
            float angle = 2.0f * PI * dot_product + phases[k * output_dim + dim_idx];
            
            // Get normalized amplitude
            float amp = amplitudes[k * output_dim + dim_idx] * amplitude_scale;
            
            // Derivative: -amplitude * 2π * k_i * sin(angle)
            float deriv = -amp * 2.0f * PI * freq_val * sinf(angle);
            
            // Chain with sigmoid derivative
            float sigmoid_deriv = x_sigmoid[input_idx] * (1.0f - x_sigmoid[input_idx]);
            
            // Accumulate gradient
            grad_sum += deriv * sigmoid_deriv * grad_output[batch_idx * output_dim + dim_idx];
        }
    }
    
    grad_input[batch_idx * num_inputs + input_idx] = grad_sum;
}

// CUDA kernel for Fourier backward pass - amplitude gradients
__global__ void fourier_backward_amplitude_kernel(
    const float* __restrict__ input,
    const float* __restrict__ frequencies,
    const float* __restrict__ amplitudes,
    const float* __restrict__ phases,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_amplitudes,
    const int batch_size,
    const int num_inputs,
    const int num_frequencies,
    const int output_dim,
    const float max_amplitude) {
    
    // Each thread handles one (frequency, output_dim) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int freq_idx = idx / output_dim;
    const int dim_idx = idx % output_dim;
    
    if (freq_idx >= num_frequencies || dim_idx >= output_dim) return;
    
    const float PI = 3.14159265358979323846f;
    const float eps = 1e-8f;
    
    // Compute amplitude normalization
    float amplitude_sum = 0.0f;
    for (int k = 0; k < num_frequencies; k++) {
        amplitude_sum += amplitudes[k * output_dim + dim_idx];
    }
    float amplitude_scale = max_amplitude / (amplitude_sum + eps);
    
    float grad_sum = 0.0f;
    
    // Sum over batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Apply sigmoid to input
        float dot_product = 0.0f;
        for (int i = 0; i < num_inputs; i++) {
            float x_val = input[batch_idx * num_inputs + i];
            float x_sig = 1.0f / (1.0f + expf(-x_val));
            dot_product += frequencies[freq_idx * num_inputs + i] * x_sig;
        }
        
        // Compute angle
        float angle = 2.0f * PI * dot_product + phases[freq_idx * output_dim + dim_idx];
        
        // Derivative w.r.t. amplitude (considering normalization)
        float deriv = amplitude_scale * cosf(angle);
        
        // Accumulate gradient
        grad_sum += deriv * grad_output[batch_idx * output_dim + dim_idx];
    }
    
    grad_amplitudes[freq_idx * output_dim + dim_idx] = grad_sum;
}

// CUDA kernel for Fourier backward pass - phase gradients
__global__ void fourier_backward_phase_kernel(
    const float* __restrict__ input,
    const float* __restrict__ frequencies,
    const float* __restrict__ amplitudes,
    const float* __restrict__ phases,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_phases,
    const int batch_size,
    const int num_inputs,
    const int num_frequencies,
    const int output_dim,
    const float max_amplitude) {
    
    // Each thread handles one (frequency, output_dim) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int freq_idx = idx / output_dim;
    const int dim_idx = idx % output_dim;
    
    if (freq_idx >= num_frequencies || dim_idx >= output_dim) return;
    
    const float PI = 3.14159265358979323846f;
    const float eps = 1e-8f;
    
    // Compute amplitude normalization
    float amplitude_sum = 0.0f;
    for (int k = 0; k < num_frequencies; k++) {
        amplitude_sum += amplitudes[k * output_dim + dim_idx];
    }
    float amplitude_scale = max_amplitude / (amplitude_sum + eps);
    
    float grad_sum = 0.0f;
    
    // Sum over batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Apply sigmoid to input
        float dot_product = 0.0f;
        for (int i = 0; i < num_inputs; i++) {
            float x_val = input[batch_idx * num_inputs + i];
            float x_sig = 1.0f / (1.0f + expf(-x_val));
            dot_product += frequencies[freq_idx * num_inputs + i] * x_sig;
        }
        
        // Compute angle
        float angle = 2.0f * PI * dot_product + phases[freq_idx * output_dim + dim_idx];
        
        // Get normalized amplitude
        float amp = amplitudes[freq_idx * output_dim + dim_idx] * amplitude_scale;
        
        // Derivative w.r.t. phase: -amplitude * sin(angle)
        float deriv = -amp * sinf(angle);
        
        // Accumulate gradient
        grad_sum += deriv * grad_output[batch_idx * output_dim + dim_idx];
    }
    
    grad_phases[freq_idx * output_dim + dim_idx] = grad_sum;
}

// CUDA kernel for Fourier backward pass - bias gradients
__global__ void fourier_backward_bias_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    const int batch_size,
    const int output_dim) {
    
    const int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dim_idx >= output_dim) return;
    
    float grad_sum = 0.0f;
    
    // Sum over batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        grad_sum += grad_output[batch_idx * output_dim + dim_idx];
    }
    
    grad_bias[dim_idx] = grad_sum;
}

// C++ interface

torch::Tensor fourier_cuda_forward(
    torch::Tensor input,
    torch::Tensor frequencies,
    torch::Tensor amplitudes,
    torch::Tensor phases,
    torch::Tensor bias,
    float max_amplitude) {
    
    const int batch_size = input.size(0);
    const int num_inputs = input.size(1);
    const int num_frequencies = frequencies.size(0);
    const int output_dim = bias.size(0);
    
    auto output = torch::zeros({batch_size, output_dim}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * output_dim + threads - 1) / threads;
    
    fourier_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        frequencies.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        phases.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_inputs,
        num_frequencies,
        output_dim,
        max_amplitude
    );
    
    return output;
}

torch::Tensor fourier_cuda_forward_eval(
    torch::Tensor input,
    torch::Tensor frequencies,
    torch::Tensor amplitudes,
    torch::Tensor phases,
    torch::Tensor bias,
    float max_amplitude) {
    
    const int batch_size = input.size(0);
    const int num_inputs = input.size(1);
    const int num_frequencies = frequencies.size(0);
    const int output_dim = bias.size(0);
    
    auto output = torch::zeros({batch_size, output_dim}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * output_dim + threads - 1) / threads;
    
    fourier_forward_eval_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        frequencies.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        phases.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_inputs,
        num_frequencies,
        output_dim,
        max_amplitude
    );
    
    return output;
}

std::vector<torch::Tensor> fourier_cuda_backward(
    torch::Tensor input,
    torch::Tensor frequencies,
    torch::Tensor amplitudes,
    torch::Tensor phases,
    torch::Tensor grad_output,
    float max_amplitude) {
    
    const int batch_size = input.size(0);
    const int num_inputs = input.size(1);
    const int num_frequencies = frequencies.size(0);
    const int output_dim = grad_output.size(1);
    
    auto grad_input = torch::zeros_like(input);
    auto grad_amplitudes = torch::zeros_like(amplitudes);
    auto grad_phases = torch::zeros_like(phases);
    auto grad_bias = torch::zeros({output_dim}, input.options());
    
    const int threads = 256;
    
    // Compute input gradients
    {
        const int blocks = (batch_size * num_inputs + threads - 1) / threads;
        fourier_backward_input_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            frequencies.data_ptr<float>(),
            amplitudes.data_ptr<float>(),
            phases.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size,
            num_inputs,
            num_frequencies,
            output_dim,
            max_amplitude
        );
    }
    
    // Compute amplitude gradients
    {
        const int blocks = (num_frequencies * output_dim + threads - 1) / threads;
        fourier_backward_amplitude_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            frequencies.data_ptr<float>(),
            amplitudes.data_ptr<float>(),
            phases.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_amplitudes.data_ptr<float>(),
            batch_size,
            num_inputs,
            num_frequencies,
            output_dim,
            max_amplitude
        );
    }
    
    // Compute phase gradients
    {
        const int blocks = (num_frequencies * output_dim + threads - 1) / threads;
        fourier_backward_phase_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            frequencies.data_ptr<float>(),
            amplitudes.data_ptr<float>(),
            phases.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_phases.data_ptr<float>(),
            batch_size,
            num_inputs,
            num_frequencies,
            output_dim,
            max_amplitude
        );
    }
    
    // Compute bias gradients
    {
        const int blocks = (output_dim + threads - 1) / threads;
        fourier_backward_bias_kernel<<<blocks, threads>>>(
            grad_output.data_ptr<float>(),
            grad_bias.data_ptr<float>(),
            batch_size,
            output_dim
        );
    }
    
    return {grad_input, grad_amplitudes, grad_phases, grad_bias};
}
