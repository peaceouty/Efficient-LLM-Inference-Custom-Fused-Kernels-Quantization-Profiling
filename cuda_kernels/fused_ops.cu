/*
 * Optimized Fused Kernels for LLM Inference
 * 
 * This file contains custom CUDA kernels for:
 * 1. RMSNorm: Fused reduction and normalization to minimize global memory reads.
 * 2. SiLU: Fused activation function.
 * 
 * Optimization Strategy:
 * - Block-level reduction for variance calculation.
 * - Register caching for weight parameters.
 * - Coalesced memory access patterns.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Fused RMSNorm kernel for Llama models
// Reduces memory bandwidth via single-pass computation
// Formula: y = x / sqrt(mean(x^2) + eps) * weight
template <typename scalar_t>
__global__ void fused_rmsnorm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const float eps) {
    
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    const scalar_t* x = input + idx * hidden_size;
    scalar_t* y = output + idx * hidden_size;
    
    // Compute mean of squares using shared memory reduction
    extern __shared__ float shared_sum[];
    
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        thread_sum += val * val;
    }
    
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float rms = sqrtf(shared_sum[0] / hidden_size + eps);
    
    // Normalize and apply weight
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y[i] = static_cast<scalar_t>(
            (static_cast<float>(x[i]) / rms) * static_cast<float>(weight[i])
        );
    }
}

// Fused SiLU activation: x * sigmoid(x)
// Single kernel reduces memory traffic
template <typename scalar_t>
__global__ void fused_silu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = static_cast<scalar_t>(x * sigmoid_x);
    }
}

// CUDA kernel for fused GeLU activation
// GeLU is used in many transformer models
template <typename scalar_t>
__global__ void fused_gelu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int n) {
    
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        float tanh_inner = tanhf(inner);
        output[idx] = static_cast<scalar_t>(0.5f * x * (1.0f + tanh_inner));
    }
}

// Fused kernel: Add + LayerNorm
// This reduces memory bandwidth by fusing two operations
template <typename scalar_t>
__global__ void fused_add_layernorm_kernel(
    const scalar_t* __restrict__ input1,
    const scalar_t* __restrict__ input2,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const float eps) {
    
    int idx = blockIdx.x;
    if (idx >= batch_size) return;
    
    const scalar_t* x1 = input1 + idx * hidden_size;
    const scalar_t* x2 = input2 + idx * hidden_size;
    scalar_t* y = output + idx * hidden_size;
    
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sq_sum = shared_data + blockDim.x;
    
    // Step 1: Add inputs and compute mean and variance
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x1[i]) + static_cast<float>(x2[i]);
        thread_sum += val;
        thread_sq_sum += val * val;
    }
    
    shared_sum[threadIdx.x] = thread_sum;
    shared_sq_sum[threadIdx.x] = thread_sq_sum;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / hidden_size;
    float variance = shared_sq_sum[0] / hidden_size - mean * mean;
    float std_inv = rsqrtf(variance + eps);
    
    // Step 2: Normalize and apply affine transformation
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float added = static_cast<float>(x1[i]) + static_cast<float>(x2[i]);
        float normalized = (added - mean) * std_inv;
        y[i] = static_cast<scalar_t>(
            normalized * static_cast<float>(weight[i]) + static_cast<float>(bias[i])
        );
    }
}

// C++ wrapper functions
torch::Tensor fused_rmsnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps) {
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_rmsnorm_cuda", ([&] {
        fused_rmsnorm_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            eps
        );
    }));
    
    return output;
}

torch::Tensor fused_silu_cuda(torch::Tensor input) {
    const int n = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_silu_cuda", ([&] {
        fused_silu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            n
        );
    }));
    
    return output;
}

torch::Tensor fused_gelu_cuda(torch::Tensor input) {
    const int n = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_gelu_cuda", ([&] {
        fused_gelu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            n
        );
    }));
    
    return output;
}

torch::Tensor fused_add_layernorm_cuda(
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    const auto batch_size = input1.size(0);
    const auto hidden_size = input1.size(1);
    
    auto output = torch::empty_like(input1);
    
    const int threads = 256;
    const int blocks = batch_size;
    const int shared_mem = 2 * threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "fused_add_layernorm_cuda", ([&] {
        fused_add_layernorm_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            input1.data_ptr<scalar_t>(),
            input2.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            eps
        );
    }));
    
    return output;
}
