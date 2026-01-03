"""Custom CUDA fused kernels for efficient LLM inference."""

import torch
import torch.nn as nn
from typing import Optional

try:
    import fused_kernels
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False
    print("Warning: Custom CUDA kernels not available. Using PyTorch native operations.")


class FusedRMSNorm(nn.Module):
    """Fused RMSNorm for Llama models. Single-kernel normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if CUDA_KERNELS_AVAILABLE and x.is_cuda:
            return fused_kernels.fused_rmsnorm(x, self.weight, self.eps)
        else:
            # Fallback to PyTorch implementation
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x


class FusedSiLU(nn.Module):
    """Fused SiLU activation: x * sigmoid(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if CUDA_KERNELS_AVAILABLE and x.is_cuda:
            return fused_kernels.fused_silu(x)
        else:
            # Fallback to PyTorch
            return x * torch.sigmoid(x)


class FusedGELU(nn.Module):
    """Fused GeLU activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if CUDA_KERNELS_AVAILABLE and x.is_cuda:
            return fused_kernels.fused_gelu(x)
        else:
            # Fallback to PyTorch
            return torch.nn.functional.gelu(x)


class FusedAddLayerNorm(nn.Module):
    """Fused Add + LayerNorm for transformer residual connections."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if CUDA_KERNELS_AVAILABLE and x1.is_cuda:
            return fused_kernels.fused_add_layernorm(
                x1, x2, self.weight, self.bias, self.eps
            )
        else:
            # Fallback to PyTorch
            x = x1 + x2
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight * x + self.bias


def benchmark_kernel(kernel_fn, pytorch_fn, input_shape, num_iterations=1000):
    """Benchmark custom kernel vs PyTorch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(input_shape, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = kernel_fn(x)
        _ = pytorch_fn(x)
    
    torch.cuda.synchronize()
    
    # Benchmark custom kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        _ = kernel_fn(x)
    end_event.record()
    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event)
    
    # Benchmark PyTorch
    start_event.record()
    for _ in range(num_iterations):
        _ = pytorch_fn(x)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event)
    
    speedup = pytorch_time / custom_time
    
    return {
        'custom_time_ms': custom_time,
        'pytorch_time_ms': pytorch_time,
        'speedup': speedup
    }


if __name__ == "__main__":
    print(f"CUDA Kernels Available: {CUDA_KERNELS_AVAILABLE}")
    
    if CUDA_KERNELS_AVAILABLE:
        print("\n=== Benchmarking Custom Kernels ===\n")
        
        # Test RMSNorm
        print("Testing Fused RMSNorm...")
        rmsnorm = FusedRMSNorm(4096).cuda()
        def pytorch_rmsnorm(x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6)
            return rmsnorm.weight * x
        
        results = benchmark_kernel(
            lambda x: rmsnorm(x),
            pytorch_rmsnorm,
            (32, 4096)
        )
        print(f"  Custom: {results['custom_time_ms']:.2f}ms")
        print(f"  PyTorch: {results['pytorch_time_ms']:.2f}ms")
        print(f"  Speedup: {results['speedup']:.2f}x\n")
        
        # Test SiLU
        print("Testing Fused SiLU...")
        silu = FusedSiLU()
        results = benchmark_kernel(
            lambda x: silu(x),
            lambda x: x * torch.sigmoid(x),
            (32, 4096)
        )
        print(f"  Custom: {results['custom_time_ms']:.2f}ms")
        print(f"  PyTorch: {results['pytorch_time_ms']:.2f}ms")
        print(f"  Speedup: {results['speedup']:.2f}x\n")
