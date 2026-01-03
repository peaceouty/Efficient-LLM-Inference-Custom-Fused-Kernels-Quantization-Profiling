"""Unit tests for custom CUDA kernels."""

import torch
import pytest
from cuda_kernels import (
    FusedRMSNorm,
    FusedSiLU,
    FusedGELU,
    FusedAddLayerNorm,
    CUDA_KERNELS_AVAILABLE
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCustomKernels:
    """Test custom CUDA kernel implementations."""
    
    def test_rmsnorm_correctness(self):
        """Verify RMSNorm output matches PyTorch reference."""
        batch_size, hidden_size = 4, 1024
        eps = 1e-6
        
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float32)
        
        # Custom implementation
        custom_norm = FusedRMSNorm(hidden_size, eps=eps).cuda()
        custom_output = custom_norm(x)
        
        # Reference implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        reference_output = x * torch.rsqrt(variance + eps) * custom_norm.weight
        
        # Check outputs match within tolerance
        torch.testing.assert_close(
            custom_output,
            reference_output,
            rtol=1e-3,
            atol=1e-5
        )
    
    def test_silu_correctness(self):
        """Verify SiLU activation matches PyTorch."""
        x = torch.randn(128, 2048, device='cuda', dtype=torch.float32)
        
        # Custom implementation
        custom_silu = FusedSiLU()
        custom_output = custom_silu(x)
        
        # Reference
        reference_output = x * torch.sigmoid(x)
        
        torch.testing.assert_close(
            custom_output,
            reference_output,
            rtol=1e-4,
            atol=1e-6
        )
    
    def test_gelu_correctness(self):
        """Verify GeLU activation matches PyTorch."""
        x = torch.randn(64, 4096, device='cuda', dtype=torch.float32)
        
        custom_gelu = FusedGELU()
        custom_output = custom_gelu(x)
        
        reference_output = torch.nn.functional.gelu(x)
        
        torch.testing.assert_close(
            custom_output,
            reference_output,
            rtol=1e-3,
            atol=1e-5
        )
    
    def test_add_layernorm_correctness(self):
        """Verify fused residual + LayerNorm."""
        batch_size, hidden_size = 8, 2048
        eps = 1e-5
        
        x1 = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float32)
        x2 = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float32)
        
        # Custom implementation
        custom_norm = FusedAddLayerNorm(hidden_size, eps=eps).cuda()
        custom_output = custom_norm(x1, x2)
        
        # Reference
        x = x1 + x2
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        reference_output = custom_norm.weight * x_normalized + custom_norm.bias
        
        torch.testing.assert_close(
            custom_output,
            reference_output,
            rtol=1e-3,
            atol=1e-5
        )
    
    @pytest.mark.skipif(not CUDA_KERNELS_AVAILABLE, reason="Custom kernels not compiled")
    def test_rmsnorm_performance(self):
        """Benchmark RMSNorm performance."""
        batch_size, hidden_size = 32, 4096
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        
        custom_norm = FusedRMSNorm(hidden_size).cuda().half()
        
        # Warmup
        for _ in range(100):
            _ = custom_norm(x)
        
        torch.cuda.synchronize()
        
        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(1000):
            _ = custom_norm(x)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        
        # Should be faster than 5ms for 1000 iterations
        assert elapsed_ms < 5000, f"Performance regression: {elapsed_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
