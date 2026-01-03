"""Usage examples for the inference pipeline."""

import torch
from cuda_kernels import FusedRMSNorm, FusedSiLU


def simple_transformer_layer_example():
    """
    Simplified transformer layer using custom kernels.
    Shows how to integrate fused operations.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Transformer Layer with Custom Kernels")
    print("="*80)
    
    batch_size = 16
    seq_length = 128
    hidden_size = 1024
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Input tensor
    x = torch.randn(batch_size, seq_length, hidden_size, device=device, dtype=torch.float16)
    
    # Layer components with custom kernels
    rmsnorm = FusedRMSNorm(hidden_size).to(device).half()
    silu = FusedSiLU()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Device: {device}")
    print(f"Dtype: {x.dtype}\n")
    
    # Forward pass timing
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Simulate transformer operations
    # 1. Pre-normalization
    x_norm = rmsnorm(x.view(-1, hidden_size)).view(batch_size, seq_length, hidden_size)
    
    # 2. Feed-forward with SiLU activation
    ff_weight = torch.randn(hidden_size, hidden_size * 4, device=device, dtype=torch.float16)
    ff_output = torch.matmul(x_norm, ff_weight)
    ff_activated = silu(ff_output)
    
    # 3. Project back
    ff_weight2 = torch.randn(hidden_size * 4, hidden_size, device=device, dtype=torch.float16)
    output = torch.matmul(ff_activated, ff_weight2)
    
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    
    print("Results:")
    print(f"  Forward pass time: {elapsed_ms:.2f} ms")
    print(f"  Output shape: {output.shape}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print("="*80 + "\n")


def quantization_example():
    """
    Show model size reduction with quantization.
    """
    from inference.quantization_utils import compare_quantization_methods, analyze_model_size
    
    print("\nRunning quantization analysis...")
    analyze_model_size("llama-3-8b", quantization="awq")
    compare_quantization_methods()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("EFFICIENT LLM INFERENCE - USAGE EXAMPLES")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. Some examples require GPU.")
        return
    
    # Example 1: Transformer layer
    simple_transformer_layer_example()
    
    # Example 2: Quantization analysis
    quantization_example()
    
    print("\nℹ️  For more examples, see:")
    print("  - profiling/benchmark.py: Performance benchmarks")
    print("  - inference/vllm_quantized.py: Full inference pipeline")


if __name__ == "__main__":
    main()
