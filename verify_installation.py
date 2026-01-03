"""Quick setup verification script."""

import sys
import torch
import warnings
warnings.filterwarnings('ignore')


def check_section(title):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_cuda():
    """Check CUDA availability."""
    check_section("1. CUDA Check")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: True")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("✗ CUDA Not Available")
        print("  Note: Custom kernels require CUDA")
        return False


def check_custom_kernels():
    """Check if custom kernels are compiled."""
    check_section("2. Custom CUDA Kernels")
    
    try:
        from cuda_kernels import CUDA_KERNELS_AVAILABLE, FusedRMSNorm, FusedSiLU
        
        if CUDA_KERNELS_AVAILABLE:
            print("✓ Custom Kernels Compiled: True")
            
            # Quick test
            if torch.cuda.is_available():
                x = torch.randn(4, 1024, device='cuda', dtype=torch.float16)
                
                # Test RMSNorm
                rmsnorm = FusedRMSNorm(1024).cuda().half()
                output = rmsnorm(x)
                print(f"  ✓ RMSNorm test passed (output shape: {output.shape})")
                
                # Test SiLU
                silu = FusedSiLU()
                output = silu(x)
                print(f"  ✓ SiLU test passed (output shape: {output.shape})")
                
            return True
        else:
            print("✓ Module imported, but kernels not compiled")
            print("  Using PyTorch fallback implementations")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import cuda_kernels: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing kernels: {e}")
        return False


def check_vllm():
    """Check vLLM installation."""
    check_section("3. vLLM (Optional)")
    
    try:
        import vllm
        print(f"✓ vLLM Installed: {vllm.__version__}")
        print("  Note: Can run quantized inference")
        return True
    except ImportError:
        print("✗ vLLM Not Installed")
        print("  Install with: pip install vllm")
        print("  Required for quantized inference demos")
        return False


def check_dependencies():
    """Check required dependencies."""
    check_section("4. Dependencies")
    
    required = {
        'torch': torch,
        'numpy': None,
        'pandas': None,
    }
    
    all_ok = True
    for name, module in required.items():
        try:
            if module is None:
                module = __import__(name)
            
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: Not installed")
            all_ok = False
    
    return all_ok


def run_quick_benchmark():
    """Run a quick performance test."""
    check_section("5. Quick Performance Test")
    
    if not torch.cuda.is_available():
        print("⚠ Skipping (CUDA not available)")
        return False
    
    try:
        from cuda_kernels import FusedRMSNorm, CUDA_KERNELS_AVAILABLE
        
        if not CUDA_KERNELS_AVAILABLE:
            print("⚠ Skipping (Custom kernels not compiled)")
            return False
        
        batch_size, hidden_size = 32, 4096
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        
        # Custom kernel
        custom_norm = FusedRMSNorm(hidden_size).cuda().half()
        
        # Warmup
        for _ in range(10):
            _ = custom_norm(x)
        
        torch.cuda.synchronize()
        
        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = custom_norm(x)
        end.record()
        
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / 100
        
        print(f"✓ RMSNorm Benchmark (Batch={batch_size}, Hidden={hidden_size})")
        print(f"  Average Time: {time_ms:.4f} ms")
        print(f"  Throughput: {1000/time_ms:.1f} ops/sec")
        
        if time_ms < 0.1:
            print("  Status: ⚡ Excellent performance!")
        elif time_ms < 0.5:
            print("  Status: ✓ Good performance")
        else:
            print("  Status: ⚠ Performance may vary")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False


def print_summary(results):
    """Print summary of checks."""
    check_section("Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests Passed: {passed}/{total}")
    print("\nStatus by component:")
    
    icons = {True: "✓", False: "✗"}
    for name, status in results.items():
        print(f"  {icons[status]} {name}")
    
    print("\n" + "-"*70)
    
    if results['cuda'] and results['kernels']:
        print("✅ Setup Complete! You can run:")
        print("   • python profiling/benchmark.py")
        print("   • python examples/usage_examples.py")
        if results['vllm']:
            print("   • python inference/vllm_quantized.py")
    elif results['cuda']:
        print("⚠️  CUDA available but kernels not compiled")
        print("   Run: python setup.py install")
    else:
        print("⚠️  CUDA not available - some features disabled")
        print("   Custom kernels require NVIDIA GPU")
    
    print("-"*70 + "\n")


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("  Efficient LLM Inference - Installation Verification")
    print("="*70)
    
    results = {
        'cuda': check_cuda(),
        'kernels': check_custom_kernels(),
        'vllm': check_vllm(),
        'dependencies': check_dependencies(),
        'benchmark': run_quick_benchmark(),
    }
    
    print_summary(results)
    
    # Exit code
    if results['cuda'] and results['kernels'] and results['dependencies']:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some checks failed


if __name__ == "__main__":
    main()
