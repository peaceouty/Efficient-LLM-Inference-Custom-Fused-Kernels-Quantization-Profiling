"""Performance profiling for LLM inference kernels."""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
import time
from typing import Dict, List, Callable
import pandas as pd
from pathlib import Path


class PerformanceProfiler:
    """Unified profiler for kernel performance analysis."""
    
    def __init__(self, output_dir: str = "./profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def profile_kernel(
        self,
        kernel_fn: Callable,
        input_data: torch.Tensor,
        name: str,
        num_iterations: int = 1000,
        warmup_iterations: int = 100
    ) -> Dict:
        """Profile single kernel with CUDA events."""
        device = input_data.device
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = kernel_fn(input_data)
        
        torch.cuda.synchronize()
        
        # Timing with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iterations):
            output = kernel_fn(input_data)
        end_event.record()
        
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_iterations
        
        # Memory stats
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        
        results = {
            'kernel_name': name,
            'avg_latency_ms': avg_time_ms,
            'total_time_ms': elapsed_time_ms,
            'num_iterations': num_iterations,
            'throughput_ops_sec': 1000 / avg_time_ms,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'input_shape': tuple(input_data.shape),
            'dtype': str(input_data.dtype)
        }
        
        self.results.append(results)
        return results
    
    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        input_data: torch.Tensor,
        num_iterations: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple implementations of the same operation.
        
        Args:
            implementations: Dict mapping names to functions
            input_data: Input tensor
            num_iterations: Number of iterations
        
        Returns:
            Pandas DataFrame with comparison results
        """
        print("\n" + "="*80)
        print(f"COMPARING {len(implementations)} IMPLEMENTATIONS")
        print("="*80)
        
        results = []
        for name, impl_fn in implementations.items():
            print(f"\nProfiling: {name}...")
            result = self.profile_kernel(
                impl_fn,
                input_data,
                name,
                num_iterations=num_iterations
            )
            results.append(result)
            print(f"  Average Time: {result['avg_time_ms']:.4f} ms")
        
        df = pd.DataFrame(results)
        
        # Calculate speedup relative to baseline (first implementation)
        baseline_time = df.iloc[0]['avg_time_ms']
        df['speedup'] = baseline_time / df['avg_time_ms']
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(df[['kernel_name', 'avg_latency_ms', 'speedup', 'memory_allocated_mb']].to_string(index=False))
        print("="*80 + "\n")
        
        return df
    
    def profile_with_pytorch_profiler(
        self,
        fn: Callable,
        input_data: torch.Tensor,
        name: str
    ):
        """
        Use PyTorch Profiler for detailed kernel analysis.
        
        Args:
            fn: Function to profile
            input_data: Input tensor
            name: Profile name
        """
        print(f"\nRunning PyTorch Profiler for: {name}")
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function(name):
                for _ in range(10):
                    _ = fn(input_data)
        
        # Print profiler results
        print("\n" + "="*80)
        print(f"PYTORCH PROFILER RESULTS: {name}")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))
        
        # Export trace
        trace_path = self.output_dir / f"{name}_trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\nTrace exported to: {trace_path}")
        print("View in Chrome: chrome://tracing")
        print("="*80 + "\n")
    
    def save_results(self, filename: str = "benchmark_results.csv"):
        """Save all profiling results to CSV."""
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}\n")


def benchmark_rmsnorm():
    """Benchmark RMSNorm implementations."""
    from cuda_kernels import FusedRMSNorm, CUDA_KERNELS_AVAILABLE
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    print("\n" + "="*80)
    print("BENCHMARKING: RMSNorm (Llama Normalization Layer)")
    print("="*80)
    
    profiler = PerformanceProfiler()
    
    # Test configurations
    configs = [
        (32, 4096, "Batch=32, Hidden=4096"),
        (64, 4096, "Batch=64, Hidden=4096"),
        (128, 4096, "Batch=128, Hidden=4096"),
    ]
    
    for batch_size, hidden_size, config_name in configs:
        print(f"\n--- Configuration: {config_name} ---")
        
        input_tensor = torch.randn(
            batch_size, hidden_size,
            device='cuda',
            dtype=torch.float16
        )
        
        # Custom implementation
        custom_rmsnorm = FusedRMSNorm(hidden_size).cuda().half()
        
        # PyTorch baseline
        class PyTorchRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.eps = eps
            
            def forward(self, x):
                variance = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(variance + self.eps)
                return self.weight * x
        
        pytorch_rmsnorm = PyTorchRMSNorm(hidden_size).cuda().half()
        
        implementations = {
            'PyTorch (Baseline)': lambda x: pytorch_rmsnorm(x),
        }
        
        if CUDA_KERNELS_AVAILABLE:
            implementations['Custom CUDA Kernel'] = lambda x: custom_rmsnorm(x)
        
        df = profiler.compare_implementations(
            implementations,
            input_tensor,
            num_iterations=1000
        )
    
    profiler.save_results("rmsnorm_benchmark.csv")


def benchmark_silu():
    """Benchmark SiLU activation implementations."""
    from cuda_kernels import FusedSiLU, CUDA_KERNELS_AVAILABLE
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    print("\n" + "="*80)
    print("BENCHMARKING: SiLU Activation (x * sigmoid(x))")
    print("="*80)
    
    profiler = PerformanceProfiler()
    
    input_tensor = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
    
    implementations = {
        'PyTorch (Baseline)': lambda x: x * torch.sigmoid(x),
        'PyTorch SiLU': lambda x: nn.functional.silu(x),
    }
    
    if CUDA_KERNELS_AVAILABLE:
        fused_silu = FusedSiLU()
        implementations['Custom CUDA Kernel'] = lambda x: fused_silu(x)
    
    df = profiler.compare_implementations(
        implementations,
        input_tensor,
        num_iterations=1000
    )
    
    # Detailed profiling with PyTorch Profiler
    if CUDA_KERNELS_AVAILABLE:
        profiler.profile_with_pytorch_profiler(
            lambda x: fused_silu(x),
            input_tensor,
            "FusedSiLU"
        )
    
    profiler.save_results("silu_benchmark.csv")


def benchmark_memory_bandwidth():
    """
    Measure memory bandwidth utilization.
    This is critical for understanding GPU efficiency.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    print("\n" + "="*80)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("="*80)
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    # Theoretical memory bandwidth (GB/s)
    # RTX 4070 Mobile: ~256 GB/s
    theoretical_bandwidth = 256.0
    
    print(f"\nGPU: {props.name}")
    print(f"Theoretical Memory Bandwidth: {theoretical_bandwidth:.1f} GB/s")
    
    # Test memory bandwidth with simple copy
    sizes_mb = [1, 10, 100, 500, 1000]
    
    print("\n" + "-"*60)
    print(f"{'Size (MB)':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20} {'Efficiency (%)'}")
    print("-"*60)
    
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        src = torch.randn(num_elements, device='cuda', dtype=torch.float32)
        dst = torch.empty_like(src)
        
        # Warmup
        for _ in range(10):
            dst.copy_(src)
        
        torch.cuda.synchronize()
        
        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            dst.copy_(src)
        end.record()
        
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / 100
        bytes_transferred = num_elements * 4 * 2  # Read + Write
        bandwidth_gb_s = (bytes_transferred / 1e9) / (time_ms / 1000)
        efficiency = (bandwidth_gb_s / theoretical_bandwidth) * 100
        
        print(f"{size_mb:<15} {time_ms:<15.4f} {bandwidth_gb_s:<20.2f} {efficiency:.1f}%")
    
    print("-"*60 + "\n")


def main():
    """Run all benchmarks."""
    print("\n" + "="*80)
    print("EFFICIENT LLM INFERENCE: COMPREHENSIVE PROFILING SUITE")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\nError: CUDA not available. This script requires a GPU.")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}\n")
    
    # Run benchmarks
    benchmark_rmsnorm()
    benchmark_silu()
    benchmark_memory_bandwidth()
    
    print("\n" + "="*80)
    print("ALL BENCHMARKS COMPLETED")
    print("Results saved in: ./profiling_results/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
