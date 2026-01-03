"""End-to-end inference profiling with latency breakdown."""

import torch
import time
from typing import Dict, List
from dataclasses import dataclass
import json


@dataclass
class InferenceMetrics:
    """Container for inference metrics."""
    total_latency_ms: float
    token_generation_ms: float
    kv_cache_overhead_ms: float
    attention_compute_ms: float
    throughput_tokens_per_sec: float
    memory_peak_mb: float


class InferencePipeline:
    """
    Simulated inference pipeline for profiling.
    Models the key stages of LLM inference.
    """
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.batch_size = model_config.get('batch_size', 1)
        self.seq_length = model_config.get('seq_length', 512)
        self.hidden_size = model_config.get('hidden_size', 4096)
        self.num_layers = model_config.get('num_layers', 32)
        
        print(f"Pipeline Config: {json.dumps(model_config, indent=2)}")
    
    def profile_inference(self, num_tokens: int = 100) -> InferenceMetrics:
        """
        Profile simulated inference for token generation.
        
        Args:
            num_tokens: Number of tokens to generate
        
        Returns:
            InferenceMetrics object
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simulate KV cache
        kv_cache = []
        for _ in range(self.num_layers):
            k = torch.randn(
                self.batch_size, 32, self.seq_length, 128,  # num_heads=32, head_dim=128
                device=device, dtype=torch.float16
            )
            v = torch.randn_like(k)
            kv_cache.append((k, v))
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        total_token_gen_time = 0
        total_kv_time = 0
        total_attn_time = 0
        
        # Token generation loop
        for token_idx in range(num_tokens):
            token_start = time.time()
            
            # 1. KV cache update
            kv_start = time.time()
            for layer_idx in range(self.num_layers):
                # Simulate cache update
                k, v = kv_cache[layer_idx]
                new_k = torch.randn(
                    self.batch_size, 32, 1, 128,
                    device=device, dtype=torch.float16
                )
                k = torch.cat([k, new_k], dim=2)
            kv_time = time.time() - kv_start
            total_kv_time += kv_time
            
            # 2. Attention computation
            attn_start = time.time()
            q = torch.randn(
                self.batch_size, 32, 1, 128,
                device=device, dtype=torch.float16
            )
            # Simulate attention: Q @ K^T
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = torch.softmax(scores / (128 ** 0.5), dim=-1)
            output = torch.matmul(scores, v)
            attn_time = time.time() - attn_start
            total_attn_time += attn_time
            
            token_time = time.time() - token_start
            total_token_gen_time += token_time
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Memory stats
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        metrics = InferenceMetrics(
            total_latency_ms=total_time * 1000,
            token_generation_ms=(total_token_gen_time / num_tokens) * 1000,
            kv_cache_overhead_ms=(total_kv_time / num_tokens) * 1000,
            attention_compute_ms=(total_attn_time / num_tokens) * 1000,
            throughput_tokens_per_sec=num_tokens / total_time,
            memory_peak_mb=peak_memory
        )
        
        return metrics
    
    def print_metrics(self, metrics: InferenceMetrics):
        """Print formatted metrics."""
        print("\n" + "="*80)
        print("INFERENCE PIPELINE PROFILING RESULTS")
        print("="*80)
        print(f"Total Latency: {metrics.total_latency_ms:.2f} ms")
        print(f"Per-Token Generation: {metrics.token_generation_ms:.4f} ms")
        print(f"  - KV Cache Update: {metrics.kv_cache_overhead_ms:.4f} ms")
        print(f"  - Attention Compute: {metrics.attention_compute_ms:.4f} ms")
        print(f"Throughput: {metrics.throughput_tokens_per_sec:.2f} tokens/s")
        print(f"Peak Memory: {metrics.memory_peak_mb:.2f} MB")
        print("="*80 + "\n")


def compare_configurations():
    """Compare different model configurations."""
    configs = [
        {
            'name': 'Llama-3-8B (Full Precision)',
            'batch_size': 1,
            'seq_length': 512,
            'hidden_size': 4096,
            'num_layers': 32
        },
        {
            'name': 'Llama-3-8B (Quantized)',
            'batch_size': 4,  # Can batch more with quantization
            'seq_length': 512,
            'hidden_size': 4096,
            'num_layers': 32
        }
    ]
    
    print("\n" + "="*80)
    print("COMPARING MODEL CONFIGURATIONS")
    print("="*80)
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        pipeline = InferencePipeline(config)
        metrics = pipeline.profile_inference(num_tokens=50)
        pipeline.print_metrics(metrics)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
    else:
        compare_configurations()
