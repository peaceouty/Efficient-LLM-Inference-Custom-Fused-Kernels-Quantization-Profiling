"""vLLM inference with AWQ 4-bit quantization for consumer GPUs."""

import argparse
import time
from typing import List
import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Install with: pip install vllm")


def load_quantized_model(
    model_name: str = "casperhansen/llama-3-8b-instruct-awq",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90
):
    """Load AWQ quantized model with vLLM engine."""
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is required. Install with: pip install vllm")
    
    print(f"\n=== Loading Model: {model_name} ===")
    print(f"Max Model Length: {max_model_len}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization * 100}%")
    print("Quantization: AWQ 4-bit")
    print("KV-Cache: PagedAttention enabled\n")
    
    llm = LLM(
        model=model_name,
        quantization="awq",
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    return llm


def run_inference_benchmark(
    llm: LLM,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Run inference and measure throughput."""
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    print(f"Running inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate metrics
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / elapsed_time
    
    metrics = {
        'total_prompts': len(prompts),
        'elapsed_time_s': elapsed_time,
        'total_tokens_generated': total_tokens,
        'throughput_tokens_per_s': throughput,
        'avg_latency_per_prompt_s': elapsed_time / len(prompts)
    }
    
    return outputs, metrics


def print_results(outputs, metrics):
    """Print inference results and metrics."""
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    for i, output in enumerate(outputs[:3]):  # Show first 3 results
        prompt = output.prompt[:100] + "..." if len(output.prompt) > 100 else output.prompt
        generated_text = output.outputs[0].text
        
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text[:200]}...")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")
    
    if len(outputs) > 3:
        print(f"\n... and {len(outputs) - 3} more outputs")
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Total Prompts: {metrics['total_prompts']}")
    print(f"Total Time: {metrics['elapsed_time_s']:.2f}s")
    print(f"Total Tokens Generated: {metrics['total_tokens_generated']}")
    print(f"Throughput: {metrics['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"Average Latency: {metrics['avg_latency_per_prompt_s']:.3f}s per prompt")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="vLLM Quantized Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="casperhansen/llama-3-8b-instruct-awq",
        help="HuggingFace model ID (AWQ quantized)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.90,
        help="GPU memory utilization (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This script requires a GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Sample prompts
    prompts = [
        "Explain the concept of attention mechanism in transformers in simple terms:",
        "Write a Python function to implement binary search:",
        "What are the key differences between AWS and Azure cloud platforms?",
        "How does quantization improve LLM inference efficiency?",
        "Describe the PagedAttention algorithm used in vLLM:",
    ]
    
    # Load model
    llm = load_quantized_model(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory
    )
    
    # Run inference
    outputs, metrics = run_inference_benchmark(
        llm,
        prompts,
        max_tokens=args.max_tokens
    )
    
    # Print results
    print_results(outputs, metrics)
    
    # Memory stats
    print("GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
