#!/usr/bin/env python3
"""measure_performance.py
Measure parameter counts, latency (median & p95), throughput (tokens/sec), and approximate peak memory.
Usage:
    python measure_performance.py --model-dir model/ --device cuda --seq-len 64 --iters 200 --batch-size 1
Notes: For accurate GPU memory measurement run on a dedicated GPU and ensure other processes are minimized.
"""
import argparse
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_latency(model, tokenizer, device, seq_len=64, iters=200, batch_size=1):
    model.eval()
    # Build a dummy input using tokenizer's pad token id
    sample = tokenizer('Hello world', return_tensors='pt')
    input_ids = sample['input_ids']
    # expand/pad to desired seq_len
    if input_ids.shape[1] < seq_len:
        pad_len = seq_len - input_ids.shape[1]
        pad = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, pad], dim=1)
    input_ids = input_ids.to(device).repeat(batch_size, 1)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids=input_ids)
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.time()
            _ = model(input_ids=input_ids)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t1 = time.time()
            # record time per token (total time / seq_len)
            times.append((t1 - t0) / input_ids.shape[1])

    arr = np.array(times)
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    tokens_per_second = 1.0 / median if median>0 else float('inf')
    return median, p95, tokens_per_second


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    # Auto-detect available device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model = model.to(args.device)

    total, trainable = count_params(model)
    print(f"total_params={total} trainable_params={trainable}")

    median, p95, tps = measure_latency(model, tokenizer, args.device, seq_len=args.seq_len, iters=args.iters, batch_size=args.batch_size)
    print(f"latency_per_token_median={median:.6f} p95={p95:.6f} tokens_per_second={tps:.2f}")

    # peak memory (approx): report max memory allocated during model instantiation (CUDA only)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        # reset peak stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = model.forward(torch.zeros((1, min(2, args.seq_len)), dtype=torch.long, device=args.device))
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"peak_gpu_memory_MB={peak:.2f}")
