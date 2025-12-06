#!/usr/bin/env python3
"""infer.py
Loads a causal LM (Hugging Face) and writes next-token probability outputs.
Produces a JSONL file with top-k token probabilities for each input line.
Usage example:
    python infer.py --model-dir model/ --input-file data/test_inputs.jsonl --output-file preds.jsonl --batch-size 8 --device cuda --seed 42 --topk 50
"""
import argparse
import json
import random
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config)
    model.to(device)
    model.eval()
    return tokenizer, model


def next_token_logits(model, input_ids, attention_mask=None):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        return logits


def softmax(logits):
    return torch.nn.functional.softmax(logits, dim=-1)


def run_inference(model_dir, input_path, output_path, batch_size, device, seed, topk):
    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer, model = load_model(model_dir, device)

    # read inputs: expect jsonl with {"input": "..."} per line. Also accept plain text lines.
    inputs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                text = j.get('input', j.get('text', None))
                if text is None:
                    # fall back to raw string
                    text = line
            except Exception:
                text = line
            inputs.append(text)

    out_f = open(output_path, 'w', encoding='utf-8')

    for i in range(0, len(inputs), batch_size):
        batch_texts = inputs[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc.get('attention_mask', None)

        logits = next_token_logits(model, input_ids, attention_mask)
        probs = softmax(logits).cpu().numpy()

        for text, p in zip(batch_texts, probs):
            # Save topk probabilities to keep output size moderate
            import numpy as np
            k = min(topk, p.shape[0])
            idx = np.argpartition(-p, k-1)[:k]
            top = sorted([(int(int(i)), float(p[int(i)])) for i in idx], key=lambda x: -x[1])
            out_f.write(json.dumps({
                'input': text,
                'topk': top
            }, ensure_ascii=False) + '\n')

    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=50)
    args = parser.parse_args()

    run_inference(args.model_dir, args.input_file, args.output_file, args.batch_size, args.device, args.seed, args.topk)
