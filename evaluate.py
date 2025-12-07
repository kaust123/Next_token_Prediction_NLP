#!/usr/bin/env python3
"""evaluate.py
Evaluate a causal language model on validation/test data.
Computes perplexity, cross-entropy, bits-per-token, and token-level accuracy.

Usage:
    python evaluate.py --model-dir model/ --data-file dataset/validation_mr.jsonl --device cuda --batch-size 8
"""
import argparse
import json
import math
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class EvalDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    # Handle nested structure
                    if 'row' in j and isinstance(j['row'], dict):
                        inp = j['row'].get('input', '')
                        tgt = j['row'].get('target', '')
                    else:
                        inp = j.get('input', '')
                        tgt = j.get('target', '')
                except Exception:
                    inp = line
                    tgt = ''

                # Concatenate input + eos + target for evaluation
                txt = inp + tokenizer.eos_token + tgt if tgt else inp
                self.examples.append(txt)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        txt = self.examples[idx]
        enc = self.tokenizer(txt, truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'labels': labels}


def collate_fn(batch):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'labels': labels}


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and compute metrics.
    Returns: perplexity, cross_entropy, bits_per_token, token_accuracy
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate loss
            # Count valid tokens (not -100)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            # Calculate token-level accuracy
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Count correct predictions (excluding padding)
            mask = (shift_labels != -100)
            correct = ((predictions == shift_labels) & mask).sum().item()
            correct_predictions += correct

    # Calculate metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    cross_entropy = avg_loss
    bits_per_token = avg_loss / math.log(2)
    token_accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0

    return {
        'perplexity': perplexity,
        'cross_entropy': cross_entropy,
        'bits_per_token': bits_per_token,
        'token_accuracy': token_accuracy,
        'total_tokens': total_tokens
    }


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}\n")

    # Load model and tokenizer
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to(device)
    print(" Model loaded\n")

    # Load dataset
    print(f"Loading dataset from {args.data_file}...")
    dataset = EvalDataset(args.data_file, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=args.num_workers)
    print(f" Loaded {len(dataset)} examples\n")

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, dataloader, device)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Perplexity:        {metrics['perplexity']:.4f}")
    print(f"Cross-entropy:     {metrics['cross_entropy']:.4f}")
    print(f"Bits-per-token:    {metrics['bits_per_token']:.4f}")
    print(f"Token accuracy:    {metrics['token_accuracy']:.4f} ({metrics['token_accuracy']*100:.2f}%)")
    print(f"Total tokens:      {metrics['total_tokens']}")
    print("="*60)

    # Save results to JSON
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'model_dir': args.model_dir,
            'data_file': args.data_file,
            'metrics': metrics,
            'config': {
                'batch_size': args.batch_size,
                'max_length': args.max_length,
                'device': str(device)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate causal language model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to validation/test JSONL file')
    parser.add_argument('--output-file', type=str, default='results/evaluation_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')

    args = parser.parse_args()
    main(args)
