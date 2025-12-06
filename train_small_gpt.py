#!/usr/bin/env python3
"""train_small_gpt.py
Minimal training script for a small causal LM using Hugging Face Transformers.
Trains on a JSONL file where each line is: {"input": "...", "target": "..."}
The training concatenates input + tokenizer.eos_token + target and trains with teacher forcing.
Example usage:
    python train_small_gpt.py --train-file data/train.jsonl --output-dir model/ --batch-size 8 --epochs 3 --device cuda
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)


class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    # Handle nested structure: {"row": {"input": "...", "target": "..."}}
                    if 'row' in j and isinstance(j['row'], dict):
                        inp = j['row'].get('input', '')
                        tgt = j['row'].get('target', '')
                    else:
                        # Fallback to flat structure
                        inp = j.get('input', '')
                        tgt = j.get('target', '')
                except Exception:
                    # allow raw text line -> treat all as input
                    j = {'input': line, 'target': ''}
                    inp = line
                    tgt = ''
                # concatenate with eos token between input and target to allow model to know boundary
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


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    # load or create tokenizer
    if args.tokenizer and args.tokenizer != 'none':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # model init: either load a pretrained small model or instantiate from config
    if args.init_model and args.init_model != 'none':
        model = AutoModelForCausalLM.from_pretrained(args.init_model)
        model.resize_token_embeddings(len(tokenizer))
    else:
        # build a small gpt2-like config
        config = AutoConfig.from_pretrained('gpt2')
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_embd = args.n_embd
        model = AutoModelForCausalLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    # dataset and loader
    train_ds = JsonlDataset(args.train_file, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    model.train()
    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if global_step % args.log_every == 0 and global_step>0:
                avg = running_loss / (args.log_every)
                print(f"epoch={epoch} step={global_step} avg_loss={avg:.4f}")
                running_loss = 0.0
            global_step += 1

        # save checkpoint each epoch
        ckpt_dir = Path(args.output_dir) / f'epoch_{epoch}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # final save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print('Training complete. Model saved to', args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--init-model', default='gpt2', help='pretrained model name or "none" to init from config')
    parser.add_argument('--tokenizer', default='gpt2', help='tokenizer name or "none" to use gpt2 tokenizer')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--n-layer', type=int, default=6)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=512)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    args = parser.parse_args()
    train(args)
