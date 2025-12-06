#!/usr/bin/env python3
"""train_tokenizer.py
Train a custom SentencePiece tokenizer for Telugu and Marathi languages.
This creates a compact tokenizer optimized for the provided training data.

Usage:
    # Train on single language
    python train_tokenizer.py --input dataset/train_mr.jsonl --output tokenizers/marathi --vocab-size 8000

    # Train on both languages (combined)
    python train_tokenizer.py --input dataset/train_mr.jsonl dataset/train_te.jsonl --output tokenizers/combined --vocab-size 16000
"""
import argparse
import json
import tempfile
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast


def extract_text_from_jsonl(input_files, output_file):
    """Extract all text from JSONL files for tokenizer training."""
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            print(f"Extracting text from {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
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

                        # Write both input and target
                        if inp:
                            out_f.write(inp + '\n')
                        if tgt:
                            out_f.write(tgt + '\n')
                    except Exception as e:
                        print(f"Warning: Failed to parse line: {e}")
                        continue

    print(f"✓ Text extracted to {output_file}\n")


def train_bpe_tokenizer(text_file, vocab_size, output_dir, min_frequency=2):
    """Train a BPE tokenizer using the tokenizers library."""
    print(f"Training BPE tokenizer with vocab size {vocab_size}...")

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Define special tokens
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )

    # Train the tokenizer
    tokenizer.train(files=[text_file], trainer=trainer)

    # Set post-processor for proper handling of special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save tokenizer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"✓ Tokenizer saved to {tokenizer_path}")

    # Convert to HuggingFace format
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>"
    )

    fast_tokenizer.save_pretrained(str(output_dir))
    print(f"✓ HuggingFace tokenizer saved to {output_dir}")

    return fast_tokenizer


def test_tokenizer(tokenizer, samples):
    """Test the tokenizer on sample texts."""
    print("\n" + "="*60)
    print("TOKENIZER TEST")
    print("="*60)

    for i, text in enumerate(samples[:3], 1):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"\nSample {i}:")
        print(f"Original:  {text[:100]}...")
        print(f"Tokens:    {len(encoded)} tokens")
        print(f"Token IDs: {encoded[:20]}...")
        print(f"Decoded:   {decoded[:100]}...")

    vocab_size = tokenizer.vocab_size
    print(f"\nVocabulary size: {vocab_size}")
    print("="*60)


def main(args):
    # Create temporary file for extracted text
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp_file = tmp.name

    try:
        # Extract text from JSONL files
        extract_text_from_jsonl(args.input, tmp_file)

        # Train tokenizer
        tokenizer = train_bpe_tokenizer(
            tmp_file,
            vocab_size=args.vocab_size,
            output_dir=args.output,
            min_frequency=args.min_frequency
        )

        # Test tokenizer on sample data
        if args.test:
            samples = []
            with open(args.input[0], 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    try:
                        j = json.loads(line.strip())
                        if 'row' in j:
                            samples.append(j['row'].get('input', ''))
                        else:
                            samples.append(j.get('input', ''))
                    except:
                        continue

            if samples:
                test_tokenizer(tokenizer, samples)

    finally:
        # Clean up temporary file
        Path(tmp_file).unlink(missing_ok=True)

    print(f"\n✓ Tokenizer training complete!")
    print(f"  Output directory: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a custom tokenizer for Telugu/Marathi')
    parser.add_argument('--input', nargs='+', required=True,
                        help='Input JSONL file(s) for training')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for tokenizer')
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='Vocabulary size (default: 8000)')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum frequency for tokens (default: 2)')
    parser.add_argument('--test', action='store_true',
                        help='Test tokenizer on sample data after training')

    args = parser.parse_args()
    main(args)
