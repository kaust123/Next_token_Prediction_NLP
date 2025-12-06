#!/usr/bin/env python3
"""preprocess_data.py
Extracts input and target fields from nested JSONL structure and creates flat JSONL files.
Also provides data statistics and validation.

Usage:
    python preprocess_data.py --input dataset/train_mr.jsonl --output processed/train_mr_flat.jsonl
    python preprocess_data.py --process-all  # Process all dataset files
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def extract_flat_data(input_path, output_path, show_stats=True):
    """Extract input and target from nested JSON structure."""
    examples = []
    errors = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
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

                if inp and tgt:  # Only include if both exist
                    examples.append({'input': inp, 'target': tgt})
                else:
                    errors += 1

            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                errors += 1

    # Write flat JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    if show_stats:
        print(f"\n{'='*60}")
        print(f"File: {input_path.name}")
        print(f"{'='*60}")
        print(f"Total examples: {len(examples)}")
        print(f"Errors/skipped: {errors}")

        # Calculate statistics
        input_lengths = [len(ex['input']) for ex in examples]
        target_lengths = [len(ex['target']) for ex in examples]

        print(f"\nInput character lengths:")
        print(f"  Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.1f}")
        print(f"Target character lengths:")
        print(f"  Min: {min(target_lengths)}, Max: {max(target_lengths)}, Avg: {sum(target_lengths)/len(target_lengths):.1f}")

        # Sample examples
        print(f"\nSample examples (first 2):")
        for i, ex in enumerate(examples[:2], 1):
            print(f"\n  Example {i}:")
            print(f"    Input:  {ex['input'][:100]}...")
            print(f"    Target: {ex['target'][:100]}...")

    return len(examples), errors


def process_all_datasets(dataset_dir, output_dir):
    """Process all dataset files in the directory."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    files = list(dataset_dir.glob('*.jsonl'))

    if not files:
        print(f"No JSONL files found in {dataset_dir}")
        return

    print(f"Found {len(files)} dataset files to process\n")

    total_examples = 0
    total_errors = 0

    for input_file in sorted(files):
        output_file = output_dir / f"{input_file.stem}_flat.jsonl"
        num_examples, num_errors = extract_flat_data(input_file, output_file, show_stats=True)
        total_examples += num_examples
        total_errors += num_errors
        print(f"✓ Saved to: {output_file}\n")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples processed: {total_examples}")
    print(f"Total errors/skipped: {total_errors}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess nested JSONL data')
    parser.add_argument('--input', type=str, help='Input JSONL file')
    parser.add_argument('--output', type=str, help='Output flat JSONL file')
    parser.add_argument('--process-all', action='store_true',
                        help='Process all files in dataset/ directory')
    parser.add_argument('--dataset-dir', type=str, default='dataset',
                        help='Dataset directory (default: dataset)')
    parser.add_argument('--output-dir', type=str, default='processed',
                        help='Output directory (default: processed)')

    args = parser.parse_args()

    if args.process_all:
        process_all_datasets(args.dataset_dir, args.output_dir)
    elif args.input and args.output:
        num_ex, num_err = extract_flat_data(Path(args.input), Path(args.output))
        print(f"\nProcessed {num_ex} examples ({num_err} errors)")
        print(f"Output: {args.output}")
    else:
        parser.print_help()
