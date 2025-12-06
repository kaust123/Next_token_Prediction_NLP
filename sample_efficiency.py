#!/usr/bin/env python3
"""sample_efficiency.py
Prepares dataset subsets at various fractions and provides a manifest CSV for running experiments.
Usage:
    python sample_efficiency.py --train-file data/train.jsonl --output-csv subsets_manifest.csv
"""
import argparse
import pandas as pd


def run_subsample_experiment(full_train_path, fractions, output_csv):
    with open(full_train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    for frac in fractions:
        k = max(1, int(len(lines) * frac))
        subfile = f"train_frac_{int(frac*100)}.jsonl"
        with open(subfile, 'w', encoding='utf-8') as out:
            out.writelines(lines[:k])
        results.append({'fraction': frac, 'num_examples': k, 'subfile': subfile})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Wrote manifest to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--output-csv', default='subsets_manifest.csv')
    args = parser.parse_args()
    run_subsample_experiment(args.train_file, [0.01, 0.05, 0.1, 0.25, 0.5, 1.0], args.output_csv)
