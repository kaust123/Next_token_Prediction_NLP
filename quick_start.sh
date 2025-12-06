#!/bin/bash
# quick_start.sh
# Quick setup and verification script

set -e

echo "=========================================="
echo "NLP Assignment - Quick Start Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Verify dataset
echo ""
echo "Verifying dataset files..."
if [ -f "dataset/train_mr.jsonl" ]; then
    echo "✓ Marathi training data found"
else
    echo "✗ Marathi training data missing!"
fi

if [ -f "dataset/train_te.jsonl" ]; then
    echo "✓ Telugu training data found"
else
    echo "✗ Telugu training data missing!"
fi

# Run data preprocessing verification
echo ""
echo "Running data preprocessing check..."
python preprocess_data.py --process-all

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the data statistics above"
echo "2. Choose a language to train:"
echo "   - For Marathi: bash train_marathi.sh"
echo "   - For Telugu:  bash train_telugu.sh"
echo ""
echo "3. Or train step-by-step:"
echo "   a. Train tokenizer: python train_tokenizer.py --input dataset/train_mr.jsonl --output tokenizers/marathi --vocab-size 8000"
echo "   b. Train model:     python train_small_gpt.py --train-file dataset/train_mr.jsonl --output-dir model/marathi --tokenizer tokenizers/marathi"
echo "   c. Evaluate:        python evaluate.py --model-dir model/marathi --data-file dataset/validation_mr.jsonl"
echo ""
echo "For more details, see README.md"
echo ""
