#!/bin/bash
# train_marathi.sh
# Training script for Marathi language model

set -e  # Exit on error

echo "=================================="
echo "Marathi Language Model Training"
echo "=================================="

# Configuration
LANGUAGE="marathi"
TRAIN_FILE="dataset/train_mr.jsonl"
VAL_FILE="dataset/validation_mr.jsonl"
OUTPUT_DIR="model/${LANGUAGE}_small"
TOKENIZER_DIR="tokenizers/${LANGUAGE}"
RESULTS_DIR="results"

# Training hyperparameters
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=5e-4
MAX_LENGTH=256
N_LAYER=6
N_HEAD=8
N_EMBD=512

# Device - Auto-detect (cuda/mps/cpu)
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS - check for MPS (Apple Silicon) or fallback to CPU
    if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        DEVICE="mps"
        echo "✓ Detected Apple Silicon - using MPS (GPU acceleration)"
    else
        DEVICE="cpu"
        echo "Detected macOS - using CPU"
    fi
else
    DEVICE="cuda"  # Linux/Windows - try CUDA
    echo "Using CUDA (if available)"
fi

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TOKENIZER_DIR}"
mkdir -p "${RESULTS_DIR}"

# Step 1: Train custom tokenizer
echo ""
echo "Step 1: Training custom tokenizer..."
python train_tokenizer.py \
    --input "${TRAIN_FILE}" \
    --output "${TOKENIZER_DIR}" \
    --vocab-size 8000 \
    --test

# Step 2: Train model
echo ""
echo "Step 2: Training model..."
python train_small_gpt.py \
    --train-file "${TRAIN_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --init-model none \
    --n-layer ${N_LAYER} \
    --n-head ${N_HEAD} \
    --n-embd ${N_EMBD} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --max-length ${MAX_LENGTH} \
    --device ${DEVICE}

# Step 3: Evaluate model
echo ""
echo "Step 3: Evaluating model..."
python evaluate.py \
    --model-dir "${OUTPUT_DIR}" \
    --data-file "${VAL_FILE}" \
    --output-file "${RESULTS_DIR}/${LANGUAGE}_eval.json" \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE}

# Step 4: Measure performance
echo ""
echo "Step 4: Measuring performance metrics..."
python measure_performance.py \
    --model-dir "${OUTPUT_DIR}" \
    --device ${DEVICE} \
    --seq-len 64 \
    --iters 200 \
    --batch-size 1

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Evaluation results: ${RESULTS_DIR}/${LANGUAGE}_eval.json"
echo ""
echo "Next steps:"
echo "1. Review evaluation results"
echo "2. Update submission_metadata.json with metrics"
echo "3. Write report.pdf"
