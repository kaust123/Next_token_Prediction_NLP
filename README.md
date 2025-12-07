# Compact Language Model Challenge - Telugu & Marathi

This repository contains a compact, sample-efficient language model for next-token prediction on Telugu and Marathi low-resource languages.

Project Structure

```
nlp_assignment/
├── dataset/                    # Raw dataset files
│   ├── train_mr.jsonl         # Marathi training data (6,900 examples)
│   ├── train_te.jsonl         # Telugu training data (5,500 examples)
│   ├── validation_mr.jsonl    # Marathi validation data (2,600 examples)
│   └── validation_te.jsonl    # Telugu validation data (2,537 examples)
├── model/                      # Trained model checkpoints (generated)
├── tokenizers/                 # Custom tokenizers (generated)
├── results/                    # Evaluation results (generated)
├── train_small_gpt.py         # Main training script
├── evaluate.py                # Evaluation script (perplexity, accuracy)
├── infer.py                   # Inference script for predictions
├── measure_performance.py     # Performance metrics (latency, FLOPs, memory)
├── sample_efficiency.py       # Sample efficiency curve generation
├── train_tokenizer.py         # Custom tokenizer training
├── preprocess_data.py         # Data preprocessing utilities
├── requirements.txt           # Python dependencies
├── submission_metadata.json   # Submission metadata
└── README.md                  # This file
```

Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Hardware Requirements:**
- Recommended: GPU with at least 8GB VRAM (CUDA compatible)
- Minimum: CPU with 16GB RAM (slower training)

2. Data Preprocessing (Optional)

The training script handles nested JSON automatically, but you can verify the data:

```bash
# View data statistics for all datasets
python preprocess_data.py --process-all

# Extract flat JSONL (optional)
python preprocess_data.py --input dataset/train_mr.jsonl --output processed/train_mr_flat.jsonl
```

### 3. Train Custom Tokenizer (Recommended)

For better performance on Telugu/Marathi, train a custom tokenizer:

```bash
# Train Marathi tokenizer
python train_tokenizer.py \
    --input dataset/train_mr.jsonl \
    --output tokenizers/marathi \
    --vocab-size 8000 \
    --test

# Train Telugu tokenizer
python train_tokenizer.py \
    --input dataset/train_te.jsonl \
    --output tokenizers/telugu \
    --vocab-size 8000 \
    --test

# Train combined tokenizer (both languages)
python train_tokenizer.py \
    --input dataset/train_mr.jsonl dataset/train_te.jsonl \
    --output tokenizers/combined \
    --vocab-size 16000 \
    --test
```

### 4. Train Model

**Option A: Small GPT from scratch (Marathi)**
```bash
python train_small_gpt.py \
    --train-file dataset/train_mr.jsonl \
    --output-dir model/marathi_small \
    --tokenizer tokenizers/marathi \
    --init-model none \
    --n-layer 6 \
    --n-head 8 \
    --n-embd 512 \
    --batch-size 16 \
    --epochs 10 \
    --lr 5e-4 \
    --max-length 256 \
    --device cuda
```

Option B: Fine-tune pretrained multilingual model**
```bash
python train_small_gpt.py \
    --train-file dataset/train_te.jsonl \
    --output-dir model/telugu_finetuned \
    --init-model ai4bharat/IndicBARTSS \
    --tokenizer ai4bharat/IndicBARTSS \
    --batch-size 8 \
    --epochs 5 \
    --lr 2e-5 \
    --device cuda
```

**Recommended Configurations:**

| Config | Layers | Heads | Embed | Params | Use Case |
|--------|--------|-------|-------|--------|----------|
| Tiny   | 4      | 4     | 256   | ~8M    | Fast baseline |
| Small  | 6      | 8     | 512   | ~30M   | Balanced |
| Medium | 8      | 12    | 768   | ~80M   | Best performance |

### 5. Evaluate Model

```bash
# Evaluate on validation set
python evaluate.py \
    --model-dir model/marathi_small \
    --data-file dataset/validation_mr.jsonl \
    --output-file results/marathi_eval.json \
    --batch-size 16 \
    --device cuda
```

**Output Metrics:**
- Perplexity (primary metric)
- Cross-entropy
- Bits-per-token
- Token-level accuracy

6. Measure Performance

```bash
# Measure efficiency metrics
python measure_performance.py \
    --model-dir model/marathi_small \
    --device cuda \
    --seq-len 64 \
    --iters 200 \
    --batch-size 1
```

**Output:**
- Total/trainable parameters
- Inference latency (median, p95)
- Throughput (tokens/second)
- Peak GPU memory usage

7. Run Inference

```bash
# Generate predictions for test data
python infer.py \
    --model-dir model/marathi_small \
    --input-file test_inputs.jsonl \
    --output-file predictions.jsonl \
    --batch-size 8 \
    --device cuda \
    --seed 42 \
    --topk 50
```

### 8. Sample Efficiency Analysis

```bash
# Create subsampled datasets
python sample_efficiency.py \
    --train-file dataset/train_mr.jsonl \
    --output-csv subsets_manifest.csv

# Train models on different data fractions and plot learning curves
```

Expected Performance

### Baseline Results (6-layer, 512-dim model)

| Language | Perplexity | Accuracy | Params | Latency (ms/token) |
|----------|------------|----------|--------|--------------------|
| Marathi  | ~45-60     | ~35-45%  | 30M    | ~2-5               |
| Telugu   | ~50-65     | ~30-40%  | 30M    | ~2-5               |

*Note: Actual results depend on training configuration and hardware*

## Advanced Usage

### Custom Model Architecture

Edit the training script parameters:
```bash
python train_small_gpt.py \
    --train-file dataset/train_mr.jsonl \
    --output-dir model/custom \
    --n-layer 8 \          # Number of transformer layers
    --n-head 12 \          # Number of attention heads
    --n-embd 768 \         # Embedding dimension
    --weight-decay 0.01 \  # L2 regularization
    --warmup-steps 500     # Learning rate warmup
```

Mixed Language Training

Train on both languages simultaneously:
```bash
# Concatenate datasets
cat dataset/train_mr.jsonl dataset/train_te.jsonl > dataset/train_combined.jsonl

# Train with combined tokenizer
python train_small_gpt.py \
    --train-file dataset/train_combined.jsonl \
    --output-dir model/combined \
    --tokenizer tokenizers/combined \
    --epochs 10
```

Submission Checklist

- [ ] Trained model checkpoint in `model/` directory
- [ ] Tokenizer files saved with model
- [ ] `infer.py` runs successfully on test inputs
- [ ] `measure_performance.py` outputs all required metrics
- [ ] `README.md` with reproduction instructions (this file)
- [ ] `requirements.txt` with exact versions
- [ ] `submission_metadata.json` with team info
- [ ] `report.pdf` (4-6 pages) with:
  - [ ] Model architecture description
  - [ ] Training procedure and hyperparameters
  - [ ] Preprocessing steps
  - [ ] Results tables and plots
  - [ ] External resources used
  - [ ] Ethical considerations and limitations

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
--batch-size 4

# Reduce sequence length
--max-length 128

# Use gradient accumulation (modify script)
```

Slow Training on CPU
```bash
# Use smaller model
--n-layer 4 --n-head 4 --n-embd 256

# Reduce number of workers
# Add to dataloader: num_workers=0
```

### CUDA Not Available
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or use CPU
--device cpu
```

References

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **IndicNLP**: https://indicnlp.ai4bharat.org/
- **Telugu/Marathi Pretrained Models**:
  - ai4bharat/IndicBARTSS
  - google/muril-base-cased

License

This project is for academic use only as part of CS6320E assignment.

Team Information

Update `submission_metadata.json` with your team details:
- Team members
- External resources used
- Hardware specifications
- Random seeds

---


