# Project Setup Summary - NLP Compact Language Model Challenge

## All Issues Fixed and Files Created

### 🔧 Issues Fixed

1. **✓ Nested JSON Data Loading**
   - Updated [train_small_gpt.py](train_small_gpt.py) to handle nested structure `{"row": {"input": "...", "target": "..."}}`
   - Added fallback for flat JSON structure
   - Works with both training and evaluation scripts

### New Files Created

#### Core Scripts
1. **[preprocess_data.py](preprocess_data.py)** - Data preprocessing and statistics
   - Extracts input/target from nested JSON
   - Shows data statistics (lengths, sample counts)
   - Validates dataset integrity

2. **[evaluate.py](evaluate.py)** - Comprehensive evaluation script
   - Computes perplexity (primary metric)
   - Calculates cross-entropy, bits-per-token, token accuracy
   - Saves results to JSON

3. **[train_tokenizer.py](train_tokenizer.py)** - Custom tokenizer training
   - Trains BPE tokenizer on Telugu/Marathi data
   - Configurable vocabulary size
   - Saves in HuggingFace format

#### Configuration Files
4. **[requirements.txt](requirements.txt)** - Python dependencies
   - PyTorch, Transformers, Tokenizers
   - All necessary libraries listed

5. **[submission_metadata.json](submission_metadata.json)** - Submission template
   - Team information
   - Model architecture details
   - Performance metrics placeholders
   - External resources declaration

6. **[.gitignore](.gitignore)** - Git ignore patterns
   - Excludes model binaries, venv, temporary files
   - Keeps dataset and important configs

#### Documentation
7. **[README.md](README.md)** - Comprehensive guide
   - Complete setup instructions
   - Usage examples for all scripts
   - Troubleshooting section
   - Performance expectations

#### Automation Scripts
8. **[train_marathi.sh](train_marathi.sh)** - Automated Marathi training
   - End-to-end training pipeline
   - Tokenizer → Model → Evaluation → Metrics

9. **[train_telugu.sh](train_telugu.sh)** - Automated Telugu training
   - Same pipeline for Telugu language

10. **[quick_start.sh](quick_start.sh)** - Setup verification
    - Creates virtual environment
    - Installs dependencies
    - Verifies dataset
    - Shows next steps

---

## Quick Start Guide

### 1. Initial Setup
```bash
cd "/Users/soumik.pramanik/National Performance Automation Dashboard/assignment/vision_gpt/nlp_assignment"

# Run quick setup
bash quick_start.sh
```

### 2. Train a Model (Choose One)

**Option A: Automated Training (Recommended)**
```bash
# Marathi
bash train_marathi.sh

# Telugu
bash train_telugu.sh
```

**Option B: Step-by-Step**
```bash
# 1. Train tokenizer
python train_tokenizer.py \
    --input dataset/train_mr.jsonl \
    --output tokenizers/marathi \
    --vocab-size 8000 \
    --test

# 2. Train model
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
    --device cuda

# 3. Evaluate
python evaluate.py \
    --model-dir model/marathi_small \
    --data-file dataset/validation_mr.jsonl \
    --output-file results/marathi_eval.json \
    --device cuda

# 4. Measure performance
python measure_performance.py \
    --model-dir model/marathi_small \
    --device cuda
```

---

## Project Structure

```
nlp_assignment/
├── dataset/                     # ✓ Provided data
│   ├── train_mr.jsonl          (6,900 examples)
│   ├── train_te.jsonl          (5,500 examples)
│   ├── validation_mr.jsonl     (2,600 examples)
│   └── validation_te.jsonl     (2,537 examples)
│
├── model/                       # Generated during training
│   ├── marathi_small/
│   └── telugu_small/
│
├── tokenizers/                  # Generated during training
│   ├── marathi/
│   └── telugu/
│
├── results/                     # Generated during evaluation
│   ├── marathi_eval.json
│   └── telugu_eval.json
│
├── Core Scripts (provided + fixed)
│   ├── train_small_gpt.py      ✓ Fixed for nested JSON
│   ├── infer.py                ✓ Original (works with fixed data)
│   ├── measure_performance.py  ✓ Original
│   └── sample_efficiency.py    ✓ Original
│
├── New Scripts (created)
│   ├── preprocess_data.py      ✓ Data preprocessing
│   ├── evaluate.py             ✓ Perplexity & metrics
│   └── train_tokenizer.py      ✓ Custom tokenizer
│
├── Automation (created)
│   ├── quick_start.sh          ✓ Setup script
│   ├── train_marathi.sh        ✓ Marathi pipeline
│   └── train_telugu.sh         ✓ Telugu pipeline
│
└── Documentation & Config
    ├── README.md               ✓ Complete guide
    ├── requirements.txt        ✓ Dependencies
    ├── submission_metadata.json ✓ Template
    ├── .gitignore              ✓ Git config
    └── PROJECT_SUMMARY.md      ✓ This file
```

---

## What You Still Need to Do

### Before Training
- [ ] Review dataset statistics: `python preprocess_data.py --process-all`
- [ ] Choose model configuration (tiny/small/medium)
- [ ] Decide: train from scratch or fine-tune pretrained model

### During Training
- [ ] Monitor training loss
- [ ] Save checkpoints
- [ ] Track training time

### After Training
- [ ] Update [submission_metadata.json](submission_metadata.json) with:
  - Team members
  - Model parameters
  - Training metrics
  - Performance measurements
- [ ] Write [report.pdf](report.pdf) (4-6 pages):
  - Model architecture
  - Training procedure
  - Results with tables/plots
  - Ethical considerations

### For Submission
- [ ] Ensure all files are in place
- [ ] Test inference script
- [ ] Verify reproducibility
- [ ] Create final ZIP or Git repo

---

## Key Metrics to Report

Based on assignment requirements, you must report:

### Performance Metrics
- ✓ Perplexity (primary) - from [evaluate.py](evaluate.py)
- ✓ Cross-entropy - from [evaluate.py](evaluate.py)
- ✓ Bits-per-token - from [evaluate.py](evaluate.py)
- ✓ Token accuracy - from [evaluate.py](evaluate.py)

### Efficiency Metrics
- Parameter counts - from [measure_performance.py](measure_performance.py)
- Inference latency (median, p95) - from [measure_performance.py](measure_performance.py)
- Throughput (tokens/sec) - from [measure_performance.py](measure_performance.py)
- Peak GPU memory - from [measure_performance.py](measure_performance.py)
- FLOPs/MACs - Need to add calculation
- Sample efficiency curve - Use [sample_efficiency.py](sample_efficiency.py)

---

## Tips for Success

1. **Start Small**: Train tiny model first to verify pipeline works
2. **Monitor GPU**: Use `nvidia-smi` to track GPU usage
3. **Save Checkpoints**: Keep intermediate models in case of crashes
4. **Log Everything**: Record all hyperparameters and results
5. **Test Early**: Run evaluation frequently to catch issues
6. **Document**: Update metadata.json as you go

---

## Common Issues & Solutions

### Issue: Out of Memory
**Solution**: Reduce `--batch-size` or `--max-length`

### Issue: Slow Training
**Solution**: Use smaller model (`--n-layer 4 --n-embd 256`) or GPU

### Issue: Poor Perplexity
**Solution**:
- Train longer (more epochs)
- Use custom tokenizer for better coverage
- Increase model size
- Fine-tune from pretrained model

### Issue: Import Errors
**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Need Help?

1. Check [README.md](README.md) for detailed instructions
2. Review assignment PDF for requirements
3. Check Hugging Face documentation
4. Test with smaller dataset first

---

## Timeline Suggestion

- **Week 1**: Setup, data exploration, tokenizer training
- **Week 2**: Model training experiments, hyperparameter tuning
- **Week 3**: Evaluation, metrics collection, report writing
- **Week 4**: Final testing, documentation, submission prep

**Deadline**: March 11, 2025

---

## You're All Set!

Everything is now properly organized in the `nlp_assignment/` folder. All scripts are ready to use, and the data loading issue is fixed. Good luck with your training!

**Next command to run:**
```bash
bash quick_start.sh
```
