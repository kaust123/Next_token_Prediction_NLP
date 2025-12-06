# macOS Setup Guide

## ⚠️ Important: CPU Training on macOS

Your Mac doesn't have CUDA support (CUDA is NVIDIA-only). The scripts have been automatically configured to use **CPU** on macOS.

### Check Your Device

Run this first to see what's available:
```bash
python check_device.py
```

This will show:
- Available devices (CPU/CUDA/MPS)
- Recommended device for training
- Performance test results
- Optimal configurations

---

## 🚀 Quick Start for macOS

### 1. Setup
```bash
cd "/Users/soumik.pramanik/National Performance Automation Dashboard/assignment/vision_gpt/nlp_assignment"

# Check device availability
python check_device.py

# Run setup
bash quick_start.sh
```

### 2. Training (CPU-Optimized)

**For faster training on macOS, use smaller configurations:**

```bash
# Train tiny model (recommended for CPU)
python train_small_gpt.py \
    --train-file dataset/train_mr.jsonl \
    --output-dir model/marathi_tiny \
    --tokenizer gpt2 \
    --init-model none \
    --n-layer 4 \
    --n-head 4 \
    --n-embd 256 \
    --batch-size 4 \
    --epochs 5 \
    --lr 5e-4 \
    --device cpu
```

**Or use the automated scripts (already configured for macOS):**
```bash
# These will automatically use CPU on macOS
bash train_marathi.sh
bash train_telugu.sh
```

---

## ⚡ Performance Optimization for CPU

### Model Size Recommendations

| Config | Layers | Heads | Embed | Params | Training Time* |
|--------|--------|-------|-------|--------|----------------|
| Tiny   | 4      | 4     | 256   | ~8M    | 2-3 hours      |
| Small  | 6      | 6     | 384   | ~18M   | 4-6 hours      |
| Medium | 6      | 8     | 512   | ~30M   | 8-12 hours     |

*Approximate for 6,900 examples, 10 epochs on M1/M2 Mac

### Recommended Settings for CPU

```bash
# Smallest viable model
--n-layer 4 --n-head 4 --n-embd 256 --batch-size 4 --epochs 5

# Balanced (may be slow)
--n-layer 6 --n-head 6 --n-embd 384 --batch-size 8 --epochs 8

# Only if you have time
--n-layer 6 --n-head 8 --n-embd 512 --batch-size 16 --epochs 10
```

### Speed Up Tips

1. **Use smaller vocabulary**
   ```bash
   python train_tokenizer.py --vocab-size 4000  # Instead of 8000
   ```

2. **Reduce sequence length**
   ```bash
   python train_small_gpt.py --max-length 128  # Instead of 256
   ```

3. **Fewer epochs initially**
   ```bash
   python train_small_gpt.py --epochs 3  # Test first, then increase
   ```

4. **Use pretrained model (fine-tuning is faster)**
   ```bash
   python train_small_gpt.py \
       --init-model gpt2 \
       --tokenizer gpt2 \
       --epochs 3 \
       --lr 2e-5
   ```

---

## 🍎 Apple Silicon (M1/M2/M3) - MPS Support

If you have Apple Silicon, you can try using MPS (Metal Performance Shaders):

```bash
# Check if MPS is available
python check_device.py

# If MPS is available, use it
python train_small_gpt.py --device mps
```

**Note:** MPS support is experimental and may not work with all operations.

---

## 🔄 Alternative: Use Cloud GPU

If training on CPU is too slow, consider:

### Google Colab (Free GPU)
1. Upload your code and data to Google Drive
2. Open a Colab notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Install requirements and run training

### Other Options
- **Kaggle Notebooks** - Free GPU (30h/week)
- **Paperspace Gradient** - Free tier available
- **AWS/GCP/Azure** - Pay per use

---

## 🧪 Test Your Setup

```bash
# 1. Check device
python check_device.py

# 2. Quick test with tiny model (should complete in ~10 min)
python train_small_gpt.py \
    --train-file dataset/train_mr.jsonl \
    --output-dir model/test \
    --tokenizer gpt2 \
    --n-layer 2 \
    --n-head 2 \
    --n-embd 128 \
    --batch-size 2 \
    --epochs 1 \
    --device cpu \
    --log-every 10

# 3. If successful, run full training
bash train_marathi.sh
```

---

## 📊 Expected Training Times (M1/M2 Mac, CPU)

| Dataset | Model Size | Epochs | Estimated Time |
|---------|------------|--------|----------------|
| Marathi (6.9K) | Tiny (4L, 256d) | 5 | 2-3 hours |
| Marathi (6.9K) | Small (6L, 384d) | 8 | 6-8 hours |
| Telugu (5.5K) | Tiny (4L, 256d) | 5 | 1.5-2.5 hours |
| Telugu (5.5K) | Small (6L, 384d) | 8 | 5-7 hours |

**Tip:** Run training overnight or during long breaks!

---

## 🐛 Troubleshooting

### "AssertionError: Torch not compiled with CUDA enabled"
✓ **Fixed!** Scripts now auto-detect macOS and use CPU.

### Training is too slow
- Use smaller model configuration
- Reduce batch size
- Reduce number of epochs
- Use pretrained model for fine-tuning
- Consider cloud GPU

### Out of memory
```bash
# Reduce batch size
--batch-size 2

# Reduce sequence length
--max-length 64

# Smaller model
--n-layer 2 --n-embd 128
```

### MPS errors
```bash
# Fall back to CPU
--device cpu
```

---

## ✅ Verification Checklist

- [ ] Run `python check_device.py` - confirms CPU available
- [ ] Run `bash quick_start.sh` - verifies setup
- [ ] Test tiny model training (2 layers, 1 epoch)
- [ ] If successful, run full training with bash scripts
- [ ] Scripts automatically use CPU on macOS

---

## 📚 Additional Resources

- PyTorch on Mac: https://pytorch.org/get-started/locally/
- Apple Silicon Performance: https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
- Hugging Face Docs: https://huggingface.co/docs/transformers

---

**You're all set for CPU training on macOS! 🚀**
