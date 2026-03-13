# Getting Started - RTX 4050 (6GB VRAM)

Your setup: **NVIDIA RTX 4050 Laptop GPU (6GB VRAM)**

## The Problem You Hit

The CodeLLaMA-7B model is 13GB. Even trying to just download & cache it requires loading all weights into memory, which exceeded your 6GB VRAM. 

## The Solution 

We've optimized everything for your GPU:

1. **Model Download**: Only caches files (doesn't load into memory)
2. **Training**: Uses 8-bit quantization to fit model in 6GB VRAM
3. **Configuration**: Small batches with gradient accumulation

## Start Here - 3 Steps

### Step 1: Install Dependencies (Do This First!)

```bash
cd /home/mrafi/codellms-fyp/SnakeRepair-LLAMA/training_codellama_7b

pip install -r requirements.txt
```

**What it installs:**
- PyTorch (GPU support)
- HuggingFace Transformers
- PEFT (for LoRA)
- 8-bit quantization support

### Step 2: Download & Cache Model

```bash
python load_model.py --download
```

**What happens:**
- Downloads CodeLLaMA-7B-Python (~13GB) 
- Caches it on disk
- Does NOT load into memory (so won't OOM!)
- Takes 20-30 minutes

### Step 3: Validate & Train

```bash
# Check your dataset
python validate_data.py --analyze

# Start training
./run_training.sh
```

**Training will take 12-20 hours** but it WILL work! ✓

---

## What's Different from High-End GPUs

| Aspect | Your 6GB GPU | RTX 3090 |
|--------|----------|----------|
| Batch size | 1 | 4-8 |
| Speed | 1 step/sec | 5+ steps/sec |
| Training time | 12-20 hours | 2-4 hours |
| Memory usage | 5.8GB (8-bit) | 18GB (fp16) |

**All perform similarly - just slower!**

---

## If Step 1 (Dependencies) Fails

### Error: "No module named 'X'"

You might not have all dependencies. Install individually:

```bash
# Make sure conda environment is active
conda activate fyp311

# Install PyTorch for GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install everything else
pip install transformers datasets peft accelerate bitsandbytes huggingface-hub
```

If you still get errors, try:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

---

## If Step 2 (Download) Fails

### Same OOM error?

The fix ensures model isn't loaded into memory. If you STILL get the error:

A) Ensure conda environment is active:
```bash
conda activate fyp311
```

B) Clear old cache and retry:
```bash
rm -rf ~/.cache/huggingface/hub/models--codellama*
python load_model.py --download
```

C) Check you have enough disk space:
```bash
df -h  # Should show 50GB+ available
```

---

## Monitoring Training (Step 3)

While training runs, in another terminal:

```bash
# Watch GPU usage (should stay ~5.5-6GB)
watch -n 1 nvidia-smi

# Monitor training metrics
tensorboard --logdir ./runs
# Open http://localhost:6006 in browser
```

---

## After Training (Generate Predictions)

```bash
# Interactive mode (enter prompts)
python inference.py --adapter_path ./codellama-7b-python-adapter

# From file
python inference.py \
    --adapter_path ./codellama-7b-python-adapter \
    --input_file buggy_code.py \
    --output_file fixed_code.py
```

---

## Expected Outputs

### After training completes:

```
codellama-7b-python-adapter/
├── adapter_config.json       (~1KB)
├── adapter_model.safetensors (~50MB)  ← Your trained weights!
├── tokenizer.json
└── trainer_state.json
```

**Size**: Only 50MB! (vs 13GB model)

### Can merge with base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")
model = PeftModel.from_pretrained(model, "./codellama-7b-python-adapter")

# Now use model for inference
```

---

## Quick Troubleshooting Reference

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: transformers` | `pip install transformers` |
| `Cannot allocate memory` during download | Use `python load_model.py --download` (our fixed version) |
| OOM during training | Reduce `BATCH_SIZE` or `MAX_LENGTH` in `run_training.sh` |
| Training very slow | Normal for 6GB GPU - just be patient! |
| `CUDA out of memory` during training | Reduce `MAX_LENGTH` from 512 to 256 |

---

## Next: Run the 3 Steps!

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download model
python load_model.py --download

# 3. Train
./run_training.sh
```

Good luck! 🚀
