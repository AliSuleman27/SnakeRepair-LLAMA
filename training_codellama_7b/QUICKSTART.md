# Quick Start Guide - CodeLLaMA-7B-Python Training

## 5-Minute Quick Start

### 1. Install Dependencies
```bash
# Make sure you're in the training folder
cd training_codellama_7b

# Activate your environment
conda activate fyp311

# Install requirements (takes 5-10 min)
pip install -r requirements.txt
```

### 2. Download Model
```bash
# Download the CodeLLaMA-7B-Python model (takes 15-30 min, ~13GB)
python load_model.py --download
```

### 3. Validate Data
```bash
# Verify your dataset is ready
python validate_data.py --analyze
```

### 4. Start Training
```bash
# Make script executable
chmod +x run_training.sh

# Run training (takes 4-12 hours depending on GPU)
./run_training.sh
```

## What Gets Created

After training completes, you'll have:
```
codellama-7b-python-adapter/  ← Your trained adapter (~50MB)
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── trainer_state.json
```

## Using Your Trained Adapter

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")

# Load your trained adapter
model = PeftModel.from_pretrained(model, "./codellama-7b-python-adapter")
model = model.to("cuda")

# Generate
input_text = "def buggy():\n    <FILL_ME>"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=256)
print(tokenizer.decode(output[0]))
```

Or use the inference script:
```bash
python inference.py --adapter_path ./codellama-7b-python-adapter \
    --prompt "def buggy():\n    <FILL_ME>"
```

## Troubleshooting

### Out of Memory?
Edit `run_training.sh`:
```bash
BATCH_SIZE=2          # Reduce from 4
GRAD_ACCUMULATION=8   # Increase from 4
```

### Model Download Issues?
```bash
# Check connection and available space:
python -c "import shutil; print(shutil.disk_usage('/')[2] / 1024**3, 'GB free')"

# Pre-download to custom location:
python load_model.py --download --cache_dir /mnt/data
```

### Training Too Slow?
- Check GPU utilization: `nvidia-smi` (should be 80%+)
- Increase batch size if memory allows
- Use multi-GPU if available

## Files Reference

| File | Purpose |
|------|---------|
| `train_adapter.py` | Main training script |
| `run_training.sh` | Default training runner |
| `load_model.py` | Download/verify CodeLLaMA model |
| `validate_data.py` | Check dataset format |
| `inference.py` | Generate predictions with adapter |
| `README.md` | Detailed documentation |

## Next Steps

1. **Monitor Training**: Check `runs/` folder for TensorBoard logs
2. **Save Checkpoints**: Automatically saved every 500 steps
3. **Evaluate**: Use validation set metrics to track progress
4. **Iterate**: Adjust hyperparameters based on results

## Estimated Timelines

| GPU | Training Time | Notes |
|-----|---------------|-------|
| RTX 3090 (24GB) | 8-12 hours | Default settings |
| A100 (40GB) | 4-6 hours | Larger batch size possible |
| RTX 4090 (24GB) | 6-8 hours | Fast |
| V100 (32GB) | 12-16 hours | Older hardware |

## Support

- **Documentation**: See `README.md` for comprehensive guide
- **Config Examples**: Check `CONFIG_EXAMPLES.sh` for your hardware
- **Logs**: Check `runs/` directory for TensorBoard outputs
- **Errors**: Read the error messages carefully, they usually explain the issue

---

**You're all set!** 🚀 Start training:
```bash
./run_training.sh
```
