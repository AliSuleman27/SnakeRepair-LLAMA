# CodeLLaMA-7B-Python LoRA Adapter Training

This folder contains everything needed to train a LoRA adapter on CodeLLaMA-7B-Python with your custom dataset.

## Overview

- **Model**: CodeLLaMA-7B-Python
- **Training Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Dataset**: Your Python bug repair dataset (train.jsonl + validation.jsonl in parquet format)
- **Framework**: HuggingFace Transformers + PEFT

## Files

- `train_adapter.py` - Main training script
- `run_training.sh` - Shell script to run training with recommended parameters
- `requirements.txt` - Python dependencies
- `config_examples.json` - (Optional) Example configurations for different hardware setups
- `README.md` - This file

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate your conda environment (fyp311)
conda activate fyp311

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Dataset

Your dataset should be in the `dataset/` folder:
```
dataset/
├── train.parquet (80,597 samples)
└── validation.parquet (8,956 samples)
```

The script expects each sample to have:
- `input`: buggy code snippet with placeholder
- `output`: fixed code snippet

You can verify this with:
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('parquet', data_files='../../dataset/train.parquet', split='train')
print(f'Samples: {len(ds)}')
print(f'Columns: {ds.column_names}')
print(f'Example: {ds[0]}')
"
```

### 3. Download the Model

Before training, the model needs to be downloaded. This happens automatically on first run, but you can pre-download:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'codellama/CodeLlama-7b-Python-hf'
print('Downloading model...')
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
print('Model downloaded successfully!')
"
```

**Note**: The model is ~13GB. Ensure you have sufficient disk space and bandwidth.

## Training

### Option 1: Quick Start (Recommended)

```bash
chmod +x run_training.sh
./run_training.sh
```

This uses default parameters optimized for a single GPU with reasonable batch size.

### Option 2: Custom Parameters

```bash
python train_adapter.py \
    --model_name_or_path codellama/CodeLlama-7b-Python-hf \
    --train_data_path ../../dataset/train.parquet \
    --validation_data_path ../../dataset/validation.parquet \
    --output_dir ./codellama-7b-python-adapter \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-4 \
    --model_max_length 1024
```

See `train_adapter.py` for all available arguments.

## Training Parameters Explained

| Parameter | Value | Notes |
|-----------|-------|-------|
| `per_device_train_batch_size` | 4 | Reduce if OOM; increase if memory available |
| `gradient_accumulation_steps` | 4 | Effective batch size = 4 × 4 = 16 |
| `learning_rate` | 5e-4 | Standard for LoRA fine-tuning |
| `num_train_epochs` | 3 | Can increase/decrease based on dataset size |
| `model_max_length` | 1024 | Max tokens per sample; increase with caution |
| `eval_steps` | 500 | Evaluate every 500 steps |
| `save_steps` | 500 | Save checkpoint every 500 steps |

## Hardware Requirements

### Minimum
- **GPU**: NVIDIA with 24GB+ VRAM (e.g., RTX 3090, A100 40GB)
- **RAM**: 32GB system RAM
- **Storage**: 50GB (model + dataset + checkpoints)

### Recommended
- **GPU**: NVIDIA with 40GB+ VRAM (A100, H100)
- **RAM**: 64GB+ system RAM
- **Storage**: 100GB+

### If you have less VRAM
Adjust batch size in `run_training.sh`:
```bash
# For 16GB GPU
BATCH_SIZE=2
GRAD_ACCUMULATION=8  # Keep effective batch size ≥ 16
```

## Output

After training completes, the adapter will be saved in `./codellama-7b-python-adapter/` containing:

```
codellama-7b-python-adapter/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── tokenizer_config.json
├── tokenizer.json
├── trainer_state.json        # Training state (resumable)
└── training_args.bin         # Serialized arguments
```

**Size**: LoRA adapter is ~50MB (much smaller than full model!)

## Using the Trained Adapter

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model_name = "codellama/CodeLlama-7b-Python-hf"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load LoRA adapter
adapter_path = "./codellama-7b-python-adapter"
model = PeftModel.from_pretrained(model, adapter_path)
model = model.to("cuda")

# Generate predictions
input_text = "def buggy_function():\n    <FILL_ME>"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=256)
print(tokenizer.decode(output[0]))
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./runs
```
Then open http://localhost:6006

### Check Checkpoints
```bash
ls -lh ./codellama-7b-python-adapter/checkpoint-*
```

## Resuming Training

If training is interrupted, resume from the latest checkpoint:

```bash
python train_adapter.py \
    --model_name_or_path codellama/CodeLlama-7b-Python-hf \
    --train_data_path ../../dataset/train.parquet \
    --validation_data_path ../../dataset/validation.parquet \
    --output_dir ./codellama-7b-python-adapter \
    --resume_from_checkpoint ./codellama-7b-python-adapter/checkpoint-1000
```

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--per_device_train_batch_size 2

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Enable 8-bit quantization (modify train_adapter.py)
# Uncomment BitsAndBytesConfig in the AutoModelForCausalLM.from_pretrained() call
```

### Model Download Issues
```bash
# Set HF cache directory
export HF_HOME=/path/to/large/disk

# Verify model access
huggingface-cli login  # If using a gated model
```

### Slow Data Loading
- Ensure `num_proc=8` in `train_adapter.py` matches your CPU cores
- Pre-compute and cache tokenized datasets if running multiple times

## Advanced Configuration

### For Multi-GPU Training
```bash
# Modify run_training.sh to use distributed training
python -m torch.distributed.launch --nproc_per_node=2 train_adapter.py \
    --model_name_or_path codellama/CodeLlama-7b-Python-hf \
    ...
```

### Using W&B for Logging
```bash
# Install wandb
pip install wandb

# Login
wandb login

# Modify run_training.sh
--report_to "wandb"
```

## Next Steps

1. **Verify data**: Run the verification command above
2. **Download model**: Pre-download the CodeLLaMA model
3. **Run training**: Execute `./run_training.sh`
4. **Monitor**: Check TensorBoard for training progress
5. **Evaluate**: Use the trained adapter on your test set

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [CodeLLaMA Model Card](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)
- [LoRA Paper](https://arxiv.org/abs/2106.09714)

## Support

For issues or questions:
1. Check training logs in `./runs/` (TensorBoard)
2. Review the training script error messages
3. Consult HuggingFace documentation for specific errors
