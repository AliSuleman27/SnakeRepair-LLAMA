# Python IR4/OR2 LoRA Training Plan (RTX 4050)

## 1) Pre-requisites

- GPU: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
- CUDA driver installed (`nvidia-smi` works).
- Use Python 3.10 or 3.11 for best package compatibility.
- Hugging Face access token configured if model download is gated.

## 2) Environment Setup

```bash
cd /home/mrafi/codellms-fyp/SnakeRepair-LLAMA
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft accelerate bitsandbytes evaluate tqdm numpy pandas
```

Optional:
- `wandb` only if you want remote logging.

## 3) Data and Script Inputs

- Train split: `final_dataset/train.parquet`
- Validation split: `final_dataset/validation.parquet`
- Expected schema: `input`, `output`
- Trainer entrypoint: `repairllama/src/lora/llama_sft.py`
- Launcher script: `repairllama/src/lora/run_llama_sft.sh`

## 4) Training Strategy for 6 GB VRAM

- Use QLoRA (`load_in_4bit=True`, `nf4`, double quantization).
- Keep `model_max_length=1024` (your dataset target).
- Micro-batch size = 1, use gradient accumulation to raise effective batch size.
- Keep gradient checkpointing enabled.

## 5) Run Order

1. Quick smoke test (sanity):

```bash
python repairllama/src/lora/llama_sft.py \
  --model_name_or_path deepseek-ai/deepseek-coder-6.7b-base \
  --train_file final_dataset/train.parquet \
  --eval_file final_dataset/validation.parquet \
  --output_dir repairllama/python-ir4-or2-smoke \
  --model_max_length 1024 \
  --max_steps 40 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy steps \
  --eval_steps 20 \
  --save_steps 20 \
  --logging_steps 5 \
  --report_to none
```

2. Full training run:

```bash
bash repairllama/src/lora/run_llama_sft.sh
```

3. Save and evaluate adapter checkpoints using your prediction/eval flow.

## 6) If You Hit OOM

- Reduce `gradient_accumulation_steps` only if runtime is too slow (does not reduce VRAM much).
- Reduce `model_max_length` from 1024 to 896 or 768 for debugging runs.
- Lower eval frequency (`--eval_steps`) to reduce interruptions.
- If still unstable, switch base model to a smaller coder model for initial experiments.
