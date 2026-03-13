#!/bin/bash
# Example training configurations for different hardware setups
# Choose one and modify run_training.sh accordingly

# ============================================
# SETUP 1: Single GPU with 24GB VRAM (RTX 3090, RTX 4090)
# ============================================
# Use default run_training.sh as-is
# Per-device batch size: 4
# Gradient accumulation: 4
# Effective batch size: 16
# Expected training time: ~8-12 hours

# ============================================
# SETUP 2: Single GPU with 40GB VRAM (A100, H100)
# ============================================
# Modify run_training.sh:
# BATCH_SIZE=8
# GRAD_ACCUMULATION=2
# Effective batch size: 16
# Expected training time: ~4-6 hours

# ============================================
# SETUP 3: Single GPU with 16GB VRAM (RTX 4060 Ti, V100)
# ============================================
# Modify run_training.sh:
# BATCH_SIZE=2
# GRAD_ACCUMULATION=8
# LEARNING_RATE=2.5e-4  # Lower LR for smaller batches
# Effective batch size: 16
# Expected training time: ~12-16 hours
# Note: May be slower due to smaller batch size

# ============================================
# SETUP 4: Multi-GPU (2x 24GB GPUs)
# ============================================
# Modify run_training.sh to use:
# python -m torch.distributed.launch --nproc_per_node=2 train_adapter.py \
# BATCH_SIZE=4
# GRAD_ACCUMULATION=2
# Effective batch size per GPU: 8, Total: 16
# Expected training time: ~4-6 hours

# ============================================
# SETUP 5: CPU-only (very slow, not recommended for production)
# ============================================
# Comment out --fp16 in run_training.sh
# BATCH_SIZE=1
# GRAD_ACCUMULATION=16
# Use --device_map="cpu" in train_adapter.py
# Expected training time: >24 hours
# Only use for testing

# ============================================
# MEMORY OPTIMIZATION TIPS
# ============================================
# 1. For OOM errors, progressively reduce BATCH_SIZE
# 2. Increase GRAD_ACCUMULATION to maintain effective batch size
# 3. Enable gradient checkpointing: --gradient_checkpointing
# 4. Reduce model_max_length from 1024 to 512
# 5. Use 8-bit quantization (bitsandbytes) for very limited VRAM

# ============================================
# PERFORMANCE TIPS
# ============================================
# 1. Use torch.compile() for faster training (if using older torch)
# 2. Enable mixed precision (fp16) - already enabled by default
# 3. Use larger batch sizes if VRAM allows
# 4. Increase num_proc in train_adapter.py for faster tokenization
# 5. Pin memory: --dataloader_pin_memory true

# ============================================
# RECOMMENDED HYPERPARAMETERS BY DATASET SIZE
# ============================================

# For 80k+ samples (your dataset):
# - num_train_epochs: 2-3
# - learning_rate: 5e-4 (default)
# - warmup_ratio: 0.03
# - eval_steps: 500-1000

# For 20k-80k samples:
# - num_train_epochs: 3-5
# - learning_rate: 1e-3
# - warmup_ratio: 0.05

# For <20k samples:
# - num_train_epochs: 5-10
# - learning_rate: 5e-4 to 1e-4
# - warmup_ratio: 0.1
# - Consider early stopping to prevent overfitting
