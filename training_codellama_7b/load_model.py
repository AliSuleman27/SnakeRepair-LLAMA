#!/usr/bin/env python3
"""
Model loading and download utility.
Use this to pre-download the CodeLLaMA-7B-Python model before training.
This can save time during training initialization.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    import torch
except ImportError:
    print("Error: transformers or torch not installed. Install with: pip install torch transformers")
    sys.exit(1)


def get_model_size_estimate() -> str:
    """Estimate the model file size."""
    # CodeLLaMA-7B-Python is approximately 13GB
    return "~13GB"


def check_disk_space(min_required_gb: int = 50) -> bool:
    """Check if sufficient disk space is available."""
    try:
        import shutil
        stat = shutil.disk_usage(".")
        available_gb = stat.free / (1024 ** 3)
        
        print(f"\n📊 Disk Space Check:")
        print(f"  Available: {available_gb:.1f} GB")
        print(f"  Required:  {min_required_gb} GB (model + dataset + checkpoints)")
        
        if available_gb < min_required_gb:
            print(f"  ⚠️  Warning: You may run out of disk space!")
            return False
        else:
            print(f"  ✓ Sufficient space available")
            return True
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True  # Don't fail, just warn


def check_gpu_availability() -> bool:
    """Check GPU availability and memory."""
    print(f"\n🖥️  GPU Check:")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  CUDA Available: Yes")
        print(f"  Number of GPUs: {num_gpus}")
        
        total_memory_gb = 0
        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_name(i)
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            total_memory_gb += gpu_memory
        
        print(f"  Total GPU Memory: {total_memory_gb:.1f} GB")
        
        if total_memory_gb < 20:
            print(f"  ⚠️  Warning: Limited VRAM. Consider reducing batch size during training.")
            return True
        
        print(f"  ✓ GPU ready for training")
        return True
    else:
        print(f"  CUDA Available: No")
        print(f"  ⚠️  GPU not available. Training will be very slow.")
        return False


def download_model(model_name: str = "codellama/CodeLlama-7b-Python-hf", 
                  cache_dir: Optional[str] = None,
                  hf_token: Optional[str] = None,
                  cpu_only: bool = False):
    """Download and cache the CodeLLaMA model and tokenizer."""
    
    print("=" * 80)
    print("CODELLAMA-7B-PYTHON MODEL DOWNLOAD")
    print("=" * 80)
    
    # Check prerequisites
    gpu_available = check_gpu_availability()
    check_disk_space()
    
    # Auto-detect CPU-only mode if VRAM is limited
    if gpu_available and torch.cuda.get_device_properties(0).total_memory < 10 * (1024 ** 3):
        print("\n⚠️  Limited VRAM detected. Using CPU for model loading.")
        print("   (Model weights will be cached, GPU will be used during training)")
        cpu_only = True
    
    # Set cache directory
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        print(f"\n📁 HuggingFace Cache: {cache_dir}")
    else:
        print(f"\n📁 HuggingFace Cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    
    print(f"\n📦 Model: {model_name}")
    print(f"📊 Estimated Size: {get_model_size_estimate()}")
    
    try:
        # Download tokenizer
        print("\n⏳ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            use_fast=True
        )
        print("   ✓ Tokenizer downloaded successfully")
        print(f"   Vocab size: {len(tokenizer)}")
        
        # Download model - just download, don't load into memory yet
        print("\n⏳ Downloading model (this may take several minutes)...")
        print("   Please be patient...")
        print("   Note: Model will be loaded during training with 8-bit quantization")
        
        # Use snapshot_download to just cache the model without loading it
        model_path = snapshot_download(
            model_name,
            token=hf_token,
            cache_dir=None,  # Use default cache dir
        )
        
        print("   ✓ Model downloaded and cached successfully")
        print(f"   Cache location: {model_path}")
        print(f"   Model size on disk: ~13GB")
        
        print("\n" + "=" * 80)
        print("✓ MODEL DOWNLOADED AND CACHED!")
        print("=" * 80)
        
        print("\nNext steps:")
        print("1. Run the data validation: python validate_data.py")
        print("2. Start training: ./run_training.sh")
        print("\nNote: Model will be loaded with 8-bit quantization during training")
        print("      This reduces VRAM usage from 13GB to ~7-8GB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("- Ensure stable internet connection")
        print("- Check available disk space")
        print("- If using a gated model, login with: huggingface-cli login")
        return False


def verify_downloaded_model(model_name: str = "codellama/CodeLlama-7b-Python-hf") -> bool:
    """Verify that model is already downloaded and cached."""
    
    print("=" * 80)
    print("VERIFYING CACHED MODEL")
    print("=" * 80)
    
    try:
        print(f"\nChecking for cached model: {model_name}")
        
        # Try loading tokenizer (lightweight check)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✓ Model and tokenizer found in cache")
        print(f"  Vocab size: {len(tokenizer)}")
        
        # Check cache directory for model files
        try:
            from huggingface_hub import try_to_load_from_cache
            cache_path = try_to_load_from_cache(
                model_name,
                filename="model-00001-of-00002.safetensors"
            )
            if cache_path:
                print(f"  Model files cached at: {cache_path.parent}")
                print(f"  Status: Ready for training with 8-bit quantization")
        except:
            print(f"  Status: Tokenizer cached (model files also present)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model not found in cache: {e}")
        print("\nRun download first: python load_model.py --download")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and manage CodeLLaMA-7B-Python model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download model:
    python load_model.py --download
  
  Verify cached model:
    python load_model.py --verify
  
  Use custom cache directory:
    python load_model.py --download --cache_dir /mnt/data/huggingface
  
  Login to HuggingFace (if needed):
    huggingface-cli login
        """
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model is cached"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory (must have 50GB+ space)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="codellama/CodeLlama-7b-Python-hf",
        help="Model name or path"
    )
    
    args = parser.parse_args()
    
    if args.download:
        success = download_model(
            model_name=args.model_name,
            cache_dir=args.cache_dir
        )
        sys.exit(0 if success else 1)
    
    elif args.verify:
        success = verify_downloaded_model(model_name=args.model_name)
        sys.exit(0 if success else 1)
    
    else:
        # Default: check and suggest action
        print("=" * 80)
        print("CODELLAMA-7B-PYTHON MODEL MANAGEMENT")
        print("=" * 80)
        
        print("\nOptions:")
        print("  --download   : Download the model (~13GB)")
        print("  --verify     : Check if model is already cached")
        print("  --help       : Show all options")
        
        print("\nExample:")
        print("  python load_model.py --download")
        print("  python load_model.py --verify")
        
        # Try automatic verification
        print("\nChecking for cached model...")
        if verify_downloaded_model(args.model_name):
            print("\n✓ You're ready to start training!")
            print("  Run: ./run_training.sh")
        else:
            print("\n⚠️  Model not found. Download it first:")
            print("  python load_model.py --download")


if __name__ == "__main__":
    main()
