#!/usr/bin/env python3
"""
Data validation and preprocessing utilities for training dataset.
Use this to verify your dataset before training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    from datasets import load_dataset
    import pandas as pd
except ImportError:
    print("Error: datasets library not installed. Install with: pip install datasets")
    sys.exit(1)


def validate_parquet_file(file_path: str) -> Tuple[bool, str, int]:
    """Validate parquet file and return status."""
    try:
        ds = load_dataset("parquet", data_files=file_path, split="train")
        num_samples = len(ds)
        
        # Check required columns
        required_cols = {"input", "output"}
        if not required_cols.issubset(set(ds.column_names)):
            return False, f"Missing columns. Expected {required_cols}, got {ds.column_names}", 0
        
        # Check for empty samples
        for i, sample in enumerate(ds):
            if not sample.get("input") or not sample.get("output"):
                return False, f"Empty sample at index {i}", 0
        
        return True, "✓ Valid parquet file", num_samples
    except Exception as e:
        return False, f"Error loading file: {str(e)}", 0


def validate_jsonl_file(file_path: str) -> Tuple[bool, str, int]:
    """Validate JSONL file and return status."""
    try:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        if not samples:
            return False, "Empty JSONL file", 0
        
        # Check required fields
        required_cols = {"input", "output"}
        for i, sample in enumerate(samples):
            if not required_cols.issubset(set(sample.keys())):
                return False, f"Missing fields at line {i}. Expected {required_cols}, got {sample.keys()}", 0
            if not sample.get("input") or not sample.get("output"):
                return False, f"Empty fields at line {i}", 0
        
        return True, "✓ Valid JSONL file", len(samples)
    except Exception as e:
        return False, f"Error loading file: {str(e)}", 0


def analyze_dataset(file_path: str, sample_count: int = 5) -> Dict:
    """Analyze dataset characteristics."""
    try:
        if file_path.endswith('.parquet'):
            ds = load_dataset("parquet", data_files=file_path, split="train")
        else:
            ds = load_dataset("json", data_files=file_path, split="train")
        
        # Basic stats
        stats = {
            "total_samples": len(ds),
            "columns": ds.column_names,
            "sample_count": min(sample_count, len(ds)),
            "samples": []
        }
        
        # Analyze input/output lengths
        if "input" in ds.column_names and "output" in ds.column_names:
            input_lengths = []
            output_lengths = []
            
            for sample in ds:
                if isinstance(sample["input"], str):
                    input_lengths.append(len(sample["input"].split()))
                if isinstance(sample["output"], str):
                    output_lengths.append(len(sample["output"].split()))
            
            stats["input_stats"] = {
                "min_words": min(input_lengths) if input_lengths else 0,
                "max_words": max(input_lengths) if input_lengths else 0,
                "avg_words": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            }
            
            stats["output_stats"] = {
                "min_words": min(output_lengths) if output_lengths else 0,
                "max_words": max(output_lengths) if output_lengths else 0,
                "avg_words": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            }
        
        # Get sample examples
        for i in range(min(sample_count, len(ds))):
            sample = ds[i]
            stats["samples"].append({
                "index": i,
                "input_preview": str(sample.get("input", ""))[:100] + "...",
                "output_preview": str(sample.get("output", ""))[:100] + "...",
            })
        
        return stats
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate and analyze training dataset")
    parser.add_argument(
        "--train_data",
        default="../../dataset/train.parquet",
        help="Path to training data file"
    )
    parser.add_argument(
        "--val_data",
        default="../../dataset/validation.parquet",
        help="Path to validation data file"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed analysis"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to show (with --analyze)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    
    # Check training data
    print("\n📊 Training Data:")
    print(f"  Path: {args.train_data}")
    if os.path.exists(args.train_data):
        valid, msg, num_samples = validate_parquet_file(args.train_data)
        print(f"  Status: {msg}")
        print(f"  Samples: {num_samples}")
        
        if valid and args.analyze:
            print("\n  📈 Analysis:")
            stats = analyze_dataset(args.train_data, args.samples)
            if "error" not in stats:
                print(f"    Total samples: {stats['total_samples']}")
                if "input_stats" in stats:
                    print(f"    Input length (words):  min={stats['input_stats']['min_words']}, "
                          f"max={stats['input_stats']['max_words']}, "
                          f"avg={stats['input_stats']['avg_words']:.1f}")
                    print(f"    Output length (words): min={stats['output_stats']['min_words']}, "
                          f"max={stats['output_stats']['max_words']}, "
                          f"avg={stats['output_stats']['avg_words']:.1f}")
                
                if stats["samples"]:
                    print(f"\n    Sample Examples:")
                    for sample in stats["samples"]:
                        print(f"      Sample {sample['index']}:")
                        print(f"        Input:  {sample['input_preview']}")
                        print(f"        Output: {sample['output_preview']}")
            else:
                print(f"    Error: {stats.get('error')}")
    else:
        print(f"  ❌ File not found: {args.train_data}")
    
    # Check validation data
    print("\n📊 Validation Data:")
    print(f"  Path: {args.val_data}")
    if os.path.exists(args.val_data):
        valid, msg, num_samples = validate_parquet_file(args.val_data)
        print(f"  Status: {msg}")
        print(f"  Samples: {num_samples}")
    else:
        print(f"  ❌ File not found: {args.val_data}")
    
    print("\n" + "=" * 80)
    print("✓ Validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
