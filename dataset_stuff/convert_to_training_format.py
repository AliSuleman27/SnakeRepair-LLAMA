"""
Convert cleaned_train.csv to HuggingFace Datasets format (JSON/Parquet)
Split into 90% training and 10% validation with stratified sampling from each type.
"""

import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    csv_path = Path("/home/mrafi/codellms-fyp/SnakeRepair-LLAMA/training_dataset/cleaned_train.csv")
    output_dir = Path("/home/mrafi/codellms-fyp/SnakeRepair-LLAMA/training_dataset/formatted_training_data")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Converting training data to HuggingFace Datasets format")
    print("=" * 80)
    
    # Read the CSV file
    print(f"\n[1/5] Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Total samples: {len(df)}")
    print(f"    Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ['IR4', 'OR2', 'type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove any rows with missing IR4 or OR2
    print(f"\n[2/5] Validating data...")
    initial_count = len(df)
    df = df.dropna(subset=['IR4', 'OR2'])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"    Removed {removed} rows with missing IR4/OR2")
    print(f"    Valid samples: {len(df)}")
    
    # Display type distribution before split
    print(f"\n[3/5] Type distribution (before split):")
    type_dist = df['type'].value_counts()
    for dtype, count in type_dist.items():
        print(f"    {dtype}: {count} ({100*count/len(df):.2f}%)")
    
    # Stratified split: 10% validation (stratified by type), 90% training
    print(f"\n[4/5] Splitting data (90% train, 10% val, stratified by type)...")
    train_list = []
    val_list = []
    
    for dtype in df['type'].unique():
        subset = df[df['type'] == dtype].reset_index(drop=True)
        subset_len = len(subset)
        train_subset, val_subset = train_test_split(
            subset,
            test_size=0.1,
            random_state=42
        )
        train_list.append(train_subset)
        val_list.append(val_subset)
        print(f"    {dtype}: {len(train_subset)} train, {len(val_subset)} val")
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    
    print(f"\n    Total Training samples: {len(train_df)}")
    print(f"    Total Validation samples: {len(val_df)}")
    
    # Convert to HuggingFace format
    print(f"\n[5/5] Converting to HuggingFace format...")
    
    # Create training data
    train_data = []
    for _, row in train_df.iterrows():
        train_data.append({
            "input": row['IR4'],
            "output": row['OR2']
        })
    
    # Create validation data
    val_data = []
    for _, row in val_df.iterrows():
        val_data.append({
            "input": row['IR4'],
            "output": row['OR2']
        })
    
    # Save as JSON Lines format
    print(f"\n    Saving training data (JSON Lines format)...")
    train_jsonl = output_dir / "train.jsonl"
    with open(train_jsonl, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"    ✓ {train_jsonl}")
    
    print(f"    Saving validation data (JSON Lines format)...")
    val_jsonl = output_dir / "validation.jsonl"
    with open(val_jsonl, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    print(f"    ✓ {val_jsonl}")
    
    # Also save as Parquet for efficient loading
    print(f"\n    Saving as Parquet format (better for HF datasets)...")
    
    train_table = pa.Table.from_pylist(train_data)
    train_parquet = output_dir / "train.parquet"
    pq.write_table(train_table, train_parquet)
    print(f"    ✓ {train_parquet} ({len(train_data)} samples)")
    
    val_table = pa.Table.from_pylist(val_data)
    val_parquet = output_dir / "validation.parquet"
    pq.write_table(val_table, val_parquet)
    print(f"    ✓ {val_parquet} ({len(val_data)} samples)")
    
    # Save metadata
    print(f"\n    Saving metadata...")
    metadata = {
        "total_samples": len(train_df) + len(val_df),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "train_percentage": 90,
        "validation_percentage": 10,
        "sampling_method": "stratified by type (10% from each)",
        "types": {
            dtype: {
                "total": int(type_dist[dtype]),
                "train": int((train_df['type'] == dtype).sum()),
                "validation": int((val_df['type'] == dtype).sum())
            }
            for dtype in df['type'].unique()
        },
        "format": {
            "input_field": "IR4 (buggy function with commented buggy lines + <FILL_ME> placeholder)",
            "output_field": "OR2 (fixed code snippet)",
        },
        "usage": {
            "python": "from datasets import load_dataset\nds = load_dataset('parquet', data_files='train.parquet')",
            "note": "Test data: use benchmark datasets from repairllama/benchmarks/"
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✓ {metadata_path}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  ✓ train.jsonl ({len(train_data)} samples)")
    print(f"  ✓ train.parquet ({len(train_data)} samples)")
    print(f"  ✓ validation.jsonl ({len(val_data)} samples)")
    print(f"  ✓ validation.parquet ({len(val_data)} samples)")
    print(f"  ✓ metadata.json")
    
    print(f"\nData distribution (stratified by type):")
    for dtype in df['type'].unique():
        train_count = (train_df['type'] == dtype).sum()
        val_count = (val_df['type'] == dtype).sum()
        total = train_count + val_count
        print(f"  {dtype:15} Total: {total:6} | Train: {train_count:6} (90%) | Val: {val_count:6} (10%)")
    
    print(f"\nNext steps:")
    print(f"  1. Use train.parquet for training with HuggingFace Datasets")
    print(f"  2. Use validation.parquet for validation during training")
    print(f"  3. Use benchmark datasets from repairllama/benchmarks/ for testing")
    print(f"\nExample loading code:")
    print(f"  from datasets import load_dataset")
    print(f"  train_ds = load_dataset('parquet', data_files='{output_dir}/train.parquet')")
    print(f"  val_ds = load_dataset('parquet', data_files='{output_dir}/validation.parquet')")
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    main()
