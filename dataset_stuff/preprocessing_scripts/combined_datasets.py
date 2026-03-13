import os
import glob
import pandas as pd

# Directory containing your CSV files
DATA_DIR = "/home/mrafi/codellms-fyp/SnakeRepair-LLAMA/finalized_datasets/actual_data"

# Output file path
OUTPUT_CSV = os.path.join(DATA_DIR, "final_original_dataset.csv")

def combine_csvs(input_dir: str, output_path: str):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dataframes = []
    for csv_file in csv_files:
        print(f"Reading: {csv_file}")
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    # Concatenate them (assumes same columns)
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save to a single CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV written to: {output_path}")

if __name__ == "__main__":
    combine_csvs(DATA_DIR, OUTPUT_CSV)