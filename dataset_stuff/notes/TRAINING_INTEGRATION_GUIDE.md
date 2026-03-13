# Quick Integration Guide - RepairLLaMA Training

## 🚀 Quick Start (3 Steps)

### Step 1: Update Your Training Script
In `repairllama/src/lora/llama_sft.py`, modify the data loading:

```python
from datasets import load_dataset

# Old (if using custom CSV loading):
# dataset = load_csv("cleaned_train.csv")

# New (HuggingFace Datasets):
dataset = load_dataset(
    'parquet',
    data_files={
        'train': 'training_dataset/formatted_training_data/train.parquet',
        'eval': 'training_dataset/formatted_training_data/validation.parquet'
    }
)

train_dataset = dataset['train']
eval_dataset = dataset['eval']
```

### Step 2: Ensure Tokenizer Handles 'input' and 'output'
Your tokenization function must work with the new field names:

```python
def generate_and_tokenize_prompt(sample, tokenizer, training_args):
    """
    Sample structure:
    {
        'input': '<IR4 - buggy function with <FILL_ME>>',
        'output': '<OR2 - fixed snippet>'
    }
    """
    input_text = sample['input']      # IR4
    target = sample['output']         # OR2
    
    full_text = input_text + target + tokenizer.eos_token
    
    # Tokenize full prompt
    tokenized_full = tokenize(full_text, tokenizer, training_args)
    
    # Tokenize input only to create attention mask
    tokenized_input = tokenize(input_text, tokenizer, training_args)
    input_len = len(tokenized_input["input_ids"])
    
    # Mask input tokens from loss (only train on output)
    tokenized_full["labels"] = (
        [-100] * input_len + 
        tokenized_full["labels"][input_len:]
    )
    
    return tokenized_full
```

### Step 3: Run Training
```bash
cd /home/mrafi/codellms-fyp/SnakeRepair-LLAMA

conda activate fyp311

python repairllama/src/lora/llama_sft.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir "./repairllama/trained_model" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 500 \
    --seed 42
```

## 📊 Data Format Validation

### Input Field (IR4)
```
Contains:
✓ Buggy function code
✓ Buggy lines commented out (prefixed with '# ')
✓ <FILL_ME> placeholder where fix goes
✓ Rest of function body

Never contains:
✗ Docstrings or comments (except buggy markers)
✗ Function signature changes
✗ Multiple functions
```

### Output Field (OR2)
```
Contains:
✓ Fixed code snippet only
✓ What should replace <FILL_ME>
✓ Proper indentation

Never contains:
✗ Function definition
✗ XML tags or markers
✗ Surrounding context
```

## 📈 Data Statistics

```
Total samples: 89,553
├── Training:   80,597 (90%)
│   ├── RunBugRun:  62,783 (78%)
│   ├── repairllama: 15,339 (19%)
│   └── PyResBugs:   2,475 (3%)
│
└── Validation:  8,956 (10%)
    ├── RunBugRun:  6,976 (78%)
    ├── repairllama: 1,705 (19%)
    └── PyResBugs:   275 (3%)

Token distribution:
├── IR4 (input):
│   ├── Mean: 228 tokens
│   ├── Median: 179 tokens
│   └── Max: 1,010 tokens
│
└── OR2 (output):
    ├── Mean: 53 tokens
    ├── Median: 29 tokens
    └── Max: 813 tokens
```

## 🔍 How to Inspect Data

### Using Python
```python
from datasets import load_dataset

# Load training data
ds = load_dataset('parquet', 
    data_files='training_dataset/formatted_training_data/train.parquet')

# Inspect a sample
sample = ds['train'][0]
print("="*80)
print("INPUT (IR4):")
print(sample['input'])
print("\n" + "="*80)
print("OUTPUT (OR2):")
print(sample['output'])
print("="*80)
```

### Using JSON Lines directly
```bash
# View first sample
head -n 1 training_dataset/formatted_training_data/train.jsonl | \
    python3 -m json.tool | less

# Count samples
wc -l training_dataset/formatted_training_data/train.jsonl

# Get random sample
shuf -n 1 training_dataset/formatted_training_data/train.jsonl | \
    python3 -m json.tool
```

## 🧪 Testing the Integration

Before full training, test your pipeline:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load one sample
ds = load_dataset('parquet', 
    data_files='training_dataset/formatted_training_data/train.parquet')
sample = ds['train'][0]

# Test tokenization
input_ids = tokenizer.encode(sample['input'] + sample['output'])
print(f"✓ Sample tokenizes successfully")
print(f"✓ Token count: {len(input_ids)}")
print(f"✓ Max tokens: 4096")
print(f"✓ Status: {'PASS' if len(input_ids) <= 4096 else 'FAIL'}")
```

## 📁 File Locations

| What | Where |
|------|-------|
| **Training data** | `training_dataset/formatted_training_data/train.parquet` |
| **Validation data** | `training_dataset/formatted_training_data/validation.parquet` |
| **Metadata** | `training_dataset/formatted_training_data/metadata.json` |
| **Conversion script** | `preprocessing_scripts/convert_to_training_format.py` |
| **Format docs** | `PEFT_LoRA_DATA_FORMAT.md` |
| **Full report** | `training_dataset/formatted_training_data/README.md` |
| **Test data** | `repairllama/benchmarks/` |

## ❓ Troubleshooting

### Error: "No dataset 'input' or 'output' keys"
→ Make sure you're loading from the formatted data directory, not the original CSV

### Error: "Token count exceeds 4096"
→ All samples are pre-validated. This shouldn't happen. Check tokenizer padding/truncation settings.

### Data looks wrong
→ Verify you're using parquet files, not JSON Lines. Both contain identical data, but parquet is faster.

### Need to change split ratio
1. Edit `preprocessing_scripts/convert_to_training_format.py`
2. Change `test_size=0.1` to desired percentage
3. Run: `python preprocessing_scripts/convert_to_training_format.py`
4. Choose new output directory

## ✅ Checklist Before Training

- [ ] `train.parquet` loaded successfully
- [ ] `validation.parquet` loaded successfully  
- [ ] Sample has both 'input' and 'output' keys
- [ ] Tokenization test passed
- [ ] Model checkpoint downloaded
- [ ] Output directory created
- [ ] Training args configured
- [ ] GPU/compute resources allocated

## 📞 Support

- **Data Format Questions**: See `PEFT_LoRA_DATA_FORMAT.md`
- **Training Issues**: See `repairllama/src/lora/llama_sft.py`
- **Data Generation**: See `preprocessing_scripts/filter_and_create_representation.py`
- **Examples**: See `repairllama/example/example.ipynb`
