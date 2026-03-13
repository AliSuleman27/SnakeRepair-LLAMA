# PEFT-LoRA Data Format for Python Function-Level Bugs

## Overview
Based on the RepairLLaMA codebase, here is the exact format needed to store data for PEFT-LoRA fine-tuning on Python function-level bugs.

## Data Structure

### Primary Format: HuggingFace Datasets (JSON/Parquet)

Your data must have these **two essential fields**:
- `input` - The Input Representation (IR)
- `output` - The Output Representation (OR)

### For Python Function-Level Bugs Specifically:

#### Input Representation: **IR4** (Recommended)
The buggy function with:
1. Commented out buggy lines
2. A sentinel `<FILL_ME>` placeholder where the fix should go
3. Rest of the function body below

**Example:**
```python
def _make_sqlite_account_info(self, env=None, last_upgrade_to_run=None):
    # Buggy code:
    # with mock.patch('os.environ', env or {'HOME': self.home}):
    <FILL_ME>
    return SqliteAccountInfo(
        file_name=self.db_path if not env else None,
        last_upgrade_to_run=last_upgrade_to_run,
    )
```

#### Output Representation: **OR2** (Recommended)
The fixed code snippet(s) only - what should replace `<FILL_ME>`:

**Example:**
```python
    with mock.patch('os.environ', env or {'HOME': self.test_home}):
```

## Detailed Format Specifications

### Python Function Requirements
- **Single function only** - Must contain exactly one function definition
- **Signature integrity** - Function name, parameters, and return type must NOT change between buggy and fixed versions
- **Intraprocedural** - All changes must be within the function body (not in signature or external code)
- **No docstring-only changes** - Must have actual code changes, not just docstring modifications

### Implementation Details (from filter_and_create_representation.py)

```python
def generate_ir4_or2(buggy_func, fixed_func):
    """Generate IR4 and OR2 representations"""
    b_lines = buggy_func.splitlines(keepends=True)
    f_lines = fixed_func.splitlines(keepends=True)
    
    # Find first and last differing lines
    matcher = difflib.SequenceMatcher(None, b_lines, f_lines)
    opcodes = matcher.get_opcodes()
    
    first_b, last_b, first_f, last_f = -1, -1, -1, -1
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            if first_b == -1:
                first_b, first_f = i1, j1
            last_b, last_f = i2, j2
    
    # Extract parts
    prefix = "".join(b_lines[:first_b])
    suffix = "".join(b_lines[last_b:])
    buggy_chunk = b_lines[first_b:last_b]
    fixed_chunk = "".join(f_lines[first_f:last_f])
    
    # Create IR4: prefix + commented buggy + <FILL_ME> + suffix
    commented_buggy = "".join([f"# {line}" for line in buggy_chunk])
    if commented_buggy and not commented_buggy.endswith('\n'):
        commented_buggy += '\n'
    
    ir4 = prefix + commented_buggy + "<FILL_ME>\n" + suffix
    
    # OR2 is just the fixed chunk
    return ir4, fixed_chunk
```

## Storage Formats

### Option 1: CSV Format (Intermediate)
Columns needed:
- `buggy_function` - Full buggy function code
- `fixed_function` - Full fixed function code
- `IR4` - Computed input representation
- `OR2` - Computed output representation
- Metadata: `repo`, `file_path`, `commit_sha` (optional)

**Example:**
```csv
buggy_function,fixed_function,IR4,OR2,repo,file_path,commit_sha
"def foo():\n  x=1\n  return x","def foo():\n  x=2\n  return x","def foo():\n# x=1\n<FILL_ME>\n  return x"," x=2","myrepo","main.py","abc123"
```

### Option 2: HuggingFace Datasets Format (Final/Production)

**Structure:** PyArrow Table or JSON Lines format

**For Training, create records like:**
```json
{
  "input": "def foo():\n# x=1\n<FILL_ME>\n  return x",
  "output": " x=2"
}
```

**Train-test split recommended:** 80-20 or 70-15-15 (train-val-test)

## Loading in PEFT-LoRA Training

From `llama_sft.py`, the training script expects:

```python
from datasets import load_dataset
from transformers import Trainer

# Load from HuggingFace
ds = load_dataset("ASSERT-KTH/repairllama-datasets", "ir4xor2-python")

# Or load from local parquet/arrow
ds = load_dataset("parquet", data_files="train.parquet")

train_dataset = ds["train"]
eval_dataset = ds["test"]

# Tokenization expects 'input' and 'output' fields
def generate_and_tokenize_prompt(sample, tokenizer, training_args):
    input_text = sample['input']
    target = sample['output']
    full_text = input_text + target + tokenizer.eos_token
    
    # Tokenize full text
    tokenized_full_text = tokenize(full_text, tokenizer, training_args)
    
    # Tokenize input only to mask it
    tokenized_input_text = tokenize(input_text, tokenizer, training_args)
    input_len = len(tokenized_input_text["input_ids"])
    
    # Mask input from labels (only train on output)
    tokenized_full_text["labels"] = [-100] * input_len + tokenized_full_text["labels"][input_len:]
    
    return tokenized_full_text
```

## Token Length Constraints

- **Maximum combined length:** 4096 tokens (for CodeLLaMA)
- **Filter rule:** Remove samples where `len(tokenizer.encode(IR4 + OR2)) > 4096`

## Validation Checklist

Before training, ensure:
- ✅ Each sample has `input` and `output` fields
- ✅ Python code is syntactically valid (parse with `ast.parse()`)
- ✅ Function signature is identical in buggy and fixed versions
- ✅ Actual code changes exist (not just docstrings/formatting)
- ✅ Single function per sample (intraprocedural)
- ✅ Token count within limits
- ✅ Train/eval split is created
- ✅ Data is in HuggingFace Datasets format (JSON/Parquet)

## Example Complete Dataset Row

```json
{
  "input": "def _validate_source(source, req):\n    if source:\n        pieces = urlparse.urlparse(source)\n        schemes = [scheme for scheme in store.get_known_schemes() if scheme != 'file']\n        # Additional scheme check missing\n        <FILL_ME>\n        for scheme in schemes:\n            if pieces.scheme == scheme:\n                return source\n        msg = (\"External sourcing not supported for \" \"store '%s'\" % pieces.scheme)\n        LOG.debug(msg)\n        raise HTTPBadRequest(explanation=msg, request=req, content_type=\"text/plain\")",
  "output": "and scheme != 'swift+config'",
  "repo": "openstack/keystone",
  "file_path": "keystone/common/wsgi.py",
  "commit_sha": "abc123def456"
}
```

## Creation Pipeline

1. **Extract** buggy and fixed functions from raw code diffs
2. **Validate** using AST parsing (function signature, single-function, actual changes)
3. **Compute** IR4 and OR2 from buggy+fixed pair
4. **Filter** by token length
5. **Deduplicate** and remove comments-only changes
6. **Save** to CSV for inspection (optional)
7. **Convert** to HuggingFace Datasets format (JSON/Parquet)
8. **Upload** to HuggingFace Hub or keep locally
9. **Load** with `load_dataset()` during training

## References
- Training script: `repairllama/src/lora/llama_sft.py`
- Data generation: `preprocessing_scripts/filter_and_create_representation.py`
- Visualization: `app.py` (Streamlit viewer)
- Example datasets: `repairllama/benchmarks/gitbugjava/*.jsonl`
