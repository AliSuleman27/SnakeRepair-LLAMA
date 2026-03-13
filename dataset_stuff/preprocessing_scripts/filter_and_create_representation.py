import pandas as pd
import ast
import difflib
from transformers import AutoTokenizer
from tqdm import tqdm

# Initialize CodeLlama tokenizer (ensure you have internet access or it's cached)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
MAX_TOKENS = 4096

def generate_ir4_or2(buggy_func, fixed_func):
    """Generates the IR4 and OR2 strings from the function pair."""
    if not isinstance(buggy_func, str) or not isinstance(fixed_func, str):
        return None, None
        
    b_lines = buggy_func.splitlines(keepends=True)
    f_lines = fixed_func.splitlines(keepends=True)
    
    matcher = difflib.SequenceMatcher(None, b_lines, f_lines)
    opcodes = matcher.get_opcodes()
    
    first_b, last_b, first_f, last_f = -1, -1, -1, -1
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            if first_b == -1:
                first_b, first_f = i1, j1
            last_b, last_f = i2, j2
            
    if first_b == -1: return None, None
        
    prefix = "".join(b_lines[:first_b])
    suffix = "".join(b_lines[last_b:])
    
    buggy_chunk = b_lines[first_b:last_b]
    fixed_chunk = "".join(f_lines[first_f:last_f])
    
    commented_buggy = "".join([f"# {line}" for line in buggy_chunk])
    if commented_buggy and not commented_buggy.endswith('\n'):
        commented_buggy += '\n'
        
    ir4 = prefix + commented_buggy + "<FILL_ME>\n" + suffix
    return ir4, fixed_chunk

def is_valid_sample(buggy_code, fixed_code):
    """Filters out signature changes, non-functions, and docstring-only changes using AST."""
    try:
        b_ast = ast.parse(buggy_code)
        f_ast = ast.parse(fixed_code)
        
        # 1. Enforce strictly intraprocedural (single function)
        if len(b_ast.body) != 1 or not isinstance(b_ast.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if len(f_ast.body) != 1 or not isinstance(f_ast.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
            
        b_func, f_func = b_ast.body[0], f_ast.body[0]
        
        # 2. Reject ANY signature changes
        b_sig = ast.dump(b_func.args) + str(ast.dump(b_func.returns) if b_func.returns else "") + b_func.name
        f_sig = ast.dump(f_func.args) + str(ast.dump(f_func.returns) if f_func.returns else "") + f_func.name
        if b_sig != f_sig:
            return False
            
        # 3. Reject if ONLY docstring changed
        b_doc, f_doc = ast.get_docstring(b_func), ast.get_docstring(f_func)
        
        def get_body_without_docstring(func, has_doc):
            if has_doc and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant):
                return func.body[1:]
            return func.body
            
        b_body = get_body_without_docstring(b_func, b_doc)
        f_body = get_body_without_docstring(f_func, f_doc)
        
        b_body_ast = [ast.dump(stmt) for stmt in b_body]
        f_body_ast = [ast.dump(stmt) for stmt in f_body]
        
        if b_body_ast == f_body_ast:
            return False # Only docstring/formatting changed
            
        return True
    except SyntaxError:
        return False # Drop unparsable syntax

# Load dataset
print("Loading dataset...")
df = pd.read_csv("/home/mrafi/codellms-fyp/SnakeRepair-LLAMA/only_rbr.csv")
valid_indices = []

print("Applying strict AST and Token filters...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    buggy = str(row.get('buggy_function', ''))
    fixed = str(row.get('fixed_function', ''))
    
    if is_valid_sample(buggy, fixed):
        ir4, or2 = generate_ir4_or2(buggy, fixed)
        if ir4 and or2:
            # 4. Token limit filter
            if len(tokenizer.encode(ir4 + or2)) <= MAX_TOKENS:
                valid_indices.append(idx)

# Compile filtered dataset
filtered_df = df.iloc[valid_indices].copy()
filtered_df['IR4'] = filtered_df.apply(lambda row: generate_ir4_or2(row['buggy_function'], row['fixed_function'])[0], axis=1)
filtered_df['OR2'] = filtered_df.apply(lambda row: generate_ir4_or2(row['buggy_function'], row['fixed_function'])[1], axis=1)

# Save final artifacts
training_data = filtered_df[['repo', 'file_path', 'commit_sha', 'IR4', 'OR2']]
training_data.to_csv("rbr_ir_or.csv", index=False)
filtered_df.to_csv("rbr_dataset_actual.csv",index=False)
print(f"Dataset reduced from {len(df)} to {len(training_data)} pristine samples.")