import ast
import difflib
import logging
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# License stripping
# ----------------------------------------------------------------------
def strip_license(code: str) -> str:
    """
    Remove leading comment lines (starting with '#') and empty lines.
    Keeps everything after the first non‑comment, non‑empty line.
    """
    lines = code.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        else:
            start = i
            break
    else:
        # Entire file consisted of comments/empty lines? Return empty string.
        return ""
    return "\n".join(lines[start:])


# ----------------------------------------------------------------------
# Function extraction using AST
# ----------------------------------------------------------------------
class FunctionExtractor(ast.NodeVisitor):
    """
    Collects all top‑level functions and methods (not nested inside other functions).
    Stores them with a unique qualname: "class_name.method" for methods, "func_name" for top‑level.
    Also stores the source code segment and (start, end) line numbers.
    """

    def __init__(self, source: str):
        self.source = source
        self.context: List[str] = []          # stack of class names
        self.functions: Dict[str, Tuple[ast.FunctionDef, str, Tuple[int, int]]] = {}
        self._in_function = False              # whether we are inside a function (skip nested)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef) -> None:
        # Only record if not inside another function (i.e., at top‑level or inside a class)
        if not self._in_function:
            self._in_function = True
            qualname = ".".join(self.context + [node.name])
            # Get source segment
            func_source = ast.get_source_segment(self.source, node)
            if func_source is None:
                # Fallback: could reconstruct, but for safety we skip
                logger.warning(f"Could not extract source for function {qualname}")
            else:
                line_range = (node.lineno, node.end_lineno)
                self.functions[qualname] = (node, func_source, line_range)
            self.generic_visit(node)
            self._in_function = False
        else:
            # Nested function – skip
            self.generic_visit(node)


def extract_functions(source: str) -> Dict[str, Tuple[str, Tuple[int, int]]]:
    """
    Parse source and return a dictionary mapping qualname -> (source_code, (start_line, end_line)).
    Returns empty dict on syntax error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.error(f"Syntax error while parsing: {e}")
        return {}
    extractor = FunctionExtractor(source)
    extractor.visit(tree)
    return {name: (src, rng) for name, (_, src, rng) in extractor.functions.items()}


# ----------------------------------------------------------------------
# Global change detection using line diff
# ----------------------------------------------------------------------
def has_global_changes(
    buggy_lines: List[str],
    fixed_lines: List[str],
    buggy_func_ranges: Dict[str, Tuple[int, int]],
    fixed_func_ranges: Dict[str, Tuple[int, int]],
) -> bool:
    """
    Returns True if there are line differences that lie completely outside any function.
    """
    sm = difflib.SequenceMatcher(None, buggy_lines, fixed_lines)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # Changes in buggy version (deleted or replaced lines)
        if tag in ("delete", "replace"):
            for line_num in range(i1 + 1, i2 + 1):  # 1‑based
                if not any(start <= line_num <= end for start, end in buggy_func_ranges.values()):
                    return True
        # Changes in fixed version (inserted or replaced lines)
        if tag in ("insert", "replace"):
            for line_num in range(j1 + 1, j2 + 1):
                if not any(start <= line_num <= end for start, end in fixed_func_ranges.values()):
                    return True
    return False


# ----------------------------------------------------------------------
# Main processing script
# ----------------------------------------------------------------------
def main(input_csv: str, output_csv: str) -> None:
    logger.info(f"Reading {input_csv}")
    df = pd.read_csv(input_csv)

    # Drop columns if they exist
    drop_cols = ["input_representation", "output_representation"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        logger.info(f"Dropped columns: {existing_drop}")

    # Strip license from original buggy/fixed code and overwrite
    logger.info("Stripping license headers...")
    tqdm.pandas(desc="Stripping license")
    df["buggy_code"] = df["buggy_code"].progress_apply(
        lambda x: strip_license(x) if pd.notna(x) else x
    )
    df["fixed_code"] = df["fixed_code"].progress_apply(
        lambda x: strip_license(x) if pd.notna(x) else x
    )

    # Prepare new columns
    df["type"] = None
    df["buggy_function"] = None
    df["fixed_function"] = None

    # Keep track of rows to keep (only single‑function)
    keep_mask = [False] * len(df)

    # Process rows with tqdm
    logger.info("Classifying samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        buggy = row["buggy_code"]
        fixed = row["fixed_code"]

        if pd.isna(buggy) or pd.isna(fixed):
            logger.warning(f"Row {idx} has missing code, skipping")
            continue

        try:
            # Extract functions
            buggy_funcs = extract_functions(buggy)
            fixed_funcs = extract_functions(fixed)

            if not buggy_funcs or not fixed_funcs:
                # Parsing error – treat as invalid
                continue

            buggy_sources = {name: src for name, (src, _) in buggy_funcs.items()}
            buggy_ranges = {name: rng for name, (_, rng) in buggy_funcs.items()}
            fixed_sources = {name: src for name, (src, _) in fixed_funcs.items()}
            fixed_ranges = {name: rng for name, (_, rng) in fixed_funcs.items()}

            # Identify changed, added, removed functions
            common_names = set(buggy_sources) & set(fixed_sources)
            changed = [name for name in common_names if buggy_sources[name] != fixed_sources[name]]
            added = set(fixed_sources) - set(buggy_sources)
            removed = set(buggy_sources) - set(fixed_sources)

            # Global changes outside functions
            buggy_lines = buggy.splitlines()
            fixed_lines = fixed.splitlines()
            global_change = has_global_changes(buggy_lines, fixed_lines, buggy_ranges, fixed_ranges)

            # Criteria for single‑function bug:
            #   exactly one changed function,
            #   no added functions,
            #   no removed functions,
            #   no global changes.
            if len(changed) == 1 and not added and not removed and not global_change:
                keep_mask[idx] = True
                key = changed[0]
                df.at[idx, "type"] = "single function"
                df.at[idx, "buggy_function"] = buggy_sources[key]
                df.at[idx, "fixed_function"] = fixed_sources[key]

        except Exception as e:
            logger.error(f"Row {idx} failed with exception: {e}")
            continue

    # Apply filter
    filtered_df = df[keep_mask].copy()
    if "commit_sha" not in filtered_df.columns:
        filtered_df["commit_sha"] = pd.Series("", index=filtered_df.index, dtype="string")

    if "filepath" not in filtered_df.columns:
        filtered_df["file_path"] = pd.Series("", index=filtered_df.index, dtype="string")

    
    logger.info(f"Kept {len(filtered_df)} rows out of {len(df)}")

    # Save
    filtered_df.to_csv(output_csv, index=False)
    logger.info(f"Saved filtered dataset to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])