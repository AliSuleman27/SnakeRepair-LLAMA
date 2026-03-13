import streamlit as st
import pandas as pd
import os

# Configure the page layout
st.set_page_config(layout="wide", page_title="SnakeRepairLLAMA Dataset Explorer")

st.title("🐍 SnakeRepairLLAMA: APR Dataset Explorer")
st.markdown("Use this interface to inspect the input representations (IR4) and output representations (OR2) for Python bug-fix pairs.")

# 1. Accept filepath directly to bypass upload limits
default_path = "filtered_training_data.csv"
file_path = st.text_input("Enter Dataset File Path:", value=default_path)

@st.cache_data
def load_data(path):
    """Caches the dataset in memory to prevent reloading on every interaction."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")
    return pd.read_csv(path)

if file_path:
    try:
        df = load_data(file_path)
        st.success(f"Dataset loaded successfully! Total pristine samples: **{len(df)}**")
        
        all_columns = df.columns.tolist()
        
        # 2. Select 4 columns to inspect
        st.subheader("Select Columns to Inspect")
        
        sel_cols = st.columns(4)
        
        def get_index(col_name, default_idx):
            return all_columns.index(col_name) if col_name in all_columns else default_idx

        with sel_cols[0]:
            col1 = st.selectbox("Top-Left", all_columns, index=get_index('buggy_function', 0))
        with sel_cols[1]:
            col2 = st.selectbox("Top-Right", all_columns, index=get_index('fixed_function', min(1, len(all_columns)-1)))
        with sel_cols[2]:
            col3 = st.selectbox("Bottom-Left", all_columns, index=get_index('IR4', min(2, len(all_columns)-1)))
        with sel_cols[3]:
            col4 = st.selectbox("Bottom-Right", all_columns, index=get_index('OR2', min(3, len(all_columns)-1)))
        
        st.divider()
        
        # 3. View individual samples
        st.subheader("Sample Viewer")
        
        nav_cols = st.columns([1, 3])
        with nav_cols[0]:
            sample_idx = st.number_input("Select Sample Index:", min_value=0, max_value=len(df)-1, value=0, step=1)
        
        row = df.iloc[sample_idx]
        
        if 'repo' in row and 'commit_sha' in row:
            st.caption(f"**Repository:** `{row.get('repo', 'N/A')}` | **Commit:** `{row.get('commit_sha', 'N/A')}`")

        # 4. Display the code in a 2x2 Grid
        
        # --- Row 1 ---
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown(f"#### {col1}")
            st.code(str(row[col1]), language="python")
            
        with row1_col2:
            st.markdown(f"#### {col2}")
            st.code(str(row[col2]), language="python")
            
        st.markdown("<br>", unsafe_allow_html=True) # Small vertical spacer
        
        # --- Row 2 ---
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.markdown(f"#### {col3}")
            st.code(str(row[col3]), language="python")
            
        with row2_col2:
            st.markdown(f"#### {col4}")
            st.code(str(row[col4]), language="python")

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred while rendering the data: {e}")