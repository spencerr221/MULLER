"""
MULLER Streamlit Demo - Interactive Multimodal Data Lake Management

This demo showcases MULLER's capabilities:
- Dataset creation and CRUD operations
- Conditional filtering and vector search
- Git-like version control (branch, merge, conflict resolution)
- Performance benchmarking vs Parquet
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_ui.utils import (
    create_dataset, create_tensors, add_samples, update_sample, delete_sample,
    run_query, dataset_to_dataframe, branch_ops, benchmark_parquet_vs_muller,
    load_dataset, commit_dataset
)
import pandas as pd
import numpy as np
from PIL import Image
import tempfile


# Page config
st.set_page_config(
    page_title="MULLER Demo",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "current_branch" not in st.session_state:
    st.session_state.current_branch = "main"


# Sidebar navigation
st.sidebar.title("🗄️ MULLER Demo")
st.sidebar.markdown("**Multimodal Data Lake with Git-like Versioning**")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dataset Management", "🔍 Query & Search", "🌿 Version Control", "⚡ Benchmarks", "ℹ️ About"]
)

st.sidebar.markdown("---")
if st.session_state.dataset is not None:
    st.sidebar.success(f"✓ Dataset loaded")
    st.sidebar.info(f"Branch: `{st.session_state.current_branch}`")
else:
    st.sidebar.warning("No dataset loaded")


# ============================================================================
# PAGE 1: Dataset Management
# ============================================================================
if page == "📊 Dataset Management":
    st.title("📊 Dataset Management")

    tab1, tab2, tab3 = st.tabs(["Create Dataset", "Add Samples", "View & Edit"])

    # --- Tab 1: Create Dataset ---
    with tab1:
        st.subheader("Create New Dataset")

        col1, col2 = st.columns(2)
        with col1:
            ds_name = st.text_input("Dataset Name", value="demo_dataset")
            ds_root = st.text_input("Root Directory", value=str(Path.home() / "muller_datasets"))

        with col2:
            overwrite = st.checkbox("Overwrite if exists", value=False)

        if st.button("Create Dataset", type="primary"):
            with st.spinner("Creating dataset..."):
                ds, error = create_dataset(ds_name, ds_root, overwrite=overwrite)

                if error:
                    st.error(error)
                else:
                    # Create default schema
                    schema = {
                        "labels": {"htype": "generic", "dtype": "int"},
                        "categories": {"htype": "text"},
                        "description": {"htype": "text"}
                    }
                    error = create_tensors(ds, schema)

                    if error:
                        st.error(error)
                    else:
                        st.session_state.dataset = ds
                        st.session_state.dataset_path = str(Path(ds_root) / ds_name)
                        st.session_state.current_branch = "main"
                        st.success(f"✓ Dataset created at: {st.session_state.dataset_path}")
                        st.rerun()

        st.markdown("---")
        st.subheader("Load Existing Dataset")
        load_path = st.text_input("Dataset Path", value="")
        if st.button("Load Dataset"):
            ds, error = load_dataset(load_path)
            if error:
                st.error(error)
            else:
                st.session_state.dataset = ds
                st.session_state.dataset_path = load_path
                st.success(f"✓ Dataset loaded from: {load_path}")
                st.rerun()

    # --- Tab 2: Add Samples ---
    with tab2:
        st.subheader("Add Samples to Dataset")

        if st.session_state.dataset is None:
            st.warning("Please create or load a dataset first")
        else:
            st.info("Add samples manually or upload CSV")

            # Manual entry
            with st.expander("➕ Add Single Sample"):
                label_val = st.number_input("Label", value=0, step=1)
                category_val = st.text_input("Category", value="")
                desc_val = st.text_area("Description", value="")

                if st.button("Add Sample"):
                    data = {
                        "labels": [label_val],
                        "categories": [category_val],
                        "description": [desc_val]
                    }
                    error = add_samples(st.session_state.dataset, data, auto_commit=True)
                    if error:
                        st.error(error)
                    else:
                        st.success("✓ Sample added and committed")
                        st.rerun()

            # Batch upload
            with st.expander("📤 Upload CSV"):
                uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())

                    if st.button("Import CSV Data"):
                        data = {col: df[col].tolist() for col in df.columns}
                        error = add_samples(st.session_state.dataset, data, auto_commit=True)
                        if error:
                            st.error(error)
                        else:
                            st.success(f"✓ Imported {len(df)} samples")
                            st.rerun()

    # --- Tab 3: View & Edit ---
    with tab3:
        st.subheader("View & Edit Dataset")

        if st.session_state.dataset is None:
            st.warning("Please create or load a dataset first")
        else:
            ds = st.session_state.dataset

            # Display summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(ds))
            with col2:
                st.metric("Tensors", len(ds.tensors))
            with col3:
                st.metric("Current Branch", st.session_state.current_branch)

            # Display data
            df, error = dataset_to_dataframe(ds, start=0, end=100)
            if error:
                st.error(error)
            else:
                st.dataframe(df, use_container_width=True)

                # Delete sample
                with st.expander("🗑️ Delete Sample"):
                    del_idx = st.number_input("Sample Index", min_value=0, max_value=len(ds)-1, value=0)
                    if st.button("Delete"):
                        error = delete_sample(ds, del_idx)
                        if error:
                            st.error(error)
                        else:
                            commit_dataset(ds, message=f"Deleted sample {del_idx}")
                            st.success(f"✓ Deleted sample {del_idx}")
                            st.rerun()


# ============================================================================
# PAGE 2: Query & Search
# ============================================================================
elif page == "🔍 Query & Search":
    st.title("🔍 Query & Search")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first")
    else:
        ds = st.session_state.dataset

        tab1, tab2 = st.tabs(["Conditional Filtering", "Vector Search"])

        # --- Tab 1: Conditional Filtering ---
        with tab1:
            st.subheader("Conditional Filtering")

            col1, col2, col3 = st.columns(3)
            with col1:
                field = st.selectbox("Field", list(ds.tensors.keys()))
            with col2:
                operator = st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "CONTAINS", "LIKE"])
            with col3:
                value = st.text_input("Value")

            if st.button("Run Query", type="primary"):
                # Convert value to appropriate type
                try:
                    if operator in [">", "<", ">=", "<="]:
                        value = float(value)
                    elif operator == "==":
                        try:
                            value = int(value)
                        except:
                            pass  # Keep as string
                except:
                    st.error("Invalid value for operator")
                    st.stop()

                conditions = [(field, operator, value)]
                result_ds, error = run_query(ds, conditions)

                if error:
                    st.error(error)
                else:
                    st.success(f"✓ Found {len(result_ds)} matching samples")
                    df, _ = dataset_to_dataframe(result_ds)
                    st.dataframe(df, use_container_width=True)

        # --- Tab 2: Vector Search ---
        with tab2:
            st.subheader("Vector Similarity Search")
            st.info("Vector search requires embeddings tensor with vector index")
            st.markdown("*Feature coming soon - requires embedding generation*")


# ============================================================================
# PAGE 3: Version Control
# ============================================================================
elif page == "🌿 Version Control":
    st.title("🌿 Version Control")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first")
    else:
        ds = st.session_state.dataset

        tab1, tab2, tab3 = st.tabs(["Branches", "Merge", "Commit Log"])

        # --- Tab 1: Branches ---
        with tab1:
            st.subheader("Branch Management")

            # List branches
            branches, error = branch_ops(ds, "list")
            if error:
                st.error(error)
            else:
                st.write("**Available Branches:**")
                for branch_name, info in branches.items():
                    if branch_name == st.session_state.current_branch:
                        st.success(f"✓ {branch_name} (current)")
                    else:
                        st.info(f"  {branch_name}")

            col1, col2 = st.columns(2)

            # Create branch
            with col1:
                st.markdown("**Create New Branch**")
                new_branch = st.text_input("Branch Name")
                if st.button("Create Branch"):
                    result, error = branch_ops(ds, "create", branch_name=new_branch)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.current_branch = new_branch
                        st.success(result)
                        st.rerun()

            # Checkout branch
            with col2:
                st.markdown("**Switch Branch**")
                target_branch = st.selectbox("Select Branch", list(branches.keys()))
                if st.button("Checkout"):
                    result, error = branch_ops(ds, "checkout", branch_name=target_branch)
                    if error:
                        st.error(error)
                    else:
                        st.session_state.current_branch = target_branch
                        st.success(result)
                        st.rerun()

        # --- Tab 2: Merge ---
        with tab2:
            st.subheader("Merge Branches")

            merge_source = st.selectbox("Merge from", [b for b in branches.keys() if b != st.session_state.current_branch])

            # Detect conflicts
            if st.button("Detect Conflicts"):
                result, error = branch_ops(ds, "detect_conflict", branch_name=merge_source)
                if error:
                    st.error(error)
                else:
                    if result["columns"]:
                        st.warning(f"⚠️ Conflicts detected in: {', '.join(result['columns'])}")

                        # Enhanced conflict visualization
                        with st.expander("📋 View Conflict Details", expanded=True):
                            for col_name in result["columns"]:
                                st.markdown(f"### Conflicts in `{col_name}`")

                                conflict_data = result["records"][col_name]

                                # Append conflicts
                                if conflict_data.get("app_ori_idx"):
                                    st.markdown("**Append Conflicts:**")
                                    st.markdown(f"- Current branch adds {len(conflict_data['app_ori_idx'])} samples")
                                    st.markdown(f"- Source branch adds {len(conflict_data['app_tar_idx'])} samples")

                                    # Show sample values
                                    if conflict_data.get("app_ori_values"):
                                        st.markdown("*Current branch samples:*")
                                        for idx, val in zip(conflict_data["app_ori_idx"][:3], conflict_data["app_ori_values"][:3]):
                                            st.code(f"[{idx}] {val}")

                                    if conflict_data.get("app_tar_values"):
                                        st.markdown("*Source branch samples:*")
                                        for idx, val in zip(conflict_data["app_tar_idx"][:3], conflict_data["app_tar_values"][:3]):
                                            st.code(f"[{idx}] {val}")

                                # Delete conflicts
                                if conflict_data.get("del_ori_idx") or conflict_data.get("del_tar_idx"):
                                    st.markdown("**Delete Conflicts:**")
                                    if conflict_data.get("del_ori_idx"):
                                        st.markdown(f"- Current branch deletes: {conflict_data['del_ori_idx']}")
                                    if conflict_data.get("del_tar_idx"):
                                        st.markdown(f"- Source branch deletes: {conflict_data['del_tar_idx']}")

                                # Update conflicts
                                if conflict_data.get("update_values"):
                                    update_ori = conflict_data["update_values"].get("update_ori", [])
                                    update_tar = conflict_data["update_values"].get("update_tar", [])

                                    if update_ori or update_tar:
                                        st.markdown("**Update Conflicts:**")

                                        # Create comparison table
                                        comparison_data = []
                                        all_indices = set()

                                        for update_dict in update_ori:
                                            all_indices.update(update_dict.keys())
                                        for update_dict in update_tar:
                                            all_indices.update(update_dict.keys())

                                        for idx in sorted(all_indices):
                                            ori_val = None
                                            tar_val = None

                                            for update_dict in update_ori:
                                                if idx in update_dict:
                                                    ori_val = update_dict[idx]

                                            for update_dict in update_tar:
                                                if idx in update_dict:
                                                    tar_val = update_dict[idx]

                                            comparison_data.append({
                                                "Index": idx,
                                                "Current Branch": str(ori_val) if ori_val is not None else "-",
                                                "Source Branch": str(tar_val) if tar_val is not None else "-"
                                            })

                                        if comparison_data:
                                            conflict_df = pd.DataFrame(comparison_data)

                                            # Style the dataframe
                                            def highlight_conflicts(row):
                                                if row["Current Branch"] != "-" and row["Source Branch"] != "-":
                                                    return ['background-color: #ffcccc'] * len(row)
                                                return [''] * len(row)

                                            styled_df = conflict_df.style.apply(highlight_conflicts, axis=1)
                                            st.dataframe(styled_df, use_container_width=True)

                                st.markdown("---")
                    else:
                        st.success("✓ No conflicts detected")

            # Merge strategy
            st.markdown("**Merge Strategy**")
            col1, col2, col3 = st.columns(3)
            with col1:
                append_res = st.radio("Append Resolution", ["ours", "theirs", "both"])
            with col2:
                pop_res = st.radio("Delete Resolution", ["ours", "theirs"])
            with col3:
                update_res = st.radio("Update Resolution", ["ours", "theirs"])

            if st.button("Merge", type="primary"):
                strategy = {
                    "append_resolution": append_res,
                    "pop_resolution": pop_res,
                    "update_resolution": update_res
                }
                result, error = branch_ops(ds, "merge", branch_name=merge_source, merge_strategy=strategy)
                if error:
                    st.error(error)
                else:
                    st.success(result)
                    st.rerun()

        # --- Tab 3: Commit Log ---
        with tab3:
            st.subheader("Commit History")
            if st.button("Show Log"):
                log_output = ds.log()
                st.text(log_output)


# ============================================================================
# PAGE 4: Benchmarks
# ============================================================================
elif page == "⚡ Benchmarks":
    st.title("⚡ Performance Benchmarks")

    if st.session_state.dataset is None:
        st.warning("Please create or load a dataset first")
    else:
        st.subheader("MULLER vs Parquet")
        st.markdown("Compare query performance and storage efficiency")

        ds = st.session_state.dataset

        # Query setup
        st.markdown("**Query Configuration**")
        col1, col2, col3 = st.columns(3)
        with col1:
            field = st.selectbox("Field", list(ds.tensors.keys()))
        with col2:
            operator = st.selectbox("Operator", [">", "<", "=="])
        with col3:
            value = st.number_input("Value", value=0)

        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                conditions = [(field, operator, value)]
                fig, error = benchmark_parquet_vs_muller(ds, conditions)

                if error:
                    st.error(error)
                else:
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("**Key Takeaways:**")
                    st.markdown("- MULLER uses chunk-based storage with lazy loading")
                    st.markdown("- Parquet requires full table scan for non-indexed queries")
                    st.markdown("- MULLER supports Git-like versioning without data duplication")


# ============================================================================
# PAGE 5: About
# ============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About MULLER")

    st.markdown("""
    ## MULLER: Multimodal Data Lake Format

    **MULLER** is a next-generation data lake format designed for collaborative AI workflows.

    ### Key Features

    - **Multimodal Support**: Images, videos, audio, text, vectors, and structured data
    - **Git-like Versioning**: Branch, merge, and track changes with conflict resolution
    - **Lazy Loading**: Efficient memory usage with on-demand data loading
    - **Query Engine**: SQL-like filtering, full-text search, and vector similarity search
    - **Compression**: 20+ formats (LZ4, JPEG, PNG, MP4, etc.)
    - **Cloud Storage**: S3, OBS, Roma, and local filesystem support

    ### Architecture

    - **Chunk Engine**: Variable-sized chunks with three compression strategies
    - **LRU Cache**: Multi-layer caching (memory → local → remote)
    - **Storage Providers**: Pluggable backends for different storage systems
    - **Version Control**: Commit DAG with merge strategies

    ### Use Cases

    - Collaborative data annotation
    - Prompt-driven dataset evolution
    - Multi-user AI training workflows
    - Large-scale multimodal data management

    ### Demo Workflow

    1. **Create Dataset** → Define schema and add samples
    2. **Query & Search** → Filter data with conditions or vector similarity
    3. **Version Control** → Create branches, make changes, merge with conflict resolution
    4. **Benchmark** → Compare performance with traditional formats

    ---

    **SIGMOD 2026 Demo Track Submission**

    *For more information, visit the [MULLER GitHub repository](#)*
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("MULLER Demo | SIGMOD 2026")
