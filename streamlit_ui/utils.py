"""
Utility functions for Streamlit demo - wraps MULLER API calls with error handling.
"""
import muller
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import time
import plotly.graph_objects as go


def create_dataset(name: str, root: str, overwrite: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """
    Create a new MULLER dataset.

    Args:
        name: Dataset name
        root: Root directory path
        overwrite: Whether to overwrite existing dataset

    Returns:
        (dataset_object, error_message)
    """
    try:
        ds_path = Path(root) / name
        ds = muller.dataset(path=str(ds_path), overwrite=overwrite)
        return ds, None
    except Exception as e:
        return None, f"Failed to create dataset: {str(e)}"


def create_tensors(ds: Any, schema: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Create tensors (columns) in the dataset.

    Args:
        ds: Dataset object
        schema: Dict mapping tensor_name -> {htype, dtype, sample_compression}

    Returns:
        error_message or None
    """
    try:
        for tensor_name, config in schema.items():
            ds.create_tensor(
                tensor_name,
                htype=config.get('htype', 'generic'),
                dtype=config.get('dtype'),
                sample_compression=config.get('sample_compression')
            )
        return None
    except Exception as e:
        return f"Failed to create tensors: {str(e)}"


def add_samples(ds: Any, data: Dict[str, List], auto_commit: bool = True) -> Optional[str]:
    """
    Add samples to dataset.

    Args:
        ds: Dataset object
        data: Dict mapping tensor_name -> list of values
        auto_commit: Whether to commit after adding

    Returns:
        error_message or None
    """
    try:
        with ds:
            for tensor_name, values in data.items():
                if hasattr(ds, tensor_name):
                    getattr(ds, tensor_name).extend(values)
                else:
                    return f"Tensor '{tensor_name}' does not exist in dataset"

        if auto_commit:
            ds.commit(message="Add samples via Streamlit UI")

        return None
    except Exception as e:
        return f"Failed to add samples: {str(e)}"


def update_sample(ds: Any, tensor_name: str, index: int, value: Any) -> Optional[str]:
    """
    Update a single sample value.

    Args:
        ds: Dataset object
        tensor_name: Name of the tensor
        index: Sample index
        value: New value

    Returns:
        error_message or None
    """
    try:
        getattr(ds, tensor_name)[index] = value
        return None
    except Exception as e:
        return f"Failed to update sample: {str(e)}"


def delete_sample(ds: Any, index: int) -> Optional[str]:
    """
    Delete a sample by index.

    Args:
        ds: Dataset object
        index: Sample index to delete

    Returns:
        error_message or None
    """
    try:
        ds.pop(index)
        return None
    except Exception as e:
        return f"Failed to delete sample: {str(e)}"


def run_query(ds: Any, conditions: List[Tuple[str, str, Any]],
              logic_ops: Optional[List[str]] = None,
              offset: int = 0, limit: Optional[int] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Run a filter query on the dataset.

    Args:
        ds: Dataset object
        conditions: List of (field, operator, value) tuples
        logic_ops: List of logic operators ("AND", "OR") between conditions
        offset: Result offset
        limit: Max number of results

    Returns:
        (filtered_dataset, error_message)
    """
    try:
        result = ds.filter_vectorized(
            conditions,
            logic_ops or [],
            offset=offset,
            limit=limit
        )
        return result, None
    except Exception as e:
        return None, f"Query failed: {str(e)}"


def dataset_to_dataframe(ds: Any, tensor_list: Optional[List[str]] = None,
                         start: int = 0, end: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Convert dataset to pandas DataFrame.

    Args:
        ds: Dataset object
        tensor_list: List of tensor names to include (None = all)
        start: Start index
        end: End index (None = all)

    Returns:
        (dataframe, error_message)
    """
    try:
        if tensor_list is None:
            tensor_list = list(ds.tensors.keys())

        data = {}
        for tensor_name in tensor_list:
            tensor = getattr(ds, tensor_name)
            if end is None:
                data[tensor_name] = tensor.numpy(aslist=True)[start:]
            else:
                data[tensor_name] = tensor.numpy(aslist=True)[start:end]

        df = pd.DataFrame(data, columns=tensor_list)
        return df, None
    except Exception as e:
        return None, f"Failed to convert to DataFrame: {str(e)}"


def branch_ops(ds: Any, action: str, branch_name: Optional[str] = None,
               merge_strategy: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Perform version control operations.

    Args:
        ds: Dataset object
        action: One of "create", "checkout", "merge", "detect_conflict", "list"
        branch_name: Branch name (for create/checkout/merge)
        merge_strategy: Dict with keys like "append_resolution", "pop_resolution", "update_resolution"

    Returns:
        (result_data, error_message)
    """
    try:
        if action == "create":
            ds.checkout(branch_name, create=True)
            return f"Branch '{branch_name}' created and checked out", None

        elif action == "checkout":
            ds.checkout(branch_name)
            return f"Switched to branch '{branch_name}'", None

        elif action == "merge":
            strategy = merge_strategy or {}
            ds.merge(
                branch_name,
                append_resolution=strategy.get("append_resolution", "ours"),
                pop_resolution=strategy.get("pop_resolution", "ours"),
                update_resolution=strategy.get("update_resolution", "ours")
            )
            return f"Merged '{branch_name}' into current branch", None

        elif action == "detect_conflict":
            conflict_cols, conflict_records = ds.detect_merge_conflict(branch_name, show_value=True)
            return {"columns": conflict_cols, "records": conflict_records}, None

        elif action == "list":
            return ds.branches, None

        else:
            return None, f"Unknown action: {action}"

    except Exception as e:
        return None, f"Branch operation failed: {str(e)}"


def benchmark_parquet_vs_muller(ds: Any, query_conditions: List[Tuple[str, str, Any]],
                                 parquet_path: Optional[str] = None) -> Tuple[Optional[go.Figure], Optional[str]]:
    """
    Benchmark query performance: Parquet vs MULLER.

    Args:
        ds: MULLER dataset
        query_conditions: Query to run
        parquet_path: Path to Parquet file (if None, will export from ds)

    Returns:
        (plotly_figure, error_message)
    """
    try:
        # Export to Parquet if not provided
        if parquet_path is None:
            parquet_path = str(Path(ds.path).parent / "benchmark_temp.parquet")
            ds.write_to_parquet(parquet_path)

        # Benchmark MULLER query
        start = time.time()
        _ = ds.filter_vectorized(query_conditions)
        muller_time = time.time() - start

        # Benchmark Parquet query (using pandas)
        import pyarrow.parquet as pq
        start = time.time()
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        # Simple filter simulation
        for field, op, value in query_conditions:
            if op == "==":
                df = df[df[field] == value]
            elif op == ">":
                df = df[df[field] > value]
            elif op == "<":
                df = df[df[field] < value]
        parquet_time = time.time() - start

        # Get file sizes
        import os
        muller_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(ds.path)
            for f in files
        ) / (1024 * 1024)  # MB

        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB

        # Create Plotly chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Query Time (s)',
            x=['MULLER', 'Parquet'],
            y=[muller_time, parquet_time],
            yaxis='y',
            marker_color=['#636EFA', '#EF553B']
        ))

        fig.add_trace(go.Bar(
            name='Storage Size (MB)',
            x=['MULLER', 'Parquet'],
            y=[muller_size, parquet_size],
            yaxis='y2',
            marker_color=['#00CC96', '#AB63FA']
        ))

        fig.update_layout(
            title='MULLER vs Parquet: Query Performance & Storage',
            xaxis=dict(title='Format'),
            yaxis=dict(title='Query Time (seconds)', side='left'),
            yaxis2=dict(title='Storage Size (MB)', overlaying='y', side='right'),
            barmode='group',
            height=400
        )

        return fig, None

    except Exception as e:
        return None, f"Benchmark failed: {str(e)}"


def load_dataset(path: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load an existing MULLER dataset.

    Args:
        path: Dataset path

    Returns:
        (dataset_object, error_message)
    """
    try:
        ds = muller.load(path)
        return ds, None
    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"


def commit_dataset(ds: Any, message: str = "Commit from Streamlit UI") -> Optional[str]:
    """
    Commit changes to dataset.

    Args:
        ds: Dataset object
        message: Commit message

    Returns:
        error_message or None
    """
    try:
        ds.commit(message=message)
        return None
    except Exception as e:
        return f"Commit failed: {str(e)}"

