---
name: muller-export
description: Export and integrate MULLER datasets with other formats - Arrow, Parquet, JSON, NumPy, MindRecord. Use when user wants to export data, convert formats, or integrate with other frameworks.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Export & Integration

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing script:
- Use `scripts/export.py` for all export and integration operations

Execute the script directly with `python3` command. Never write new scripts to the project root.

## When to Use This Skill

Use this skill when the user wants to:
- Export datasets to Arrow format
- Export datasets to Parquet files
- Export datasets to JSON format
- Convert datasets to NumPy arrays
- Export to MindRecord format (for MindSpore)
- Integrate with PyTorch, TensorFlow, or other frameworks

## Available Script

### scripts/export.py

Handles export and format conversion operations.

**Operations:**
- `to-arrow` - Export to Apache Arrow format
- `to-parquet` - Export to Parquet files
- `to-json` - Export to JSON format
- `to-numpy` - Convert tensors to NumPy arrays
- `to-mindrecord` - Export to MindRecord format
- `get-info` - Get export information

**Usage:**
```bash
# Export to Arrow
python3 .claude/skills/muller-export/scripts/export.py to-arrow \
  --path ./my_dataset --output ./output.arrow

# Export to Parquet
python3 .claude/skills/muller-export/scripts/export.py to-parquet \
  --path ./my_dataset --output ./output_dir

# Export to JSON
python3 .claude/skills/muller-export/scripts/export.py to-json \
  --path ./my_dataset --output ./output.json

# Convert tensor to NumPy
python3 .claude/skills/muller-export/scripts/export.py to-numpy \
  --path ./my_dataset --tensor embeddings --output ./embeddings.npy

# Export to MindRecord
python3 .claude/skills/muller-export/scripts/export.py to-mindrecord \
  --path ./my_dataset --output ./output.mindrecord
```

## Common Workflows

### Export for Data Analysis

```bash
# Export to Parquet for analysis with Pandas/Polars
python3 .claude/skills/muller-export/scripts/export.py to-parquet \
  --path ./my_dataset --output ./analysis_data

# Export to JSON for web applications
python3 .claude/skills/muller-export/scripts/export.py to-json \
  --path ./my_dataset --output ./data.json --limit 1000
```

### Integration with ML Frameworks

```bash
# Export embeddings to NumPy for PyTorch/TensorFlow
python3 .claude/skills/muller-export/scripts/export.py to-numpy \
  --path ./embeddings_dataset --tensor embeddings \
  --output ./embeddings.npy

# Export to MindRecord for MindSpore training
python3 .claude/skills/muller-export/scripts/export.py to-mindrecord \
  --path ./training_dataset --output ./train.mindrecord
```

### Data Sharing

```bash
# Export to Arrow for efficient data sharing
python3 .claude/skills/muller-export/scripts/export.py to-arrow \
  --path ./my_dataset --output ./shared_data.arrow

# Export subset to JSON
python3 .claude/skills/muller-export/scripts/export.py to-json \
  --path ./my_dataset --output ./sample.json \
  --offset 0 --limit 100
```

## Export Formats

### Apache Arrow
- **Use for:** Efficient columnar data exchange
- **Best for:** Large datasets, cross-language compatibility
- **Features:** Zero-copy reads, memory-mapped files

### Parquet
- **Use for:** Long-term storage, data analytics
- **Best for:** Compressed columnar storage
- **Features:** Efficient compression, schema evolution

### JSON
- **Use for:** Web APIs, human-readable exports
- **Best for:** Small to medium datasets
- **Features:** Universal compatibility, easy debugging

### NumPy
- **Use for:** ML/DL frameworks integration
- **Best for:** Tensor data, numerical computations
- **Features:** Direct array access, framework compatibility

### MindRecord
- **Use for:** MindSpore framework training
- **Best for:** Large-scale deep learning
- **Features:** Optimized for MindSpore, efficient I/O

## Export Options

### Filtering
- Use `--offset` and `--limit` to export subsets
- Combine with muller-dataset query for filtered exports

### Tensor Selection
- Export specific tensors with `--tensor` parameter
- Export all tensors by default

### Compression
- Parquet: Automatic compression (snappy, gzip)
- Arrow: Optional compression
- JSON: Optional pretty-printing

## Reference Documentation

- [Export Guide](references/export-guide.md) - Detailed export workflows
- [Format Comparison](references/format-comparison.md) - Choose the right format
- [Integration Examples](references/integration-examples.md) - Framework integration
- [Full Documentation](../../docs/api/dataset-export.md) - Complete API reference

## Notes

- Arrow and Parquet are best for large datasets
- JSON is human-readable but less efficient
- NumPy exports are tensor-specific
- MindRecord requires MindSpore installation
- Use read-only mode for exports (no modifications)
