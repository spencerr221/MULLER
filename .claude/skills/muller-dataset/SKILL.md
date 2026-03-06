---
name: muller-dataset
description: Create, query, and manage MULLER datasets (multimodal data lake with version control). Use when user wants to work with datasets, create tensors, append data, query samples, or inspect dataset information.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Dataset Management

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing scripts provided in this skill:
- Use `scripts/dataset_manager.py` for dataset and tensor management
- Use `scripts/data_operations.py` for data CRUD operations

Execute these scripts directly with `python3` command. Never write new scripts to the project root.

## When to Use This Skill

Use this skill when the user wants to:
- Create or load MULLER datasets
- Add data to datasets (images, text, labels, vectors, etc.)
- Query or filter dataset samples
- Manage dataset structure (create/delete/rename tensors)
- Inspect dataset information (summary, statistics)
- Import data from files or directories

## Available Scripts

### scripts/dataset_manager.py

Manages dataset lifecycle and structure.

**Operations:**
- `create` - Create a new dataset
- `load` - Load existing dataset info
- `delete` - Delete a dataset
- `info` - Get dataset information
- `stats` - Get dataset statistics
- `create-tensor` - Create a new tensor
- `delete-tensor` - Delete a tensor
- `rename-tensor` - Rename a tensor

**Usage:**
```bash
# Create dataset
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create --path ./my_dataset

# Create with tensors
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create --path ./my_dataset \
  --tensors "images:image:jpg,labels:class_label:uint32"

# Get info
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py info --path ./my_dataset

# Create tensor
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create-tensor --path ./my_dataset \
  --name embeddings --htype vector --dtype float32
```

### scripts/data_operations.py

Handles data CRUD operations.

**Operations:**
- `append` - Add single sample
- `extend` - Add multiple samples
- `update` - Update existing sample
- `delete` - Delete samples
- `query` - Query and filter samples
- `import` - Import data from files

**Usage:**
```bash
# Append sample
python3 .claude/skills/muller-dataset/scripts/data_operations.py append --path ./my_dataset \
  --data '{"images": "path/to/img.jpg", "labels": 1}'

# Query samples
python3 .claude/skills/muller-dataset/scripts/data_operations.py query --path ./my_dataset \
  --filter "labels > 5" --limit 10

# Import from file
python3 .claude/skills/muller-dataset/scripts/data_operations.py import --path ./my_dataset \
  --source data.jsonl
```

## Common Workflows

### Create Image Classification Dataset

```bash
# 1. Create dataset with image and label tensors
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create --path ./image_dataset \
  --tensors "images:image:jpg,labels:class_label:uint32"

# 2. Import images from directory
python3 .claude/skills/muller-dataset/scripts/data_operations.py import --path ./image_dataset \
  --source ./photos/ --tensor images --pattern "*.jpg"

# 3. Query specific samples
python3 .claude/skills/muller-dataset/scripts/data_operations.py query --path ./image_dataset \
  --filter "labels == 5"
```

### Create Text Dataset

```bash
# 1. Create dataset
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py create --path ./text_dataset \
  --tensors "text:text,embeddings:vector:float32"

# 2. Import from JSONL file
python3 .claude/skills/muller-dataset/scripts/data_operations.py import --path ./text_dataset \
  --source data.jsonl
```

### Inspect Dataset

```bash
# Get summary
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py info --path ./my_dataset

# Get statistics
python3 .claude/skills/muller-dataset/scripts/dataset_manager.py stats --path ./my_dataset

# Query samples
python3 .claude/skills/muller-dataset/scripts/data_operations.py query --path ./my_dataset --limit 5
```

## Data Types (htypes)

MULLER supports 12+ data types:

| htype | Use Case | Compression |
|-------|----------|-------------|
| `image` | Images | jpg, png, webp, etc. |
| `video` | Videos | mp4, mkv, avi |
| `audio` | Audio files | mp3, wav, flac |
| `text` | Text/strings | None, lz4 |
| `vector` | Embeddings | None, lz4 |
| `class_label` | Labels | None, lz4 |
| `bbox` | Bounding boxes | None, lz4 |
| `json` | JSON objects | None, lz4 |
| `list` | Lists | None, lz4 |
| `generic` | Generic data | None, lz4 |

See [references/htypes-guide.md](references/htypes-guide.md) for details.

## Reference Documentation

- [Quick Start Guide](references/quick-start.md) - 5-minute tutorial
- [API Cheatsheet](references/api-cheatsheet.md) - Common operations
- [Data Types Guide](references/htypes-guide.md) - Supported htypes
- [Full Documentation](../../docs/) - Complete API reference

## Error Handling

Scripts output JSON with clear error messages:

```json
{
  "success": false,
  "error": "DatasetNotExistsError",
  "message": "Dataset not found at ./my_dataset",
  "suggestion": "Create dataset first: dataset_manager.py create --path ./my_dataset"
}
```

## Notes

- Always use `with ds:` context in Python for better write performance
- Use compression for images/video/audio (jpg, mp4, mp3)
- Avoid compression for text/labels unless storage is critical
- Query operations support: `==`, `!=`, `>`, `<`, `>=`, `<=`, `CONTAINS`
