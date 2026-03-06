---
name: muller-advanced-query
description: Advanced query operations for MULLER datasets - indexing, vector search, aggregation, and complex filtering. Use when user wants to create indexes, perform vector similarity search, or run aggregation queries.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Advanced Query

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing script:
- Use `scripts/advanced_query.py` for all advanced query operations

Execute the script directly with `python3` command. Never write new scripts to the project root.

## When to Use This Skill

Use this skill when the user wants to:
- Create inverted indexes for text search
- Create vector indexes for similarity search
- Perform vector similarity search
- Run aggregation queries (GROUP BY, COUNT, AVG, etc.)
- Complex filtering with multiple conditions
- Full-text search with CONTAINS operator

## Available Script

### scripts/advanced_query.py

Handles advanced query and indexing operations.

**Operations:**
- `create-index` - Create inverted index for text search
- `create-vector-index` - Create vector index (FLAT, HNSW, DISKANN)
- `load-vector-index` - Load vector index into memory
- `vector-search` - Perform vector similarity search
- `aggregate` - Run aggregation queries
- `filter-advanced` - Complex filtering with multiple conditions

**Usage:**
```bash
# Create inverted index for text search
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-index \
  --path ./my_dataset --tensors "description,title"

# Create vector index
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-vector-index \
  --path ./my_dataset --tensor embeddings --index-name hnsw \
  --index-type HNSWFLAT --metric l2

# Vector search
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py vector-search \
  --path ./my_dataset --tensor embeddings --index-name hnsw \
  --query-file query.npy --topk 10

# Aggregation
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py aggregate \
  --path ./my_dataset --group-by categories --select labels,categories \
  --aggregate-tensors "*"
```

## Common Workflows

### Full-Text Search

```bash
# 1. Create inverted index
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-index \
  --path ./text_dataset --tensors "description"

# 2. Search with CONTAINS (use muller-dataset skill)
python3 .claude/skills/muller-dataset/scripts/data_operations.py query \
  --path ./text_dataset --filter "description CONTAINS 'machine learning'"
```

### Vector Similarity Search

```bash
# 1. Create vector index
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py create-vector-index \
  --path ./embeddings_dataset --tensor embeddings --index-name flat \
  --index-type FLAT --metric l2

# 2. Load index
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py load-vector-index \
  --path ./embeddings_dataset --tensor embeddings --index-name flat

# 3. Search
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py vector-search \
  --path ./embeddings_dataset --tensor embeddings --index-name flat \
  --query-file query_vectors.npy --topk 5
```

### Data Aggregation

```bash
# Group by category and count
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py aggregate \
  --path ./my_dataset --group-by categories \
  --select categories --aggregate-tensors "*"

# Complex aggregation with ordering
python3 .claude/skills/muller-advanced-query/scripts/advanced_query.py aggregate \
  --path ./my_dataset --group-by categories \
  --select labels,categories --order-by labels \
  --aggregate-tensors "labels:AVG,labels:COUNT"
```

## Index Types

### Inverted Index (Text Search)
- Used for: Full-text search with CONTAINS operator
- Best for: Text fields, descriptions, titles
- Create before: Using CONTAINS in queries

### Vector Indexes

**FLAT Index:**
- Exact nearest neighbor search
- Best for: Small datasets (<100K vectors)
- Highest accuracy, slower search

**HNSW Index:**
- Approximate nearest neighbor search
- Best for: Medium to large datasets
- Good balance of speed and accuracy
- Parameters: `ef_construction`, `m`

**DISKANN Index:**
- Disk-based approximate search
- Best for: Very large datasets (millions of vectors)
- Memory efficient

## Aggregation Functions

Supported aggregate functions:
- `COUNT` - Count records
- `AVG` - Average value
- `MIN` - Minimum value
- `MAX` - Maximum value
- `SUM` - Sum of values

## Reference Documentation

- [Indexing Guide](references/indexing-guide.md) - Index types and usage
- [Vector Search Guide](references/vector-search-guide.md) - Vector similarity search
- [Aggregation Guide](references/aggregation-guide.md) - SQL-like aggregations
- [Full Documentation](../../docs/api/dataset-query.md) - Complete API reference

## Notes

- Always commit before creating indexes
- Vector indexes require loading before search
- HNSW parameters affect speed/accuracy tradeoff
- Aggregation supports SQL-like GROUP BY and ORDER BY
- Use inverted index for text search performance
