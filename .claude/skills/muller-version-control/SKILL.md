---
name: muller-version-control
description: Git-like version control for MULLER datasets - commit, branch, merge, diff, and conflict resolution. Use when user wants to version datasets, create branches, merge changes, or view history.
compatibility: Requires Python 3.11+, muller package installed
---

# MULLER Version Control

## IMPORTANT: How to Use This Skill

**DO NOT create new Python files.** Always use the existing script:
- Use `scripts/version_control.py` for all version control operations

Execute the script directly with `python3` command. Never write new scripts to the project root.

## When to Use This Skill

Use this skill when the user wants to:
- Commit dataset changes
- Create and switch branches
- Merge branches with conflict resolution
- View commit history and logs
- Compare differences between commits/branches
- Checkout specific commits or branches

## Available Script

### scripts/version_control.py

Manages Git-like version control operations.

**Operations:**
- `commit` - Commit current changes
- `checkout` - Switch or create branches
- `branch` - List all branches
- `merge` - Merge branches with conflict resolution
- `log` - View commit history
- `diff` - Compare commits or branches
- `commits` - List all commits

**Usage:**
```bash
# Commit changes
python3 .claude/skills/muller-version-control/scripts/version_control.py commit \
  --path ./my_dataset --message "Added new samples"

# Create and checkout new branch
python3 .claude/skills/muller-version-control/scripts/version_control.py checkout \
  --path ./my_dataset --branch dev-1 --create

# List branches
python3 .claude/skills/muller-version-control/scripts/version_control.py branch \
  --path ./my_dataset

# Merge branch
python3 .claude/skills/muller-version-control/scripts/version_control.py merge \
  --path ./my_dataset --branch dev-1 --append-resolution both

# View log
python3 .claude/skills/muller-version-control/scripts/version_control.py log \
  --path ./my_dataset

# Compare branches
python3 .claude/skills/muller-version-control/scripts/version_control.py diff \
  --path ./my_dataset --id1 main --id2 dev-1
```

## Common Workflows

### Collaborative Data Annotation

```bash
# 1. Create a new branch for annotation
python3 .claude/skills/muller-version-control/scripts/version_control.py checkout \
  --path ./dataset --branch annotation-v1 --create

# 2. Make changes (use muller-dataset skill)
# ... add/update/delete samples ...

# 3. Commit changes
python3 .claude/skills/muller-version-control/scripts/version_control.py commit \
  --path ./dataset --message "Annotated 100 new samples"

# 4. Switch back to main
python3 .claude/skills/muller-version-control/scripts/version_control.py checkout \
  --path ./dataset --branch main

# 5. Merge annotation branch
python3 .claude/skills/muller-version-control/scripts/version_control.py merge \
  --path ./dataset --branch annotation-v1
```

### Parallel Development

```bash
# Team member 1: Create branch dev-1
python3 .claude/skills/muller-version-control/scripts/version_control.py checkout \
  --path ./dataset@main --branch dev-1 --create

# Team member 2: Create branch dev-2
python3 .claude/skills/muller-version-control/scripts/version_control.py checkout \
  --path ./dataset@main --branch dev-2 --create

# After both commit changes, merge with conflict resolution
python3 .claude/skills/muller-version-control/scripts/version_control.py merge \
  --path ./dataset@main --branch dev-1 --append-resolution both

python3 .claude/skills/muller-version-control/scripts/version_control.py merge \
  --path ./dataset@main --branch dev-2 --update-resolution theirs
```

## Merge Conflict Resolution

When merging branches, you can specify resolution strategies:

- `append-resolution`: How to handle new samples
  - `ours` - Keep only our appended samples
  - `theirs` - Keep only their appended samples
  - `both` - Keep both (default)

- `pop-resolution`: How to handle deleted samples
  - `ours` - Use our deletions
  - `theirs` - Use their deletions
  - `manual` - Require manual resolution

- `update-resolution`: How to handle updated samples
  - `ours` - Keep our updates
  - `theirs` - Keep their updates
  - `manual` - Require manual resolution

## Reference Documentation

- [Version Control Guide](references/version-control-guide.md) - Detailed workflows
- [Conflict Resolution](references/conflict-resolution.md) - Handling merge conflicts
- [Full Documentation](../../docs/api/dataset-version-control.md) - Complete API reference

## Notes

- Always commit before switching branches
- Use descriptive commit messages
- Check for conflicts before merging
- Fast-forward merges happen automatically when possible
- Three-way merges handle concurrent modifications
