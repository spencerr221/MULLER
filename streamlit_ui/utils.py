# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Utility functions for Streamlit demo - wraps MULLER API calls with error handling.
"""
import muller
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import time
import os
import plotly.graph_objects as go


def create_dataset(name: str, root: str, overwrite: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """Create a new MULLER dataset."""
    try:
        ds_path = Path(root) / name
        ds = muller.dataset(path=str(ds_path), overwrite=overwrite)
        return ds, None
    except Exception as e:
        return None, f"Failed to create dataset: {e}"


def create_tensors(ds: Any, schema: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Create tensors (columns) in the dataset.

    schema: {tensor_name: {htype, dtype, sample_compression}}
    """
    try:
        for tensor_name, config in schema.items():
            kwargs = {"htype": config.get("htype", "generic")}
            if config.get("dtype"):
                kwargs["dtype"] = config["dtype"]
            if config.get("sample_compression"):
                kwargs["sample_compression"] = config["sample_compression"]
            ds.create_tensor(tensor_name, **kwargs)
        return None
    except Exception as e:
        return f"Failed to create tensors: {e}"


def add_samples(ds: Any, data: Dict[str, List], auto_commit: bool = True) -> Optional[str]:
    """Add samples to dataset using per-tensor extend."""
    try:
        with ds:
            for tensor_name, values in data.items():
                ds[tensor_name].extend(values)
        if auto_commit:
            ds.commit(message="Add samples via Streamlit UI")
        return None
    except Exception as e:
        return f"Failed to add samples: {e}"


def update_sample(ds: Any, tensor_name: str, index: int, value: Any) -> Optional[str]:
    """Update a single sample value."""
    try:
        ds[tensor_name][index] = value
        return None
    except Exception as e:
        return f"Failed to update sample: {e}"


def delete_sample(ds: Any, index: int) -> Optional[str]:
    """Delete a sample by index."""
    try:
        ds.pop(index)
        return None
    except Exception as e:
        return f"Failed to delete sample: {e}"


def run_query(ds: Any, conditions: List[Tuple[str, str, Any]],
              connectors: Optional[List[str]] = None,
              offset: int = 0, limit: Optional[int] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Run a filter query on the dataset."""
    try:
        kwargs = {"offset": offset, "limit": limit}
        if connectors:
            kwargs["connector_list"] = connectors
        result = ds.filter_vectorized(conditions, **kwargs)
        return result, None
    except Exception as e:
        return None, f"Query failed: {e}"


def dataset_to_dataframe(ds: Any, tensor_list: Optional[List[str]] = None,
                         start: int = 0, end: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Convert dataset (or view) to pandas DataFrame."""
    try:
        if end is not None:
            view = ds[start:end]
        elif start > 0:
            view = ds[start:]
        else:
            view = ds
        df = view.to_dataframe(tensor_list=tensor_list)
        return df, None
    except Exception as e:
        # Fallback: manual conversion for filtered views that may not support to_dataframe
        try:
            if tensor_list is None:
                tensor_list = list(ds.tensors.keys())
            data = {}
            for tname in tensor_list:
                tensor = ds[tname]
                vals = tensor.numpy(aslist=True)
                if end is not None:
                    vals = vals[start:end]
                elif start > 0:
                    vals = vals[start:]
                data[tname] = vals
            return pd.DataFrame(data), None
        except Exception as e2:
            return None, f"Failed to convert to DataFrame: {e2}"


def branch_ops(ds: Any, action: str, branch_name: Optional[str] = None,
               merge_strategy: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Perform version control operations."""
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
                update_resolution=strategy.get("update_resolution", "ours"),
            )
            return f"Merged '{branch_name}' into current branch", None

        elif action == "detect_conflict":
            conflict_cols, conflict_records = ds.detect_merge_conflict(branch_name, show_value=True)
            return {"columns": conflict_cols, "records": conflict_records}, None

        elif action == "list":
            return ds.branches, None

        elif action == "commit":
            cid = ds.commit(message=merge_strategy.get("message", "Commit from UI") if merge_strategy else "Commit from UI")
            return f"Committed: {cid}", None

        elif action == "log":
            return ds.commits(ordered_by_date=True), None

        elif action == "diff":
            diff = ds.diff(as_dict=True, show_value=True)
            return diff, None

        else:
            return None, f"Unknown action: {action}"

    except Exception as e:
        return None, f"Branch operation failed: {e}"


def build_commit_graph_data(ds: Any) -> Dict[str, Any]:
    """Extract the full commit DAG from dataset version state for visualization.

    Returns a JSON-serializable dict with commits, branches, and lane assignments.
    """
    from muller.constants import FIRST_COMMIT_ID
    from collections import deque

    commit_node_map = ds.version_state.get("commit_node_map", {})
    branch_commit_map = ds.version_state.get("branch_commit_map", {})
    branch_info = ds.version_state.get("branch_info", {})
    current_branch = ds.version_state.get("branch", "main")
    current_node = ds.version_state.get("commit_node")
    current_commit_id = current_node.commit_id if current_node else ""

    # Assign lanes: main=0, others sorted by create_time
    all_branch_names = set(branch_commit_map.keys())
    for node in commit_node_map.values():
        all_branch_names.add(node.branch)

    def _branch_sort_key(name):
        if name == "main":
            return (0, "")
        info = branch_info.get(name, {})
        ct = info.get("create_time")
        if ct is None:
            return (2, name)
        if hasattr(ct, "timestamp"):
            return (1, ct.timestamp())
        return (1, str(ct))

    sorted_branches = sorted(all_branch_names, key=_branch_sort_key)
    branch_lanes = {name: i for i, name in enumerate(sorted_branches)}

    # BFS to collect all reachable nodes
    root = commit_node_map.get(FIRST_COMMIT_ID)
    if not root:
        return {"commits": [], "branches": [], "lane_count": 0}

    all_nodes = {}
    visited = set()
    queue = deque([root])
    visited.add(root.commit_id)
    while queue:
        node = queue.popleft()
        all_nodes[node.commit_id] = node
        for child in node.children:
            if child.commit_id not in visited:
                visited.add(child.commit_id)
                queue.append(child)

    # Assign depth by commit_time order so that left-to-right = chronological.
    # Nodes with commit_time are sorted by time; uncommitted heads go to the end.
    committed = [n for n in all_nodes.values() if n.commit_time is not None]
    uncommitted = [n for n in all_nodes.values() if n.commit_time is None]
    committed.sort(key=lambda n: n.commit_time)
    uncommitted.sort(key=lambda n: branch_lanes.get(n.branch, 0))

    depth = {}
    d = 0
    prev_time = None
    for n in committed:
        if prev_time is not None and n.commit_time != prev_time:
            d += 1
        depth[n.commit_id] = d
        prev_time = n.commit_time
    # Uncommitted heads each get their own depth at the end
    if uncommitted:
        d += 1
        for n in uncommitted:
            depth[n.commit_id] = d

    sorted_nodes = sorted(all_nodes.values(),
                          key=lambda n: (depth[n.commit_id], branch_lanes.get(n.branch, 0)))
    topo_order = list(reversed(sorted_nodes))
    max_depth = max(depth.values()) if depth else 0

    # Build commit records
    commits = []
    for row_idx, node in enumerate(topo_order):
        lane = branch_lanes.get(node.branch, 0)
        commits.append({
            "id": node.commit_id,
            "short_id": node.commit_id[:8],
            "branch": node.branch,
            "lane": lane,
            "depth": depth[node.commit_id],
            "message": node.commit_message or "",
            "time": str(node.commit_time)[:-7] if node.commit_time else "",
            "author": node.commit_user_name or "",
            "parent_id": node.parent.commit_id if node.parent else None,
            "merge_parent_id": node.merge_parent if node.merge_parent else None,
            "is_merge": node.is_merge_node,
            "is_head": node.is_head_node,
            "is_checkout": getattr(node, "checkout_node", False),
            "is_current": (node.commit_id == current_commit_id),
            "row": row_idx,
        })

    branches_list = []
    for name in sorted_branches:
        branches_list.append({
            "name": name,
            "lane": branch_lanes[name],
            "is_current": name == current_branch,
            "head_commit_id": branch_commit_map.get(name, ""),
        })

    return {
        "commits": commits,
        "branches": branches_list,
        "lane_count": len(branch_lanes),
        "max_depth": max_depth,
    }


def render_commit_graph_html(graph_data: Dict[str, Any], height: int = 500) -> str:
    """Render the commit graph as self-contained horizontal HTML with SVG + vanilla JS."""
    import json
    json_data = json.dumps(graph_data)

    return f'''<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "SF Mono", "Cascadia Code", "Fira Code", Menlo, monospace; background: #fafbfc; }}
  .graph-wrap {{ position: relative; overflow-x: auto; overflow-y: auto; max-height: {height}px; padding: 0; }}
  .tooltip {{
    position: absolute; display: none; background: #24292e; color: #e1e4e8;
    padding: 8px 12px; border-radius: 6px; font-size: 11.5px; line-height: 1.6;
    pointer-events: none; z-index: 100; white-space: pre-line; max-width: 300px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }}
  .tooltip b {{ color: #79b8ff; }}
  .tooltip .t-branch {{ color: #85e89d; }}
  .tooltip .t-msg {{ color: #d1d5da; }}
  .tooltip .t-meta {{ color: #959da5; font-size: 10.5px; }}
  .tooltip .t-tag {{ display: inline-block; padding: 0 5px; border-radius: 3px; font-size: 10px;
    margin-left: 4px; font-weight: 600; }}
  .tooltip .t-merge {{ background: #b3920020; color: #ffdf5d; border: 1px solid #ffdf5d50; }}
  .tooltip .t-cur   {{ background: #34d05820; color: #85e89d; border: 1px solid #85e89d50; }}
</style>
</head>
<body>
<div class="graph-wrap" id="wrap">
  <svg id="graph" xmlns="http://www.w3.org/2000/svg"></svg>
  <div class="tooltip" id="tip"></div>
</div>
<script>
(function() {{
  const D = {json_data};
  if (!D.commits || !D.commits.length) {{
    document.getElementById("wrap").innerHTML = '<p style="padding:16px;color:#8b949e;font-size:13px;">No commits yet.</p>';
    return;
  }}

  /* ── Layout constants ── */
  const COL  = 70;          /* horizontal spacing between depth levels */
  const LANE = 46;          /* vertical spacing between branch lanes */
  const R    = 5;           /* node radius */
  const LM   = 90;          /* left margin for branch labels */
  const TM   = 22;          /* top margin */

  /* ── Palette: muted, professional tones ── */
  const PAL = [
    "#2ea043",  /* green  – main */
    "#388bfd",  /* blue   */
    "#d29922",  /* amber  */
    "#a371f7",  /* purple */
    "#f85149",  /* red    */
    "#3fb950",  /* lime   */
    "#56d4dd",  /* cyan   */
    "#db6d28",  /* orange */
    "#e275ad",  /* pink   */
    "#768390",  /* gray   */
  ];
  function clr(lane) {{ return PAL[lane % PAL.length]; }}

  /* ── SVG helpers ── */
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, a) {{
    const e = document.createElementNS(NS, tag);
    for (const k in a) e.setAttribute(k, a[k]);
    return e;
  }}
  function g() {{ return document.createElementNS(NS, "g"); }}

  const svg = document.getElementById("graph");
  const tip = document.getElementById("tip");
  const maxD = D.max_depth || 0;
  const W = LM + (maxD + 1) * COL + 40;
  const H = TM + D.lane_count * LANE + 24;
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);

  /* ── Compute node positions ── */
  const pos = {{}};
  D.commits.forEach(c => {{
    pos[c.id] = {{ x: LM + c.depth * COL, y: TM + c.lane * LANE }};
  }});

  /* ── Layer 1: branch lane rails (subtle dashed lines) ── */
  D.branches.forEach(b => {{
    const y = TM + b.lane * LANE;
    svg.appendChild(el("line", {{
      x1: LM - 4, y1: y, x2: W - 16, y2: y,
      stroke: clr(b.lane), "stroke-width": "1", "stroke-dasharray": "2,4", opacity: "0.18"
    }}));
  }});

  /* ── Layer 2: branch labels ── */
  D.branches.forEach(b => {{
    const y = TM + b.lane * LANE;
    const isCur = b.is_current;
    /* Rounded-rect badge */
    const label = b.name;
    const charW = isCur ? 7.4 : 6.8;
    const pw = label.length * charW + 14;
    const ph = 18;
    const bx = LM - 8 - pw;
    const bg = g();
    bg.appendChild(el("rect", {{
      x: bx, y: y - ph / 2, width: pw, height: ph, rx: "9", ry: "9",
      fill: isCur ? clr(b.lane) : "#f6f8fa",
      stroke: clr(b.lane), "stroke-width": isCur ? "0" : "1",
      opacity: isCur ? "1" : "0.7"
    }}));
    const txt = el("text", {{
      x: bx + pw / 2, y: y + 4, "text-anchor": "middle",
      fill: isCur ? "#ffffff" : clr(b.lane),
      "font-size": "11px", "font-weight": isCur ? "700" : "500",
      "font-family": "'SF Mono', Menlo, monospace"
    }});
    txt.textContent = label;
    bg.appendChild(txt);
    svg.appendChild(bg);
  }});

  /* ── Arrow head helper (two-line style) ── */
  function arrow(tx, ty, dx, dy, color, op) {{
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len, uy = dy / len;
    const px = -uy, py = ux;
    const aL = 6, aW = 3;
    const bx = tx - ux * aL, by = ty - uy * aL;
    svg.appendChild(el("line", {{
      x1: tx, y1: ty, x2: bx + px * aW, y2: by + py * aW,
      stroke: color, "stroke-width": "1.5", opacity: op, "stroke-linecap": "round"
    }}));
    svg.appendChild(el("line", {{
      x1: tx, y1: ty, x2: bx - px * aW, y2: by - py * aW,
      stroke: color, "stroke-width": "1.5", opacity: op, "stroke-linecap": "round"
    }}));
  }}

  /* ── Layer 3: edges ── */
  const edgeG = g();
  svg.appendChild(edgeG);
  D.commits.forEach(c => {{
    [["parent_id", false], ["merge_parent_id", true]].forEach(([key, isMergeEdge]) => {{
      const pid = c[key];
      if (!pid || !pos[pid]) return;
      const s = pos[pid], d = pos[c.id];
      const pc = D.commits.find(x => x.id === pid);
      const lane = pc ? pc.lane : c.lane;
      const sc = clr(lane);
      const op = isMergeEdge ? "0.35" : "0.50";
      const sw = isMergeEdge ? "1.5" : "2";
      const dash = isMergeEdge ? "4,3" : "none";

      if (s.y === d.y) {{
        /* Same lane: straight */
        edgeG.appendChild(el("line", {{
          x1: s.x, y1: s.y, x2: d.x, y2: d.y,
          stroke: sc, "stroke-width": sw, opacity: op, "stroke-dasharray": dash
        }}));
        arrow(d.x - R - 1, d.y, 1, 0, sc, op);
      }} else {{
        /* Cross-lane: cubic bezier for smooth S-curve */
        const mx = (s.x + d.x) / 2;
        edgeG.appendChild(el("path", {{
          d: `M ${{s.x}} ${{s.y}} C ${{mx}} ${{s.y}}, ${{mx}} ${{d.y}}, ${{d.x}} ${{d.y}}`,
          stroke: sc, "stroke-width": sw, opacity: op, fill: "none",
          "stroke-dasharray": dash
        }}));
        arrow(d.x - R - 1, d.y, 1, 0, sc, op);
      }}
    }});
  }});

  /* ── Layer 4: commit nodes ── */
  D.commits.forEach(c => {{
    const cx = pos[c.id].x, cy = pos[c.id].y, cc = clr(c.lane);
    const ng = g();
    ng.style.cursor = "pointer";

    /* Current node: static highlight ring */
    if (c.is_current) {{
      ng.appendChild(el("circle", {{
        cx, cy, r: R + 5, fill: "none", stroke: cc, "stroke-width": "2", opacity: "0.25"
      }}));
    }}

    /* White outline to separate from edges */
    ng.appendChild(el("circle", {{ cx, cy, r: R + 1.5, fill: "#fafbfc" }}));

    if (c.is_merge) {{
      /* Merge node: double ring */
      ng.appendChild(el("circle", {{ cx, cy, r: R, fill: "none", stroke: cc, "stroke-width": "2" }}));
      ng.appendChild(el("circle", {{ cx, cy, r: R - 2.5, fill: cc }}));
    }} else if (c.is_head && !c.time) {{
      /* Uncommitted head: dashed outline */
      ng.appendChild(el("circle", {{
        cx, cy, r: R, fill: "#fafbfc", stroke: cc, "stroke-width": "1.5",
        "stroke-dasharray": "2,2"
      }}));
    }} else {{
      /* Normal commit: solid circle */
      ng.appendChild(el("circle", {{ cx, cy, r: R, fill: cc }}));
    }}

    /* Short hash label below node */
    const label = el("text", {{
      x: cx, y: cy + R + 12, "text-anchor": "middle", fill: "#8b949e",
      "font-size": "9px", "font-family": "'SF Mono', Menlo, monospace"
    }});
    label.textContent = c.short_id;
    ng.appendChild(label);

    /* Tooltip interaction */
    ng.addEventListener("mouseenter", () => {{
      let h = '<b>' + c.short_id + '</b> <span class="t-branch">' + c.branch + '</span>';
      if (c.is_merge) h += '<span class="t-tag t-merge">merge</span>';
      if (c.is_current) h += '<span class="t-tag t-cur">HEAD</span>';
      h += '\\n';
      if (c.message) h += '<span class="t-msg">' + c.message + '</span>\\n';
      else if (c.is_head && !c.time) h += '<span class="t-msg" style="opacity:0.6">(uncommitted)</span>\\n';
      if (c.author || c.time) h += '<span class="t-meta">' + (c.author || '') + (c.time ? ' \u00b7 ' + c.time : '') + '</span>';
      tip.innerHTML = h;
      tip.style.display = "block";
    }});
    ng.addEventListener("mousemove", (e) => {{
      const r = document.getElementById("wrap").getBoundingClientRect();
      tip.style.left = (e.clientX - r.left + 14) + "px";
      tip.style.top  = (e.clientY - r.top  - 8)  + "px";
    }});
    ng.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});

    svg.appendChild(ng);
  }});
}})();
</script>
</body>
</html>'''


def benchmark_parquet_vs_muller(ds: Any, query_conditions: List[Tuple[str, str, Any]],
                                parquet_path: Optional[str] = None,
                                num_runs: int = 3) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Benchmark query performance: Parquet vs MULLER with multiple runs."""
    try:
        # Export to Parquet if not provided
        if parquet_path is None:
            parquet_path = str(Path(ds.path).parent / "benchmark_temp.parquet")
            ds.write_to_parquet(parquet_path)

        # Benchmark MULLER query (average over num_runs)
        muller_times = []
        for _ in range(num_runs):
            t0 = time.time()
            _ = ds.filter_vectorized(query_conditions)
            muller_times.append(time.time() - t0)
        muller_time = np.mean(muller_times)

        # Benchmark Parquet query via pandas
        import pyarrow.parquet as pq
        parquet_times = []
        for _ in range(num_runs):
            t0 = time.time()
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            for field, op, value in query_conditions:
                if op == "==":
                    df = df[df[field] == value]
                elif op == "!=":
                    df = df[df[field] != value]
                elif op == ">":
                    df = df[df[field] > value]
                elif op == "<":
                    df = df[df[field] < value]
                elif op == ">=":
                    df = df[df[field] >= value]
                elif op == "<=":
                    df = df[df[field] <= value]
            parquet_times.append(time.time() - t0)
        parquet_time = np.mean(parquet_times)

        # File sizes
        muller_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(ds.path)
            for f in files
        ) / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)

        # Create two-panel Plotly chart
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Query Latency", "Storage Size"))

        fig.add_trace(
            go.Bar(x=["MULLER", "Parquet"], y=[muller_time, parquet_time],
                   marker_color=["#636EFA", "#EF553B"], text=[f"{muller_time:.4f}s", f"{parquet_time:.4f}s"],
                   textposition="auto"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=["MULLER", "Parquet"], y=[muller_size, parquet_size],
                   marker_color=["#636EFA", "#EF553B"], text=[f"{muller_size:.2f}MB", f"{parquet_size:.2f}MB"],
                   textposition="auto"),
            row=1, col=2
        )

        fig.update_yaxes(title_text="Seconds", row=1, col=1)
        fig.update_yaxes(title_text="MB", row=1, col=2)
        fig.update_layout(title=f"MULLER vs Parquet (avg of {num_runs} runs)", height=400, showlegend=False)

        return fig, None

    except Exception as e:
        return None, f"Benchmark failed: {e}"


def load_dataset(path: str) -> Tuple[Optional[Any], Optional[str]]:
    """Load an existing MULLER dataset."""
    try:
        ds = muller.load(path)
        return ds, None
    except Exception as e:
        return None, f"Failed to load dataset: {e}"


def commit_dataset(ds: Any, message: str = "Commit from Streamlit UI") -> Tuple[Optional[str], Optional[str]]:
    """Commit changes to dataset. Returns (commit_id, error)."""
    try:
        cid = ds.commit(message=message)
        return cid, None
    except Exception as e:
        return None, f"Commit failed: {e}"


def get_dataset_info(ds: Any) -> Dict[str, Any]:
    """Return summary info about the dataset."""
    try:
        return {
            "path": ds.path,
            "branch": ds.branch,
            "num_samples": len(ds),
            "tensors": {name: {"htype": t.htype, "dtype": str(t.dtype)} for name, t in ds.tensors.items()},
            "commit_id": ds.commit_id,
            "has_uncommitted": ds.has_head_changes,
        }
    except Exception as e:
        return {"error": str(e)}
