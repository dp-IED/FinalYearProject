"""
Compare evaluation results across methods (GNN, KG->LLM, KAG, LLM, RAG).

Subcommands:
  run (or generate): run default evals (GNN, KG->LLM, LLM) then compare. Only needs --limit, --gdn-model; optional --output-dir. Output goes to <output-dir>/eval_<limit>/.
  compare: load/compare by --method (optional --run to run evals first). Needs --method; optional --run, --limit, --output-dir, etc.
No HTML; website can consume the JSON/markdown separately.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root (repo root when running from evaluations/eval_llm.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sanitize(name: str) -> str:
    """Sanitize model/checkpoint name for filename (e.g. granite-4.0-h-micro -> granite_4_0_h_micro)."""
    return name.strip().replace("-", "_") if name else ""


def infer_results_path(
    method_type: str, model_or_checkpoint: str, output_dir: Path, output_tag: str = ""
) -> Path:
    """Infer result JSON path from method type and model name or checkpoint path. If output_tag is set, append _<tag> before .json."""
    model_or_checkpoint = (model_or_checkpoint or "").strip()
    san = _sanitize(model_or_checkpoint)
    suffix = f"_{output_tag}" if output_tag else ""

    def _path(base: str) -> Path:
        return output_dir / f"{base}{suffix}.json"

    if method_type == "GNN":
        stem = Path(model_or_checkpoint).stem if model_or_checkpoint else "gdn"
        return _path(f"gdn_kg_{stem}")
    if method_type == "KG->LLM":
        return _path(f"serialised_kg_{san}") if san else _path("serialised_kg")
    if method_type == "KAG":
        return _path(f"kag_{san}") if san else _path("kag")
    if method_type == "LLM":
        return _path("llm_baseline") if not san or san == "baseline" else _path(f"llm_baseline_{san}")
    if method_type == "RAG":
        return _path("rag") if not san else _path(f"rag_{san}")
    return output_dir / f"{method_type}_{san}{suffix}.json".replace("->", "_")


def load_results(path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def flatten_result_to_row(result: Dict[str, Any], method_type: str, model_id: str) -> Dict[str, Any]:
    """Build a flat row dict from one result JSON: method_type, model_id, and all metric keys."""
    metrics = result.get("metrics", {})
    row: Dict[str, Any] = {
        "method_type": method_type,
        "model_id": model_id or "",
    }
    wl = metrics.get("window_level", {})
    sl = metrics.get("sensor_level", {})
    eff = metrics.get("efficiency", {})
    # Metrics already use window_*, sensor_* keys; add efficiency_ prefix for nested keys
    for key, val in wl.items():
        if isinstance(val, (int, float)):
            row[key] = val
    for key, val in sl.items():
        if isinstance(val, (int, float)):
            row[key] = val
    for key, val in eff.items():
        if isinstance(val, (int, float)):
            row[f"efficiency_{key}"] = val
    row["num_windows"] = result.get("num_windows")
    return row


def select_best_per_type(
    entries: List[Tuple[str, str, Path]],
    key_metric: str = "window_f1",
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    For each (method_type, model_id), keep the result with highest key_metric.
    entries: list of (method_type, model_id, path).
    Returns list of (method_type, model_id, row_dict) with one row per (type, model).
    """
    from collections import defaultdict

    by_type_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for method_type, model_id, path in entries:
        if not path.exists():
            continue
        result = load_results(path)
        row = flatten_result_to_row(result, method_type, model_id)
        row["_path"] = str(path)
        by_type_model[(method_type, model_id)].append(row)

    best: List[Tuple[str, str, Dict[str, Any]]] = []
    for (method_type, model_id), rows in by_type_model.items():
        if not rows:
            continue
        # Remove internal key for output
        for r in rows:
            r.pop("_path", None)
        chosen = max(rows, key=lambda r: float(r.get(key_metric, 0) or 0))
        best.append((method_type, model_id, chosen))
    return best


def all_metric_keys(rows: List[Dict[str, Any]]) -> List[str]:
    """Collect all metric-like keys (excluding method_type, model_id) and return sorted."""
    keys = set()
    for r in rows:
        for k in r:
            if k not in ("method_type", "model_id"):
                keys.add(k)
    return sorted(keys)


def build_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Build markdown table in a loop; columns = ['method_type', 'model_id', ...metrics]."""
    lines: List[str] = []
    # Header
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _resolve_dataset_path(dataset_dir: Path, dataset_suffix: str) -> Path:
    """Resolve dataset path: dataset_dir / f'{dataset_suffix}.npz'. If dataset_dir is relative, resolve from PROJECT_ROOT."""
    d = Path(dataset_dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    return d / f"{dataset_suffix}.npz"


def _run_eval(
    method_type: str,
    model_or_checkpoint: str,
    output_path: Path,
    dataset_path: Path,
    gdn_model_path: Optional[Path],
    limit: Optional[int],
) -> bool:
    """Run the evaluation script for the given method. Returns True on success."""
    limit_args = ["--limit", str(limit)] if limit is not None else []

    if method_type == "GNN":
        if not gdn_model_path or not gdn_model_path.exists():
            print("  Skip GNN (missing --gdn-model or file not found)", file=sys.stderr)
            return False
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "llm/evaluation/evaluate_gdn_kg.py"),
            "--dataset", str(dataset_path),
            "--model-path", str(gdn_model_path),
            "--output", str(output_path),
            "--device", "cpu",
        ] + limit_args
    elif method_type == "KG->LLM":
        if not gdn_model_path or not gdn_model_path.exists():
            print("  Skip KG->LLM (missing --gdn-model or file not found)", file=sys.stderr)
            return False
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "llm/evaluation/evaluate_gdn_kg_llm.py"),
            "--dataset", str(dataset_path),
            "--model-path", str(gdn_model_path),
            "--output", str(output_path),
            "--model-repo", model_or_checkpoint or "granite-4.0-h-micro-GGUF",
            "--no-neo4j-sync",
            "--device", "cpu",
        ] + limit_args
    elif method_type == "KAG":
        if not gdn_model_path or not gdn_model_path.exists():
            print("  Skip KAG (missing --gdn-model or file not found)", file=sys.stderr)
            return False
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "llm/evaluation/evaluate_kag_v2.py"),
            "--dataset", str(dataset_path),
            "--gdn-model", str(gdn_model_path),
            "--output", str(output_path),
            "--device", "cpu",
        ]
        if model_or_checkpoint:
            cmd.extend(["--model-repo", model_or_checkpoint])
        cmd.extend(limit_args)
    elif method_type == "LLM":
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "llm/evaluation/evaluate_llm_baseline.py"),
            "--dataset", str(dataset_path),
            "--output", str(output_path),
        ]
        if model_or_checkpoint and model_or_checkpoint.lower() != "baseline":
            cmd.extend(["--model-repo", model_or_checkpoint])
        cmd.extend(limit_args)
    elif method_type == "RAG":
        print("  Skip RAG (run RAG eval separately; ChromaDB etc. not wired here)", file=sys.stderr)
        return False
    else:
        print(f"  Unknown method type: {method_type}", file=sys.stderr)
        return False

    out = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return out.returncode == 0


def _do_compare(
    entries: List[Tuple[str, str, Path]],
    output_dir: Path,
    output_tag: str,
    best_metric: str,
) -> int:
    """Run comparison on entries, print table, write compare.md and compare.json. Returns 0 on success."""
    best_rows_raw = select_best_per_type(entries, key_metric=best_metric)
    rows = [r for (_, _, r) in best_rows_raw]

    if not rows:
        print("No results to compare (no valid JSON paths?).")
        return 1

    columns = ["method_type", "model_id"] + all_metric_keys(rows)
    for r in rows:
        for c in columns:
            if c not in r:
                r[c] = None
    rows.sort(key=lambda r: (r["method_type"], r["model_id"]))

    out = {"columns": columns, "rows": rows, "best_metric_used": best_metric}
    print("\nComparison (best per method_type, model_id):")
    print(build_markdown_table(rows, columns))

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{output_tag}" if output_tag else ""
    md_path = output_dir / f"compare{tag}.md"
    md_path.write_text(build_markdown_table(rows, columns), encoding="utf-8")
    print(f"\nMarkdown table saved to: {md_path}")
    json_path = output_dir / f"compare{tag}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON saved to: {json_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run default evals (GNN, KG->LLM, LLM) then compare. Output to <output-dir>/eval_<limit>/."""
    limit = args.limit
    gdn_path = Path(args.gdn_model)
    if not gdn_path.is_absolute():
        gdn_path = PROJECT_ROOT / gdn_path
    if not gdn_path.exists():
        raise SystemExit(f"--gdn-model path not found: {gdn_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_dir = output_dir / f"eval_{limit}"
    dataset_suffix = f"test_{limit}"
    dataset_path = _resolve_dataset_path(Path(args.dataset_dir), dataset_suffix)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path} (need test_{limit}.npz in shared_dataset)")

    # Default methods: GNN (checkpoint), KG->LLM (default micro), LLM baseline
    gdn_str = str(gdn_path)
    entries: List[Tuple[str, str, Path]] = [
        ("GNN", gdn_str, infer_results_path("GNN", gdn_str, run_dir, "")),
        ("KG->LLM", "granite-4.0-h-micro", infer_results_path("KG->LLM", "granite-4.0-h-micro", run_dir, "")),
        ("LLM", "baseline", infer_results_path("LLM", "baseline", run_dir, "")),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run evals (limit={limit}) -> {run_dir}")
    for method_type, model_or_checkpoint, out_path in entries:
        print(f"  Running {method_type} {model_or_checkpoint or '(default)'} -> {out_path.name} ...")
        ok = _run_eval(method_type, model_or_checkpoint, out_path, dataset_path, gdn_path, limit)
        if not ok:
            print(f"    Warning: eval failed", file=sys.stderr)
    return _do_compare(entries, run_dir, "", args.best_metric)


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare by --method; optionally --run evals first."""
    entries: List[Tuple[str, str, Path]] = []
    for s in args.methods:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) < 2:
            raise SystemExit(f"Each --method must be 'METHOD_TYPE,MODEL_OR_CHECKPOINT', got: {s!r}")
        method_type, model_or_checkpoint = parts[0], parts[1]
        path = infer_results_path(method_type, model_or_checkpoint, args.output_dir, args.output_tag)
        entries.append((method_type, model_or_checkpoint, path))

    if args.run:
        dataset_suffix = f"test_{args.limit}" if args.limit is not None else "test"
        dataset_path = _resolve_dataset_path(args.dataset_dir, dataset_suffix)
        if not dataset_path.exists():
            raise SystemExit(f"Dataset not found: {dataset_path}")
        need_gdn = any(m in ("GNN", "KG->LLM", "KAG") for m, _, _ in entries)
        gdn_path = None
        if need_gdn:
            if not args.gdn_model:
                raise SystemExit("--gdn-model is required when --run with GNN, KG->LLM, or KAG.")
            gdn_path = Path(args.gdn_model)
            if not gdn_path.is_absolute():
                gdn_path = PROJECT_ROOT / gdn_path
            if not gdn_path.exists():
                raise SystemExit(f"--gdn-model path not found: {gdn_path}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for method_type, model_or_checkpoint, out_path in entries:
            print(f"Running {method_type} {model_or_checkpoint or '(default)'} -> {out_path.name} ...")
            ok = _run_eval(method_type, model_or_checkpoint, out_path, dataset_path, gdn_path, args.limit)
            if not ok:
                print(f"  Warning: eval failed for {method_type},{model_or_checkpoint}", file=sys.stderr)

    return _do_compare(entries, args.output_dir, args.output_tag, args.best_metric)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evals and/or compare results (markdown + JSON, no HTML)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run / generate: --limit, --gdn-model, optional --output-dir
    for name, help_ in [("run", "Run default evals (GNN, KG->LLM, LLM) then compare; output to <output-dir>/eval_<limit>/."), ("generate", "Alias for run.")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--limit", type=int, required=True, help="Window limit; dataset is test_<limit>.npz.")
        p.add_argument("--gdn-model", type=Path, required=True, help="Path to GDN checkpoint.")
        p.add_argument("--output-dir", type=Path, default=Path("results"), help="Base output directory (default: results).")
        p.add_argument("--dataset-dir", type=Path, default=Path("llm/evaluation/shared_dataset"), help="Shared dataset directory.")
        p.add_argument("--best-metric", type=str, default="window_f1", help="Best-metric for comparison.")
        p.set_defaults(_run=_cmd_run)

    # compare: --method, optional --run, --limit, --output-dir, etc.
    cmp_p = sub.add_parser("compare", help="Compare by --method; optionally --run evals first.")
    cmp_p.add_argument("--method", action="append", dest="methods", metavar="METHOD_TYPE,MODEL", required=True)
    cmp_p.add_argument("--output-dir", type=Path, default=Path("results"))
    cmp_p.add_argument("--output-tag", type=str, default="")
    cmp_p.add_argument("--best-metric", type=str, default="window_f1")
    cmp_p.add_argument("--run", action="store_true", help="Run evals before comparing.")
    cmp_p.add_argument("--limit", type=int, default=None)
    cmp_p.add_argument("--dataset-dir", type=Path, default=Path("llm/evaluation/shared_dataset"))
    cmp_p.add_argument("--gdn-model", type=Path, default=None)
    cmp_p.set_defaults(_run=_cmd_compare)

    args = parser.parse_args()
    return args._run(args)


if __name__ == "__main__":
    raise SystemExit(main())
