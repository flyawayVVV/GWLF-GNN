# -*- coding: utf-8 -*-
"""
Run and summarize the daily-flow simulation experiments.

Default behavior reproduces the final experiment matrix used in the paper:
- baseline_lstm: 5 random seeds
- baseline_transformer: 5 random seeds
- stgnn_direct: 4 edge-weight modes x 5 random seeds
- stgnn_reachability: 4 edge-weight modes x 5 random seeds

Examples
--------
Run the full final matrix:
    python run_experiments.py --data-dir /path/to/data --result-root results_final

Run only the two graph models with one edge mode:
    python run_experiments.py --models stgnn_direct stgnn_reachability --edge-weight-modes inv_time_kirpich

Summarize an existing result directory without launching new training:
    python run_experiments.py --result-root results_final --summary-only
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


MODEL_SCRIPTS = {
    "baseline_lstm": "baseline_lstm.py",
    "baseline_transformer": "baseline_transformer.py",
    "stgnn_direct": "stgnn_direct.py",
    "stgnn_reachability": "stgnn_reachability.py",
}

GRAPH_MODELS = {"stgnn_direct", "stgnn_reachability"}

DEFAULT_SEEDS = [42, 123, 2024, 3407, 777]
DEFAULT_EDGE_MODES = ["binary", "inv_length", "inv_time_v1", "inv_time_kirpich"]
METRIC_COLS = ["NSE", "KGE", "RMSE", "MAE", "PBIAS_percent", "r"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run GWLF-STGNN daily-flow experiments.")
    parser.add_argument("--data-dir", default=".", help="Directory containing the Excel input files.")
    parser.add_argument("--result-root", default="results_final", help="Root directory for all model runs.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SCRIPTS), choices=list(MODEL_SCRIPTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--edge-weight-modes", nargs="+", default=DEFAULT_EDGE_MODES,
                        choices=DEFAULT_EDGE_MODES, help="Edge modes used by STGNN models.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a run if its metrics.csv already exists.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only summarize existing runs under --result-root.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to launch model scripts.")
    parser.add_argument("--allow-duplicate-openmp", action="store_true",
                        help="Set KMP_DUPLICATE_LIB_OK=TRUE. Use only as a Windows/Anaconda workaround.")

    # Optional overrides passed to model scripts through environment variables.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def run_name(model_name: str, seed: int, edge_mode: str = "") -> str:
    if model_name in GRAPH_MODELS:
        return f"{model_name}_{edge_mode}_seed{seed}"
    return f"{model_name}_seed{seed}"


def stream_subprocess(cmd, env, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parent,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def build_env(args, out_dir: Path, seed: int, edge_mode: str = ""):
    env = os.environ.copy()
    env["DATA_DIR"] = str(Path(args.data_dir))
    env["OUTPUT_DIR"] = str(out_dir)
    env["SEED"] = str(seed)

    # Required by PyTorch for deterministic CUDA algorithms. Harmless on CPU.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.allow_duplicate_openmp:
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    optional_overrides = {
        "EPOCHS": args.epochs,
        "PATIENCE": args.patience,
        "SEQ_LEN": args.seq_len,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.learning_rate,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            env[key] = str(value)
    if edge_mode:
        env["EDGE_WEIGHT_MODE"] = edge_mode
    return env


def iter_requested_runs(args):
    for model_name in args.models:
        script_name = MODEL_SCRIPTS[model_name]
        edge_modes = args.edge_weight_modes if model_name in GRAPH_MODELS else [""]
        for edge_mode in edge_modes:
            for seed in args.seeds:
                yield model_name, script_name, edge_mode, seed


def run_one(args, model_name: str, script_name: str, edge_mode: str, seed: int) -> Path:
    out_dir = Path(args.result_root) / run_name(model_name, seed, edge_mode)
    metrics_path = out_dir / "metrics.csv"
    if args.skip_existing and metrics_path.exists():
        print(f"[skip] {out_dir}")
        return out_dir

    env = build_env(args, out_dir, seed, edge_mode)
    label = run_name(model_name, seed, edge_mode)
    print(f"\n=== Running {label} ===")
    code = stream_subprocess([args.python, script_name], env, out_dir / "run_stdout.log")
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}. See {out_dir / 'run_stdout.log'}")
    return out_dir


def parse_variant_from_dir(run_dir: Path):
    name = run_dir.name
    if name.startswith("baseline_lstm_seed"):
        return "baseline_lstm", "baseline_lstm", "", int(name.split("seed")[-1])
    if name.startswith("baseline_transformer_seed"):
        return "baseline_transformer", "baseline_transformer", "", int(name.split("seed")[-1])
    match = re.match(r"^(stgnn_(?:direct|reachability))_(.+)_seed(\d+)$", name)
    if match:
        family, edge_mode, seed = match.group(1), match.group(2), int(match.group(3))
        return f"{family}_{edge_mode}", family, edge_mode, seed
    return name, "", "", None


def write_markdown_table(df: pd.DataFrame, path: Path):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_results(result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted([p for p in result_root.iterdir() if p.is_dir()])

    metrics_frames = []
    config_frames = []
    training_frames = []
    split_frames = []
    missing_metrics = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        variant, family, edge_mode, seed = parse_variant_from_dir(run_dir)
        common = {
            "run_dir": run_dir.name,
            "model_variant": variant,
            "model_family": family,
            "edge_weight_mode": edge_mode,
            "seed_from_dir": seed,
        }

        metrics = pd.read_csv(metrics_path)
        for key, value in common.items():
            metrics[key] = value
        metrics_frames.append(metrics)

        for file_name, target in [
            ("config.csv", config_frames),
            ("training_summary.csv", training_frames),
            ("split_info.csv", split_frames),
        ]:
            path = run_dir / file_name
            if path.exists():
                df = pd.read_csv(path)
                for key, value in common.items():
                    df[key] = value
                target.append(df)

    if missing_metrics:
        raise RuntimeError("Missing metrics files:\n" + "\n".join(missing_metrics))
    if not metrics_frames:
        raise RuntimeError(f"No metrics.csv files found under {result_root}")

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_metrics.to_csv(result_root / "all_metrics.csv", index=False, encoding="utf-8-sig")

    if config_frames:
        pd.concat(config_frames, ignore_index=True).to_csv(
            result_root / "configs_used.csv", index=False, encoding="utf-8-sig"
        )
    if training_frames:
        pd.concat(training_frames, ignore_index=True).to_csv(
            result_root / "training_summaries.csv", index=False, encoding="utf-8-sig"
        )
    if split_frames:
        pd.concat(split_frames, ignore_index=True).to_csv(
            result_root / "split_infos.csv", index=False, encoding="utf-8-sig"
        )

    test = all_metrics.loc[all_metrics["split"] == "test"].copy()
    test.to_csv(result_root / "test_metrics_by_seed.csv", index=False, encoding="utf-8-sig")

    group_cols = ["model_variant", "model_family", "edge_weight_mode"]
    summary = test.groupby(group_cols, dropna=False)[METRIC_COLS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns.to_flat_index()]
    counts = test.groupby(group_cols, dropna=False).size().reset_index(name="n_seeds")
    summary = summary.merge(counts, on=group_cols)
    summary = summary.sort_values(["NSE_mean", "KGE_mean"], ascending=[False, False]).reset_index(drop=True)
    summary.insert(0, "rank_by_NSE", range(1, len(summary) + 1))
    summary.to_csv(result_root / "test_summary_all_variants.csv", index=False, encoding="utf-8-sig")

    pretty = summary[["rank_by_NSE", "model_variant", "model_family", "edge_weight_mode", "n_seeds"]].copy()
    for metric in METRIC_COLS:
        pretty[metric] = summary.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}",
            axis=1,
        )
    pretty.to_csv(result_root / "test_summary_all_variants_mean_std.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(pretty, result_root / "test_summary_all_variants_mean_std.md")

    for family in GRAPH_MODELS:
        family_summary = summary.loc[summary["model_family"] == family].copy()
        if family_summary.empty:
            continue
        out_name = family.replace("stgnn_", "")
        family_summary.to_csv(result_root / f"{out_name}_edge_ablation_summary.csv", index=False, encoding="utf-8-sig")
        family_pretty = family_summary[["rank_by_NSE", "model_variant", "edge_weight_mode", "n_seeds"]].copy()
        for metric in METRIC_COLS:
            family_pretty[metric] = family_summary.apply(
                lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}",
                axis=1,
            )
        family_pretty.to_csv(
            result_root / f"{out_name}_edge_ablation_summary_mean_std.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_markdown_table(family_pretty, result_root / f"{out_name}_edge_ablation_summary_mean_std.md")

    print(f"\nSummary files written to: {result_root.resolve()}")
    print(pretty[["rank_by_NSE", "model_variant", "NSE", "KGE"]].to_string(index=False))


def main():
    args = parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())
    args.result_root = str(Path(args.result_root).resolve())
    result_root = Path(args.result_root)
    if not args.summary_only:
        for model_name, script_name, edge_mode, seed in iter_requested_runs(args):
            run_one(args, model_name, script_name, edge_mode, seed)
    summarize_results(result_root)


if __name__ == "__main__":
    main()
