# GWLF-STGNN Daily Streamflow Simulation

[简体中文](README.md) | English

This repository contains the code and processed input data used to reproduce the daily streamflow simulation experiments for the GWLF-STGNN study in the upper Yangtze River basin above Cuntan station.

## Contents

- `baseline_lstm.py`: LSTM baseline using subbasin runoff and groundwater time series.
- `baseline_transformer.py`: Transformer Encoder baseline using the same inputs.
- `stgnn_direct.py`: STGNN using the direct upstream-downstream river graph.
- `stgnn_reachability.py`: STGNN using the hydrological reachability graph.
- `run_experiments.py`: batch runner and result summarizer for the final multi-seed experiments.
- `requirements.txt`: minimal Python dependencies.
- `data/`: processed input workbooks required by the released code.

## Runtime Environment

The experiments were developed with Python 3.9 and PyTorch. A typical installation is:

```bash
pip install -r requirements.txt
```

If you use conda, install PyTorch following the official PyTorch command for your CUDA/CPU platform, then install the remaining packages from `requirements.txt`.

## Input Data

The `data/` directory contains the four processed Excel workbooks required by the model scripts:

- `runoff-Volume.xlsx`
- `groundwater-Volume.xlsx`
- `Flow.xlsx`
- `WatershedInfo_new.xlsx`

The included workbooks are the processed model inputs used for neural-network reproduction. Raw hydrological yearbook records, meteorological station observations, land-use rasters, DEM data, and GIS preprocessing products are not redistributed here.

## Run a Single Model

Examples:

```bash
python baseline_lstm.py
python baseline_transformer.py
python stgnn_direct.py
python stgnn_reachability.py
```

Configuration can be overridden through environment variables:

```bash
DATA_DIR=data OUTPUT_DIR=results_lstm_seed42 SEED=42 python baseline_lstm.py
DATA_DIR=data OUTPUT_DIR=results_direct_kirpich_seed42 SEED=42 EDGE_WEIGHT_MODE=inv_time_kirpich python stgnn_direct.py
```

On Windows PowerShell, use:

```powershell
$env:DATA_DIR = "data"
$env:OUTPUT_DIR = "results_direct_kirpich_seed42"
$env:SEED = "42"
$env:EDGE_WEIGHT_MODE = "inv_time_kirpich"
python stgnn_direct.py
```

## Reproduce the Final Experiment Matrix

The final matrix contains 50 independent runs:

- LSTM baseline: 5 seeds
- Transformer baseline: 5 seeds
- Direct STGNN: 4 edge-weight modes x 5 seeds
- Reachability STGNN: 4 edge-weight modes x 5 seeds

Run:

```bash
python run_experiments.py --data-dir data --result-root results_final
```

To skip runs that already have `metrics.csv`:

```bash
python run_experiments.py --data-dir data --result-root results_final --skip-existing
```

To summarize an existing result directory only:

```bash
python run_experiments.py --result-root results_final --summary-only
```

The runner writes per-run outputs and summary tables under `results_final/`, including:

- `all_metrics.csv`
- `test_metrics_by_seed.csv`
- `test_summary_all_variants.csv`
- `test_summary_all_variants_mean_std.csv`
- `direct_edge_ablation_summary_mean_std.csv`
- `reachability_edge_ablation_summary_mean_std.csv`

## Edge-Weight Modes

The STGNN scripts support four edge-weight modes:

- `binary`
- `inv_length`
- `inv_time_v1`
- `inv_time_kirpich`

For the final experiments, `stgnn_direct.py` uses 6 GNN layers by default, while `stgnn_reachability.py` uses 1 GNN layer by default.

## Notes on Environment Variables

The scripts set:

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

This is used by PyTorch when deterministic CUDA algorithms are enabled. It helps make GPU runs more reproducible and is harmless for CPU-only runs.

The earlier internal scripts also set:

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

That line is not included as a default in this public code release. It is a Windows/Anaconda workaround for duplicate Intel OpenMP runtime errors. If needed, launch:

```bash
python run_experiments.py --allow-duplicate-openmp
```

## Outputs

Each model run writes:

- `config.csv`
- `split_info.csv`
- `training_log.csv`
- `training_summary.csv`
- `metrics.csv`
- `train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`
- `train_hydrograph.png`, `val_hydrograph.png`, `test_hydrograph.png`
- `best_model.pt`

The STGNN scripts additionally write `edge_table_used.csv`.
