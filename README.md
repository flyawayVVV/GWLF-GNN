# GWLF-STGNN Daily Streamflow Simulation

[简体中文](README.zh-CN.md) | English

This repository provides the code and processed input data required to reproduce the daily streamflow simulation experiments of the GWLF-STGNN study. The study focuses on the upper Yangtze River basin above Cuntan hydrological station. A calibrated GWLF model is first used to estimate daily surface runoff and groundwater/baseflow generation for 16 subbasins. These physically informed water-generation inputs are then coupled with spatiotemporal graph neural networks to represent river-network-constrained water transfer, and the proposed models are compared with LSTM and Transformer time-series baselines.

## Repository Structure

- `baseline_lstm.py`: LSTM baseline using historical surface runoff and groundwater/baseflow series from 16 subbasins.
- `baseline_transformer.py`: Transformer Encoder baseline using the same inputs as the LSTM model.
- `stgnn_direct.py`: STGNN model based on the direct upstream-downstream river network.
- `stgnn_reachability.py`: STGNN model based on hydrological reachability connections.
- `run_experiments.py`: batch runner for the final multi-seed experiments and result summaries.
- `requirements.txt`: minimal Python dependency list.
- `data/`: processed input data required for reproducing the released experiments.

## Runtime Environment

The experiments were developed with Python 3.9 and PyTorch. A typical installation is:

```bash
pip install -r requirements.txt
```

If you use conda, install PyTorch first using the official command for your CUDA/CPU platform, and then install the remaining dependencies from `requirements.txt`.

## Input Data

The `data/` directory contains four processed Excel workbooks used directly by the model scripts:

- `runoff-Volume.xlsx`: daily GWLF surface runoff volume for 16 subbasins.
- `groundwater-Volume.xlsx`: daily GWLF groundwater/baseflow volume for 16 subbasins.
- `Flow.xlsx`: observed daily outlet streamflow at Cuntan hydrological station.
- `WatershedInfo_new.xlsx`: subbasin attributes and river-network topology.

These workbooks are the processed model inputs used for neural-network reproduction. Raw hydrological yearbook records, raw meteorological station observations, land-use rasters, DEM data, and GIS preprocessing products are not redistributed in this repository. SHA256 checksums for the released data files are provided in `data/CHECKSUMS_SHA256.txt`.

## Run a Single Model

Examples:

```bash
python baseline_lstm.py
python baseline_transformer.py
python stgnn_direct.py
python stgnn_reachability.py
```

Model settings can be overridden using environment variables. For example:

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

The final experiment matrix contains 50 independent training runs:

- LSTM baseline: 5 random seeds.
- Transformer baseline: 5 random seeds.
- Direct STGNN: 4 edge-weight modes x 5 random seeds.
- Reachability STGNN: 4 edge-weight modes x 5 random seeds.

Run the full experiment matrix with:

```bash
python run_experiments.py --data-dir data --result-root results_final
```

To skip completed runs that already contain `metrics.csv`:

```bash
python run_experiments.py --data-dir data --result-root results_final --skip-existing
```

To summarize an existing result directory only:

```bash
python run_experiments.py --result-root results_final --summary-only
```

The batch runner stores per-run outputs and summary tables under `results_final/`, including:

- `all_metrics.csv`
- `test_metrics_by_seed.csv`
- `test_summary_all_variants.csv`
- `test_summary_all_variants_mean_std.csv`
- `direct_edge_ablation_summary_mean_std.csv`
- `reachability_edge_ablation_summary_mean_std.csv`

## Edge-Weight Modes

Both STGNN scripts support four edge-weight modes:

- `binary`: hydrological connectivity only.
- `inv_length`: inverse river length.
- `inv_time_v1`: inverse travel time estimated from river length and velocity.
- `inv_time_kirpich`: inverse relative concentration time estimated using the Kirpich empirical equation.

In the final experiments, `stgnn_direct.py` uses 6 GNN layers by default, while `stgnn_reachability.py` uses 1 GNN layer by default.

## Environment Variable Notes

The scripts keep the following setting:

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

This setting is used by PyTorch when deterministic CUDA algorithms are enabled. It improves reproducibility for GPU runs and is harmless for CPU-only runs.

Earlier internal scripts also included:

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

This is not a generally required setting. It is a temporary workaround for duplicate Intel OpenMP runtime errors sometimes encountered in Windows/Anaconda environments. It is not enabled by default in this public release. If the error occurs, first consider fixing the Python environment. As a temporary workaround, the batch runner provides:

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

The STGNN scripts additionally write `edge_table_used.csv`, which records the graph edges and edge attributes used by the model.
