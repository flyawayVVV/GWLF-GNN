# GWLF-STGNN 日尺度流量模拟

[English](README.md) | 简体中文

本仓库提供 GWLF-STGNN 研究中用于复现日尺度流量模拟实验的代码与处理后输入数据。研究对象为寸滩水文站以上长江上游流域。建模框架首先使用校准后的 GWLF 模型得到 16 个子流域的逐日地表径流与地下潜流/基流产流量，再通过时空图神经网络表征河网约束下的水量传输过程，并与 LSTM、Transformer 时间序列基准模型进行比较。

## 文件结构

- `baseline_lstm.py`：LSTM 基准模型，输入为 16 个子流域的地表径流和地下潜流历史序列。
- `baseline_transformer.py`：Transformer Encoder 基准模型，使用与 LSTM 相同的输入。
- `stgnn_direct.py`：基于直接上游-下游河网边的 STGNN 模型。
- `stgnn_reachability.py`：基于水文可达性边的 STGNN 模型。
- `run_experiments.py`：用于批量运行最终多随机种子实验并汇总结果。
- `requirements.txt`：最小 Python 依赖列表。
- `data/`：复现实验所需的处理后输入数据。

## 运行环境

实验代码使用 Python 3.9 和 PyTorch 开发。可先安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 conda，建议根据本机 CUDA/CPU 环境先按照 PyTorch 官方命令安装 PyTorch，再安装 `requirements.txt` 中的其他依赖。

## 输入数据

`data/` 文件夹中包含模型脚本需要读取的 4 个处理后 Excel 数据文件：

- `runoff-Volume.xlsx`：16 个子流域逐日 GWLF 地表径流产流量。
- `groundwater-Volume.xlsx`：16 个子流域逐日 GWLF 地下潜流/基流产流量。
- `Flow.xlsx`：寸滩水文站逐日实测出口流量。
- `WatershedInfo_new.xlsx`：子流域属性与河网拓扑信息。

这些文件是神经网络复现实验使用的处理后输入数据。原始《中华人民共和国水文年鉴》资料、气象站原始观测、土地利用栅格、DEM 数据和 GIS 预处理中间成果未在本仓库中再分发。数据文件的 SHA256 校验值见 `data/CHECKSUMS_SHA256.txt`。

## 单个模型运行

示例：

```bash
python baseline_lstm.py
python baseline_transformer.py
python stgnn_direct.py
python stgnn_reachability.py
```

模型配置可通过环境变量覆盖。例如：

```bash
DATA_DIR=data OUTPUT_DIR=results_lstm_seed42 SEED=42 python baseline_lstm.py
DATA_DIR=data OUTPUT_DIR=results_direct_kirpich_seed42 SEED=42 EDGE_WEIGHT_MODE=inv_time_kirpich python stgnn_direct.py
```

Windows PowerShell 中可使用：

```powershell
$env:DATA_DIR = "data"
$env:OUTPUT_DIR = "results_direct_kirpich_seed42"
$env:SEED = "42"
$env:EDGE_WEIGHT_MODE = "inv_time_kirpich"
python stgnn_direct.py
```

## 复现最终实验矩阵

最终实验矩阵共包含 50 次独立训练：

- LSTM 基准模型：5 个随机种子。
- Transformer 基准模型：5 个随机种子。
- direct STGNN：4 种边权设置 x 5 个随机种子。
- reachability STGNN：4 种边权设置 x 5 个随机种子。

完整运行命令为：

```bash
python run_experiments.py --data-dir data --result-root results_final
```

如果结果目录中已有完成的 `metrics.csv`，可跳过已有实验：

```bash
python run_experiments.py --data-dir data --result-root results_final --skip-existing
```

只汇总已有结果时可运行：

```bash
python run_experiments.py --result-root results_final --summary-only
```

批处理脚本会在 `results_final/` 下保存每次训练的结果，并生成汇总表，包括：

- `all_metrics.csv`
- `test_metrics_by_seed.csv`
- `test_summary_all_variants.csv`
- `test_summary_all_variants_mean_std.csv`
- `direct_edge_ablation_summary_mean_std.csv`
- `reachability_edge_ablation_summary_mean_std.csv`

## 边权设置

两个 STGNN 脚本均支持 4 种边权设置：

- `binary`：仅表示是否存在水文连通关系。
- `inv_length`：按河道长度倒数构造边权。
- `inv_time_v1`：按长度/流速估计传播时间，并使用传播时间倒数构造边权。
- `inv_time_kirpich`：按 Kirpich 经验公式估计相对汇流时间，并使用时间倒数构造边权。

最终实验中，`stgnn_direct.py` 默认使用 6 层 GNN，`stgnn_reachability.py` 默认使用 1 层 GNN。

## 环境变量说明

脚本中保留了：

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

该设置用于 PyTorch 在启用确定性 CUDA 算法时提高 GPU 计算的可复现性；对 CPU 运行通常没有影响，用户也可以在运行前自行覆盖。

早期内部脚本中曾包含：

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

这不是通用必要设置，而是 Windows/Anaconda 环境中 Intel OpenMP 重复运行库报错时的临时规避方法。本公开版本不再默认启用它。如确实遇到 OpenMP 重复库错误，可优先修正 Python 环境，或临时运行：

```bash
python run_experiments.py --allow-duplicate-openmp
```

## 输出文件

每个模型运行会输出：

- `config.csv`
- `split_info.csv`
- `training_log.csv`
- `training_summary.csv`
- `metrics.csv`
- `train_predictions.csv`、`val_predictions.csv`、`test_predictions.csv`
- `train_hydrograph.png`、`val_hydrograph.png`、`test_hydrograph.png`
- `best_model.pt`

STGNN 模型还会额外输出 `edge_table_used.csv`，用于检查实际参与训练的图边及其属性。
