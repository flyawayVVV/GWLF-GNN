# -*- coding: utf-8 -*-
"""
baseline_transformer.py

经典 Transformer Encoder 基准模型：
- 输入：16 个子流域过去若干天的 GWLF 地表径流 runoff 和地下潜流 groundwater，单位 m3/day。
- 输出：流域出口当天实测流量，单位 m3/day。
- 仅使用 PyTorch，不依赖 PyTorch Geometric。

运行方式：
1. 将 runoff-Volume.xlsx、groundwater-Volume.xlsx、Flow.xlsx 放到 DATA_DIR 指定的文件夹。
2. 在 main() 中修改 DATA_DIR、SEQ_LEN、EPOCHS 等超参数。
3. 在 Anaconda/IDLE 中打开本文件，Run Module 即可。
"""

import os

# Required by PyTorch for deterministic CUDA matrix operations when
# torch.use_deterministic_algorithms(True) is enabled. It is harmless on CPU
# and can be overridden by users before launching the script.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 基础工具函数
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def get_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def find_date_column(df: pd.DataFrame):
    for col in df.columns:
        if "date" in str(col).strip().lower() or "time" in str(col).strip().lower():
            return col
    return df.columns[0]


def find_subbasin_column(columns, sid: int):
    """在 Excel 列名中查找 1~16 号子流域列。兼容整数列名和字符串列名。"""
    sid_text = str(sid)
    for col in columns:
        if str(col).strip() == sid_text:
            return col
    for col in columns:
        try:
            if int(float(str(col).strip())) == sid:
                return col
        except Exception:
            pass
    raise ValueError(f"没有找到子流域 {sid} 对应的列。请检查 Excel 列名是否为 1, 2, ..., 16。")


def load_subbasin_volume_excel(file_path: str, prefix: str, n_subbasins: int = 16) -> pd.DataFrame:
    """读取 runoff 或 groundwater 体积数据。返回列：date, prefix_1, ..., prefix_16。"""
    df = pd.read_excel(file_path, sheet_name=0)
    date_col = find_date_column(df)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col])
    for sid in range(1, n_subbasins + 1):
        col = find_subbasin_column(df.columns, sid)
        out[f"{prefix}_{sid}"] = pd.to_numeric(df[col], errors="coerce")
    return out


def load_flow_excel(file_path: str, flow_unit: str = "m3/day") -> pd.DataFrame:
    """
    读取出口流量数据。返回列：date, flow。

    flow_unit:
        "m3/day"：输入已经是立方米每天。
        "m3/s"  ：输入是立方米每秒，代码会乘以 86400 转成 m3/day。
    """
    df = pd.read_excel(file_path, sheet_name=0)
    date_col = find_date_column(df)
    numeric_cols = [c for c in df.columns if c != date_col]
    if not numeric_cols:
        raise ValueError("Flow.xlsx 中没有找到流量列。")
    flow_col = numeric_cols[0]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col])
    flow = pd.to_numeric(df[flow_col], errors="coerce").astype(float)
    if flow_unit.lower() in ["m3/s", "cms", "m^3/s"]:
        flow = flow * 86400.0
    out["flow"] = flow
    return out


def load_all_data(data_dir: str,
                  runoff_file: str,
                  groundwater_file: str,
                  flow_file: str,
                  flow_unit: str,
                  n_subbasins: int = 16) -> pd.DataFrame:
    runoff = load_subbasin_volume_excel(os.path.join(data_dir, runoff_file), "runoff", n_subbasins)
    groundwater = load_subbasin_volume_excel(os.path.join(data_dir, groundwater_file), "groundwater", n_subbasins)
    flow = load_flow_excel(os.path.join(data_dir, flow_file), flow_unit)

    df = runoff.merge(groundwater, on="date", how="inner").merge(flow, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"注意：合并后删除了 {before - after} 行含缺失值的数据。")
    if len(df) < 100:
        print("警告：有效样本数较少。demo 数据可以做 smoke test，但正式训练请使用完整逐日数据。")
    return df


class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


def get_target_indices(n_days: int, seq_len: int, use_current_day: bool) -> np.ndarray:
    if use_current_day:
        # 用 [t-seq_len+1, ..., t] 预测 y(t)
        return np.arange(seq_len - 1, n_days)
    else:
        # 用 [t-seq_len, ..., t-1] 预测 y(t)
        return np.arange(seq_len, n_days)


def make_sequences(x: np.ndarray,
                   y: np.ndarray,
                   dates: np.ndarray,
                   seq_len: int,
                   use_current_day: bool):
    xs, ys, ds, target_indices = [], [], [], []
    n_days = len(y)
    for target_idx in get_target_indices(n_days, seq_len, use_current_day):
        if use_current_day:
            start = target_idx - seq_len + 1
            end = target_idx + 1
        else:
            start = target_idx - seq_len
            end = target_idx
        xs.append(x[start:end])
        ys.append(y[target_idx])
        ds.append(dates[target_idx])
        target_indices.append(target_idx)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ds), np.asarray(target_indices)


def prepare_lstm_data(df: pd.DataFrame,
                      seq_len: int,
                      train_ratio: float,
                      val_ratio: float,
                      use_current_day: bool,
                      log_transform: bool,
                      n_subbasins: int = 16):
    runoff_cols = [f"runoff_{i}" for i in range(1, n_subbasins + 1)]
    gw_cols = [f"groundwater_{i}" for i in range(1, n_subbasins + 1)]
    x_raw = df[runoff_cols + gw_cols].values.astype(float)
    y_raw = df["flow"].values.astype(float).reshape(-1, 1)
    dates = df["date"].values

    target_indices_all = get_target_indices(len(df), seq_len, use_current_day)
    if len(target_indices_all) < 10:
        raise ValueError("序列样本太少。请减小 SEQ_LEN 或使用完整数据。")
    n_samples = len(target_indices_all)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    if n_train < 1 or n_val < 1 or (n_samples - n_train - n_val) < 1:
        raise ValueError("训练/验证/测试样本太少。请检查数据长度或调整划分比例。")

    # 只用训练期拟合标准化参数，避免测试期信息泄漏。
    last_train_target_idx = int(target_indices_all[n_train - 1])

    if log_transform:
        x_trans = np.log1p(np.maximum(x_raw, 0.0))
        y_trans = np.log1p(np.maximum(y_raw, 0.0))
    else:
        x_trans = x_raw.copy()
        y_trans = y_raw.copy()

    x_scaler = StandardScalerNP().fit(x_trans[:last_train_target_idx + 1])
    y_scaler = StandardScalerNP().fit(y_trans[:last_train_target_idx + 1])
    x_scaled = x_scaler.transform(x_trans)
    y_scaled = y_scaler.transform(y_trans)

    x_seq, y_seq, date_seq, target_indices = make_sequences(
        x_scaled, y_scaled, dates, seq_len, use_current_day
    )

    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_samples),
    }
    return x_seq, y_seq, date_seq, target_indices, splits, y_scaler, log_transform


def inverse_y(y_scaled: np.ndarray, y_scaler: StandardScalerNP, log_transform: bool) -> np.ndarray:
    y_trans = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    if log_transform:
        y = np.expm1(y_trans)
        y = np.maximum(y, 0.0)
    else:
        y = y_trans
    return y


def make_loader(x_seq, y_seq, start, end, batch_size, shuffle, seed: int = None):
    x = torch.tensor(x_seq[start:end], dtype=torch.float32)
    y = torch.tensor(y_seq[start:end], dtype=torch.float32)
    ds = TensorDataset(x, y)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=generator)


# -----------------------------------------------------------------------------
# 模型定义
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 n_heads: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, 32]
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        # 日流量模拟通常更重视最近时刻状态，因此使用最后一个 token。
        h_last = h[:, -1, :]
        return self.head(h_last)


# -----------------------------------------------------------------------------
# 训练、评估、输出
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float):
    model.train()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total += loss.item() * len(xb)
        count += len(xb)
    return total / max(count, 1)


@torch.no_grad()
def predict_scaled(model, loader, device):
    model.eval()
    preds, obs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        obs.append(yb.numpy())
    return np.vstack(preds).reshape(-1), np.vstack(obs).reshape(-1)


def calc_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = float(1.0 - np.sum((pred - obs) ** 2) / denom) if denom > 0 else np.nan
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs)) if np.sum(obs) != 0 else np.nan
    if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0:
        r = float(np.corrcoef(obs, pred)[0, 1])
    else:
        r = np.nan
    alpha = float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else np.nan
    beta = float(np.mean(pred) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)) if np.isfinite(r) else np.nan
    return {"NSE": nse, "KGE": kge, "RMSE": rmse, "MAE": mae, "PBIAS_percent": pbias, "r": r}


def save_split_info(out_dir: Path, date_seq, target_indices, splits, seq_len: int, use_current_day: bool):
    rows = []
    for split_name, (start, end) in splits.items():
        split_dates = pd.to_datetime(date_seq[start:end])
        split_targets = target_indices[start:end]
        first_target = int(split_targets[0])
        last_target = int(split_targets[-1])
        first_window_start = first_target - seq_len + 1 if use_current_day else first_target - seq_len
        last_window_start = last_target - seq_len + 1 if use_current_day else last_target - seq_len
        rows.append({
            "split": split_name,
            "sample_start": start,
            "sample_end_exclusive": end,
            "n_samples": end - start,
            "target_start_date": split_dates.min(),
            "target_end_date": split_dates.max(),
            "first_target_day_index": first_target,
            "last_target_day_index": last_target,
            "first_input_window_start_day_index": first_window_start,
            "last_input_window_start_day_index": last_window_start,
        })
    pd.DataFrame(rows).to_csv(out_dir / "split_info.csv", index=False, encoding="utf-8-sig")


def save_predictions_and_plot(out_dir: Path, dates, obs, pred, title: str, split_name: str):
    pred_df = pd.DataFrame({"date": pd.to_datetime(dates), "observed_m3_day": obs, "predicted_m3_day": pred})
    pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 4))
    plt.plot(pred_df["date"], pred_df["observed_m3_day"], label="Observed")
    plt.plot(pred_df["date"], pred_df["predicted_m3_day"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Flow (m3/day)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{split_name}_hydrograph.png", dpi=200)
    plt.close()


def main():
    # -------------------------------------------------------------------------
    # 用户需要改的设置都集中放在这里
    # -------------------------------------------------------------------------
    MODEL_NAME = "baseline_transformer"
    MODEL_LABEL = "Transformer baseline"

    DATA_DIR = r"."                   # 数据所在文件夹。IDLE 下建议填绝对路径，例如 r"D:\\GWLF_GNN\\data"
    RUNOFF_FILE = "runoff-Volume.xlsx"
    GROUNDWATER_FILE = "groundwater-Volume.xlsx"
    FLOW_FILE = "Flow.xlsx"
    FLOW_UNIT = "m3/day"              # 如果 Flow.xlsx 仍是 m3/s，则改成 "m3/s"

    OUTPUT_DIR = "results_transformer"
    N_SUBBASINS = 16

    SEQ_LEN = 90                       # 输入窗口长度：默认用当天及前 89 天预测当天流量
    USE_CURRENT_DAY = True             # True: [t-L+1, ..., t] -> Q(t)；False: [t-L, ..., t-1] -> Q(t)
    LOG_TRANSFORM = True               # 对输入和流量做 log1p，通常更稳定

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15                   # TEST_RATIO 自动为 1 - TRAIN_RATIO - VAL_RATIO

    D_MODEL = 96
    NUM_LAYERS = 3
    N_HEADS = 4
    DIM_FEEDFORWARD = 192
    DROPOUT = 0.20
    EPOCHS = 300
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 40
    GRAD_CLIP = 1.0
    SEED = 42

    DATA_DIR = os.environ.get("DATA_DIR", DATA_DIR)
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)
    FLOW_UNIT = os.environ.get("FLOW_UNIT", FLOW_UNIT)
    SEQ_LEN = get_env_int("SEQ_LEN", SEQ_LEN)
    USE_CURRENT_DAY = get_env_bool("USE_CURRENT_DAY", USE_CURRENT_DAY)
    LOG_TRANSFORM = get_env_bool("LOG_TRANSFORM", LOG_TRANSFORM)
    TRAIN_RATIO = get_env_float("TRAIN_RATIO", TRAIN_RATIO)
    VAL_RATIO = get_env_float("VAL_RATIO", VAL_RATIO)
    D_MODEL = get_env_int("D_MODEL", D_MODEL)
    NUM_LAYERS = get_env_int("NUM_LAYERS", NUM_LAYERS)
    N_HEADS = get_env_int("N_HEADS", N_HEADS)
    DIM_FEEDFORWARD = get_env_int("DIM_FEEDFORWARD", DIM_FEEDFORWARD)
    DROPOUT = get_env_float("DROPOUT", DROPOUT)
    EPOCHS = get_env_int("EPOCHS", EPOCHS)
    BATCH_SIZE = get_env_int("BATCH_SIZE", BATCH_SIZE)
    LEARNING_RATE = get_env_float("LEARNING_RATE", LEARNING_RATE)
    WEIGHT_DECAY = get_env_float("WEIGHT_DECAY", WEIGHT_DECAY)
    PATIENCE = get_env_int("PATIENCE", PATIENCE)
    GRAD_CLIP = get_env_float("GRAD_CLIP", GRAD_CLIP)
    SEED = get_env_int("SEED", SEED)
    # -------------------------------------------------------------------------

    set_seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": MODEL_NAME,
        "data_dir": DATA_DIR,
        "runoff_file": RUNOFF_FILE,
        "groundwater_file": GROUNDWATER_FILE,
        "flow_file": FLOW_FILE,
        "flow_unit": FLOW_UNIT,
        "n_subbasins": N_SUBBASINS,
        "seq_len": SEQ_LEN,
        "use_current_day": USE_CURRENT_DAY,
        "log_transform": LOG_TRANSFORM,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "n_heads": N_HEADS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
    }
    pd.DataFrame([config]).to_csv(out_dir / "config.csv", index=False, encoding="utf-8-sig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_all_data(DATA_DIR, RUNOFF_FILE, GROUNDWATER_FILE, FLOW_FILE, FLOW_UNIT, N_SUBBASINS)
    print(f"有效逐日数据量: {len(df)} 天，起止日期: {df['date'].min()} 至 {df['date'].max()}")

    x_seq, y_seq, date_seq, target_indices, splits, y_scaler, log_transform = prepare_lstm_data(
        df, SEQ_LEN, TRAIN_RATIO, VAL_RATIO, USE_CURRENT_DAY, LOG_TRANSFORM, N_SUBBASINS
    )
    print(f"序列样本数: {len(x_seq)}，输入形状: {x_seq.shape}")
    print(f"Train/Val/Test: {splits['train']}, {splits['val']}, {splits['test']}")

    save_split_info(out_dir, date_seq, target_indices, splits, SEQ_LEN, USE_CURRENT_DAY)

    train_loader = make_loader(x_seq, y_seq, *splits["train"], BATCH_SIZE, True, SEED)
    val_loader = make_loader(x_seq, y_seq, *splits["val"], BATCH_SIZE, False)
    test_loader = make_loader(x_seq, y_seq, *splits["test"], BATCH_SIZE, False)

    model = TransformerRegressor(
        input_dim=x_seq.shape[-1],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=SEQ_LEN,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_state = None
    best_epoch = 0
    wait = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, GRAD_CLIP)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
            break

    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    stopped_epoch = int(logs[-1]["epoch"]) if logs else 0
    pd.DataFrame([{
        "model": MODEL_NAME,
        "seed": SEED,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val,
        "early_stopped": stopped_epoch < EPOCHS,
    }]).to_csv(out_dir / "training_summary.csv", index=False, encoding="utf-8-sig")

    metrics_rows = []
    for split_name in ["train", "val", "test"]:
        loader = make_loader(x_seq, y_seq, *splits[split_name], BATCH_SIZE, False)
        pred_s, obs_s = predict_scaled(model, loader, device)
        pred = inverse_y(pred_s, y_scaler, log_transform)
        obs = inverse_y(obs_s, y_scaler, log_transform)
        m = calc_metrics(obs, pred)
        m["model"] = MODEL_NAME
        m["seed"] = SEED
        m["best_epoch"] = best_epoch
        m["stopped_epoch"] = stopped_epoch
        m["split"] = split_name
        metrics_rows.append(m)
        print(split_name, m)
        split_start, split_end = splits[split_name]
        save_predictions_and_plot(out_dir, date_seq[split_start:split_end], obs, pred, f"{MODEL_LABEL} ({split_name})", split_name)

    metrics_df = pd.DataFrame(metrics_rows)[["model", "seed", "split", "best_epoch", "stopped_epoch", "NSE", "KGE", "RMSE", "MAE", "PBIAS_percent", "r"]]
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    print(f"完成。结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
