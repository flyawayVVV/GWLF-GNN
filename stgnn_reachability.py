# -*- coding: utf-8 -*-
"""
stgnn_reachability.py

STGNN-reachability：水文可达性河网图版本。
- 节点：16 个子流域 + 1 个虚拟出口 output。
- 边：把所有上游节点连接到其所有可达下游节点，边属性为沿河网累计长度、累计落差、累计传播时间等。
- 输入：每个子流域过去若干天的 GWLF 地表径流 runoff 和地下潜流 groundwater，单位 m3/day。
- 模型：两个并行的时空图网络流，分别处理 runoff 和 groundwater，再融合预测出口逐日流量。
- 实现：纯 PyTorch，不依赖 PyTorch Geometric。

运行方式：
1. 将 WatershedInfo_new.xlsx、runoff-Volume.xlsx、groundwater-Volume.xlsx、Flow.xlsx 放到 DATA_DIR 指定的文件夹。
2. 在 main() 中修改 DATA_DIR、SEQ_LEN、EDGE_WEIGHT_MODE、GNN_LAYERS 等超参数。
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
from collections import defaultdict

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
    df = pd.read_excel(file_path, sheet_name=0)
    date_col = find_date_column(df)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col])
    for sid in range(1, n_subbasins + 1):
        col = find_subbasin_column(df.columns, sid)
        out[f"{prefix}_{sid}"] = pd.to_numeric(df[col], errors="coerce")
    return out


def load_flow_excel(file_path: str, flow_unit: str = "m3/day") -> pd.DataFrame:
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

    def fit(self, x: np.ndarray, axis=0):
        self.mean_ = np.nanmean(x, axis=axis)
        self.std_ = np.nanstd(x, axis=axis)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


def get_target_indices(n_days: int, seq_len: int, use_current_day: bool) -> np.ndarray:
    if use_current_day:
        return np.arange(seq_len - 1, n_days)
    else:
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
# WatershedInfo_new.xlsx 解析与图构建
# -----------------------------------------------------------------------------

def _contains(text, key):
    return key.lower() in str(text).strip().lower()


def parse_node_static_features(info_file: str, n_subbasins: int = 16) -> np.ndarray:
    """读取节点静态属性：local area 和 total upstream area。输出 [17, 2]，最后一行为 output 节点。"""
    df = pd.read_excel(info_file, sheet_name=0)
    cols = list(df.columns)
    ws_col = None
    local_col = None
    upstream_col = None
    for c in cols:
        if _contains(c, "WatershedID"):
            ws_col = c
        if _contains(c, "Local catchment area"):
            local_col = c
        if _contains(c, "Total upstream area"):
            upstream_col = c
    if ws_col is None or local_col is None or upstream_col is None:
        print("警告：没有完整找到节点面积属性，静态节点特征将设为 0。")
        return np.zeros((n_subbasins + 1, 2), dtype=np.float32)

    ids = pd.to_numeric(df[ws_col], errors="coerce")
    static = np.zeros((n_subbasins + 1, 2), dtype=float)
    for sid in range(1, n_subbasins + 1):
        row = df.loc[ids == sid]
        if len(row) == 0:
            continue
        static[sid - 1, 0] = float(pd.to_numeric(row[local_col], errors="coerce").iloc[0])
        static[sid - 1, 1] = float(pd.to_numeric(row[upstream_col], errors="coerce").iloc[0])

    # 对面积做 log1p，并用 16 个真实子流域标准化；output 节点保留为 0，表示无本地产流属性。
    real = np.log1p(np.maximum(static[:n_subbasins], 0.0))
    mean = real.mean(axis=0)
    std = real.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    static_scaled = np.zeros_like(static, dtype=np.float32)
    static_scaled[:n_subbasins] = (real - mean) / std
    static_scaled[n_subbasins] = 0.0
    return static_scaled.astype(np.float32)


def find_column_by_keywords(columns, keywords):
    for c in columns:
        c_text = str(c).strip().lower()
        if all(k.lower() in c_text for k in keywords):
            return c
    raise ValueError(f"没有在列名 {list(columns)} 中找到关键词 {keywords}")


def parse_to_node_value(v):
    text = str(v).strip().lower()
    if text == "output":
        return "output"
    return int(float(v))


def parse_edge_table(info_file: str) -> pd.DataFrame:
    """读取直接边表。兼容 Sheet1 中上半部分是节点表、下半部分是边表的格式。"""
    raw = pd.read_excel(info_file, sheet_name=0, header=None)
    edge_header_row = None
    for i in range(raw.shape[0]):
        values = [str(v).strip() for v in raw.iloc[i].values]
        if "Edge" in values and "FromWatershedID" in values and "ToWatershedID" in values:
            edge_header_row = i
            break
    if edge_header_row is None:
        raise ValueError("没有在 WatershedInfo_new.xlsx 中找到边表标题行 Edge / FromWatershedID / ToWatershedID。")

    header = [str(v).strip() for v in raw.iloc[edge_header_row].values]
    edge_df = raw.iloc[edge_header_row + 1:].copy()
    edge_df.columns = header
    edge_col = find_column_by_keywords(edge_df.columns, ["Edge"])
    from_col = find_column_by_keywords(edge_df.columns, ["FromWatershedID"])
    to_col = find_column_by_keywords(edge_df.columns, ["ToWatershedID"])

    # 中文全角括号不影响关键词查找。
    length_col = None
    drop_col = None
    sinuosity_col = None
    for c in edge_df.columns:
        c_text = str(c).lower()
        if "length" in c_text:
            length_col = c
        if "elevation" in c_text or "drop" in c_text:
            drop_col = c
        if "sinuosity" in c_text:
            sinuosity_col = c
    if length_col is None or drop_col is None or sinuosity_col is None:
        raise ValueError("边表中必须包含 Length、Elevation drop 和 Sinuosity 列。")

    records = []
    for _, row in edge_df.iterrows():
        if pd.isna(row[edge_col]) or pd.isna(row[from_col]) or pd.isna(row[to_col]):
            continue
        try:
            edge_id = int(float(row[edge_col]))
            from_ws = int(float(row[from_col]))
            to_ws = parse_to_node_value(row[to_col])
            length_m = float(row[length_col])
            drop_m = float(row[drop_col])
            sinuosity = float(row[sinuosity_col])
        except Exception:
            continue
        records.append({
            "edge_id": edge_id,
            "from": from_ws,
            "to": to_ws,
            "length_m": length_m,
            "drop_m": drop_m,
            "sinuosity": sinuosity,
        })
    if not records:
        raise ValueError("边表解析失败，没有读到任何有效边。")
    return pd.DataFrame(records)


def add_hydraulic_edge_features(edge_df: pd.DataFrame, velocity_m_per_s: float = 1.0) -> pd.DataFrame:
    out = edge_df.copy()
    out["length_m"] = out["length_m"].clip(lower=1.0)
    out["drop_m"] = out["drop_m"].clip(lower=0.0)
    out["slope"] = (out["drop_m"] / out["length_m"]).clip(lower=1e-8)
    out["time_days_v1"] = out["length_m"] / max(velocity_m_per_s, 1e-6) / 86400.0
    # Kirpich 公式常用于小流域汇流时间估计。这里不把它视为精确旅行时间，只作为相对传播时间属性。
    # Tc(min) = 0.01947 * L(m)^0.77 * S^-0.385
    out["time_days_kirpich"] = 0.01947 * (out["length_m"] ** 0.77) * (out["slope"] ** -0.385) / 60.0 / 24.0
    return out


def node_to_index(node, n_subbasins: int = 16):
    if str(node).strip().lower() == "output":
        return n_subbasins
    node = int(node)
    if not (1 <= node <= n_subbasins):
        raise ValueError(f"节点 {node} 不在 1~{n_subbasins} 或 output 范围内。")
    return node - 1


def build_edge_tensors(edge_df: pd.DataFrame,
                       edge_weight_mode: str,
                       n_subbasins: int = 16):
    edge_df = add_hydraulic_edge_features(edge_df)

    src = [node_to_index(v, n_subbasins) for v in edge_df["from"].values]
    dst = [node_to_index(v, n_subbasins) for v in edge_df["to"].values]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    raw_features = np.column_stack([
        np.log1p(edge_df["length_m"].values),
        np.log1p(edge_df["drop_m"].values),
        edge_df["sinuosity"].values,
        edge_df["slope"].values,
        np.log1p(edge_df["time_days_v1"].values),
        np.log1p(edge_df["time_days_kirpich"].values),
    ]).astype(np.float32)
    feat_mean = raw_features.mean(axis=0, keepdims=True)
    feat_std = raw_features.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    edge_attr = torch.tensor((raw_features - feat_mean) / feat_std, dtype=torch.float32)

    mode = edge_weight_mode.lower()
    if mode == "binary":
        w = np.ones(len(edge_df), dtype=np.float32)
    elif mode == "inv_length":
        w = 1.0 / np.maximum(edge_df["length_m"].values, 1.0)
    elif mode == "inv_time_v1":
        w = 1.0 / np.maximum(edge_df["time_days_v1"].values, 1e-6)
    elif mode == "inv_time_kirpich":
        w = 1.0 / np.maximum(edge_df["time_days_kirpich"].values, 1e-6)
    else:
        raise ValueError("EDGE_WEIGHT_MODE 必须是 binary, inv_length, inv_time_v1, inv_time_kirpich 之一。")

    # 归一化到均值为 1，避免权重尺度影响训练稳定性。
    w = w / np.mean(w)
    edge_weight = torch.tensor(w.astype(np.float32), dtype=torch.float32)
    return edge_index, edge_attr, edge_weight, edge_df


def build_reachability_edge_table(direct_edge_df: pd.DataFrame,
                                  n_subbasins: int = 16) -> pd.DataFrame:
    """
    从直接河网边构造可达性图：
    若 j 位于 i 的任意下游路径上，则加入 i -> j。
    length/drop/time 等属性按路径累计；sinuosity 采用长度加权平均。
    """
    direct_edge_df = add_hydraulic_edge_features(direct_edge_df)
    adj = defaultdict(list)
    for _, row in direct_edge_df.iterrows():
        adj[row["from"]].append(row.to_dict())

    records = []

    def dfs(start, current, cum_len, cum_drop, cum_sinu_len, steps, visited):
        for e in adj.get(current, []):
            nxt = e["to"]
            if nxt in visited:
                continue
            new_len = cum_len + float(e["length_m"])
            new_drop = cum_drop + float(e["drop_m"])
            new_sinu_len = cum_sinu_len + float(e["sinuosity"]) * float(e["length_m"])
            new_steps = steps + 1
            avg_sinuosity = new_sinu_len / max(new_len, 1.0)
            records.append({
                "edge_id": len(records) + 1,
                "from": start,
                "to": nxt,
                "length_m": new_len,
                "drop_m": new_drop,
                "sinuosity": avg_sinuosity,
                "path_steps": new_steps,
            })
            if str(nxt).strip().lower() != "output":
                dfs(start, nxt, new_len, new_drop, new_sinu_len, new_steps, visited | {nxt})

    for sid in range(1, n_subbasins + 1):
        dfs(sid, sid, 0.0, 0.0, 0.0, 0, {sid})

    if not records:
        raise ValueError("可达性图构建失败，没有产生任何边。")
    return pd.DataFrame(records)


def build_reachability_graph(info_file: str,
                             edge_weight_mode: str,
                             n_subbasins: int = 16):
    direct_edges = parse_edge_table(info_file)
    reachability_edges = build_reachability_edge_table(direct_edges, n_subbasins)
    edge_index, edge_attr, edge_weight, used_df = build_edge_tensors(reachability_edges, edge_weight_mode, n_subbasins)
    return edge_index, edge_attr, edge_weight, used_df


# -----------------------------------------------------------------------------
# 数据准备
# -----------------------------------------------------------------------------

def prepare_graph_data(df: pd.DataFrame,
                       info_file: str,
                       seq_len: int,
                       train_ratio: float,
                       val_ratio: float,
                       use_current_day: bool,
                       log_transform: bool,
                       n_subbasins: int = 16,
                       scale_mode: str = "per_node",
                       use_mass_shortcut: bool = True):
    runoff_cols = [f"runoff_{i}" for i in range(1, n_subbasins + 1)]
    gw_cols = [f"groundwater_{i}" for i in range(1, n_subbasins + 1)]

    runoff = df[runoff_cols].values.astype(float)
    groundwater = df[gw_cols].values.astype(float)
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
    last_train_target_idx = int(target_indices_all[n_train - 1])

    dyn_raw = np.stack([runoff, groundwater], axis=-1)  # [T, 16, 2]
    if log_transform:
        dyn_trans = np.log1p(np.maximum(dyn_raw, 0.0))
        y_trans = np.log1p(np.maximum(y_raw, 0.0))
    else:
        dyn_trans = dyn_raw.copy()
        y_trans = y_raw.copy()

    scale_mode = scale_mode.lower()
    if scale_mode == "global":
        train_dyn_flat = dyn_trans[:last_train_target_idx + 1].reshape(-1, 2)
        dyn_mean = train_dyn_flat.mean(axis=0).reshape(1, 1, 2)
        dyn_std = train_dyn_flat.std(axis=0).reshape(1, 1, 2)
    elif scale_mode == "per_node":
        dyn_mean = dyn_trans[:last_train_target_idx + 1].mean(axis=0, keepdims=True)
        dyn_std = dyn_trans[:last_train_target_idx + 1].std(axis=0, keepdims=True)
    else:
        raise ValueError("GRAPH_SCALE_MODE 必须是 per_node 或 global。")
    dyn_std = np.where(dyn_std < 1e-8, 1.0, dyn_std)
    dyn_scaled = (dyn_trans - dyn_mean) / dyn_std

    y_scaler = StandardScalerNP().fit(y_trans[:last_train_target_idx + 1], axis=0)
    y_scaled = y_scaler.transform(y_trans)

    static_scaled = parse_node_static_features(info_file, n_subbasins)  # [17, 2]
    n_days = len(df)
    n_nodes = n_subbasins + 1
    node_feature_dim = 6 if use_mass_shortcut else 4
    x_full = np.zeros((n_days, n_nodes, node_feature_dim), dtype=np.float32)
    x_full[:, :n_subbasins, 0:2] = dyn_scaled.astype(np.float32)

    # output 节点是虚拟汇出口，没有本地产流；在标准化空间中设为中性 0。
    x_full[:, n_subbasins, 0:2] = 0.0

    # 静态节点属性复制到所有日期。
    x_full[:, :, 2:4] = static_scaled.reshape(1, n_nodes, 2)

    if use_mass_shortcut:
        total_raw = np.stack([runoff.sum(axis=1), groundwater.sum(axis=1)], axis=-1)
        if log_transform:
            total_trans = np.log1p(np.maximum(total_raw, 0.0))
        else:
            total_trans = total_raw.copy()
        total_scaler = StandardScalerNP().fit(total_trans[:last_train_target_idx + 1], axis=0)
        total_scaled = total_scaler.transform(total_trans).astype(np.float32)
        x_full[:, :, 4:6] = total_scaled.reshape(n_days, 1, 2)

    x_seq, y_seq, date_seq, target_indices = make_sequences(
        x_full, y_scaled, dates, seq_len, use_current_day
    )
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_samples),
    }
    return x_seq, y_seq, date_seq, target_indices, splits, y_scaler, log_transform


# -----------------------------------------------------------------------------
# 模型定义：双水源通道 STGNN
# -----------------------------------------------------------------------------

class EdgeAwareGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        # x: [batch_like, n_nodes, in_dim]
        src = edge_index[0]
        dst = edge_index[1]
        h_self = self.lin_self(x)
        msg = self.lin_msg(x[:, src, :])  # [B, E, out_dim]
        gate = self.edge_gate(edge_attr).unsqueeze(0)  # [1, E, out_dim]
        msg = msg * gate * edge_weight.view(1, -1, 1)

        agg = torch.zeros_like(h_self)
        agg.index_add_(1, dst, msg)

        denom = torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, edge_weight.to(dtype=x.dtype))
        denom = denom.clamp_min(1e-6).view(1, -1, 1)
        agg = agg / denom

        out = h_self + agg
        out = self.norm(out)
        out = torch.relu(out)
        return self.dropout(out)


class GraphTemporalStream(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 edge_dim: int,
                 gnn_layers: int,
                 dropout: float,
                 output_node_index: int):
        super().__init__()
        layers = []
        for i in range(gnn_layers):
            layers.append(EdgeAwareGraphConv(input_dim if i == 0 else hidden_dim, hidden_dim, edge_dim, dropout))
        self.layers = nn.ModuleList(layers)
        self.outlet_proj = nn.Linear(hidden_dim * gnn_layers, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_node_index = output_node_index

    def forward(self, x_stream, edge_index, edge_attr, edge_weight):
        # x_stream: [B, L, N, F]
        B, L, N, F = x_stream.shape
        h = x_stream.reshape(B * L, N, F)
        outlet_states = []
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, edge_weight)
            outlet_states.append(h[:, self.output_node_index, :])
        outlet_h = torch.cat(outlet_states, dim=-1)
        outlet_h = self.outlet_proj(outlet_h).reshape(B, L, -1)
        _, hn = self.gru(outlet_h)
        return hn[-1]


class DualStreamSTGNN(nn.Module):
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int,
                 edge_dim: int,
                 gnn_layers: int,
                 dropout: float,
                 output_node_index: int,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 edge_weight: torch.Tensor,
                 use_mass_shortcut: bool = True):
        super().__init__()
        # x_full 的特征为 [runoff, groundwater, local_area, upstream_area]。
        # runoff stream 使用 [runoff, local_area, upstream_area]。
        # groundwater stream 使用 [groundwater, local_area, upstream_area]。
        stream_input_dim = 3
        self.use_mass_shortcut = use_mass_shortcut
        self.runoff_stream = GraphTemporalStream(stream_input_dim, hidden_dim, edge_dim, gnn_layers, dropout, output_node_index)
        self.groundwater_stream = GraphTemporalStream(stream_input_dim, hidden_dim, edge_dim, gnn_layers, dropout, output_node_index)
        if self.use_mass_shortcut:
            self.mass_gru = nn.GRU(2, hidden_dim, num_layers=1, batch_first=True)
        head_dim = hidden_dim * (4 if self.use_mass_shortcut else 3)
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)
        self.register_buffer("edge_weight", edge_weight)

    def forward(self, x):
        # x: [B, L, N, 4]
        runoff_x = torch.stack([x[..., 0], x[..., 2], x[..., 3]], dim=-1)
        groundwater_x = torch.stack([x[..., 1], x[..., 2], x[..., 3]], dim=-1)
        hr = self.runoff_stream(runoff_x, self.edge_index, self.edge_attr, self.edge_weight)
        hg = self.groundwater_stream(groundwater_x, self.edge_index, self.edge_attr, self.edge_weight)
        pieces = [hr, hg, hr * hg]
        if self.use_mass_shortcut:
            if x.size(-1) >= 6:
                mass_x = x[:, :, 0, 4:6]
            else:
                mass_x = torch.zeros(x.size(0), x.size(1), 2, device=x.device, dtype=x.dtype)
            _, hm = self.mass_gru(mass_x)
            pieces.append(hm[-1])
        h = torch.cat(pieces, dim=-1)
        return self.head(h)


class HybridHydroLoss(nn.Module):
    def __init__(self, bias_weight: float = 0.05, std_weight: float = 0.02):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bias_weight = bias_weight
        self.std_weight = std_weight

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        bias_penalty = (pred.mean() - target.mean()).pow(2)
        pred_std = pred.reshape(-1).std(unbiased=False)
        target_std = target.reshape(-1).std(unbiased=False)
        std_penalty = (pred_std - target_std).pow(2)
        return mse + self.bias_weight * bias_penalty + self.std_weight * std_penalty


def make_criterion(loss_mode: str, bias_weight: float, std_weight: float):
    if loss_mode.lower() == "mse":
        return nn.MSELoss()
    if loss_mode.lower() == "hybrid":
        return HybridHydroLoss(bias_weight=bias_weight, std_weight=std_weight)
    raise ValueError("LOSS_MODE 必须是 mse 或 hybrid。")


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
    MODEL_NAME = "stgnn_reachability"
    MODEL_LABEL = "STGNN-reachability"

    DATA_DIR = r"."                   # 数据所在文件夹。IDLE 下建议填绝对路径，例如 r"D:\\GWLF_GNN\\data"
    INFO_FILE = "WatershedInfo_new.xlsx"
    RUNOFF_FILE = "runoff-Volume.xlsx"
    GROUNDWATER_FILE = "groundwater-Volume.xlsx"
    FLOW_FILE = "Flow.xlsx"
    FLOW_UNIT = "m3/day"              # 如果 Flow.xlsx 仍是 m3/s，则改成 "m3/s"

    OUTPUT_DIR = "results_stgnn_reachability"
    N_SUBBASINS = 16

    SEQ_LEN = 90                       # 输入窗口长度：默认用当天及前 89 天预测当天流量
    USE_CURRENT_DAY = True             # True: [t-L+1, ..., t] -> Q(t)；False: [t-L, ..., t-1] -> Q(t)
    LOG_TRANSFORM = True
    GRAPH_SCALE_MODE = "per_node"       # per_node 与基准模型一样逐子流域标准化；global 为旧版本
    USE_MASS_SHORTCUT = True            # 加入全流域总 runoff/groundwater 的质量通量时间通道

    # 边消融实验主要改这里：binary, inv_length, inv_time_v1, inv_time_kirpich
    EDGE_WEIGHT_MODE = "inv_time_v1"

    # reachability 图已经把每个上游节点连接到所有可达下游节点，1~2 层通常足够。
    GNN_LAYERS = 1
    HIDDEN_DIM = 32
    DROPOUT = 0.40

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    EPOCHS = 180
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-5
    PATIENCE = 35
    MIN_DELTA = 1e-5
    LOSS_MODE = "mse"                   # mse 或 hybrid；hybrid 额外约束批量均值和变幅
    BIAS_LOSS_WEIGHT = 0.05
    STD_LOSS_WEIGHT = 0.02
    LR_PATIENCE = 10
    LR_FACTOR = 0.5
    MIN_LR = 1e-5
    GRAD_CLIP = 1.0
    SEED = 42

    DATA_DIR = os.environ.get("DATA_DIR", DATA_DIR)
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)
    FLOW_UNIT = os.environ.get("FLOW_UNIT", FLOW_UNIT)
    SEQ_LEN = get_env_int("SEQ_LEN", SEQ_LEN)
    USE_CURRENT_DAY = get_env_bool("USE_CURRENT_DAY", USE_CURRENT_DAY)
    LOG_TRANSFORM = get_env_bool("LOG_TRANSFORM", LOG_TRANSFORM)
    GRAPH_SCALE_MODE = os.environ.get("GRAPH_SCALE_MODE", GRAPH_SCALE_MODE)
    USE_MASS_SHORTCUT = get_env_bool("USE_MASS_SHORTCUT", USE_MASS_SHORTCUT)
    EDGE_WEIGHT_MODE = os.environ.get("EDGE_WEIGHT_MODE", EDGE_WEIGHT_MODE)
    GNN_LAYERS = get_env_int("GNN_LAYERS", GNN_LAYERS)
    HIDDEN_DIM = get_env_int("HIDDEN_DIM", HIDDEN_DIM)
    DROPOUT = get_env_float("DROPOUT", DROPOUT)
    TRAIN_RATIO = get_env_float("TRAIN_RATIO", TRAIN_RATIO)
    VAL_RATIO = get_env_float("VAL_RATIO", VAL_RATIO)
    EPOCHS = get_env_int("EPOCHS", EPOCHS)
    BATCH_SIZE = get_env_int("BATCH_SIZE", BATCH_SIZE)
    LEARNING_RATE = get_env_float("LEARNING_RATE", LEARNING_RATE)
    WEIGHT_DECAY = get_env_float("WEIGHT_DECAY", WEIGHT_DECAY)
    PATIENCE = get_env_int("PATIENCE", PATIENCE)
    MIN_DELTA = get_env_float("MIN_DELTA", MIN_DELTA)
    LOSS_MODE = os.environ.get("LOSS_MODE", LOSS_MODE)
    BIAS_LOSS_WEIGHT = get_env_float("BIAS_LOSS_WEIGHT", BIAS_LOSS_WEIGHT)
    STD_LOSS_WEIGHT = get_env_float("STD_LOSS_WEIGHT", STD_LOSS_WEIGHT)
    LR_PATIENCE = get_env_int("LR_PATIENCE", LR_PATIENCE)
    LR_FACTOR = get_env_float("LR_FACTOR", LR_FACTOR)
    MIN_LR = get_env_float("MIN_LR", MIN_LR)
    GRAD_CLIP = get_env_float("GRAD_CLIP", GRAD_CLIP)
    SEED = get_env_int("SEED", SEED)
    # -------------------------------------------------------------------------

    set_seed(SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": MODEL_NAME,
        "data_dir": DATA_DIR,
        "info_file": INFO_FILE,
        "runoff_file": RUNOFF_FILE,
        "groundwater_file": GROUNDWATER_FILE,
        "flow_file": FLOW_FILE,
        "flow_unit": FLOW_UNIT,
        "n_subbasins": N_SUBBASINS,
        "seq_len": SEQ_LEN,
        "use_current_day": USE_CURRENT_DAY,
        "log_transform": LOG_TRANSFORM,
        "graph_scale_mode": GRAPH_SCALE_MODE,
        "use_mass_shortcut": USE_MASS_SHORTCUT,
        "edge_weight_mode": EDGE_WEIGHT_MODE,
        "gnn_layers": GNN_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "loss_mode": LOSS_MODE,
        "bias_loss_weight": BIAS_LOSS_WEIGHT,
        "std_loss_weight": STD_LOSS_WEIGHT,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "min_lr": MIN_LR,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
    }
    pd.DataFrame([config]).to_csv(out_dir / "config.csv", index=False, encoding="utf-8-sig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    info_path = os.path.join(DATA_DIR, INFO_FILE)
    df = load_all_data(DATA_DIR, RUNOFF_FILE, GROUNDWATER_FILE, FLOW_FILE, FLOW_UNIT, N_SUBBASINS)
    print(f"有效逐日数据量: {len(df)} 天，起止日期: {df['date'].min()} 至 {df['date'].max()}")

    edge_index, edge_attr, edge_weight, edge_df_used = build_reachability_graph(info_path, EDGE_WEIGHT_MODE, N_SUBBASINS)
    edge_df_used.to_csv(out_dir / "edge_table_used.csv", index=False, encoding="utf-8-sig")
    print(f"Reachability graph: {edge_index.shape[1]} 条边，EDGE_WEIGHT_MODE={EDGE_WEIGHT_MODE}")

    x_seq, y_seq, date_seq, target_indices, splits, y_scaler, log_transform = prepare_graph_data(
        df, info_path, SEQ_LEN, TRAIN_RATIO, VAL_RATIO, USE_CURRENT_DAY, LOG_TRANSFORM,
        N_SUBBASINS, GRAPH_SCALE_MODE, USE_MASS_SHORTCUT
    )
    print(f"序列样本数: {len(x_seq)}，输入形状: {x_seq.shape}")
    print(f"Train/Val/Test: {splits['train']}, {splits['val']}, {splits['test']}")

    save_split_info(out_dir, date_seq, target_indices, splits, SEQ_LEN, USE_CURRENT_DAY)

    train_loader = make_loader(x_seq, y_seq, *splits["train"], BATCH_SIZE, True, SEED)
    val_loader = make_loader(x_seq, y_seq, *splits["val"], BATCH_SIZE, False)
    test_loader = make_loader(x_seq, y_seq, *splits["test"], BATCH_SIZE, False)

    model = DualStreamSTGNN(
        node_feature_dim=x_seq.shape[-1],
        hidden_dim=HIDDEN_DIM,
        edge_dim=edge_attr.shape[-1],
        gnn_layers=GNN_LAYERS,
        dropout=DROPOUT,
        output_node_index=N_SUBBASINS,
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        edge_weight=edge_weight.to(device),
        use_mass_shortcut=USE_MASS_SHORTCUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    criterion = make_criterion(LOSS_MODE, BIAS_LOSS_WEIGHT, STD_LOSS_WEIGHT)

    best_val = math.inf
    best_state = None
    best_epoch = 0
    wait = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, GRAD_CLIP)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}")

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
        save_predictions_and_plot(out_dir, date_seq[split_start:split_end], obs, pred, f"{MODEL_LABEL} ({EDGE_WEIGHT_MODE}, {split_name})", split_name)

    metrics_df = pd.DataFrame(metrics_rows)[["model", "seed", "split", "best_epoch", "stopped_epoch", "NSE", "KGE", "RMSE", "MAE", "PBIAS_percent", "r"]]
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    print(f"完成。结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
