# -*- coding: utf-8 -*-
"""数据接入：真实 CSV / JARVIS-DFT 快照 / 合成 demo 数据（自动降级）

设计原则：不要赌沙箱联网。真实数据到位就用真实数据，没到位就用合成数据
先把整条 pipeline 跑通（对应方案 §13 降级路径）。
"""

import hashlib
import os

import numpy as np
import pandas as pd


def snapshot_hash(path):
    """数据快照哈希 —— 用于论文与提交物中的可复现性声明"""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_synthetic(n=3000, d=20, seed=0):
    """生成含三个目标任务列的合成数据。

    - bandgap:      中等相关（BO 应有增益）
    - bulk_modulus: 高相关（BO 强增益）
    - synthetic_weak: 弱相关 + 多峰（人为构造的失效区，对应 T3）
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))

    def _target(r, n_modes=1):
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        signal = X @ w
        noise = rng.normal(size=n)
        y = r * signal + np.sqrt(max(1e-9, 1 - r * r)) * noise
        if n_modes > 1:
            k = n // n_modes
            for i in range(1, n_modes):
                y[i * k:(i + 1) * k] += 3.0 * i
        y = (y - y.mean()) / (y.std() + 1e-9)
        return y

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["bandgap"] = _target(r=0.55) * 0.5 + 1.2           # T1
    df["bulk_modulus"] = _target(r=0.75) * 60 + 150       # T2
    df["synthetic_weak"] = _target(r=0.05, n_modes=3)     # T3：弱相关 + 三峰

    # 成本模型所需列（真实数据接入时替换为真实字段）
    df["n_atoms"] = rng.integers(5, 60, size=n)
    df["has_rare"] = rng.random(n) < 0.08
    df["material_id"] = [f"demo-{i:05d}" for i in range(n)]
    return df


def load_any(path=None, save_if_generated=True):
    """优先读真实数据；不存在则生成合成数据并落盘（保证后续可复现）"""
    path = path or os.path.join(os.path.dirname(__file__), "..", "data", "materials.csv")
    path = os.path.normpath(path)

    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[data] 读取真实数据: {path} | n={len(df)} | sha256={snapshot_hash(path)}")
        return df, snapshot_hash(path)

    print(f"[data] 未找到 {path}，使用合成 demo 数据（仅用于跑通 pipeline）")
    df = make_synthetic(seed=0)
    if save_if_generated:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    return df, snapshot_hash(path)
