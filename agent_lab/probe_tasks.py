# -*- coding: utf-8 -*-
"""探测各真实任务的"空间结构属性"，验证可学习性梯度是否存在（N2 的关键验证）

这是整个方案的生死检查：如果所有任务的 RF 可分性 AUC 都挤在很窄的区间，
就画不出失效边界，必须改走"受控降质构造扫描轴"（C6 方案 B）。

输出每个任务的：
  - 可学习性 AUC（RF 交叉验证，命中标签 = 目标 top-q 分位）
  - 描述符-目标最大 |Spearman|
  - 目标分布峰数
  - 基线命中率（= q，或分类任务的正类比例）
  - 稀有元素占比、n_atoms 分布（成本模型是否真的有区分度）
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from config import TASKS_DIR as TASK_DIR   # 自动定位工程根，不写死绝对路径
QUANTILES = (0.90,)          # 命中 = 目标 top 10%


def hit_labels(y, q):
    thr = np.quantile(y, q)
    return (y >= thr).astype(int)


def probe(path, q=0.90):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(float)
    y = df["target"].values.astype(float)
    name = os.path.basename(path).replace(".csv", "")

    uniq = np.unique(y)
    is_binary = len(uniq) <= 2
    label = (y >= 0.5).astype(int) if is_binary else hit_labels(y, q)
    base_rate = float(label.mean())

    auc = np.nan
    if 0 < base_rate < 1:
        try:
            prob = cross_val_predict(
                RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1),
                X, label, cv=5, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(label, prob))
        except Exception as e:
            print(f"  [warn] {name}: AUC 失败 {e}")

    rs = []
    for j in range(X.shape[1]):
        if X[:, j].std() < 1e-12:
            continue
        c = spearmanr(X[:, j], y).correlation
        if not np.isnan(c):
            rs.append(abs(c))
    r_proxy = float(np.nanmax(rs)) if rs else np.nan

    n_modes = np.nan
    if not is_binary and len(uniq) > 10:
        try:
            kde = gaussian_kde(y[::max(1, len(y) // 500)])
            grid = np.linspace(y.min(), y.max(), 200)
            n_modes = int(len(find_peaks(kde(grid))[0]))
        except Exception:
            pass

    return {"task": name, "n": len(df), "binary": is_binary,
            "base_hit_rate": round(base_rate, 3),
            "separability_auc": round(auc, 3) if not np.isnan(auc) else np.nan,
            "r_proxy": round(r_proxy, 3) if not np.isnan(r_proxy) else np.nan,
            "n_modes": n_modes,
            "y_std": round(float(y.std()), 3),
            "rare_frac": round(float(df["has_rare"].mean()), 3),
            "n_atoms_med": float(df["n_atoms"].median()),
            "n_atoms_max": float(df["n_atoms"].max())}


def main():
    files = sorted(f for f in os.listdir(TASK_DIR) if f.endswith(".csv")
                   and f != "tasks_manifest.csv")
    rows = [probe(os.path.join(TASK_DIR, f)) for f in files]
    out = pd.DataFrame(rows).sort_values("separability_auc")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    aucs = out["separability_auc"].dropna()
    print(f"\n[AUC 跨度] min={aucs.min():.3f}  max={aucs.max():.3f}  "
          f"跨度={aucs.max()-aucs.min():.3f}  中位={aucs.median():.3f}")
    print(f"[判定] {'✅ 天然梯度存在，可直接画边界' if aucs.max()-aucs.min() >= 0.15 else '⚠️ 梯度过窄，需受控降质补档（C6-B）'}")

    out_path = os.path.join(TASK_DIR, "task_probe.csv")
    out.to_csv(out_path, index=False)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
