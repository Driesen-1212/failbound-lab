# -*- coding: utf-8 -*-
"""构造「描述符消融」稳健性臂 v2（2026-10-16，登记于 preregistration.md §10）

动机（当前方案最大的评委攻击面）：
    主扫描轴靠"给真实目标加噪"制造低可学习性任务 → **失效边界的存在性 100% 依赖合成数据**。
    评委一句话就能打过来："你研究的是加噪标签，不是材料发现。"

消融臂走另一条**独立**的路：
    **目标值一个字不动（100% 真实 DFT / 实验数据）**，只削减描述符空间。
    候选池、材料标识、成本模型列（n_atoms / has_rare）全部保持真实。

为什么用"多算子"而不是单一 PCA：
    实测单一 PCA 的 AUC 梯度太窄（k=1,2,3,5 → 0.61/0.80/0.83/0.88），
    低端只有一个点，拟合不出边界。改用**三族削减算子**拼出 0.55–0.90 的覆盖：
      spectral   pca{k}   保留前 k 个主成分（信息量最有序的削减）
      random     rp{k}    随机投影到 k 维（信息量最无序的削减）
      subset     elem{j}  只保留出现频率最高的 j 种元素（化学上"只认识几种元素"）
    三族给出的 (AUC, 增益) 点混在一起拟合边界 —— 若边界与算子族无关，
    就说明失效边界是"可学习性"的函数，而不是某个特定削减手法的产物。
    这本身就是一条额外的稳健性证据。

实测 AUC（3 基底均值，RF 5-fold，top-10% 标签）：
    pca1 0.606 | rp2 0.642 | rp5 0.755 | pca2 0.799 | elem20 0.819 | pca5 0.877
    （full88 = 0.888，作为对照已在主臂）

    ⚠️ 以上是**开发期**（2026-10-16 构造消融臂时）的 3 基底均值，
       用于论证"算子族覆盖到 0.55–0.90"这一设计目标；
       **逐任务最终值以 results/task_props.csv 为准**
       （例：pca2 的最终单任务值 0.769、rp2 为 0.593，与上图均值不同属正常）。
"""

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import ABLATE_DIR, TASKS_DIR

POOL_SIZE = 3000

# (方法名, (族, 参数)) —— 6 个算子 × 3 个真实基底 = 18 个消融任务
OPERATORS = (
    ("pca1", ("pca", 1)),
    ("rp2", ("rp", 2)),
    ("rp5", ("rp", 5)),
    ("pca2", ("pca", 2)),
    ("elem20", ("elem", 20)),
    ("pca5", ("pca", 5)),
)
BASES = ("matbench_expt_gap", "matbench_dielectric", "matbench_log_gvrh")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduce_dims(X_std, kind, k, rng):
    if kind == "pca":
        return PCA(n_components=k, svd_solver="full", random_state=0).fit_transform(X_std)
    if kind == "rp":
        R = rng.normal(0.0, 1.0, size=(X_std.shape[1], k))
        return X_std @ R / np.sqrt(X_std.shape[1])
    if kind == "elem":
        freq = (X_std != 0).mean(axis=0)          # 元素出现频率（原空间非零比例）
        idx = np.argsort(-freq)[:k]
        return X_std[:, idx]
    raise ValueError(kind)


def main():
    os.makedirs(ABLATE_DIR, exist_ok=True)
    rows = []

    for base in BASES:
        df = pd.read_csv(os.path.join(TASKS_DIR, base + ".csv"))
        if len(df) > POOL_SIZE:
            df = df.sample(n=POOL_SIZE, random_state=0).reset_index(drop=True)

        feat_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feat_cols].values.astype(float)
        y = df["target"].values.astype(float)       # ← 真实目标，一个字不动
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        X_std = (X - mu) / sd
        rng = np.random.default_rng(7)              # 随机投影的固定种子

        for op_name, (kind, k) in OPERATORS:
            Z = reduce_dims(X_std, kind, k, rng)

            out = pd.DataFrame({f"f{i}": Z[:, i] for i in range(Z.shape[1])})
            out["target"] = y
            out["material_id"] = [f"{base}-{op_name}-{i:05d}" for i in range(len(out))]
            if "n_atoms" in df.columns:
                out["n_atoms"] = df["n_atoms"].values
            if "has_rare" in df.columns:
                out["has_rare"] = df["has_rare"].values

            name = f"{base}__{op_name}"
            path = os.path.join(ABLATE_DIR, name + ".csv")
            out.to_csv(path, index=False)
            rows.append({"task": name, "base": base, "operator": op_name,
                         "kind": kind, "dim": int(Z.shape[1]),
                         "n": len(out), "csv_sha256": sha256_file(path)})
            print(f"  {name:<34} dim={Z.shape[1]:>3}")

    man = pd.DataFrame(rows)
    man_path = os.path.join(ABLATE_DIR, "ablate_manifest.csv")
    man.to_csv(man_path, index=False)
    print(f"\n[save] {len(man)} 个消融任务 → {ABLATE_DIR}")
    print(f"[manifest] {man_path}")
    return man


if __name__ == "__main__":
    main()
