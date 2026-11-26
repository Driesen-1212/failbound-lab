# -*- coding: utf-8 -*-
"""样本外新基底检验 —— 构造两个「从未进入过冻结包」的基底消融任务。

为什么要做（这是全项目最大的攻击面）：
    消融臂只覆盖 3 个基底（expt_gap / dielectric / log_gvrh），
    "失效边界在基底层"这个结论的独立单位只有 3 个 —— n=3 说不出统计。

做法（不下载任何数据、不改冻结包）：
    从本地已落地但**从未进过消融臂**的两个基底重建同一套 6 个削减算子：
        matbench_log_kvrh     体积模量，与 log_gvrh 同批 10987 个结构（同化学、不同物理量）
        matbench_perovskites  ABO3 钙钛矿，e_form，18928 → 池 3000（**不同化学体系**）
    算子代码与 build_ablation.py 逐行同构（standardize → pca/rp/elem），
    随机投影种子同为 7，池子同为 3000，成本列同为真实的 n_atoms / has_rare。
    另外补一个 full 算子（原样 85 维），作为这两个基底自己的全维参照点。

诚实边界（必须写进报告，不能藏）：
    1. log_kvrh 与 log_gvrh 共享同一批结构 → 化学上不是独立基底，
       只有**目标函数不同**这一点独立性。有效新增化学基底 ≈ 1（perovskites）。
    2. perovskites 的 85 维特征矩阵有效秩只有 54（ABO3 只用到少数元素），
       所以它的 "full" 参照点本身就是低秩的 —— 报告里要标注。

用法：python holdout/build_holdout.py
输出：holdout/data/ablate/*.csv + holdout/ablate_manifest.csv
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "agent_lab"))

from build_ablation import reduce_dims  # noqa: E402  算子实现与冻结包逐行同一份

TASKS = os.path.join(HERE, "data", "tasks")
OUT_DIR = os.path.join(HERE, "data", "ablate")

BASES = ("matbench_log_kvrh", "matbench_perovskites")
OPERATORS = (
    ("pca1", ("pca", 1)),
    ("rp2", ("rp", 2)),
    ("rp5", ("rp", 5)),
    ("pca2", ("pca", 2)),
    ("elem20", ("elem", 20)),
    ("pca5", ("pca", 5)),
    ("full", ("full", 85)),
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for base in BASES:
        df = pd.read_csv(os.path.join(TASKS, base + ".csv"))
        feat_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feat_cols].values.astype(float)
        y = df["target"].values.astype(float)        # 目标一个字不动
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        X_std = (X - mu) / sd
        rng = np.random.default_rng(7)               # 与 build_ablation.py 同种子
        s = np.linalg.svd(X - mu, compute_uv=False)
        rank = int((s > 1e-8).sum())
        print(f"[{base}] n={len(df)} dim={X.shape[1]} rank={rank}")

        for op_name, (kind, k) in OPERATORS:
            Z = X_std if kind == "full" else reduce_dims(X_std, kind, k, rng)
            out = pd.DataFrame({f"f{i}": Z[:, i] for i in range(Z.shape[1])})
            # 全维时列名必须沿用原 f0..f84，否则 downstream 的列序断言会错位
            out["target"] = y
            out["material_id"] = [f"{base}-{op_name}-{i:05d}" for i in range(len(out))]
            out["n_atoms"] = df["n_atoms"].values
            out["has_rare"] = df["has_rare"].values
            name = f"{base}__{op_name}"
            path = os.path.join(OUT_DIR, name + ".csv")
            out.to_csv(path, index=False)
            rows.append({"task": name, "base": base, "operator": op_name,
                         "kind": kind, "dim": int(Z.shape[1]),
                         "n": len(out), "base_rank": rank,
                         "csv_sha256": sha256_file(path)})
            print(f"  {name:<36} dim={Z.shape[1]:>3}")

    man = pd.DataFrame(rows)
    mp = os.path.join(HERE, "ablate_manifest.csv")
    man.to_csv(mp, index=False)
    print(f"\n[save] {len(man)} 个样本外任务 → {OUT_DIR}")
    print(man[["task", "dim", "base_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
