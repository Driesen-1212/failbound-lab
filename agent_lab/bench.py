# -*- coding: utf-8 -*-
"""实测单次搜索循环耗时，用于判断 12 档扫描（方案 A）是否跑得起。

不写任何结果文件，只输出每个策略的单 run 耗时并外推到 A/B 两种规模。
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import N_INIT, SEED, STRATEGIES, TASKS
from cost_model import cost_vector, gate_pool
from data_io import load_any
from run_experiments import run_one
from strategies import standardize

MAX_BUDGET = 100
TNAME = "T2_bulk_modulus"


def main():
    df, sha = load_any()
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_all = standardize(df[feat_cols].values.astype(float))
    pool = gate_pool(df, use_gate=True)
    idx_map = pool.index.to_numpy()
    X_std = X_all[idx_map]
    cost = cost_vector(pool, use_gate=True)
    task = TASKS[TNAME]
    y = pool[task["target_col"]].values.astype(float)
    print(f"[pool] n={len(pool)} dim={X_std.shape[1]} | sha={sha}")

    per_run = {}
    for strat in STRATEGIES:
        ts = time.perf_counter()
        run_one(X_std, y, cost, task, strat, MAX_BUDGET, SEED, use_gate=True)
        per_run[strat] = time.perf_counter() - ts
        print(f"[time] {strat:<16} {per_run[strat]:.2f}s")

    avg = float(np.mean(list(per_run.values())))
    # 每 run 里最贵的策略决定整批耗时上界（同一批里 4 个策略各占 1/4）
    worst = max(per_run.values())
    batch = sum(per_run.values())          # 1 任务 × 4 策略 × 1 种子

    print()
    print(f"[avg] 单 run {avg:.2f}s | 最慢策略 {worst:.2f}s | "
          f"1 任务×4 策略×1 种子 = {batch:.2f}s")
    for label, n_task in (("B 锚点精简 (3 档)", 3), ("A 全量扫描 (12 档)", 12)):
        for n_seed in (20,):
            total = batch * n_task * n_seed
            print(f"[{label}] {n_task} 任务 × 4 策略 × {n_seed} 种子 = "
                  f"{n_task*4*n_seed} runs → {total/60:.1f} 分钟 "
                  f"({total/3600:.2f} 小时)")


if __name__ == "__main__":
    main()
