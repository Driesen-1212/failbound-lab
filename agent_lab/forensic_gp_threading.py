# -*- coding: utf-8 -*-
"""取证测试：gate_ablate 组 3 个分歧 run 的浮点环境敏感性。

背景：本地 gate_ablate 有 3 个 run 截断到 spent=205，沙箱承诺 max=204
（即沙箱这几个 run 的成本轨迹与本地不同）。分歧全部发生在 BO 策略
（gp_ei/gp_ucb），RNG 类策略（random_uniform/lhs_stratified）零分歧——
指向 GP 超参优化的浮点环境差异（库版本 / BLAS 线程），而非代码或数据。

本测试在本地用**同一份代码、同一份数据、同一个种子**复现这 3 个 run，
只改变 BLAS 线程数（threadpoolctl），观察 truncated spent 与 chosen
序列是否翻转。若翻转 → 证明这是环境级 FP 差异，两环境结果都"正确"，
差异粒度在浮点非结合性层面，不在代码/数据层面。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import SEED, discover_tasks
from cost_model import gate_pool, threshold_of
from run_experiments import _task_arrays, derive_seed
from trial import run_trial

# (任务, 策略, seed_index) —— 本地截断到 205 的三个 run
CASES = [("matbench_dielectric__pca2", "gp_ei", 9),
         ("matbench_expt_gap__rp2", "gp_ucb", 17),
         ("matbench_log_gvrh__elem20", "gp_ei", 9)]

BUDGET = 200


def truncate_spent(cost_seq, budget=BUDGET):
    spent = 0.0
    for c in cost_seq:
        if spent >= budget:
            break
        spent += c
    return spent


def run_once(tname, strategy, s, n_threads):
    """在指定 BLAS 线程数下跑一次，返回 (chosen, truncated_spent)。"""
    from threadpoolctl import threadpool_limits
    task, X_std, y, cost, thr = _task_arrays(tname, True, "ablate")
    seed = derive_seed(SEED, tname, strategy, s)
    with threadpool_limits(limits=n_threads):
        r = run_trial(X_std, y, cost, thr, task, BUDGET, seed, strategy)
    return r["chosen"], truncate_spent(r["cost_seq"])


def main():
    print("=== 取证：3 个分歧 run 的浮点环境敏感性 ===\n")
    for tname, strategy, s in CASES:
        print(f"--- {tname} / {strategy} / seed={s} ---")
        base = None
        for nt in (1, 2, 4, 8):
            chosen, spent = run_once(tname, strategy, s, nt)
            if base is None:
                base = (chosen, spent)
            same = (chosen == base[0])
            print(f"  threads={nt}: truncated_spent={spent:6.1f}  "
                  f"chosen与threads=1{'一致' if same else '❌不一致'}"
                  f"{'' if same else f'（len {len(base[0])}→{len(chosen)}）'}")
        print()


if __name__ == "__main__":
    main()
