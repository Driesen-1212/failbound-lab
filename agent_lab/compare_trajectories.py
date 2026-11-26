# -*- coding: utf-8 -*-
"""对比两个环境的轨迹：找第一个分歧点 + 成本结构分析。"""
import json
import sys

sys.path.insert(0, ".")
from run_experiments import _task_arrays
from cost_model import threshold_of

a = json.load(open("results/traj_default.json", encoding="utf-8"))
b = json.load(open("results/traj_sbstack.json", encoding="utf-8"))

for ca, cb in zip(a["cases"], b["cases"]):
    print(f"\n=== {ca['task']} / {ca['strategy']} / seed={ca['seed_index']} ===")
    print(f"default: n_exp={ca['n_exp']} truncated={ca['truncated_spent']}")
    print(f"sbstack: n_exp={cb['n_exp']} truncated={cb['truncated_spent']}")
    # 第一个分歧点（按 chosen 序列逐点比）
    ch_a, ch_b = ca["chosen"], cb["chosen"]
    n = min(len(ch_a), len(ch_b))
    div = next((k for k in range(n) if ch_a[k] != ch_b[k]), None)
    if div is None:
        print(f"前 {n} 点完全一致（长度不同：{len(ch_a)} vs {len(ch_b)}）")
    else:
        print(f"第 1 个分歧点 = 第 {div} 次选择（0-based）："
              f"default 选 {ch_a[div]}，sbstack 选 {ch_b[div]}")
        # 前缀累计花费
        pa = sum(ca["cost_seq"][:div]); pb = sum(cb["cost_seq"][:div])
        print(f"  分歧前前缀累计：default={pa:.1f} sbstack={pb:.1f}")
        print(f"  该步成本：default={ca['cost_seq'][div]:.3f} "
              f"sbstack={cb['cost_seq'][div]:.3f}")
        # 之后是否重新汇合
        sa, sb = set(ch_a[div:]), set(ch_b[div:])
        print(f"  分歧后集合交集大小：{len(sa & sb)} / {len(sa)} vs {len(sb)}")
    # 成本结构：恰好 200.0 怎么来的
    cs = cb["cost_seq"]
    cum = 0.0
    for k, c in enumerate(cs):
        if cum >= 200:
            break
        cum += c
    print(f"sbstack 截断点：第 {k} 步（n_exp={cb['n_exp']}），累计={cum}")
    print(f"  末 3 步成本：{[round(x,3) for x in cs[max(0,k-3):k+1]]}")
    # 该任务成本分布
    task, X_std, y, cost, thr = _task_arrays(ca["task"], True, "ablate")
    import numpy as np
    print(f"  任务成本分布：min={cost.min():.3f} max={cost.max():.3f} "
          f"mean={cost.mean():.3f} 唯一值数={len(np.unique(cost))}")
    # 200.0 是否任务成本的自然组合
    exact = cost[np.abs(cost - (200 - (cum - cs[k]))) < 1e-9]
    print(f"  是否存在单个成本恰好 = 200-前缀 = {200-(cum-cs[k]):.3f} 的候选点："
          f"{len(exact)} 个")
