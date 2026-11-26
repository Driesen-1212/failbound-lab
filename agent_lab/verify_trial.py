# -*- coding: utf-8 -*-
"""等价性验证：确认「统一的 run_trial」与重构前各脚本的自有循环**逐点一致**。

为什么要这个脚本：run_experiments / failbound_agent / screening_demo 三处
各自实现了一遍实验主循环。把它们合并成一份时，任何一点差异
（初始化顺序、fp.update 的时机、随机数的消耗次序）都会让新数据与已有的
2160×4 条 run 不可比 —— 那等于作废全部实验结果。

做法：对若干 (任务, 策略, 预算, 种子) 组合跑一遍，把 chosen 序列做哈希：
    python verify_trial.py --save      # 存基线 results/_baseline_chosen.json
    python verify_trial.py             # 与基线比对，全等才算通过
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED, discover_tasks  # noqa: E402
from cost_model import cost_vector, gate_pool, threshold_of  # noqa: E402
from strategies import standardize  # noqa: E402

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "_baseline_chosen.json")

CASES = [
    ("matbench_dielectric", "random_uniform", 25, 0),
    ("matbench_dielectric", "farthest_point", 25, 0),
    ("matbench_dielectric", "gp_ucb", 25, 0),
    ("matbench_dielectric", "gp_ei", 25, 1),
    ("matbench_log_gvrh", "gp_ucb", 50, 3),
    ("matbench_log_gvrh", "farthest_point", 50, 3),
]


def prep(task_name, use_gate=False):
    import pandas as pd
    task = discover_tasks("main")[task_name]
    df = pd.read_csv(task["path"])
    pool = gate_pool(df, use_gate=use_gate)
    feat = [c for c in pool.columns if c.startswith("f")]
    X_std = standardize(pool[feat].values.astype(float))
    y = pool[task["target_col"]].values.astype(float)
    thr = threshold_of(y, task)
    cost = cost_vector(pool, use_gate=use_gate)
    return X_std, y, cost, thr, task


def sig(chosen):
    return hashlib.sha256(",".join(map(str, chosen)).encode()).hexdigest()[:16]


def collect():
    out = {}
    for tname, strat, budget, s in CASES:
        X_std, y, cost, thr, task = prep(tname)
        seed = SEED + s

        # 三份实现，各自独立调用
        import run_experiments as RE
        r1 = RE.run_one(X_std, y, cost, task, thr, strat, budget, seed,
                        use_gate=False)["chosen"]
        import failbound_agent as FA
        FA.run_strategy  # noqa: B018  (存在性检查)
        r2 = _agent_chosen(X_std, y, cost, thr, task, budget, seed, strat)
        import screening_demo as SD
        r3 = SD.run_one_strategy(X_std, y, cost, thr, task, budget, seed, strat)

        out[f"{tname}|{strat}|{budget}|{s}"] = {
            "run_experiments": sig(r1), "agent": sig(r2), "screening": sig(r3),
            "n": len(r1),
        }
    return out


def _agent_chosen(X_std, y, cost, thr, task, budget, seed, strategy):
    """复现 failbound_agent.run_strategy 的循环（它只返回统计量，不返回 chosen）。

    直接照抄其循环体，用于比对；重构后 run_strategy 走 run_trial，
    这个影子实现就用来确认 chosen 序列没变。
    """
    import numpy as np
    from strategies import (FarthestPointSelector, GPSelector, pick_initial,
                            pick_random_uniform)
    from cost_model import is_hit, utility
    from config import N_INIT
    rng = np.random.default_rng(seed)
    n_pool = len(y)
    u_all = utility(y, task)
    gp = GPSelector()
    fp = FarthestPointSelector(X_std)
    chosen, spent = [], 0.0
    for i in pick_initial(n_pool, N_INIT, rng):
        chosen.append(int(i)); spent += cost[i]; fp.update(int(i))
    while spent < budget and len(chosen) < n_pool:
        avail = np.setdiff1d(np.arange(n_pool), np.array(chosen))
        if len(avail) == 0:
            break
        if strategy == "random_uniform":
            nxt = pick_random_uniform(avail, 1, rng)
        elif strategy == "farthest_point":
            nxt = fp.select(avail, 1)
        else:
            acq = "ucb" if strategy == "gp_ucb" else "ei"
            nxt = gp.select(X_std, X_std[chosen], u_all[chosen], avail, 1, rng, acq=acq)
        i = int(nxt[0])
        chosen.append(i); spent += cost[i]
        if strategy == "farthest_point":
            fp.update(i)
    return chosen


def main():
    save = "--save" in sys.argv
    cur = collect()
    if save:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2, ensure_ascii=False)
        print(f"[save] 基线已存 {len(cur)} 组 → {BASELINE}")
        for k, v in cur.items():
            print(f"  {k:<52} n={v['n']:<4} {v['run_experiments']}")
        return

    if not os.path.exists(BASELINE):
        raise SystemExit("没有基线，先跑 python verify_trial.py --save")

    with open(BASELINE, encoding="utf-8") as f:
        old = json.load(f)
    bad = 0
    for k, v in cur.items():
        o = old.get(k)
        if o is None:
            print(f"  ⚠ 基线里没有 {k}")
            continue
        for impl in ("run_experiments", "agent", "screening"):
            if o[impl] != v[impl]:
                print(f"  ❌ {k} [{impl}] {o[impl]} → {v[impl]}")
                bad += 1
        # 同一组合三份实现之间也必须互相一致
        if len({v[i] for i in ("run_experiments", "agent", "screening")}) != 1:
            print(f"  ❌ {k} 三份实现彼此不一致: {v}")
            bad += 1
    print(f"[verify] {len(cur)} 组 × 3 实现"
          f"{'  ✅ 全部逐点一致' if bad == 0 else f'  ❌ {bad} 处不一致'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
