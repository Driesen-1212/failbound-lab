# -*- coding: utf-8 -*-
"""单次实验循环 —— **全项目唯一一份**。

🔴 为什么要抽出来（2026-10-17 重构）：
    run_experiments.run_one / failbound_agent.run_strategy /
    screening_demo.run_one_strategy 各自写了一遍几乎逐行相同的 while 循环，
    只有返回字段不同。三份拷贝 = 三个可能悄悄漂移的地方。最典型的一处隐患：
        run_experiments 用 `GPSelector(kappa=GP_KAPPA)`
        failbound_agent 与 screening_demo 用 `GPSelector()`（默认 kappa）
    今天 config.GP_KAPPA 恰好等于默认值 2.0，所以没出事；但只要有人调 kappa，
    **Agent 的校准数字就会和主实验脱钩**，而这种错误不会报任何异常。

✅ 现在三处共用本文件的 run_trial()，参数只有一处来源，不可能再脱钩。
   已用 verify_trial.py 证明：重构前后 6 组 × 3 实现的 chosen 序列逐点一致。

统计口径的每一步都不能动，所以这里严格保留原有顺序：
    初始化 → 逐个加入（含最远点集合更新）→ while 循环 → 每步更新 → 记命中
"""

import numpy as np

from config import GP_KAPPA, N_INIT
from cost_model import is_hit, utility
from strategies import (FarthestPointSelector, GPSelector, pick_initial,
                        pick_random_uniform)


def run_trial(X_std, y, cost, thr, task, budget, seed, strategy, n_init=None):
    """跑一次完整实验，返回 chosen 序列与派生统计量。

    参数
    ----
    X_std : (n_pool, d) 标准化后的描述符
    y     : (n_pool,)   真实目标值
    cost  : (n_pool,)   每个候选的实验成本
    thr   : float       命中阈值（top-q 分位）
    task  : dict        任务元信息（utility / is_hit 需要）
    budget: float       预算上限（while spent < budget）
    seed  : int         随机种子
    strategy: str       random_uniform | farthest_point | gp_ucb | gp_ei

    返回
    ----
    dict: chosen / hits / y_seq / cost_seq / n_hits / n_exp / spent /
          hit_rate / ttfh（ttfh=None 表示从未命中）
    """
    if strategy not in ("random_uniform", "farthest_point", "gp_ucb", "gp_ei"):
        raise ValueError(f"未知策略 {strategy}")

    n_init = N_INIT if n_init is None else n_init
    rng = np.random.default_rng(seed)
    n_pool = len(y)
    u_all = utility(y, task)
    # ⚠️ kappa 必须来自 config，不能省略成默认值 —— 见文件头说明
    gp_sel = GPSelector(kappa=GP_KAPPA)
    fp_sel = FarthestPointSelector(X_std)

    chosen, spent = [], 0.0
    init = pick_initial(n_pool, n_init, rng)
    for i in init:
        chosen.append(int(i))
        spent += cost[i]
        fp_sel.update(int(i))          # 初始点也要进入已选集合

    while spent < budget and len(chosen) < n_pool:
        avail = np.setdiff1d(np.arange(n_pool), np.array(chosen))
        if len(avail) == 0:
            break

        if strategy == "random_uniform":
            nxt = pick_random_uniform(avail, 1, rng)
        elif strategy == "farthest_point":
            nxt = fp_sel.select(avail, 1)
        else:
            acq = "ucb" if strategy == "gp_ucb" else "ei"
            nxt = gp_sel.select(X_std, X_std[chosen], u_all[chosen], avail, 1,
                                rng, acq=acq)

        i = int(nxt[0])
        chosen.append(i)
        spent += cost[i]
        if strategy == "farthest_point":
            fp_sel.update(i)

    y_seq = [float(y[i]) for i in chosen]
    hits = [int(is_hit(v, task, thr)) for v in y_seq]
    cost_seq = [float(cost[i]) for i in chosen]
    ttfh = next((k + 1 for k, h in enumerate(hits) if h), None)

    return {
        "chosen": chosen, "y_seq": y_seq, "hits": hits, "cost_seq": cost_seq,
        "n_hits": int(sum(hits)), "ttfh": ttfh, "n_exp": len(chosen),
        "spent": float(spent),
        "hit_rate": float(sum(hits) / max(len(hits), 1)),
    }
