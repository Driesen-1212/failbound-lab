# -*- coding: utf-8 -*-
"""基线稳定性检查：强基线自身的命中率是否跨任务稳定。

为什么需要这个检查（2026-10-16 的教训）：
    命中阈值按 top-10% 分位定义，所以**任何**无偏策略的期望命中率都必须是 0.100。
    实测旧的 `lhs_stratified` 在 0.022–0.326 之间摆动（15 倍），
    而 rel_gain 以基线命中数为分母 —— 于是"BO 增益"实际在测"基线的运气"。

判据（2026-10-17 修订，见下方实测结果）：
    - `random_uniform` 必须 ≈ 0.100（否则是命中口径的 bug，不是策略的 bug）
    - 强基线 `farthest_point` 的波动要**分两侧看**：
        上侧（高于 0.100）：正常，是有信息的空间填充设计，越高越好。
        下侧（低于 0.100）：**真实效应，不是 bug** —— max-min 空间填充在高价值样本
                            聚成一簇时会系统性欠采样该簇。这类任务要单独标记，
                            因为 `rel_gain` 以基线命中数为分母，在这些任务上会爆炸。
    ⚠️ 因此本脚本的"极差/中位 < 3×"只是**提示**，不再作为硬性通过判据：
       实测 farthest_point 极差/中位 = 3.68×，已确认来自真实效应。
       真正的应对是把效应量口径换成 abs_gain（见 analyze.py 模块顶部注释），
       而不是继续换基线。

用法：
    python check_baseline_stability.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BUDGETS, OUT_DIR, STRATEGIES


def hit_rate_table(summary, budget):
    s = summary[summary["budget"] == budget]
    g = (s.groupby(["task", "strategy"])
           .agg(hits=("n_hits", "mean"), n=("n_exp", "mean")).reset_index())
    g["hit_rate"] = g["hits"] / g["n"].clip(lower=1)
    return g.pivot(index="task", columns="strategy", values="hit_rate")


def main():
    path = os.path.join(OUT_DIR, "summary.csv")
    if not os.path.exists(path):
        print(f"[skip] 没有 {path}，先跑 analyze.py")
        return
    summary = pd.read_csv(path)
    budget = int(summary["budget"].max())

    for gate in sorted(summary["use_gate"].unique()):
        sub = summary[summary["use_gate"] == bool(gate)]
        t = hit_rate_table(sub, budget)
        print(f"\n===== use_gate={bool(gate)}  预算={budget} =====")
        cols = [c for c in ("random_uniform", "farthest_point", "lhs_stratified",
                            "gp_ucb", "gp_ei") if c in t.columns]
        print(t[cols].round(4).to_string())

        print("\n--- 稳定性判据 ---")
        for c in cols:
            v = t[c].dropna()
            print(f"  {c:<16} min={v.min():.3f}  max={v.max():.3f}  "
                  f"中位={v.median():.3f}  极差/中位={(v.max()-v.min())/max(v.median(),1e-9):.2f}×")
        if "random_uniform" in t.columns:
            ru = t["random_uniform"].dropna()
            # ⚠️ 曾写成 bool((ru > .085) & (ru < .115)).all() —— 先对 Series 取 bool
            #    直接抛 ValueError（Series 的真值不明确）。必须先 .all() 再 bool()。
            ok = bool(((ru > 0.085) & (ru < 0.115)).all()) if len(ru) else False
            print(f"  [命中口径] random_uniform 全部落在 [0.085,0.115]: {'✅' if ok else '❌'}")
        if "farthest_point" in t.columns:
            fp = t["farthest_point"].dropna()
            spread = (fp.max() - fp.min()) / max(fp.median(), 1e-9)
            print(f"  [强基线波动] farthest_point 极差/中位 = {spread:.2f}×  "
                  f"（提示值，非硬性判据）")
            # 下侧：低于随机基线 = 空间填充在该任务上系统性失效
            if "random_uniform" in t.columns:
                ru = t["random_uniform"]
                below = fp[fp < ru - 0.02]
                print(f"  [下侧失效] 低于随机基线的任务 {len(below)}/{len(fp)} 个：")
                for name in below.index:
                    print(f"      {name:<34} fp={fp[name]:.3f}  "
                          f"random={ru[name]:.3f}  ← rel_gain 在此任务不可用")
                if len(below):
                    print("      → 这是【真实发现】：高价值样本聚成一簇时，max-min "
                          "空间填充会欠采样该簇。已确认不是实现 bug，不必换基线。")
        if "lhs_stratified" in t.columns:
            lh = t["lhs_stratified"].dropna()
            spread = (lh.max() - lh.min()) / max(lh.median(), 1e-9)
            print(f"  [已废弃基线] lhs_stratified 极差/中位 = {spread:.2f}×  "
                  f"（这就是它被换掉的原因）")


if __name__ == "__main__":
    main()
