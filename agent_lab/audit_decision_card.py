# -*- coding: utf-8 -*-
"""准则卡审计 —— 用实测真值逐任务核对决策规则的假阳/假阴。

=== 为什么要有这个脚本 ===
报告 3.5 节自己拆了自己一次：可学习性 AUC 与 BO 增益的秩相关在消融臂上
只有 ρ=+0.513，且 AUC 0.593 增益 −20.6 而 AUC 0.633 增益 +20.7 ——
**AUC 不是干净的预测因子**。而准则卡（failbound_agent.decide）恰恰用
AUC 阈值（0.60/0.75）和维度阈值（≤2）做拒绝/采用决策。

一个必然会被问到的问题：**"你说 AUC 预测力有限，那你的 Agent 用 AUC
做决策，错了怎么办？"** 回避这个问题等于把命门交给评委。

本脚本把这个问题变成可回答的：对每个任务，用与 decide() 完全相同的
输入（AUC、dim）跑出卡的决策，再与主实验真值（tests.csv，预算 400
等成本口径，gp_ucb vs farthest_point，Holm 校正）对照，输出：

    采用/拒绝 × 显著为正/显著为负/不显著  的混淆式核对表

以及三个汇总指标：
    - 漏报率：卡放行（用 BO）但实测显著为负的任务占比
    - 误拒率：卡拒绝但实测显著为正的任务占比
    - 卡的保护对象：被卡拒绝且实测确实显著为负的任务（真拒绝）

=== 结论的诚实边界（写进报告的说法）===
准则卡是**熔断器，不是预报器**：
  - 它不声称"AUC 单调预测增益"（3.5 已证伪这个声称）；
  - 它只声称"在这两个阈值围出的区域里，实测出现过灾难性损失
    （最低 −33.9 命中），不值得把预算赌在 BO 上"；
  - 因此正确的评价指标是**风险调整后的收益**，不是逐任务胜率 ——
    误拒的代价是少赚（+14 ~ +32），漏报的代价是巨亏（−17.5 ~ −33.9）。
  这个不对称正是"保守卡"的设计意图，本脚本把两侧的账都算出来。

用法：
    python audit_decision_card.py                # 默认预算 400 等成本口径
    python audit_decision_card.py --budget 200   # 换预算档
    python audit_decision_card.py --gate         # 换门控口径
输出：results/decision_card_audit.csv + 控制台核对表
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import discover_tasks  # noqa: E402
from failbound_agent import decide  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "results", "tests.csv")
PROPS = os.path.join(HERE, "results", "task_props.csv")
OUT = os.path.join(HERE, "results", "decision_card_audit.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=400,
                    help="用哪一档预算的真值做对照（默认 400 = 主口径）")
    ap.add_argument("--gate", action="store_true",
                    help="用门控口径的真值（默认等成本口径）")
    ap.add_argument("--two-param", action="store_true",
                    help="复现预注册的两参数卡（dim≤2 或 AUC<0.60）；默认跑交付版"
                         "（唯一否决条件 dim≤2，见 §4.4.2 与 §4.10②）")
    a = ap.parse_args()

    t = pd.read_csv(TESTS)
    props = pd.read_csv(PROPS)
    truth = t[(t.budget == a.budget) & (t.use_gate == a.gate) &
              (t.strategy == "gp_ucb")][["task", "arm", "abs_gain",
                                         "p_holm", "significant", "mean_base"]]
    props = props[["task", "separability_auc", "dim", "base_hit_rate"]]

    # 诊断命中率：卡的 base_hit_rate 输入来自 25 预算的空间填充探测。
    # 这里用 tests.csv 同档位的 mean_base（已是 farthest_point 均值）折算成
    # "每单位预算命中率"作近似 —— 注意 decide() 中该输入**只影响置信度标签、
    # 从不翻转 use_bo**（见 failbound_agent.decide 的 base_hit_rate 分支），
    # 所以这个近似不影响审计核心问题（拒绝/放行），只影响 confidence 列的精度。
    df = truth.merge(props, on="task")
    df["probe_hit_rate"] = df["mean_base"] / a.budget

    rows = []
    for _, r in df.iterrows():
        d = decide(r["separability_auc"], int(r["dim"]),
                   float(r["probe_hit_rate"]), {},
                   legacy_auc_reject=a.two_param)
        rows.append({
            "task": r["task"], "arm": r["arm"],
            "auc": round(r["separability_auc"], 3), "dim": int(r["dim"]),
            "card_use_bo": d["use_bo"], "card_strategy": d["strategy"],
            "card_confidence": d["confidence"],
            "actual_gain": round(r["abs_gain"], 2),
            "actual_significant": bool(r["significant"]),
            "trigger_reasons": "；".join(d["reasons"]),
        })
    out = pd.DataFrame(rows)

    # ---- 三类真值标签 ----
    def label(r):
        if r["actual_significant"] and r["actual_gain"] < 0:
            return "显著为负（BO 真输）"
        if r["actual_significant"] and r["actual_gain"] > 0:
            return "显著为正（BO 真赢）"
        return "不显著（噪声）"
    out["truth"] = out.apply(label, axis=1)

    # ---- 汇总 ----
    def summarize(o):
        neg = o[o.truth == "显著为负（BO 真输）"]
        pos = o[o.truth == "显著为正（BO 真赢）"]
        ns = o[o.truth == "不显著（噪声）"]
        n = len(o)
        return {
            "任务数": n,
            "显著为负": len(neg), "显著为正": len(pos), "不显著": len(ns),
            "卡拒绝的任务数": int((~o.card_use_bo).sum()),
            "真拒绝（拒绝且真输）": int(((~o.card_use_bo) &
                                      (o.truth == "显著为负（BO 真输）")).sum()),
            "误拒（拒绝但真赢）": int(((~o.card_use_bo) &
                                     (o.truth == "显著为正（BO 真赢）")).sum()),
            # 主页/视频公布的「误杀」口径 = 拒绝但实测没有真输（显著为正 + 不显著）。
            # 两者分开列：严格的假阳只算显著为正那一列，噪声属于「本可以不拒、但也不亏」。
            "误杀（拒绝但未真输，含噪声）": int(((~o.card_use_bo) &
                                       (o.truth != "显著为负（BO 真输）")).sum()),
            "拒绝但仅噪声": int(((~o.card_use_bo) &
                               (o.truth == "不显著（噪声）")).sum()),
            "漏报（放行且真输）": int(((o.card_use_bo) &
                                     (o.truth == "显著为负（BO 真输）")).sum()),
            "正确放行（放行且真赢）": int(((o.card_use_bo) &
                                        (o.truth == "显著为正（BO 真赢）")).sum()),
            "漏报率（放行中真输占比）": round(
                len(o[(o.card_use_bo) & (o.truth == "显著为负（BO 真输）")]) /
                max((o.card_use_bo).sum(), 1), 3),
        }

    print(f"\n=== 准则卡审计 | 预算 {a.budget} | "
          f"{'门控' if a.gate else '等成本'}口径 | "
          f"{'预注册两参数卡（dim≤2 或 AUC<0.60）' if a.two_param else '交付版（仅 dim≤2）'} | "
          f"真值来自 tests.csv ===")
    for arm in ("main", "ablate", "ALL"):
        o = out if arm == "ALL" else out[out.arm == arm]
        if len(o) == 0:
            continue
        s = summarize(o)
        print(f"\n--- {arm if arm != 'ALL' else '全部'} 臂 ---")
        print(pd.Series(s).to_string())

    print("\n=== 逐任务核对表（按 AUC 升序）===")
    cols = ["task", "arm", "auc", "dim", "card_strategy", "card_confidence",
            "actual_gain", "truth", "trigger_reasons"]
    print(out.sort_values("auc")[cols].to_string(index=False))

    out.to_csv(OUT, index=False)
    print(f"\n[save] {OUT}")

    # ---- 风险调整视角：把误拒的代价和漏报的代价摆到同一张桌子上 ----
    print("\n=== 风险调整视角（不对称代价）===")
    for arm in ("main", "ablate", "ALL"):
        o = out if arm == "ALL" else out[out.arm == arm]
        if len(o) == 0:
            continue
        miss = o[(o.card_use_bo) & (o.truth == "显著为负（BO 真输）")]
        false_rej = o[(~o.card_use_bo) & (o.truth == "显著为正（BO 真赢）")]
        caught = o[(~o.card_use_bo) & (o.truth == "显著为负（BO 真输）")]
        print(f"[{arm if arm != 'ALL' else '全部'}] "
              f"卡拦住的损失 {caught['actual_gain'].sum():+.1f} 命中；"
              f"漏报的损失 {miss['actual_gain'].sum():+.1f}；"
              f"误拒少赚 {false_rej['actual_gain'].sum():+.1f}")


if __name__ == "__main__":
    main()
