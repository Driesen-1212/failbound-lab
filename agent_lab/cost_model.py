# -*- coding: utf-8 -*-
"""成本模型 v2：把"预算 N 次"变成有真实依据的约束

一次"实验" = 把一个候选推进到可合成/可表征状态的成本：
    成本 = 基础成本 × 结构复杂度惩罚 × 稀有元素惩罚
预算 = 实验室在该周期内可负担的合成/表征产能。

v2 变更：命中口径新增 `topq` 模式（命中 = 目标 top-q 分位），
使所有任务基线命中率恒定，消除"基线命中率"与"可学习性"两个自变量的混淆。
"""

import numpy as np

from config import HARD_EXCLUDE_ATOMS, MAX_ATOMS, RARE_COST_FACTOR


def experiment_cost(row, use_gate=True):
    """返回单个候选的实验成本（越大越贵）。use_gate=False 时退化为等成本。"""
    if not use_gate:
        return 1.0
    cost = 1.0
    if row.get("n_atoms", 0) > MAX_ATOMS:
        cost *= 2.0                      # 结构复杂 → 合成难度翻倍
    if bool(row.get("has_rare", False)):
        cost *= RARE_COST_FACTOR         # 稀有元素：能合成但贵（软约束）
    return float(cost)


def gate_pool(df, use_gate=True):
    """门控（硬约束）：过滤掉明确不可合成的候选。

    硬约束只剔除"成本高到不可接受"的极端复杂结构；稀有元素走软约束（加价）
    而非一票否决，避免评委质疑"你人为缩小了搜索空间"。
    """
    if not use_gate:
        return df
    if "n_atoms" not in df.columns:
        return df.reset_index(drop=True)
    keep = df["n_atoms"] <= HARD_EXCLUDE_ATOMS
    return df[keep].reset_index(drop=True)


def cost_vector(df, use_gate=True):
    """向量化版本。原实现用 df.iterrows() 逐行算，3000 行要 ~1.5s，
    而每个子进程每个任务都要算一次（18 任务 × 6 进程 = 108 次 = 白白烧 2.7 分钟）。
    结果与逐行版逐元素一致。"""
    if not use_gate:
        return np.ones(len(df), dtype=float)
    n_atoms = df["n_atoms"].values if "n_atoms" in df.columns else np.zeros(len(df))
    has_rare = df["has_rare"].values.astype(bool) if "has_rare" in df.columns \
        else np.zeros(len(df), dtype=bool)
    cost = np.ones(len(df), dtype=float)
    cost = np.where(n_atoms > MAX_ATOMS, cost * 2.0, cost)
    cost = np.where(has_rare, cost * RARE_COST_FACTOR, cost)
    return cost


def threshold_of(y, task):
    """按任务模式算命中阈值。

    - topq   : 目标分布的 q 分位数（基线命中率恒为 1-q）
    - greater: 固定 threshold
    """
    y = np.asarray(y, dtype=float)
    if task["mode"] == "topq":
        return float(np.quantile(y, task.get("q", 0.90)))
    if task["mode"] == "greater":
        return float(task["threshold"])
    raise ValueError(f"该模式不支持阈值判定: {task['mode']}")


def utility(y, task):
    """统一效用：越大越好。区间型目标用负距离，其余直接用原值。"""
    y = np.asarray(y, dtype=float)
    if task["mode"] == "range":
        return -np.abs(y - task["center"])
    return y


def is_hit(y, task, threshold=None):
    """命中判定。topq / greater 用阈值；range 用区间。"""
    if task["mode"] == "range":
        return abs(float(y) - task["center"]) <= task["half_width"]
    if threshold is None:
        threshold = task.get("threshold")
    return float(y) >= float(threshold)
