# -*- coding: utf-8 -*-
"""命中率跨栈稳健性：sbstack 探测文件没存 hits，这里用任务真值（同一份数据、
同一个 is_hit 口径）从 chosen 序列重算，与本地文件的 hits 逐策略对比。
这是科学结论的核心统计量（abs_gain / hit_rate 的分母分子）。

用法：python compare_hits.py <group>
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import discover_tasks
from cost_model import is_hit
from run_experiments import _task_arrays

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
GROUP = sys.argv[1] if len(sys.argv) > 1 else "gate_ablate"
ARM = "main" if GROUP == "gate" else "ablate"
GATE = True
BUDGET = 200

cache = {}


def task_ctx(tname):
    if tname not in cache:
        task, X_std, y, cost, thr = _task_arrays(tname, GATE, ARM)
        cache[tname] = (task, y, thr)
    return cache[tname]


def prefix(d):
    spent = 0.0
    k = 0
    for c in d["cost_seq"]:
        if spent >= BUDGET:
            break
        spent += c
        k += 1
    return k


def hits_of(d):
    task, y, thr = task_ctx(d["task"])
    k = prefix(d)
    return [int(is_hit(float(y[i]), task, thr)) for i in d["chosen"][:k]]


def load_local():
    out = defaultdict(list)
    with open(os.path.join(RESULTS, f"runs_{GROUP}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"strategy"' not in line:
                continue
            d = json.loads(line)
            k = prefix(d)
            out[d["strategy"]].append(sum(d["hits"][:k]))
    return out


def load_sb():
    out = defaultdict(list)
    with open(os.path.join(RESULTS, f"sbstack_{GROUP}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["strategy"]].append(sum(hits_of(d)))
    return out


def main():
    local, sb = load_local(), load_sb()
    print(f"=== {GROUP}：budget-200 口径命中数（30 种子 × 18 任务均值）===")
    print(f"{'策略':<16}{'本地':>10}{'sbstack':>10}{'绝对差':>10}{'相对差':>10}")
    for strat in sorted(local):
        L = sum(local[strat]) / len(local[strat])
        S = sum(sb[strat]) / len(sb[strat])
        print(f"{strat:<16}{L:>10.4f}{S:>10.4f}{S-L:>+10.4f}{(S-L)/L*100:>+9.3f}%")
    # BO 增益（相对最强基线 farthest_point）的跨栈差异
    for group, name in ((local, "本地"), (sb, "sbstack")):
        base = sum(group["farthest_point"]) / len(group["farthest_point"])
        gains = {s: sum(group[s]) / len(group[s]) - base
                 for s in ("gp_ei", "gp_ucb")}
        print(f"{name}：BO 增益 vs farthest_point = "
              + ", ".join(f"{s} {g:+.4f}" for s, g in gains.items()))


if __name__ == "__main__":
    main()
