# -*- coding: utf-8 -*-
"""组级统计量稳健性：sbstack vs 本地默认栈，按策略汇总 budget-200 口径的
n_exp / n_hits / spent，看「~60% 轨迹翻转」是否撼动组级结论。

若组级均值差异在千分位以内 → 轨迹翻转是微观混沌，统计结论不受影响
（这正是"分析用本地全量数据"约定的安全性所在）。

用法：python compare_aggregates.py <group>
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
GROUP = sys.argv[1] if len(sys.argv) > 1 else "gate_ablate"
BUDGET = 200


def prefix_stats(d):
    """budget-200 口径的前缀统计：n_exp / n_hits / spent / hit_rate。"""
    spent = 0.0
    n = 0
    hits = 0
    for c, h in zip(d["cost_seq"], d["hits"]):
        if spent >= BUDGET:
            break
        spent += c
        n += 1
        hits += h
    return n, hits, spent


def load(path, with_hits):
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"strategy"' not in line:
                continue
            d = json.loads(line)
            if with_hits:
                n, hits, spent = prefix_stats(d)
            else:
                # sbstack 探测文件没存 hits —— 用 y_seq 重算不可行（无 task 信息），
                # 改用 chosen 长度与 cost 前缀即可，hits 留空
                spent = 0.0
                n = 0
                for c in d["cost_seq"]:
                    if spent >= BUDGET:
                        break
                    spent += c
                    n += 1
                hits = None
            out[d["strategy"]].append((n, hits, spent))
    return out


def main():
    local = load(os.path.join(RESULTS, f"runs_{GROUP}.jsonl"), True)
    sb = load(os.path.join(RESULTS, f"sbstack_{GROUP}.jsonl"), False)

    print(f"=== {GROUP}：组级统计量（budget 200 口径，30 种子 × 18 任务）===")
    print(f"{'策略':<16}{'n_exp 本地':>12}{'n_exp sb':>12}{'差%':>8}"
          f"{'spent 本地':>12}{'spent sb':>12}{'差%':>8}")
    for strat in sorted(local):
        L, S = local[strat], sb[strat]
        nL = sum(x[0] for x in L) / len(L)
        nS = sum(x[0] for x in S) / len(S)
        sL = sum(x[2] for x in L) / len(L)
        sS = sum(x[2] for x in S) / len(S)
        print(f"{strat:<16}{nL:>12.2f}{nS:>12.2f}{(nS-nL)/nL*100:>7.2f}%"
              f"{sL:>12.3f}{sS:>12.3f}{(sS-sL)/sL*100:>7.3f}%")

    # 命中数组容（本地有 hits）：本地 vs 本地截断到 200 的命中均值，
    # 与 sbstack 的 n_exp 差异对照 —— hits 的跨栈对比需要 y_seq，这里只报告
    # 本地侧口径供参照
    print("\n本地侧 budget-200 命中均值（供参照，跨栈 hits 对比需 task 真值，略）：")
    for strat in sorted(local):
        L = local[strat]
        hL = sum(x[1] for x in L) / len(L)
        print(f"  {strat:<16} mean_hits={hL:.4f}  mean_n_exp="
              f"{sum(x[0] for x in L)/len(L):.2f}  mean_spent="
              f"{sum(x[2] for x in L)/len(L):.3f}")


if __name__ == "__main__":
    main()
