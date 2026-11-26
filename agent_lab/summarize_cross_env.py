# -*- coding: utf-8 -*-
"""汇总两次全组探测的对比结果 → results/cross_env_quantified.json。
一次跑完，存档供文档引用。"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BUDGET = 200


def truncate(cost_seq):
    spent = 0.0
    for c in cost_seq:
        if spent >= BUDGET:
            break
        spent += c
    return spent


def analyze(group):
    local = {}
    with open(os.path.join(RESULTS, f"runs_{group}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"strategy"' not in line:
                continue
            d = json.loads(line)
            local[(d["task"], d["strategy"], d["seed"])] = d

    n = stable = 0
    flips = []
    sb_max = 0.0
    by_strat_flips = Counter()
    by_strat_total = Counter()
    diff_dist = Counter()
    with open(os.path.join(RESULTS, f"sbstack_{group}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d["task"], d["strategy"], d["seed"])
            n += 1
            by_strat_total[d["strategy"]] += 1
            sbt = truncate(d["cost_seq"])
            sb_max = max(sb_max, sbt)
            ld = local[key]
            k = len(d["chosen"])
            if ld["chosen"][:k] == d["chosen"]:
                stable += 1
            else:
                flips.append(key)
                by_strat_flips[d["strategy"]] += 1
                diff_dist[abs(sbt - truncate(ld["cost_seq"]))] += 1

    local_max = max(truncate(d["cost_seq"]) for d in local.values())
    return {
        "group": group, "n_runs": n, "stable": stable, "flipped": len(flips),
        "flip_rate": round(len(flips) / n, 4),
        "by_strategy": {s: {"flipped": by_strat_flips[s], "total": by_strat_total[s],
                            "flip_rate": round(by_strat_flips[s] / by_strat_total[s], 4)}
                        for s in sorted(by_strat_total)},
        "truncated_spent_diff_dist": {str(k): v for k, v in sorted(diff_dist.items())},
        "max_spent": {"local_default": local_max, "sbstack": sb_max},
    }


def main():
    out = {"budget": BUDGET, "threading": "both stacks 1 BLAS thread",
           "groups": {}}
    for g in ("gate", "gate_ablate"):
        out["groups"][g] = analyze(g)

    # 组级统计量（budget-200 前缀）
    def prefix_stats(d):
        spent = 0.0
        k = 0
        for c in d["cost_seq"]:
            if spent >= BUDGET:
                break
            spent += c
            k += 1
        return k, spent

    agg = {}
    for g in ("gate", "gate_ablate"):
        loc = defaultdict(list)
        with open(os.path.join(RESULTS, f"runs_{g}.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or '"strategy"' not in line:
                    continue
                d = json.loads(line)
                k, s = prefix_stats(d)
                loc[d["strategy"]].append((k, s, sum(d["hits"][:k])))
        sb = defaultdict(list)
        with open(os.path.join(RESULTS, f"sbstack_{g}.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                k, s = prefix_stats(d)
                sb[d["strategy"]].append((k, s, None))
        agg[g] = {}
        for strat in sorted(loc):
            L, S = loc[strat], sb[strat]
            agg[g][strat] = {
                "n_exp_local": round(sum(x[0] for x in L) / len(L), 3),
                "n_exp_sb": round(sum(x[0] for x in S) / len(S), 3),
                "spent_local": round(sum(x[1] for x in L) / len(L), 4),
                "spent_sb": round(sum(x[1] for x in S) / len(S), 4),
                "hits_local": round(sum(x[2] for x in L) / len(L), 4),
            }
    out["group_level_stats_budget200"] = agg

    p = os.path.join(RESULTS, "cross_env_quantified.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n→ {p}")


if __name__ == "__main__":
    main()
