# -*- coding: utf-8 -*-
"""比对 sbstack 全组探测结果 vs 本地默认栈结果，量化「只换库版本」的翻转率。

用法：python compare_probe.py <group>
  group = gate | gate_ablate

输入：
  results/sbstack_<group>.jsonl        沙箱同库版本栈（budget 200）重跑产物
  results/runs_<group>.jsonl           本地默认栈（budget 400）全量产物
  results/sandbox_claims.json          沙箱承诺（文件级 max_spent）

比对口径：sbstack 的 chosen 序列 vs 本地 chosen 序列的前 k 个（k = len(sbstack chosen)）。
本地跑的是预算 400，其前缀与预算 200 逐点等价（run_trial 的选择与 budget 无关，
只决定何时停 —— 前缀性质，L5 即以此截断）。前缀逐点相同 = 该 run 跨环境未翻转。
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

GROUP = sys.argv[1] if len(sys.argv) > 1 else "gate_ablate"
LOCAL = os.path.join(RESULTS, f"runs_{GROUP}.jsonl")
SB = os.path.join(RESULTS, f"sbstack_{GROUP}.jsonl")
CLAIMS = os.path.join(RESULTS, "sandbox_claims.json")

BUDGET = 200


def truncate(cost_seq):
    spent = 0.0
    for c in cost_seq:
        if spent >= BUDGET:
            break
        spent += c
    return spent


def load_local():
    out = {}
    with open(LOCAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"strategy"' not in line:
                continue
            d = json.loads(line)
            out[(d["task"], d["strategy"], d["seed"])] = d
    return out


def main():
    local = load_local()
    claims = json.load(open(CLAIMS, encoding="utf-8"))["files"][f"runs_{GROUP}.jsonl"]
    local_max = max(truncate(d["cost_seq"]) for d in local.values())

    n = n_match = 0
    flips = []
    sb_max = 0.0
    with open(SB, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (d["task"], d["strategy"], d["seed"])
            n += 1
            sb_trunc = truncate(d["cost_seq"])
            sb_max = max(sb_max, sb_trunc)
            ld = local.get(key)
            if ld is None:
                print(f"❌ 本地缺少 {key}")
                continue
            k = len(d["chosen"])
            if ld["chosen"][:k] == d["chosen"]:
                n_match += 1
            else:
                div = next((j for j in range(min(k, len(ld["chosen"])))
                            if ld["chosen"][j] != d["chosen"][j]), None)
                flips.append((key, sb_trunc, truncate(ld["cost_seq"]), div))

    print(f"=== {GROUP}：sbstack（沙箱同库版本栈）vs 本地默认栈 ===")
    print(f"run 总数 {n} | 前缀逐点一致 {n_match} | 翻转 {len(flips)}"
          f"（翻转率 {len(flips)/max(n,1):.2%}）")
    print(f"文件级 max_spent：本地默认栈 {local_max} | sbstack {sb_max} | "
          f"沙箱承诺 {claims['max_spent']}")

    if flips:
        print("\n翻转 run 明细（div = 首个分歧的选择步序号）：")
        for (task, strat, seed), sbt, lt, div in sorted(flips):
            print(f"  {task}/{strat}/seed{seed}: div=第{div}步 "
                  f"sbstack={sbt:g} 本地={lt:g}")
        print(f"\n按策略分布：{dict(Counter(k[1] for k, *_ in flips))}")
        print(f"截断值差分布：{dict(sorted(Counter(abs(s - l) for _, s, l, _ in flips).items()))}")

    ok = abs(sb_max - claims["max_spent"]) < 1e-9
    print(f"\nsbstack 文件级 max {sb_max:g} vs 沙箱承诺 {claims['max_spent']:g}："
          f"{'精确一致 ✅' if ok else '不一致'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
