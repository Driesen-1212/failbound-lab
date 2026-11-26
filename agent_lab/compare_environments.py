# -*- coding: utf-8 -*-
"""跨环境复现对照 —— 本地全量数据 vs 沙箱预算 200 数据。

🔴 为什么需要：预注册 §10 登记过「运行环境分级」——沙箱跑预算 200、本地跑全档 400，
   两环境用同一套代码、同一种子派生规则、同一统计口径，**构成跨环境复现检验**。
   检验方式必须是全项目最强的那个：**逐 run 比对 chosen 选择序列**。
   因为 run_one 在给定 (任务, 策略, 种子, 数据) 后是完全确定性的，
   所以序列一致 ⟺ 代码与数据在两环境逐比特一致。只比汇总统计量会漏掉
   「数字碰巧一样但路径不同」的情况。

用法（沙箱的 4 个 jsonl 下载到本地后）：
    python compare_environments.py \
        --local results/runs_nogate.jsonl  --sandbox /path/runs_nogate_sandbox.jsonl \
        --local results/runs_gate.jsonl   --sandbox /path/runs_gate_sandbox.jsonl

    # 一次比对一组（gate / nogate），arm 可用 --arm ablate 指定

输出：逐组的一致/不一致统计；不一致时打印前若干个差异样例（任务/策略/种子 + 首个分歧位置）。
退出码 0 = 全部一致。
"""

import argparse
import json
import os
import sys


def load(path):
    meta, runs = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "meta" in d:
                meta = d["meta"]
            elif "task_meta" in d:
                continue
            elif "strategy" in d:
                runs[(d["task"], d["strategy"], d["seed"], bool(d["use_gate"]))] = d
    return meta, runs


def compare(local_path, sandbox_path, budget, verbose=5):
    lm, lr = load(local_path)
    sm, sr = load(sandbox_path)
    print(f"\n=== {os.path.basename(local_path)} vs {os.path.basename(sandbox_path)} ===")
    print(f"[load] 本地 {len(lr)} run | 沙箱 {len(sr)} run")

    # ---- 1. 口径元信息 ----
    keys = ("seed", "n_init", "hit_q", "n_seeds", "budgets")
    for k in keys:
        a, b = lm.get(k), sm.get(k)
        if a != b:
            print(f"  ⚠ meta.{k}: 本地 {a} vs 沙箱 {b}")

    # 只比对预算 ≤ budget 的 run（沙箱只跑到这个档）
    common = set(lr) & set(sr)
    only_local = len(set(lr) - set(sr))
    only_sandbox = len(set(sr) - set(lr))
    print(f"[key] 两边都有 {len(common)} 个 run | 仅本地 {only_local} | 仅沙箱 {only_sandbox}")
    if not common:
        print("  ❌ 没有可比对的 run——先检查任务名/种子/meta 是否一致")
        return 1

    # ---- 2. 逐 run 比对 chosen 序列（最强检验）----
    seq_bad, hit_bad, spent_bad, shown = [], [], [], 0
    for k in sorted(common):
        L, S = lr[k], sr[k]
        lc, sc = L["chosen"], S["chosen"]
        if lc[:len(sc)] != sc:
            seq_bad.append(k)
            if shown < verbose:
                # 找第一个分歧位置
                i = next((j for j in range(len(sc)) if lc[j] != sc[j]), None)
                print(f"  ❌ 序列分歧 {k} 首个分歧位置={i} "
                      f"本地 {lc[i] if i is not None else '?'} / 沙箱 {sc[i] if i is not None else '?'}")
                shown += 1
        if L["n_hits"] != S["n_hits"]:
            hit_bad.append((k, L["n_hits"], S["n_hits"]))
        if abs(L["spent"] - S["spent"]) > 1e-6:
            spent_bad.append((k, L["spent"], S["spent"]))

    n = len(common)
    print(f"[seq] chosen 序列一致 {n - len(seq_bad)}/{n} "
          f"{'✅' if not seq_bad else '❌'}")
    print(f"[hit] 命中数一致 {n - len(hit_bad)}/{n} "
          f"{'✅' if not hit_bad else '❌'}")
    print(f"[spend] 总成本一致 {n - len(spent_bad)}/{n} "
          f"{'✅' if not spent_bad else '❌'}")
    if hit_bad[:verbose]:
        print("  命中数差异样例：", hit_bad[:verbose])
    if spent_bad[:verbose]:
        print("  成本差异样例：", spent_bad[:verbose])

    ok = not (seq_bad or hit_bad or spent_bad)
    if ok:
        print("  ✅ 该组跨环境完全一致：同一份代码在两环境给出逐比特相同的选择序列")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--sandbox", required=True)
    ap.add_argument("--budget", type=int, default=200,
                    help="沙箱跑到的预算档，只比对 ≤ 该档的选择序列")
    a = ap.parse_args()
    for p in (a.local, a.sandbox):
        if not os.path.exists(p):
            raise SystemExit(f"文件不存在：{p}")
    sys.exit(compare(a.local, a.sandbox, a.budget))


if __name__ == "__main__":
    main()
