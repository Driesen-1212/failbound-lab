# -*- coding: utf-8 -*-
"""合并 --shard 分片产生的 runs_*.shardKofN.jsonl → runs_*.jsonl

为什么需要：
    沙箱单次会话有时长上限。run_experiments.py --shard K/N 让每个会话只领走 1/N 的任务，
    各片独立落盘、互不覆盖；会话被中断只丢当前片。所有片跑完后用本脚本合并，
    合并结果与「一次性跑完」在内容上等价（行序按片序拼接，分析端不依赖行序）。

用法：
    python merge_shards.py            # 合并 results/ 下所有分片，并做完整性自检
    python merge_shards.py --keep     # 合并后保留分片文件（默认删除）

输出：runs_gate.jsonl / runs_nogate.jsonl / runs_*_ablate.jsonl
自检：每个 (arm × 口径) 的 run 数应等于 任务数 × 策略数 × 种子数，缺片会明确报警。
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUT_DIR, STRATEGIES, discover_tasks
from out_guard import guard_out_path

SHARD_RE = re.compile(r"^(runs_(?:no)?gate(?:_\w+)?)\.shard(\d+)of(\d+)\.jsonl$")


def parse_base(base):
    """runs_nogate_ablate → (use_gate=False, arm='ablate')"""
    rest = base[len("runs_"):]
    gate = not rest.startswith("nogate")
    arm = "ablate" if rest.endswith("_ablate") else "main"
    return gate, arm


def merge_group(base, shard_files, keep=False, overwrite=False):
    out_path = os.path.join(OUT_DIR, base + ".jsonl")
    guard_out_path(out_path, overwrite, what="合并结果")
    meta, task_meta, runs, order_seen = {}, {}, [], []
    shards = []

    for k, path in sorted(shard_files):
        shards.append(k)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "meta" in obj:
                    m = dict(obj["meta"])
                    m.pop("shard", None)          # 合并后不再是某一片
                    for t in m.get("task_order", []):
                        if t not in order_seen:
                            order_seen.append(t)
                    meta = m                       # 各片 meta 除 task_order 外相同
                elif "task_meta" in obj:
                    task_meta.update(obj["task_meta"])
                else:
                    runs.append(line)              # 原样保留，避免二次序列化差异

    meta["task_order"] = order_seen
    meta["shards_merged"] = shards

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for line in runs:
            f.write(line + "\n")
        f.write(json.dumps({"task_meta": task_meta}, ensure_ascii=False) + "\n")

    if not keep:
        for _, path in shard_files:
            try:
                os.remove(path)
            except OSError:
                pass

    # ---- 完整性自检 ----
    gate, arm = parse_base(base)
    n_tasks = len(discover_tasks(arm=arm))
    n_seeds = int(meta.get("n_seeds", 0))
    expect = n_tasks * len(STRATEGIES) * n_seeds
    got = len(runs)
    ok = (got == expect)
    print(f"[merge] {base:<24} shards={shards} runs={got}/{expect} "
          f"tasks={len(task_meta)}/{n_tasks} "
          f"{'✅ 完整' if ok else '❌ 缺片（补齐后重新合并）'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="合并后保留分片文件")
    ap.add_argument("--overwrite", action="store_true",
                    help="目标 runs_*.jsonl 已有数据时仍覆盖（会先自动备份）")
    args = ap.parse_args()

    groups = {}
    for name in sorted(os.listdir(OUT_DIR)):
        m = SHARD_RE.match(name)
        if m:
            groups.setdefault(m.group(1), []).append(
                (int(m.group(2)), os.path.join(OUT_DIR, name)))

    if not groups:
        print(f"[无分片] {OUT_DIR} 下没有 runs_*.shardNofM.jsonl")
        return

    all_ok = True
    for base, files in sorted(groups.items()):
        all_ok &= merge_group(base, files, keep=args.keep, overwrite=args.overwrite)
    print("[done] 全部完整 ✅" if all_ok else "[done] 存在缺片，请勿用不完整数据分析 ❌")


if __name__ == "__main__":
    main()
