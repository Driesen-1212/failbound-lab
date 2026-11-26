# -*- coding: utf-8 -*-
"""输出每个 run 的紧凑指纹，一行一个：

    task|strategy|seed|n_exp|spent_trunc|chosen_sha16

设计目的：沙箱侧的 runs 文件太大无法贴回对话（全量约 6.5MB），
但逐 run 指纹只有几十字节。沙箱对本文件输出执行一次、把结果贴回，
本地用 compare_digests.py 复算同一种指纹并比对 —— 等价于逐 run
chosen 序列比对，且不需要文件本体。

序列化是显式的（json.dumps separators=(",",":")），跨 Python 版本稳定；
截断口径与 L5 一致（while spent < budget 的前缀）。

用法：
    python emit_run_digests.py <runs.jsonl> [--budget 200]
"""
import argparse
import hashlib
import json
import sys


def truncate(cost_seq, budget):
    spent = 0.0
    k = 0
    for c in cost_seq:
        if spent >= budget:
            break
        spent += c
        k += 1
    return k, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--budget", type=int, default=200)
    a = ap.parse_args()

    n = 0
    out = []
    with open(a.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"strategy"' not in line:
                continue
            d = json.loads(line)
            k, spent = truncate(d["cost_seq"], a.budget)
            chosen = d["chosen"][:k]
            h = hashlib.sha256(
                json.dumps(chosen, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            out.append(f'{d["task"]}|{d["strategy"]}|{d["seed"]}|{k}'
                       f'|{spent:g}|{h}')
            n += 1
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    print(f"# {n} runs, budget={a.budget}", file=sys.stderr)


if __name__ == "__main__":
    main()
