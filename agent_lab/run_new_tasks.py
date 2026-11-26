# -*- coding: utf-8 -*-
"""只跑"尚未跑过"的任务（增量补跑），结果**单独写到一个新文件**，供 merge_runs.py 合并。

为什么不用 --shard：分片是按索引取模，会把已跑完的任务也重新领一遍。
这里是显式指定任务名清单，只跑新增的。

🔴 曾经的致命缺陷（10/17）：本脚本调用 RE.main() 时用的是默认输出名，
    也就是直接往 runs_nogate.jsonl 上写 —— 一旦被中途 kill，
    2160 run 的完整数据就被残片冲掉了（那次靠手工备份才救回来）。
✅ 现在默认写到 runs_<group>_add.jsonl，绝不会碰主数据文件；
    跑完打印一行可直接复制的 merge_runs.py 命令。

用法：
    python run_new_tasks.py no-gate 400 6 <任务名1> <任务名2> ...
    python run_new_tasks.py no-gate 400 6 --tag v2 <任务名>      # 自定义后缀
    python run_new_tasks.py no-gate 200 4 --dry <任务名>         # 只列出要跑什么

跑完之后：
    python merge_runs.py results/runs_nogate.jsonl results/runs_nogate_add.jsonl \
                         results/runs_nogate_full.jsonl
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiments as RE  # noqa: E402
from config import discover_tasks  # noqa: E402


def parse_args(argv):
    """手写解析，兼容旧的「位置参数 + 任务名列表」用法，另加 --tag / --dry。"""
    positional, tag, dry = [], "add", False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--tag":
            i += 1
            tag = argv[i] if i < len(argv) else "add"
        elif a == "--dry":
            dry = True
        else:
            positional.append(a)
        i += 1
    return positional, tag, dry


def main(argv):
    positional, tag, dry = parse_args(argv)
    group = positional[0] if positional else "no-gate"
    max_budget = int(positional[1]) if len(positional) > 1 else 400
    n_proc = int(positional[2]) if len(positional) > 2 else 6
    wanted = set(positional[3:])
    if not wanted:
        raise SystemExit(__doc__)

    full = discover_tasks("main")
    missing = wanted - set(full)
    if missing:
        raise SystemExit(
            f"这些任务不在注册表里：{sorted(missing)}\n"
            f"注册表现有任务（main 臂）共 {len(full)} 个。")

    # ⚠️ 文件名前缀与 group 参数不是同一个字面量：
    #    group="no-gate" → 文件前缀 "nogate"（run_experiments.main 里的规则）。
    #    这里必须同步，否则打印出来的合并命令会指向不存在的文件。
    prefix = "gate" if group != "no-gate" else "nogate"
    if dry:
        print(f"[dry] group={group} max_budget={max_budget} proc={n_proc} "
              f"out=runs_{prefix}_{tag}.jsonl")
        for t in sorted(wanted):
            print(f"  - {t}")
        return None

    # 只保留指定任务（monkeypatch 模块级的 discover_tasks）
    RE.discover_tasks = lambda arm="main": {k: v for k, v in full.items() if k in wanted}
    if len(wanted) != len(RE.discover_tasks()):
        raise SystemExit("任务过滤结果与 wanted 不一致，请检查注册表")

    out = RE.main(use_gate=(group != "no-gate"), n_seeds=30, n_proc=n_proc,
                  arm="main", max_budget=max_budget, out_suffix=f"_{tag}")
    base = os.path.basename(out)
    target = f"runs_{prefix}.jsonl"
    full = f"runs_{prefix}_full.jsonl"
    print(f"[run_new_tasks] 完成 {len(wanted)} 个任务 → {base}")
    print("[next] 合并进主数据（会先做重复校验）：")
    print(f"    python merge_runs.py results/{target} results/{base} results/{full}")
    print(f"    确认无误后：mv results/{full} results/{target}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
