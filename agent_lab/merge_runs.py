# -*- coding: utf-8 -*-
"""把两份 runs_*.jsonl 合并成一份（增量补跑新任务时用）。

为什么需要：主臂任务集扩容（18 → 27）后，若整组重跑要 ~8 小时；
而已跑完的 18 个任务完全没必要重跑。于是：
    1. 只跑新增任务 → 单独文件
    2. 用本脚本与旧文件合并 → 与"一次性跑完"在内容上等价

用法：
    python merge_runs.py results/runs_nogate_base.jsonl results/runs_nogate_new.jsonl \
                         results/runs_nogate.jsonl

自检：合并后 run 数 = 两份之和，任务数 = 两份任务的并集，且预算档一致。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from out_guard import check_no_overlap, guard_out_path  # noqa: E402


def load(path):
    meta, task_meta, runs = {}, {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
            elif "task_meta" in obj:
                task_meta.update(obj["task_meta"])
            else:
                runs.append(line)
    return meta, task_meta, runs


def main():
    argv = [a for a in sys.argv[1:] if a != "--overwrite"]
    overwrite = "--overwrite" in sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit(__doc__)
    p_old, p_new, p_out = argv

    for p in (p_old, p_new):
        if not os.path.exists(p):
            raise SystemExit(f"输入文件不存在：{p}")

    m_old, t_old, r_old = load(p_old)
    m_new, t_new, r_new = load(p_new)
    if not r_old and not r_new:
        raise SystemExit("两份输入都没有 run 记录，没什么可合并的。")

    budgets_old = m_old.get("budgets")
    budgets_new = m_new.get("budgets")
    if budgets_old and budgets_new and budgets_old != budgets_new:
        raise SystemExit(f"预算档不一致，不能合并：{budgets_old} vs {budgets_new}")

    # 重复 (task, strategy, seed) 会让合并后重复计数 —— 显式拦下
    check_no_overlap(r_old, r_new, os.path.basename(p_old), os.path.basename(p_new))

    # 输出文件已有数据 → 默认拒绝（与 run_experiments.py 同一套保护）
    guard_out_path(p_out, overwrite, what="合并结果")

    meta = dict(m_old)
    # 任务顺序取并集（保持先旧后新），便于与一次性跑完的结果对照
    order = list(m_old.get("task_order", []))
    for t in m_new.get("task_order", []):
        if t not in order:
            order.append(t)
    meta["task_order"] = order
    meta["merged_from"] = [os.path.basename(p_old), os.path.basename(p_new)]

    task_meta = dict(t_old)
    task_meta.update(t_new)

    with open(p_out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for line in r_old + r_new:
            f.write(line + "\n")
        f.write(json.dumps({"task_meta": task_meta}, ensure_ascii=False) + "\n")

    tasks = {json.loads(l)["task"] for l in r_old + r_new}
    print(f"[merge] {p_out}")
    print(f"  run 数 = {len(r_old)} + {len(r_new)} = {len(r_old)+len(r_new)}")
    print(f"  任务数 = {len(tasks)}  |  预算档 = {budgets_old}")


if __name__ == "__main__":
    main()
