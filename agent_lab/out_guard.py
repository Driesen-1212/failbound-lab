# -*- coding: utf-8 -*-
"""输出文件写前保护 —— 所有会写 runs_*.jsonl 的脚本统一走这里。

🔴 为什么要有这个文件（10/17 事故复盘）：
    run_experiments.main() 以前直接 `open(out_path, "w")`。
    一个「补跑 9 个新任务」的后台任务跑到一半被 kill，把 runs_nogate.jsonl
    从 2160 run（18 个任务）冲成 168 run（2 个残片）—— 完整数据没了，
    最后靠手工备份才救回来。根因是「覆盖模式 + 长任务 + 可中断」三者叠加。

✅ 现在的三条防线：
    1. 目标文件已有 run 记录 → 默认拒绝启动，给出三条安全替代方案；
    2. 显式 --overwrite 才允许覆盖，且覆盖前自动备份到 results/_backup/；
    3. 合并类脚本额外校验「内容维度一致」（预算档 / 任务重名），防止无声污染。

只依赖标准库，导入开销可忽略，任何脚本都能安全 import。
"""

import json
import os
import shutil
import time


def count_existing_runs(path):
    """返回已有文件里的 run 行数（0 表示不存在 / 空 / 不可读 / 无有效记录）。

    判定"是不是一条 run"：能被 json 解析且含 "strategy" 键。
    meta / task_meta 行不含该键，不会被计入。
    """
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if "strategy" in json.loads(line):
                        n += 1
                except Exception:
                    continue
    except Exception:
        return 0
    return n


def backup_file(path):
    """把 path 复制一份到同级 _backup/ 目录，返回备份路径。"""
    bak_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "_backup")
    os.makedirs(bak_dir, exist_ok=True)
    bak = os.path.join(bak_dir, f"{os.path.basename(path)}."
                                f"{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(path, bak)
    return bak


def guard_out_path(out_path, overwrite, what="输出文件"):
    """写前检查。已有 run 记录且未显式 overwrite → SystemExit。

    成功时返回备份路径（若做了备份），否则 None。
    """
    n = count_existing_runs(out_path)
    if n == 0:
        return None
    name = os.path.basename(out_path)
    if not overwrite:
        raise SystemExit(
            f"\n[拒绝覆盖] {name} 已有 {n} 条 run 记录。\n"
            f"  直接覆盖会永久丢失这些数据（10/17 真实发生过一次）。\n"
            f"  三条安全出路，任选其一：\n"
            f"    (1) 换输出名：加 --shard K/N 或改用 run_new_tasks.py 写到新文件，\n"
            f"        之后用 merge_shards.py / merge_runs.py 合并；\n"
            f"    (2) 人工确认后覆盖：加 --overwrite（会先自动备份到 results/_backup/）；\n"
            f"    (3) 只跑新增任务：用 run_new_tasks.py。\n")
    bak = backup_file(out_path)
    print(f"[backup] {what}覆盖前已备份 {n} 条旧记录 → {os.path.basename(bak)}")
    return bak


def check_no_overlap(runs_old, runs_new, label_old="旧", label_new="新"):
    """合并前检查两边是否有重复的 (task, strategy, seed) 三元组。

    重复意味着同一条 run 被写进两份文件，合并后会重复计数，
    统计上等于凭空把某些组合的权重翻倍 —— 必须显式报错而不是静默合并。
    """
    def keys(lines):
        s = set()
        for line in lines:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "strategy" in d:
                s.add((d.get("task"), d.get("strategy"), d.get("seed")))
        return s

    a, b = keys(runs_old), keys(runs_new)
    dup = a & b
    if dup:
        sample = sorted(dup)[:5]
        raise SystemExit(
            f"[{label_old}/{label_new} 重复] 有 {len(dup)} 个 (task, strategy, seed) "
            f"组合在两份文件里都出现，例如 {sample}。\n"
            f"  合并会重复计数。请先确认哪一份是权威数据，或用 --keep-new 只取新的一份。")
    return len(a) + len(b)
