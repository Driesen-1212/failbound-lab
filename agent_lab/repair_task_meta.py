# -*- coding: utf-8 -*-
"""修复 runs_nogate.jsonl 缺失的 task_meta 行 + 验证再生算法逐字节正确。

背景：本地 runs_nogate.jsonl 只有 2162 行（1 meta + 2160 run + 1 task_meta），
而沙箱交付版为 2179 行（1 meta + 2160 run + 18 task_meta）。差的 17 行是
「每任务 120 条 run 后跟一条的 task_meta 元信息头」——10/17 数据事故恢复时的
合并残留（唯一幸存的那行被追加到了文件末尾，内容还是第一个任务的）。

task_meta 的每个字段都由数据确定性算出（n_pool/阈值/dim/数据文件sha256/命中率），
因此可以无损再生。本脚本分两步：
  第 1 步（证明）：对 runs_gate / runs_gate_ablate / runs_nogate_ablate 三个完整文件，
    用当前代码再生全部 task_meta 行，与文件中的现有行逐字节比对 —— 必须全部一致。
  第 2 步（修复）：备份 runs_nogate.jsonl 后，把 18 条再生行插到各自任务的 120 条
    run 之后，删掉末尾的残留行；随后验证 run 行逐字节未动、总行数 2179。

用法：python repair_task_meta.py            # 证明 + 修复 + 验证
      python repair_task_meta.py --dry-run  # 只证明，不动文件
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from config import discover_tasks
from cost_model import gate_pool, threshold_of
from run_experiments import file_sha256

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def regen_task_meta(arm, use_gate):
    """按 run_experiments.py 写入时的同一算法再生 task_meta 行文本。"""
    tasks = discover_tasks(arm=arm)
    out = {}
    for tname, task in tasks.items():
        df = pd.read_csv(task["path"])
        p = gate_pool(df, use_gate=use_gate)
        yv = p[task["target_col"]].values.astype(float)
        thr = threshold_of(yv, task)
        out[tname] = json.dumps(
            {"task_meta": {tname: {
                "n_pool": int(len(p)),
                "n_pool_before_gate": int(len(df)),
                "threshold": float(thr),
                "dim": int(sum(1 for c in p.columns if c.startswith("f"))),
                "sha256": file_sha256(task["path"]),
                "hit_rate": float((yv >= thr).mean()),
            }}}, ensure_ascii=False)
    return out


def step1_prove():
    """在三个完整文件上证明再生算法逐字节正确。"""
    print("=== 第 1 步 · 证明再生算法逐字节正确（三个完整文件）===")
    cases = [("runs_gate.jsonl", "main", True),
             ("runs_gate_ablate.jsonl", "ablate", True),
             ("runs_nogate_ablate.jsonl", "ablate", False)]
    all_ok = True
    for fname, arm, gate in cases:
        path = os.path.join(RESULTS, fname)
        regen = regen_task_meta(arm, gate)
        existing, n_match, n_total = {}, 0, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                d = json.loads(line)
                if "task_meta" in d:
                    tname = next(iter(d["task_meta"]))
                    existing[tname] = line
        for tname, line in existing.items():
            n_total += 1
            if regen.get(tname) == line:
                n_match += 1
            else:
                print(f"  ❌ {fname} / {tname} 再生结果与文件不一致")
                print(f"     文件: {line[:160]}")
                print(f"     再生: {regen.get(tname, '<缺失>')[:160]}")
        ok = n_match == n_total == 18
        all_ok &= ok
        print(f"  {fname:26s} task_meta 行 {n_match}/{n_total} 逐字节一致 "
              f"{'✅' if ok else '❌'}")
    return all_ok


def step2_repair(dry_run=False):
    """修复 runs_nogate.jsonl：补齐 18 条 task_meta 到正确位置。"""
    fname = "runs_nogate.jsonl"
    path = os.path.join(RESULTS, fname)
    regen = regen_task_meta("main", False)   # nogate, main arm

    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    meta_lines = [l for l in lines if '"meta"' in l[:20]]
    run_lines = [l for l in lines if '"strategy"' in l]
    tm_lines = [l for l in lines if '"task_meta"' in l]
    print(f"\n=== 第 2 步 · 修复 {fname} ===")
    print(f"  修复前: {len(lines)} 行 = {len(meta_lines)} meta + "
          f"{len(run_lines)} run + {len(tm_lines)} task_meta")

    # 任务块切分：run 按 task_order 顺序连续排列，每任务 4 策略 × 30 种子 = 120 条
    blocks, cur, cur_task = [], [], None
    for l in run_lines:
        d = json.loads(l)
        if d["task"] != cur_task:
            if cur:
                blocks.append((cur_task, cur))
            cur, cur_task = [], d["task"]
        cur.append(l)
    if cur:
        blocks.append((cur_task, cur))
    print(f"  识别到 {len(blocks)} 个任务块，各 {sorted(set(len(b[1]) for b in blocks))} 条 run")

    out = [meta_lines[0]]
    for tname, blk in blocks:
        out.extend(blk)
        out.append(regen[tname])       # 每任务 120 run 后插一条 task_meta

    # 校验：run 行集合与顺序必须与原文件完全一致
    orig_runs = [l for l in lines if '"strategy"' in l]
    new_runs = [l for l in out if '"strategy"' in l]
    assert orig_runs == new_runs, "run 行发生变动！中止"
    assert len([l for l in out if '"task_meta"' in l]) == 18

    print(f"  修复后: {len(out)} 行 = 1 meta + {len(new_runs)} run + 18 task_meta")
    if dry_run:
        print("  --dry-run：不写盘")
        return True

    backup = path + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    print(f"  已备份 → {os.path.basename(backup)}")

    with open(path, "w", encoding="utf-8") as f:
        for l in out:
            f.write(l + "\n")

    # 终检
    with open(path, encoding="utf-8") as f:
        chk = [l.rstrip("\n") for l in f if l.strip()]
    n_run = len([l for l in chk if '"strategy"' in l])
    n_tm = len([l for l in chk if '"task_meta"' in l])
    keys = set()
    for l in chk:
        if '"strategy"' in l:
            d = json.loads(l)
            keys.add((d["task"], d["strategy"], d["seed"], bool(d["use_gate"])))
    ok = (len(chk) == 2179 and n_run == 2160 and n_tm == 18 and len(keys) == 2160)
    print(f"  终检: {len(chk)} 行 | run {n_run} | task_meta {n_tm} | "
          f"唯一键 {len(keys)} {'✅' if ok else '❌'}")
    return ok


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    ok = step1_prove()
    if not ok:
        print("\n❌ 再生算法未能逐字节复现，放弃修复（避免污染数据）")
        sys.exit(1)
    ok = step2_repair(dry_run=dry)
    print("\n" + ("✅ 全部完成" if ok else "❌ 修复失败"))
    sys.exit(0 if ok else 1)
