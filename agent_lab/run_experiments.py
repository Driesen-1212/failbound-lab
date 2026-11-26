# -*- coding: utf-8 -*-
"""主实验循环 v3 —— 多进程并行版（统计口径与 v2 完全一致）

关键设计：
  1. 只跑到最大预算，完整记录"选择顺序"；分析阶段对任意预算档直接截断，无需重复跑。
  2. 每个任务一个独立数据文件（data/scan + data/tasks），命中阈值按 top-q 分位现算。
  3. 门控开/关分别写 runs_gate.jsonl / runs_nogate.jsonl，供 H4 对比。
  4. **多进程并行**：外层次序固定（任务 → 策略 → 种子），内层用 imap（保序）并行。
     输出文件的行序与单进程版逐字节一致，便于 diff 验证；只是更快。
     ⚠️ 并行只改变"算得多快"，不改变任何一次 run 的随机数、选择顺序或指标。

用法：
    python run_experiments.py                       # 门控组，30 种子，全 18 任务
    python run_experiments.py no-gate 30            # 等成本组（H1/H2 主口径）
    python run_experiments.py gate 30 3             # 只跑前 3 个任务（冒烟）
    python run_experiments.py gate 30 3 --proc 4    # 指定并行度

    # 省算力：预算上限 200（实测约 9.5 倍加速，事前预估 11 倍），种子一个不减
    python run_experiments.py no-gate 30 --max-budget 200 --proc 4

    # 沙箱会话限时：把任务拆 3 片，每个会话领一片，跑完合并
    python run_experiments.py gate 30 --max-budget 200 --shard 0/3
    python run_experiments.py gate 30 --max-budget 200 --shard 1/3
    python run_experiments.py gate 30 --max-budget 200 --shard 2/3
    python merge_shards.py
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (BUDGETS, GP_KAPPA, N_INIT, N_SEEDS, OUT_DIR, SEED,
                    STRATEGIES, discover_tasks)
from cost_model import cost_vector, gate_pool, threshold_of
from strategies import standardize
from trial import run_trial

# ⚠️ 这里的 MAX_BUDGET 只是「直接调用 run_one() 时」的默认值。
# 🔴 曾经的真 bug（10/16 修）：子进程 _worker 以前直接读这个全局，而它是恒定的
#    max(BUDGETS)=400 —— 于是 `--max-budget 200` 只改了 meta["budgets"]（分析端据此截断），
#    实验却照样跑到 400，**说好的 9.5 倍算力节省一点没发生**，沙箱 15 小时危机原封不动。
# ✅ 现在预算上限作为 job 元组的一员显式传给子进程，不再依赖任何全局。
#    安全性：run_one 的选择序列与预算视野无关（`while spent < budget`），
#    所以「真跑到 200」与「跑到 400 再截断到 200」逐点等价，旧数据不受影响。
MAX_BUDGET = int(max(BUDGETS))

# ---------- BLAS 线程数 ----------
# ⚠️ 不要把 BLAS 线程锁成 1！实测：8 核 8 进程 × 1 线程 = 每个 GEMM 慢 8 倍，
#    反而比不限制更慢。正确的做法是 每个进程的 BLAS 线程数 = 总核数 / 进程数，
#    这样 8 个进程加起来正好占满 8 个核，既不超订也不闲置。
#    用 threadpoolctl 在运行时限制（sklearn 的硬依赖，一定装了）。
_LIMITER = None


def _init_worker(n_threads):
    global _LIMITER
    try:
        from threadpoolctl import threadpool_limits
        _LIMITER = threadpool_limits(limits=max(1, int(n_threads)))
    except Exception:
        _LIMITER = None


# ---------- 子进程内的一次性任务数据缓存 ----------
# 为什么需要：一次 run 要反复用同一份 (X_std, y, cost, thr)。
# 若把 3000×88 的数组逐个 job 序列化传给子进程，IPC 开销会吃掉并行收益；
# 若每个 job 都重新读 CSV，又要读 2000+ 次。折中：子进程按任务名缓存，每任务只读一次。
_CACHE = {}


def _task_arrays(tname, use_gate, arm="main"):
    # key 必须带 arm：主臂与消融臂存在同名前缀以外的任务，混用会查错表（曾 KeyError）
    key = (tname, use_gate, arm)
    if key not in _CACHE:
        task = discover_tasks(arm=arm)[tname]
        df = pd.read_csv(task["path"])
        pool = gate_pool(df, use_gate=use_gate)
        feat_cols = [c for c in pool.columns if c.startswith("f")]
        X_std = standardize(pool[feat_cols].values.astype(float))
        y = pool[task["target_col"]].values.astype(float)
        thr = threshold_of(y, task)
        cost = cost_vector(pool, use_gate=use_gate)
        _CACHE[key] = (task, X_std, y, cost, thr)
    return _CACHE[key]


def derive_seed(base, *parts):
    """由主种子派生互不重复的种子，保证整体可复现。

    ⚠️ 原实现用 Python 内置 hash(str)，受 PYTHONHASHSEED 随机化影响，
    同一份代码两次运行会得到不同种子 —— 与"可复现性"卖点直接冲突。
    改用 blake2b，跨进程、跨机器稳定。
    """
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return (base + int.from_bytes(digest, "big")) % (2 ** 31)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------- 写前保护 ----------
# 🔴 曾经的事故（10/17）：main() 用 open(out_path, "w") 直接覆盖，
#    一个被中途 kill 的补跑任务把 2160 run 的完整数据冲成 168 run 残片。
# ✅ 保护逻辑统一在 out_guard.py（merge_shards / merge_runs / run_new_tasks 共用）：
#    已有 run 记录 → 默认拒绝；显式 --overwrite → 先自动备份再覆盖。
from out_guard import count_existing_runs, guard_out_path  # noqa: E402


def run_one(X_std, y, cost, task, threshold, strategy, budget, seed, use_gate=True):
    """一次 run。循环体已统一到 trial.run_trial（见该文件头说明）。

    保留本函数是为了兼容 _worker 与既有调用点；返回字段与重构前**完全一致**
    （已用 verify_trial.py 在 6 组 × 3 实现上逐点验证）。
    """
    # ⚠️ 注意参数次序：run_one 是 (..., task, threshold, ...)，
    #    run_trial 是 (..., thr, task, ...)。写反不会报错，只会静默算错 ——
    #    正是本次重构踩到的坑，靠 verify_trial.py 才抓出来。
    r = run_trial(X_std, y, cost, threshold, task, budget, seed, strategy)
    # 字段顺序与旧版一致，保证 jsonl 行的键序不变（便于 diff）
    return {
        "chosen": r["chosen"], "y_seq": r["y_seq"], "hits": r["hits"],
        "cost_seq": r["cost_seq"], "n_hits": r["n_hits"], "ttfh": r["ttfh"],
        "n_exp": r["n_exp"], "spent": r["spent"],
    }


def _worker(job):
    """子进程入口：跑一次 run，返回记录。

    job 元组末位 max_b 是本次运行的预算上限（来自 --max-budget，不是全局常量）。
    """
    tname, strategy, s, use_gate, arm, max_b = job
    task, X_std, y, cost, thr = _task_arrays(tname, use_gate, arm)
    seed = derive_seed(SEED, tname, strategy, s)
    res = run_one(X_std, y, cost, task, thr, strategy,
                  int(max_b), seed, use_gate=use_gate)
    return {"task": tname, "strategy": strategy, "seed": s,
            "use_gate": bool(use_gate), **res}


def main(use_gate=True, n_seeds=N_SEEDS, task_limit=None, n_proc=None,
         arm="main", max_budget=None, shard=None, threads=None,
         overwrite=False, out_suffix=""):
    """跑一组实验。

    out_suffix：给输出文件名加后缀（如 "_new"），用于「补跑任务写到独立文件」。
    这样增量补跑永远不会碰已有的完整数据文件。
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = discover_tasks(arm=arm)
    if task_limit:
        tasks = dict(list(tasks.items())[:task_limit])

    # ---- 分片 ----------------------------------------------------------
    # 为什么需要：沙箱单次会话有时长上限。全量一组 ≈ 2000+ run，
    # 一次会话跑不完就得从零重来。分片后每个会话只领走 1/N 的任务（按任务索引取模），
    # 各片独立成文件、互不覆盖；会话被中断也只丢当前片，前面的片已经落盘。
    # 领完所有片后用 merge_shards.py 合并（meta 取最后一版，task_meta 累加合并）。
    shard_tag = ""
    if shard:
        k, n = int(shard[0]), int(shard[1])
        if not (0 <= k < n):
            raise SystemExit(f"--shard 需要 0 <= K < N，收到 {k}/{n}")
        names = list(tasks)
        tasks = {t: tasks[t] for i, t in enumerate(names) if i % n == k}
        shard_tag = f".shard{k}of{n}"
        if not tasks:
            raise SystemExit(f"--shard {k}/{n} 没有分到任何任务（共 {len(names)} 个）")

    # ⚠️ 为什么需要 --max-budget：实测沙箱单次 run（预算 400）要 25.3 秒，
    #    全量 gate 30 = 15 小时，超出沙箱会话预算。GP 的单步开销随已观测数**三次方**增长，
    #    把预算上限从 400 降到 200 实测能省约 9.5 倍时间（事前预估 11 倍），而统计上只损失最右一个预算档。
    #    预算档写进 meta["budgets"]，分析端从这里读，两端永远一致。
    budgets = tuple(b for b in BUDGETS if (max_budget is None or b <= max_budget))
    if not budgets:
        raise SystemExit(f"--max-budget={max_budget} 过小，最小的预算档是 {min(BUDGETS)}")
    max_b = int(max(budgets))

    suffix = "" if arm == "main" else f"_{arm}"
    out_path = os.path.join(
        OUT_DIR, f"runs_{'gate' if use_gate else 'nogate'}{suffix}"
                 f"{out_suffix}{shard_tag}.jsonl")
    guard_out_path(out_path, overwrite)

    t0 = time.time()
    n_runs = 0
    n_proc = n_proc or max(1, min((os.cpu_count() or 2) - 1, 8))
    print(f"[start] arm={arm} use_gate={use_gate} tasks={len(tasks)} seeds={n_seeds} "
          f"procs={n_proc} max_budget={max_b} budgets={list(budgets)} "
          f"→ {os.path.basename(out_path)}")

    with open(out_path, "w", encoding="utf-8") as f:
        meta = {"arm": arm, "use_gate": bool(use_gate), "max_budget": max_b,
                "budgets": list(budgets),
                "shard": [int(shard[0]), int(shard[1])] if shard else None,
                "n_init": N_INIT, "seed": SEED, "n_seeds": n_seeds,
                "hit_q": 0.90, "n_proc": n_proc,
                "task_order": list(tasks),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        f.flush()

        task_meta = {}
        # 每个进程的 BLAS 线程数 = 总逻辑核 / 进程数，合起来正好占满。
        # ⚠️ 注意 os.cpu_count() 数的是**逻辑核**：本机 i5-1135G7 报 8，实际只有 4 个物理核，
        #    所以 --proc 6 会是 6 个单线程进程抢 4 个物理核（超订 1.5×）。
        #    物理核少的机器上，宁可 --proc 4 --threads 2。可用 --threads 直接覆盖。
        n_threads = threads or max(1, (os.cpu_count() or 2) // max(1, n_proc))
        pool = Pool(processes=n_proc, initializer=_init_worker,
                    initargs=(n_threads,))
        try:
            for tname, task in tasks.items():
                # 任务级元信息（在主进程算，避免子进程重复读 CSV）
                df = pd.read_csv(task["path"])
                p = gate_pool(df, use_gate=use_gate)
                yv = p[task["target_col"]].values.astype(float)
                thr = threshold_of(yv, task)
                task_meta[tname] = {
                    "n_pool": int(len(p)),
                    "n_pool_before_gate": int(len(df)),
                    "threshold": float(thr),
                    "dim": int(sum(1 for c in p.columns if c.startswith("f"))),
                    "sha256": file_sha256(task["path"]),
                    "hit_rate": float((yv >= thr).mean()),
                }

                jobs = [(tname, strategy, s, use_gate, arm, max_b)
                        for strategy in STRATEGIES
                        for s in range(n_seeds)]
                # imap 保序 → 输出行序与单进程版一致
                for rec in pool.imap(_worker, jobs, chunksize=1):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_runs += 1
                # 任务级元信息**随任务逐条落盘**（原来只在结尾写一次）：
                # 进程被中断时已完成任务的元信息不丢，分析端照样能读。
                f.write(json.dumps({"task_meta": {tname: task_meta[tname]}},
                                   ensure_ascii=False) + "\n")
                f.flush()
                el = time.time() - t0
                print(f"[done] {tname:<34} n_pool={len(p):>5} thr={thr:+.3f} "
                      f"| {n_runs} runs | {el:.0f}s "
                      f"| eta {el / max(1, len(task_meta)) * (len(tasks) - len(task_meta)):.0f}s")
        finally:
            pool.close()
            pool.join()

        # task_meta 已在每个任务跑完后逐条写入，这里不再汇总重写：
        # 会话中断时也不会丢元信息。

    print(f"[save] {out_path} | runs={n_runs} | {time.time()-t0:.0f}s")
    return out_path


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("group", nargs="?", default="gate",
                    help="gate（默认，真实约束）| no-gate（等成本，H1/H2 主口径）")
    ap.add_argument("seeds", nargs="?", type=int, default=N_SEEDS)
    ap.add_argument("limit", nargs="?", type=int, default=None,
                    help="只跑前 N 个任务（冒烟用）")
    ap.add_argument("--arm", default="main", choices=["main", "ablate"],
                    help="main=预注册主实验；ablate=描述符消融稳健性臂")
    ap.add_argument("--max-budget", type=int, default=None,
                    help="预算上限（如 200 = 只跑 25/50/100/200 四档），省算力用")
    ap.add_argument("--proc", type=int, default=None,
                    help="并行进程数，默认 min(CPU-1, 8)")
    ap.add_argument("--threads", type=int, default=None,
                    help="每个进程的 BLAS 线程数，默认 逻辑核数/proc（物理核少的机器建议显式设 2）")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="只跑第 K 片（共 N 片），按任务索引取模；沙箱会话限时用。"
                         "全部片跑完后用 merge_shards.py 合并")
    ap.add_argument("--overwrite", action="store_true",
                    help="目标文件已有 run 记录时仍要覆盖（会先自动备份到 results/_backup/）")
    a = ap.parse_args(argv)
    shard = None
    if a.shard:
        parts = a.shard.replace("of", "/").split("/")
        if len(parts) != 2:
            raise SystemExit(f"--shard 格式应为 K/N（如 0/3），收到 {a.shard}")
        shard = (int(parts[0]), int(parts[1]))
    return (a.group != "no-gate"), a.seeds, a.limit, a.proc, a.arm, \
        a.max_budget, shard, a.threads, a.overwrite


if __name__ == "__main__":
    gate, seeds, limit, proc, arm, maxb, shard, threads, ow = parse_args(sys.argv[1:])
    main(use_gate=gate, n_seeds=seeds, task_limit=limit, n_proc=proc,
         arm=arm, max_budget=maxb, shard=shard, threads=threads, overwrite=ow)
