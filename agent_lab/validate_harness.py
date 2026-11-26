# -*- coding: utf-8 -*-
"""验证 1：探针 harness 忠实性 —— 在本地默认栈下用 sbstack_probe 的同一套调用
（budget 200 直跑）复算本地文件（budget 400）的抽样 run，比对 chosen 前缀。
若不一致 → harness 有 bug 或文件由不同代码产生，翻转率结论不可信。

⚠️ 线程数必须与生产一致：run_experiments.py 生产时 n_threads = cpu_count // n_proc
= 8 // 6 = 1（meta.n_proc=6），故这里也必须限 1 线程。实测不限线程（8 线程）时
个别 GP run 会因归约顺序不同而翻转（如 log_gvrh__elem20/gp_ei/7）——
BLAS 线程数本身也是浮点环境变量，只是敏感性比库版本稀有得多。

用法：python validate_harness.py [group] [n_sample]
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from threadpoolctl import threadpool_limits

from config import N_INIT, SEED, discover_tasks
from run_experiments import _task_arrays, derive_seed
from trial import run_trial

GROUP = sys.argv[1] if len(sys.argv) > 1 else "gate_ablate"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ARM = "main" if GROUP == "gate" else "ablate"

records = []
with open(os.path.join("results", f"runs_{GROUP}.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and '"strategy"' in line:
            records.append(json.loads(line))

rng = random.Random(42)
# 分层抽样：每个策略抽一些，保证覆盖 4 种策略
by_strat = {}
for r in records:
    by_strat.setdefault(r["strategy"], []).append(r)
sample = []
for s, rs in by_strat.items():
    sample.extend(rng.sample(rs, min(N // len(by_strat) + 1, len(rs))))

tasks_cache = {}
ok = bad = 0
for r in sample:
    key = (r["task"], r["use_gate"], ARM)
    if key not in tasks_cache:
        tasks_cache[key] = _task_arrays(r["task"], r["use_gate"], ARM)
    task, X_std, y, cost, thr = tasks_cache[key]
    seed = derive_seed(SEED, r["task"], r["strategy"], r["seed"])
    with threadpool_limits(limits=1):  # 与生产一致（n_proc=6 → 1 线程/进程）
        out = run_trial(X_std, y, cost, thr, task, 200, seed, r["strategy"],
                        n_init=N_INIT)
    k = len(out["chosen"])
    if r["chosen"][:k] == out["chosen"]:
        ok += 1
    else:
        bad += 1
        div = next((j for j in range(min(k, len(r["chosen"])))
                    if r["chosen"][j] != out["chosen"][j]), None)
        print(f"  ❌ {r['task']}/{r['strategy']}/{r['seed']}: div=第{div}步")

print(f"harness 忠实性（本地默认栈，budget 200 直跑 vs 文件 budget 400 前缀）："
      f"一致 {ok} | 不一致 {bad}")
