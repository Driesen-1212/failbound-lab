# -*- coding: utf-8 -*-
"""全组探测：在沙箱同版本库栈（sbstack venv）下重跑整个 gate 组，量化
「只换库版本」造成的轨迹翻转率与文件级 max spent。

用法：python sbstack_probe.py <group> <out.jsonl>
  group = gate        → arm=main,  use_gate=True   （对标 runs_gate.jsonl）
  group = gate_ablate → arm=ablate, use_gate=True  （对标 runs_gate_ablate.jsonl）

与本地默认栈结果（budget 400，L5 截断到 200）的比对见 compare_probe.py。
每个 run 独立（derive_seed 按 (task, strategy, seed) 派生），顺序不影响结果。
BLAS 固定 1 线程：forensic_gp_threading.py 已证明线程数不影响结果，
固定线程只是为了避免两个并行探测进程互相争抢。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy
import scipy
import sklearn

from config import N_INIT, N_SEEDS, SEED, STRATEGIES, discover_tasks
from run_experiments import _task_arrays, derive_seed
from trial import run_trial

GROUP = sys.argv[1]
OUT = sys.argv[2]
ARM, GATE = ("main", True) if GROUP == "gate" else ("ablate", True)
BUDGET = 200


def truncate_spent(cost_seq, budget=BUDGET):
    spent = 0.0
    for c in cost_seq:
        if spent >= budget:
            break
        spent += c
    return spent


def main():
    tasks = sorted(discover_tasks(arm=ARM))
    total = len(tasks) * len(STRATEGIES) * N_SEEDS
    print(f"env: python {sys.version.split()[0]} numpy {numpy.__version__} "
          f"scipy {scipy.__version__} sklearn {sklearn.__version__}", flush=True)
    print(f"group={GROUP} arm={ARM} gate={GATE} budget={BUDGET} "
          f"tasks={len(tasks)} total_runs={total}", flush=True)

    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for tname in tasks:
            task, X_std, y, cost, thr = _task_arrays(tname, GATE, ARM)
            for strategy in STRATEGIES:
                for s in range(N_SEEDS):
                    seed = derive_seed(SEED, tname, strategy, s)
                    r = run_trial(X_std, y, cost, thr, task, BUDGET, seed,
                                  strategy, n_init=N_INIT)
                    rec = {"task": tname, "strategy": strategy, "seed": s,
                           "arm": ARM, "use_gate": GATE, "budget": BUDGET,
                           "chosen": r["chosen"], "cost_seq": r["cost_seq"],
                           "n_exp": r["n_exp"],
                           "truncated_spent": truncate_spent(r["cost_seq"])}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 50 == 0:
                        f.flush()
                        print(f"  {n}/{total}", flush=True)
    print(f"DONE {n} runs → {OUT}", flush=True)


if __name__ == "__main__":
    main()
