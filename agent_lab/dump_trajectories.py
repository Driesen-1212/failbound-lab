# -*- coding: utf-8 -*-
"""导出 3 个分歧 run 的完整轨迹（chosen + cost_seq），用于跨环境找第一个分歧点。

用法（分别用两个解释器跑，输出到不同文件）：
  default: python dump_trajectories.py out_default.json
  sbstack: python dump_trajectories.py out_sbstack.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy
import scipy
import sklearn

from config import SEED
from run_experiments import _task_arrays, derive_seed
from trial import run_trial

CASES = [("matbench_dielectric__pca2", "gp_ei", 9),
         ("matbench_expt_gap__rp2", "gp_ucb", 17),
         ("matbench_log_gvrh__elem20", "gp_ei", 9)]

BUDGET = 200
OUT = sys.argv[1] if len(sys.argv) > 1 else "traj.json"

data = {"env": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "cases": []}
for tname, strategy, s in CASES:
    task, X_std, y, cost, thr = _task_arrays(tname, True, "ablate")
    seed = derive_seed(SEED, tname, strategy, s)
    r = run_trial(X_std, y, cost, thr, task, BUDGET, seed, strategy)
    # 复刻 while spent < budget 的截断口径
    spent = 0.0
    for c in r["cost_seq"]:
        if spent >= BUDGET:
            break
        spent += c
    data["cases"].append({"task": tname, "strategy": strategy, "seed_index": s,
                          "chosen": r["chosen"], "cost_seq": r["cost_seq"],
                          "truncated_spent": spent, "n_exp": r["n_exp"]})

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print(f"{data['env']}")
for c in data["cases"]:
    print(f"  {c['task']}/{c['strategy']}/s{c['seed_index']}: "
          f"n_exp={c['n_exp']} truncated={c['truncated_spent']}")
print(f"→ {OUT}")
