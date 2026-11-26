# -*- coding: utf-8 -*-
"""根因定位：在沙箱同版本库栈下复现 gate_ablate 组 3 个分歧 run。

实验设计（三组对照，逐个排除变量）：
  A = 本地默认栈   Python 3.13 + numpy 2.5.2 / sklearn 1.9.1 / scipy 1.16.x  → 205.0（已观测）
  B = sbstack      Python 3.13 + numpy 2.3.0 / sklearn 1.8.0 / scipy 1.15.3  → 本脚本
  C = 沙箱         Python 3.11 + numpy 2.3.0 / sklearn 1.8.0 / scipy 1.15.3  → 204.0（承诺值）

若 B == 204.0：Python 版本被排除（A/B 同为 3.13 结果不同）→ 差异驱动 = 库版本。
若 B == 205.0：库版本不足以翻转，剩余变量为 Python 3.11 vs 3.13 本身。
两种结果都缩小了根因范围，且都不触碰"代码/数据不一致"的解释空间。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy
import pandas
import scipy
import sklearn

from config import SEED, discover_tasks
from run_experiments import _task_arrays, derive_seed
from trial import run_trial

CASES = [("matbench_dielectric__pca2", "gp_ei", 9),
         ("matbench_expt_gap__rp2", "gp_ucb", 17),
         ("matbench_log_gvrh__elem20", "gp_ei", 9)]

BUDGET = 200
SANDBOX_CLAIM = 204.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "rootcause_sbstack.json")


def truncate_spent(cost_seq, budget=BUDGET):
    spent = 0.0
    for c in cost_seq:
        if spent >= budget:
            break
        spent += c
    return spent


def main():
    print("=== 运行环境 ===")
    print(f"python  {sys.version.split()[0]}")
    print(f"numpy   {numpy.__version__}")
    print(f"scipy   {scipy.__version__}")
    print(f"sklearn {sklearn.__version__}")
    print(f"pandas  {pandas.__version__}")
    print(f"\n=== 复现 3 个分歧 run（本地=205.0，沙箱承诺={SANDBOX_CLAIM}）===\n")

    results = []
    for tname, strategy, s in CASES:
        task, X_std, y, cost, thr = _task_arrays(tname, True, "ablate")
        seed = derive_seed(SEED, tname, strategy, s)
        r = run_trial(X_std, y, cost, thr, task, BUDGET, seed, strategy)
        spent = truncate_spent(r["cost_seq"])
        match = "✅ 复现沙箱 204.0" if spent == SANDBOX_CLAIM else (
            f"仍为本地值 {spent}" if spent == 205.0 else f"出现第三值 {spent}")
        print(f"{tname} / {strategy} / seed={s}: truncated_spent={spent}  {match}")
        results.append({"task": tname, "strategy": strategy, "seed_index": s,
                        "truncated_spent": spent, "sandbox_claim": SANDBOX_CLAIM,
                        "local_default": 205.0, "n_chosen": len(r["chosen"])})

    flipped = sum(1 for r in results if r["truncated_spent"] == SANDBOX_CLAIM)
    print(f"\n=== 结论：{flipped}/3 个 run 翻转为沙箱值 ===")
    if flipped == 3:
        print("根因证实：库版本级 GP 浮点差异（Python 版本已排除，A/B 同为 3.13）")
        verdict = "library_versions"
    elif flipped == 0:
        print("库版本不是驱动因素；剩余变量 = Python 3.11 vs 3.13 本身")
        verdict = "python_version_or_other"
    else:
        print("部分翻转：需逐 run 检查分歧点位置")
        verdict = "partial"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"env": {"python": sys.version.split()[0],
                           "numpy": numpy.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn.__version__, "pandas": pandas.__version__},
                   "cases": results, "flipped": flipped, "verdict": verdict},
                  f, ensure_ascii=False, indent=2)
    print(f"\n存档 → {OUT}")
    return 0 if flipped == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
