# -*- coding: utf-8 -*-
"""本地复算全部任务的 n_pool / thr（门控双口径），输出 CSV 供与沙箱 [done] 行比对。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from config import discover_tasks
from cost_model import gate_pool, threshold_of

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "local_task_params.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
rows = []
for arm in ("main", "ablate"):
    ts = discover_tasks(arm=arm)
    for t in sorted(ts):
        df = pd.read_csv(ts[t]["path"])
        rec = {"arm": arm, "task": t}
        for g in (True, False):
            pool = gate_pool(df, use_gate=g)
            y = pool[ts[t]["target_col"]].values.astype(float)
            rec[f"n_pool_gate{int(g)}"] = len(pool)
            rec[f"thr_gate{int(g)}"] = round(float(threshold_of(y, ts[t])), 3)
        rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(df.to_string(index=False))
print("\nwritten:", out)
