# -*- coding: utf-8 -*-
"""样本外任务的属性（AUC / r_proxy / 峰数）—— 与 analyze.probe_one 完全同一口径。

为什么单独算：analyze.task_props 有缓存且只认主臂/消融臂的登记任务，
样本外任务不在缓存里；直接调用同一个函数（不改它）保证 AUC 口径逐字一致。
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "unzipped", "agent_lab"))

from analyze import probe_one  # noqa: E402

ABL = os.path.join(HERE, "data", "ablate")
SKIP = {"ablate_manifest.csv"}


def main():
    rows = []
    for fn in sorted(os.listdir(ABL)):
        if not fn.endswith(".csv") or fn in SKIP:
            continue
        name = fn[:-4]
        df = pd.read_csv(os.path.join(ABL, fn))
        p = probe_one(df)
        base, op = name.split("__", 1)
        rows.append({"task": name, "base": base, "operator": op, **p})
        print(f"  {name:<36} AUC={p['separability_auc']:.3f} "
              f"r={p['r_proxy']:.3f} modes={p['n_modes']} dim={p['dim']}", flush=True)
    out = pd.DataFrame(rows).sort_values(["base", "separability_auc"])
    out.to_csv(os.path.join(HERE, "holdout_props.csv"), index=False)
    print(f"\n[save] {os.path.join(HERE, 'holdout_props.csv')}")
    print(out[["task", "separability_auc", "r_proxy", "n_modes"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
