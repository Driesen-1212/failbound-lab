# -*- coding: utf-8 -*-
"""验证 2：X_std 跨栈逐比特性 —— 数据管线（pandas read_csv + standardize）
在不同库版本下是否产生比特级相同的标准化描述符。

若 X_std 不同 → farthest_point/BO 的翻转源自数据管线浮点差异（ pandas 解析 /
numpy 归约顺序）；若相同 → 差异在距离/GP 计算内部。

用法：python dump_xstd_hash.py <out.json>   （两个解释器各跑一次，比对哈希）
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy
import pandas
import scipy
import sklearn

from config import discover_tasks
from cost_model import gate_pool
from run_experiments import standardize

OUT = sys.argv[1] if len(sys.argv) > 1 else "xstd_hash.json"

data = {"env": {"numpy": numpy.__version__, "pandas": pandas.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "tasks": {}}
for arm in ("main", "ablate"):
    for tname, task in sorted(discover_tasks(arm=arm).items()):
        df = pandas.read_csv(task["path"])
        pool = gate_pool(df, use_gate=True)
        feat_cols = [c for c in pool.columns if c.startswith("f")]
        raw = pool[feat_cols].values.astype(float)
        X_std = standardize(raw)
        # 三个粒度：原始矩阵、标准化矩阵、逐行首元素
        data["tasks"][f"{arm}/{tname}"] = {
            "raw_sha": hashlib.sha256(
                numpy.ascontiguousarray(raw).tobytes()).hexdigest()[:16],
            "xstd_sha": hashlib.sha256(
                numpy.ascontiguousarray(X_std).tobytes()).hexdigest()[:16],
            "xstd_00": repr(float(X_std[0, 0])),
            "n": int(X_std.shape[0]), "d": int(X_std.shape[1]),
        }

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print(f"{data['env']}")
n_diff = sum(1 for v in data["tasks"].values())
print(f"{len(data['tasks'])} tasks → {OUT}")
