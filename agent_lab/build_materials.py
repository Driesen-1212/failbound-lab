# -*- coding: utf-8 -*-
"""从 matbench v0.1 原始快照构建真实材料任务表（N2：数据落地）

设计要点（答辩会被问，务必理解）：

1. **不依赖 pymatgen / 不联网**。matbench 的 json.gz 是 pandas split-orient 结构，
   `structure` 列是 pymatgen Structure 的 dict 序列化，直接读 sites 即可取元素组成。
2. **所有任务共享同一个描述符空间** = 全部任务出现元素的并集（元素分数向量）。
   这样不同任务的"可学习性 AUC"才具有可比性 —— 这是画失效边界的前提。
3. **成本模型列全部来自真实化学式**：n_atoms = 化学式原子总数；
   has_rare = 是否含受限/贵金属元素。不再用合成数据的随机值。
4. 每个任务固定随机子采样到 POOL_SIZE（保证预算公平），并落盘 + 算 sha256。
"""

import gzip
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# ---------- 配置 ----------
# 路径由 config._find_data_root() 自动定位（支持 FAILBOUND_PROJECT 环境变量），
# 不再写死本机绝对路径，换机器/换布局都不会静默找不到数据。
from config import DATA_ROOT

MATBENCH_DIR = os.path.join(DATA_ROOT, "matbench")   # 原始 matbench .json.gz（不随冻结包分发）
OUT_DIR = os.path.join(DATA_ROOT, "tasks")
POOL_SIZE = 3000
SUBSET_SEED = 0

# 命中/门控用的受限元素（与 config.RARE_ELEMENTS 保持同一口径）
RARE_ELEMENTS = {"Pt", "Ir", "Rh", "Pd", "Au", "Ru", "Re", "Os"}

# 元素周期表原子序数（1-92），用于额外统计特征
_ATOMIC_NUMBER = {}
_SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
            "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
            "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
            "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split()
for _i, _s in enumerate(_SYMBOLS, start=1):
    _ATOMIC_NUMBER[_s] = _i

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]+)?")

# 只保留样本量足够做预算实验的任务
TASK_SPEC = {
    "matbench_dielectric": "n",
    "matbench_expt_gap": "gap expt",
    "matbench_glass": "gfa",
    "matbench_expt_is_metal": "is_metal",
    "matbench_log_kvrh": "log10(K_VRH)",
    "matbench_log_gvrh": "log10(G_VRH)",
    "matbench_perovskites": "e_form",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_formula(s):
    """'Fe0.620C0.000953Mn0.000521' / 'Al' -> {element: amount}"""
    out = {}
    for sym, num in _FORMULA_TOKEN.findall(s):
        if sym not in _ATOMIC_NUMBER:
            continue
        v = float(num) if num else 1.0
        out[sym] = out.get(sym, 0.0) + v
    return out


def parse_structure(obj):
    """pymatgen Structure dict -> {element: amount}（只取组成，忽略坐标）"""
    out = {}
    for site in obj.get("sites", []):
        for sp in site.get("species", []):
            el = sp.get("element")
            if el is None:
                continue
            occu = sp.get("occu", 1.0)
            out[el] = out.get(el, 0.0) + float(occu)
    return out


def load_task(name):
    path = os.path.join(MATBENCH_DIR, name + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        obj = json.load(fh)
    cols = obj["columns"]
    rows = obj["data"]
    df = pd.DataFrame(rows, columns=cols)
    return df, sha256_file(path)


def composition_series(df):
    if "composition" in df.columns:
        return df["composition"].apply(parse_formula)
    if "structure" in df.columns:
        return df["structure"].apply(parse_structure)
    raise KeyError("既无 composition 也无 structure 列")


def featurize(comps, vocab):
    """元素分数向量 + 元素种类数 + 原子序数加权均值/标准差"""
    n = len(comps)
    d = len(vocab)
    X = np.zeros((n, d), dtype=float)
    n_elem = np.zeros(n, dtype=float)
    z_mean = np.zeros(n, dtype=float)
    z_std = np.zeros(n, dtype=float)
    n_atoms = np.zeros(n, dtype=float)

    for i, c in enumerate(comps):
        tot = sum(c.values())
        if tot <= 0:
            continue
        n_atoms[i] = tot
        zs = np.array([_ATOMIC_NUMBER.get(e, 0) for e in c], dtype=float)
        ws = np.array(list(c.values()), dtype=float)
        ws = ws / ws.sum()
        for e, v in c.items():
            if e in vocab:
                X[i, vocab[e]] = v / tot
        n_elem[i] = len(c)
        z_mean[i] = float((zs * ws).sum())
        z_std[i] = float(np.sqrt(((zs - z_mean[i]) ** 2 * ws).sum()))

    return X, n_elem, z_mean, z_std, n_atoms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SUBSET_SEED)

    # ---- 第一遍：读全部任务，建立统一元素词汇表 ----
    raw = {}
    for name, target in TASK_SPEC.items():
        df, sha = load_task(name)
        comps = composition_series(df).tolist()
        y = pd.to_numeric(df[target], errors="coerce")
        keep = [i for i, c in enumerate(comps) if c and not np.isnan(y.iloc[i])]
        raw[name] = {"comps": [comps[i] for i in keep],
                     "y": np.asarray([y.iloc[i] for i in keep], dtype=float),
                     "sha": sha, "n_raw": len(df), "target": target}
        print(f"[load] {name:<24} n_raw={len(df):>6} usable={len(keep):>6} "
              f"sha={sha[:16]}")

    vocab = {}
    for name in TASK_SPEC:
        for c in raw[name]["comps"]:
            for e in c:
                if e not in vocab:
                    vocab[e] = len(vocab)
    print(f"\n[vocab] 统一描述符空间维度 = {len(vocab)} 种元素\n")

    # ---- 第二遍：逐任务特征化 + 固定子采样 + 落盘 ----
    manifest = []
    for name, spec in raw.items():
        n = len(spec["comps"])
        idx = rng.choice(n, size=min(POOL_SIZE, n), replace=False)
        idx.sort()
        comps = [spec["comps"][i] for i in idx]
        y = spec["y"][idx]

        X, n_elem, z_mean, z_std, n_atoms = featurize(comps, vocab)
        has_rare = np.array([any(e in RARE_ELEMENTS for e in c) for c in comps])

        df_out = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df_out["n_elements"] = n_elem
        df_out["z_mean"] = z_mean
        df_out["z_std"] = z_std
        df_out["target"] = y
        df_out["n_atoms"] = n_atoms
        df_out["has_rare"] = has_rare
        df_out["material_id"] = [f"{name}-{i:05d}" for i in range(len(df_out))]

        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        df_out.to_csv(out_path, index=False)
        out_sha = sha256_file(out_path)

        manifest.append({"task": name, "target_col": spec["target"],
                         "n_raw": spec["n_raw"], "n_pool": len(df_out),
                         "dim": X.shape[1] + 3, "src_sha256": spec["sha"],
                         "csv_sha256": out_sha, "n_rare": int(has_rare.sum()),
                         "y_mean": float(y.mean()), "y_std": float(y.std())})
        print(f"[save] {name:<24} pool={len(df_out):>5} sha={out_sha[:16]} "
              f"rare={int(has_rare.sum())}")

    man = pd.DataFrame(manifest)
    man_path = os.path.join(OUT_DIR, "tasks_manifest.csv")
    man.to_csv(man_path, index=False)

    vocab_path = os.path.join(OUT_DIR, "element_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(sorted(vocab, key=vocab.get), f, ensure_ascii=False)

    print(f"\n[manifest] {man_path}")
    print(man[["task", "n_raw", "n_pool", "dim", "y_mean", "y_std"]].to_string(index=False))
    return man


if __name__ == "__main__":
    main()
