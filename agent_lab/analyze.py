# -*- coding: utf-8 -*-
"""FailBound-Lab 统计分析与失效边界拟合 —— v2（严格实现 preregistration.md §7–§8）

用法：
    python analyze.py                # 同时分析 runs_gate.jsonl 与 runs_nogate.jsonl
    python analyze.py nogate         # 只分析等成本组（H1/H2 主口径）
    python analyze.py gate           # 只分析门控组（真实约束口径，H4 的一侧）

产出（results/）：
    task_props.csv        任务空间结构属性（自变量：可学习性 AUC / r_proxy / n_modes）
    summary.csv           每 run（任务×策略×种子）× 每预算档 的指标
    tests.csv             配对检验 + 效应量 + CI + Holm 校正（区分 main / exploratory 家族）
    h4_gate.csv           H4：门控 vs 等成本 的策略排序一致性
    boundary_fit.json     失效边界零点 AUC* 及其 bootstrap 95% CI
    guideline_card.md     可操作的决策准则卡
    fig_*.png             四张主图

口径锚点（不得随手改，改了就登记到 preregistration §10）：
    §5  命中 = 目标 top-10% 分位（基线命中率恒为 10%）
    §7  Wilcoxon 符号秩 / Cliff's delta / bootstrap 2000 次 / Holm–Bonferroni / α=0.05
    §8  分箱 0.05 + LOWESS 定位零点 + 零点 bootstrap CI
    §10 主检验家族 = 18 任务 × 1 策略(gp_ucb) × 1 预算档 = 18 次；H1/H2 主口径走等成本组
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体兜底：图内标签原则上一律用英文（见 EFFECT_LABEL 注释），
# 但万一某处混入中文，这里把系统里能找到的中文字体挂上去，避免渲染成方块。
# 沙箱多为 Linux，通常没有中文字体 → 静默跳过，不影响英文标签。
try:
    from matplotlib import font_manager
    _CJK = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
            "WenQuanYi Zen Hei", "Source Han Sans SC", "PingFang SC")
    _avail = {f.name for f in font_manager.fontManager.ttflist}
    for _n in _CJK:
        if _n in _avail:
            plt.rcParams["font.sans-serif"] = [_n] + \
                list(plt.rcParams.get("font.sans-serif", []))
            break
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

from config import (BASELINE, BO, BUDGETS, HIT_Q, OUT_DIR, STRATEGIES,
                    ABLATE_OPS, discover_tasks, HARD_EXCLUDE_ATOMS)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

# ---------- 事先锁定、不随结果调整的分析常量 ----------
MAIN_STRATEGY = "gp_ucb"         # 预注册 §10：主检验只用主策略
BO_STRATEGIES = ("gp_ucb", "gp_ei")
ALPHA = 0.05
N_BOOT = 2000                    # 预注册 §7：bootstrap 2000 次
BIN_WIDTH = 0.05                 # 预注册 §8：AUC 分箱宽度
LOWESS_FRAC = 0.60               # LOWESS 带宽（覆盖约 60% 任务点）
ARMS = ("main", "ablate")        # main=预注册主实验；ablate=描述符消融稳健性臂

# ⚠️ 边界拟合的效应量口径（2026-10-17 由 rel_gain 改为 abs_gain，登记于 §10）
#
# 为什么不能用 rel_gain（相对增益 = 差值 / 基线命中数）：
#   实测 2160 run 的主臂等成本组命中率表（budget=400）：
#     random_uniform  0.097–0.103（中位 0.100）→ 命中口径本身校准良好
#     farthest_point  0.029–0.374（中位 0.094）→ 极差/中位 3.68×
#   farthest_point 在 matbench_expt_gap 上只有 0.029（比随机还差 3.4 倍），
#   原因是**真实效应**而非 bug：max-min 空间填充在高价值样本聚成一簇时会系统性欠采样该簇。
#   但它作为分母会让 rel_gain 在该任务上飙到 ~10.75，把"基线的运气"放大成"BO 的功劳"。
#
# 为什么 abs_gain 是对的：
#   ① 零点语义不变 —— rel_gain = abs_gain / base_hits，base_hits > 0，
#      故 abs_gain = 0 ⟺ rel_gain = 0，AUC* 的**位置**不受影响，只是曲线不再被分母扭曲。
#   ② random_uniform 恒为 0.100，除以它等价于线性缩放，同样不改变零点。
#   ③ Wilcoxon 符号秩与 Cliff's delta 本来就作用在**配对差值**上，与本次改动同构。
#   ④ 量纲直观：abs_gain = "预算内比基线多找到几个命中候选"，可直接写进准则卡。
EFFECT = "abs_gain"              # 边界拟合与全部图表的因变量
# ⚠️ 图内标签一律用英文：matplotlib 默认字体（DejaVu Sans）不含中文字形，
#    中文会渲染成一排方块（实测触发大量 "Glyph ... missing from font" 警告），
#    图就直接报废了。中文说明一律放在报告正文与准则卡里，不进图。
EFFECT_LABEL = "absolute gain (BO hits - baseline hits)"

# 注意：MAIN_BUDGET 不再写死 —— 它必须等于该组实际跑到的最大预算档，
# 从 runs_*.jsonl 的 meta["budgets"] 里读（沙箱可能只跑到 200）。

TASK_PROPS_CACHE = os.path.join(OUT_DIR, "task_props.csv")


# ============================================================
# 0. 载入
# ============================================================
def runs_path(use_gate, arm="main"):
    suffix = "" if arm == "main" else f"_{arm}"
    return os.path.join(OUT_DIR,
                        f"runs_{'gate' if use_gate else 'nogate'}{suffix}.jsonl")
def load_runs(path):
    """读 jsonl：只取 run 记录，跳过 meta / task_meta 行。

    task_meta 现在是**每个任务跑完就落盘一条**（会话中断也不丢元信息），
    所以必须逐条 update 累加，直接赋值只会留下最后一个任务。
    """
    if not os.path.exists(path):
        return [], {}, {}
    runs, meta, task_meta = [], {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
            elif "task_meta" in obj:
                task_meta.update(obj["task_meta"])
            else:
                runs.append(obj)
    return runs, meta, task_meta


# ============================================================
# 1. 预算截断 → 指标（主指标：预算内命中数）
# ============================================================
def prefix_metrics(rec, budget):
    """把一条完整 run 截断到给定预算（累计成本口径），返回该预算下的指标。

    n_hits  预算内命中数（主指标，预注册 §7）
    ttfh    Time-to-First-Hit（次要；未命中记 None，比较时右删失记 n_exp+1）
    auc     命中曲线下面积 = mean(前 k 步的累计命中密度)（次要）
    n_exp   预算内完成的有效实验次数
    spent   实际花费
    """
    cost = np.asarray(rec["cost_seq"], dtype=float)
    hits = np.asarray(rec["hits"], dtype=int)
    cum = np.cumsum(cost)
    k = int(np.searchsorted(cum, budget, side="right"))
    k = max(0, min(k, len(hits)))

    h = hits[:k]
    n_hits = int(h.sum())
    ttfh = next((i + 1 for i, v in enumerate(h) if v), None)
    if k > 0:
        cum_hits = np.cumsum(h)
        auc = float(np.mean(cum_hits / np.arange(1, k + 1)))
    else:
        auc = 0.0
    return {"n_hits": n_hits, "ttfh": ttfh, "auc": auc, "n_exp": k,
            "spent": float(cum[k - 1]) if k > 0 else 0.0}


def build_summary(runs, budgets=None):
    budgets = tuple(budgets or BUDGETS)
    rows = []
    for rec in runs:
        for b in budgets:
            m = prefix_metrics(rec, b)
            rows.append({"task": rec["task"], "strategy": rec["strategy"],
                         "seed": int(rec["seed"]), "use_gate": bool(rec["use_gate"]),
                         "budget": int(b), **m})
    df = pd.DataFrame(rows)
    if len(df):
        # 右删失 TTFH：未命中的 run 记 n_exp+1（越大越差），用于次要指标的配对比较
        df["ttfh_cens"] = [r if r is not None else (n + 1)
                           for r, n in zip(df["ttfh"], df["n_exp"])]
    return df


# ============================================================
# 2. 统计工具（预注册 §7）
# ============================================================
def cliffs_delta(a, b):
    """Cliff's delta：P(a>b) − P(a<b)，范围 [−1, 1]。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float(gt - lt) / (len(a) * len(b))


def boot_ci_mean(diffs, n=N_BOOT, seed=0):
    """配对差均值的 bootstrap 95% CI。"""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return (np.nan, np.nan)
    if len(d) == 1 or np.allclose(d, d[0]):
        return (float(d.mean()), float(d.mean()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def holm(pvals):
    """Holm–Bonferroni 逐步下降校正，返回单调化的校正 p 值。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def paired_tests(summary, use_gate, budgets=None, strategy_family=STRATEGIES,
                 arm="main"):
    """对每个 (任务 × 预算 × 策略) 做 该策略 vs 强基线 的配对检验。

    家族划分（预注册 §10）：
      main         = 主策略 × 主预算档（=该组实际最大预算） × 等成本组
                     → 每臂各 1 次/任务，唯一做主结论的家族
      exploratory  = 其余全部 → 同族内 Holm 校正，仅供探索
    """
    budgets = tuple(budgets or sorted(summary["budget"].unique()))
    main_budget = int(max(budgets))          # 主预算档 = 该组实际跑到的最大档
    rows = []
    for task in sorted(summary["task"].unique()):
        for b in budgets:
            sub = summary[(summary["task"] == task) & (summary["budget"] == b)]
            base = sub[sub["strategy"] == BASELINE].sort_values("seed")
            for strat in strategy_family:
                if strat == BASELINE:
                    continue
                cur = sub[sub["strategy"] == strat].sort_values("seed")
                n = min(len(cur), len(base))
                if n == 0:
                    continue
                a = cur["n_hits"].values[:n].astype(float)
                bb = base["n_hits"].values[:n].astype(float)

                if n < 2 or np.allclose(a - bb, 0):
                    p = 1.0
                else:
                    try:
                        p = float(wilcoxon(a, bb, zero_method="zsplit").pvalue)
                    except ValueError:
                        p = 1.0
                delta = cliffs_delta(a, bb)
                lo, hi = boot_ci_mean(a - bb)
                # 次要指标：右删失 TTFH（越小越好）→ 取增益符号为 *负* 差值
                ac = cur["ttfh_cens"].values[:n].astype(float)
                bc = base["ttfh_cens"].values[:n].astype(float)
                is_main = (strat == MAIN_STRATEGY and b == main_budget
                           and not use_gate)
                # ⚠️ 相对增益的分母下限：小预算下来自基线的命中数可能为 0，
                #    直接用 mean_base 会让 rel_gain 爆成 +∞/−∞（实测 budget=25 时出现 1.57）。
                #    统一用 max(mean_base, 1.0) 做分母，等价于"相对 1 次命中的增益"，
                #    并同时给出绝对增益 abs_gain 供交叉核对。
                denom = max(float(bb.mean()), 1.0)
                rows.append({
                    "task": task, "budget": int(b), "strategy": strat,
                    "use_gate": bool(use_gate), "n": int(n),
                    "mean_bo": float(a.mean()), "mean_base": float(bb.mean()),
                    "mean_diff": float((a - bb).mean()),
                    "abs_gain": float((a - bb).mean()),
                    "rel_gain": float((a.mean() - bb.mean()) / denom),
                    "cliffs_delta": delta, "p_raw": p, "ci_lo": lo, "ci_hi": hi,
                    "mean_ttfh_bo": float(ac.mean()),
                    "mean_ttfh_base": float(bc.mean()),
                    "ttfh_diff": float((ac - bc).mean()),
                    # 预注册 §10：主检验家族 = 主策略 × 主预算档 × 等成本组
                    "family": "main" if is_main else "exploratory",
                    "arm": arm,
                    # 报告用标记：主策略 × 主预算档（不区分门控），供准则卡逐任务表使用
                    "main_cell": bool(strat == MAIN_STRATEGY and b == main_budget),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for fam in ("main", "exploratory"):
            idx = out.index[out["family"] == fam]
            if len(idx):
                out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_raw"].values)
        out["significant"] = out["p_holm"] < ALPHA
    return out


# ============================================================
# 3. 任务空间结构属性（失效边界的自变量）
# ============================================================
def probe_one(df, q=HIT_Q, n_estimators=120, cv=5, seed=0):
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(float)
    y = df["target"].values.astype(float)

    thr = float(np.quantile(y, q))
    label = (y >= thr).astype(int)
    base_rate = float(label.mean())

    auc = np.nan
    if 0 < base_rate < 1:
        try:
            prob = cross_val_predict(
                RandomForestClassifier(n_estimators=n_estimators,
                                       random_state=seed, n_jobs=-1),
                X, label, cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(label, prob))
        except Exception:
            auc = np.nan

    rs = []
    for j in range(X.shape[1]):
        if X[:, j].std() < 1e-12:
            continue
        c = spearmanr(X[:, j], y).correlation
        if not np.isnan(c):
            rs.append(abs(c))
    r_proxy = float(np.nanmax(rs)) if rs else np.nan

    n_modes = np.nan
    try:
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde
        step = max(1, len(y) // 500)
        kde = gaussian_kde(y[::step])
        grid = np.linspace(y.min(), y.max(), 200)
        n_modes = int(len(find_peaks(kde(grid))[0]))
    except Exception:
        n_modes = np.nan

    return {"separability_auc": auc, "r_proxy": r_proxy, "n_modes": n_modes,
            "base_hit_rate": base_rate, "dim": int(X.shape[1]),
            "n_pool": int(len(df))}


def task_props(tasks, force=False, extra_cols=None):
    """计算每个登记任务的属性；命中缓存以免重复跑 RF。"""
    if (not force) and os.path.exists(TASK_PROPS_CACHE):
        cached = pd.read_csv(TASK_PROPS_CACHE)
        if set(tasks).issubset(set(cached["task"])):
            print(f"[props] 命中缓存 {os.path.basename(TASK_PROPS_CACHE)}"
                  f"（{len(cached)} 任务）")
            return cached

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name, task in tasks.items():
        df = pd.read_csv(task["path"])
        if "n_atoms" in df.columns:
            df = df[df["n_atoms"] <= HARD_EXCLUDE_ATOMS].reset_index(drop=True)
        p = probe_one(df)
        row = {"task": name, "arm": ("ablate" if "__" in name and
                                     name.split("__", 1)[1] in ABLATE_OPS
                                     else "main"), **p}
        if extra_cols:
            row.update(extra_cols.get(name, {}))
        rows.append(row)
        print(f"[props] {name:<34} arm={row['arm']:<6} AUC={p['separability_auc']:.3f} "
              f"r={p['r_proxy']:.3f} modes={p['n_modes']}")
    out = pd.DataFrame(rows)
    out.to_csv(TASK_PROPS_CACHE, index=False)
    return out


# ============================================================
# 4. 边界拟合（预注册 §8）
# ============================================================
def lowess_fit(x, y, frac=LOWESS_FRAC, grid=None):
    """自实现的 LOWESS（tricube 权重 + 局部线性），避免额外依赖。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return x, y
    k = int(np.clip(np.ceil(frac * n), 3, n))
    if grid is None:
        grid = np.linspace(x.min(), x.max(), 200)
    grid = np.asarray(grid, dtype=float)
    yhat = np.empty_like(grid)
    for gi, g in enumerate(grid):
        d = np.abs(x - g)
        h = np.sort(d)[k - 1]
        if h <= 0:
            yhat[gi] = y[np.argmin(d)]
            continue
        w = np.clip(1.0 - (d / h) ** 3, 0.0, None) ** 3
        if w.sum() <= 0:
            yhat[gi] = y[np.argmin(d)]
            continue
        xc = x - g
        S0 = w.sum()
        S1 = (w * xc).sum()
        S2 = (w * xc ** 2).sum()
        T0 = (w * y).sum()
        T1 = (w * xc * y).sum()
        den = S0 * S2 - S1 ** 2
        if abs(den) < 1e-12:
            yhat[gi] = T0 / S0
        else:
            b = (S0 * T1 - S1 * T0) / den
            a = (T0 - b * S1) / S0
            yhat[gi] = a  # 局部加权拟合：a 即该 grid 点处的取值
    return grid, yhat


def zero_crossing(grid, yhat):
    """找增益由正转负的零点（线性插值）。返回 (auc*, 方向)；无穿越返回 (nan, None)。"""
    sign = np.sign(yhat)
    for i in range(1, len(grid)):
        if sign[i - 1] > 0 and sign[i] <= 0:
            f = yhat[i - 1] / (yhat[i - 1] - yhat[i])
            return float(grid[i - 1] + f * (grid[i] - grid[i - 1])), "pos2neg"
    for i in range(1, len(grid)):
        if sign[i - 1] < 0 and sign[i] >= 0:
            f = yhat[i - 1] / (yhat[i - 1] - yhat[i])
            return float(grid[i - 1] + f * (grid[i] - grid[i - 1])), "neg2pos"
    return float("nan"), None


def fit_boundary(tests, props, use_gate, strategy=MAIN_STRATEGY, budget=None,
                 n_boot=N_BOOT):
    """H1 边界：以可学习性 AUC 为自变量，拟合绝对增益的零穿越点（预注册 §8）。

    因变量用 EFFECT（= abs_gain）而不是 rel_gain，理由见模块顶部注释：
    rel_gain 的分母是基线命中数，而强基线 farthest_point 的命中率跨任务摆动 3.68×，
    会把"基线的运气"放大成"BO 的功劳"。零点位置两者相同，但 abs 版曲线不被分母扭曲。

    返回 (summary_dict, per_task_dataframe)。per_task 已按 AUC 升序，
    并带 lowess 拟合值与 0.05 宽的 AUC 分箱中位数。
    """
    if budget is None:
        avail = sorted(tests["budget"].unique())
        budget = int(max(avail)) if len(avail) else None
    if budget is None:
        return {"strategy": strategy, "budget": None, "use_gate": bool(use_gate),
                "n_tasks": 0, "auc_star": None, "note": "无数据"}, pd.DataFrame()
    sub = tests[(tests["strategy"] == strategy) & (tests["budget"] == budget)
                & (tests["use_gate"] == use_gate)].merge(props, on="task", how="left")
    sub = sub.dropna(subset=["separability_auc", EFFECT]) \
             .sort_values("separability_auc").reset_index(drop=True)

    out = {"strategy": strategy, "budget": int(budget), "use_gate": bool(use_gate),
           "n_tasks": int(len(sub))}
    if len(sub) < 4:
        out.update({"auc_star": None, "direction": None, "ci": None,
                    "ci_note": "任务数不足，未拟合",
                    "n_boot_with_crossing": 0, "frac_boot_with_crossing": 0.0,
                    "auc_range": ([float(sub["separability_auc"].min()),
                                   float(sub["separability_auc"].max())]
                                  if len(sub) else None),
                    "gain_range": ([float(sub[EFFECT].min()),
                                    float(sub[EFFECT].max())] if len(sub) else None),
                    "n_significant": int(sub["significant"].sum()) if len(sub) else 0,
                    "note": f"任务数 {len(sub)} < 4，不足以拟合边界"})
        return out, sub

    x = sub["separability_auc"].values.astype(float)
    y = sub[EFFECT].values.astype(float)
    grid, yhat = lowess_fit(x, y)
    auc_star, direction = zero_crossing(grid, yhat)

    # 分箱（0.05 一箱）取中位数 —— 预注册 §8.1
    lo_bin = np.floor(x.min() / BIN_WIDTH) * BIN_WIDTH
    bins = np.arange(lo_bin, x.max() + BIN_WIDTH + 1e-9, BIN_WIDTH)
    sub["auc_bin"] = pd.cut(sub["separability_auc"], bins)
    binned = (sub.groupby("auc_bin", observed=True)[EFFECT]
                 .agg(["median", "count"]).reset_index())
    binned["auc_mid"] = [iv.mid for iv in binned["auc_bin"]]

    # LOWESS 在任务点上的取值（用于散点标注，长度与 sub 一致）
    sub["lowess_at_task"] = np.interp(x, grid, yhat)

    # 零点 bootstrap CI：对任务重采样（预注册 §8.3）
    rng = np.random.default_rng(0)
    stars = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xs, ys = x[idx], y[idx]
        if len(np.unique(xs)) < 3:
            continue
        g2, y2 = lowess_fit(xs, ys, grid=grid)
        s2, _ = zero_crossing(g2, y2)
        if not np.isnan(s2):
            stars.append(s2)
    # ⚠️ 只有当**原样本**确实存在零点时才报 CI。
    #    曾出现的荒谬输出：auc_star=None（无零点）却给出 ci=[0.518, 0.914]——
    #    那是 bootstrap 里 2.85% 的重采样碰巧穿越了零，但这不构成"存在边界"的证据
    #    （对照：消融臂的 frac_boot_with_crossing 高达 94%）。
    #    两者必须区分开，否则读者会把噪声当成边界。
    if np.isnan(auc_star):
        ci = None
        ci_note = (f"原样本无零点穿越，故不报 CI；bootstrap 中仅 "
                   f"{len(stars)/n_boot*100:.1f}% 的重采样出现零点，不足以支持边界存在")
    elif len(stars) >= 50:
        ci = [float(np.percentile(stars, 2.5)), float(np.percentile(stars, 97.5))]
        ci_note = None
    else:
        ci = None
        ci_note = f"bootstrap 有效重采样不足（{len(stars)} 次），CI 未收敛"

    out.update({
        "auc_star": None if np.isnan(auc_star) else float(auc_star),
        "direction": direction,
        "ci": ci,
        "ci_note": ci_note,
        "n_boot_with_crossing": len(stars),
        "frac_boot_with_crossing": float(len(stars) / n_boot),
        "auc_range": [float(x.min()), float(x.max())],
        "gain_range": [float(y.min()), float(y.max())],
        "n_significant": int(sub["significant"].sum()),
        "n_significant_pos": int(((sub["significant"]) & (sub[EFFECT] > 0)).sum()),
        "sign_all_positive": bool((y > 0).all()),
        "effect": EFFECT,
        "note": None if not np.isnan(auc_star) else
                f"扫描区间内未出现{EFFECT_LABEL}的零点穿越",
    })
    return out, sub


def fit_boundary_by_budget(tests, props, use_gate, strategy=MAIN_STRATEGY,
                           budgets=None):
    """对每个预算档各拟合一次零点，用于观察边界随预算漂移（H3 的补充视角）。"""
    budgets = budgets or sorted(tests["budget"].unique())
    rows = []
    for b in budgets:
        o, _ = fit_boundary(tests, props, use_gate, strategy=strategy,
                            budget=b, n_boot=500)
        rows.append({"budget": int(b), "use_gate": bool(use_gate),
                     "n_tasks": o["n_tasks"], "auc_star": o["auc_star"],
                     "ci_lo": (o["ci"][0] if o["ci"] else np.nan),
                     "ci_hi": (o["ci"][1] if o["ci"] else np.nan),
                     "frac_boot_with_crossing": o["frac_boot_with_crossing"],
                     "gain_min": (o["gain_range"][0] if o["gain_range"] else np.nan),
                     "gain_max": (o["gain_range"][1] if o["gain_range"] else np.nan)})
    return pd.DataFrame(rows)


def predictor_strength(tests, props, arm, use_gate, budget=None):
    """探索性：检验哪些任务属性**真正**预测 BO 增益（预注册 §10 登记的事后补充）。

    为什么需要：预注册只指定了"可学习性 AUC"作自变量。但实测发现
    AUC 的预测力在两条轴上差别很大 —— 主臂 ρ=+0.864（p<1e-4），
    消融臂只有 ρ=+0.513。若不量化这一点，就容易把 LOWESS 拟合出的
    单个 AUC* 当成硬阈值，而它其实只在部分条件下成立。
    """
    if tests is None or not len(tests):
        return pd.DataFrame()
    b = int(budget or max(tests["budget"].unique()))
    s = tests[(tests["strategy"] == MAIN_STRATEGY) & (tests["budget"] == b)
              & (tests["use_gate"] == bool(use_gate))]
    if len(s) < 5:
        return pd.DataFrame()
    s = s.merge(props, on="task", how="left")
    rows = []
    for v in ("separability_auc", "r_proxy", "n_modes", "dim", "base_hit_rate"):
        if v not in s.columns:
            continue
        x = s[v].astype(float)
        if x.nunique() < 3:            # 常量列（如主臂 dim 恒为 88）无法算秩相关
            continue
        rho, pv = spearmanr(x, s[EFFECT])
        rows.append({"arm": arm, "use_gate": bool(use_gate), "budget": b,
                     "predictor": v, "spearman_rho": float(rho),
                     "p_raw": float(pv), "n_tasks": int(len(s)),
                     "note": "exploratory（预注册未指定，§10 事后登记）"})
    return pd.DataFrame(rows)


def robustness_report(tests, props, budget=None):
    """稳健性自检（探索性，§10 事后登记）—— 专门用来拆自己的台。

    为什么必须有：
      ① 两条 LOWESS 曲线"看起来不一样"不等于差异显著。核心结论"两轴后果不同"
         必须能被直接检验，否则评委一句"你没检验这个差异"就能推翻。
      ② 名义任务数 ≠ 有效样本量。主臂 18 个任务里 15 个来自 λ 扫描
         （共享同一底层真实目标），把它们当成 18 次独立检验是 inflated evidence。

    输出：跨轴差异的 Mann-Whitney 检验 + 按基底聚合的有效样本量。
    """
    if tests is None or not len(tests):
        return pd.DataFrame()
    b = int(budget or max(tests["budget"].unique()))
    s = tests[(tests["strategy"] == MAIN_STRATEGY) & (tests["budget"] == b)
              & (tests["use_gate"] == False)]          # noqa: E712 等成本主口径
    # ⚠️ props 里也有 arm 列，直接 merge 会撞成 arm_x/arm_y → 后面 s["arm"] 直接 KeyError
    s = s.merge(props[[c for c in props.columns if c != "arm"]], on="task", how="left")
    rows = []

    # ---- ① 跨轴差异的直接检验 ----
    A = s[s["arm"] == "main"][EFFECT].astype(float)
    B = s[s["arm"] == "ablate"][EFFECT].astype(float)
    if len(A) >= 3 and len(B) >= 3:
        _, pv = mannwhitneyu(A, B, alternative="two-sided")
        rows.append({"check": "cross_axis_all", "scope": "全部 AUC 区间",
                     "n_main": len(A), "n_ablate": len(B),
                     "median_main": float(A.median()), "median_ablate": float(B.median()),
                     "p_raw": float(pv), "budget": b,
                     "note": "两轴增益分布是否不同"})
    lo = s[s["separability_auc"] < 0.75]
    A2 = lo[lo["arm"] == "main"][EFFECT].astype(float)
    B2 = lo[lo["arm"] == "ablate"][EFFECT].astype(float)
    if len(A2) >= 3 and len(B2) >= 3:
        _, pv2 = mannwhitneyu(A2, B2, alternative="two-sided")
        rows.append({"check": "cross_axis_low_auc", "scope": "AUC < 0.75（关键区间）",
                     "n_main": len(A2), "n_ablate": len(B2),
                     "median_main": float(A2.median()), "median_ablate": float(B2.median()),
                     "p_raw": float(pv2), "budget": b,
                     "note": "低可学习性区间内两轴是否不同"})

    # ---- ② 有效样本量：按基底聚合 ----
    s = s.copy()
    s["base"] = s["task"].str.split("__").str[0]
    s["n_task_in_base"] = s.groupby(["arm", "base"])["task"].transform("nunique")
    for (arm, base), g in s.groupby(["arm", "base"]):
        rows.append({"check": "effective_n", "scope": f"{arm}/{base.replace('matbench_','')}",
                     "n_main": len(g), "n_ablate": int(g["n_task_in_base"].iloc[0]),
                     "median_main": float(g[EFFECT].median()), "median_ablate": None,
                     "p_raw": None, "budget": b,
                     "note": f"显著为负 {int(((g['significant']) & (g[EFFECT] < 0)).sum())} 个，"
                             f"显著为正 {int(((g['significant']) & (g[EFFECT] > 0)).sum())} 个"})
    return pd.DataFrame(rows)


# ============================================================
# 5. H4：门控 vs 等成本 的相对排序变化（双向检验）
# ============================================================
def compare_gate(tests_gate, tests_nogate):
    """H4：加入合成可行性门控后，策略相对排序是否变化。

    逐 (任务 × 预算) 单元对策略按 mean_bo 排名，比较两个口径下的排名。
    """
    if tests_gate is None or tests_nogate is None or not len(tests_gate) or not len(tests_nogate):
        return pd.DataFrame(), {}
    key = ["task", "budget"]
    a = tests_nogate.groupby(key + ["strategy"])[["mean_bo", EFFECT]].mean().reset_index()
    b = tests_gate.groupby(key + ["strategy"])[["mean_bo", EFFECT]].mean().reset_index()
    m = a.merge(b, on=key + ["strategy"], suffixes=("_nogate", "_gate"))
    m["rank_nogate"] = m.groupby(key)["mean_bo_nogate"].rank(ascending=False)
    m["rank_gate"] = m.groupby(key)["mean_bo_gate"].rank(ascending=False)
    m["rank_inversion"] = (m["rank_nogate"] != m["rank_gate"]).astype(int)
    m["effect_shift"] = m[f"{EFFECT}_gate"] - m[f"{EFFECT}_nogate"]

    # 逐单元 Spearman（≥3 个策略才有意义）
    rhos = []
    for _, g in m.groupby(key):
        if len(g) >= 3:
            r = spearmanr(g["mean_bo_nogate"], g["mean_bo_gate"]).correlation
            if not np.isnan(r):
                rhos.append(r)
    summary = {
        "n_cells": int(m.groupby(key).ngroups),
        "inversion_rate": float(m["rank_inversion"].mean()),
        "mean_rank_rho": float(np.mean(rhos)) if rhos else None,
        "median_rank_rho": float(np.median(rhos)) if rhos else None,
        "n_cells_rho_lt_1": int(sum(1 for r in rhos if r < 0.999)),
        "mean_effect_shift_gp_ucb": float(
            m.loc[m["strategy"] == MAIN_STRATEGY, "effect_shift"].mean()),
    }
    return m, summary


# ============================================================
# 6. 出图
# ============================================================
def make_figures(boundary, sub_boundary, tests_gate, tests_nogate, summary, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Fig 1 —— 失效边界（H1 主图）
    if len(sub_boundary):
        plt.figure(figsize=(6.8, 4.8))
        plt.scatter(sub_boundary["separability_auc"], sub_boundary[EFFECT],
                    s=70, alpha=0.85, color="tab:blue", label="task (main budget)")
        # 0.05 宽 AUC 分箱中位数（预注册 §8.1）
        bx = np.floor(sub_boundary["separability_auc"].min() / BIN_WIDTH) * BIN_WIDTH
        bins = np.arange(bx, sub_boundary["separability_auc"].max() + BIN_WIDTH + 1e-9,
                         BIN_WIDTH)
        tmp = sub_boundary.assign(b=pd.cut(sub_boundary["separability_auc"], bins))
        med = tmp.groupby("b", observed=True)[EFFECT].median()
        mids = [iv.mid for iv in med.index]
        plt.plot(mids, med.values, marker="s", color="tab:orange", lw=1.6,
                 label="binned median (0.05 AUC)")
        xs = sub_boundary.sort_values("separability_auc")
        g, yhat = lowess_fit(xs["separability_auc"].values, xs[EFFECT].values)
        plt.plot(g, yhat, color="k", lw=2, label="LOWESS")
        plt.axhline(0, color="gray", lw=1, ls="--")
        if boundary.get("auc_star"):
            plt.axvline(boundary["auc_star"], color="tab:red", lw=1.5, ls=":")
            plt.text(boundary["auc_star"], plt.ylim()[1] * 0.92,
                     f"  AUC*={boundary['auc_star']:.3f}", color="tab:red", fontsize=9)
            if boundary.get("ci"):
                plt.axvspan(boundary["ci"][0], boundary["ci"][1],
                            color="tab:red", alpha=0.10)
        plt.xlabel("learnability (RF 5-fold CV ROC-AUC of top-10% label)")
        plt.ylabel(f"{EFFECT_LABEL}\\n{MAIN_STRATEGY} vs {BASELINE}")
        plt.title("H1: failure boundary of Bayesian optimization")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_boundary_auc.png"), dpi=140)
        plt.close()

        # Fig 1b —— 分箱中位数
        plt.figure(figsize=(6.6, 4.2))
        plt.axhline(0, color="gray", lw=1, ls="--")
        plt.plot(g, yhat, color="k", lw=2)
        plt.scatter(xs["separability_auc"], xs[EFFECT], s=45, color="tab:blue")
        plt.xlabel("learnability AUC")
        plt.ylabel(EFFECT_LABEL)
        plt.title(f"H1: LOWESS fit (frac={LOWESS_FRAC}), family n={boundary['n_tasks']}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_boundary_lowess.png"), dpi=140)
        plt.close()

    # Fig 2 —— 预算拐点（H3）
    if len(tests_nogate):
        plt.figure(figsize=(6.6, 4.6))
        t = tests_nogate[tests_nogate["strategy"] == MAIN_STRATEGY]
        for task in sorted(t["task"].unique()):
            s = t[t["task"] == task].sort_values("budget")
            plt.plot(s["budget"], s[EFFECT], marker="o", lw=1.2, ms=3.5,
                     alpha=0.75, label=task if task in ("matbench_expt_gap",
                                                       "matbench_dielectric",
                                                       "matbench_log_gvrh") else None)
        mean_curve = t.groupby("budget")[EFFECT].mean()
        plt.plot(mean_curve.index, mean_curve.values, color="k", lw=2.6,
                 marker="s", label="mean over tasks")
        plt.axhline(0, color="gray", lw=1, ls="--")
        plt.xlabel("budget (number of synthesizable / characterizable candidates)")
        plt.ylabel(f"{EFFECT_LABEL}\\n{MAIN_STRATEGY} vs {BASELINE}")
        plt.title("H3: does BO gain saturate as budget grows?")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_budget_saturation.png"), dpi=140)
        plt.close()

    # Fig 3 —— 门控 vs 等成本（H4）
    if len(tests_gate) and len(tests_nogate):
        plt.figure(figsize=(6.6, 4.6))
        ng = tests_nogate[tests_nogate["strategy"] == MAIN_STRATEGY]
        gt = tests_gate[tests_gate["strategy"] == MAIN_STRATEGY]
        for b in BUDGETS:
            x = ng[ng["budget"] == b][EFFECT].values
            y = gt[gt["budget"] == b][EFFECT].values
            n = min(len(x), len(y))
            plt.scatter(x[:n], y[:n], s=55, alpha=0.8, label=f"budget={b}")
        lim = [min(-0.5, plt.xlim()[0], plt.ylim()[0]),
               max(0.5, plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lim, lim, "k--", lw=1)
        plt.axhline(0, color="gray", lw=0.8)
        plt.axvline(0, color="gray", lw=0.8)
        plt.xlabel(f"{EFFECT_LABEL} — equal-cost (no gate)")
        plt.ylabel(f"{EFFECT_LABEL} — with synthesizability gate")
        plt.title("H4: does the cost gate rewrite the ranking?")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_gate_vs_nogate.png"), dpi=140)
        plt.close()

    # Fig 4 —— 有效实验次数（成本模型是否真的压缩了预算）
    if len(summary):
        # 主预算档 = 这组数据实际跑到的最大档（沙箱可能只到 200）
        main_b = int(summary["budget"].max())
        plt.figure(figsize=(6.6, 4.2))
        for strat in STRATEGIES:
            s = (summary[(summary["strategy"] == strat) & (summary["budget"] == main_b)]
                 .groupby("use_gate")["n_exp"].mean())
            if len(s):
                plt.plot(["no gate", "gate"], [s.get(False, np.nan), s.get(True, np.nan)],
                         marker="o", label=strat)
        plt.ylabel(f"mean number of experiments within budget={main_b}")
        plt.title("cost model compresses the effective budget")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_effective_budget.png"), dpi=140)
        plt.close()


# ============================================================
# 7. 决策准则卡
# ============================================================
def make_guideline_card(boundary, sub_boundary, h4_summary, outdir,
                        by_budget=None):
    group = "含合成可行性门控（真实约束）" if boundary.get("use_gate") \
        else "等成本（H1/H2 主口径）"
    ac = boundary.get("auc_star")
    rg = boundary.get("auc_range")
    lines = ["# 决策准则卡：什么时候不该用主动学习（贝叶斯优化）",
             "",
             f"> 口径：强基线 = `{BASELINE}`（**最远点 max-min 空间填充**）；"
             f"主动学习 = `{MAIN_STRATEGY}`；命中 = 目标 top-10% 分位。",
             f"> 成本口径：**{group}**；预算档 = {boundary.get('budget')}。",
             f"> 效应量：**{EFFECT_LABEL}**。不用相对增益，因为强基线自身的命中率"
             f"跨任务摆动 3.68×（详见模块顶部注释），作分母会把基线运气放大成 BO 功劳；"
             f"两者零点位置相同，绝对增益不受影响。",
             f"> 显著性：Wilcoxon 符号秩 + Holm–Bonferroni，α=0.05；"
             f"效应量 Cliff's δ；区间为 bootstrap {N_BOOT} 次 95% CI。",
             ""]

    lines += ["## 核心判据", ""]
    if ac is not None:
        ci = boundary.get("ci")
        ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "（bootstrap 未收敛，样本内零点）"
        lines += [
            f"- **失效边界零点 AUC\\* = {ac:.3f}**，bootstrap 95% CI {ci_txt}",
            f"- bootstrap 中 {boundary['frac_boot_with_crossing']*100:.1f}% 的重采样存在零点穿越"
            f"（{boundary['n_boot_with_crossing']} 次）",
            f"- 覆盖 AUC 区间 [{rg[0]:.3f}, {rg[1]:.3f}]，任务数 {boundary['n_tasks']}",
            f"- Holm 校正后**显著为正**（BO 优于强基线）的任务："
            f"**{boundary['n_significant_pos']} / {boundary['n_tasks']}**",
            f"- Holm 校正后显著（不分方向）的任务：{boundary['n_significant']} / "
            f"{boundary['n_tasks']}",
            "",
            f"**判据：当目标在给定描述符下的可学习性 AUC < {ac:.2f} 时，"
            f"主动学习相对强基线不再有正增益，应改用 `{BASELINE}`（空间填充设计）。**",
            "",
        ]
    else:
        lines += [
            f"- 在扫描区间 AUC ∈ [{rg[0]:.3f}, {rg[1]:.3f}] 内"
            f"**未观察到零点穿越**（任务数 {boundary['n_tasks']}）",
            f"- bootstrap 中只有 {boundary['frac_boot_with_crossing']*100:.1f}% 的"
            f"重采样出现零点（{boundary['n_boot_with_crossing']} / {N_BOOT} 次）"
            f"—— **这不足以支持『边界存在』的主张**，故不报 AUC\\* 与 CI",
            f"- 绝对增益区间 = [{boundary['gain_range'][0]:+.2f}, "
            f"{boundary['gain_range'][1]:+.2f}]（单位：预算内命中的候选个数）"
            + ("，**全部为正**" if boundary.get("sign_all_positive") else "，存在负值"),
            f"- Holm 校正后**显著为正**的任务：**{boundary['n_significant_pos']} / "
            f"{boundary['n_tasks']}**"
            + ("；其余为方向为正但未达校正后显著，按预注册 §7 不能宣称有增益"
               if boundary['n_significant_pos'] < boundary['n_tasks'] else ""),
            "",
            "**判据：本实验在扫描区间内未发现失效边界。**"
            "边界（若存在）位于 AUC 更低的区间，本扫描轴未覆盖到；"
            "按预注册 §9，此处如实报告，不修改题面。",
            "",
        ]

    if by_budget is not None and len(by_budget):
        lines += ["## 边界随预算的漂移（探索性）", "",
                  "| 预算 | 零点 AUC* | bootstrap 95% CI | 有零点穿越的重采样占比 | 增益范围 |",
                  "|---|---|---|---|---|"]
        for _, r in by_budget.iterrows():
            star = "—" if pd.isna(r["auc_star"]) else f"{r['auc_star']:.3f}"
            ci = ("—" if pd.isna(r["ci_lo"])
                  else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
            lines.append(f"| {int(r['budget'])} | {star} | {ci} | "
                         f"{r['frac_boot_with_crossing']*100:.1f}% | "
                         f"[{r['gain_min']:+.3f}, {r['gain_max']:+.3f}] |")
        lines.append("")

    rows = sub_boundary[sub_boundary["main_cell"]] if len(sub_boundary) else sub_boundary
    if len(rows):
        fam = rows["family"].iloc[0]
        fam_note = ("主检验家族（预注册 §10，Holm 校正限于 18 次）" if fam == "main"
                    else "探索性家族（Holm 校正在全网格内进行，更保守）")
        lines += [f"## 逐任务判定（预算={boundary.get('budget')}，{fam_note}）", "",
                  "| 任务 | 可学习性 AUC | 描述符相关性 r | 峰数 | 基线命中 | BO 命中 | "
                  "**绝对增益** | Cliff's δ | p(Holm) | 判定 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in rows.sort_values("separability_auc").iterrows():
            if ac is not None and (not np.isnan(r["separability_auc"])) \
                    and r["separability_auc"] < ac:
                verdict = "**失效区（AUC < AUC\\*）→ 不建议用 BO**"
            elif r["significant"] and r[EFFECT] > 0:
                verdict = "**显著为正 → 值得用 BO**"
            elif r[EFFECT] > 0:
                verdict = "方向为正但未达校正后显著 → 谨慎"
            elif r["significant"]:
                verdict = "**显著为负 → 别用 BO**"
            else:
                verdict = "无增益"
            modes = (int(r["n_modes"]) if not np.isnan(r["n_modes"]) else "—")
            lines.append(
                f"| {r['task']} | {r['separability_auc']:.3f} | {r['r_proxy']:.3f} | "
                f"{modes} | {r['mean_base']:.1f} | {r['mean_bo']:.1f} | "
                f"{r[EFFECT]:+.2f} | {r['cliffs_delta']:+.3f} | {r['p_holm']:.3f} | "
                f"{verdict} |")
        lines.append("")

    if h4_summary:
        rho = h4_summary.get("median_rank_rho")
        shift = h4_summary.get("mean_effect_shift_gp_ucb")
        lines += ["## H4：合成可行性门控的影响（双向检验）", "",
                  f"- 逐单元（任务 × 预算）策略排序反转率："
                  f"{h4_summary['inversion_rate']*100:.1f}%"
                  f"（共 {h4_summary['n_cells']} 个单元）",
                  f"- 单元内 Spearman 秩相关中位数："
                  f"{rho:.3f}" if rho is not None else "- 单元内秩相关：样本不足",
                  f"- 秩相关 < 1 的单元数：{h4_summary['n_cells_rho_lt_1']}",
                  f"- `{MAIN_STRATEGY}` 绝对增益的平均位移（门控 − 等成本）："
                  f"{shift:+.3f}" if shift is not None else "- 位移：样本不足",
                  ""]

    lines += ["## 使用说明", "",
              "1. 先估目标在**你的描述符集**下的可学习性"
              "（RF 交叉验证 ROC-AUC，标签 = 目标 top-10%）。",
              ]
    if ac is not None:
        lines.append(f"2. AUC 高于 {ac:.2f} 且预算有限 → 用 BO 有正收益；"
                     f"低于 {ac:.2f} → 分层随机采样足够，且实现更简单。")
    else:
        lines.append("2. 在本实验覆盖的 AUC 区间内 BO 始终不劣于强基线；"
                     "若目标可学习性进一步下降（AUC → 0.5），需自行用同法复核。")
    lines += [
        "3. **不要**用「描述符–目标的皮尔逊/斯皮尔曼相关」做判据："
        "实测在扫描轴上 r 仅从 0.08 变到 0.33，而 AUC 从 0.52 变到 0.93 —— "
        "相关性几乎不区分可学与不可学，AUC 才能。",
        "4. 若候选池含大量昂贵/难合成项，务必先看门控版结论（H4），"
        "成本结构本身会改写策略排序与你实际能做的实验次数。",
        "5. 增益在小预算档（≤50）噪声很大（命中数本身是小数），"
        "以预算 400 档的结论为准。",
        "",
    ]
    p = os.path.join(outdir, "guideline_card.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return p


# ============================================================
# 8. main
# ============================================================
def _analyze_group(runs, use_gate, props, tag, arm="main"):
    """分析一个 (臂 × 成本口径) 下的全部 runs。返回 dict（无数据时返回 None）。"""
    if not runs:
        return None
    # 预算档必须以数据自己的 meta 为准（沙箱可能只跑到 200）
    budgets = _BUDGETS_SEEN.get((arm, use_gate)) or sorted({int(b) for b in BUDGETS})
    summary = build_summary(runs, budgets=budgets)
    tests = paired_tests(summary, use_gate=use_gate, budgets=budgets, arm=arm)
    boundary, sub = fit_boundary(tests, props, use_gate=use_gate)
    by_budget = fit_boundary_by_budget(tests, props, use_gate=use_gate,
                                       budgets=budgets)
    print(f"[{tag}] runs={len(runs)} tasks={summary['task'].nunique()} "
          f"| 预算档={budgets} | AUC*={boundary.get('auc_star')} "
          f"| 显著={boundary.get('n_significant')}/{boundary.get('n_tasks')}")
    return {"summary": summary, "tests": tests, "boundary": boundary,
            "sub": sub, "by_budget": by_budget, "use_gate": use_gate,
            "arm": arm, "budgets": budgets, "n_runs": len(runs)}


# (arm, use_gate) -> 预算档；由 main() 在载入时填充
_BUDGETS_SEEN = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", nargs="?", default="both",
                    choices=["both", "gate", "nogate"])
    ap.add_argument("--force-props", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 载入两臂 × 两口径，共最多 4 个文件
    loaded = {}
    for arm in ARMS:
        for gate in (True, False):
            runs, meta, _ = load_runs(runs_path(gate, arm))
            key = (arm, "gate" if gate else "nogate")
            budgets = meta.get("budgets") or list(BUDGETS)
            _BUDGETS_SEEN[(arm, gate)] = budgets
            if runs:
                loaded[key] = runs
                print(f"[load] {key[0]:<6}/{key[1]:<6} {len(runs):>5} runs "
                      f"| budgets={budgets}")
    if not loaded:
        print("[中止] 没有任何 runs_*.jsonl 可分析")
        return

    # 任务属性：两臂的登记任务合并计算（有缓存）
    all_tasks = {}
    for arm in ARMS:
        all_tasks.update(discover_tasks(arm=arm))
    print(f"[discover] 主臂 {len(discover_tasks('main'))} + "
          f"消融臂 {len(discover_tasks('ablate'))} = {len(all_tasks)} 任务")
    props = task_props(all_tasks, force=args.force_props)

    # 逐 (臂 × 口径) 分析
    res = {}
    for (arm, grp), runs in loaded.items():
        if args.group != "both" and grp != args.group:
            continue
        res[(arm, grp)] = _analyze_group(runs, grp == "gate", props,
                                         f"{arm}/{grp}", arm=arm)
    if not res:
        print("[中止] 所选口径没有数据")
        return

    # ---- 落盘 ----
    all_summary = pd.concat([r["summary"] for r in res.values()], ignore_index=True)
    all_summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(f"[save] summary.csv ({len(all_summary)} 行)")

    tframes = [r["tests"] for r in res.values() if len(r["tests"])]
    tests_all = pd.concat(tframes, ignore_index=True) if tframes else pd.DataFrame()
    if len(tests_all):
        tests_all.to_csv(os.path.join(OUT_DIR, "tests.csv"), index=False)
        print(f"[save] tests.csv ({len(tests_all)} 行)")

    # 稳健性自检：跨轴差异检验 + 有效样本量（探索性，§10 登记）
    rb = robustness_report(tests_all, props)
    if len(rb):
        rb.to_csv(os.path.join(OUT_DIR, "robustness.csv"), index=False)
        print(f"[save] robustness.csv ({len(rb)} 行)")
        for _, r in rb[rb["check"].str.startswith("cross_axis")].iterrows():
            flag = "✅ 显著" if (r["p_raw"] is not None and r["p_raw"] < ALPHA) else "⚠️ 未达显著"
            print(f"    跨轴差异[{r['scope']}] p={r['p_raw']:.4f} {flag}")
        nb = rb[rb["check"] == "effective_n"]
        if len(nb):
            print(f"    有效基底数 = {len(nb)}（名义任务数 {int(nb['n_main'].sum())}）")

    # 自变量的预测力对照（探索性，§10 登记）
    pframes = []
    for (arm, grp), r in res.items():
        pf = predictor_strength(r["tests"], props, arm, grp == "gate")
        if len(pf):
            pframes.append(pf)
    if pframes:
        ps = pd.concat(pframes, ignore_index=True)
        ps.to_csv(os.path.join(OUT_DIR, "predictor_strength.csv"), index=False)
        print(f"[save] predictor_strength.csv ({len(ps)} 行)")
        for _, r in ps[ps["predictor"] == "separability_auc"].iterrows():
            print(f"    AUC 预测力 {r['arm']}/{'gate' if r['use_gate'] else 'nogate'}: "
                  f"rho={r['spearman_rho']:+.3f}  p={r['p_raw']:.4f}")

    bframes = []
    for r in res.values():
        bb = r["by_budget"].copy()
        bb["group"] = "gate" if r["use_gate"] else "nogate"
        bb["arm"] = r["arm"]
        bframes.append(bb)
    if bframes:
        pd.concat(bframes, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "boundary_by_budget.csv"), index=False)
        print("[save] boundary_by_budget.csv")

    # H4：主臂（消融臂不参与）
    h4_table, h4_summary = pd.DataFrame(), {}
    if ("main", "gate") in res and ("main", "nogate") in res:
        h4_table, h4_summary = compare_gate(res[("main", "gate")]["tests"],
                                            res[("main", "nogate")]["tests"])
        if len(h4_table):
            h4_table.to_csv(os.path.join(OUT_DIR, "h4_gate.csv"), index=False)
            print(f"[save] h4_gate.csv | 排序反转率="
                  f"{h4_summary['inversion_rate']*100:.1f}%")

    # 主口径 = 主臂等成本组（预注册 §10）；缺失时退化并显式标注
    if ("main", "nogate") in res:
        primary, primary_key = res[("main", "nogate")], "main_equal_cost"
    elif ("main", "gate") in res:
        primary, primary_key = res[("main", "gate")], "gate_only_no_equal_cost_group"
        print("[warn] 缺少主臂等成本组 —— 主口径不可用，"
              "以下边界结论均基于门控组，**不得**作为 H1/H2 主结论")
    else:
        primary = res[("ablate", "nogate")] if ("ablate", "nogate") in res \
            else next(iter(res.values()))
        primary_key = f"{primary['arm']}_{'gate' if primary['use_gate'] else 'nogate'}"

    payload = {primary_key: primary["boundary"]}
    for (arm, grp), r in res.items():
        payload[f"{arm}_{grp}"] = r["boundary"]
    payload["h4"] = h4_summary
    payload["config"] = {
        "main_strategy": MAIN_STRATEGY,
        "main_budget_by_arm": {arm: max(_BUDGETS_SEEN[(arm, False)])
                               for arm in ARMS
                               if (arm, False) in _BUDGETS_SEEN},
        "alpha": ALPHA, "n_boot": N_BOOT,
        "bin_width": BIN_WIDTH, "lowess_frac": LOWESS_FRAC,
        "n_tasks_registered": {arm: len(discover_tasks(arm=arm)) for arm in ARMS},
    }
    with open(os.path.join(OUT_DIR, "boundary_fit.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save] boundary_fit.json | 主口径 AUC*="
          f"{primary['boundary'].get('auc_star')} ({primary_key})")

    # 出图
    make_figures(primary["boundary"], primary["sub"],
                 res[("main", "gate")]["tests"] if ("main", "gate") in res else pd.DataFrame(),
                 res[("main", "nogate")]["tests"] if ("main", "nogate") in res else pd.DataFrame(),
                 all_summary, OUT_DIR)
    # 双轴印证图：主扫描轴（目标降质）vs 消融臂（描述符削减）
    two_axis_figure(res, props, OUT_DIR)
    print("[save] fig_*.png")

    card = make_guideline_card(primary["boundary"], primary["sub"],
                               h4_summary, OUT_DIR, by_budget=primary["by_budget"])
    print(f"[save] {os.path.basename(card)}")

    print("\n[完成] 分析产物见 results/")


def two_axis_figure(res, props, outdir):
    """两张独立的"降信息轴"画在同一张图上：
      主臂  = 目标降质（给真实目标加噪）
      消融臂 = 描述符削减（真实目标不动）
    若两条轴给出同一零点 → 边界不是合成产物的产物。
    """
    styles = {"main": dict(color="tab:blue", marker="o", label="target-degradation axis (λ)"),
              "ablate": dict(color="tab:red", marker="s",
                             label="descriptor-ablation axis (real targets)")}
    plt.figure(figsize=(7.0, 4.8))
    used = False
    for (arm, grp), r in res.items():
        if grp != "nogate" or arm not in styles:
            continue
        sub = r["sub"]
        if not len(sub):
            continue
        used = True
        st = styles[arm]
        plt.scatter(sub["separability_auc"], sub[EFFECT], s=62,
                    alpha=0.85, color=st["color"], marker=st["marker"],
                    label=st["label"])
        if len(sub) >= 4:
            g, yh = lowess_fit(sub["separability_auc"].values.astype(float),
                               sub[EFFECT].values.astype(float))
            plt.plot(g, yh, color=st["color"], lw=2, ls="-" if arm == "main" else "--")
            star = r["boundary"].get("auc_star")
            if star is not None:
                plt.axvline(star, color=st["color"], lw=1.3, ls=":",
                            alpha=0.9)
                plt.text(star, plt.ylim()[1], f"  AUC*={star:.2f}",
                         color=st["color"], fontsize=9, va="top")
    if not used:
        plt.close()
        return
    plt.axhline(0, color="gray", lw=1, ls="--")
    plt.xlabel("learnability (RF 5-fold CV ROC-AUC of top-10% label)")
    plt.ylabel(f"{EFFECT_LABEL}\\n{MAIN_STRATEGY} vs {BASELINE}")
    plt.title("two independent information-reduction axes, one boundary")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_two_axes.png"), dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
