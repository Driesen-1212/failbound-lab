# -*- coding: utf-8 -*-
"""样本外基底检验：新基底上失效边界还在不在。

口径（关键，别搞错）：
    主口径是**等成本 no-gate 臂**——材料里 "最低 −33.9"、"AUC*≈0.73" 全部出自
    `runs_nogate_ablate.jsonl`（见 boundary_by_budget.csv 的 nogate/ablate 行）。
    gate 臂是成本约束下的现实版，AUC* 与效应量都不同尺度，不能混着比。

    指标 = 预算 400 内命中数（命中 = 目标 top-10%），效应量 = gp_ucb − farthest_point。
    直接 import analyze.build_summary / prefix_metrics / fit_boundary，口径逐字一致。

三条预判（先写死，避免事后挑）：
    P1 AUC<0.65 的段：增益是否为负
    P2 AUC≥0.80 的段：增益是否为正
    P3 增益随 AUC 是否单调（Spearman）—— 边界存在的必要条件
外加：用 fit_boundary 在样本外 14 个点上重拟合，看 AUC* 是否落在冻结臂 CI 内。

用法：python holdout/analyze_holdout.py
输出：holdout/holdout_results.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 代码位置自动探测：兼容主项目（unzipped/agent_lab）、补充复算包（agent_lab）、以及平铺布局
CODE = None
for _cand in (os.path.join(ROOT, "agent_lab"),
              os.path.join(ROOT, "unzipped", "agent_lab"),
              ROOT):
    if os.path.exists(os.path.join(_cand, "analyze.py")):
        sys.path.insert(0, _cand)
        CODE = _cand
        break
else:
    raise SystemExit("找不到 analyze.py：请把 agent_lab/ 放在本目录同级（见 README）")

from analyze import build_summary, fit_boundary, load_runs  # noqa: E402

BUDGET = 400
BO = "gp_ucb"
BASE = "farthest_point"
EI = "gp_ei"
RUNS = "runs_nogate_ablate.jsonl"


def holm(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    out = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def build_arm(path, prop_csv, tag):
    runs, meta, tmeta = load_runs(path)
    if not runs:
        raise SystemExit(f"没有 run 记录：{path}")
    df = build_summary(runs, budgets=(BUDGET,))
    props = pd.read_csv(prop_csv)
    auc = dict(zip(props["task"], props["separability_auc"]))
    rows = []
    for t, g in df.groupby("task"):
        pv = g.pivot_table(index="seed", columns="strategy", values="n_hits")
        if BO not in pv or BASE not in pv:
            continue
        # 只保留两条策略都跑完的种子（补跑中断时可能有半截种子）
        pv = pv[[BO, BASE]].dropna()
        if len(pv) < 6:
            continue
        d = (pv[BO] - pv[BASE]).values.astype(float)
        mb, mo = pv[BASE].mean(), pv[BO].mean()
        p = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d) else np.nan
        rec = {"task": t, "tag": tag, "base": t.split("__")[0],
               "op": t.split("__")[1], "auc": float(auc.get(t, np.nan)),
               "hits_base": float(mb), "hits_bo": float(mo),
               "abs_gain": float(d.mean()), "gain_sd": float(d.std(ddof=1)),
               "rel_pct": float(100.0 * d.mean() / mb) if mb else np.nan,
               "n_seeds": int(len(d)), "p_wilcoxon": p,
               "dim": int(tmeta.get(t, {}).get("dim", -1))}
        if EI in pv:
            rec["abs_gain_ei"] = float((pv[EI] - pv[BASE]).mean())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("auc").reset_index(drop=True)
    out["p_holm"] = holm(out["p_wilcoxon"].fillna(1.0).values)
    out["significant"] = out["p_holm"] < 0.05
    return out


def main():
    frozen = build_arm(os.path.join(CODE, "results", RUNS),
                       os.path.join(CODE, "results", "task_props.csv"), "frozen")
    hold = build_arm(os.path.join(HERE, "agent_lab", "results", RUNS),
                     os.path.join(HERE, "holdout_props.csv"), "holdout")
    both = pd.concat([frozen, hold], ignore_index=True)

    f = lambda v: f"{v:+6.1f}" if pd.notna(v) else "   nan"
    for name, d in (("冻结臂 3 基底（样本内）", frozen), ("样本外 2 新基底", hold)):
        print(f"\n==== {name} ====")
        print(f"{'任务':<36}{'AUC':>7}{'基线':>7}{'BO':>7}{'Δ':>7}{'Δ%':>8}{'p_holm':>9}")
        for _, r in d.iterrows():
            print(f"{r['task']:<36}{r['auc']:>7.3f}{r['hits_base']:>7.1f}"
                  f"{r['hits_bo']:>7.1f}{f(r['abs_gain']):>7}{f(r['rel_pct']):>8}"
                  f"{r['p_holm']:>9.3f}")

    print("\n==== 逐基底（独立单位） ====")
    byb = []
    for tag, d in (("frozen", frozen), ("holdout", hold)):
        for b, g in d.groupby("base"):
            byb.append({"arm": tag, "base": b, "n": len(g),
                        "mean_gain": round(float(g["abs_gain"].mean()), 2),
                        "auc_lo": round(float(g["auc"].min()), 3),
                        "auc_hi": round(float(g["auc"].max()), 3),
                        "n_neg": int((g["abs_gain"] < 0).sum()),
                        "n_sig_neg": int(((g["abs_gain"] < 0) & g["significant"]).sum()),
                        "n_sig_pos": int(((g["abs_gain"] > 0) & g["significant"]).sum())})
    byb = pd.DataFrame(byb)
    print(byb.to_string(index=False))

    res = {"budget": BUDGET, "arm": "nogate",
           "frozen": frozen.round(4).to_dict(orient="records"),
           "holdout": hold.round(4).to_dict(orient="records"),
           "by_base": byb.to_dict(orient="records")}

    # ---- P1/P2/P3 ----
    lo = hold[hold["auc"] < 0.65]
    hi = hold[hold["auc"] >= 0.80]
    res["P1_low_auc"] = {"n": len(lo), "n_neg": int((lo["abs_gain"] < 0).sum()),
                         "mean_gain": round(float(lo["abs_gain"].mean()), 2) if len(lo) else None,
                         "tasks": lo["task"].tolist()}
    res["P2_high_auc"] = {"n": len(hi), "n_pos": int((hi["abs_gain"] > 0).sum()),
                          "mean_gain": round(float(hi["abs_gain"].mean()), 2) if len(hi) else None,
                          "tasks": hi["task"].tolist()}
    rh = spearmanr(hold["auc"], hold["abs_gain"])
    rf = spearmanr(frozen["auc"], frozen["abs_gain"])
    res["P3_monotone"] = {"rho_holdout": float(rh.correlation), "p_holdout": float(rh.pvalue),
                          "rho_frozen": float(rf.correlation), "p_frozen": float(rf.pvalue)}

    print("\n==== 三条预判 ====")
    print(f"P1 AUC<0.65：n={res['P1_low_auc']['n']}，负增益 "
          f"{res['P1_low_auc']['n_neg']} 个，均 Δ={f(res['P1_low_auc']['mean_gain'])}")
    print(f"P2 AUC≥0.80：n={res['P2_high_auc']['n']}，正增益 "
          f"{res['P2_high_auc']['n_pos']} 个，均 Δ={f(res['P2_high_auc']['mean_gain'])}")
    print(f"P3 Spearman：样本外 ρ={rh.correlation:+.3f}(p={rh.pvalue:.3f}) | "
          f"冻结臂 ρ={rf.correlation:+.3f}(p={rf.pvalue:.3f})")

    # ---- 边界重拟合（用冻结包同一个 fit_boundary）----
    def to_tests(d):
        x = d.copy()
        x["strategy"] = BO
        x["budget"] = BUDGET
        x["use_gate"] = False
        return x

    print("\n==== 边界重拟合（fit_boundary，lowess + bootstrap 2000）====")
    fits = {}
    for name, d in (("frozen", frozen), ("holdout_only", hold), ("frozen+holdout", both)):
        # fit_boundary 会把 tests 与 props 按 task 合并；两者是同一张表时必须只给 props
        # 两列，否则同名列被加 _x/_y 后缀（abs_gain 消失 → KeyError）。
        props = d[["task", "auc"]].rename(columns={"auc": "separability_auc"})
        s, _ = fit_boundary(to_tests(d), props, use_gate=False, budget=BUDGET)
        fits[name] = {k: s.get(k) for k in
                      ("n_tasks", "auc_star", "direction", "ci", "frac_boot_with_crossing")}
        ci = s.get("ci") or [None, None]
        print(f"{name:<16} n={s['n_tasks']:>2} AUC*={s['auc_star'] if s['auc_star'] is None else round(s['auc_star'], 3)}"
              f"  CI=[{ci[0] if ci[0] is None else round(ci[0], 3)},"
              f"{ci[1] if ci[1] is None else round(ci[1], 3)}]"
              f"  含穿零bootstrap={s['frac_boot_with_crossing']:.3f}")
    res["boundary_fits"] = fits

    op = hold.groupby("op").agg(auc=("auc", "mean"),
                                gain=("abs_gain", "mean")).reset_index()
    print("\n==== 算子族（样本外 2 基底均值） ====")
    print(op.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    ro = spearmanr(op["auc"], op["gain"])
    res["by_operator"] = op.round(4).to_dict(orient="records")
    res["operator_level"] = {"rho": float(ro.correlation), "p": float(ro.pvalue)}
    print(f"算子层（n={len(op)}）：ρ={ro.correlation:+.3f} p={ro.pvalue:.3f}")

    p = os.path.join(HERE, "holdout_results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"\n[save] {p}")


if __name__ == "__main__":
    main()
