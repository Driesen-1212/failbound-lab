# -*- coding: utf-8 -*-
r"""提交文档的数字一致性自检。

🔴 为什么需要：4 份提交文档（5 页说明 / 原创性声明 / 基本信息 / 视频脚本）
   里散落着几十个数字——AUC* = 0.731、CI [0.557, 0.806]、−33.9、773/780/833……
   它们全部来自 `results/` 下的产物。只要重跑一次实验或改一次口径，
   文档里的数字就会**静默过期**，而评审交叉比对时一眼就能看出来。
   人工核对几十个数字必然出错，所以让机器来核对。

做法：从产物里**读真值**，再去文档里**找这个数字**，找不到/对不上就报警。

用法：
    python check_report_consistency.py
退出码 0 = 全部一致；1 = 有不一致（必须修）。
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # .../agent_lab
# 目录布局有三种：
#   本地      <proj>/unzipped/agent_lab/      文档在 <proj>/
#   沙箱      <proj>/agent_lab/               文档在 <proj>/
#   提交包    04-鼓励项-代码/…/agent_lab/      文档在 01-必交/
# 逐级往上找，找到含 submission-*.md（或含 01-必交/submission-*.md）的那一层为止。
_h = HERE
PROJ = None
for _ in range(5):
    _hit = any(f.startswith("submission-") and f.endswith(".md")
               for f in os.listdir(_h))
    _sub = os.path.join(_h, "01-必交")
    _hit_pkg = (os.path.isdir(_sub) and
                any(f.startswith("submission-") and f.endswith(".md")
                    for f in os.listdir(_sub)))
    if _hit or _hit_pkg:
        PROJ = _h if _hit else _sub
        break
    _h = os.path.dirname(_h)
if PROJ is None:
    PROJ = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")

# ⚠️ 提交用的 5 页说明是 submission-05页说明.md（≤5 页硬要求，用 md_to_pdf.py 核验）；
#    submission-技术报告-full.md 是完整展开版（无页数限制）。两份都要核。
DOCS = [
    os.path.join(PROJ, "submission-05页说明.md"),
    os.path.join(PROJ, "submission-技术报告-full.md"),
    os.path.join(PROJ, "submission-原创性声明.md"),
    os.path.join(PROJ, "submission-基本信息.md"),
    os.path.join(PROJ, "submission-视频脚本.md"),
]


def load_docs():
    text, missing = "", []
    for p in DOCS:
        if not os.path.exists(p):
            missing.append(os.path.basename(p))
            continue
        with open(p, encoding="utf-8") as f:
            text += f.read() + "\n"
    return text, missing


def read_json(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collect_expected():
    """从产物里取出「必须在文档里出现」的数字，每条带人类可读的说明。"""
    exp = []

    bf = read_json("boundary_fit.json")
    if bf:
        # 消融臂等成本：报告的主结论
        for key in ("ablate_nogate", "ablate_equal_cost"):
            d = bf.get(key)
            if not d:
                continue
            if d.get("auc_star") is not None:
                exp.append((f"{d['auc_star']:.3f}", f"消融臂零点 AUC*（{key}）"))
            if d.get("ci"):
                lo, hi = d["ci"]
                exp.append((f"{lo:.3f}", f"消融臂 AUC* CI 下界（{key}）"))
                exp.append((f"{hi:.3f}", f"消融臂 AUC* CI 上界（{key}）"))
            if d.get("gain_range"):
                exp.append((f"{d['gain_range'][0]:.1f}", f"消融臂增益下界（{key}）"))
            if d.get("n_significant_neg") is not None:
                exp.append((str(d["n_significant_neg"]),
                            f"消融臂显著负增益任务数（{key}）"))
            break

        # 主臂等成本：18/18 全正
        for key in ("main_nogate", "main_equal_cost"):
            d = bf.get(key)
            if not d:
                continue
            if d.get("gain_range"):
                exp.append((f"{d['gain_range'][0]:.1f}", f"主臂增益下界（{key}）"))
            if d.get("n_significant_pos") is not None:
                exp.append((str(d["n_significant_pos"]),
                            f"主臂显著正增益任务数（{key}）"))
            if d.get("frac_boot_with_crossing") is not None:
                pct = d["frac_boot_with_crossing"] * 100
                exp.append((f"{pct:.1f}", f"主臂支持零点的重采样占比%（{key}）"))
            break

    # Agent 校准数字：从决策日志实算
    sys.path.insert(0, HERE)
    try:
        import failbound_agent as FA
        c, measured, _ = FA.load_calibration()
        if measured:
            for v, what in ((c["agent"], "Agent 累计命中"),
                            (c["all_bo"], "无脑全用 BO 命中"),
                            (c["oracle"], "Oracle 命中")):
                exp.append((str(v), what))
    except Exception:
        pass

    # 端到端筛选（普通模式）：锚点体系的命中率，来自 screening_report.md
    rp = os.path.join(RES, "screening", "screening_report.md")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            txt = f.read()
        for m in re.finditer(r"命中率 (0\.\d+)", txt):
            exp.append((m.group(1), "端到端筛选命中率（锚点）"))

    # 端到端筛选（反事实模式）：从裁决表读真值（逐种子命中均值 + 差值）。
    # 只用 decision_summary.csv —— 它是反事实演示的权威产物；
    # 反事实报告 screening_counterfactual_report.md 由同一脚本生成，内容同源。
    ds = os.path.join(RES, "screening", "decision_summary.csv")
    if os.path.exists(ds):
        try:
            import pandas as pd
            d = pd.read_csv(ds)
            for _, r in d.iterrows():
                for col, what in (("chosen_hits_mean", "筛选逐种子命中均值"),
                                  ("counterfactual_hits_mean", "筛选反事实命中均值")):
                    exp.append((f"{r[col]:.1f}", f"{what}（{r['task']}）"))
                exp.append((f"{abs(r['hits_diff']):.1f}",
                            f"筛选裁决差值（{r['task']}）"))
        except Exception:
            pass

    return exp


def main():
    text, missing = load_docs()
    if missing:
        print(f"[warn] 以下文档不存在，未纳入核对：{missing}")
    if not text.strip():
        print("[fail] 没有可核对的文档")
        return 1

    exp = collect_expected()
    if not exp:
        print("[fail] results/ 不存在或为空，无法核对。")
        print("       本脚本只做「results 产物 → 文档」的单向比对，")
        print("       需要配套的 results/*.json（boundary_fit.json、task_props.csv 等）。")
        print("       拿到 results 后重跑即可；此为「无法验证」，不是「验证通过」。")
        return 2
    bad = []
    for val, what in exp:
        # 允许 0.731 写成 0.73、-33.9 写成 −33.9（全角减号）
        cands = {val, val.lstrip("-"), "−" + val.lstrip("-")}
        if not any(c in text for c in cands):
            bad.append((val, what))

    print(f"[check] 需核对 {len(exp)} 个数字，来自 {len(DOCS)-len(missing)} 份文档")
    if bad:
        print(f"[❌ 不一致] 以下 {len(bad)} 个数字在文档里找不到（产物已变，文档没跟上）：")
        for v, w in bad:
            print(f"    {v:<10} ← {w}")
        return 1
    print("[✅ 全部一致] 文档中的关键数字与 results/ 产物吻合")
    return 0


if __name__ == "__main__":
    sys.exit(main())
