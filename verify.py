# -*- coding: utf-8 -*-
"""verify.py —— 一条命令复算报告核心数字。

原则（E10）：
1. **重算，不是复读**：从 agent_lab/results/tests.csv + task_props.csv 实时重算，
   再与报告中的数字做逐项比对（abs(diff) < 容差）。报告数字改了、数据改了，
   这里立刻 FAIL——它验证的是「数据 → 数字」这条链，不是把结论打印一遍。
2. **零依赖**：只用 Python 标准库（csv/math/sys），任何机器直接跑。
3. **判定与增益分开打印**：先打印卡的全部判定逻辑（三行），再打印增益，
   两者不混排——为什么用 dim 是公开的代理选择，不是隐藏在计算里。

用法：
    python verify.py            # 复算全部核心数字
    python verify.py --budget 200   # 换预算档（数字随档位变化，仅 400 对报告）

预期输出末尾：ALL CHECKS PASSED（或任一 FAIL，退出码 1）。
"""
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FAILBOUND_RESULTS") or os.path.join(HERE, "agent_lab", "results")

# ---- 报告中的预期值（比对目标，容差见 TOL）----
EXPECTED = {
    "all_bo_total": 2735.40,      # 全部 36 任务-臂用 BO 的总命中（budget=400, gp_ucb）
    "card_total": 2761.57,        # 主卡（dim>2 放行 / dim≤2 拒绝）总命中
    "card_gain": 26.17,           # 主卡 − 全BO
    "old_card_total": 2726.17,    # 旧两条件卡（dim≤2 或 AUC<0.60 拒绝）总命中
    "old_card_gain": -9.23,       # 旧卡 − 全BO
    "n_tasks": 36,
    "n_rejected_main": 9,         # 主卡拒绝数（全部 dim≤2）
    "n_true_reject": 4,           # 真拒（拒绝且 BO 确实更差）
    "n_false_kill": 5,            # 误杀（拒绝但 BO 更好）
    "n_sig_miss": 0,              # 显著漏报（放行且 BO 显著更差）
    # —— 风险拦截的统计证据（§4.4.3 补）——
    "n_disaster": 4,              # 显著负增益的「灾难臂」数
    "hyper_p": 0.0021,            # 随机拒绝 9/36 时拦到全部 4 个的概率（超几何，单侧）
    "gain_d1": 11.80,             # 门槛取 1 时的卡增益（门槛敏感性）
    "gain_d5": -212.47,           # 门槛取 5 时的卡增益（跨过 2 维即翻负）
    "lobo_dielectric": -51.77,    # 留一基底：去掉 dielectric 后的卡增益
}
TOL = 0.02

DIM_THRESHOLD = 2
AUC_THRESHOLD = 0.60          # 仅旧卡使用；主卡不用 AUC
MAIN_STRATEGY = "gp_ucb"
BUDGET = 400


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load():
    props = {r["task"]: r for r in
             csv.DictReader(open(os.path.join(DATA, "task_props.csv"), encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(os.path.join(DATA, "tests.csv"), encoding="utf-8")):
        if (r["budget"] == str(BUDGET) and r["strategy"] == MAIN_STRATEGY
                and r["use_gate"] == "False"):
            rows.append(r)
    return props, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=400, choices=[25, 50, 100, 200, 400])
    a = ap.parse_args()
    global BUDGET
    BUDGET = a.budget

    props, rows = load()
    print("=" * 62)
    print("FailBound 核心数字复算  (budget=%d, strategy=%s, use_gate=False)"
          % (BUDGET, MAIN_STRATEGY))
    print("数据源: %s" % os.path.normpath(DATA))
    print("=" * 62)

    if a.budget != 400:
        print("\n注意：报告口径是 budget=400；当前档位仅做行为展示，比对将跳过。\n")

    # ---------- 第一部分：卡的全部判定逻辑（三行，没有隐藏分支） ----------
    print("【判定逻辑】主卡（交付物）")
    print("  dim > 2  → 放行：用满预算跑 BO")
    print("  dim <= %d → 拒绝：改用空间填充（或截 60%% 预算）" % DIM_THRESHOLD)
    print("  判据只读描述符维度，不读目标值——维度在开跑前可得，这是可部署的原因")
    print()
    print("【判定逻辑】旧两条件卡（已作废，仅供对照）")
    print("  放行当且仅当 dim > %d 且 AUC >= %.2f；其余拒绝" % (DIM_THRESHOLD, AUC_THRESHOLD))
    print()

    # ---------- 第二部分：从数据重算 ----------
    all_bo = sum(f(r["mean_bo"]) for r in rows)
    card_total = 0.0
    old_total = 0.0
    rej, true_rej, false_kill, sig_miss = [], 0, 0, 0
    disaster = []          # 显著负增益的臂（不管卡拦没拦）
    for r in rows:
        t = r["task"]
        dim = int(f(props[t]["dim"]))
        auc = f(props[t].get("separability_auc"))
        bo, base = f(r["mean_bo"]), f(r["mean_base"])
        diff = bo - base
        p = f(r.get("p_holm"))
        sig = str(r.get("significant")).strip().lower() == "true"
        if diff < 0 and sig:
            disaster.append((t, dim, diff))

        # 主卡
        if dim > DIM_THRESHOLD:
            card_total += bo
            if diff < 0 and sig:
                sig_miss += 1
        else:
            card_total += base
            rej.append((t, dim, diff, p, sig))
            if diff < 0:
                true_rej += 1
            else:
                false_kill += 1

        # 旧两条件卡
        if dim > DIM_THRESHOLD and auc >= AUC_THRESHOLD:
            old_total += bo
        else:
            old_total += base

    card_gain = card_total - all_bo
    old_gain = old_total - all_bo

    # ---- 风险拦截证据（三件，全部从同一张表算出）----
    n_disaster = len(disaster)
    n_hit = sum(1 for _t, d, _g in disaster if d <= DIM_THRESHOLD)
    # 超几何单侧 P(X >= n_hit)：从 36 臂里随机拒 len(rej) 个，能拦到几个灾难
    _N, _K, _n = len(rows), n_disaster, len(rej)
    hyper_p = sum(math.comb(_K, i) * math.comb(_N - _K, _n - i)
                  for i in range(n_hit, min(_K, _n) + 1)) / math.comb(_N, _n)

    def gain_at(d):
        """门槛取 d：dim<=d 的臂改用空间填充，其余用 BO；返回卡增益。"""
        return sum(f(r["mean_base"]) - f(r["mean_bo"]) for r in rows
                   if int(f(props[r["task"]]["dim"])) <= d)

    lobo = sum(((f(r["mean_base"]) - f(r["mean_bo"])) if int(f(props[r["task"]]["dim"])) <= DIM_THRESHOLD else 0.0)
               for r in rows if not r["task"].startswith("matbench_dielectric"))

    print("【重算结果】（实时从 CSV 计算，非硬编码）")
    print("  全部用 BO            总命中 = %10.2f  (n=%d 任务-臂)" % (all_bo, len(rows)))
    print("  主卡                 总命中 = %10.2f" % card_total)
    print("  主卡 − 全BO          增益   = %+10.2f" % card_gain)
    print("  旧两条件卡           总命中 = %10.2f" % old_total)
    print("  旧卡 − 全BO          增益   = %+10.2f   (已作废，仅供对照)" % old_gain)
    print()
    print("  主卡拒绝 %d 个任务-臂：真拒 %d / 误杀 %d；放行侧显著漏报 %d"
          % (len(rej), true_rej, false_kill, sig_miss))
    print()
    print("  被主卡拒绝的 %d 个，逐条（diff = BO − 空间填充，负 = 拒对）：" % len(rej))
    for t, dim, diff, p, sig in sorted(rej, key=lambda x: x[2]):
        tag = "拒对" if diff < 0 else ("误杀(显著)" if sig else "误杀(不显著)")
        print("    dim=%d  %-36s  diff=%+7.2f  p_holm=%.4f  %s" % (dim, t, diff, p, tag))
    print()

    # ---------- 第三部分之一：风险拦截证据 ----------
    print("【风险拦截】（同批数据内的一致性检验；增益均值不显著，两项分开报）")
    print("  灾难臂（显著负增益）       %d 个" % n_disaster)
    print("  卡拦住其中                 %d 个 / 随机拒绝 %d/%d 的期望拦截数 %.2f"
          % (n_hit, _n, _N, _n * n_disaster / float(_N)))
    print("  超几何单侧 P(X>=%d)        %.4f" % (n_hit, hyper_p))
    print("  门槛敏感性   d=1 %+9.2f   d=2 %+9.2f   d=5 %+9.2f" % (gain_at(1), gain_at(2), gain_at(5)))
    print("  留一基底     去掉 dielectric 后 %+9.2f（增益集中在单一基底，如实报）" % lobo)
    print()

    # ---------- 第三部分：与报告比对 ----------
    checks = [
        ("任务-臂总数 n", len(rows), EXPECTED["n_tasks"], TOL),
        ("全BO 总命中", round(all_bo, 2), EXPECTED["all_bo_total"], TOL),
        ("主卡总命中", round(card_total, 2), EXPECTED["card_total"], TOL),
        ("主卡增益", round(card_gain, 2), EXPECTED["card_gain"], TOL),
        ("旧卡总命中", round(old_total, 2), EXPECTED["old_card_total"], TOL),
        ("旧卡增益", round(old_gain, 2), EXPECTED["old_card_gain"], TOL),
        ("主卡拒绝数", len(rej), EXPECTED["n_rejected_main"], TOL),
        ("真拒数", true_rej, EXPECTED["n_true_reject"], TOL),
        ("误杀数", false_kill, EXPECTED["n_false_kill"], TOL),
        ("显著漏报数", sig_miss, EXPECTED["n_sig_miss"], TOL),
        ("灾难臂数", n_disaster, EXPECTED["n_disaster"], TOL),
        ("拦截超几何 P", round(hyper_p, 4), EXPECTED["hyper_p"], 0.001),
        ("门槛 d=1 增益", round(gain_at(1), 2), EXPECTED["gain_d1"], TOL),
        ("门槛 d=5 增益", round(gain_at(5), 2), EXPECTED["gain_d5"], TOL),
        ("留一基底 dielectric", round(lobo, 2), EXPECTED["lobo_dielectric"], TOL),
    ]
    if a.budget != 400:
        print("【比对】budget≠400，跳过与报告比对。")
        return

    print("【与报告比对】")
    failed = 0
    for name, got, exp, tol in checks:
        ok = abs(got - exp) <= tol
        print("  %-16s 重算=%-12g 预期=%-12g  %s" % (name, got, exp, "PASS" if ok else "FAIL"))
        if not ok:
            failed += 1
    print()
    if failed:
        print("RESULT: %d 项 FAIL —— 数据与报告不一致，禁止用于答辩。" % failed)
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED —— 重算数字与报告逐位一致。")


if __name__ == "__main__":
    main()
