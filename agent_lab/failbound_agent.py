# -*- coding: utf-8 -*-
"""FailBound Advisor —— 一个会**质疑自己的优化器**的科研 Agent。

=== 它解决什么问题 ===
普通材料发现 Agent 的默认行为是"上贝叶斯优化"。但我们的实验证明：
在描述符不足的任务上，BO 会显著劣于空间填充（实测 −33.9 个命中，Cliff's δ = −0.629）。
**Agent 必须知道自己什么时候不该信自己的优化。**

=== 🔴 关于"能不能自动判断"的诚实结论（2026-10-17 校准）===
我们**没能**构造出一个能可靠预测失效的判据。在 18 个消融臂任务上
（预算 400 等成本口径，与 BO 增益的 Spearman 秩相关，数字由
`predictor_strength.csv` 实测，可由 analyze.py 复现）：

    基线命中率  vs BO 增益   ρ = −0.289  p = 0.246   （不显著）
    可学习性AUC vs BO 增益   ρ = +0.513  p = 0.030   （显著，但弱）
    描述符维度  vs BO 增益   ρ = +0.452  p = 0.060   （边缘不显著）

注意 AUC 这行与旧版文档的差别：它**是**显著的，但弱到不足以逐任务下注 ——
只解释约 26% 的秩方差，且在最需要它的不确定带内**翻转方向**
（AUC 0.593 增益 −20.6，AUC 0.633 却 +20.7）。所以正确的结论不是
"信号不存在"，而是"**信号太弱，不能当自动开关**"。
按最优规则重做后的累计命中 773，**仍低于"无脑全用 BO"的 780**
（Oracle 833，无脑空间填充 539）—— 逐任务胜率上我们确实打不过无脑。

所以本 Agent **不宣称能自动做对决策**。它的价值是：
  ① 在**极端**情况下（≤2 维、AUC<0.60）明确发出拒绝信号 ——
     这些正是实测出现 −33.9 的那类任务，宁可误伤也要避免灾难性失败；
     （误拒/漏报的账已逐任务算清，见 audit_decision_card.py：
       漏报 0/23，误拒 7/36 —— 卡把 −33.9 的右尾换成了 +14~+32 的左尾）
  ② 输出一份可审计的诊断报告，把判断依据交给人类科学家；
  ③ **把自己被校准的数据一并打印出来**——包括"我打不过无脑 BO"这个事实。
把不确定性写在脸上，比假装确定更有用。这也是本工作的核心科学结论之一：
**不存在一个单值指标能可靠到让你把预算赌在它上面。**

=== 它的决策闭环（四步，可审计）===
   ① 诊断    用一小笔预算（默认 25）跑空间填充，测出"不优化能拿多少"
   ② 标定    用与实验完全一致的口径算可学习性 AUC（RF 5-fold，top-10% 标签）
   ③ 决策    对照边界，输出 USE_BO / USE_SPACE_FILLING，并**显式给出不确定性**
   ④ 执行    按自己的决定跑完剩余预算，并用实际结果检验这个决定

=== 为什么这不只是"又跑一次 BO" ===
   - 它会**拒绝**使用自己的优化器，并在输出里说明拒绝理由（可审计）
   - 它会报告"我对这个判断有多不确定"，而不是假装确定
   - 决定与结果一起落盘，形成可复核的证据链

用法：
    python failbound_agent.py matbench_log_gvrh__pca5
    python failbound_agent.py matbench_dielectric__pca2 --budget 200 --execute
    python failbound_agent.py --all              # 对全部登记任务跑一遍决策（不出图）

输出：stdout 决策卡 + results/agent_decisions.jsonl（追加，机器可读）
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (HIT_Q, N_INIT, OUT_DIR, SEED, discover_tasks)  # noqa: E402
from cost_model import cost_vector, gate_pool, is_hit, threshold_of  # noqa: E402
from strategies import (FarthestPointSelector, pick_initial, standardize)  # noqa: E402
from trial import run_trial  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

DIAG_BUDGET = 25          # ① 诊断阶段预算（预注册口径里的最小档）
DECISION_LOG = os.path.join(OUT_DIR, "agent_decisions.jsonl")

# ---------------------------------------------------------------- 决策阈值
# 🔴 曾经这几个数（0.60 / 0.75 / 2 / 0.20）硬编码在 decide() 的函数体里。
#    问题不只是可读性：要提交"Agent 配置导出"时，只能手写一份 YAML，
#    而手写的 YAML 和代码里的数字**没有任何机制保证一致**——改了一处就会漂。
# ✅ 提为模块级常量后，export_agent_config.py 直接从这里读，
#    导出的配置文件与运行时行为**同源**，不可能对不上。
AUC_FAILZONE = 0.60     # AUC < 此值 → 实测失效区上界；**只标低置信，不否决**
AUC_UNCERTAIN = 0.75    # AUC ∈ [AUC_FAILZONE, 此值) → 不确定带，低置信度
DIM_REJECT = 2          # 描述符维度 ≤ 此值 → 拒绝 BO（唯一会翻转 use_bo 的规则）
BASE_HIT_WARN = 0.20    # 空间填充基线命中率 > 此值 → 提示 exploitation 未必有优势


# ---------------------------------------------------------------- ⑤ 校准
# 🔴 曾经的问题：校准段的 773/780/833 是**硬编码**在 print 里的。
#    一旦补跑新任务或换预算档，这些数字就与报告对不上，
#    而"把校准数据写在脸上"恰恰是本 Agent 的核心卖点 —— 写错等于自毁信誉。
# ✅ 现在从 agent_decisions.jsonl 实测累加；日志不足时才回退到预置值，
#    并显式标注"（预置值，非本次实测）"，绝不冒充实测。
PRESET_CALIB = {"n_tasks": 18, "agent": 773, "all_bo": 780, "oracle": 833,
                "all_spacefill": 539}
# 预测力预置值 —— 🔴 旧版这里写的是 (−0.362, +0.365, +0.393)，**任何当前口径
#    都复现不出来**（已用 tests.csv + task_props.csv 逐口径重算比对）。
#    它们与 predictor_strength.csv 的实测值矛盾，而后者可由 analyze.py 复现。
# ✅ 改为从 predictor_strength.csv 动态读取（见 load_predictor_rhos），
#    预置值与 CSV 的消融臂/等成本/预算400 一格保持一致，仅作兜底。
#    同时修正叙述：AUC 在消融臂**显著**（p=0.030）但弱，不是"不显著"。
PRESET_RHOS = {"base_hit_rate": (-0.289, 0.246), "auc": (0.513, 0.030),
               "dim": (0.452, 0.060)}


def load_predictor_rhos():
    """从 predictor_strength.csv 读消融臂/等成本/最高预算档的三个秩相关。

    为什么必须读文件而不是写死：这三个数字是"Agent 把自己被校准的数据
    打印出来"的核心内容，写死就会和 analyze.py 的产物漂移（已经漂过一次，
    见 PRESET_RHOS 上方的说明）。读不到文件时才回退预置值并标注来源。
    """
    path = os.path.join(OUT_DIR, "predictor_strength.csv")
    if not os.path.exists(path):
        return dict(PRESET_RHOS), False
    try:
        ps = pd.read_csv(path)
        cell = ps[(ps.arm == "ablate") & (ps.use_gate == False)]
        if cell.empty:
            return dict(PRESET_RHOS), False
        b = int(cell["budget"].max())
        cell = cell[cell.budget == b]
        out = {}
        for key, pred in (("base_hit_rate", "base_hit_rate"),
                          ("auc", "separability_auc"), ("dim", "dim")):
            row = cell[cell.predictor == pred]
            if row.empty:
                out[key] = PRESET_RHOS[key]
            else:
                out[key] = (float(row.iloc[0]["spearman_rho"]),
                            float(row.iloc[0]["p_raw"]))
        return out, True
    except Exception:
        return dict(PRESET_RHOS), False


def load_calibration():
    """从决策日志累加校准数字。

    为什么能算：每条记录同时有 chosen_hits（按 Agent 决策执行）和
    counterfactual_hits（按另一策略执行）。于是
        无脑全用 BO   = Σ(决策为BO ? chosen : counterfactual)
        无脑空间填充  = Σ(决策为SF ? chosen : counterfactual)
        Oracle        = Σ max(chosen, counterfactual)
    返回 (dict, 是否来自实测)。同一任务重复跑只取最后一次。
    """
    if not os.path.exists(DECISION_LOG):
        return dict(PRESET_CALIB), False, 0
    rows = {}
    try:
        with open(DECISION_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("executed") and "chosen_hits" in d:
                    rows[d.get("task")] = d      # 后写的覆盖先写的
    except Exception:
        return dict(PRESET_CALIB), False, 0

    if not rows:
        return dict(PRESET_CALIB), False, 0

    agent = bo = sf = oracle = 0
    for d in rows.values():
        ch, cf = int(d["chosen_hits"]), int(d["counterfactual_hits"])
        is_bo = bool(d.get("use_bo"))
        agent += ch
        bo += ch if is_bo else cf
        sf += cf if is_bo else ch
        oracle += max(ch, cf)
    calib = {"n_tasks": len(rows), "agent": agent, "all_bo": bo,
             "oracle": oracle, "all_spacefill": sf}
    # 任务数明显不足（例如只跑了 1 个任务就来看校准）→ 仍然回退，避免用小样本吓人
    measured = len(rows) >= 10
    return (calib if measured else dict(PRESET_CALIB)), measured, len(rows)


def calibration_block():
    """生成 ⑤ 校准段的几行文本。"""
    c, measured, n_rows = load_calibration()
    src = f"本日志实测（{c['n_tasks']} 个任务）" if measured else \
          f"预置值（{PRESET_CALIB['n_tasks']} 任务实测；本次日志仅 {n_rows} 条，不足以重算）"
    gap = c["agent"] - c["all_bo"]
    lines = [
        f"⑤ 校准（{src}，务必连同结论一起读）：本规则累计命中 {c['agent']}，",
        f"   {'低于' if gap < 0 else '高于'}『无脑全用 BO』的 {c['all_bo']}"
        f"（差 {gap:+d}）；Oracle {c['oracle']}，无脑空间填充 {c['all_spacefill']}。",
    ]
    r, from_csv = load_predictor_rhos()
    rsrc = "predictor_strength.csv 实测" if from_csv else "预置值"
    lines.append(
        f"   → 判据信号本身预测力有限（{rsrc}，消融臂等成本口径）："
        f"AUC ρ={r['auc'][0]:+.3f} p={r['auc'][1]:.3f}（显著但弱，仅解释约 26% 秩方差）；"
        f"维度 ρ={r['dim'][0]:+.3f} p={r['dim'][1]:.3f}；"
        f"基线命中率 ρ={r['base_hit_rate'][0]:+.3f} p={r['base_hit_rate'][1]:.3f}。")
    lines.append("   → 不确定带内 AUC 还会翻转方向（0.593→−20.6，0.633→+20.7），"
                 "故只做保守熔断，不做自动开关。")
    lines.append("   → 本 Agent 只保证在极端情形下拒绝优化，不保证逐任务做对。")
    return lines


# ---------------------------------------------------------------- ② 可学习性
def learnability_auc(X, y, q=HIT_Q, seed=0):
    """与实验完全一致的口径：标签 = 目标 top-(1-q) 分位，RF 5-fold 交叉验证 ROC-AUC。"""
    thr = float(np.quantile(y, q))
    lab = (y >= thr).astype(int)
    if lab.sum() < 5 or len(lab) - lab.sum() < 5:
        return None
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    try:
        p = cross_val_predict(clf, X, lab, cv=5, method="predict_proba")[:, 1]
        return float(roc_auc_score(lab, p))
    except Exception:
        return None


# ---------------------------------------------------------------- ① 诊断采样
def probe_space_filling(X_std, y, cost, thr, task, budget, seed):
    """用一小笔预算跑空间填充，测出"完全不优化能拿到多少命中"。"""
    rng = np.random.default_rng(seed)
    n_pool = len(y)
    fp = FarthestPointSelector(X_std)

    chosen, spent = [], 0.0
    for i in pick_initial(n_pool, N_INIT, rng):
        chosen.append(int(i)); spent += cost[i]; fp.update(int(i))
    while spent < budget and len(chosen) < n_pool:
        avail = np.setdiff1d(np.arange(n_pool), np.array(chosen))
        if len(avail) == 0:
            break
        i = int(fp.select(avail, 1)[0])
        chosen.append(i); spent += cost[i]; fp.update(i)

    hits = [int(is_hit(y[i], task, thr)) for i in chosen]
    # 🧹 2026-10-17 清理：这里曾创建 GPSelector 并把 (u_all, gp) 塞进返回值
    #    "供后续复用"，但**全项目无人消费**，而 GP 对象不可 pickle ——
    #    一旦有人把 probe 结果跨进程传递就会炸。死代码直接删。
    return {"n_hits": int(sum(hits)), "n_exp": len(chosen),
            "hit_rate": float(sum(hits) / max(len(hits), 1)),
            "chosen": chosen, "spent": float(spent)}


def run_strategy(X_std, y, cost, thr, task, budget, seed, strategy):
    """按给定策略跑完整预算（循环体统一到 trial.run_trial）。

    ⚠️ 重构前这里写的是 `GPSelector()`（默认 kappa），与主实验的
       `GPSelector(kappa=GP_KAPPA)` 是两份独立默认值 —— 今天是 2.0 vs 2.0 才没出事。
       统一后不可能再脱钩。
    """
    r = run_trial(X_std, y, cost, thr, task, budget, seed, strategy)
    return {"n_hits": r["n_hits"], "n_exp": r["n_exp"], "hit_rate": r["hit_rate"]}


# ---------------------------------------------------------------- ③ 决策
def decide(auc, dim, base_hit_rate, boundary, legacy_auc_reject=False):
    """对照失效边界做决策，并**显式给出不确定性**。

    决策依据全部来自实测（见 preregistration.md §10 与报告 P3）：
      - 轴 A（目标加噪，描述符完整）：18/18 显著为正，从不有害
      - 轴 B（描述符削减）：存在失效，最低 −33.9，但**只在 2/3 个基底上出现**，
        且 AUC 的秩相关只有 ρ=+0.513 —— 所以 AUC 不足以单独定决策
    """
    reasons, warns = [], []
    use_bo = True
    confidence = "medium"

    if auc is None:
        warns.append("可学习性 AUC 无法计算（正负样本过少），按默认策略执行")
        confidence = "low"
    elif auc < AUC_FAILZONE:
        # 默认不否决：AUC 条款在 8 个基底上净亏 35.40（主臂拦下 4 个实测全为正），
        # 已退为低置信标注；唯一会翻转 use_bo 的规则只剩 dim ≤ DIM_REJECT。
        # legacy_auc_reject=True 可复现预注册的两参数版（放行 23 / 拒绝 13）。
        if legacy_auc_reject:
            use_bo = False
            reasons.append(f"AUC={auc:.3f} < {AUC_FAILZONE:.2f}，"
                           f"落入实测失效区（轴 B 最低增益 −33.9）")
            confidence = "medium"
        else:
            reasons.append(f"AUC={auc:.3f} < {AUC_FAILZONE:.2f}：落入实测失效区，"
                           f"但该条规则实测净亏 35.40，已从否决条件降为低置信标注")
            warns.append("该区间确有灾难性损失（最低 −33.9），代价是漏报右侧，"
                         "不是靠 AUC 阈值本身兜住的")
            confidence = "low"
    elif auc < AUC_UNCERTAIN:
        reasons.append(f"AUC={auc:.3f} 处于 {AUC_FAILZONE:.2f}–{AUC_UNCERTAIN:.2f} 的不确定带")
        warns.append("该区间内逐任务结果不一致（AUC 0.593 的增益是 −20.6，"
                     "AUC 0.633 却是 +20.7），AUC 在此区间预测力有限（ρ=+0.513）")
        confidence = "low"

    if dim is not None and dim <= DIM_REJECT:
        use_bo = False
        reasons.append(f"描述符仅 {dim} 维：实测 4 个显著负增益任务"
                       f"全部来自 ≤{DIM_REJECT} 维")
        confidence = "medium"

    if base_hit_rate is not None and base_hit_rate > BASE_HIT_WARN:
        warns.append(f"空间填充基线命中率已高达 {base_hit_rate:.3f}"
                     f"—— 高价值区域可能分散且广阔，BO 的 exploitation 未必有优势"
                     f"（注：此机制为待验证假设，代理指标相关性未达显著 p=0.246）")
        if confidence == "medium":
            confidence = "low"

    if not reasons:
        reasons.append("未触发任何失效条件，按默认（BO）执行")

    return {"strategy": "gp_ucb" if use_bo else "farthest_point",
            "use_bo": bool(use_bo), "confidence": confidence,
            "reasons": reasons, "warnings": warns}


# ---------------------------------------------------------------- 主流程
def run_agent(task_name, task, budget=200, seed_offset=0, execute=True,
              use_gate=False):
    df = pd.read_csv(task["path"])
    pool = gate_pool(df, use_gate=use_gate)
    feat = [c for c in pool.columns if c.startswith("f")]
    X_std = standardize(pool[feat].values.astype(float))
    y = pool[task["target_col"]].values.astype(float)
    thr = threshold_of(y, task)
    cost = cost_vector(pool, use_gate=use_gate)
    dim = len(feat)
    seed = SEED + seed_offset

    t0 = time.time()
    # ① 诊断
    probe = probe_space_filling(X_std, y, cost, thr, task, DIAG_BUDGET, seed)
    # ② 标定
    auc = learnability_auc(pool[feat].values.astype(float), y)
    # ③ 决策
    boundary = {}
    bpath = os.path.join(OUT_DIR, "boundary_fit.json")
    if os.path.exists(bpath):
        try:
            boundary = json.load(open(bpath, encoding="utf-8"))
        except Exception:
            boundary = {}
    d = decide(auc, dim, probe["hit_rate"], boundary)

    out = {"task": task_name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "budget": int(budget), "dim": int(dim), "auc": auc,
           "diag_hit_rate": probe["hit_rate"], "diag_n_hits": probe["n_hits"],
           "decision": d["strategy"], "use_bo": d["use_bo"],
           "confidence": d["confidence"], "reasons": d["reasons"],
           "warnings": d["warnings"], "executed": False}

    # ④ 执行并检验
    if execute:
        r = run_strategy(X_std, y, cost, thr, task, budget, seed, d["strategy"])
        # 反事实：如果按另一个决定执行会怎样
        alt = "farthest_point" if d["use_bo"] else "gp_ucb"
        r_alt = run_strategy(X_std, y, cost, thr, task, budget, seed, alt)
        out.update({"executed": True, "chosen_hits": r["n_hits"],
                    "chosen_hit_rate": r["hit_rate"],
                    "counterfactual_strategy": alt,
                    "counterfactual_hits": r_alt["n_hits"],
                    "counterfactual_hit_rate": r_alt["hit_rate"],
                    "decision_was_better": bool(r["n_hits"] >= r_alt["n_hits"])})

    out["elapsed_s"] = round(time.time() - t0, 1)

    # ---- 决策卡 ----
    print(f"\n{'='*66}")
    print(f"任务：{task_name}   （描述符 {dim} 维，候选 {len(y)}）")
    print(f"{'='*66}")
    print(f"① 诊断：预算 {DIAG_BUDGET} 的空间填充命中 {probe['n_hits']}/{probe['n_exp']}"
          f" = {probe['hit_rate']:.3f}（随机基线期望 0.100）")
    print(f"② 标定：可学习性 AUC = "
          f"{'无法计算' if auc is None else f'{auc:.3f}'}")
    print(f"③ 决策：{'【用贝叶斯优化】' if d['use_bo'] else '【改用空间填充 —— 拒绝优化】'}"
          f"  置信度 {d['confidence']}")
    for r_ in d["reasons"]:
        print(f"      · {r_}")
    for w in d["warnings"]:
        print(f"    ⚠ {w}")
    if out["executed"]:
        verdict = "✅ 决定正确" if out["decision_was_better"] else "❌ 决定劣于反事实"
        print(f"④ 执行：预算 {budget} → 命中 {out['chosen_hits']}"
              f"（{out['chosen_hit_rate']:.3f}）"
              f"  |  若改用 {out['counterfactual_strategy']} 则为 "
              f"{out['counterfactual_hits']}  {verdict}")
    for line in calibration_block():
        print(line)

    with open(DECISION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="任务名；不给则用 --all")
    ap.add_argument("--all", action="store_true", help="对全部登记任务跑一遍")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--arm", default="main", choices=["main", "ablate"])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--no-execute", action="store_true", help="只决策不执行")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = discover_tasks(arm=a.arm)
    if a.all:
        names = list(tasks)
    elif a.task:
        if a.task not in tasks:
            raise SystemExit(f"未知任务 {a.task}（{a.arm} 臂共 {len(tasks)} 个）")
        names = [a.task]
    else:
        raise SystemExit("给一个任务名，或用 --all")

    print(f"[agent] 待决策任务 {len(names)} 个 | 预算 {a.budget} | "
          f"{'仅决策' if a.no_execute else '决策+执行+反事实检验'}")
    ok = 0
    for n in names:
        o = run_agent(n, tasks[n], budget=a.budget, execute=not a.no_execute,
                      use_gate=a.gate)
        if o.get("decision_was_better"):
            ok += 1
    if not a.no_execute:
        print(f"\n[agent] 决定优于（或持平）反事实：{ok}/{len(names)}")
    print(f"[agent] 决策日志 → {DECISION_LOG}")


if __name__ == "__main__":
    main()
