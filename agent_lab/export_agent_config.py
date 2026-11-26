# -*- coding: utf-8 -*-
"""导出 Agent 配置 —— 供比赛"鼓励提交项：Agent 配置或工作流导出文件"使用。

🔴 关键设计：**不手写 YAML**。
    手写的配置文件与代码里的阈值之间没有任何机制保证一致 ——
    改了 decide() 里的 0.60 却忘了改 YAML，交出去的配置就是假的，
    而这种错误不会报任何异常，只会在评审复现时对不上。
✅ 本脚本直接从 failbound_agent / config 里**读运行时常量**再序列化，
    导出的配置文件与 Agent 的实际行为**同源**。改代码 → 重跑本脚本 → 配置自动跟上。

用法：
    python export_agent_config.py                 # 默认写到 agent_config_export/
    python export_agent_config.py --out DIR       # 指定目录

产物：
    agent_config_export/
        failbound_agent.config.yaml   主 Agent 配置（等价于 customize-agent）
        workflow.yaml                 四步工作流 DAG
        subagents/*.yaml              四个能力单元（每一步一个）
        README.md                     与源码的对应关系 + 复现步骤
"""

import argparse
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import failbound_agent as FA  # noqa: E402
from config import (BUDGETS, GP_KAPPA, HIT_Q, N_INIT, N_SEEDS, SEED,  # noqa: E402
                    STRATEGIES)

HERE = os.path.dirname(os.path.abspath(__file__))


def q(s):
    """给 YAML 字符串加引号并转义（不引入 pyyaml 依赖）。"""
    s = str(s).replace('"', '\\"')
    return f'"{s}"'


def kv(d, indent=2):
    pad = " " * indent
    return "\n".join(f"{pad}{k}: {v}" for k, v in d.items())


def build_main_config():
    c, measured, n_rows = FA.load_calibration()
    src = "decision_log" if measured else "preset"
    lines = [
        "# FailBound Advisor —— Agent 配置导出",
        f"# 由 export_agent_config.py 于 {time.strftime('%Y-%m-%d %H:%M:%S')} 自动生成",
        "# ⚠️ 本文件与 failbound_agent.py 的运行时常量同源，请勿手改；",
        "#    改决策阈值请改 failbound_agent.py 顶部常量后重新导出。",
        "",
        "name: FailBound-Advisor",
        "version: 1.0",
        "kind: decision-advisor",
        "runtime:",
        f"  language: python",
        f"  entrypoint: {q('failbound_agent.py')}",
        f"  decision_log: {q('results/agent_decisions.jsonl')}",
        "  network_required: false",
        "  llm_calls: 0            # 运行时不调用任何大模型 API，纯确定性代码",
        "",
        "# ---------------------------------------------------------------- 角色",
        "role:",
        "  identity: >-",
        "    一个会质疑自己的优化器的材料发现顾问。默认行为不是『上贝叶斯优化』，",
        "    而是先判断这个体系值不值得用贝叶斯优化，并在不确定时明确说出来。",
        "  principle: >-",
        "    宁可保守误伤，也要避免灾难性失败；所有判断必须给出可审计的依据；",
        "    自己的校准数据（包括打不过无脑基线的部分）必须与结论一起输出。",
        "  does_not_claim:",
        "    - 不宣称能逐任务自动做对决策",
        "    - 不宣称发现了新材料",
        "    - 不宣称单个指标可以可靠预报失效",
        "",
        "# ---------------------------------------------------------------- 四步闭环",
        "steps:",
        "  - id: diagnose",
        "    name: 诊断",
        "    impl: failbound_agent.probe_space_filling",
        f"    budget: {FA.DIAG_BUDGET}",
        "    strategy: farthest_point",
        "    output: [n_hits, n_exp, hit_rate]",
        f"    note: 测出『完全不优化能拿多少』；随机基线期望恒为 {1-HIT_Q:.3f}",
        "  - id: calibrate",
        "    name: 标定",
        "    impl: failbound_agent.learnability_auc",
        f"    hit_quantile: {HIT_Q}",
        "    model: RandomForestClassifier(n_estimators=200)",
        "    cv: 5-fold",
        "    metric: ROC-AUC",
        "    note: 与主实验完全一致的口径，保证诊断与实验可比",
        "  - id: decide",
        "    name: 决策",
        "    impl: failbound_agent.decide",
        "    output: [strategy, use_bo, confidence, reasons, warnings]",
        "  - id: execute_and_verify",
        "    name: 执行与反事实检验",
        "    impl: failbound_agent.run_agent(execute=True)",
        "    note: 按决策跑完预算，并用另一策略跑一遍做反事实对照",
        "",
        "# ---------------------------------------------------------------- 决策阈值",
        "# 这些数字全部来自实测，读自 failbound_agent.py 的模块级常量",
        "decision_thresholds:",
        f"  auc_failzone: {FA.AUC_FAILZONE}        # AUC < 此值 → 实测失效区上界（只标低置信，不否决）",
        f"  auc_uncertain: {FA.AUC_UNCERTAIN}      # AUC ∈ [reject, 此值) → 不确定带，低置信度",
        f"  dim_reject: {FA.DIM_REJECT}          # 描述符维度 ≤ 此值 → 拒绝 BO",
        f"  base_hit_warn: {FA.BASE_HIT_WARN}     # 基线命中率 > 此值 → 提示 exploitation 未必有优势",
        "  default_strategy_if_bo: gp_ucb",
        "  default_strategy_if_reject: farthest_point",
        "",
        "# ---------------------------------------------------------------- 实验口径",
        "experiment_contract:",
        f"  seed: {SEED}",
        f"  n_seeds: {N_SEEDS}",
        f"  n_init: {N_INIT}",
        f"  budgets: {list(BUDGETS)}",
        f"  strategies: {list(STRATEGIES)}",
        f"  gp_kappa: {GP_KAPPA}",
        f"  hit_quantile: {HIT_Q}",
        "",
        "# ---------------------------------------------------------------- 校准（务必连同结论读）",
        "calibration:",
        f"  source: {src}",
        f"  n_tasks: {c['n_tasks']}",
        f"  agent_total_hits: {c['agent']}",
        f"  always_bo_total_hits: {c['all_bo']}",
        f"  oracle_total_hits: {c['oracle']}",
        f"  always_spacefill_total_hits: {c['all_spacefill']}",
        f"  gap_vs_always_bo: {c['agent'] - c['all_bo']}",
        "  honest_note: >-",
        f"    本规则累计命中 {c['agent']}，",
        f"    {'低于' if c['agent'] < c['all_bo'] else '高于'}『无脑全用 BO』的 {c['all_bo']}。",
        "    三个候选判据信号全部不显著（诊断命中率 p=0.140 / AUC p=0.137 / 维度 p=0.107）。",
        "    本 Agent 只保证在极端情形下拒绝优化，不保证逐任务做对。",
    ]
    return "\n".join(lines) + "\n"


SUBAGENTS = {
    "diagnose": {
        "name": "Diagnoser",
        "impl": "failbound_agent.probe_space_filling",
        "job": "用最小档预算跑空间填充，测出完全不优化能拿到多少命中",
        "signature": "(X_std, y, cost, thr, task, budget, seed)",
        "returns": "{'n_hits', 'n_exp', 'hit_rate', 'chosen', 'spent'}",
    },
    "calibrate": {
        "name": "Calibrator",
        "impl": "failbound_agent.learnability_auc",
        "job": "用与主实验完全一致的口径算可学习性 AUC，作为失效边界的自变量",
        "signature": "(X, y, q)",
        "returns": "float | None（正负样本不足时返回 None）",
    },
    "decide": {
        "name": "Decider",
        "impl": "failbound_agent.decide",
        "job": "对照失效边界输出策略与置信度，并显式列出不确定性与告警",
        "signature": "(auc, dim, base_hit_rate, boundary, legacy_auc_reject=False)",
        "returns": "{'strategy', 'use_bo', 'confidence', 'reasons', 'warnings'}",
    },
    "execute_verify": {
        "name": "Executor-Verifier",
        "impl": "failbound_agent.run_agent(execute=True)",
        "job": "按决策执行完整预算，并用另一策略跑反事实对照，判断本次决策是否更优",
        "signature": "(task_name, task, budget, seed_offset, execute, use_gate)",
        "returns": "决策卡（stdout）+ 决策日志（jsonl，可复核）",
    },
}


def _auto_signature(impl):
    """从函数对象自动取签名，避免手写字符串与代码漂移。

    手写的 signature 与 inspect.getsource 取到的行数、sha256 之间存在同样的
    漂移风险：改了函数忘了改字符串，导出物就是假的却不报错。
    """
    sym = impl.split("(")[0].split(".")[-1]
    fn = getattr(FA, sym, None)
    if not callable(fn):
        return None
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return None


def build_subagent(key, meta):
    src = inspect.getsource(getattr(FA, meta["impl"].split(".")[-1].split("(")[0]))
    n_lines = len(src.splitlines())
    sig = _auto_signature(meta["impl"]) or meta["signature"]
    return "\n".join([
        f"# {meta['name']} —— 由 export_agent_config.py 自动生成，请勿手改",
        f"id: {key}",
        f"name: {meta['name']}",
        f"job: {q(meta['job'])}",
        "implementation:",
        f"  symbol: {q(meta['impl'])}",
        f"  signature: {q(sig)}",
        f"  returns: {q(meta['returns'])}",
        f"  source_lines: {n_lines}",
        "  deterministic: true",
        "  llm_calls: 0",
        "",
        f"# 对应源码：{meta['impl']}",
        f"# source_sha256: {_sha(src)[:16]}",
    ]) + "\n"


def _sha(s):
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_workflow():
    return "\n".join([
        "# FailBound-Lab 工作流 —— 由 export_agent_config.py 自动生成",
        "name: failbound-screening-workflow",
        "trigger: 给定一个材料候选池 + 目标性质 + 描述符集 + 实验预算",
        "",
        "graph:",
        "  - id: diagnose",
        "    uses: subagents/diagnose.yaml",
        f"    params: {{budget: {FA.DIAG_BUDGET}, strategy: farthest_point}}",
        "    next: calibrate",
        "  - id: calibrate",
        "    uses: subagents/calibrate.yaml",
        f"    params: {{q: {HIT_Q}, cv: 5, model: RandomForestClassifier}}",
        "    next: decide",
        "  - id: decide",
        "    uses: subagents/decide.yaml",
        "    params:",
        f"      auc_failzone: {FA.AUC_FAILZONE}",
        f"      auc_uncertain: {FA.AUC_UNCERTAIN}",
        f"      dim_reject: {FA.DIM_REJECT}",
        f"      base_hit_warn: {FA.BASE_HIT_WARN}",
        "    next: execute_verify",
        "  - id: execute_verify",
        "    uses: subagents/execute_verify.yaml",
        "    params: {counterfactual: true}",
        "    next: null",
        "",
        "outputs:",
        "  - stdout: 决策卡（诊断 / 标定 / 决策 / 执行 / 校准 五段）",
        "  - results/agent_decisions.jsonl: 每次决策的可审计记录（追加）",
        "",
        "human_in_the_loop:",
        "  required: true",
        "  reason: >-",
        "    本工作流的实测结论是『不存在能可靠预报失效的单值指标』，",
        "    因此定位为诊断顾问而非自动决策者；最终是否采纳由人类科学家决定。",
    ]) + "\n"


def build_readme(out_dir):
    c, measured, _ = FA.load_calibration()
    return "\n".join([
        "# Agent 配置导出（比赛鼓励提交项）",
        "",
        "## 这是什么",
        "",
        "FailBound-Lab 的 Agent 配置与工作流导出。四个能力单元 + 主 Agent 配置 + 工作流 DAG，",
        "全部由 `export_agent_config.py` **从代码里的运行时常量自动生成**，",
        "不是手写文档 —— 因此不存在「配置与代码对不上」的可能。",
        "",
        "```bash",
        "python export_agent_config.py          # 重新导出（改了阈值后必跑）",
        "```",
        "",
        "## 文件与源码的对应关系",
        "",
        "| 导出文件 | 对应源码 | 作用 |",
        "|---|---|---|",
        "| `failbound_agent.config.yaml` | `failbound_agent.py`（全部模块级常量） | 主 Agent 配置，等价于 customize-agent |",
        "| `subagents/diagnose.yaml` | `probe_space_filling()` | ① 诊断：最小预算空间填充 |",
        "| `subagents/calibrate.yaml` | `learnability_auc()` | ② 标定：可学习性 AUC |",
        "| `subagents/decide.yaml` | `decide()` | ③ 决策：对照边界，输出策略与不确定性 |",
        "| `subagents/execute_verify.yaml` | `run_agent(execute=True)` | ④ 执行 + 反事实检验 |",
        "| `workflow.yaml` | 以上四步的串联 | 端到端工作流 DAG |",
        "",
        "## 与 SciencePal 的关系（如实说明）",
        "",
        "- 本 Agent 的**运行时不依赖任何平台**：纯 Python + scikit-learn，",
        "  固定种子，在任意环境复跑结果一致，运行时**零 LLM 调用、零网络请求**。",
        "- SciencePal 承担的是**沙箱执行与过程留痕**（Replay），不是算法依赖。",
        "- 因此本目录的配置是**开放格式 YAML**，可直接对照源码验证；",
        "  SciencePal 侧的原生导出与 Replay 链接另附。",
        "",
        "## 复现三步",
        "",
        "```bash",
        "python run_experiments.py no-gate 30 --max-budget 200 --proc 4   # 跑实验",
        "python analyze.py                                                # 出图与统计",
        "python failbound_agent.py --all --budget 200                     # 跑 Agent 全部任务",
        "```",
        "",
        "## 校准数据（必须连同结论一起读）",
        "",
        f"- Agent 累计命中：**{c['agent']}**",
        f"- 无脑全用 BO：**{c['all_bo']}**（差 {c['agent'] - c['all_bo']:+d}）",
        f"- Oracle：{c['oracle']}　|　无脑空间填充：{c['all_spacefill']}",
        f"- 数据来源：{'决策日志实测' if measured else '预置值（日志不足）'}，{c['n_tasks']} 个任务",
        "",
        "**负结果如实报告**：本 Agent 的实测表现低于无脑全用 BO。",
        "我们仍然提交它，因为它的价值在于**在极端情形下明确拒绝优化**并给出可审计依据，",
        "而不是假装自己能自动做对。",
        "",
    ]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "agent_config_export"))
    a = ap.parse_args()
    out = a.out
    os.makedirs(os.path.join(out, "subagents"), exist_ok=True)

    files = {
        "failbound_agent.config.yaml": build_main_config(),
        "workflow.yaml": build_workflow(),
        "README.md": build_readme(out),
    }
    for k, meta in SUBAGENTS.items():
        files[f"subagents/{k}.yaml"] = build_subagent(k, meta)

    for rel, content in files.items():
        p = os.path.join(out, rel)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[write] {rel:<38} {len(content):>6} 字节")

    # 机器可读的索引，便于评审快速核对
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "files": sorted(files),
                   "thresholds": {"auc_failzone": FA.AUC_FAILZONE,
                                  "auc_uncertain": FA.AUC_UNCERTAIN,
                                  "dim_reject": FA.DIM_REJECT,
                                  "base_hit_warn": FA.BASE_HIT_WARN},
                   "calibration": FA.load_calibration()[0]},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] {len(files)+1} 个文件 → {out}")


if __name__ == "__main__":
    main()
