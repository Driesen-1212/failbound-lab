# -*- coding: utf-8 -*-
"""ask.py —— 现场演示入口：输入一个任务，输出「该不该用 BO」以及这个判断的实测对错。

设计约束（为什么这样写）：
1. **不联网、不读大文件**：只读 agent_lab/results/ 下的 tests.csv / task_props.csv
   （共约 300 KB），现场断网也能跑，2 秒内出结果。
2. **输出里必须带"实测"**：光说"我判断该拒绝"没有说服力；必须当场给出
   "如果真跑 400 步，BO 拿多少命中、空间填充拿多少命中"，让判断立刻可被判真假。
3. **判据只读描述符维度**，不读目标值——这是卡的诚实性所在，代码里必须看得见。

用法：
    python ask.py --task matbench_dielectric__pca1
    python ask.py --list                 # 列出全部 36 个任务-臂
    python ask.py --task X --budget 200  # 换预算档，看判断是否稳定
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FAILBOUND_RESULTS") or os.path.join(HERE, "agent_lab", "results")
TESTS = os.path.join(DATA, "tests.csv")
PROPS = os.path.join(DATA, "task_props.csv")

DIM_THRESHOLD = 2      # 卡的唯一判据：描述符维度 ≤ 2 就拒绝优化
MAIN_STRATEGY = "gp_ucb"
BASE_STRATEGY = "farthest_point"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_props():
    if not os.path.exists(PROPS):
        sys.exit("找不到 %s" % PROPS)
    return {r["task"]: r for r in csv.DictReader(open(PROPS, encoding="utf-8"))}


def load_rows(budget):
    if not os.path.exists(TESTS):
        sys.exit("找不到 %s" % TESTS)
    out = {}
    for r in csv.DictReader(open(TESTS, encoding="utf-8")):
        if r["budget"] != str(budget) or r["use_gate"] != "False":
            continue
        if r["strategy"] != MAIN_STRATEGY:
            continue
        out[r["task"]] = r
    return out


def decide(dim):
    """卡的全部逻辑。三行，没有隐藏分支。"""
    if dim <= DIM_THRESHOLD:
        return False, "描述符维度 %d ≤ %d：表示不足，代理模型在拟合噪声" % (dim, DIM_THRESHOLD)
    return True, "描述符维度 %d > %d：表示足够，按 BO 执行" % (dim, DIM_THRESHOLD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--budget", type=int, default=400, choices=[25, 50, 100, 200, 400])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    props = load_props()
    rows = load_rows(a.budget)

    if a.list:
        print("共 %d 个任务配置（budget=%d）：" % (len(rows), a.budget))
        for t in sorted(rows, key=lambda x: (int(f(props[x]["dim"])), x)):
            dim = int(f(props[t]["dim"]))
            print("  dim=%-3d %s" % (dim, t))
        return

    if not a.task:
        sys.exit("用 --task <任务名> 或 --list")

    if a.task not in rows:
        print("没有这个任务的实测数据：%s" % a.task)
        near = [t for t in rows if a.task.split("__")[0] in t][:8]
        if near:
            print("相近的有：")
            for t in near:
                print("  ", t)
        sys.exit(2)

    dim = int(f(props[a.task]["dim"]))
    use_bo, why = decide(dim)
    r = rows[a.task]
    bo, base = f(r["mean_bo"]), f(r["mean_base"])
    diff = bo - base
    p = f(r.get("p_holm"))
    sig = str(r.get("significant")).strip().lower() == "true"

    print("")
    print("任务      %s" % a.task)
    print("描述符    %d 维        （判据只看这一个数，不看目标值）" % dim)
    print("判断      %s" % ("USE_BO" if use_bo else "拒绝优化 → 改用空间填充"))
    print("理由      %s" % why)
    print("")
    print("实测      budget=%d，30 个种子平均命中数（命中 = 目标 top 10%%）" % a.budget)
    print("          BO(gp_ucb)      %7.2f" % bo)
    print("          空间填充        %7.2f" % base)
    print("          差(BO−空间填充) %+7.2f   Holm 校正后 p=%.4f（%s）"
          % (diff, p, "显著" if sig else "不显著"))
    print("")
    if abs(diff) < 1e-9:
        verdict = "持平"
    elif use_bo:
        if diff > 0:
            verdict = "放行是对的（BO 更好）"
        else:
            verdict = ("放行错了：漏报（BO 显著更差）" if sig
                       else "放行有偏但未达显著（BO 更差，p 不显著）")
    else:
        if diff < 0:
            verdict = "拒绝是对的（BO 显著更差）" if sig else "拒绝方向对（BO 更差，但未达显著）"
        else:
            verdict = ("拒错了：误杀（BO 显著更好）" if sig
                       else "拒错但为噪声（BO 略好，未达显著）")
    print("裁决      %s" % verdict)
    print("")


if __name__ == "__main__":
    main()
