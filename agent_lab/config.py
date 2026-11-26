# -*- coding: utf-8 -*-
"""FailBound-Lab 全局配置 v2 —— 适配真实 matbench 数据

相对 v1 的四处变更（对应预注册文档 preregistration.md）：
  1. 任务集由 3 个硬编码任务 → 从 data/scan + data/tasks 目录自动发现（主臂 18 个）
  2. 命中口径由固定阈值 → **top-10% 分位数**（消除基线命中率与可学习性的混淆）
  3. 预算 3 档 → 5 档（25/50/100/200/400），用于 H3 拐点拟合
  4. 种子 20 → 30
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data_root():
    """自动定位含 data/tasks 的工程根目录。

    为什么不能写死：本工程有两种等价布局 ——
        本地： <proj>/unzipped/agent_lab/  +  <proj>/data/
        沙箱： <root>/agent_lab/          +  <root>/data/
    写死 ".." 或 "../.." 会在另一种布局下静默失败（数据找不到 → 任务数为 0），
    所以这里逐级向上探测，取第一个真正含 data/tasks 的目录。
    """
    env = os.environ.get("FAILBOUND_PROJECT")
    cands = ([env] if env else []) + [
        os.path.normpath(os.path.join(_HERE, "..")),
        os.path.normpath(os.path.join(_HERE, "..", "..")),
        os.path.normpath(os.path.join(_HERE, "..", "..", "..")),
    ]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "data", "tasks")):
            return c
    return os.path.normpath(os.path.join(_HERE, ".."))


_PROJECT = _find_data_root()
DATA_ROOT = os.path.join(_PROJECT, "data")

SCAN_DIR = os.path.join(DATA_ROOT, "scan")      # 受控降质扫描轴（12 档 + 3 档抽查）
TASKS_DIR = os.path.join(DATA_ROOT, "tasks")    # 真实任务锚点（不降质）
ABLATE_DIR = os.path.join(DATA_ROOT, "ablate")  # 描述符消融臂（真实目标 + PCA 降维）

# ---------- 复现性 ----------
SEED = 20261015
N_SEEDS = 30

# ---------- 预算（成本模型口径：可合成/可表征的候选数）----------
BUDGETS = (25, 50, 100, 200, 400)
N_INIT = 5

# ---------- 命中口径 ----------
HIT_Q = 0.90            # 命中 = 目标 top 10%，所有任务基线命中率恒为 10%

# ---------- 路径 ----------
OUT_DIR = os.path.join(_HERE, "results")

# ---------- 策略集：2 基线 + 2 主动学习 ----------
# ⚠️ 2026-10-16：强基线由 lhs_stratified 改为 farthest_point（最远点空间填充）。
#    原因：lhs_stratified 以 n=1 逐步调用时退化成"均匀随机 + 最近邻吸附"，
#    实测基线命中率跨任务摆动 15 倍（0.022–0.326），使相对增益失去意义。
#    详见 strategies.py 模块 docstring 与 preregistration.md §10。
STRATEGIES = ("random_uniform", "farthest_point", "gp_ucb", "gp_ei")
BASELINE = "farthest_point"     # 强基线（稳定的空间填充设计）
BO = "gp_ucb"                   # 主主动学习策略
GP_KAPPA = 2.0

# ---------- 合成可行性门控（成本模型）----------
RARE_ELEMENTS = ("Pt", "Ir", "Rh", "Pd", "Au", "Ru", "Re", "Os")
MAX_ATOMS = 40           # 晶胞原子数软上限，超过则成本加倍
HARD_EXCLUDE_ATOMS = 80  # 硬门控：超过则排除出候选池
RARE_COST_FACTOR = 3.0

_SKIP = {"tasks_manifest.csv", "task_probe.csv", "scan_manifest.csv",
         "ablate_manifest.csv"}

# ---- 预注册登记的任务集（主臂共 18 个，见 preregistration.md §3.2）----
# 主扫描轴：基底 log_kvrh × 12 档 λ
_SCAN_LAMBDAS = ("0.02", "0.05", "0.08", "0.12", "0.18", "0.25",
                 "0.35", "0.45", "0.60", "0.75", "0.88", "1.00")
# 跨基底抽查：基底 perovskites × 3 档
_SPOT_LAMBDAS = ("0.25", "0.60", "1.00")
# 2026-10-17 新增：给 3 个真实锚点基底各补 3 档降质扫描（build_scan.py 的 EXTRA_SCAN）
# 目的：把"真正被扫过"的独立基底从 1 个（log_kvrh）提到 5 个，修复有效样本量被高估的问题。
# 不含 λ=1.00：它与同名锚点完全等价（standardize 不改变 top-10% 命中集合）。
_ANCHOR_SCAN_BASES = ("matbench_expt_gap", "matbench_dielectric", "matbench_log_gvrh")
_ANCHOR_SCAN_LAMBDAS = ("0.05", "0.25", "0.60")
# 真实任务锚点（不降质）：排除 glass / expt_is_metal（分类任务，基线命中率与可学习性混淆）
# ⚠️ 已剔除 matbench_log_kvrh 与 matbench_perovskites：
#    实测它们与扫描轴的 __lam1.00 档 Spearman=1.0000、命中集合重合度 100%，
#    是完全等价的重复任务（保留会被读作"凑数"）。扫描轴的 lam1.00 即为其无降质版本。
_REAL_ANCHORS = ("matbench_expt_gap", "matbench_dielectric", "matbench_log_gvrh")

# ---- 描述符消融臂（2026-10-16 新增的稳健性臂，登记于 preregistration.md §10）----
# 动机：主扫描轴靠"给真实目标加噪"制造低可学习性任务，边界的存在性因此完全依赖合成数据。
# 消融臂走另一条独立的路：**目标值一个字不动（100% 真实）**，只削减描述符空间。
# 三族削减算子（spectral / random / subset）× 3 个真实基底 = 18 个任务，
# 实测 AUC 覆盖 0.55–0.92（单靠 PCA 只能覆盖 0.61–0.88 且低端只有 1 个点）。
ABLATE_BASES = _REAL_ANCHORS
ABLATE_OPS = ("pca1", "rp2", "rp5", "pca2", "elem20", "pca5")


def _in_registry(name):
    """只跑预注册里登记过的任务，防止事后挑任务（等同 p-hacking）。"""
    if name.startswith("matbench_log_kvrh__lam"):
        return name.split("__lam")[1] in _SCAN_LAMBDAS
    if name.startswith("matbench_perovskites__lam"):
        return name.split("__lam")[1] in _SPOT_LAMBDAS
    for b in _ANCHOR_SCAN_BASES:
        if name.startswith(b + "__lam"):
            return name.split("__lam")[1] in _ANCHOR_SCAN_LAMBDAS
    return name in _REAL_ANCHORS


def _is_anchor_scan(name):
    """是否为「锚点补扫」任务（3 个真实基底 × 3 档 λ），这类任务尚未执行。"""
    return any(name.startswith(b + "__lam") for b in _ANCHOR_SCAN_BASES)


def _in_ablate_registry(name):
    if "__" not in name:
        return False
    base, op = name.split("__", 1)
    return base in ABLATE_BASES and op in ABLATE_OPS


def discover_tasks(arm="main", include_extra_scan=False):
    """扫描数据目录，每个 CSV 是一个任务。

    arm="main"   → 预注册主实验的 **18 个任务**（12 档扫描轴 + 3 档跨基底 + 3 个真实锚点）
                   —— 报告中**全部结论**都建立在这 18 个上，也是复现清单应当跑到的集合。
    arm="ablate" → 描述符消融臂的 18 个任务（3 基底 × 6 种削减算子，目标 100% 真实）

    ⚠️ include_extra_scan=True（仅 main 臂）会额外纳入 9 个**锚点补扫任务**
       （3 个真实基底 × λ∈{0.05, 0.25, 0.60}），使主臂变 27 个。
       🔴 为什么默认关掉：这 9 个任务**尚未执行**（补跑任务在 10/17 被中断，
       且已决定不再重启）。若默认纳入，评审按复现清单跑出来会是 27 个任务，
       而我们报告的所有统计都基于 18 个 —— **复现结果与报告对不上**，
       这是比"少跑几个任务"严重得多的问题。**跑完之前不要打开这个开关。**

    返回 {task_name: {"path":..., "target_col":"target", "mode":"topq", "q":HIT_Q}}
    """
    tasks = {}
    dirs = ((SCAN_DIR, TASKS_DIR) if arm == "main" else (ABLATE_DIR,))
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".csv") or fn in _SKIP:
                continue
            name = fn[:-4]
            if arm == "main":
                if not _in_registry(name):
                    continue
                # 锚点补扫任务默认排除（原因见 discover_tasks 的 docstring）
                if not include_extra_scan and _is_anchor_scan(name):
                    continue
            else:
                if not _in_ablate_registry(name):
                    continue
            tasks[name] = {"path": os.path.join(d, fn),
                           "target_col": "target",
                           "mode": "topq",
                           "q": HIT_Q}
    return tasks


if __name__ == "__main__":
    for arm in ("main", "ablate"):
        t = discover_tasks(arm=arm)
        print(f"[discover:{arm}] {len(t)} 个任务")
        for k in list(t)[:5]:
            print("  ", k)
