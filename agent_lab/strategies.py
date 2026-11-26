# -*- coding: utf-8 -*-
"""策略集：2 个基线 + 2 个主动学习（对应 v4 修正 #4：强基线，不刷分）

- random_uniform   随机参照（用于验证命中口径本身：理论上必须恒为 10%）
- **farthest_point**  强基线：最远点（max-min）空间填充 —— 见下方 ⚠️
- gp_ucb / gp_ei    主动学习：高斯过程 + 采集函数

⚠️ 2026-10-16 重大修正：废弃原 `lhs_stratified` 强基线
------------------------------------------------------
原实现 `pick_lhs_stratified` 以 n=1 逐步调用时**根本不是拉丁超立方采样**：
    perms = rng.permutation(1) 恒等于 [0]  →  pts = (0 + U)/1 = U
即每步只是"在标准化包围盒里撒一个均匀随机点，再吸附到最近邻候选"。
LHS 的分层结构在 n=1 时完全消失。

后果是灾难性的：实测 36 个任务上基线自身的命中率在 **0.022 ~ 0.326** 之间摆动
（随机应为 0.100，实测 random_uniform 恒为 0.100，说明命中口径没错，是基线不稳定）。
而 `rel_gain` 以基线命中数为分母 —— 于是测出的"BO 增益"实际在测"这个基线的运气"。
主臂"BO 处处获胜"和消融臂"边界在 AUC=0.551"两个结论**都是这个混淆的产物**。

替代方案选最远点（max-min）而不是"修正版 LHS"：
  1. 它直接挑真实的候选点，不需要"查询点 → 吸附最近邻"这步（吸附正是偏 bias 的来源）；
  2. 在 85 维里 LHS 的边际分层几乎无效（400 个样本 / 85 维），max-min 才是真正有效的空间填充；
  3. 无超参、确定性强、跨任务稳定。
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd


def pick_initial(n_pool, n_init, rng):
    return rng.choice(n_pool, size=min(n_init, n_pool), replace=False)


def pick_random_uniform(avail_idx, n, rng):
    n = min(n, len(avail_idx))
    return rng.choice(avail_idx, size=n, replace=False)


class FarthestPointSelector:
    """最远点（max-min）空间填充选择器 —— 新的强基线。

    每步挑"离所有已选点最远"的候选。这是标准的确定性空间填充设计：
      - 不需要"查询点 → 吸附最近邻"，因此没有吸附偏差；
      - 在高维里仍然有效（LHS 的边际分层在 85 维、400 样本下几乎无效）；
      - 无超参，跨任务稳定（实测命中率不再出现 15 倍的摆动）。

    实现要点：维护一个长度 = 池大小的 `dmin`（每个候选到已选集合的最小距离），
    每步只更新一次 O(N·d)，避免每步重算 3000×400 的距离矩阵（那样每 run 要 21 秒）。
    """

    def __init__(self, X_std):
        self.X = X_std
        self.dmin = None

    def update(self, i):
        d = np.sum((self.X - self.X[i]) ** 2, axis=1)
        self.dmin = d if self.dmin is None else np.minimum(self.dmin, d)

    def select(self, avail_idx, n):
        n = min(n, len(avail_idx))
        idx = np.asarray(avail_idx, dtype=int)
        if self.dmin is None:
            return idx[:n]                       # 第一步：任意（外层已随机选了初始点）
        sub = self.dmin[idx]
        order = np.argsort(-sub)
        return idx[order[:n]]


def pick_lhs_stratified(X_std, avail_idx, n, rng):
    """⚠️ 已废弃（保留仅为对照）。以 n=1 调用时不是 LHS，详见模块 docstring。

    在标准化特征空间做拉丁超立方采样，再取最近邻候选（分层覆盖）"""
    n = min(n, len(avail_idx))
    d = X_std.shape[1]
    lo, hi = X_std.min(axis=0), X_std.max(axis=0)
    perms = np.array([rng.permutation(n) for _ in range(d)]).T   # (n, d)
    pts = (perms + rng.random((n, d))) / n
    q = lo + pts * (hi - lo)

    chosen, used = [], set()
    avail_set = set(avail_idx.tolist())
    for p in q:
        dist = np.linalg.norm(X_std - p, axis=1)
        order = np.argsort(dist)
        for o in order:
            if int(o) in avail_set and int(o) not in used:
                chosen.append(int(o))
                used.add(int(o))
                break
    return np.array(chosen, dtype=int)


class GPSelector:
    """带状态的高斯过程选择器。

    为什么需要它：真实描述符 88 维 + 预算 400 时，若每次都重新优化核超参数
    （sklearn 默认会做多重启 L-BFGS，每次都要重算边际似然 O(n³)），
    单次 run 会慢到 30s 以上，2400 个 run 无法在期限内跑完。

    三项优化（均为 BO 工程常规做法，不改动统计模型本身）：
      1. **热启动核参数**：复用上一次拟合得到的 kernel，n_restarts_optimizer=0
      2. **周期性重拟合**：每 refit_every 步才优化一次超参数，其余步固定参数（仅做 Cholesky）
      3. **候选子集**：每步只在随机 cand_subset 个候选上评估采集函数，而非全池 3000
    """

    def __init__(self, kappa=2.0, refit_every=20, cand_subset=1000):
        self.kappa = kappa
        self.refit_every = refit_every
        self.cand_subset = cand_subset
        self.kernel = None
        self.step = 0

    def select(self, X_std, X_obs, u_obs, avail_idx, n, rng, acq="ucb"):
        self.step += 1
        n = min(n, len(avail_idx))

        if len(avail_idx) > self.cand_subset:
            sub = rng.choice(avail_idx, size=self.cand_subset, replace=False)
        else:
            sub = avail_idx

        refit = (self.kernel is None) or (self.step % self.refit_every == 0)
        if refit:
            base = self.kernel or (ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5))
            gp = GaussianProcessRegressor(kernel=base, alpha=1e-6,
                                          normalize_y=True, n_restarts_optimizer=0,
                                          random_state=0)
        else:
            gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6,
                                          normalize_y=True, optimizer=None,
                                          random_state=0)
        gp.fit(X_obs, u_obs)
        self.kernel = gp.kernel_

        mu, sigma = gp.predict(X_std[sub], return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        if acq == "ucb":
            score = mu + self.kappa * sigma
        else:
            best = float(np.max(u_obs))
            z = (mu - best) / sigma
            score = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)

        order = np.argsort(-score)
        return np.asarray(sub, dtype=int)[order[:n]]


def _fit_gp(X_obs, u_obs):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                  normalize_y=True, random_state=0)
    gp.fit(X_obs, u_obs)
    return gp


def pick_gp(X_std, X_obs, u_obs, avail_idx, n, rng, acq="ucb", kappa=2.0):
    """一次选 n 个候选。效用 u 越大越好（见 cost_model.utility）"""
    n = min(n, len(avail_idx))
    gp = _fit_gp(X_obs, u_obs)
    mu, sigma = gp.predict(X_std[avail_idx], return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    if acq == "ucb":
        score = mu + kappa * sigma
    else:  # EI
        best = float(np.max(u_obs))
        z = (mu - best) / sigma
        score = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)

    order = np.argsort(-score)
    return avail_idx[order[:n]]
