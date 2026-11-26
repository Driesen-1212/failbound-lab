# REFREEZE.md · 证据包重冻结记录

**v1** 2026-11-17（初版冻结）→ **v2** 2026-11-24（口径修订）

本文件是变更日志，**不参与** `SHA256SUMS.txt` 的自哈希（否则会造成自指）。它存在
的唯一目的，是让任何审计者能独立回答："改了什么、为什么改、数据动没动"。

---

## 1. 改了什么

只有 4 个文件变；其余 32 个字节未动，**26 个数据文件全部未动，没有任何实验被重跑**。

| 文件 | v1 sha256（前 16 位） | v2 sha256（前 16 位） | 字节 |
|---|---|---|---|
| `EVIDENCE_GUIDE.md` | `ddd5eee9bd609211` | 见 `SHA256SUMS.txt` | 11,155 → 13,412 |
| `README.md`（GUIDE 的副本） | `ddd5eee9bd609211` | 见 `SHA256SUMS.txt` | 11,155 → 13,412 |
| `reconciliation/replay_card_on_holdout.py` | `7a216a25fc6bda07` | `1592807185e42e3a` | 4,217 → 7,005 |
| `reconciliation/replay_card_on_holdout.csv` | `f473e4f6d2bf3b55` | `be8b951b2de0d0c5` | 984 → 1,321 |

根哈希（`SHA256SUMS.txt` 本身的 sha256）：

```
v1  a10684d573a75e849032b7203aea5698e1c9542a14676be07a437c4bee2aa8d6
v2  34691388b7ddefcd8f77a0382d4990cd7dde61dc62f556a53ffb40e020864fbd
```

本表的 v2 列只锚定 v1 旧值用于追溯；**v2 的权威值一律以 `checksums/SHA256SUMS.txt`
为准**（该清单同时是 `MANIFEST.root_hash` 的输入），变更日志不重复硬编码，避免两处漂移。

清单 `checksums/SHA256SUMS.txt` 存为 **LF**（v1 也是 LF）。v2 首次写入时误转为
CRLF，导致 `sha256sum -c` 报满屏 FAILED；已改回 LF 并复验 36/36 通过。数据文件
（`.jsonl` / `.csv`）本身仍是 CRLF，这与哈希无关——哈希算的是字节。

## 2. 为什么改

v1 的 `replay_card_on_holdout.py` 用了一个**与交付卡口径不同**的比较基准：

- 它取 `max(gp_ucb, gp_ei)` 对 `max(farthest_point, random_uniform)`，即"每个任务都
  挑最好的那一支"。这是**逐任务 oracle**，等于把卡没用过的采集函数也算进卡的成绩。
- 交付卡（`agent_lab/results/guideline_card.md`）名明的只有两支：主动学习 = `gp_ucb`，
  强基线 = `farthest_point`。技术报告和路演写的 −8.60 是这个口径。
- 但 v1 的列名是 `bo_best_mean_n_hits` / `space_filling_best_mean_n_hits`，读起来像
  卡的口径。结果：**csv 跑出 −13.13，报告和 PPT 写 −8.60，同一份证据包两个数。**
  评委会先跑脚本、再翻 PPT，这一步就会被抓住。

差异集中在 `log_kvrh__pca2`：`gp_ucb` = 92.967，`gp_ei` = 101.500，max 取了后者。
另外 `log_kvrh__elem20` 在 `random_uniform`（40.667）下高于 `farthest_point`（22.033），
导致 v1 把这张卡在放行侧记成 7/8，而严格口径下是 **8/8**。

## 3. 改成了什么

v2 的 csv 同时给出两套基准，列名不再含混：

| 列 | 含义 |
|---|---|
| `bo_gpucb` / `sf_farthest` / `gap_strict` / `ok_strict` | **契约口径**，卡名明的那两支策略。头行数就是这一套 |
| `bo_bestmax` / `sf_bestmax` / `gap_max` / `ok_max` | **敏感性口径**，逐任务取最优，仅作对照 |

重放结果（14 个留出任务，预算 400，gate-off）：

| 口径 | 净价值 | 拒绝 6 个 | 放行 8 个 |
|---|---|---|---|
| 契约（`gp_ucb` vs `farthest_point`） | **−8.60** | 真拒 1、误杀 5 | BO 赢 8/8 |
| 敏感性（best-of-family） | −13.13 | 真拒 1、误杀 5 | BO 赢 7/8 |

**引用时用契约口径的 −8.60**，它和报告 §4.4.5、路演 P9 是同一个数。敏感性口径只用来
说明"即便给卡一支它没跑的采集函数，净值仍然是负的"。

## 4. 怎么自证只改了这些

```
cd evidence
sha256sum -c checksums/SHA256SUMS.txt          # 期望 36/36 OK
python reconciliation/reconcile_runs.py        # ALL CHECKS PASS，17,520 条逐条自洽
python reconciliation/replay_card_on_holdout.py   # 打印 HEADLINE = -8.60
git log --stat -- evidence/                    # 若在版本库内，直接看变更文件列表
```

> 注意：`checksums/SHA256SUMS.txt` 本身存为 **LF**，这是刻意的。若被编辑器另存为
> CRLF，`sha256sum -c` 会把行尾 `\r` 当成文件名的一部分，报 36 行全 FAILED —— 那是
> 换行符问题，不是文件损坏。v2 修过一次这个：第一次写入时把清单转成了 CRLF，
> 已改回 LF 并复验 36/36 通过。

`reconcile_runs.py` 不依赖任何被改动的文件（它读 `main_8640/`、`ext11_7200/`、
`holdout_1680/` 和 `evidence_4320runs.csv`），所以 **v2 之后它仍然全部通过**，
这本身构成"数据层未变"的独立证据。

## 5. 影响到的对外材料

| 材料 | 用的数 | 是否需改 |
|---|---|---|
| 技术报告 §4.4.5 | −8.60 | 否，本来就是契约口径 |
| 路演 P9 / P15 | −8.60 | 否 |
| 答辩手册（`_defense_site`） | −8.60 | 否 |
| 证据包 csv（v1） | −13.13 | **已改为双列，头行 −8.60** |

也就是说：**结论没变，改的是证据包里那个跑偏的明细表。**

---

## 2. v2 同日补充声明 · β 扫描（2026-11-24）

技术报告正文两处提到 β 扫描（样本外卫生清单、复现路线图）。其逐 run 台账
`beta_runs.csv`（1,080 run，2026-10-22 产出）**不在本包 36 条锚定范围内，也从未
在本包的收录范围内**——它是探索性分析，非预注册终点，不进入报告任何结论的数字链。

三点说在前面，防误读：

1. **本次补充不改任何锚定。**MANIFEST.json 与本文件依设计不自哈希（见 MANIFEST
   note 原文），故此声明不改变 36 条锚定清单与 root_hash（仍为
   `34691388b7ddefcd…`）。这不是"用不改哈希绕过审计"，而是锚定清单的语义边界
   本来就把元数据排除在外；两文件的完整性由 git 提交历史与文件时间戳背书。
2. **排除是范围理由，不是质量理由。**β 扫描未纳入，是因为它不承载任何报告结论，
   不是因为结果不利。`mismatch_vs_frozen=0` 只说明该台账与冻结协议的字段/口径
   一致，**不构成对其结论的背书，也不使其成为报告结论的证据**。
3. **台账与脚本（`run_beta.py`、`analyze_beta.py`）可应要求即时交付**（51KB）。

若未来 β 扫描升级为预注册终点，将按本文件第 1 节的完整流程 re-freeze 并另行记录。
