# FailBound-Lab

> 同一批材料任务，换一种描述符表示，贝叶斯优化的收益会换向。换向不是处处发生，它只落在部分体系上。我们用 17,520 条轨迹把这条边界测出来，做成一张开跑前就能用的判决卡，并把这张卡自己的错误率一起公开。

**它在做什么**：任务进来，先看描述符维度。大于 2，放行进满预算；小于等于 2，拒绝，改走均匀铺点或砍到六成预算。判据只读开工前拿得到的信息，不读目标值，所以判决发生在第一枪之前。

## 三层数据，加起来 17,520 条

| 层 | 轨迹 | 用途 |
|---|---|---|
| `evidence/main_8640/` | 8,640 | 主集：36 个任务臂（主臂 18 + 消融臂 18），四个条件各 2,160 |
| `evidence/ext11_7200/` | 7,200 | 扩展集：11 个任务，检验结论是否只在主集成立 |
| `evidence/holdout_1680/` | 1,680 | 严格留出：14 个任务，卡从未在这批上标定 |

所有数字都按预算 400、采集函数 `gp_ucb`、强基线 `farthest_point` 记。换采集函数或换预算，数字要重新算。

## 三个数字，先看它们的分母

| 数字 | 含义 | 分母 |
|---|---|---|
| **+26.17** | 主集上，卡相对「全部交给优化器」的净命中差（省下 98.53，让掉 72.37） | 36 个任务臂 |
| **误杀 5 / 真拒 4** | 卡拒掉的 9 个里，5 个拒错，4 个躲过了最差 −33.9 的灾难 | 拒绝的 9 个 |
| **漏报 0 / 27** | 卡放行的 27 个里，没有一个「优化器显著更差」；95% 上界 12.8% | 放行的 27 个 |

`+26.17` 有前提，写在数字旁边：它来自参与标定的那批基底。换 3 个没标定过的新体系，净增益 +0.90；走到严格留出集，−8.60。所以它只证明机制存在过，不是普适成绩。

## 一条命令复算

```bash
python verify.py
```

从 `agent_lab/results/` 的实测结果实时重算上面这些数，和报告逐位比对，不一致就输出 FAIL。零第三方依赖。

```bash
python ask.py --task matbench_log_gvrh__pca2     # 拒错的例子：卡当场承认"拒错了：误杀"
python ask.py --task matbench_dielectric__pca2   # 拒对的例子
python ask.py --list                             # 列出全部任务
```

## 证据包自证

```bash
cd evidence
sha256sum -c checksums/SHA256SUMS.txt            # 36/36 OK
python reconciliation/reconcile_runs.py          # ALL CHECKS PASS，17,520 条逐条自洽
python reconciliation/replay_card_on_holdout.py  # HEADLINE = -8.60
```

## 文件导航

| 位置 | 内容 |
|---|---|
| `index.html` | 项目主页 |
| `tech-report.html` / `.pdf` | 技术报告，HTML 与 PDF 同源 |
| `project-overview-5p.pdf` | 五页概览 |
| `agent_lab/` | 实验代码：bench / strategies / analyze / audit 等 |
| `agent_lab/results/` | 逐 run 轨迹与聚合：`tests.csv`、`task_props.csv`、`runs_gate*.jsonl`、`runs_nogate*.jsonl`、`summary.csv`、`decision_card_audit.csv` |
| `evidence/` | 冻证据包：39 个文件、17,520 条轨迹、36 条 sha256 锚定、重冻结记录 |
| `evidence_4320runs.csv` | 30-run 均值摘要，**不能相减复现配对增益**（口径见报告 §4.4.7） |
| `failbound-console.html` | 复算台：18 个任务臂的判决与账目 |
| `replay/` | 四个场景的决策回放 |
| `ask.py` / `verify.py` | 上面几条命令 |

## 复现环境

Python 3.8+，`verify.py` 与 `ask.py` 只用标准库。全量重跑见 `agent_lab/run_experiments.py`（需材料数据集，配置在 `agent_lab/config.py`）。

## 边界

结论是回溯性的：卡在冻结数据上标定，前瞻验证还没做。四条不适用情形写在技术报告与主页上——任务与这 36 个臂不同源、预算远超标定区间、指标换成与命中数无关的口径、任务之间强耦合。

原创性声明见 `originality-statement.pdf`。
