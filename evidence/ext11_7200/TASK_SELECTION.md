# Task selection: 60-run set → 48-task subset (ext11_7200)

The extension file `ext11_7200/runs_nogate_ablate.jsonl` ships **all 60 tasks**
(10 base datasets × 6 representations × 4 strategies × 30 seeds = 7,200 runs).
Nothing was removed or filtered in this pack.

## Selection rule (quoted from the technical report, §4.4.2 / L461)

> 这一批是独立补跑的扩展消融臂(`ext11/`)：10 个基底 × 6 种表示 × 4 种策略 × 30 个种子 = 7200 条 run，预算 400、门控关闭；按命中口径剔除 2 个分类基底与 1 个退化基底（`matbench_glass`，其 top-10% 分位在 400 次采样里期望命中 282–285 次）后，剩 8 个有效基底 × 6 种表示 = 48 个任务。

English gloss: from the 10 base datasets, tasks are excluded on a hit-rate
feasibility criterion; `matbench_glass` is degenerate under this criterion
(its top-10% quantile would be hit ~282–285 times within a 400-experiment
budget, i.e. almost every draw is a "hit", so hit-based metrics carry no
signal). The report's 48-task figures use the remaining 8 bases × 6
representations.

**Caveat on the quoted sentence · 报告原句与数据文件的差异（主动写明）**: the
quoted sentence describes the campaign's planned exclusion of three bases
("2 个分类基底与 1 个退化基底"), i.e. an 11-base plan. The shipped run file
contains **10 bases**: `matbench_expt_is_metal` (one of the planned
classification exclusions) was never included in the extension campaign, so
only **two** exclusions take effect at file level (`matbench_glass`,
`matbench_steels`) and 10 − 2 = 8 valid bases. This is why
`_sig_worse.py` (repo `agent_lab/`, line 13) lists three names in
`EXCLUDE = {"matbench_glass", "matbench_expt_is_metal", "matbench_steels"}`
while the file yields 48 = 8 × 6: one of the three names matches nothing in
the data. Both readings produce the same 48-task subset; the discrepancy is
between the report's wording and the campaign files, not between any two
numbers.

## File-level facts (reproducible from the shipped data)

- The extension run file contains these 10 bases: `matbench_dielectric`,
  `matbench_expt_gap`, `matbench_glass`, `matbench_jdft2d`,
  `matbench_log_gvrh`, `matbench_log_kvrh`, `matbench_mp_e_form`,
  `matbench_perovskites`, `matbench_phonons`, `matbench_steels`
  (each × 6 representations: elem20, pca1, pca2, pca5, rp2, rp5).
- Therefore the 48-task subset is obtained by removing **12 tasks = 2 bases × 6
  representations**: `matbench_glass` (degenerate hit-rate) and
  `matbench_steels` (hit-criterion inapplicable). The full list of the 48 kept
  tasks is in `ext11_7200/tasks_48.txt` (48 lines).
- Expected subset count: 48 × 4 strategies × 30 seeds = **5,760 runs**.

## Reproduce the subset count

```
cd reconciliation
python reconcile_runs.py --task-filter ../ext11_7200/tasks_48.txt
```

Expected output line: `48-task subset = 5760 / 5760 PASS`.
