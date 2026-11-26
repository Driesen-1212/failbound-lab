# Evidence Pack Guide · 证据包指南

**FailBound-Lab — frozen run-level evidence bundle (frozen 2026-11-17, revised 2026-11-24).**
No experiment was re-run for this bundle; every data file is a byte-identical
copy of the original campaign outputs, and every count below is a
re-enumeration of those files. The 2026-11-24 revision changed **four files**:
the replay script and its csv (the substantive fix — the card's contract basis
is now the headline number), plus this guide and its README copy, restated to
match. **All 26 data files are unchanged and no experiment was re-run.** See
`REFREEZE.md` for the old→new root hash.

Integrity scope, stated up front: `MANIFEST.json` and
`checksums/SHA256SUMS.txt` cover all 36 hashed files (the 26 data files below, this guide and its README copy, `EXCLUDED.md`/`TASK_SELECTION.md`/`tasks_48.txt`, and the frozen seed-42 reconciliation reference outputs) and detect
**transfer corruption**. They do not, by themselves, prove tamper-resistance
(the manifests are not self-hashed); the root hash of `SHA256SUMS.txt` is
recorded inside `MANIFEST.json` (`root_hash`) for cross-anchoring against the
submission record. `REFREEZE.md` documents the 2026-11-24 revision and is not
itself hashed. The `reconciliation/` outputs are **regenerated on every
run** of the script (they are run products, not frozen evidence); a frozen
reference run with seed 42 is included in the hash list.

## 1. What is here · 三层证据

| Layer | Directory | Files | Runs | Design |
|---|---|---|---|---|
| Main benchmark | `main_8640/` | 4 jsonl | **8,640** | 36 task-arms × 4 strategies × 30 seeds × 2 evaluation settings: 18 main-arm tasks (bare dataset names + lam-operator tasks) and 18 ablation-arm tasks (3 bases × 6 representations) are **different task sets by design**; gate/nogate are the two settings, 2,160 runs per file |
| Task-extension set | `ext11_7200/` | 1 jsonl + 7 aux shards + `TASK_SELECTION.md` + `tasks_48.txt` | **7,200** (+720 aux) | 60 tasks (10 bases × 6 representations) × 4 strategies × 30 seeds, budget 400, gate-off. The report's 48-task figures use the 8 valid bases = 48 tasks = 5,760-run subset — selection rule and reproducible task list in `TASK_SELECTION.md`. Aux shards: `shufgap*` = expt_gap permutation-robustness runs (720 run records); `shufprobe` = a wiring probe kept for lineage (24 records, **not** an experimental result, excluded from all statistics) |
| Held-out set | `holdout_1680/` | 1 jsonl + 4 data files + 4 scripts + 2 logs | **1,680** | 2 bases × 7 representations × 4 strategies × 30 seeds, budget 400, **gate-off only** (the gate arm was not run on the holdout — see §5). Ships with `holdout_results.json`, `ablate_manifest.csv`, `holdout_props.csv` (includes the per-task `dim` used by the card), build/analysis scripts and run logs |

Files that exist in the campaign directories but were intentionally **not**
shipped are listed with reasons in `EXCLUDED.md` (backups / aborted shards /
debug stubs).

Each run record carries the full audit trail:
`task, strategy, seed, use_gate, chosen (full selection-index sequence),
y_seq, hits (per-experiment 0/1 sequence), cost_seq, n_hits, ttfh, n_exp, spent`.

## 2. Verify the counts in one command · 一条命令对账

```
cd evidence/reconciliation
python reconcile_runs.py          # standard library only, no installation
```

Expected: exit code 0 and `ALL CHECKS PASS`. The script re-enumerates the
**six design files** (runs per strategy, distinct tasks, seeds 0–29 per
group, budget compliance, `use_gate` consistency, index uniqueness, ttfh
range), checks the grand totals 8,640 / 7,200 / 1,680, validates cross-file
design facts (gate/nogate share identical task sets within each arm; ext11
contains the full main-ablate task-name set; holdout does not collide with
the main arm), validates internal field consistency for **all 17,520 runs**
(`len(chosen) == len(y_seq) == n_exp`, `n_hits == sum(hits)`,
`spent == sum(cost_seq)`), enumerates the **8 auxiliary shards** (counts
reported, no design assertions), and cross-checks **all 144 rows** of the
aggregate table `evidence_4320runs.csv` against recomputed group means from
the ablation-arm JSONL runs (numeric equality within 5e-4 — the csv stores
3-decimal means), including each row's `n_runs` field. The csv itself is
asserted complete (144 rows, all mappable). `--sample-n N` restricts the
crosscheck to a seeded sample of N rows if a quick pass is wanted; the
default verifies everything.

The aggregate csv is read from the evidence root (a legacy copy at the repo
root also works). Keep `evidence/` and `evidence_4320runs.csv` together when
copying.

Outputs written to `reconciliation/` (regenerated each run):
`reconciliation_report.csv`, `crosscheck_samples.csv`,
`base_system_overlap.csv`, `summary_by_task_strategy.csv` (584 groups over
all three layers), `RECONCILE_RESULT.txt`. A frozen seed-42 reference run of
each is in the hash list.

## 3. Spot-check any single run · 抽查任意一条轨迹

```
python -c "import json; lines=open('../main_8640/runs_gate.jsonl',encoding='utf-8').readlines(); r=json.loads(lines[1]); assert 'seed' in r; print(json.dumps(r, indent=1)[:1500])"
```

Or open any `.jsonl` in a text editor — one JSON object per line. Meta/summary
lines carry no `task`/`strategy`/`seed` fields; the main files have exactly
19 such lines each (2,179 = 2,160 + 19; the reconcile script asserts this).
File integrity (run from the `evidence/` directory — the checksum list uses
paths relative to it):

```
sha256sum -c checksums/SHA256SUMS.txt
```

Note: this list is stored **LF on purpose**. If an editor re-saves it as
CRLF, `sha256sum -c` reads the trailing `\r` as part of the filename and
reports **every line as FAILED** — a line-ending artifact, not corruption.
Keep it LF. Portable equivalent (works either way, standard library):

```
python -c "import hashlib;r=[l.rstrip().split('  ',1) for l in open('checksums/SHA256SUMS.txt') if l.strip()];b=[p for h,p in r if hashlib.sha256(open(p,'rb').read()).hexdigest()!=h];print(f'{len(r)-len(b)}/{len(r)} OK',b or '')"
```

Expected: `36/36 OK`.

## 4. Mapping to report numbers · 与报告口径对照

- "36 task-arms × 4 strategies × 30 seeds × 2 evaluation settings" = the four
  `main_8640/` files: `runs_gate` / `runs_nogate` are the **main arm** (18
  tasks: 3 bare dataset names + 12 `log_kvrh__lam*` + 3 `perovskites__lam*`);
  `runs_gate_ablate` / `runs_nogate_ablate` are the **ablation arm** (18
  tasks: `dielectric/expt_gap/log_gvrh` × 6 representations). 4 × 2,160 =
  8,640. All four ran at budget 200.
- `evidence_4320runs.csv` (144 rows) = the **ablation-arm** aggregate: 18
  task-arms × 4 strategies × 2 gate settings. Row key
  `(dataset, representation, strategy, gate)` maps to the JSONL runs as
  task = `{dataset}__{representation}`; `gate_on` ↔ `runs_gate_ablate.jsonl`
  (`use_gate: true`), `gate_off` ↔ `runs_nogate_ablate.jsonl`
  (`use_gate: false`). The reconcile script's crosscheck uses exactly this
  mapping (spot-verified: dielectric/elem20/farthest_point gives 28.200/5.900
  gate-off and 26.900/6.200 gate-on).
- "48-task out-of-base-sample replication" = subset of the 60 tasks in
  `ext11_7200/`: 48 × 4 × 30 = **5,760 runs** (the full 60-task file, 7,200
  runs, is shipped unfiltered). Selection rule, the report-quote caveat, and
  the reproducible task list are in `ext11_7200/TASK_SELECTION.md`;
  reproduce with
  `python reconcile_runs.py --task-filter ../ext11_7200/tasks_48.txt`
  (counts and 8×6 structure both asserted).
- "held-out test" = `holdout_1680/` (2 bases × 7 representations × 4
  strategies × 30 seeds, gate-off, budget 400) with its manifest, properties
  table and analysis scripts. What "held out" means here, and its limits:
  see §5.
- Independence across layers is **limited and visible**:
  `base_system_overlap.csv` enumerates, for every base system, how many
  tasks it contributes to each layer (10 bases total; 3 bases appear in
  main/ablate/ext11; the holdout re-uses 2 of the main bases with new
  representations).

## 5. Held-out evidence for the frozen card · 决策卡的留出证据（与边界）

Two things are shipped here; their limits are stated in the same breath.

**(a) The holdout runs themselves.** 14 tasks (2 bases × 7 representations)
that did not participate in boundary fitting or rule search — held out at
the **representation level**: they re-use two base systems already present in
the main benchmark (see `base_system_overlap.csv`). The holdout campaign ran
**gate-off only, budget 400** (the gate arm aborted; see `EXCLUDED.md`), so
**the holdout cannot test the gate arm's runtime behavior itself** — it can
only test the card's task-level rule offline.

**(b) Offline replay of the frozen card** (`reconciliation/replay_card_on_holdout.py`,
standard library):

```
python replay_card_on_holdout.py
```

The card's rule uses only task-level properties known before any experiment
(descriptor dimensionality: dim ≤ 2 → withhold BO), so it replays on the
gate-off runs. Output `replay_card_on_holdout.csv` (one row per task, with
`dim`, card decision, best-BO vs best-space-filling mean n_hits, and whether
the outcome matches the decision). Reading the result honestly:

- The card withholds **6/14** holdout tasks (pca1/pca2/rp2 of both bases).
- On **5 of those 6**, BO would in fact have beaten space filling at budget
  400 — **false kills**, consistent with the card's stated conservative
  pricing (it prefers false kills over misses; the calibration set priced
  this at 5/9 false kills vs 0/27 significant misses, report §4.4.2).
  Two caveats: the holdout ran at **budget 400, double the card's
  calibration budget**, which favors BO recovery; and this replay compares
  raw means, not the report's normalized-gain significance methodology.
- **Net value on the holdout is negative: −8.60 hits.** Contract basis
  (`gp_ucb` vs `farthest_point` — the two strategies the card actually
  names): the card avoids 27.73 hits on its one correct rejection and
  gives up 36.33 across the five false kills, net −8.60. On the 8 tasks
  where the card lets BO run, BO beat the baseline on **8/8**. The same
  ledger is **+26.15 in-sample** (calibration set, 36 task-arms); the sign
  flips out of sample, which is the same non-transfer result as §3.5 of
  the report.
- **Two reporting bases, not interchangeable.** The csv carries both:
  `gap_strict` (contract; headline **−8.60**) and `gap_max` (sensitivity:
  best-of-BO vs best-of-space-filling — an oracle-per-task comparison that
  credits the card with an acquisition function it did not run; **−13.13**).
  The sensitivity basis is also why `use_bo` reads 7/8 rather than 8/8:
  `log_kvrh__elem20` loses to `random_uniform` but not to
  `farthest_point`. v1 of the replay script reported the sensitivity basis
  under contract-sounding column names, so the shipped csv disagreed with
  the −8.60 printed in the report and deck; that mismatch is recorded in
  `REFREEZE.md`. **Quote the strict number when citing the card.**
- No catastrophic BO failure occurred on any held-out task, so the card's
  protection was not needed here; the card's value proposition (avoiding
  disasters at a known cost) is therefore **not validated** by the holdout —
  it is validated in-sample (report §4.4.2) and by the out-of-base-sample
  48-task extension replication (+18.47, report §4.4), whose run-level
  outputs are shipped in `ext11_7200/`.

For the calibration-set audit of the card ("0/27 significant misses, 5/9
false kills"), the frozen audit output is shipped at
`agent_lab/results/decision_card_audit.csv`, reproducible with
`agent_lab/audit_decision_card.py` (requires pandas; reads
`agent_lab/results/tests.csv` and `task_props.csv`, both shipped).

## 6. Known limitations (stated up front · 已知局限)

- Evidence is retrospective benchmark behavior; the holdout is held out from
  boundary fitting and rule search at the representation level, not a live
  prospective trial, and it cannot test the gate arm (gate-off only).
- Tasks share base material systems; independence is limited and
  enumerated per base in `base_system_overlap.csv`.
- The full-pool AUC diagnostic is reported for analysis only; the deployed
  decision code does not use it. AUC requires candidate-level scores and
  labels, which run records do not carry — it can be recomputed only by
  re-running the main experiment plus `analyze.py`.
- The **25-point substitute** (predicting the 400-budget outcome from the
  first 25 experiments) was evaluated and found weak (report §3.4 /
  §4.4); it is retained only as reported evidence and is **not** part of the
  deployed card.
- The deployed rejection rule is **descriptor dimensionality (dim ≤ 2 →
  withhold BO)** — a task-level prior available before experiments. The
  report states why dimensionality is the operative proxy and prices its
  error rates (5/9 false kills, 0/27 significant misses on the calibration
  set); the offline replay in §5 shows the same conservative behavior on
  held-out representations.
