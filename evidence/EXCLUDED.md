# Excluded files · 有意排除的文件清单

The following files exist in the original campaign working directories but are
**intentionally not part of this evidence pack**. They are listed here so that
their absence is a documented decision, not an omission.

| Source file (relative to campaign dir) | Lines | Reason |
|---|---|---|
| `ext11/agent_lab/results/runs_nogate_ablate.BAK14tasks.jsonl` | 1,695 | Historical backup snapshot of an earlier 14-task stage of the extension campaign; superseded by the final 60-task file. |
| `ext11/agent_lab/results/runs_gate_ablate.ABORTED.jsonl` | 152 | Extension-campaign gate arm, aborted mid-campaign (incomplete shard). **Consequence, stated openly: the extension set has no gate-on data — it is gate-off only.** The main experiment's gate arm is a different campaign (`main_8640/runs_gate*.jsonl`, different task sets). |
| `ext11/agent_lab/results/runs_nogate_ablate_probe.jsonl` | 6 | 6-line debug stub written while wiring the extension campaign; referenced by no analysis, not an experimental result. |
| `ext11/agent_lab/results/runs_nogate.jsonl` | 448 | Early partial shard from extension-campaign setup; superseded by the final 60-task file. |
| `holdout/agent_lab/results/runs_gate_ablate.ABORTED.jsonl` | 152 | Holdout gate arm, aborted mid-campaign. **Consequence, stated openly: the holdout set is gate-off only and cannot test the gate arm's runtime behavior; the frozen card is replayed on it offline instead (see `reconciliation/replay_card_on_holdout.py` and `EVIDENCE_GUIDE.md` §5).** |

Note on the two probes: `runs_nogate_ablate_shufprobe.jsonl` **is** shipped
(24 records) because it belongs to the shufgap permutation-robustness series
and is labeled in `EVIDENCE_GUIDE.md` §1 as a wiring probe excluded from all
statistics; the ext11 probe above is excluded because it is a pre-campaign
debug stub with no lineage in any shipped analysis series.

No file in this pack was modified, filtered, or re-generated: every JSONL here
is a byte-identical copy of the original campaign output. Integrity of every
shipped data file can be checked against `checksums/SHA256SUMS.txt` and
`MANIFEST.json`.
