#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FailBound-Lab evidence reconciliation.

Re-enumerates the six design JSONL files in this evidence pack against the
experiment design reported in the technical report, validates internal field
consistency for ALL runs, validates cross-file design facts (identical task
sets across the four main files, use_gate consistency with file names,
budget compliance, index uniqueness, ttfh semantics, 48-subset structure),
enumerates the auxiliary permutation-robustness shards, and cross-checks a
seeded random sample of rows from the aggregate table evidence_4320runs.csv
(ablation arm).

The EXPECTED constants below are a design snapshot, manually aligned with
tech-report §4.4.2 / §3.4 (they are not parsed from the report).

Standard library only. No experiment is re-run; every count is a
re-enumeration of the shipped files.

Exit codes: 0 = all checks pass, 1 = any check failed, 2 = missing/corrupt file.
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]  # .../evidence

# ---------------------------------------------------------------------------
# Design snapshot (manually aligned with tech-report §4.4.2 / §3.4)
EXPECTED = {
    "main_8640/runs_gate.jsonl":            {"runs": 2160, "tasks": 18, "strategies": 4, "seeds": 30, "budget": 200, "gate": True},
    "main_8640/runs_nogate.jsonl":          {"runs": 2160, "tasks": 18, "strategies": 4, "seeds": 30, "budget": 200, "gate": False},
    "main_8640/runs_gate_ablate.jsonl":     {"runs": 2160, "tasks": 18, "strategies": 4, "seeds": 30, "budget": 200, "gate": True},
    "main_8640/runs_nogate_ablate.jsonl":   {"runs": 2160, "tasks": 18, "strategies": 4, "seeds": 30, "budget": 200, "gate": False},
    "ext11_7200/runs_nogate_ablate.jsonl":  {"runs": 7200, "tasks": 60, "strategies": 4, "seeds": 30, "budget": 400, "gate": False},
    "holdout_1680/runs_nogate_ablate.jsonl":{"runs": 1680, "tasks": 14, "strategies": 4, "seeds": 30, "budget": 400, "gate": False},
}
GRAND_TOTALS = {"main_8640": 8640, "ext11_7200": 7200, "holdout_1680": 1680}
REQUIRED_FIELDS = ["task", "strategy", "seed", "use_gate", "chosen", "y_seq",
                   "hits", "cost_seq", "n_hits", "ttfh", "n_exp", "spent"]
MAIN_FILES = [r for r in EXPECTED if r.startswith("main_8640/")]
EXCLUDED_BASES_48 = {"matbench_glass", "matbench_steels"}  # per report + _sig_worse.py
SUBSET_48_EXPECTED = 48 * 4 * 30  # = 5760

# Auxiliary permutation-robustness shards: shipped and enumerated (counts
# reported, no design assertions — they are supplementary robustness runs).
AUX_SHARDS = [
    "ext11_7200/runs_nogate_ablate_shufgap.jsonl",
    "ext11_7200/runs_nogate_ablate_shufgap.tmatbench_expt_gap__elem20.jsonl",
    "ext11_7200/runs_nogate_ablate_shufgap.tmatbench_expt_gap__pca1.jsonl",
    "ext11_7200/runs_nogate_ablate_shufgap.tmatbench_expt_gap__pca2.jsonl",
    "ext11_7200/runs_nogate_ablate_shufgap.tmatbench_expt_gap__pca5.jsonl",
    "ext11_7200/runs_nogate_ablate_shufgap.tmatbench_expt_gap__rp2.jsonl",
    "ext11_7200/runs_nogate_ablate_shufgap.tmatbench_expt_gap__rp5.jsonl",
    "ext11_7200/runs_nogate_ablate_shufprobe.jsonl",
]

# evidence_4320runs.csv = ablation-arm aggregate (144 rows =
# 18 task-arms x 4 strategies x 2 gate settings). Verified mapping:
#   csv task  = f"{dataset}__{representation}"   -> jsonl "task"
#   gate_on   -> runs_gate_ablate.jsonl   (use_gate == True)
#   gate_off  -> runs_nogate_ablate.jsonl (use_gate == False)
# The csv stores means rounded to 3 decimals, so the comparison tolerance is
# half of the last stored digit (5e-4), not machine epsilon.


def is_run_record(obj):
    return isinstance(obj, dict) and {"task", "strategy", "seed"} <= set(obj.keys())


def load_runs(path):
    runs, meta_lines, bad_lines = [], 0, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue
            if is_run_record(obj):
                runs.append(obj)
            else:
                meta_lines += 1
    return runs, meta_lines, bad_lines


def group_stats(runs):
    g = {}
    for r in runs:
        k = (r["task"], r["strategy"])
        d = g.setdefault(k, {"seeds": set(), "n": 0, "n_hits": [], "ttfh": [], "spent": []})
        d["seeds"].add(r["seed"])
        d["n"] += 1
        d["n_hits"].append(r["n_hits"])
        d["ttfh"].append(r["ttfh"])
        d["spent"].append(r["spent"])
    return g


def check_file(relpath, spec, root, report_rows, problems):
    path = root / relpath
    if not path.exists():
        problems.append(f"MISSING FILE: {relpath}")
        return None
    runs, meta_lines, bad_lines = load_runs(path)
    if bad_lines:
        problems.append(f"{relpath}: {bad_lines} unparseable line(s)")
    if relpath.startswith("main_8640/") and meta_lines != 19:
        problems.append(f"{relpath}: {meta_lines} meta lines, expected 19")
    g = group_stats(runs)
    strategies = sorted({k[1] for k in g})
    tasks = sorted({k[0] for k in g})
    if len(runs) != spec["runs"]:
        problems.append(f"{relpath}: {len(runs)} runs, expected {spec['runs']}")
    if len(tasks) != spec["tasks"]:
        problems.append(f"{relpath}: {len(tasks)} distinct tasks, expected {spec['tasks']}")
    if len(strategies) != spec["strategies"]:
        problems.append(f"{relpath}: {len(strategies)} strategies, expected {spec['strategies']}")
    for s in strategies:
        n_s = sum(d["n"] for (t, st), d in g.items() if st == s)
        if n_s != spec["tasks"] * spec["seeds"]:
            problems.append(f"{relpath}: strategy {s} has {n_s} runs, expected {spec['tasks']*spec['seeds']}")
    for k, d in g.items():
        if d["n"] != spec["seeds"]:
            problems.append(f"{relpath}: group {k} has {d['n']} runs, expected {spec['seeds']}")
        if d["seeds"] != set(range(spec["seeds"])):
            problems.append(f"{relpath}: group {k} seeds != 0..{spec['seeds']-1}")
    for r in runs:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            problems.append(f"{relpath}: run {r.get('task')}/{r.get('strategy')}/{r.get('seed')} missing fields {missing}")
            break
        if bool(r["use_gate"]) != spec["gate"]:
            problems.append(f"{relpath}: use_gate={r['use_gate']} contradicts file design ({spec['gate']})")
            break
        if r["n_exp"] > spec["budget"]:
            problems.append(f"{relpath}: run {r.get('task')}/{r.get('strategy')}/{r.get('seed')} n_exp={r['n_exp']} exceeds budget {spec['budget']}")
            break
        c = r["chosen"]
        if len(set(c)) != len(c):
            problems.append(f"{relpath}: run {r.get('task')}/{r.get('strategy')}/{r.get('seed')} has duplicate chosen indices")
            break
        t = r["ttfh"]
        if not (isinstance(t, (int, float)) and 1 <= t <= r["n_exp"]):
            problems.append(f"{relpath}: run {r.get('task')}/{r.get('strategy')}/{r.get('seed')} ttfh={t} outside 1..n_exp")
            break
    for strat in strategies:
        report_rows.append({"file": relpath, "strategy": strat,
                            "n_tasks": len({t2 for (t2, st) in g if st == strat}),
                            "n_runs": sum(d["n"] for (t2, st), d in g.items() if st == strat),
                            "seeds_per_group": spec["seeds"],
                            "expected_runs": spec["tasks"] * spec["seeds"],
                            "meta_lines": "", "bad_lines": ""})
    return {"runs": runs, "meta_lines": meta_lines, "bad_lines": bad_lines, "groups": g}


def cross_file_design(runs_by_file, problems, overlap_rows):
    # (1) within each arm, the gate and nogate files must cover the identical
    # task set. NOTE: the main arm and the ablation arm intentionally use
    # different task sets (36 task-arms = 18 main + 18 ablate, report §4.4.2),
    # so equality is asserted per arm, not across arms.
    arms = {}
    for rel in MAIN_FILES:
        arm = "ablate" if "ablate" in rel else "main"
        arms.setdefault(arm, {})[rel] = sorted({r["task"] for r in runs_by_file[rel]})
    for arm, files_ in arms.items():
        sets_ = list(files_.values())
        if any(s != sets_[0] for s in sets_[1:]):
            problems.append(f"cross-file: gate/nogate task sets differ within {arm} arm")
    main_ref = arms["main"][MAIN_FILES[0]] if "main" in arms else []
    # (2) ext11 must contain the full main-ablate task-name set (re-run at budget 400)
    ext = {r["task"] for r in runs_by_file["ext11_7200/runs_nogate_ablate.jsonl"]}
    ablate_names = set(arms.get("ablate", {None: []})[MAIN_FILES[2]]) if "ablate" in arms else set()
    missing = [t for t in ablate_names if t not in ext]
    if missing:
        problems.append(f"cross-file: ext11 missing main-ablate task names: {sorted(missing)[:5]}")
    # (3) holdout task names must not collide with the main arm's lam-operator tasks
    hold = {r["task"] for r in runs_by_file["holdout_1680/runs_nogate_ablate.jsonl"]}
    if hold & set(main_ref):
        problems.append(f"cross-file: holdout task names overlap main arm: {sorted(hold & set(main_ref))[:5]}")
    # (4) overlap matrix (informational; base-level dedup across layers)
    layers = {}
    for rel in MAIN_FILES:
        arm = "main_ablate" if "ablate" in rel else "main"
        layers.setdefault(arm, set()).update(r["task"] for r in runs_by_file[rel])
    layers["ext11"] = ext
    layers["holdout"] = hold
    bases = {}
    for layer, tasks_ in layers.items():
        for t in tasks_:
            b = t.split("__")[0]
            bases.setdefault(b, {}).setdefault(layer, set()).add(t)
    all_layers = sorted(layers)
    for b in sorted(bases):
        row = {"base": b}
        for layer in all_layers:
            row[layer + "_tasks"] = len(bases[b].get(layer, set()))
        shared = [l for l in all_layers if bases[b].get(l)]
        row["layers_present"] = len(shared)
        overlap_rows.append(row)


def internal_consistency(all_runs, problems, limit_report=5):
    bad = 0
    shown = 0
    for relpath, runs in all_runs.items():
        for r in runs:
            ok = True
            chosen, y, n_exp = r.get("chosen"), r.get("y_seq"), r.get("n_exp")
            if not (isinstance(chosen, list) and isinstance(y, list)
                    and len(chosen) == len(y) == n_exp):
                ok = False
            # hits is the per-experiment 0/1 sequence (len == n_exp);
            # n_hits is its sum (number of hits), not its length.
            if r.get("n_hits") != sum(r.get("hits") or []):
                ok = False
            if isinstance(r.get("hits"), list) and len(r["hits"]) != n_exp:
                ok = False
            try:
                if abs(float(r.get("spent") or 0.0) - float(sum(r.get("cost_seq") or []))) > 1e-6:
                    ok = False
            except Exception:
                ok = False
            if not ok:
                bad += 1
                if shown < limit_report:
                    problems.append(f"internal inconsistency: {relpath} {r.get('task')}/{r.get('strategy')}/{r.get('seed')}")
                    shown += 1
    return bad


def crosscheck_aggregates(root, runs_by_file, n_groups, rng, problems, rows):
    csv_path = root / "evidence_4320runs.csv"
    if not csv_path.exists():
        csv_path = root.parent / "evidence_4320runs.csv"  # legacy location: repo root
    if not csv_path.exists():
        problems.append(f"MISSING aggregate table: {root / 'evidence_4320runs.csv'}")
        return
    ablate = {}
    for rel in ("main_8640/runs_gate_ablate.jsonl", "main_8640/runs_nogate_ablate.jsonl"):
        for r in runs_by_file[rel]:
            ablate[(r["task"], r["strategy"], bool(r["use_gate"]))] = r
    with open(csv_path, encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    # the aggregate table must be complete: 144 rows, all mappable
    if len(csv_rows) != 144:
        problems.append(f"aggregate csv: {len(csv_rows)} rows, expected 144")
    usable = [c for c in csv_rows
              if (f"{c['dataset']}__{c['representation']}", c["strategy"], c["gate"] == "gate_on") in ablate]
    if len(usable) != 144:
        problems.append(f"aggregate csv: only {len(usable)}/144 rows mappable to ablation jsonl runs")
    if not usable:
        problems.append("crosscheck: no csv row could be mapped to ablation jsonl runs")
        return
    # default: verify ALL rows (cheap). --sample-n > 0 restricts to a seeded sample.
    if n_groups and n_groups < len(usable):
        pick = rng.sample(usable, n_groups)
    else:
        pick = usable
    for c in pick:
        key = (f"{c['dataset']}__{c['representation']}", c["strategy"], c["gate"] == "gate_on")
        runs = [r for r in runs_by_file["main_8640/runs_gate_ablate.jsonl" if key[2] else "main_8640/runs_nogate_ablate.jsonl"]
                if (r["task"], r["strategy"], bool(r["use_gate"])) == key]
        row = {"dataset": c["dataset"], "representation": c["representation"],
               "strategy": c["strategy"], "gate": c["gate"],
               "csv_n_runs": c["n_runs"], "recomputed_n_runs": len(runs),
               "csv_n_hits_mean": c["n_hits_mean"],
               "recomputed_n_hits_mean": round(statistics.mean(r["n_hits"] for r in runs), 3),
               "csv_ttfh_mean": c["ttfh_mean"],
               "recomputed_ttfh_mean": round(statistics.mean(r["ttfh"] for r in runs), 3)}
        # csv stores 3-decimal means -> tolerance 5e-4 (half of last stored digit)
        row["PASS"] = (abs(row["recomputed_n_hits_mean"] - float(c["n_hits_mean"])) < 5e-4
                       and abs(row["recomputed_ttfh_mean"] - float(c["ttfh_mean"])) < 5e-4
                       and row["recomputed_n_runs"] == int(float(c["n_runs"])))
        if not row["PASS"]:
            problems.append(f"crosscheck FAIL: {key} csv=({c['n_hits_mean']},{c['ttfh_mean']}) recomputed=({row['recomputed_n_hits_mean']},{row['recomputed_ttfh_mean']})")
        rows.append(row)


def write_summary(all_groups, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "task", "strategy", "n_runs", "n_hits_mean", "ttfh_mean", "spent_mean"])
        for rel in sorted(all_groups):
            for (task, strat), d in sorted(all_groups[rel].items()):
                w.writerow([rel, task, strat, d["n"],
                            round(statistics.mean(d["n_hits"]), 4),
                            round(statistics.mean(d["ttfh"]), 4),
                            round(statistics.mean(d["spent"]), 4)])


def main():
    ap = argparse.ArgumentParser(description="FailBound-Lab evidence reconciliation")
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="evidence/ directory (default: parent of this script's folder; cwd-independent)")
    ap.add_argument("--sample-n", type=int, default=0,
                    help="0 = verify ALL 144 aggregate rows (default); N>0 = seeded sample of N rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task-filter", default=None,
                    help="txt file with 48 task names (relative paths resolve against --root; expect 5760 runs)")
    args = ap.parse_args()
    root = Path(args.root)
    rng = random.Random(args.seed)
    problems, report_rows, cross_rows, overlap_rows = [], [], [], []

    tf = None
    if args.task_filter:
        tf_path = Path(args.task_filter)
        if not tf_path.is_absolute():
            # resolution order: evidence root, this script's directory, then cwd
            for base in (root, Path(__file__).resolve().parent, Path.cwd()):
                cand = base / tf_path
                if cand.exists():
                    tf_path = cand
                    break
        if not tf_path.exists():
            print(f"FAIL: task-filter file not found: {tf_path}")
            sys.exit(2)
        tf = {t.strip() for t in open(tf_path, encoding="utf-8") if t.strip()}

    runs_by_file, groups_by_file = {}, {}
    for rel, spec in EXPECTED.items():
        res = check_file(rel, spec, root, report_rows, problems)
        if res is None:
            print(f"FAIL {rel}: missing")
            sys.exit(2)
        runs_by_file[rel] = res["runs"]
        groups_by_file[rel] = res["groups"]
        report_rows.append({"file": rel, "strategy": "(all)",
                            "n_tasks": len({t for (t, s) in res["groups"]}),
                            "n_runs": len(res["runs"]), "seeds_per_group": spec["seeds"],
                            "expected_runs": spec["runs"],
                            "meta_lines": res["meta_lines"], "bad_lines": res["bad_lines"]})

    # grand totals
    for layer, total in GRAND_TOTALS.items():
        got = sum(len(runs_by_file[rel]) for rel in EXPECTED if rel.startswith(layer))
        if got != total:
            problems.append(f"grand total {layer}: {got} != {total}")
        print(f"{layer} total = {got} / {total} {'PASS' if got == total else 'FAIL'}")

    # cross-file design facts + overlap matrix
    cross_file_design(runs_by_file, problems, overlap_rows)
    print("cross-file design checks done (see base_system_overlap.csv)")

    # internal consistency over ALL runs
    bad = internal_consistency(runs_by_file, problems)
    print(f"internal consistency: {bad} bad run(s) over {sum(len(v) for v in runs_by_file.values())} runs")

    # auxiliary shards: enumerate only (supplementary robustness runs)
    for rel in AUX_SHARDS:
        p = root / rel
        if not p.exists():
            problems.append(f"MISSING auxiliary shard: {rel}")
            continue
        runs, meta_lines, bad_lines = load_runs(p)
        print(f"aux shard {rel}: {len(runs)} run records, {meta_lines} meta lines, {bad_lines} bad lines")

    # 48-task subset: count AND structure (8 valid bases x 6 representations)
    if tf:
        got = sum(1 for r in runs_by_file["ext11_7200/runs_nogate_ablate.jsonl"] if r["task"] in tf)
        if got != SUBSET_48_EXPECTED:
            problems.append(f"48-task subset: {got} runs != expected {SUBSET_48_EXPECTED}")
        bases_in = {t.split("__")[0] for t in tf if "__" in t}
        reps_in = {t.split("__", 1)[1] for t in tf if "__" in t}
        ext_bases = {t.split("__")[0] for t in (r["task"] for r in runs_by_file["ext11_7200/runs_nogate_ablate.jsonl"])}
        struct_ok = (len(bases_in) == 8 and len(reps_in) == 6
                     and bases_in == ext_bases - EXCLUDED_BASES_48)
        if not struct_ok:
            problems.append(f"48-task subset structure: {len(bases_in)} bases / {len(reps_in)} reps "
                            f"(expected 8 x 6 = ext11 bases minus {sorted(EXCLUDED_BASES_48)})")
        print(f"48-task subset = {got} / {SUBSET_48_EXPECTED} runs, "
              f"structure = {len(bases_in)} bases x {len(reps_in)} reps "
              f"{'PASS' if got == SUBSET_48_EXPECTED and struct_ok else 'FAIL'}")

    # crosscheck vs aggregate csv
    crosscheck_aggregates(root, runs_by_file, args.sample_n, rng, problems, cross_rows)
    print(f"crosscheck vs evidence_4320runs.csv: {sum(1 for r in cross_rows if r['PASS'])}/{len(cross_rows)} rows PASS")

    # outputs
    rec = root / "reconciliation"
    rec.mkdir(exist_ok=True)
    with open(rec / "reconciliation_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "strategy", "n_tasks", "n_runs", "seeds_per_group", "expected_runs", "meta_lines", "bad_lines"])
        w.writeheader()
        w.writerows(report_rows)
    with open(rec / "crosscheck_samples.csv", "w", newline="", encoding="utf-8") as fh:
        if cross_rows:
            w = csv.DictWriter(fh, fieldnames=list(cross_rows[0].keys()))
            w.writeheader()
            w.writerows(cross_rows)
    with open(rec / "base_system_overlap.csv", "w", newline="", encoding="utf-8") as fh:
        if overlap_rows:
            w = csv.DictWriter(fh, fieldnames=list(overlap_rows[0].keys()))
            w.writeheader()
            w.writerows(overlap_rows)
    write_summary(groups_by_file, rec / "summary_by_task_strategy.csv")
    with open(rec / "RECONCILE_RESULT.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(problems) if problems else "ALL CHECKS PASS")
        fh.write("\n")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:20]:
            print(" -", p)
        sys.exit(1)
    print("\nALL CHECKS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
