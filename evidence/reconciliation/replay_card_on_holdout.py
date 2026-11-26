#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline replay of the frozen decision card on the held-out set.

The deployed decision card withholds BO on tasks whose descriptor
dimensionality is <= 2 (use space filling instead) and lets BO run
elsewhere. This dimension rule uses only task-level properties known
BEFORE any experiment, so it can be replayed offline on the held-out set:
for every held-out task we read its `dim` from holdout_props.csv, apply the
frozen rule, and compare the outcome with the shipped gate-off runs in
holdout_1680/runs_nogate_ablate.jsonl (budget 400, 4 strategies x 30 seeds).

The holdout consists of 2 bases (log_kvrh, perovskites) x 7 representations
(elem20, full, pca1, pca2, pca5, rp2, rp5) that did not participate in the
boundary fitting or rule search of the card (see tech-report §4.4.5).
Note the honest scope: the holdout re-uses the same two base systems with
new representations; it is held out at the representation level, not the
base level.

## Contract vs sensitivity (v2, 2026-11-24)

Two reporting bases are emitted, and they are NOT interchangeable:

  strict      - the card's own contract: BO = `gp_ucb`, baseline =
                `farthest_point`. These are the two strategies named in
                guideline_card.md, so this is the basis the card can be
                held to. Card net value vs always-BO on the holdout: -8.60.
  sensitivity - best-of-BO (`max` over gp_ucb/gp_ei) vs best-of-space-filling
                (`max` over farthest_point/random_uniform). This is an
                oracle-per-task comparison: it credits the card with a
                better acquisition function it did not run. Net value: -13.13.
                Retained only so the two bases can be inspected side by side.

v1 of this script reported the sensitivity basis under column names that
implied the contract basis (`bo_best` / `space_filling_best`), so the
shipped csv disagreed with the -8.60 printed in the report and the deck.
Fixed here: `gap_strict` is the contract number, `gap_max` is the
sensitivity number, and the printed headline is the contract one.

Outputs (written next to this script):
  replay_card_on_holdout.csv   one row per held-out task, both bases
Prints a short summary. Standard library only.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../evidence
RUNS = ROOT / "holdout_1680" / "runs_nogate_ablate.jsonl"
PROPS = ROOT / "holdout_1680" / "holdout_props.csv"
OUT = Path(__file__).resolve().parent / "replay_card_on_holdout.csv"

# Contract basis: the two strategies named in the frozen card.
BO_CONTRACT = "gp_ucb"
BASELINE_CONTRACT = "farthest_point"
# Sensitivity basis: best-of-family, per task (oracle per task).
BO_FAMILY = ["gp_ucb", "gp_ei"]
SF_FAMILY = ["farthest_point", "random_uniform"]

CARD_DIM_THRESHOLD = 2   # frozen card: dim <= 2 -> withhold BO


def main():
    dims = {}
    with open(PROPS, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dims[row["task"]] = int(row["dim"])
    groups = {}
    with open(RUNS, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if "task" not in d:
                continue
            g = groups.setdefault(d["task"], {}).setdefault(d["strategy"], [])
            g.append(d["n_hits"])

    rows = []
    missing = [t for t in sorted(groups) if t not in dims]
    if missing:
        print(f"FAIL: holdout_props.csv has no dim for {len(missing)} task(s): {missing[:5]}")
        sys.exit(2)
    absent = sorted(t for t in groups
                    if BO_CONTRACT not in groups[t] or BASELINE_CONTRACT not in groups[t])
    if absent:
        print(f"FAIL: contract strategies missing on {len(absent)} task(s): {absent[:5]}")
        sys.exit(2)

    for task in sorted(groups):
        dim = dims[task]
        decision = "withhold_bo" if dim <= CARD_DIM_THRESHOLD else "use_bo"
        bo_c = statistics.mean(groups[task][BO_CONTRACT])
        sf_c = statistics.mean(groups[task][BASELINE_CONTRACT])
        bo_m = max(statistics.mean(groups[task][s]) for s in BO_FAMILY if s in groups[task])
        sf_m = max(statistics.mean(groups[task][s]) for s in SF_FAMILY if s in groups[task])
        rows.append({
            "task": task, "dim": dim, "card_decision": decision,
            "bo_gpucb": round(bo_c, 3),
            "sf_farthest": round(sf_c, 3),
            "gap_strict": round(bo_c - sf_c, 3),
            "ok_strict": (bo_c - sf_c <= 0) if decision == "withhold_bo" else (bo_c - sf_c > 0),
            "bo_bestmax": round(bo_m, 3),
            "sf_bestmax": round(sf_m, 3),
            "gap_max": round(bo_m - sf_m, 3),
            "ok_max": (bo_m - sf_m <= 0) if decision == "withhold_bo" else (bo_m - sf_m > 0),
        })

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    withheld = [r for r in rows if r["card_decision"] == "withhold_bo"]
    proceeded = [r for r in rows if r["card_decision"] == "use_bo"]

    def basis(gap_key, ok_key, label):
        # Card net value vs always-BO: on withheld tasks the card runs the
        # baseline instead of BO, so it gains (sf - bo) = -gap there; on
        # proceeded tasks it runs BO, i.e. the difference is 0 by construction.
        net = sum(-r[gap_key] for r in withheld)
        ok_w = sum(1 for r in withheld if r[ok_key])
        ok_p = sum(1 for r in proceeded if r[ok_key])
        print(f"  [{label}]")
        print(f"    withhold_bo (dim<=2): {len(withheld)} tasks; "
              f"correct rejections {ok_w}/{len(withheld)}, "
              f"false kills {len(withheld) - ok_w} "
              f"({100 * (len(withheld) - ok_w) / len(withheld):.1f}% of rejections)")
        print(f"    use_bo     (dim>2)  : {len(proceeded)} tasks; "
              f"BO beat baseline on {ok_p}/{len(proceeded)}")
        print(f"    card net value vs always-BO: {net:+.3f}")
        return net

    print(f"held-out tasks: {len(rows)}  (card replay, dim rule only, budget 400)")
    net_strict = basis("gap_strict", "ok_strict", "contract: gp_ucb vs farthest_point")
    net_max = basis("gap_max", "ok_max", "sensitivity: best-of-BO vs best-of-SF (oracle per task)")
    print()
    print(f"HEADLINE (contract basis, matches report and deck): {net_strict:+.2f}")
    print(f"Sensitivity basis for reference only: {net_max:+.2f}")
    for r in withheld:
        print(f"  {r['task']}: dim={r['dim']} gp_ucb={r['bo_gpucb']} "
              f"farthest={r['sf_farthest']} gap={r['gap_strict']:+.3f} "
              f"-> {'false kill' if r['gap_strict'] > 0 else 'correct'}")
    print(f"written: {OUT}")
    sys.exit(0)


if __name__ == "__main__":
    main()
