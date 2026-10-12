#!/usr/bin/env python3
"""Matched inference-mode analysis: paired-bootstrap contrasts (paper Sec. 5.1-5.2).

Consumes the per-question outputs of scripts/matched_mode.py (or the shipped copies in
results/matched_mode/) and reports two families of contrasts:

  DROP vs ACTIVE for the SAME checkpoint      -> the cost of an active pathway;
  DROP-mode contrasts between configurations  -> does training-time geometry separate on any
                                                 axis (GQA-rel / GQA-attr / VSR)?

All contrasts are PAIRED on the shared question set (same qids / same (image, caption)), so the
difference is a within-item comparison, not two independent accuracies. Reports observed delta
accuracy, 95% bootstrap CI and a two-sided bootstrap p-value.

  python scripts/analyze_matched_mode.py [--dir results/matched_mode]
"""
import json
import os

import numpy as np

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
OUT = os.environ.get("HRA_E1_DIR", os.path.join(RESULTS, "matched_mode"))
RNG = np.random.RandomState(42)
NBOOT = 10000


# ---- loaders: map config-name -> {item_id: correct(0/1)} per metric --------------
def load_gqa(mode_dir, name, sem):
    p = os.path.join(OUT, mode_dir, f"{name}_{sem}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {qid: int(v["correct"]) for qid, v in d.items()}


def load_vsr(name):
    p = os.path.join(OUT, "vsr", name, "vsr_predictions.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {(r["image"], r["caption"]): int(r["correct"]) for r in d}


def metric_map(name, metric):
    """metric in {rel, attr, vsr}. DROP names live in gqa_drop, ACT in gqa_active."""
    if metric == "vsr":
        return load_vsr(name)
    mode_dir = "gqa_active" if name.endswith("_ACT") else "gqa_drop"
    return load_gqa(mode_dir, name, metric)


# ---- paired bootstrap ------------------------------------------------------------
def paired_bootstrap(a, b):
    """a, b: dict item_id->0/1. Returns (accA, accB, dAcc, ci_lo, ci_hi, p2)."""
    if a is None or b is None:
        return None
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    av = np.array([a[i] for i in ids], dtype=float)
    bv = np.array([b[i] for i in ids], dtype=float)
    n = len(ids)
    d0 = (av.mean() - bv.mean()) * 100
    boots = np.empty(NBOOT)
    for k in range(NBOOT):
        idx = RNG.randint(0, n, n)
        boots[k] = (av[idx].mean() - bv[idx].mean()) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p2 = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p2 = min(1.0, p2)
    return (av.mean() * 100, bv.mean() * 100, d0, lo, hi, p2, n)


def row(tag, a_name, b_name, metric):
    a, b = metric_map(a_name, metric), metric_map(b_name, metric)
    r = paired_bootstrap(a, b)
    if r is None:
        print(f"  {tag:34s} [{metric}]  (missing)")
        return
    accA, accB, d, lo, hi, p, n = r
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
    print(f"  {tag:34s} [{metric:4s}] {accA:6.2f} vs {accB:6.2f}  "
          f"Δ={d:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.1e} {star}  (n={n})")


def main():
    print(f"matched-mode analysis  (dir={OUT}, NBOOT={NBOOT})\n")

    print("=== PRIMARY: DROP vs ACTIVE, same checkpoint  (positive Δ = dropping helps) ===")
    same = [("rmsPub", "rmsPub_DROP", "rmsPub_ACT"),
            ("rmsR2",  "rmsR2_DROP",  "rmsR2_ACT"),
            ("eucGeo", "eucGeo_DROP", "eucGeo_ACT"),
            ("soft",   "soft_DROP",   "soft_ACT"),
            ("euc",    "euc_DROP",    "euc_ACT"),
            ("g2",     "g2_DROP",     "g2_ACT"),
            ("smallA", "smallA_DROP", "smallA_ACT")]
    for metric in ("rel", "attr", "vsr"):
        for tag, d, a in same:
            row(f"{tag}: DROP-ACTIVE", d, a, metric)
        print()

    print("=== DROP-mode contrasts: geometry vs no geometry / vanilla  (positive Δ = geometry separates) ===")
    check = [("rmsR2 - soft",   "rmsR2_DROP",  "soft_DROP"),
             ("rmsR2 - euc",    "rmsR2_DROP",  "euc_DROP"),
             ("rmsR2 - g2",     "rmsR2_DROP",  "g2_DROP"),
             ("rmsR2 - vanilla","rmsR2_DROP",  "vanilla"),
             ("eucGeo - soft",  "eucGeo_DROP", "soft_DROP"),
             ("eucGeo - euc",   "eucGeo_DROP", "euc_DROP"),
             ("eucGeo - vanilla","eucGeo_DROP","vanilla"),
             ("smallA - vanilla","smallA_DROP","vanilla"),
             ("soft - vanilla", "soft_DROP",   "vanilla"),
             ("euc - vanilla",  "euc_DROP",    "vanilla"),
             ("g2 - vanilla",   "g2_DROP",     "vanilla")]
    for metric in ("rel", "attr", "vsr"):
        for tag, a, b in check:
            row(tag, a, b, metric)
        print()


def cli():
    import argparse
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=OUT, help="directory with gqa_drop/, gqa_active/, vsr/")
    args = ap.parse_args()
    OUT = args.dir
    main()


if __name__ == "__main__":
    cli()
