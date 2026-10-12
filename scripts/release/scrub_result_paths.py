#!/usr/bin/env python3
"""Rewrite machine-specific strings inside the shipped result JSONs (results/**/*.json).

The knock-out / sweep JSONs record the command line and checkpoint path they were produced with.
Those strings carry the training machine's absolute paths; this script maps them to repository
paths, registry names and `$VAR` placeholders so the files are portable. Numbers are never touched
(only string leaves are rewritten, and dict keys listed in KEY_RENAMES).

    python scripts/release/scrub_result_paths.py --manifest /path/to/private_manifest.json   # rewrite
    python scripts/release/scrub_result_paths.py --check                                    # assert clean
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
FORBIDDEN = re.compile(r"/mnt/|/home/|/opt/conda|jupyter|moshiur|rebuttal|erratum|ERRATA|hypertrip-accv|signal_use_vlm|gate-objective/results")
KEY_RENAMES = {"x_chance_rebuttal_convention": "x_chance_mean_over_layers",
               "rebuttal_reference": "reference_probe"}


def build_rules(manifest):
    """Ordered (pattern, replacement) list, longest / most specific first."""
    rules = []
    if manifest:
        from hra import hub  # noqa: E402  (registry repo ids)
        for name, entry in manifest.items():
            src = entry["src"] if isinstance(entry, dict) else entry
            if name in hub.REGISTRY:
                rules.append((src.rstrip("/"), f"hf://{hub.repo_id(name)}"))
            for extra in (entry.get("extra", {}) if isinstance(entry, dict) else {}).values():
                rules.append((extra, f"hf://{hub.repo_id(name)}/{os.path.basename(extra)}"))
    rules += [
        (r"/home/jupyter/gcs-data/conda_envs/[a-z]+/bin/python", "python"),
        (r"/opt/conda/envs/[a-z]+/bin/python", "python"),
        (r"/mnt/md0_raid/research/hypertrip-accv/journal/scripts/", "scripts/knockout/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/journal/results/", "results/knockouts/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/outputs/iclr_e1/", "results/matched_mode/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/outputs/final/", "results/generation/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/rebuttal/(results|archive)/", "results/probes/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/scripts/", "scripts/probes/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/src/", "hra/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/models/llava-1\.5-7b", "llava-hf/llava-1.5-7b-hf"),
        (r"/mnt/md0_raid/research/hypertrip-accv/data/", "$HRA_DATA/"),
        (r"/mnt/md0_raid/research/hypertrip-accv/?", "$HRA_ROOT/"),
        (r"/mnt/md0_raid/research/hypertrip/rebuttal_results/revision_evals/ckpt_v102_noln", "$HYPERVIS_CKPT"),
        (r"/mnt/md0_raid/research/hypertrip/models/llava-1\.5-7b", "llava-hf/llava-1.5-7b-hf"),
        (r"/mnt/md0_raid/research/hypertrip/data/", "$HRA_DATA/"),
        (r"/mnt/md0_raid/research/hypertrip/?", "$HYPERVIS_ROOT/"),
        (r"/mnt/md0_raid/research/gate-objective/results/", "results/external/gate-objective/"),
        (r"/mnt/md0_raid/research/gate-objective/?", "$GATE_OBJECTIVE_ROOT/"),
        (r"/mnt/md0_raid/research/signal_use_vlm/results/journal/j6_llava/", "results/probes/j6/"),
        (r"/mnt/md0_raid/research/signal_use_vlm/external/accv_journal/scripts/", "scripts/knockout/"),
        (r"/mnt/md0_raid/research/signal_use_vlm/?", "$HRA_ROOT/"),
        (r"/mnt/md0_raid/research/hf_home", "$HF_HOME"),
        (r"/dev/shm/hf_cache/hub", "$HF_HUB_CACHE"),
        (r"outputs/rebuttal/internvl/seed(\d)", r"hf://mrfarazi/tgac-internvl3-internvl3-seed\1"),
        (r"outputs/rebuttal/", "runs/"),
        (r"rebuttal/results/", "results/probes/"),
        (r"x_chance_rebuttal_convention", "x_chance_mean_over_layers"),
        (r"rebuttal text-query set", "text-query set"),
        (r"rebuttal convention", "mean-over-layers convention"),
        (r"the rebuttal's", "the original probe's"),
        (r"6x6 grid \(rebuttal_eval\.py / layerwise_attention\.py protocol\)", "6x6 grid (layerwise_attention.py protocol)"),
        (r"rebuttal_eval\.py", "eval_gqa_spatial.py"),
        (r"erratum-E4-corrected prefix-on condition", "prefix-on condition (RMS/None cell)"),
        (r"\(J5 semantics, erratum E4\)", "(J5 semantics)"),
        (r"\brebuttal\b", "original probe"),
        (r"rebuttal", "original-probe"),
    ]
    literal = [(re.escape(a) if not a.startswith("/home") and "(" not in a and "\\" not in a and "?" not in a and "[" not in a and "+" not in a else a, b) for a, b in rules]
    return literal


def rewrite(obj, rules, stats):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = KEY_RENAMES.get(k, k)
            if nk != k:
                stats["keys"] += 1
            out[nk] = rewrite(v, rules, stats)
        return out
    if isinstance(obj, list):
        return [rewrite(v, rules, stats) for v in obj]
    if isinstance(obj, str):
        new = obj
        for pat, rep in rules:
            new = re.sub(pat, rep, new)
        if new != obj:
            stats["strings"] += 1
        return new
    return obj


def offending(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if FORBIDDEN.search(str(k)):
                yield f"{path}.{k} (key)"
            yield from offending(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from offending(v, f"{path}[{i}]")
    elif isinstance(obj, str) and FORBIDDEN.search(obj):
        yield f"{path} = {obj[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="private manifest (name -> {src, extra}) used to map checkpoint paths to hub ids")
    ap.add_argument("--check", action="store_true", help="only report offending strings; exit 1 if any")
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest)) if args.manifest and os.path.exists(args.manifest) else {}
    rules = build_rules(manifest)
    files = []
    for dp, _dn, fn in os.walk(args.results):
        files += [os.path.join(dp, f) for f in fn if f.endswith(".json")]
    bad_total, changed = 0, 0
    for f in sorted(files):
        with open(f) as fh:
            data = json.load(fh)
        if not args.check:
            stats = {"strings": 0, "keys": 0}
            new = rewrite(data, rules, stats)
            if stats["strings"] or stats["keys"]:
                with open(f, "w") as fh:
                    json.dump(new, fh, indent=1)
                changed += 1
                data = new
        bad = list(offending(data))
        if bad:
            bad_total += len(bad)
            print(f"{os.path.relpath(f, ROOT)}: {len(bad)} offending string(s)")
            for b in bad[:5]:
                print("   ", b)
    print(f"{len(files)} files scanned; {changed} rewritten; {bad_total} offending strings remain")
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
