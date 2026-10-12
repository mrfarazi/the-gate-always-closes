#!/usr/bin/env python3
"""Matched inference-mode evaluation (paper Sec. 5.1, Table 5; supplement Tables S5-S6).

Evaluates every configuration under BOTH inference modes on the same question sets:

  DROP   = auxiliary pathway removed at inference (LoRA-only).
  ACTIVE = auxiliary pathway on at inference (prefix at learned gate / scale 1 if ungated;
           attention bias at its learned scale).

Outputs -> <out>/ (default results/matched_mode when HRA_E1_DIR is unset; use --out to
write elsewhere, e.g. runs/matched_mode, so the shipped files stay untouched):
  gqa_drop/{name}_{rel,attr}.json  gqa_active/{name}_{rel,attr}.json  results_{rel,attr}.json
  vsr/<name>/vsr_{summary,predictions}.json
  E1_SUMMARY.json

    python scripts/matched_mode.py --gpus 0,1,2,3 --out runs/matched_mode
Then: python scripts/analyze_matched_mode.py (paired bootstrap contrasts) or scripts/make_tables.py.
"""
import argparse
import json
import os
import subprocess

from hra.paths import PYTHON, RESULTS, RUNS

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HRA_E1_DIR", os.path.join(RESULTS, "matched_mode"))
GPUS = [0]

# registry names (hra.hub); resolved to directories by the evaluators
CKPT = {
    "rms_pub": "rms_ungated",
    "rms_r2":  "rms_ungated_r2",
    "euc_geo": "euclidean_geo_r1",
    "soft":    "soft_prompt",
    "euc":     "euclidean",
    "g2":      "g2",
    "small_a": "small_alpha",
}

# (name, gqa_mtype, vsr_model, ckpt, variant)   variant only used by prefix mode
DROP = [
    ("vanilla",      "vanilla",   "vanilla",   None,             None),
    ("rmsPub_DROP",  "lora_only", "lora_only", CKPT["rms_pub"],  None),
    ("rmsR2_DROP",   "lora_only", "lora_only", CKPT["rms_r2"],   None),
    ("eucGeo_DROP",  "lora_only", "lora_only", CKPT["euc_geo"],  None),
    ("soft_DROP",    "lora_only", "lora_only", CKPT["soft"],     None),
    ("euc_DROP",     "lora_only", "lora_only", CKPT["euc"],      None),
    ("g2_DROP",      "lora_only", "lora_only", CKPT["g2"],       None),
    ("smallA_DROP",  "lora_only", "lora_only", CKPT["small_a"],  None),
]
ACTIVE = [
    ("rmsPub_ACT",   "prefix", "prefix", CKPT["rms_pub"],  "rms_ungated"),
    ("rmsR2_ACT",    "prefix", "prefix", CKPT["rms_r2"],   "rms_ungated"),
    ("eucGeo_ACT",   "prefix", "prefix", CKPT["euc_geo"],  "euclidean_geo"),
    ("soft_ACT",     "prefix", "prefix", CKPT["soft"],     "soft_prompt"),
    ("euc_ACT",      "prefix", "prefix", CKPT["euc"],      "euclidean"),
    ("g2_ACT",       "prefix", "prefix", CKPT["g2"],       "g2_no_geo"),
    ("smallA_ACT",   "hra",    "hra",    CKPT["small_a"],  None),
]


def log(msg):
    print(msg, flush=True)


def run_gqa(configs, out_dir, sem):
    """One eval_gqa_spatial.py invocation: all configs in parallel, one GPU each."""
    if not configs:
        return
    os.makedirs(out_dir, exist_ok=True)
    specs = []
    for i, (name, mtype, _vsr, ckpt, variant) in enumerate(configs):
        gpu = GPUS[i % len(GPUS)]
        specs.append(f"{name}:{mtype}:{ckpt or ''}:{variant or ''}:{gpu}")
    models_arg = ";".join(specs)
    log(f"\n=== GQA {sem} -> {out_dir} ({len(configs)} configs) ===")
    subprocess.run(
        [PYTHON, os.path.join(SCRIPTS, "eval_gqa_spatial.py"),
         "--semantic_type", sem, "--output_dir", out_dir, "--models", models_arg],
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        check=False,
    )


def run_vsr(configs):
    """VSR per-config, pooled NGPU-wide (eval_vsr.py handles one model per call)."""
    if not configs:
        return
    vsr_root = os.path.join(OUT, "vsr")
    os.makedirs(vsr_root, exist_ok=True)
    pending = list(configs)
    while pending:
        wave, pending = pending[:len(GPUS)], pending[len(GPUS):]
        procs = []
        for i, (name, _gm, vsr_model, ckpt, variant) in enumerate(wave):
            gpu = GPUS[i % len(GPUS)]
            od = os.path.join(vsr_root, name)
            cmd = [PYTHON, os.path.join(SCRIPTS, "eval_vsr.py"),
                   "--model", vsr_model, "--output_dir", od]
            if ckpt:
                cmd += ["--checkpoint", ckpt]
            if variant:
                cmd += ["--variant", variant]
            log(f"  VSR {name} (model={vsr_model}) on GPU {gpu}")
            lf = open(os.path.join(vsr_root, f"{name}.log"), "w")
            procs.append(subprocess.Popen(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                stdout=lf, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()


def assemble():
    """Collect all per-config numbers into one table."""
    def load_gqa(out_dir, sem):
        p = os.path.join(out_dir, f"results_{sem}.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    table = {}
    for out_dir in (os.path.join(OUT, "gqa_drop"), os.path.join(OUT, "gqa_active")):
        for sem in ("rel", "attr"):
            for name, r in load_gqa(out_dir, sem).items():
                table.setdefault(name, {})[f"gqa_{sem}"] = r
    vsr_root = os.path.join(OUT, "vsr")
    if os.path.isdir(vsr_root):
        for name in os.listdir(vsr_root):
            sp = os.path.join(vsr_root, name, "vsr_summary.json")
            if os.path.exists(sp):
                table.setdefault(name, {})["vsr"] = json.load(open(sp))

    with open(os.path.join(OUT, "E1_SUMMARY.json"), "w") as f:
        json.dump(table, f, indent=2)

    log("\n================ matched-mode summary ================")
    log(f"{'config':16s} {'GQA-rel':>9s} {'GQA-attr':>9s} {'VSR':>7s}")
    order = [c[0] for c in DROP] + [c[0] for c in ACTIVE]
    for name in order:
        r = table.get(name, {})
        rel = r.get("gqa_rel", {}).get("accuracy", "-")
        attr = r.get("gqa_attr", {}).get("accuracy", "-")
        vsr = r.get("vsr", {}).get("accuracy", "-")
        log(f"{name:16s} {str(rel):>9s} {str(attr):>9s} {str(vsr):>7s}")
    log(f"\nSaved -> {os.path.join(OUT, 'E1_SUMMARY.json')}")


def main():
    global OUT, GPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "matched_mode"))
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    ap.add_argument("--only", default="", help="comma-separated config names to run (default all)")
    args = ap.parse_args()
    OUT, GPUS = args.out, [int(g) for g in args.gpus.split(",")]
    only = set(filter(None, args.only.split(",")))
    drop = [c for c in DROP if not only or c[0] in only]
    active = [c for c in ACTIVE if not only or c[0] in only]
    os.makedirs(OUT, exist_ok=True)
    log(f"matched-mode evaluation starting. OUT={OUT} GPUs={GPUS}")

    # GQA: 4 sequential invocations, each internally parallel across the given GPUs.
    run_gqa(drop,   os.path.join(OUT, "gqa_drop"),   "rel")
    run_gqa(drop,   os.path.join(OUT, "gqa_drop"),   "attr")
    run_gqa(active, os.path.join(OUT, "gqa_active"), "rel")
    run_gqa(active, os.path.join(OUT, "gqa_active"), "attr")

    # VSR: pooled per-config.
    run_vsr(drop + active)

    assemble()
    log("\nmatched-mode evaluation done.")


if __name__ == "__main__":
    main()
