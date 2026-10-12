#!/usr/bin/env python3
"""Package and upload the paper's checkpoints to the Hugging Face Hub (maintainer tool).

    python scripts/release/upload_checkpoints.py --manifest /path/to/private_manifest.json --dry-run
    python scripts/release/upload_checkpoints.py --manifest ... --private [--only raw_gated,hra_v1]
    python scripts/release/upload_checkpoints.py --make-public          # after the paper is public

The manifest maps registry names to local directories (see manifest.example.json). For each
name a staging directory is assembled with lora/, prefix_modules.pt | hra_modules.pt (+ extras such
as stage2_checkpoint.pt / FINAL.json), a `variant.json` (schema v1, learned gate read from the
weights on CPU), a small telemetry file if present, and a model card; then it is uploaded to
`<HRA_HF_ORG>/tgac-<backbone>-<name>`. Idempotent: an existing remote whose weights file has the
same sha256 is skipped. Authenticate once with `hf auth login` (a write token); nothing is stored
in the repository.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile

from hra import hub
from hra.checkpoints import write_variant_json
from hra.paths import INTERNVL_ID, LLAVA_PATH, QWEN2VL_ID

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE_MODEL = {"llava-1.5-7b": "llava-hf/llava-1.5-7b-hf", "internvl3-8b": INTERNVL_ID, "qwen2-vl-7b": QWEN2VL_ID}
DESCRIPTIONS = {
    "hra": "Attention-bias design: cone scores from the hyperbolic relational graph are added to image-patch self-attention through a learnable scale beta (softplus).",
    "hra_stage2": "Attention-bias design with the Stage-2 margin-loss gates (LoRA frozen); `stage2_checkpoint.pt` holds beta and the head / layer gates.",
    "prefix": "Prefix design: K=4 tokens from the hyperbolic relational graph are spliced after the image patches; `variant.json` says whether the prefix is RMS-normalised and whether it is gated.",
    "prefix_internvl": "S5 recipe (raw gated hyperbolic prefix) ported to InternVL3-8B; prefix projector width 3584.",
    "prefix_qwen2vl": "S5 recipe (raw gated hyperbolic prefix) ported to Qwen2-VL-7B; prefix projector width 3584.",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def learned_gate(stage_dir, kind):
    import torch
    if kind.startswith("prefix"):
        st = torch.load(os.path.join(stage_dir, "prefix_modules.pt"), map_location="cpu")
        g = st.get("gate")
        if g is None:
            sd = st["prefix_graph"]
            g = sd.get("prefix_proj.gate", sd.get("gate"))
        if g is None:
            return {}
        g = float(torch.as_tensor(g).float())
        return {"gate_raw": g, "gate_tanh": math.tanh(g)}
    st = torch.load(os.path.join(stage_dir, "hra_modules.pt"), map_location="cpu")
    beta = float(st["hra_graph"]["cone_attn.bias_scale"]) if "cone_attn.bias_scale" in st["hra_graph"] else None
    out = {"beta": beta, "softplus_beta": math.log1p(math.exp(beta)) if beta is not None else None}
    s2 = os.path.join(stage_dir, "stage2_checkpoint.pt")
    if os.path.exists(s2):
        st2 = torch.load(s2, map_location="cpu")
        out["stage2_beta"] = float(st2["bias_scale"])
    return out


def knockout_block(name):
    """Markdown rows from results/knockouts/j7_<name>_*.json if present."""
    rows = []
    for bench, label in (("gqa_rel", "GQA-rel (n=1,000)"), ("sugarcrepe", "SugarCrepe (n=1,000)"), ("gqa", "GQA testdev")):
        p = os.path.join(ROOT, "results", "knockouts", f"j7_{name}_{bench}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        vk = d.get("value_key", "gamma")
        by = {}
        for r in d["rows"]:
            by[r.get("label") or r.get(vk)] = r
        off = by.get(0.0) or by.get(0)
        lg = d.get("learned_gate")
        on = None
        if vk == "gamma":
            on = next((r for r in d["rows"] if r.get("gamma") is not None and lg is not None and abs(r["gamma"] - lg) < 1e-9), None)
        else:
            on = by.get(1.0) or by.get(1)
        if off and on:
            rows.append(f"| {label} (n={d.get('n')}) | {off['acc']:.1f} | {on['acc']:.1f} | {on['delta_vs_0_pp']:+.1f} [{on['ci95'][0]:+.1f}, {on['ci95'][1]:+.1f}] |")
    if not rows:
        return "_No knock-out file for this checkpoint in `results/knockouts/`._"
    return "| benchmark | pathway off | pathway on (learned) | delta [95% CI] |\n|---|---|---|---|\n" + "\n".join(rows)


def stage(name, entry, tmp):
    spec = hub.REGISTRY[name]
    src = entry["src"] if isinstance(entry, dict) else entry
    d = os.path.join(tmp, name)
    os.makedirs(d)
    for f in ("lora", "prefix_modules.pt", "hra_modules.pt", "FINAL.json"):
        p = os.path.join(src, f)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(d, f))
        elif os.path.exists(p):
            shutil.copy2(p, d)
    for fname, p in (entry.get("extra", {}) if isinstance(entry, dict) else {}).items():
        shutil.copy2(p, os.path.join(d, fname))
    tel = os.path.join(src, "telemetry.jsonl")
    if os.path.exists(tel) and os.path.getsize(tel) < 5 * 1024 * 1024:
        shutil.copy2(tel, d)
    gate = learned_gate(d, spec["kind"])
    base = BASE_MODEL[spec["backbone"]]
    write_variant_json(d, name=name, paper_label=spec["paper"], kind=spec["kind"], variant=spec["variant"],
                       backbone=base, notes=spec["notes"], **gate)
    files = ", ".join(sorted(os.listdir(d)))
    gate_line = ", ".join(f"{k} = {v:.4f}" for k, v in gate.items() if isinstance(v, float)) or "n/a"
    load_kind = "hra" if spec["kind"].startswith("hra") else "prefix"
    card = open(os.path.join(HERE, "model_card_template.md")).read().format(
        base_model=base, backbone_tag=spec["backbone"], title=f"{spec['paper']} ({name})", name=name,
        paper_label=spec["paper"], description=DESCRIPTIONS[spec["kind"]] + (f" {spec['notes']}" if spec["notes"] else ""),
        kind=spec["kind"], variant=spec["variant"] or "-", gate_line=gate_line, trainer="see variant.json",
        files=files, load_kind=load_kind, knockout_block=knockout_block(name))
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write(card)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--make-public", action="store_true", help="flip every registry repo to public and exit")
    ap.add_argument("--stage-dir", default=None, help="keep the staged directories here instead of a temp dir")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    if args.make_public:
        for n in hub.names():
            api.update_repo_settings(repo_id=hub.repo_id(n), private=False)
            print("public:", hub.repo_id(n))
        return
    manifest = json.load(open(args.manifest))
    names = [n for n in args.only.split(",") if n] or [n for n in hub.names() if n in manifest]
    tmp = args.stage_dir or tempfile.mkdtemp(prefix="tgac_upload_")
    os.makedirs(tmp, exist_ok=True)
    for n in names:
        d = stage(n, manifest[n], tmp)
        weights = next((f for f in ("prefix_modules.pt", "hra_modules.pt") if os.path.exists(os.path.join(d, f))), None)
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs) / 1e6
        print(f"{n:22s} -> {hub.repo_id(n):45s} {size:7.1f} MB  {sorted(os.listdir(d))}")
        if args.dry_run:
            continue
        api.create_repo(repo_id=hub.repo_id(n), private=not args.public, exist_ok=True)
        try:
            remote = api.list_repo_files(hub.repo_id(n))
        except Exception:
            remote = []
        if weights and weights in remote:
            info = api.get_paths_info(hub.repo_id(n), [weights])
            lfs = getattr(info[0], "lfs", None) if info else None
            if lfs and lfs.sha256 == sha256(os.path.join(d, weights)):
                print("   up to date, skipped")
                continue
        api.upload_folder(repo_id=hub.repo_id(n), folder_path=d,
                          commit_message=f"{n}: {hub.REGISTRY[n]['paper']} (ACCV 2026 release)")
        print("   uploaded")
    if not args.stage_dir:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
