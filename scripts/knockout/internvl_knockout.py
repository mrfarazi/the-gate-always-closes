#!/usr/bin/env python3
"""Forced-gamma knockout on an InternVL3-8B closure checkpoint.

For LLaVA and Qwen2-VL we claim *behavioural* closure, not merely a small gate value.
This makes the third backbone's claim equally strong: it measures whether forcing the
pathway open changes what the model actually predicts.

We use teacher-forced cross-entropy on the gold answer tokens rather than generation:
InternVL's generate() would need a KV-cache path for spliced prefix embeddings that the
training wrapper does not implement, and CE on the answer is a direct, paired,
interpretable read of the same quantity. Reported per-example and paired-bootstrapped, so
"no effect" is a measured CI rather than an eyeballed similarity.

  python scripts/knockout/internvl_knockout.py --ckpt internvl3_seed0 --n 300 --gammas 0.0,1.0,2.0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS, LOCAL_FILES_ONLY  # noqa: E402
from hra import hub  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="registry name (internvl3_seed0..2) or dir with prefix_modules.pt and lora/")
    ap.add_argument("--backbone", default="internvl3-8b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gammas", default="0.0,1.0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from hra.backbones.internvl import (BACKBONES, GQARelDataset, make_collate,
                                        InternVLGatedPrefix, build_labels)
    args.ckpt = hub.resolve(args.ckpt)
    import transformers
    from transformers import AutoProcessor
    from peft import PeftModel
    from hra.train_prefix_fixed import FixedPrefixGraph
    from torch.utils.data import DataLoader

    spec = BACKBONES[args.backbone]
    dev = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(spec["model_id"], local_files_only=LOCAL_FILES_ONLY)
    model = getattr(transformers, spec["cls"]).from_pretrained(
        spec["model_id"], dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
    model.requires_grad_(False)
    lora = os.path.join(args.ckpt, "lora")
    if os.path.exists(lora):
        model.model.language_model = PeftModel.from_pretrained(
            model.model.language_model, lora)

    pg = FixedPrefixGraph(vis_dim=spec["vis_dim"], k=4, prefix_dropout=0.0,
                          no_rms=True, ungated=False)
    pg.prefix_proj.llm_dim = spec["llm_dim"]
    pg.prefix_proj.proj = torch.nn.Linear(pg.embed_dim, 4 * spec["llm_dim"])
    st = torch.load(os.path.join(args.ckpt, "prefix_modules.pt"), map_location="cpu")
    pg.load_state_dict(st["prefix_graph"], strict=True)
    learned = float(torch.tanh(pg.prefix_proj.gate))
    wrapper = InternVLGatedPrefix(model, processor, pg,
                                  vision_attr=spec["vision_attr"]).to(dev).eval()
    print(f"[knockout] {spec['nice']}  learned gate = {learned:+.6f}", flush=True)

    ds = GQARelDataset(processor, limit=args.n * 3)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=make_collate(processor))
    gammas = [float(g) for g in args.gammas.split(",")]
    per = {g: [] for g in gammas}
    n = 0
    for batch in dl:
        if n >= args.n:
            break
        px = batch["pixel_values"].to(dev, dtype=torch.bfloat16)
        ids = batch["input_ids"].to(dev)
        am = batch["attention_mask"].to(dev)
        bx = batch["boxes"].to(dev)
        bm = batch["box_mask"].to(dev)
        lb = build_labels(batch["input_ids"], batch["answers"],
                          processor.tokenizer).to(dev)
        if (lb != -100).sum() == 0:      # answer not locatable in the prompt
            continue
        try:
            with torch.no_grad():
                vals = {}
                for g in gammas:
                    loss, _, _, _ = wrapper(px, ids, am, bx, bm, labels=lb,
                                            gamma_override=g)
                    vals[g] = float(loss)
            if not all(np.isfinite(v) for v in vals.values()):
                continue
            for g in gammas:
                per[g].append(vals[g])
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{args.n}", flush=True)
        except Exception as e:
            print(f"  [warn] {type(e).__name__}: {e}", flush=True)
            continue

    base = gammas[0]
    a = np.array(per[base])
    print(f"\n[knockout] n={n}   answer cross-entropy (lower = better)")
    res = {"backbone": spec["nice"], "ckpt": args.ckpt, "n": n,
           "learned_gate": learned, "gammas": {}}
    rng = np.random.default_rng(0)
    for g in gammas:
        b = np.array(per[g])
        d = b - a
        idx = rng.integers(0, len(d), size=(10000, len(d)))
        boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        tag = "" if g == base else ("*" if (lo > 0 or hi < 0) else " ")
        print(f"  gamma={g:<5} CE={b.mean():.4f}   dCE vs gamma={base}: "
              f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{tag}")
        res["gammas"][str(g)] = {"ce": float(b.mean()), "d_ce": float(d.mean()),
                                 "ci_lo": float(lo), "ci_hi": float(hi)}
    out = args.out or os.path.join(RUNS, "probes", f"internvl_knockout_{os.path.basename(args.ckpt.rstrip('/'))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
