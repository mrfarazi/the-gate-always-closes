#!/usr/bin/env python3
"""POPE with a prefix checkpoint's prefix tokens multiplied by --scale (slots kept).

    python scripts/eval_pope_prefix_on.py --checkpoint rms_ungated [--variant rms_ungated] [--scale 1]

--scale 1 (default) is the prefix genuinely on (scale 1 for ungated variants, the learned gate
otherwise): the last row of supplement Table S10. --scale 0 is the zeroed condition (prefix x 0,
slots kept) in which the "RMS prefix, ungated" row of that table was measured (docs/PROTOCOL.md).
6x6 grid proposals, the eval_pope loop unchanged. Output: <out>/<name>_pope_prefix_on.json
(scale 1) or <out>/<name>_pope_prefix_s<scale>.json.
"""
import argparse
import json
import os
import time

import torch

from hra.checkpoints import load_prefix, read_variant, set_prefix_scale
from hra.eval.pope import eval_pope_variant
from hra.paths import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rms_ungated", help="registry name or directory")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out_dir", default=os.path.join(RUNS, "generation"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on the prefix tokens (0 = zeroed, slots kept)")
    ap.add_argument("--splits", default="random,popular,adversarial")
    args = ap.parse_args()
    device = torch.device("cuda")
    wrapper, processor = load_prefix(args.checkpoint, device, variant=args.variant)
    set_prefix_scale(wrapper, args.scale)
    pp = getattr(wrapper.prefix_graph, "prefix_proj", None)
    info = {"checkpoint": str(args.checkpoint), "checkpoint_dir": wrapper.checkpoint_dir,
            "variant": wrapper.variant, "variant_json": read_variant(wrapper.checkpoint_dir),
            "graph": type(wrapper.prefix_graph).__name__,
            "ungated": getattr(pp, "ungated", None), "no_rms": getattr(pp, "no_rms", None),
            "gate_raw": float(pp.gate.detach().float()) if pp is not None and hasattr(pp, "gate") else None,
            "prefix_scale": args.scale, "boxes": "6x6 grid (hra.eval.pope)"}
    print("[graph]", json.dumps(info, default=str), flush=True)
    out = {"info": info, "pope": {}}
    t0 = time.time()
    for v in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(f"  {v}...", flush=True)
        r = eval_pope_variant(v, wrapper.vlm, processor, device, wrapper)
        out["pope"][v] = r.get("metrics", r) if isinstance(r, dict) else r
        print("   ", json.dumps(out["pope"][v]), flush=True)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(str(args.checkpoint).rstrip("/"))
    suffix = "on" if args.scale == 1.0 else f"s{args.scale:g}"
    dst = os.path.join(args.out_dir, f"{name}_pope_prefix_{suffix}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", dst)


if __name__ == "__main__":
    main()
