#!/usr/bin/env python3
"""J5: ungated-prefix scale sweep.

  python j5_prefix_scale.py --model hypervis --bench gqa --n 2000 \
      --scales 1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0,-1 --out <json> [--boxes repo|grid]
  python j5_prefix_scale.py --model accv_rms_ungated --bench sugarcrepe --n 2000 --out <json>

hypervis          : no-LN checkpoint (ckpt_v102_noln, HYPERVIS_ROOT / HYPERVIS_CKPT),
                    prefix * s right before the merge (hypervis_lib.HyperVisModel).
accv_rms_ungated  : ACCV rms_ungated checkpoint with the prefix genuinely on
                    (ungated=True semantics) then * s.
Both: fixed first-n item lists, all scales of an item in one batched LM call (paired),
prefix / text token L2 and rms recorded on the first 64 items.
The two repos share module names under src/, so only ONE repo's src is put on sys.path,
chosen by --model.
"""
import argparse
import os
import sys

from hra.knockout.common import setup_env  # noqa: E402
setup_env()

from hra.knockout.common import ACCV_BASE, HV_BASE, ensure_baseline_first, parse_values  # noqa: E402
from hra.knockout.job import run_job  # noqa: E402

DEFAULT_SCALES = "1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0,-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hypervis", "accv_rms_ungated"])
    ap.add_argument("--bench", required=True, choices=["gqa", "sugarcrepe", "gqa_rel"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--scales", default=DEFAULT_SCALES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--norm_items", type=int, default=64)
    ap.add_argument("--boxes", default="repo", choices=["repo", "grid"],
                    help="hypervis GQA only: 'repo' = eval_gqa_parallel.py scene-graph boxes "
                         "(all-zero fallback on testdev), 'grid' = 6x6 grid")
    args = ap.parse_args()

    n = args.smoke if args.smoke else args.n
    scales = ensure_baseline_first(parse_values(args.scales), 0.0)

    if args.model == "hypervis":
        from hra.knockout.hypervis_adapter import HyperVisModel, NOLN_CKPT
        data_base = ACCV_BASE  # GQA / SugarCrepe items come from this repo's data/ for both models

        def loader():
            return HyperVisModel("cuda", gqa_boxes=args.boxes)
        meta = {"model": "hypervis", "ckpt": "ckpt_v102_noln", "ckpt_path": NOLN_CKPT,
                "arm": "J5", "gqa_boxes": args.boxes,
                "scale_semantics": "prefix_tokens * s passed to V10VLMWrapper._merge_with_vision_and_prefix",
                "gqa_items": "testdev dict order, first n", "sc_items": "round-robin over 7 subtasks, first n"}
    else:
        from hra.knockout.llava_gate_lib import PrefixModel, ckpt_path
        data_base = ACCV_BASE

        def loader():
            m = PrefixModel("rms_ungated", "cuda")
            m.value_name = "scale"
            return m
        meta = {"model": "accv_rms_ungated", "ckpt": "rms_ungated", "ckpt_path": ckpt_path("rms_ungated"),
                "arm": "J5", "gqa_boxes": "6x6 grid",
                "scale_semantics": ("prefix genuinely on (ungated=True semantics; pre-gate tokens) "
                                    "* s; s=1 is the prefix-on condition (RMS/None cell)"),
                "gqa_items": "testdev dict order, first n", "sc_items": "round-robin over 7 subtasks, first n"}

    run_job("J5", args.out, loader, lambda m: scales, args.bench, n, data_base,
            eps_spec="native", value_key="scale", meta=meta, smoke=args.smoke,
            chunk=args.chunk, norm_items=args.norm_items)


if __name__ == "__main__":
    main()
