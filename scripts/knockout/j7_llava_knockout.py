#!/usr/bin/env python3
"""J7-LLaVA: open != useful. Forced-gamma knockout at native eps on every gated design,
plus bias-multiplier sweep on the attention-bias designs.

  python j7_llava_knockout.py --ckpt euclidean --bench gqa_rel --n 1000 --out <json>
  python j7_llava_knockout.py --ckpt hra_v1 --bench sugarcrepe --n 1000 --out <json>
  python j7_llava_knockout.py --ckpt stage2 --bench gqa_rel --n 1000 --out <json>

Prefix checkpoints: --gammas (default 0,learned,-learned,0.001,...,1,-1).
Attention-bias checkpoints (hra_v1, small_alpha, stage2): the code allows a scalar
multiplier on the (B,1,P,P) patch bias, so we sweep --bias_mults 0,0.1,0.3,1,3
(0 = bias off, 1 = learned bias on). For stage2 the learned Stage-2 head/layer gates
(sigmoid means) and beta = -8.496 are applied before the multiplier.
Same JSON schema as J4 (paired bootstrap vs gamma = 0 / bias off).
"""
import argparse
import os
import sys

from hra.knockout.common import setup_env  # noqa: E402
setup_env()

from hra.knockout.common import ACCV_BASE, ensure_baseline_first, parse_values  # noqa: E402
from hra.knockout.job import run_job  # noqa: E402

DEFAULT_GAMMAS = "0,learned,-learned,0.001,0.003,0.01,0.03,0.1,0.3,1,-1"
DEFAULT_BIAS_MULTS = "0,0.1,0.3,1,3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", required=True, choices=["sugarcrepe", "gqa_rel", "gqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--gammas", default=DEFAULT_GAMMAS)
    ap.add_argument("--bias_mults", default=DEFAULT_BIAS_MULTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--norm_items", type=int, default=64)
    args = ap.parse_args()

    n = args.smoke if args.smoke else args.n
    from hra.knockout.llava_gate_lib import CKPTS, load_model, ckpt_path
    spec = CKPTS[args.ckpt]
    is_hra = spec["kind"] != "prefix"

    def loader():
        return load_model(args.ckpt, "cuda")

    def values_fn(model):
        if is_hra:
            vals = parse_values(args.bias_mults)
        else:
            vals = parse_values(args.gammas, learned=model.learned_gate)
        return ensure_baseline_first(vals, 0.0)

    meta = {"ckpt": args.ckpt, "ckpt_path": ckpt_path(args.ckpt), "arm": "J7-LLaVA",
            "kind": spec["kind"], "variant": spec.get("variant"),
            "gqa_boxes": "6x6 grid", "sc_items": "round-robin over the 7 subtasks, first n",
            "gqa_items": "testdev dict order, semantic==rel filter, first n"}
    if is_hra:
        meta["gamma_semantics"] = ("value = scalar multiplier on the learned attention bias "
                                   "(0 = off, 1 = learned on); the 'gamma' key holds the multiplier")
        meta["stage2_path"] = spec.get("stage2")
    else:
        meta["gamma_semantics"] = "prefix = pre-gate tokens * gamma; gamma=0 = closed gate"
    value_key = "gamma"
    res = run_job("J7-LLaVA", args.out, loader, values_fn, args.bench, n, ACCV_BASE,
                  eps_spec="native", value_key=value_key, meta=meta, smoke=args.smoke,
                  chunk=args.chunk, norm_items=args.norm_items)


if __name__ == "__main__":
    main()
