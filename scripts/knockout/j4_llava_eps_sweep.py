#!/usr/bin/env python3
"""J4-LLaVA: sqrt(eps) law on LLaVA-1.5 -- forced gate gamma x RMSNorm epsilon.

  python j4_llava_eps_sweep.py --ckpt base_r1 --bench sugarcrepe --n 1000 --eps native \
      --gammas 0,1e-4,... --out results/j4_base_r1_sugarcrepe_eps_native.json [--smoke 16]

For every item the pre-gate prefix is computed once; all gammas are evaluated in one
batched LM call (paired). eps is applied by setting `variance_epsilon` on every
LlamaRMSNorm of the language model (vision tower excluded) and restored afterwards.
Env: the LLaVA-1.5 stack of requirements.txt; paths from hra.paths.
"""
import argparse
import os
import sys

from hra.knockout.common import setup_env  # noqa: E402
setup_env()

from hra.knockout.common import ACCV_BASE, ensure_baseline_first, parse_values  # noqa: E402
from hra.knockout.job import run_job  # noqa: E402

DEFAULT_GAMMAS = ("0,1e-4,3e-4,1e-3,2e-3,3e-3,5e-3,1e-2,3e-2,0.1,0.3,1,"
                  "-1e-3,-1e-2,-1,learned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", required=True, choices=["sugarcrepe", "gqa_rel", "gqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--eps", default="native", help="float or 'native' (1e-5 for LLaVA-1.5)")
    ap.add_argument("--gammas", default=DEFAULT_GAMMAS,
                    help="comma list; 'learned' / '-learned' allowed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="use only the first N items")
    ap.add_argument("--chunk", type=int, default=16, help="max gammas per LM call")
    ap.add_argument("--norm_items", type=int, default=64)
    args = ap.parse_args()

    n = args.smoke if args.smoke else args.n
    import torch
    from hra.knockout.llava_gate_lib import CKPTS, PrefixModel, ckpt_path

    def loader():
        return PrefixModel(args.ckpt, "cuda")

    def values_fn(model):
        vals = parse_values(args.gammas, learned=model.learned_gate)
        return ensure_baseline_first(vals, 0.0)

    meta = {"ckpt": args.ckpt, "ckpt_path": ckpt_path(args.ckpt),
            "variant": CKPTS[args.ckpt]["variant"], "arm": "J4-LLaVA",
            "gamma_semantics": ("prefix = pre-gate tokens * gamma (== gate := atanh(gamma), "
                                "ungated=False); gamma=0 is the closed-gate baseline"),
            "gqa_boxes": "6x6 grid (layerwise_attention.py protocol)",
            "sc_items": "round-robin over the 7 subtasks, first n",
            "gqa_items": "testdev dict order, semantic==rel filter, first n"}
    run_job("J4-LLaVA", args.out, loader, values_fn, args.bench, n, ACCV_BASE,
            eps_spec=args.eps, value_key="gamma", meta=meta, smoke=args.smoke,
            chunk=args.chunk, norm_items=args.norm_items)


if __name__ == "__main__":
    main()
