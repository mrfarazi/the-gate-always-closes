#!/usr/bin/env python3
"""Gate-closure run on Qwen2-VL-7B (paper Sec. 3.3, Table 1): the S5 recipe, seeds 0-2.

Data: GQA train_balanced, semantic == "rel", scene-graph proposal boxes (6x6 grid fallback), Qwen
chat template, answer-token labels, K=4 prefix slots reserved after the image span (see
hra/backbones/qwen2vl.py). `--steps` counts OPTIMISER steps (micro-batch 4 x grad-accum 4 =
effective 16). Every hyper-parameter is a constant below and is dumped to <out>/config.json.

Outputs under --out (default runs/qwen2vl_seed<N>):
  telemetry.jsonl                    one record per micro-batch (step, loss, gate, ...)
  checkpoints/final/{lora, prefix_modules.pt, variant.json}
  FINAL.json                         seed, final_gate_tanh, opt_steps, passes_g1b_seed / closed

  python scripts/train_qwen2vl.py --seed 0                       # 4000 optimiser steps
  python scripts/train_qwen2vl.py --seed 0 --smoke --out /tmp/q  # 8 steps + tiny eval, ~5 min
"""
import argparse
import json
import os
import random
import time

import torch

from hra.paths import DATA, RUNS
from hra.backbones.qwen2vl import (K_PREFIX, MAX_PIXELS, MIN_PIXELS, MAX_PROPOSALS, Collator,
                                   build_gated_qwen)

# ---- the recipe (constants) -------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32              # 2 x rank, as in the adapter_config.json of the released checkpoints
LORA_TARGETS = ["q_proj", "v_proj"]
LR_LORA = 5e-5               # AdamW, weight decay 0.01
WEIGHT_DECAY = 0.01
LR_GRAPH = 1e-4              # Adam over every prefix-graph parameter (incl. kappa, eta, gate)
OPT_STEPS = 4000
BATCH = 4
GRAD_ACCUM = 4               # effective batch 16
GRAD_CLIP = 1.0
LAMBDA_ANG, LAMBDA_ENT = 1.0, 0.1
ALPHA_INIT, ALPHA_MAX, ALPHA_RAMP = 0.001, 0.05, 100   # geometric-loss weight ramp (opt steps)
KAPPA_RANGE = (0.1, 5.0)
CKPT_EVERY = 250             # periodic checkpoints (optimiser steps)
SMOKE_STEPS = 8
SMOKE_EVAL_N = 4
CLOSED_ABS = 0.05            # pass rule: |tanh(gate)| < 0.05 (or gate <= 0)

CONFIG = dict(
    lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_targets=LORA_TARGETS, lr_lora=LR_LORA,
    weight_decay=WEIGHT_DECAY, lr_graph=LR_GRAPH, opt_steps=OPT_STEPS, batch=BATCH,
    grad_accum=GRAD_ACCUM, grad_clip=GRAD_CLIP, lambda_ang=LAMBDA_ANG, lambda_ent=LAMBDA_ENT,
    alpha_init=ALPHA_INIT, alpha_max=ALPHA_MAX, alpha_ramp=ALPHA_RAMP, kappa_range=KAPPA_RANGE,
    k_prefix=K_PREFIX, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
    max_proposals=MAX_PROPOSALS, closed_abs=CLOSED_ABS)


# ---- data ----------------------------------------------------------------------------------
def load_train_items(n_max, seed):
    """GQA train relational questions (shuffled with `seed`) whose image exists, plus the train
    scene graphs used for proposal boxes."""
    with open(os.path.join(DATA, "gqa/train_balanced_questions.json")) as f:
        qs = json.load(f)
    items = [(q, v) for q, v in qs.items() if v["types"]["semantic"] == "rel"]
    random.Random(seed).shuffle(items)
    with open(os.path.join(DATA, "gqa/train_sceneGraphs.json")) as f:
        sgs = json.load(f)
    out = []
    for qid, v in items:
        img = os.path.join(DATA, "gqa/images", f"{v['imageId']}.jpg")
        if os.path.exists(img):
            out.append({"qid": qid, "image": img, "image_id": v["imageId"],
                        "question": v["question"], "answer": v["answer"]})
        if len(out) >= n_max:
            break
    return out, sgs


def load_testdev_rel(n):
    with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
        qs = json.load(f)
    out = []
    for qid, v in sorted(qs.items()):
        if v["types"]["semantic"] != "rel":
            continue
        img = os.path.join(DATA, "gqa/images", f"{v['imageId']}.jpg")
        if os.path.exists(img):
            out.append({"qid": qid, "image": img, "image_id": v["imageId"],
                        "question": v["question"], "answer": v["answer"].strip().lower()})
        if len(out) >= n:
            break
    return out


def norm_answer(text):
    t = text.strip().split("\n")[0]
    for sep in [",", ".", ";", "!", "?"]:
        if sep in t:
            t = t[:t.index(sep)]
    return t.strip().lower()


@torch.no_grad()
def tiny_eval(wrapper, coll, device, n):
    """Greedy GQA-rel testdev accuracy on n items with the live gate (smoke only)."""
    wrapper.eval()
    tok = coll.tok
    correct, rows = 0, []
    for it in load_testdev_rel(n):
        enc = coll.encode_one(it, with_answer=False)
        ids = wrapper.generate(**Collator.prompt_batch(enc, device), max_new_tokens=12)[0]
        pred = norm_answer(tok.decode(ids, skip_special_tokens=True))
        correct += int(pred == it["answer"])
        rows.append({"qid": it["qid"], "pred": pred, "gold": it["answer"]})
    wrapper.train()
    return {"n": len(rows), "acc": 100.0 * correct / max(len(rows), 1), "rows": rows}


# ---- training ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=0, help="optimiser steps (default 4000)")
    ap.add_argument("--out", default="", help="run directory (default runs/qwen2vl_seed<N>)")
    ap.add_argument("--smoke", action="store_true", help="8 optimiser steps + tiny eval")
    args = ap.parse_args()

    opt_steps = args.steps or OPT_STEPS
    if args.smoke:
        opt_steps = SMOKE_STEPS
    run_dir = args.out or os.path.join(RUNS, f"qwen2vl_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**CONFIG, "seed": args.seed, "opt_steps": opt_steps, "smoke": args.smoke},
                  f, indent=1)

    torch.manual_seed(args.seed)      # before the model build: the graph init draws from the RNG
    random.seed(args.seed)
    device = "cuda"
    wrapper, processor = build_gated_qwen(lora_rank=LORA_R, lora_alpha=LORA_ALPHA,
                                          lora_targets=LORA_TARGETS, device=device)
    items, sgs = load_train_items(n_max=opt_steps * BATCH * GRAD_ACCUM + 512, seed=args.seed)
    coll = Collator(processor, sgs, image_token_id=wrapper.image_token_id)
    print(f"[qwen2vl seed{args.seed}] items={len(items)} opt_steps={opt_steps} "
          f"gate_tanh0={wrapper.gate_tanh:+.4f}", flush=True)

    lora_params = [p for _, p in wrapper.vlm.named_parameters() if p.requires_grad]
    graph_params = wrapper.prefix_parameters()
    opt_lora = torch.optim.AdamW(lora_params, lr=LR_LORA, weight_decay=WEIGHT_DECAY)
    opt_graph = torch.optim.Adam(graph_params, lr=LR_GRAPH)
    print(f"[qwen2vl] trainable: lora={sum(p.numel() for p in lora_params):,} "
          f"graph={sum(p.numel() for p in graph_params):,}", flush=True)

    tel = open(os.path.join(run_dir, "telemetry.jsonl"), "a")
    alpha, micro, opt_step, idx, t0 = ALPHA_INIT, 0, 0, 0, time.time()
    wrapper.train()
    while opt_step < opt_steps:
        batch_items = items[idx:idx + BATCH]
        idx = (idx + BATCH) % max(len(items) - BATCH, 1)
        try:
            batch = coll(batch_items)
        except Exception as e:                       # unreadable image etc.: skip the batch
            print(f"[qwen2vl] collate skip: {e}", flush=True)
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        out, aux = wrapper(**batch)
        loss = out.loss + alpha * (LAMBDA_ANG * aux["ang_loss"] + LAMBDA_ENT * aux["ent_loss"])
        (loss / GRAD_ACCUM).backward()
        if torch.isnan(loss):
            raise RuntimeError(f"NaN loss at micro-batch {micro}")
        micro += 1
        if micro % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(graph_params, GRAD_CLIP)
            opt_lora.step(); opt_graph.step()
            opt_lora.zero_grad(); opt_graph.zero_grad()
            with torch.no_grad():
                wrapper.prefix_graph.kappa.data.clamp_(*KAPPA_RANGE)
            opt_step += 1
            alpha = min(ALPHA_MAX, ALPHA_INIT + (ALPHA_MAX - ALPHA_INIT) * opt_step / ALPHA_RAMP)
        rec = {"step": opt_step, "micro": micro, "loss": float(loss.detach()),
               "loss_task": float(out.loss.detach()), "loss_ang": float(aux["ang_loss"].detach()),
               "loss_ent": float(aux["ent_loss"].detach()), "gate": aux["gate_tanh"],
               "gate_raw": aux["gate_raw"], "kappa": float(aux["kappa"].detach()),
               "alpha": alpha, "elapsed": round(time.time() - t0, 1)}
        tel.write(json.dumps(rec) + "\n")
        if micro % 40 == 0 or args.smoke:
            tel.flush()
            print(f"[qwen2vl s{args.seed}] opt{opt_step}/{opt_steps} task={rec['loss_task']:.3f} "
                  f"gate_tanh={rec['gate']:+.5f} kappa={rec['kappa']:.3f}", flush=True)
        if micro % GRAD_ACCUM == 0 and opt_step % CKPT_EVERY == 0 and opt_step < opt_steps:
            wrapper.save_checkpoint(os.path.join(run_dir, "checkpoints", f"step{opt_step}"),
                                    seed=args.seed, steps=opt_step, trainer="scripts/train_qwen2vl.py")
    tel.close()

    wrapper.save_checkpoint(os.path.join(run_dir, "checkpoints", "final"),
                            seed=args.seed, steps=opt_step, trainer="scripts/train_qwen2vl.py")
    gate = wrapper.gate_tanh
    final = {"seed": args.seed, "final_gate_tanh": gate, "opt_steps": opt_step,
             "passes_g1b_seed": bool(gate <= 0 or abs(gate) < CLOSED_ABS),
             "closed": bool(abs(gate) < CLOSED_ABS), "kappa": float(wrapper.prefix_graph.kappa),
             "smoke": args.smoke, "elapsed_s": round(time.time() - t0, 1)}
    if args.smoke:
        final["smoke_eval"] = tiny_eval(wrapper, coll, device, SMOKE_EVAL_N)
    tmp = os.path.join(run_dir, "FINAL.json.tmp")
    with open(tmp, "w") as f:
        json.dump(final, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, "FINAL.json"))
    print(f"[qwen2vl seed{args.seed}] FINAL gate_tanh={gate:+.6f} closed={final['closed']}", flush=True)


if __name__ == "__main__":
    main()
