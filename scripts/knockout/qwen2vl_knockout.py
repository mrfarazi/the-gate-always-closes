#!/usr/bin/env python3
"""Forced-gamma knock-out on a Qwen2-VL-7B closure checkpoint (paper Sec. 3.3, Table 1).

The learned gate closes (|tanh g| < 0.05); this measures whether forcing the pathway OPEN changes
what the model predicts. GQA-rel testdev (semantic == rel, n = 5,308; no scene graphs, so the
proposals are the 6x6 grid), greedy decoding, strict string match, paired bootstrap of accuracy
against gamma = 0 (10,000 resamples). Output JSON has the same `rows` schema as the shipped
results/external/gate-objective/gsweep/*.json files:
    {gamma, acc, delta_vs_0_pp, ci95: [lo, hi]}  (+ n_correct, p_two_sided)

  python scripts/knockout/qwen2vl_knockout.py --ckpt qwen2vl_seed0 --gammas 0,1,1.5,2 --n 5308
  python scripts/knockout/qwen2vl_knockout.py --ckpt qwen2vl_seed0 --n 20        # smoke
"""
import argparse
import json
import os
import random
import time

import numpy as np

from hra import hub
from hra.knockout.common import atomic_write_json, ensure_baseline_first, parse_values, paired_bootstrap
from hra.paths import DATA, RUNS

N_BOOT = 10000
ITEM_SHUFFLE_SEED = 3      # fixed item order (matters only when --n takes a subset)


def load_gqa_rel(n):
    with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
        qs = json.load(f)
    rel = sorted((q, v) for q, v in qs.items() if v["types"]["semantic"] == "rel")
    random.Random(ITEM_SHUFFLE_SEED).shuffle(rel)
    items = []
    for qid, v in rel:
        img = os.path.join(DATA, "gqa/images", f"{v['imageId']}.jpg")
        if os.path.exists(img):
            items.append({"qid": qid, "image": img, "image_id": v["imageId"],
                          "question": v["question"], "answer": v["answer"].strip().lower()})
        if n and len(items) >= n:
            break
    return items


def norm_answer(text):
    t = text.strip().split("\n")[0]
    for sep in [",", ".", ";", "!", "?"]:
        if sep in t:
            t = t[:t.index(sep)]
    return t.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="qwen2vl_seed0",
                    help="registry name (qwen2vl_seed0..2) or dir with lora/ and prefix_modules.pt")
    ap.add_argument("--gammas", default="0,1,1.5,2", help="forced gate values; 0 is the paired baseline")
    ap.add_argument("--n", type=int, default=0, help="items (0 = all 5,308 GQA-rel testdev questions)")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default="", help="default runs/probes/qwen2vl_knockout_<ckpt>.json")
    args = ap.parse_args()

    from hra.backbones.qwen2vl import Collator, load_gated_qwen
    ckpt_dir = hub.resolve(args.ckpt)
    tag = args.ckpt if args.ckpt in hub.REGISTRY else os.path.basename(ckpt_dir.rstrip("/"))
    out = args.out or os.path.join(RUNS, "probes", f"qwen2vl_knockout_{tag}.json")
    gammas = ensure_baseline_first(parse_values(args.gammas), 0.0)

    device = "cuda"
    wrapper, processor = load_gated_qwen(ckpt_dir, device=device, trainable=False)
    wrapper.eval()
    learned = wrapper.gate_tanh
    print(f"[knockout] {ckpt_dir}  learned tanh(gate) = {learned:+.6f}  gammas={gammas}", flush=True)
    coll = Collator(processor, {}, image_token_id=wrapper.image_token_id)   # testdev: grid boxes
    tok = processor.tokenizer
    items = load_gqa_rel(args.n)

    correct = {g: [] for g in gammas}
    peritem = []
    t0 = time.time()
    for j, it in enumerate(items):
        try:
            enc = coll.encode_one(it, with_answer=False)
            batch = Collator.prompt_batch(enc, device)
            row = {"qid": it["qid"], "gold": it["answer"]}
            for g in gammas:
                ids = wrapper.generate(**batch, gamma_override=g,
                                       max_new_tokens=args.max_new_tokens)[0]
                pred = norm_answer(tok.decode(ids, skip_special_tokens=True))
                row[f"pred_{g:g}"] = pred
                row[f"{g:g}"] = int(pred == it["answer"])
        except Exception as e:                      # keep the gamma lists paired
            print(f"  [warn] {it['qid']}: {type(e).__name__}: {e}", flush=True)
            continue
        for g in gammas:
            correct[g].append(row[f"{g:g}"])
        peritem.append(row)
        if (j + 1) % 100 == 0:
            print(f"[knockout] {j + 1}/{len(items)} ({(time.time() - t0) / (j + 1):.2f}s/item)", flush=True)

    base = np.asarray(correct[0.0], dtype=np.float64)
    rows = []
    for g in gammas:
        c = np.asarray(correct[g], dtype=np.float64)
        d, lo, hi = paired_bootstrap(c, base, n_boot=N_BOOT, seed=0)
        rng = np.random.default_rng(0)
        boots = (c - base)[rng.integers(0, len(c), size=(N_BOOT, len(c)))].mean(1)
        p = min(1.0, 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean())))
        rows.append({"gamma": g, "acc": round(100 * float(c.mean()), 3),
                     "delta_vs_0_pp": round(100 * d, 3),
                     "ci95": [round(100 * lo, 3), round(100 * hi, 3)],
                     "n_correct": int(c.sum()), "p_two_sided": p if g != 0.0 else 1.0})
        print(f"  gamma={g:<5g} acc={rows[-1]['acc']:.3f}  delta={rows[-1]['delta_vs_0_pp']:+.3f} pp "
              f"CI95 [{rows[-1]['ci95'][0]:+.3f}, {rows[-1]['ci95'][1]:+.3f}]", flush=True)
    res = {"tag": f"knockout_{tag}", "bench": "gqa", "subset": "gqa_rel_testdev", "ckpt": ckpt_dir,
           "n": int(len(base)), "learned_gate_tanh": learned, "decoding": "greedy",
           "n_boot": N_BOOT, "rows": rows}
    atomic_write_json(out, res)
    with open(os.path.splitext(out)[0] + "_peritem.jsonl", "w") as f:
        for r in peritem:
            f.write(json.dumps(r) + "\n")
    print(f"DONE n={res['n']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
