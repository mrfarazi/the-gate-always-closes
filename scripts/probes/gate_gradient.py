#!/usr/bin/env python3
"""Direct measurement of the gate gradient at gamma=0 (paper Sec. 3.4; supplement Table S7).

Proposition 1 (restated) says the first-order discriminative gradient through the gate is
  d(s+ - s-)/dgamma |_0 = <Delta(I), g(b(I))>,  Delta = dl/dh|c+ - dl/dh|c-
and that this is not distinguishable from zero under two hypotheses:
  H1 support disjointness, H2 weak alignment across images.
Caption-invariance alone does NOT imply it, so we measure it.

For each image we take a positive/negative caption pair (SugarCrepe-style) and compute:
  * grad_pair : dL_pair/dgamma at gamma=0  -- the DISCRIMINATIVE gradient (should be ~0)
  * grad_task : dL_task/dgamma at gamma=0  -- single-caption CE (need NOT be 0)
  * curv      : d2L_task/dgamma2 at 0 by central finite difference. Positive curvature
                makes gamma=0 a local minimum along the gate axis -- the "negative
                utility" statement, measured rather than asserted.

We report mean, std across images, and |mean| / (std/sqrt(N)), which is exactly the
signal-to-noise ratio H2 is about. A |t| well below ~2 means the batch gradient estimator
cannot resolve the term from zero, which is the operational content of "dead gradient".
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH


def grid_boxes(n=6):
    s = 1.0 / n
    return [[j * s, i * s, (j + 1) * s, (i + 1) * s] for i in range(n) for j in range(n)]


def load_pairs(limit):
    """(PIL image, positive caption, hard-negative caption) triples from SugarCrepe.

    SugarCrepe is stored as a HuggingFace DatasetDict with the image embedded and the
    positive caption identified through the ClassLabel feature, so we mirror exactly the
    accessor used in eval_compositionality.eval_sugarcrepe -- same pos/neg convention.
    Sampling round-robin across the seven subtasks keeps the perturbation types balanced.
    """
    from datasets import load_from_disk
    subtasks = ["SugarCrepe_add_att", "SugarCrepe_add_obj", "SugarCrepe_replace_att",
                "SugarCrepe_replace_obj", "SugarCrepe_replace_rel",
                "SugarCrepe_swap_att", "SugarCrepe_swap_obj"]
    loaded = []
    for st in subtasks:
        p = os.path.join(DATA, "sugarcrepe", st)
        if os.path.exists(p):
            try:
                loaded.append(load_from_disk(p))
            except Exception as ex:
                print(f"  [warn] {st}: {ex}", flush=True)
    pairs = []
    idx = 0
    while len(pairs) < limit and loaded:
        progressed = False
        for ds in loaded:
            if idx >= len(ds):
                continue
            progressed = True
            s = ds[idx]
            try:
                true_label = ds.features["true_label"].int2str(s["true_label"])
                labels = s["tested_labels"]
                neg = labels[1] if labels[0] == true_label else labels[0]
                pairs.append((s["image"].convert("RGB"), true_label, neg))
            except Exception:
                pass
            if len(pairs) >= limit:
                break
        if not progressed:
            break
        idx += 1
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rms_ungated")
    ap.add_argument("--checkpoint", default="rms_ungated", help="registry name or checkpoint directory")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--out", default=os.path.join(RUNS, "probes", "gate_gradient.json"))
    args = ap.parse_args()

    from hra import hub
    from hra.checkpoints import load_llava, attach_lora, build_prefix_graph_from_dir
    from hra.train_prefix_fixed import FixedPrefixWrapper

    device = torch.device("cuda")
    ckpt_dir = hub.resolve(args.checkpoint)
    vlm, processor = load_llava()
    attach_lora(vlm, ckpt_dir)
    pg, _variant = build_prefix_graph_from_dir(ckpt_dir, args.variant)
    # Force the gate to be an explicit multiplicative scalar we can differentiate at 0.
    if hasattr(pg, "prefix_proj"):
        pg.prefix_proj.ungated = True          # bypass the built-in tanh gate
    wrapper = FixedPrefixWrapper(vlm, processor, pg).to(device).eval()

    boxes = torch.tensor(grid_boxes(6), dtype=torch.float32).unsqueeze(0).to(device)
    bmask = torch.ones(1, boxes.shape[1]).to(device)
    tok = processor.tokenizer
    img_tok = wrapper.vlm.config.image_token_index

    def seq_logprob(inp, imf, prefix, gamma, caption_ids):
        """-CE of the caption tokens with the prefix scaled by gamma."""
        ie = wrapper._get_embed_fn()(inp["input_ids"])
        imf = imf[0] if imf.dim() == 3 else imf
        pos = int((inp["input_ids"][0] == img_tok).nonzero(as_tuple=True)[0][0])
        npatch, K = imf.shape[0], prefix.shape[1]
        S = inp["input_ids"].shape[1]
        merged = torch.zeros(1, S - 1 + npatch + K, ie.shape[-1],
                             device=device, dtype=ie.dtype)
        merged[0, :pos] = ie[0, :pos]
        merged[0, pos:pos + npatch] = imf
        ps = pos + npatch
        merged[0, ps:ps + K] = prefix[0] * gamma
        merged[0, ps + K:] = ie[0, pos + 1:]
        am = torch.ones(1, merged.shape[1], device=device, dtype=torch.long)
        out = wrapper.vlm.language_model(inputs_embeds=merged, attention_mask=am)
        logits = out.logits[0, -caption_ids.shape[0] - 1:-1].float()
        return -F.cross_entropy(logits, caption_ids, reduction="sum")

    pairs = load_pairs(args.n)
    print(f"[gate-grad] {len(pairs)} caption pairs", flush=True)

    g_pair, g_task, curv = [], [], []
    used = 0
    for img, pos_c, neg_c in pairs:
        if used >= args.n:
            break
        try:
            base_prompt = "USER: <image>\nDescribe the image. ASSISTANT:"
            inp = processor(text=base_prompt, images=img, return_tensors="pt").to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                raw_vision, image_features = wrapper._get_vision_features(inp["pixel_values"])
                feats = wrapper._extract_roi_features(raw_vision, boxes, bmask)
                ie0 = wrapper._get_embed_fn()(inp["input_ids"])
                prefix, _, _, _ = wrapper.prefix_graph(feats, boxes, bmask, ref_embeds=ie0)
            prefix = prefix.detach()
            image_features = image_features.detach()

            ids_p = torch.tensor(tok.encode(" " + pos_c, add_special_tokens=False),
                                 device=device)[:48]
            ids_n = torch.tensor(tok.encode(" " + neg_c, add_special_tokens=False),
                                 device=device)[:48]
            if len(ids_p) < 2 or len(ids_n) < 2:
                continue

            def margin(gam):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    sp = seq_logprob(inp, image_features, prefix, gam, ids_p)
                    sn = seq_logprob(inp, image_features, prefix, gam, ids_n)
                return float(sp - sn), float(-sp)   # (margin, task CE on positive)

            with torch.no_grad():
                e = args.eps
                m_p, t_p = margin(+e)
                m_m, t_m = margin(-e)
                m_0, t_0 = margin(0.0)
            g_pair.append((m_p - m_m) / (2 * e))
            g_task.append((t_p - t_m) / (2 * e))
            curv.append((t_p - 2 * t_0 + t_m) / (e * e))
            used += 1
            if used % 20 == 0:
                print(f"  {used}/{args.n}", flush=True)
        except Exception as ex:
            print(f"  [warn] {type(ex).__name__}: {ex}", flush=True)
            continue

    def summarise(x, label):
        x = np.array([v for v in x if np.isfinite(v)])
        if len(x) == 0:
            return {}
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        t = abs(x.mean()) / se if se and se > 0 else float("nan")
        print(f"  {label:26s} mean={x.mean():+.5f}  std={x.std(ddof=1):.5f}  "
              f"se={se:.5f}  |t|={t:.2f}  n={len(x)}", flush=True)
        return dict(mean=float(x.mean()), std=float(x.std(ddof=1)),
                    se=float(se), abs_t=float(t), n=int(len(x)))

    print(f"\n[gate-grad] results ({used} images):", flush=True)
    res = dict(variant=args.variant, n=used, eps=args.eps,
               grad_pair_discriminative=summarise(g_pair, "dL_pair/dgamma (discrim.)"),
               grad_task_single_caption=summarise(g_task, "dL_task/dgamma (single cap.)"),
               curvature_task=summarise(curv, "d2L_task/dgamma2"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
