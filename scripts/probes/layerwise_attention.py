#!/usr/bin/env python3
"""Layer-wise attention mass onto the injected prefix tokens (supplement Table S8a).

This measures, for every one of the 32 decoder layers, the fraction of attention that
answer-side (text) query positions place on the K=4 injected prefix positions:

    mass(layer) = mean over text queries of  sum_{j in prefix} A[layer, head, q, j]

reported at (a) the learned gate and (b) the gate forced open. If the pathway is
behaviourally closed, mass should sit at chance (K / context_length) at every depth and
barely move when the gate is forced -- i.e. no layer absorbs the signal.

Chance level is reported alongside, because 4 positions out of ~600 is ~0.7% and a raw
number without that reference is uninterpretable.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH

# (name, variant, registry name)
DEFAULT_CKPTS = [
    ("rms_ungated", "rms_ungated", "rms_ungated"),
    ("soft_prompt", "soft_prompt", "soft_prompt"),
    ("raw_gated", "raw_gated", "raw_gated"),
]


def grid_boxes(n=6):
    s = 1.0 / n
    return [[j * s, i * s, (j + 1) * s, (i + 1) * s] for i in range(n) for j in range(n)]


def run_one(name, variant, ckpt, qids, questions, n_layers_report, gate_mode):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from hra import hub
    from hra.checkpoints import attach_lora, build_prefix_graph_from_dir
    from hra.train_prefix_fixed import FixedPrefixWrapper

    device = torch.device("cuda")
    ckpt_dir = hub.resolve(ckpt)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    vlm = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager")
    vlm.requires_grad_(False)
    attach_lora(vlm, ckpt_dir)
    pg, _variant = build_prefix_graph_from_dir(ckpt_dir, variant)

    # The gate lives in different places per class: FixedPrefixGraph / EuclideanGeo* keep it
    # on .prefix_proj, while SoftPromptGraph / EuclideanPrefixGraph own it directly. Resolve
    # it rather than assuming, or "forced open" silently becomes a no-op.
    gate_owner = pg.prefix_proj if (hasattr(pg, "prefix_proj")
                                    and hasattr(pg.prefix_proj, "gate")) else \
        (pg if hasattr(pg, "gate") else None)
    learned_gate = float(torch.tanh(gate_owner.gate).item()) if gate_owner is not None \
        else float("nan")
    ungated = bool(getattr(gate_owner, "ungated", False)) if gate_owner is not None else False
    if gate_mode == "open" and gate_owner is not None:
        # tanh(3.0) ~ 0.995: effectively fully open without an infinite parameter.
        gate_owner.gate.data.fill_(3.0)
        if hasattr(gate_owner, "ungated"):
            gate_owner.ungated = False   # make the (now-open) gate actually apply

    wrapper = FixedPrefixWrapper(vlm, processor, pg).to(device).eval()

    boxes = torch.tensor(grid_boxes(6), dtype=torch.float32).unsqueeze(0).to(device)
    bmask = torch.ones(1, boxes.shape[1]).to(device)
    per_layer = None
    chance = []
    n_ok = 0

    for qid in qids:
        e = questions[qid]
        p = os.path.join(DATA, "gqa/images", f"{e['imageId']}.jpg")
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        prompt = (f"USER: <image>\n{e['question']}\nAnswer the question using a single "
                  f"word or phrase.\nASSISTANT:")
        inp = processor(text=prompt, images=img, return_tensors="pt").to(device)
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # replicate the wrapper's merge, then call the LM with attentions on
                raw_vision, imf = wrapper._get_vision_features(inp["pixel_values"])
                feats = wrapper._extract_roi_features(raw_vision, boxes, bmask)
                embed = wrapper._get_embed_fn()
                ie = embed(inp["input_ids"])
                pref, _, _, _ = wrapper.prefix_graph(feats, boxes, bmask, ref_embeds=ie)
                img_tok = wrapper.vlm.config.image_token_index
                pos = (inp["input_ids"][0] == img_tok).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    continue
                pos = int(pos[0])
                imf = imf[0] if imf.dim() == 3 else imf
                npatch = imf.shape[0]
                K = pref.shape[1]
                S = inp["input_ids"].shape[1]
                new_len = S - 1 + npatch + K
                merged = torch.zeros(1, new_len, ie.shape[-1],
                                     device=device, dtype=ie.dtype)
                am = torch.ones(1, new_len, device=device, dtype=inp["attention_mask"].dtype)
                merged[0, :pos] = ie[0, :pos]
                merged[0, pos:pos + npatch] = imf
                ps = pos + npatch
                merged[0, ps:ps + K] = pref[0]
                merged[0, ps + K:] = ie[0, pos + 1:]
                out = wrapper.vlm.language_model(
                    inputs_embeds=merged, attention_mask=am, output_attentions=True)
                atts = out.attentions           # tuple(L) of (1, H, Q, Q)
                text_start = ps + K
                vals = []
                for a in atts:
                    a = a[0].float()            # (H, Q, Q)
                    q = a[:, text_start:, :]    # text queries only
                    if q.shape[1] == 0:
                        vals.append(float("nan")); continue
                    vals.append(float(q[:, :, ps:ps + K].sum(-1).mean().item()))
                v = np.array(vals)
                per_layer = v if per_layer is None else per_layer + v
                chance.append(K / float(new_len))
                n_ok += 1
        except Exception as ex:
            print(f"  [warn] {qid}: {type(ex).__name__}: {ex}", flush=True)
            continue

    if n_ok == 0:
        return None
    return dict(name=name, gate_mode=gate_mode, learned_gate=learned_gate,
                trained_ungated=ungated, n=n_ok,
                chance=float(np.mean(chance)),
                mass_per_layer=[float(x) for x in (per_layer / n_ok)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_examples", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(RUNS, "probes", "layerwise_attention.json"))
    args = ap.parse_args()

    qs = json.load(open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")))
    rel = [k for k, v in qs.items() if v.get("types", {}).get("semantic") == "rel"]
    qids = rel[: args.n_examples]

    results = []
    for name, variant, ckpt in DEFAULT_CKPTS:
        if not os.path.exists(os.path.join(BASE, ckpt, "prefix_modules.pt")):
            print(f"[skip] {name}: no checkpoint", flush=True); continue
        for mode in ("learned", "open"):
            print(f"=== {name} / gate={mode} ===", flush=True)
            try:
                r = run_one(name, variant, ckpt, qids, qs, 32, mode)
            except Exception as ex:
                print(f"  [fail] {type(ex).__name__}: {ex}", flush=True); r = None
            if r:
                results.append(r)
                m = r["mass_per_layer"]
                print(f"  n={r['n']} chance={r['chance']:.4f} "
                      f"L0={m[0]:.4f} L8={m[8]:.4f} L16={m[16]:.4f} "
                      f"L24={m[24]:.4f} L31={m[-1]:.4f} mean={np.nanmean(m):.4f}",
                      flush=True)
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
