#!/usr/bin/env python3
"""HyperVis (BMVC 2026 repository) adapter: the ungated 4-token prefix, scaled by s right
before it is merged into the embedding sequence.

Loader: train_v10.V10VLMWrapper.load_for_eval on the no-LN checkpoint ckpt_v102_noln
(its injected `config` has
use_prefix_ln=False, so topk_gate.norm is Identity; verified at load time).

The scale hook does not touch the repo: we call the wrapper's own vision / roi /
hyper_graph / _merge_with_vision_and_prefix / _expand_labels methods and pass
`prefix_tokens * s` into the merge, exactly where V10VLMWrapper.score/generate would
pass the raw prefix. All s values of one item share the vision features and run in one
batched LM call.

Protocol fidelity: SugarCrepe uses 6x6 grid boxes and the eval_compositionality prompt
(as the published 79.94); GQA uses the eval_gqa_parallel.py prompt/normalisation and, by
default (`boxes='repo'`), the repo's scene-graph boxes -- which for GQA testdev are the
all-zero fallback for every image (0/398 testdev images have a train or val scene graph),
i.e. the published prefix-on 34.75 was produced with an image-independent prefix.
`boxes='grid'` is offered as the image-dependent alternative.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

from hra.knockout.common import (HV_BASE, GQA_PROMPT, SC_PROMPT, caption_labels, grid_boxes)

# The HyperVis model lives in its own repository (BMVC 2026; same authors). Set
#   HYPERVIS_ROOT=/path/to/hypertrip        (checkout containing src/train_v10.py)
#   HYPERVIS_CKPT=/path/to/ckpt_v102_noln   (default: $HYPERVIS_ROOT/checkpoints/ckpt_v102_noln)
# to re-run the Raw/None rows; the shipped results/knockouts/j5_hypervis_*.json do not need it.
NOLN_CKPT = os.environ.get("HYPERVIS_CKPT") or (
    os.path.join(HV_BASE, "checkpoints", "ckpt_v102_noln") if HV_BASE else None)


def _import_hypervis():
    if not HV_BASE:
        raise RuntimeError("HYPERVIS_ROOT is not set: the HyperVis adapter needs the BMVC repository "
                           "(github.com/mrfarazi/hypertrip) on disk; see docs/REPRODUCE.md")
    os.environ.setdefault("HYPERTRIP_BASE", HV_BASE)
    if os.path.join(HV_BASE, "src") not in sys.path:
        sys.path.insert(0, os.path.join(HV_BASE, "src"))
    from train_v10 import V10VLMWrapper
    return V10VLMWrapper


def _per_row_neg_ce(logits, labels):
    lg = logits[:, :-1]
    lb = labels[:, 1:]
    m = lb != -100
    n_tok = int(m[0].sum())
    ce = F.cross_entropy(lg[m], lb[m], reduction="none").view(logits.shape[0], n_tok)
    return (-ce.mean(dim=1)).tolist()


class HyperVisModel:
    value_name = "scale"
    max_new_tokens = 20

    def __init__(self, device="cuda", ckpt=NOLN_CKPT, gqa_boxes="repo"):
        V10VLMWrapper = _import_hypervis()
        self.device = torch.device(device)
        self.ckpt = ckpt
        self.wrapper, self.processor = V10VLMWrapper.load_for_eval(ckpt, self.device)
        self.vlm = self.wrapper.vlm
        self.lm = self.vlm.language_model
        self.tok = self.processor.tokenizer
        self.embed = self.wrapper._get_embed_fn()
        self.img_tok = self.vlm.config.image_token_index
        hg = self.wrapper.hyper_graph
        self.norm_class = type(hg.topk_gate.norm).__name__
        self.use_prefix_ln = bool(getattr(hg, "use_prefix_ln", True))
        self.kappa = float(hg.kappa.item())
        self.gqa_boxes = gqa_boxes
        self.learned_gate = float("nan")       # no gate in this design
        self.learned_gate_raw = float("nan")
        self.trained_ungated = True
        self.grid = torch.tensor(grid_boxes(6), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.grid_mask = torch.ones(1, self.grid.shape[1], device=self.device)
        self.zero_boxes = torch.zeros(1, 36, 4, device=self.device)
        self.zero_mask = torch.zeros(1, 36, device=self.device)

    # ---- prep ---------------------------------------------------------------------
    @torch.no_grad()
    def _vision_and_prefix(self, pixel_values, boxes, bmask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vo = self.vlm.vision_tower(pixel_values, output_hidden_states=True)
            raw = vo.last_hidden_state                     # (1, 577, 1024) incl. CLS (repo)
            imf = self.vlm.multi_modal_projector(raw)      # (1, 577, 4096)  (repo)
            feats = self.wrapper._extract_roi_features(raw, boxes, bmask)
            pref, _, _, _ = self.wrapper.hyper_graph(feats, boxes, bmask)
        return imf, pref

    @torch.no_grad()
    def _prepare(self, image, prompt, boxes, bmask, labels_fn=None, reuse=None):
        inp = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        if reuse is not None and "imf" in reuse:
            imf, pref = reuse["imf"], reuse["base"]
        else:
            imf, pref = self._vision_and_prefix(inp["pixel_values"], boxes, bmask)
        ids, am = inp["input_ids"], inp["attention_mask"]
        ie = self.embed(ids)
        special = ids == self.img_tok
        with torch.autocast("cuda", dtype=torch.bfloat16):
            merged, attn = self.wrapper._merge_with_vision_and_prefix(
                ie, imf, torch.zeros_like(pref), ids, am, special)
        pos = int(special[0].nonzero(as_tuple=True)[0][0])
        npatch, K = imf.shape[1], pref.shape[1]
        ps = pos + npatch
        labels = None
        if labels_fn:
            labels = self.wrapper._expand_labels(labels_fn(inp), special, npatch, K)
        return dict(inp=inp, imf=imf, base=pref, ie=ie, merged=merged, attn=attn,
                    labels=labels, ps=ps, K=K, pos=pos, boxes=boxes, bmask=bmask)

    def prepare_sc(self, image, caption, reuse=None):
        return self._prepare(image, SC_PROMPT.format(caption=caption), self.grid, self.grid_mask,
                             labels_fn=lambda inp: caption_labels(self.processor, inp["input_ids"]),
                             reuse=reuse)

    def prepare_gqa(self, image, question):
        if self.gqa_boxes == "grid":
            b, m = self.grid, self.grid_mask
        else:   # 'repo': scene-graph boxes -> none exist for testdev -> zero fallback
            b, m = self.zero_boxes, self.zero_mask
        return self._prepare(image, GQA_PROMPT.format(q=question), b, m)

    # ---- batched eval ---------------------------------------------------------------
    def _batch(self, ctx, values):
        G = len(values)
        merged = ctx["merged"].expand(G, -1, -1).clone()
        ps, K = ctx["ps"], ctx["K"]
        base = ctx["base"][0]
        for g, v in enumerate(values):
            merged[g, ps:ps + K] = (base * float(v)).to(merged.dtype)
        return merged, ctx["attn"].expand(G, -1)

    @torch.no_grad()
    def score(self, ctx, values, chunk=16):
        out = []
        for c0 in range(0, len(values), chunk):
            vals = values[c0:c0 + chunk]
            merged, attn = self._batch(ctx, vals)
            labels = ctx["labels"].expand(len(vals), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.lm(inputs_embeds=merged, attention_mask=attn)
            out += _per_row_neg_ce(o.logits.float(), labels)
        return out

    @torch.no_grad()
    def generate(self, ctx, values, chunk=16):
        out = []
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        for c0 in range(0, len(values), chunk):
            vals = values[c0:c0 + chunk]
            merged, attn = self._batch(ctx, vals)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = self.lm.generate(inputs_embeds=merged, attention_mask=attn,
                                       pad_token_id=pad, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
            out += [self.tok.decode(g, skip_special_tokens=True) for g in gen]
        return out

    @staticmethod
    def _token_norms(x):
        x = x.float()
        return x.pow(2).mean(dim=-1).sqrt().flatten(), x.norm(dim=-1).flatten()

    # ---- prefix slot removed (published LoRA-only condition, 61.03 GQA / 73.87 SC) ------
    supports_absent = True

    def _absent(self, ctx):
        ps, K = ctx["ps"], ctx["K"]
        m = torch.cat([ctx["merged"][:, :ps], ctx["merged"][:, ps + K:]], dim=1)
        a = torch.cat([ctx["attn"][:, :ps], ctx["attn"][:, ps + K:]], dim=1)
        lab = None
        if ctx["labels"] is not None:
            lab = torch.cat([ctx["labels"][:, :ps], ctx["labels"][:, ps + K:]], dim=1)
        return m, a, lab

    @torch.no_grad()
    def score_absent(self, ctx):
        m, a, lab = self._absent(ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.lm(inputs_embeds=m, attention_mask=a)
        return _per_row_neg_ce(o.logits.float(), lab)[0]

    @torch.no_grad()
    def generate_absent(self, ctx):
        m, a, _ = self._absent(ctx)
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = self.lm.generate(inputs_embeds=m, attention_mask=a, pad_token_id=pad,
                                   max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(gen[0], skip_special_tokens=True)

    def norms(self, ctx):
        prms, pl2 = self._token_norms(ctx["base"][0])
        ie = ctx["ie"][0]
        pos = ctx["pos"]
        text = torch.cat([ie[:pos], ie[pos + 1:]], dim=0)
        trms, tl2 = self._token_norms(text)
        return {"prefix_rms": float(prms.mean()), "prefix_l2": float(pl2.mean()),
                "prefix_rms_per_token": prms.tolist(), "prefix_l2_per_token": pl2.tolist(),
                "text_rms_median": float(trms.median()), "text_l2_median": float(tl2.median())}

    # ---- unit check -------------------------------------------------------------------
    @torch.no_grad()
    def unit_check(self, image, prompt_kind="gqa", question="What is this?", caption="A photo."):
        res = {"norm_class": self.norm_class, "use_prefix_ln": self.use_prefix_ln,
               "kappa": self.kappa, "gqa_boxes": self.gqa_boxes}
        res["passed_noln_loaded"] = (self.norm_class == "Identity")
        if prompt_kind == "sugarcrepe":
            ctx = self.prepare_sc(image, caption)
        else:
            ctx = self.prepare_gqa(image, question)
        m1, _ = self._batch(ctx, [1.0])
        m01, _ = self._batch(ctx, [0.1])
        ps, K = ctx["ps"], ctx["K"]
        n1 = m1[0, ps:ps + K].float().norm(dim=-1)
        n01 = m01[0, ps:ps + K].float().norm(dim=-1)
        res["scale_hook_norm_ratio_at_0p1"] = float((n01 / n1.clamp(min=1e-12)).mean())
        res["passed_scale_hook"] = abs(res["scale_hook_norm_ratio_at_0p1"] - 0.1) < 0.01
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l1 = self.lm(inputs_embeds=m1, attention_mask=ctx["attn"]).logits.float()
            m0, _ = self._batch(ctx, [0.0])
            l0 = self.lm(inputs_embeds=m0, attention_mask=ctx["attn"]).logits.float()
            l0b = self.lm(inputs_embeds=m0, attention_mask=ctx["attn"]).logits.float()
        res["max_abs_dlogit_repeat_0_vs_0"] = float((l0b - l0).abs().max())   # noise floor
        res["max_abs_dlogit_gamma1_vs_0"] = float((l1 - l0).abs().max())
        res["passed_dlogit_gt_0"] = res["max_abs_dlogit_gamma1_vs_0"] > 0
        if prompt_kind == "sugarcrepe":
            inp = ctx["inp"]
            labels = caption_labels(self.processor, inp["input_ids"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_repo = self.wrapper.score(pixel_values=inp["pixel_values"],
                                            input_ids=inp["input_ids"],
                                            attention_mask=inp["attention_mask"],
                                            boxes=self.grid, box_mask=self.grid_mask,
                                            labels=labels)
            s_mine = self.score(ctx, [1.0])[0]
            res["score_repo_wrapper"] = float(s_repo)
            res["score_mine_at_1"] = float(s_mine)
            res["score_abs_diff"] = abs(float(s_repo) - float(s_mine))
        res["prompt_kind"] = prompt_kind
        return res
