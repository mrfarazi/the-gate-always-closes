#!/usr/bin/env python3
"""LLaVA-1.5-7B adapters for the knock-out and sweep scripts.

PrefixModel : gated-prefix checkpoints (FixedPrefixGraph / SoftPromptGraph /
              EuclideanPrefixGraph ...). The sweep value is the gate multiplier gamma.
HRAModel    : attention-bias checkpoints (HRAWrapper), incl. the Stage-2 checkpoint.
              The sweep value is a scalar multiplier on the (B,1,P,P) patch bias.

Forced gamma. The spec: set gate := atanh(gamma) (clipped to +-0.999999) and make sure
the gate is applied (ungated=False). `set_gate()` does literally that and is used for the
unit check. The batched sweep path computes the *pre-gate* prefix once by forcing
tanh(gate) == 1 exactly (gate := +inf, ungated=False) and multiplies by gamma itself; the
two are the same arithmetic (tokens * tanh(gate) vs tokens * gamma) and the unit check
records their max abs difference at gamma = 0.3.

Loading: build_prefix_graph_for_eval(variant) + FixedPrefixWrapper, LoRA via PeftModel; the
variant comes from the registry / variant.json (see hra.checkpoints).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

from hra.knockout.common import (ACCV_BASE, GQA_PROMPT, SC_PROMPT, caption_labels, grid_boxes)

from hra.paths import LLAVA_PATH as MODEL_PATH  # noqa: E402

# name -> spec: the checkpoint registry (hra.hub.REGISTRY). `variant` is the VARIANT_CONFIGS key
# used to rebuild the prefix graph; directories are resolved (and downloaded) by hra.hub.
from hra import hub  # noqa: E402

CKPTS = hub.REGISTRY


def ckpt_path(name):
    return hub.resolve(name)


def _load_llava():
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    vlm = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    vlm.requires_grad_(False)
    return vlm, processor


def _per_row_neg_ce(logits, labels):
    """-mean CE over label tokens per row (== -outputs.loss of a single-row HF call).
    logits (G, L, V) float, labels (G, L) with -100 ignored; every row has the same
    number of scored tokens."""
    lg = logits[:, :-1]
    lb = labels[:, 1:]
    m = lb != -100
    n_tok = int(m[0].sum())
    ce = F.cross_entropy(lg[m], lb[m], reduction="none").view(logits.shape[0], n_tok)
    return (-ce.mean(dim=1)).tolist()


class _LlavaBase:
    """Common pieces: processor, prompts, decoding."""
    max_new_tokens = 20

    def _inputs(self, image, prompt):
        inp = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        return inp

    def _decode(self, gen_ids):
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen_ids]

    def _lm_generate(self, merged, attn):
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        return self.lm.generate(inputs_embeds=merged, attention_mask=attn,
                                pad_token_id=pad, max_new_tokens=self.max_new_tokens,
                                do_sample=False)

    @staticmethod
    def _token_norms(x):
        """x (..., D) -> (rms per token, l2 per token) as flat float tensors."""
        x = x.float()
        rms = x.pow(2).mean(dim=-1).sqrt().flatten()
        l2 = x.norm(dim=-1).flatten()
        return rms, l2


# =============================================================================== prefix
class PrefixModel(_LlavaBase):
    value_name = "gamma"

    def __init__(self, name, device="cuda"):
        from peft import PeftModel
        from hra.train_prefix_fixed import build_prefix_graph_for_eval, FixedPrefixWrapper
        spec = CKPTS[name]
        assert spec["kind"] == "prefix", name
        self.name, self.spec = name, spec
        self.device = torch.device(device)
        ckpt = ckpt_path(name)
        vlm, processor = _load_llava()
        lora = os.path.join(ckpt, "lora")
        if os.path.exists(lora):
            vlm.language_model = PeftModel.from_pretrained(vlm.language_model, lora)
        pg = build_prefix_graph_for_eval(spec["variant"])
        st = torch.load(os.path.join(ckpt, "prefix_modules.pt"), map_location="cpu")
        pg.load_state_dict(st["prefix_graph"], strict=True)
        # gate lives on .prefix_proj (FixedPrefixGraph/EuclideanGeo) or on the graph itself
        self.gate_owner = pg.prefix_proj if (hasattr(pg, "prefix_proj")
                                             and hasattr(pg.prefix_proj, "gate")) else \
            (pg if hasattr(pg, "gate") else None)
        assert self.gate_owner is not None, "no gate parameter found"
        self.learned_gate_raw = float(self.gate_owner.gate.item())
        self.learned_gate = math.tanh(self.learned_gate_raw)
        self.trained_ungated = bool(getattr(self.gate_owner, "ungated", False))
        self.has_ungated_flag = hasattr(self.gate_owner, "ungated")
        self.wrapper = FixedPrefixWrapper(vlm, processor, pg).to(self.device).eval()
        self.vlm = vlm
        self.lm = vlm.language_model
        self.processor = processor
        self.tok = processor.tokenizer
        self.embed = self.wrapper._get_embed_fn()
        self.img_tok = vlm.config.image_token_index
        self.boxes = torch.tensor(grid_boxes(6), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.bmask = torch.ones(1, self.boxes.shape[1], device=self.device)
        self._orig_gate = self.gate_owner.gate.data.clone()
        self._orig_ungated = self.trained_ungated

    # ---- gate control -------------------------------------------------------------
    def set_gate(self, gamma):
        """Literal forced gate: gate := atanh(clip(gamma)), gate applied."""
        g = max(-0.999999, min(0.999999, float(gamma)))
        self.gate_owner.gate.data.fill_(math.atanh(g))
        if self.has_ungated_flag:
            self.gate_owner.ungated = False

    def set_gate_unity(self):
        """tanh(+inf) == 1.0 exactly -> the projector returns the pre-gate tokens."""
        self.gate_owner.gate.data.fill_(float("inf"))
        if self.has_ungated_flag:
            self.gate_owner.ungated = False

    def restore_gate(self):
        self.gate_owner.gate.data.copy_(self._orig_gate)
        if self.has_ungated_flag:
            self.gate_owner.ungated = self._orig_ungated

    # ---- feature prep -------------------------------------------------------------
    @torch.no_grad()
    def _vision(self, pixel_values):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw_vision, imf = self.wrapper._get_vision_features(pixel_values)
            feats = self.wrapper._extract_roi_features(raw_vision, self.boxes, self.bmask)
        return imf, feats

    @torch.no_grad()
    def _prefix(self, feats, ie):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pref, _, _, _ = self.wrapper.prefix_graph(feats, self.boxes, self.bmask, ref_embeds=ie)
        return pref

    @torch.no_grad()
    def _template(self, inp, imf, base_prefix, labels=None):
        """Merged sequence with the prefix slot zeroed + attn mask (+ expanded labels).
        Same layout as FixedPrefixWrapper.forward: [text<img, 576 patches, K prefix, text]"""
        ids = inp["input_ids"]
        ie = self.embed(ids)
        imf1 = imf[0] if imf.dim() == 3 else imf
        pos = int((ids[0] == self.img_tok).nonzero(as_tuple=True)[0][0])
        npatch, K = imf1.shape[0], base_prefix.shape[1]
        S = ids.shape[1]
        L = S - 1 + npatch + K
        merged = torch.zeros(1, L, ie.shape[-1], device=self.device, dtype=ie.dtype)
        merged[0, :pos] = ie[0, :pos]
        merged[0, pos:pos + npatch] = imf1.to(ie.dtype)
        ps = pos + npatch
        merged[0, ps + K:] = ie[0, pos + 1:]
        attn = torch.ones(1, L, device=self.device, dtype=inp["attention_mask"].dtype)
        lab = None
        if labels is not None:
            lab = torch.full((1, L), -100, dtype=labels.dtype, device=self.device)
            lab[0, :pos] = labels[0, :pos]
            lab[0, ps + K:] = labels[0, pos + 1:]
        return dict(merged=merged, attn=attn, labels=lab, ps=ps, K=K, ie=ie, pos=pos)

    def _prepare(self, image, prompt, labels_fn=None, reuse=None):
        inp = self._inputs(image, prompt)
        if reuse is not None and "imf" in reuse:
            imf, feats = reuse["imf"], reuse["feats"]
        else:
            imf, feats = self._vision(inp["pixel_values"])
        ie = self.embed(inp["input_ids"])
        self.set_gate_unity()
        try:
            base = self._prefix(feats, ie)            # (1, K, D), pre-gate (gamma = 1)
        finally:
            self.restore_gate()
        labels = labels_fn(inp) if labels_fn else None
        ctx = self._template(inp, imf, base, labels)
        ctx.update(inp=inp, imf=imf, feats=feats, base=base)
        return ctx

    def prepare_sc(self, image, caption, reuse=None):
        return self._prepare(image, SC_PROMPT.format(caption=caption),
                             labels_fn=lambda inp: caption_labels(self.processor, inp["input_ids"]),
                             reuse=reuse)

    def prepare_gqa(self, image, question):
        return self._prepare(image, GQA_PROMPT.format(q=question))

    # ---- batched evaluation -------------------------------------------------------
    def _batch(self, ctx, values):
        G = len(values)
        merged = ctx["merged"].expand(G, -1, -1).clone()
        ps, K = ctx["ps"], ctx["K"]
        base = ctx["base"][0]
        for g, v in enumerate(values):
            merged[g, ps:ps + K] = (base * float(v)).to(merged.dtype)
        attn = ctx["attn"].expand(G, -1)
        return merged, attn

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
        for c0 in range(0, len(values), chunk):
            vals = values[c0:c0 + chunk]
            merged, attn = self._batch(ctx, vals)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = self._lm_generate(merged, attn)
            out += self._decode(gen)
        return out

    @torch.no_grad()
    def logits_at(self, ctx, value):
        merged, attn = self._batch(ctx, [value])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.lm(inputs_embeds=merged, attention_mask=attn).logits.float()

    # ---- prefix slot removed (published LoRA-only / prefix-dropped condition) ---------
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = self._lm_generate(m, a)
        return self._decode(gen)[0]

    def norms(self, ctx):
        prms, pl2 = self._token_norms(ctx["base"][0])
        ie = ctx["ie"][0]
        pos = ctx["pos"]
        text = torch.cat([ie[:pos], ie[pos + 1:]], dim=0)   # drop the <image> placeholder
        trms, tl2 = self._token_norms(text)
        return {"prefix_rms": float(prms.mean()), "prefix_l2": float(pl2.mean()),
                "prefix_rms_per_token": prms.tolist(), "prefix_l2_per_token": pl2.tolist(),
                "text_rms_median": float(trms.median()), "text_l2_median": float(tl2.median())}

    # ---- unit check -----------------------------------------------------------------
    @torch.no_grad()
    def unit_check(self, image, prompt_kind="gqa", question="What is this?", caption="A photo."):
        """(1) literal forced gate gamma=1 vs 0 changes the logits; (2) base*gamma equals
        the literal gate path; (3) my batched score equals wrapper.score at the learned gate."""
        res = {}
        if prompt_kind == "sugarcrepe":
            prompt = SC_PROMPT.format(caption=caption)
            ctx = self.prepare_sc(image, caption)
        else:
            prompt = GQA_PROMPT.format(q=question)
            ctx = self.prepare_gqa(image, question)
        inp = ctx["inp"]
        ie = ctx["ie"]

        def logits_with_gate(gamma):
            self.set_gate(gamma)
            try:
                pref = self._prefix(ctx["feats"], ie)
            finally:
                self.restore_gate()
            merged = ctx["merged"].clone()
            merged[0, ctx["ps"]:ctx["ps"] + ctx["K"]] = pref[0].to(merged.dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.lm(inputs_embeds=merged, attention_mask=ctx["attn"]).logits.float(), pref

        l1, _ = logits_with_gate(1.0)
        l0, _ = logits_with_gate(0.0)
        l0b, _ = logits_with_gate(0.0)
        res["max_abs_dlogit_repeat_0_vs_0"] = float((l0b - l0).abs().max())   # noise floor
        res["max_abs_dlogit_gamma1_vs_0"] = float((l1 - l0).abs().max())
        res["max_abs_dlogit_lastpos_gamma1_vs_0"] = float((l1[0, -1] - l0[0, -1]).abs().max())
        res["passed_dlogit_gt_0"] = res["max_abs_dlogit_gamma1_vs_0"] > 0
        # literal-gate vs base*gamma consistency at gamma = 0.3
        _, p_lit = logits_with_gate(0.3)
        p_mul = (ctx["base"] * 0.3)
        res["prefix_consistency_max_abs_diff_gamma0p3"] = float((p_lit.float() - p_mul.float()).abs().max())
        res["prefix_abs_max_gamma0p3"] = float(p_lit.float().abs().max())
        # score consistency vs the repo's wrapper.score at the learned gate (SC only)
        if prompt_kind == "sugarcrepe":
            labels = caption_labels(self.processor, inp["input_ids"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_repo = self.wrapper.score(pixel_values=inp["pixel_values"],
                                            input_ids=inp["input_ids"],
                                            attention_mask=inp["attention_mask"],
                                            boxes=self.boxes, box_mask=self.bmask,
                                            labels=labels)
            g_eff = 1.0 if self.trained_ungated else self.learned_gate
            s_mine = self.score(ctx, [g_eff])[0]
            res["score_repo_wrapper"] = float(s_repo)
            res["score_mine_at_learned"] = float(s_mine)
            res["score_abs_diff"] = abs(float(s_repo) - float(s_mine))
        res["prompt_kind"] = prompt_kind
        return res


# =============================================================================== HRA
class HRAModel(_LlavaBase):
    """Attention-bias checkpoints. value = scalar multiplier on the patch bias
    (0 = bias off, 1 = learned bias on)."""
    value_name = "bias_mult"

    def __init__(self, name, device="cuda"):
        from hra.model import HRAWrapper
        spec = CKPTS[name]
        assert spec["kind"] in ("hra", "hra_stage2"), name
        self.name, self.spec = name, spec
        self.device = torch.device(device)
        self.wrapper, self.processor = HRAWrapper.load_for_eval(ckpt_path(name), self.device)
        self.vlm = self.wrapper.vlm
        self.lm = self.vlm.language_model
        self.tok = self.processor.tokenizer
        self.embed = self.wrapper._get_embed_fn()
        self.gate_mult = 1.0
        self.stage2 = None
        if spec["kind"] == "hra_stage2":
            s2 = torch.load(hub.stage2_path(name), map_location="cpu")
            self.wrapper.hra_graph.cone_attn.bias_scale.data.fill_(float(s2["bias_scale"]))
            hw = float(torch.sigmoid(s2["head_gates"]).mean())
            lw = float(torch.sigmoid(s2["layer_gates"]).mean())
            self.gate_mult = hw * lw                      # Stage2Wrapper.get_gated_bias
            self.stage2 = {"beta": float(s2["bias_scale"]), "head_gate_mean": hw,
                           "layer_gate_mean": lw, "effective_gate_mult": hw * lw}
        self.beta = float(self.wrapper.hra_graph.cone_attn.bias_scale.item())
        self.softplus_beta = math.log1p(math.exp(self.beta))
        self.learned_gate = self.beta            # reported as 'learned_gate' in the JSON
        self.learned_gate_raw = self.beta
        self.trained_ungated = False
        self.boxes = torch.tensor(grid_boxes(6), dtype=torch.float32).unsqueeze(0).to(self.device)
        self.bmask = torch.ones(1, self.boxes.shape[1], device=self.device)

    @torch.no_grad()
    def _bias(self, pixel_values, boxes=None, bmask=None):
        boxes = self.boxes if boxes is None else boxes
        bmask = self.bmask if bmask is None else bmask
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw_vision, imf = self.wrapper._get_vision_features(pixel_values)
            feats = self.wrapper._extract_roi_features(raw_vision, boxes, bmask)
            patch_bias, _, _, _ = self.wrapper.hra_graph(feats, boxes, bmask)
        return imf, patch_bias.unsqueeze(1) * self.gate_mult      # (1,1,P,P)

    def _prepare(self, image, prompt, labels_fn=None, reuse=None):
        inp = self._inputs(image, prompt)
        if reuse is not None and "imf" in reuse:
            imf, bias = reuse["imf"], reuse["bias"]
        else:
            imf, bias = self._bias(inp["pixel_values"])
        ie = self.embed(inp["input_ids"])
        merged, attn, img_start = self.wrapper._merge_with_vision(
            ie, imf.to(ie.dtype), inp["input_ids"], inp["attention_mask"])
        npatch = imf.shape[1]
        labels = None
        if labels_fn:
            labels = self.wrapper._expand_labels(labels_fn(inp), inp["input_ids"], npatch)
        return dict(inp=inp, imf=imf, bias=bias, ie=ie, merged=merged, attn=attn,
                    labels=labels, img_range=(img_start, img_start + npatch),
                    pos=img_start, npatch=npatch)

    def prepare_sc(self, image, caption, reuse=None):
        return self._prepare(image, SC_PROMPT.format(caption=caption),
                             labels_fn=lambda inp: caption_labels(self.processor, inp["input_ids"]),
                             reuse=reuse)

    def prepare_gqa(self, image, question):
        return self._prepare(image, GQA_PROMPT.format(q=question))

    def _with_bias(self, ctx, values, fn):
        G = len(values)
        merged = ctx["merged"].expand(G, -1, -1).contiguous()
        attn = ctx["attn"].expand(G, -1).contiguous()
        bias = torch.cat([ctx["bias"] * float(v) for v in values], dim=0)
        self.wrapper._hra_bias = bias
        self.wrapper._image_range = ctx["img_range"]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return fn(merged, attn)
        finally:
            self.wrapper._hra_bias = None
            self.wrapper._image_range = None

    @torch.no_grad()
    def score(self, ctx, values, chunk=16):
        out = []
        for c0 in range(0, len(values), chunk):
            vals = values[c0:c0 + chunk]
            labels = ctx["labels"].expand(len(vals), -1)
            o = self._with_bias(ctx, vals, lambda m, a: self.lm(inputs_embeds=m, attention_mask=a))
            out += _per_row_neg_ce(o.logits.float(), labels)
        return out

    @torch.no_grad()
    def generate(self, ctx, values, chunk=16):
        out = []
        for c0 in range(0, len(values), chunk):
            vals = values[c0:c0 + chunk]
            gen = self._with_bias(ctx, vals, lambda m, a: self._lm_generate(m, a))
            out += self._decode(gen)
        return out

    def norms(self, ctx):
        b = ctx["bias"].float()
        ie = ctx["ie"][0]
        pos = ctx["pos"]
        text = torch.cat([ie[:pos], ie[pos + 1:]], dim=0)
        trms, tl2 = self._token_norms(text)
        return {"prefix_rms": float(b.abs().mean()), "prefix_l2": float(b.abs().max()),
                "note": "for HRA 'prefix_rms' = mean|bias|, 'prefix_l2' = max|bias| (logit units)",
                "text_rms_median": float(trms.median()), "text_l2_median": float(tl2.median())}

    @torch.no_grad()
    def unit_check(self, image, prompt_kind="gqa", question="What is this?", caption="A photo."):
        res = {}
        if prompt_kind == "sugarcrepe":
            ctx = self.prepare_sc(image, caption)
        else:
            ctx = self.prepare_gqa(image, question)
        fwd = lambda m, a: self.lm(inputs_embeds=m, attention_mask=a)  # noqa: E731
        o1 = self._with_bias(ctx, [1.0], fwd).logits.float()
        o0 = self._with_bias(ctx, [0.0], fwd).logits.float()
        o0b = self._with_bias(ctx, [0.0], fwd).logits.float()
        o3 = self._with_bias(ctx, [3.0], fwd).logits.float()
        res["max_abs_dlogit_repeat_0_vs_0"] = float((o0b - o0).abs().max())   # noise floor
        res["max_abs_dlogit_gamma1_vs_0"] = float((o1 - o0).abs().max())
        res["max_abs_dlogit_mult3_vs_0"] = float((o3 - o0).abs().max())
        res["passed_dlogit_gt_0"] = res["max_abs_dlogit_gamma1_vs_0"] > 0
        b = ctx["bias"].float()
        res["bias_abs_mean_grid_boxes"] = float(b.abs().mean())
        res["bias_abs_max_grid_boxes"] = float(b.abs().max())
        # Published testdev protocol (eval_gqa_parallel.py): scene-graph boxes, and no
        # testdev image has a scene graph -> all-zero boxes/mask. Measure what that gives.
        zb = torch.zeros_like(self.boxes)
        zm = torch.zeros_like(self.bmask)
        _, bz = self._bias(ctx["inp"]["pixel_values"], zb, zm)
        res["bias_abs_max_zero_boxes_protocol"] = float(bz.float().abs().max())
        # score consistency vs wrapper.score (SC)
        if prompt_kind == "sugarcrepe":
            inp = ctx["inp"]
            labels = caption_labels(self.processor, inp["input_ids"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_repo = self.wrapper.score(pixel_values=inp["pixel_values"],
                                            input_ids=inp["input_ids"],
                                            attention_mask=inp["attention_mask"],
                                            boxes=self.boxes, box_mask=self.bmask,
                                            labels=labels)
            # wrapper.score uses the un-gated raw bias (no Stage-2 head/layer gates)
            s_mine = self.score(ctx, [1.0 / self.gate_mult if self.gate_mult else 1.0])[0]
            res["score_repo_wrapper"] = float(s_repo)
            res["score_mine_at_learned"] = float(s_mine)
            res["score_abs_diff"] = abs(float(s_repo) - float(s_mine))
        res["prompt_kind"] = prompt_kind
        return res


def load_model(name, device="cuda"):
    kind = CKPTS[name]["kind"]
    if kind == "prefix":
        return PrefixModel(name, device)
    return HRAModel(name, device)
