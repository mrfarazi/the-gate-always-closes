#!/usr/bin/env python3
"""J4-InternVL: forced-gamma x eps knockout on the InternVL3-8B closure checkpoints.

Variant of scripts/knockout/internvl_knockout.py (functions imported) that
adds (a) an eps patch on every text-stack RMSNorm (Qwen2RMSNorm.variance_epsilon; modules
whose name contains 'vision' are excluded) and (b) a batched forced-gamma grid: the
pre-gate prefix (gamma = 1) is computed once per item in fp32 as pref / tanh(gate)
(same arithmetic as the original gamma_override) and all gammas of an item run in
chunked batched LM calls with logits computed only at the answer positions.
Teacher-forced answer cross-entropy, n = 300, paired bootstrap of dCE vs gamma = 0.

  backbone env:  python scripts/knockout/j4_internvl_eps.py --ckpt internvl3_seed0 \
      --eps native --n 300 --out runs/knockouts/j4_internvl_seed0_eps_native.json [--smoke 4]
"""
import argparse
import math
import os
import sys
import time
import traceback

from hra.knockout.common import setup_env  # noqa: E402
setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hra.knockout.common import (ACCV_BASE, atomic_write_json, cmdline, ensure_baseline_first, log_line,  # noqa: E402
                    native_rmsnorm_eps, now, paired_bootstrap, parse_values, patch_rmsnorm_eps,
                    skip_if_exists)

from hra import hub  # noqa: E402
from hra.paths import LOCAL_FILES_ONLY  # noqa: E402

DEFAULT_GAMMAS = "0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1,2,-1e-3,-1e-2"


@torch.no_grad()
def item_context(wrapper, px, ids, am, bx, bm, labels):
    """Everything that does not depend on gamma: spliced template, labels, base prefix."""
    model = wrapper.model
    ie = model.get_input_embeddings()(ids)
    img_feat = model.get_image_features(pixel_values=px)
    if isinstance(img_feat, (list, tuple)):
        img_feat = img_feat[0]
    mask = ids == wrapper.image_token_id
    ie = ie.clone()
    ie[mask] = img_feat.reshape(-1, ie.shape[-1]).to(ie.dtype)
    roi = wrapper._roi(px, bx, bm)
    pref, _, _, _ = wrapper.prefix_graph(roi, bx.float(), bm.float(), ref_embeds=ie.float())
    g = wrapper.prefix_graph.prefix_proj
    base = pref.float() / (torch.tanh(g.gate).float() + 1e-8) if not g.ungated else pref.float()
    B, S, D = ie.shape
    assert B == 1
    K = pref.shape[1]
    pos = mask[0].nonzero(as_tuple=True)[0]
    cut = int(pos[-1]) + 1 if len(pos) else 0
    tmpl = torch.zeros(1, S + K, D, device=ie.device, dtype=ie.dtype)
    tmpl[0, :cut] = ie[0, :cut]
    tmpl[0, cut + K:] = ie[0, cut:]
    out_a = torch.zeros(1, S + K, device=ie.device, dtype=am.dtype)
    out_a[0, :cut] = am[0, :cut]
    out_a[0, cut:cut + K] = 1
    out_a[0, cut + K:] = am[0, cut:]
    out_l = torch.full((1, S + K), -100, device=ie.device, dtype=labels.dtype)
    out_l[0, :cut] = labels[0, :cut]
    out_l[0, cut + K:] = labels[0, cut:]
    sel = out_l[0, 1:] != -100                 # positions whose next token is scored
    tgt = out_l[0, 1:][sel]
    text = torch.cat([ie[0, :cut][~mask[0, :cut]], ie[0, cut:]], dim=0)  # text tokens only
    return dict(tmpl=tmpl, attn=out_a, sel=sel, tgt=tgt, base=base, cut=cut, K=K, text=text)


@torch.no_grad()
def eval_gammas(wrapper, ctx, gammas, chunk=4, keep_logits=False):
    model = wrapper.model
    ces, kept = [], []
    for c0 in range(0, len(gammas), chunk):
        gs = gammas[c0:c0 + chunk]
        G = len(gs)
        e = ctx["tmpl"].expand(G, -1, -1).clone()
        for i, gm in enumerate(gs):
            e[i, ctx["cut"]:ctx["cut"] + ctx["K"]] = (ctx["base"][0] * float(gm)).to(e.dtype)
        a = ctx["attn"].expand(G, -1)
        out = model.model.language_model(inputs_embeds=e, attention_mask=a)
        h = out.last_hidden_state[:, :-1][:, ctx["sel"]]          # (G, n_ans, D)
        logits = model.lm_head(h).float()
        tgt = ctx["tgt"].unsqueeze(0).expand(G, -1)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                             reduction="none").view(G, -1).mean(1)
        ces += ce.tolist()
        if keep_logits:
            kept.append(logits)
    return ces, (torch.cat(kept, 0) if keep_logits else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir with prefix_modules.pt and lora/ (rel. to ACCV base ok)")
    ap.add_argument("--backbone", default="internvl3-8b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gammas", default=DEFAULT_GAMMAS)
    ap.add_argument("--eps", default="native", help="float or 'native' (1e-6 for Qwen2.5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()
    script = "J4-InternVL"
    if skip_if_exists(args.out, script):
        return
    n = args.smoke if args.smoke else args.n
    ckpt = hub.resolve(args.ckpt)
    t_start = time.time()
    log_line(f"{script} START | ckpt={ckpt} n={n} eps={args.eps} out={args.out} | cmd: {cmdline()}")
    try:
        from hra.backbones.internvl import (BACKBONES, GQARelDataset, make_collate,
                                            InternVLGatedPrefix, build_labels)
        import transformers
        from transformers import AutoProcessor
        from peft import PeftModel
        from hra.train_prefix_fixed import FixedPrefixGraph
        from torch.utils.data import DataLoader

        spec = BACKBONES[args.backbone]
        dev = torch.device("cuda")
        processor = AutoProcessor.from_pretrained(spec["model_id"], local_files_only=LOCAL_FILES_ONLY)
        model = getattr(transformers, spec["cls"]).from_pretrained(
            spec["model_id"], dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
        model.requires_grad_(False)
        lora = os.path.join(ckpt, "lora")
        if os.path.exists(lora):
            model.model.language_model = PeftModel.from_pretrained(model.model.language_model, lora)
        pg = FixedPrefixGraph(vis_dim=spec["vis_dim"], k=4, prefix_dropout=0.0, no_rms=True, ungated=False)
        pg.prefix_proj.llm_dim = spec["llm_dim"]
        pg.prefix_proj.proj = torch.nn.Linear(pg.embed_dim, 4 * spec["llm_dim"])
        st = torch.load(os.path.join(ckpt, "prefix_modules.pt"), map_location="cpu")
        pg.load_state_dict(st["prefix_graph"], strict=True)
        learned_raw = float(pg.prefix_proj.gate)
        learned = math.tanh(learned_raw)
        wrapper = InternVLGatedPrefix(model, processor, pg, vision_attr=spec["vision_attr"]).to(dev).eval()
        print(f"[{script}] {spec['nice']} learned gate tanh={learned:+.6f} (raw {learned_raw:+.6f})", flush=True)

        gammas = ensure_baseline_first(parse_values(args.gammas, learned=learned), 0.0)
        # eps patch: text stack only (exclude 'vision'); count includes final norm
        lm_root = model.model.language_model
        native = native_rmsnorm_eps(lm_root)
        eps_val = native if args.eps == "native" else float(args.eps)
        n_patched, restore, natives = patch_rmsnorm_eps(model, eps_val, exclude_substrings=("vision",))
        print(f"[{script}] eps={eps_val:g} (native {native:g}); patched {n_patched} RMSNorm modules "
              f"(text stack, 'vision' excluded); gammas={gammas}", flush=True)

        ds = GQARelDataset(processor, limit=n * 3)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_collate(processor))
        per = {g: [] for g in gammas}
        ids_used = []
        norms = {"prefix_rms": [], "prefix_l2": [], "text_rms": [], "text_l2": []}
        unit = None
        t0 = time.time()
        k = 0
        try:
            for bi, batch in enumerate(dl):
                if k >= n:
                    break
                px = batch["pixel_values"].to(dev, dtype=torch.bfloat16)
                ids = batch["input_ids"].to(dev)
                am = batch["attention_mask"].to(dev)
                bx = batch["boxes"].to(dev)
                bm = batch["box_mask"].to(dev)
                lb = build_labels(batch["input_ids"], batch["answers"], processor.tokenizer).to(dev)
                if (lb != -100).sum() == 0:
                    continue
                try:
                    ctx = item_context(wrapper, px, ids, am, bx, bm, lb)
                    if unit is None:
                        ce01, lg = eval_gammas(wrapper, ctx, [0.0, 1.0], chunk=2, keep_logits=True)
                        d = float((lg[1] - lg[0]).abs().max())
                        unit = {"max_abs_dlogit_gamma1_vs_0": d, "passed_dlogit_gt_0": d > 0,
                                "n_patched_gt_0": n_patched > 0, "ce_gamma0": ce01[0], "ce_gamma1": ce01[1],
                                "n_answer_tokens_item0": int(ctx["sel"].sum())}
                        print(f"[{script}] unit_check: {unit}", flush=True)
                    ces, _ = eval_gammas(wrapper, ctx, gammas, chunk=args.chunk)
                    if not all(np.isfinite(ces)):
                        continue
                    for g, c in zip(gammas, ces):
                        per[g].append(c)
                    if len(norms["prefix_rms"]) < 64:
                        b = ctx["base"][0].float()
                        norms["prefix_rms"].append(float(b.pow(2).mean(-1).sqrt().mean()))
                        norms["prefix_l2"].append(float(b.norm(dim=-1).mean()))
                        t = ctx["text"].float()
                        norms["text_rms"].append(float(t.pow(2).mean(-1).sqrt().median()))
                        norms["text_l2"].append(float(t.norm(dim=-1).median()))
                    ids_used.append(ds.qids[bi])
                    k += 1
                    if k % 100 == 0 or k == n:
                        el = time.time() - t0
                        print(f"[{script}] {k}/{n} items {el:.0f}s {k / max(el, 1e-9):.2f} items/s "
                              f"CE@0={np.mean(per[0.0]):.4f}", flush=True)
                except Exception as ex:
                    print(f"  [warn] {type(ex).__name__}: {ex}", flush=True)
                    traceback.print_exc()
                    continue
        finally:
            restore()
        elapsed = time.time() - t0
        base_ce = np.array(per[0.0])
        rows = []
        for g in gammas:
            b = np.array(per[g])
            d, lo, hi = paired_bootstrap(b, base_ce, n_boot=5000, seed=0)
            rows.append({"gamma": g, "ce": float(b.mean()), "d_ce_vs_0": d, "ci95": [lo, hi],
                         "significant": bool(lo > 0 or hi < 0)})
        prefix_rms = float(np.median(norms["prefix_rms"])) if norms["prefix_rms"] else float("nan")
        res = {"script": script, "cmd": cmdline(), "started": now(), "arm": "J4-InternVL",
               "backbone": spec["nice"], "model_id": spec["model_id"], "ckpt": ckpt,
               "learned_gate": learned, "learned_gate_raw": learned_raw, "eps": eps_val,
               "eps_native": native, "eps_native_values_seen": natives, "n_patched": n_patched,
               "bench": "gqa_rel_train_teacher_forced_ce", "n": k, "n_requested": n,
               "smoke": bool(args.smoke), "values": gammas, "value_key": "gamma",
               "gamma_semantics": "prefix = (pref / tanh(gate)) * gamma, fp32 base then cast to bf16",
               "unit_check": unit, "rows": rows,
               "token_norms": {"prefix_rms_at_gamma1": prefix_rms,
                               "prefix_l2_at_gamma1": float(np.median(norms["prefix_l2"])) if norms["prefix_l2"] else None,
                               "text_rms_median": float(np.median(norms["text_rms"])) if norms["text_rms"] else None,
                               "text_l2_median": float(np.median(norms["text_l2"])) if norms["text_l2"] else None,
                               "pred_gamma_c": math.sqrt(eps_val) / max(prefix_rms, 1e-12),
                               "pred_gamma_c_0p4": 0.4 * math.sqrt(eps_val) / max(prefix_rms, 1e-12),
                               "n_items_measured": len(norms["prefix_rms"])},
               "bootstrap": {"n_resamples": 5000, "seed": 0, "paired_vs": 0.0,
                             "note": "original internvl_knockout.py used 10000 resamples"},
               "per_item": {"ids": ids_used, "ce": {str(g): per[g] for g in gammas}},
               "timing": {"runtime_sec": elapsed, "items_per_sec": k / max(elapsed, 1e-9), "n_items": k},
               "finished": now(), "wall_sec": time.time() - t_start}
        atomic_write_json(args.out, res)
        print(f"[{script}] DONE n={k}  " + "  ".join(f"g={r['gamma']:g}:dCE={r['d_ce_vs_0']:+.4f}" for r in rows), flush=True)
        log_line(f"{script} END | n={k} runtime={elapsed:.0f}s ({k / max(elapsed, 1e-9):.2f} items/s) out={args.out}")
    except Exception as ex:
        traceback.print_exc()
        log_line(f"{script} FAILED | {type(ex).__name__}: {ex} | cmd: {cmdline()}")
        raise


if __name__ == "__main__":
    main()
