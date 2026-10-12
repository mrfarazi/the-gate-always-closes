#!/usr/bin/env python3
"""J6-LLaVA: layer-wise attention mass onto the injected prefix positions, measured under the
J7 knock-out protocol, so every LLaVA-1.5 point in the indicator-vs-truth figure has an
attention ratio from the same items, proposals and gate arithmetic as its knock-out.

  python j6_llava_attention.py --ckpt base_r1 --n 100 [--values landed,0,1] [--out <json>]
  python j6_llava_attention.py --ckpt base_r1 --n 4 --smoke            # smoke -> results/.../smoke/

Reuses hra.knockout.llava_gate_lib.PrefixModel (checkpoint registry, loaders, gate forcing, 6x6
grid proposals, merged-sequence template) and common.load_gqa_items (same item order as
j7_llava_knockout.py: testdev_balanced dict order, semantic == rel, first n). The measurement
follows scripts/probes/layerwise_attention.py: eager attention, single forward pass over the
prompt, per layer the mean over heads and text query positions after the prefix of the attention
mass on the K prefix positions, chance = K / L. Output: by_gamma[key] with all / all_ratio /
answer / answer_ratio per layer, chance_all, mean_layer_ratio_all, ...

Gate values. key 'landed' = the checkpoint's own tanh(gate) ('learned' in the J7 JSONs); '0' =
closed gate (four zero vectors keep the slot); '1' = pre-gate prefix (gate forced fully open).
Prefix = pre-gate tokens x value, exactly as PrefixModel._batch does for J4 / J7. For rms_ungated
(gate stored 0, trained with ungated=True) the value is the J5 prefix scale s, and
'landed' is s = 1 (prefix genuinely on), which coincides with key '1'.

Query sets. 'all' = every query position after the prefix (question, instruction, 'ASSISTANT:'),
positions before the prefix cannot attend to it (causal) and are excluded. 'answer' = the last
query position, whose logits generate the first answer token (the position J7's greedy decoding
starts from). chance_all = K / L per item (averaged over items); the causal chance mean_q K/(q+1)
is stored alongside as chance_all_causal / all_ratio_causal. For the last position both coincide.

Outputs go to runs/probes/j6/ by default (the shipped files are results/probes/j6/).
"""
import argparse
import datetime
import json
import math
import os
import sys
import time
import traceback

from hra.knockout.common import setup_env  # noqa: E402
setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hra.knockout.common import (ACCV_BASE, atomic_write_json, bootstrap_mean_ci, cmdline, gqa_image_path,  # noqa: E402
                    load_gqa_items, parse_values)
from hra.knockout import llava_gate_lib as LG  # noqa: E402

from hra.paths import RESULTS, RUNS  # noqa: E402

SCRIPT = "J6-LLaVA"
RESULTS_DIR = os.environ.get("HRA_J6_DIR", os.path.join(RUNS, "probes", "j6"))
REFERENCE_JSON = os.path.join(RESULTS, "probes", "layerwise_attention.json")   # read-only reference


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg, out_dir):
    """One timestamped line in <out_dir>/LOG.md."""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "LOG.md")
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("# J6-LLaVA log\n\nOne line per script start / end / failure.\n\n")
        f.write(f"- [{now()}] {msg}\n")


# --------------------------------------------------------------------------- eager loader
def _load_llava_eager():
    """PrefixModel's loader with attn_implementation='eager' (weights materialised for
    output_attentions=True), as scripts/layerwise_attention.py loads it."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(LG.MODEL_PATH)
    vlm = LlavaForConditionalGeneration.from_pretrained(LG.MODEL_PATH, torch_dtype=torch.bfloat16,
                                                        attn_implementation="eager")
    vlm.requires_grad_(False)
    return vlm, processor


def load_prefix_model_eager(name):
    orig = LG._load_llava
    LG._load_llava = _load_llava_eager
    try:
        model = LG.PrefixModel(name, "cuda")
    finally:
        LG._load_llava = orig
    # assert the text stack really is eager (class chosen at construction in transformers 4.40)
    kinds = {}
    for mod_name, m in model.lm.named_modules():
        if mod_name.endswith("self_attn") and "vision" not in mod_name:
            kinds[type(m).__name__] = kinds.get(type(m).__name__, 0) + 1
    if not kinds or any(k != "LlamaAttention" for k in kinds):
        raise RuntimeError(f"text self-attention modules are not eager LlamaAttention: {kinds}")
    return model, kinds


# --------------------------------------------------------------------------- value keys
def value_table(spec, learned_eff):
    """'landed,0,1' -> ordered [(key, value)], keys as in the Qwen J6 JSON ('landed' = own gate)."""
    table = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("landed", "learned"):
            table.append(("landed", float(learned_eff)))
        elif tok in ("-landed", "-learned"):
            table.append(("-landed", -float(learned_eff)))
        else:
            v = float(tok)
            table.append((f"{v:g}", v))
    return table


def unique_values(table):
    vals = []
    for _, v in table:
        if not any(abs(v - u) <= 1e-12 * max(1.0, abs(v)) for u in vals):
            vals.append(v)
    return vals


def value_index(vals, v):
    for i, u in enumerate(vals):
        if abs(v - u) <= 1e-12 * max(1.0, abs(v)):
            return i
    raise KeyError(v)


# --------------------------------------------------------------------------- measurement
@torch.no_grad()
def attention_forward(model, ctx, vals):
    """One batched forward over the prompt (rows = gate values) with attentions on.
    Returns (attentions tuple(L) of (G,H,Q,Q) bf16, logits (G,Q,V))."""
    merged, attn = model._batch(ctx, vals)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model.lm(inputs_embeds=merged, attention_mask=attn, output_attentions=True)
    return o.attentions, o.logits


def masses(atts, ps, K):
    """Per layer, per row: mean over heads and text queries (positions >= ps+K) of the mass on
    [ps, ps+K), and the same at the last query position. Returns (all (L,G), answer (L,G),
    row_sum_err, pre_prefix_leak)."""
    n_layers = len(atts)
    G = atts[0].shape[0]
    L = atts[0].shape[-1]
    all_m = np.zeros((n_layers, G))
    ans_m = np.zeros((n_layers, G))
    row_err = 0.0
    leak = 0.0
    for li, a in enumerate(atts):
        a = a.float()                                   # (G, H, Q, Q)
        sl = a[:, :, :, ps:ps + K].sum(-1)              # (G, H, Q) mass on the prefix per query
        all_m[li] = sl[:, :, ps + K:].mean(dim=(1, 2)).cpu().numpy()
        ans_m[li] = sl[:, :, L - 1].mean(dim=1).cpu().numpy()
        if li == 0:
            row_err = float((a[:, :, ps + K:, :].sum(-1) - 1.0).abs().max())
            leak = float(sl[:, :, :ps].abs().max())     # causal: queries before the prefix see none of it
    return all_m, ans_m, row_err, leak


def run(args):
    out_dir = RESULTS_DIR                      # LOG.md always in results/journal/j6_llava/ (this repo)
    if os.path.exists(args.out) and not args.force:
        print(f"[{SCRIPT}] SKIP | output exists: {args.out}", flush=True)
        log_line(f"{SCRIPT} SKIP | output exists: {args.out}", out_dir)
        return None
    t_start = time.time()
    log_line(f"{SCRIPT} START | ckpt={args.ckpt} n={args.n} values={args.values} out={args.out} | cmd: {cmdline()}", out_dir)
    spec = LG.CKPTS[args.ckpt]
    assert spec["kind"] == "prefix", f"{args.ckpt}: attention-bias designs have no prefix positions"
    result = {"script": SCRIPT, "arm": "J6-LLaVA", "probe": "attn", "tag": f"j6_llava_{args.ckpt}",
              "cmd": cmdline(), "started": now(), "smoke": bool(args.smoke),
              "ckpt": args.ckpt, "ckpt_path": LG.ckpt_path(args.ckpt), "kind": spec["kind"],
              "variant": spec.get("variant"), "model_path": LG.MODEL_PATH,
              "bench": "gqa_rel", "n_requested": args.n,
              "gqa_boxes": "6x6 grid", "gqa_items": "testdev dict order, semantic==rel filter, first n (as J7)",
              "attn_implementation": "eager", "output_attentions": True,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch": torch.__version__}
    try:
        import transformers
        result["transformers"] = transformers.__version__
        t0 = time.time()
        model, attn_kinds = load_prefix_model_eager(args.ckpt)
        result["load_sec"] = time.time() - t0
        result["attn_module_classes"] = attn_kinds
        result["gpu"] = torch.cuda.get_device_name(0)
        result["model_mem_gb"] = torch.cuda.memory_allocated() / 1e9
        result["learned_gate"] = model.learned_gate
        result["learned_gate_raw"] = model.learned_gate_raw
        result["trained_ungated"] = model.trained_ungated
        learned_eff = 1.0 if model.trained_ungated else model.learned_gate
        result["landed_gate_tanh"] = model.learned_gate          # J6 field (0.0 for rms_ungated)
        result["landed_value"] = learned_eff                    # value actually applied for 'landed'
        if model.trained_ungated:
            result["value_key"] = "scale"
            result["value_semantics"] = ("prefix = genuinely-on prefix x scale (J5 semantics); "
                                         "'landed' = scale 1 = key '1'")
        else:
            result["value_key"] = "gamma"
            result["value_semantics"] = "prefix = pre-gate tokens x gamma; 'landed' = tanh(learned gate); 0 = closed gate"
        table = value_table(args.values, learned_eff)
        vals = unique_values(table)
        result["gammas"] = [k for k, _ in table]
        result["gamma_values"] = {k: v for k, v in table}
        result["values_run"] = vals
        result["aliases"] = {"learned": "landed"}
        cfg = model.lm.config
        n_layers, n_heads = int(cfg.num_hidden_layers), int(cfg.num_attention_heads)
        result["n_layers"], result["n_heads"] = n_layers, n_heads
        print(f"[{SCRIPT}] {args.ckpt}: learned tanh gate {model.learned_gate:+.5f} (raw {model.learned_gate_raw:+.5f}), "
              f"trained_ungated={model.trained_ungated}; keys {result['gammas']} -> values {vals}; "
              f"attn modules {attn_kinds}; load {result['load_sec']:.0f}s", flush=True)

        items = load_gqa_items(ACCV_BASE, "gqa_rel", args.n)
        result["n_items_listed"] = len(items)
        result["item_ids"] = [it["id"] for it in items]

        # ---- unit check on item 0 (single-row forwards, attentions off; then the batched path)
        uc = {}
        img0 = Image.open(gqa_image_path(ACCV_BASE, items[0])).convert("RGB")
        ctx0 = model.prepare_gqa(img0, items[0]["question"])
        l0 = model.logits_at(ctx0, 0.0)
        l0b = model.logits_at(ctx0, 0.0)
        l1 = model.logits_at(ctx0, 1.0)
        uc["max_abs_dlogit_repeat_0_vs_0"] = float((l0b - l0).abs().max())
        uc["max_abs_dlogit_1_vs_0"] = float((l1 - l0).abs().max())
        uc["max_abs_dlogit_lastpos_1_vs_0"] = float((l1[0, -1] - l0[0, -1]).abs().max())
        if model.trained_ungated:
            uc["landed_vs_0"] = "skipped: rms_ungated stores gate 0 (landed = scale 1, see 1_vs_0)"
            uc["passed_dlogit_landed_gt_0"] = uc["max_abs_dlogit_1_vs_0"] > 0
        else:
            ll = model.logits_at(ctx0, learned_eff)
            uc["max_abs_dlogit_landed_vs_0"] = float((ll - l0).abs().max())
            uc["max_abs_dlogit_lastpos_landed_vs_0"] = float((ll[0, -1] - l0[0, -1]).abs().max())
            uc["passed_dlogit_landed_gt_0"] = uc["max_abs_dlogit_landed_vs_0"] > 0
        # batched (attentions on) vs single-row logits at value 0; attention rows sum to 1; causal leak 0
        atts, lg = attention_forward(model, ctx0, vals)
        i0 = value_index(vals, 0.0)
        # bf16 kernels are shape-dependent (batch 3 vs 1): logits differ at the ~0.5 level on a
        # few entries; what must agree is the attention mass, checked below on the same rows
        uc["max_abs_dlogit_batched_vs_single_at_0"] = float((lg[i0].float() - l0[0]).abs().max())
        uc["mean_abs_dlogit_batched_vs_single_at_0"] = float((lg[i0].float() - l0[0]).abs().mean())
        uc["lastpos_argmax_agree_batched_vs_single_at_0"] = bool(int(lg[i0, -1].float().argmax()) == int(l0[0, -1].argmax()))
        ps, K = ctx0["ps"], ctx0["K"]
        Lq = atts[0].shape[-1]
        uc["n_attention_layers"] = len(atts)
        uc["attention_shape_layer0"] = list(atts[0].shape)
        a_b, n_b, row_err, leak = masses(atts, ps, K)
        atts1, _ = attention_forward(model, ctx0, [0.0])            # single-row attentions at 0
        a_s, n_s, _, _ = masses(atts1, ps, K)
        del atts1
        uc["max_abs_dmass_layer_batched_vs_single_at_0"] = float(np.abs(a_b[:, i0] - a_s[:, 0]).max())
        uc["mean_layer_ratio_all_at_0_batched"] = float((a_b[:, i0] / (K / float(Lq))).mean())
        uc["mean_layer_ratio_all_at_0_single"] = float((a_s[:, 0] / (K / float(Lq))).mean())
        uc["row_sum_abs_err_layer0"] = row_err
        uc["pre_prefix_mass_on_prefix_layer0"] = leak
        uc["prefix_positions_item0"] = [ps, ps + K]
        uc["seq_len_item0"] = int(Lq)
        uc["chance_item0"] = K / float(Lq)
        uc["passed"] = bool(uc["passed_dlogit_landed_gt_0"] and len(atts) == n_layers and row_err < 1e-2 and leak < 1e-6
                            and uc["max_abs_dmass_layer_batched_vs_single_at_0"] < 5e-3)
        result["unit_check"] = uc
        print(f"[{SCRIPT}] unit_check: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                    for k, v in uc.items()), flush=True)
        if not uc["passed"]:
            raise RuntimeError(f"unit check failed: {uc}")
        del atts, lg, l0, l0b, l1
        torch.cuda.empty_cache()

        # ---- sweep
        G = len(vals)
        all_m = []      # per item: (n_layers, G)
        ans_m = []
        chance_l, chance_causal, seq_lens, ps_list, ids = [], [], [], [], []
        first_tok = []  # per item: [G] argmax token at the last position (first answer token)
        prefix_rms, prefix_l2, text_rms, text_l2 = [], [], [], []
        n_lm_calls = 0
        t0 = time.time()
        for i, it in enumerate(items):
            p = gqa_image_path(ACCV_BASE, it)
            if not os.path.exists(p):
                print(f"  [warn] missing image {p}; skipping item {it['id']}", flush=True)
                continue
            try:
                img = Image.open(p).convert("RGB")
                ctx = model.prepare_gqa(img, it["question"])
                ps, K = ctx["ps"], ctx["K"]
                am = np.zeros((n_layers, G))
                an = np.zeros((n_layers, G))
                ft = [None] * G
                for c0 in range(0, G, args.chunk):
                    sub = vals[c0:c0 + args.chunk]
                    atts, lg = attention_forward(model, ctx, sub)
                    n_lm_calls += 1
                    a_, n_, _, _ = masses(atts, ps, K)
                    am[:, c0:c0 + len(sub)] = a_
                    an[:, c0:c0 + len(sub)] = n_
                    tk = lg[:, -1].float().argmax(-1).tolist()
                    for j, t in enumerate(tk):
                        ft[c0 + j] = model.tok.decode([t])
                    Lq = atts[0].shape[-1]
                    del atts, lg
                q = np.arange(ps + K, Lq)
                all_m.append(am); ans_m.append(an)
                chance_l.append(K / float(Lq))
                chance_causal.append(float(np.mean(K / (q + 1.0))))
                seq_lens.append(int(Lq)); ps_list.append(int(ps)); ids.append(it["id"])
                first_tok.append(ft)
                if len(prefix_rms) < args.norm_items:
                    nm = model.norms(ctx)
                    prefix_rms.append(nm["prefix_rms"]); prefix_l2.append(nm["prefix_l2"])
                    text_rms.append(nm["text_rms_median"]); text_l2.append(nm["text_l2_median"])
            except Exception as ex:
                if isinstance(ex, torch.OutOfMemoryError) or "CUDA" in str(ex) or "device-side" in str(ex):
                    raise
                print(f"  [warn] item {it.get('id')}: {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()
                continue
            if (i + 1) % args.progress_every == 0 or (i + 1) == len(items):
                el = time.time() - t0
                mean_ratio = {k: float(np.mean([x[:, value_index(vals, v)].mean() / c for x, c in zip(all_m, chance_l)]))
                              for k, v in table}
                print(f"[{SCRIPT}] {i + 1}/{len(items)} items  {el:.0f}s  {el / max(len(ids), 1):.2f} s/item  "
                      f"mean-layer x chance: " + "  ".join(f"{k}={r:.2f}" for k, r in mean_ratio.items()), flush=True)
        elapsed = time.time() - t0
        n = len(ids)
        if n == 0:
            raise RuntimeError("no items were evaluated; refusing to write an empty result")

        # ---- aggregate (J6 schema)
        A = np.stack(all_m)                 # (n, layers, G)
        N = np.stack(ans_m)
        ch = np.asarray(chance_l)           # (n,)
        chc = np.asarray(chance_causal)
        by = {}
        per_item = {"ids": ids, "seq_len": seq_lens, "prefix_start": ps_list, "chance": [float(c) for c in ch],
                    "mean_layer_ratio_all": {}, "mean_layer_ratio_answer": {}, "first_token": {}}
        for k, v in table:
            g = value_index(vals, v)
            a = A[:, :, g]                  # (n, layers)
            an = N[:, :, g]
            r_all = a / ch[:, None]         # per-item ratio, K/L chance 
            r_ans = an / ch[:, None]
            r_causal = a / chc[:, None]     # Qwen J6 causal chance
            item_mean_all = r_all.mean(axis=1)      # (n,) mean over layers
            item_mean_ans = r_ans.mean(axis=1)
            lo, hi = bootstrap_mean_ci(item_mean_all, n_boot=5000, seed=0)
            lo_a, hi_a = bootstrap_mean_ci(item_mean_ans, n_boot=5000, seed=0)
            mean_mass = float(a.mean())
            agg = {"value": float(v),
                   "all": a.mean(axis=0).tolist(), "all_ratio": r_all.mean(axis=0).tolist(),
                   "answer": an.mean(axis=0).tolist(), "answer_ratio": r_ans.mean(axis=0).tolist(),
                   "all_ratio_causal": r_causal.mean(axis=0).tolist(),
                   "chance_all": float(ch.mean()), "chance_answer": float(ch.mean()),
                   "chance_all_causal": float(chc.mean()),
                   "max_layer_ratio_all": float(r_all.mean(axis=0).max()),
                   "mean_layer_ratio_all": float(r_all.mean()),
                   "mean_layer_ratio_all_ci95": [lo, hi],
                   "max_layer_ratio_answer": float(r_ans.mean(axis=0).max()),
                   "mean_layer_ratio_answer": float(r_ans.mean()),
                   "mean_layer_ratio_answer_ci95": [lo_a, hi_a],
                   "mean_layer_ratio_all_causal": float(r_causal.mean()),
                   # table columns: per-layer mass averaged over items, mean over layers, / mean chance
                   "mean_mass_all_over_layers": mean_mass,
                   "x_chance_mean_over_layers": mean_mass / float(ch.mean()),
                   "mass_L0_L8_L16_L24_L31": [float(a.mean(axis=0)[j]) for j in (0, 8, 16, 24, n_layers - 1)]}
            by[k] = agg
            per_item["mean_layer_ratio_all"][k] = item_mean_all.tolist()
            per_item["mean_layer_ratio_answer"][k] = item_mean_ans.tolist()
            per_item["first_token"][k] = [ft[g] for ft in first_tok]
        by["learned"] = by["landed"] if "landed" in by else None
        result["by_gamma"] = by
        result["n"] = n
        result["prefix_positions"] = {"K": int(K), "start_unique": sorted(set(ps_list)), "layout":
                                      "[text before <image> (5 tokens), 576 patches, K prefix, text after]",
                                      "note": "prefix occupies [start, start+K); text queries = positions >= start+K"}
        result["query_sets"] = {"all": "query positions >= prefix_end (text after the prefix; text-query set)",
                                "answer": "last query position L-1 (its logits generate the first answer token)"}
        result["chance_convention"] = ("chance_all = K / L per item, mean over items (layerwise_attention.py); "
                                       "all_ratio = mean over items of mass / (K/L); x_chance_mean_over_layers = "
                                       "mean over layers of item-mean mass / mean chance (RESULTS.md section 6 column); "
                                       "chance_all_causal = mean_q K/(q+1) (Qwen J6 convention)")
        result["per_item"] = per_item
        norms = {"n_items_measured": len(prefix_rms),
                 "prefix_rms_at_1": float(np.median(prefix_rms)) if prefix_rms else None,
                 "prefix_l2_at_1": float(np.median(prefix_l2)) if prefix_l2 else None,
                 "text_rms_median": float(np.median(text_rms)) if text_rms else None,
                 "text_l2_median": float(np.median(text_l2)) if text_l2 else None}
        if prefix_l2 and text_l2:
            norms["prefix_over_text_l2_ratio"] = norms["prefix_l2_at_1"] / max(norms["text_l2_median"], 1e-12)
        result["token_norms"] = norms
        result["timing"] = {"runtime_sec": elapsed, "n_items": n, "sec_per_item": elapsed / n,
                            "items_per_sec": n / max(elapsed, 1e-9), "n_lm_calls": n_lm_calls,
                            "chunk": args.chunk, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                            "load_sec": result["load_sec"]}
        # ---- the paper's original layer-wise probe for this checkpoint (read-only, for the comparison note)
        if os.path.exists(REFERENCE_JSON):
            try:
                reb = json.load(open(REFERENCE_JSON))
                ref = {}
                for r in reb:
                    if r.get("name") == args.ckpt:
                        m = r["mass_per_layer"]
                        ref[r["gate_mode"]] = {"n": r["n"], "chance": r["chance"], "mean_mass": float(np.mean(m)),
                                               "x_chance": float(np.mean(m) / r["chance"]),
                                               "mass_L0_L8_L16_L24_L31": [m[0], m[8], m[16], m[24], m[-1]]}
                result["reference_probe"] = {"src": REFERENCE_JSON, "gate_open_means": "gate := 3.0 (tanh 0.995)",
                                                "entries": ref} if ref else {"src": REFERENCE_JSON, "entries": {}}
            except Exception as ex:  # reference only
                result["reference_probe"] = {"src": REFERENCE_JSON, "error": repr(ex)}
        result["finished"] = now()
        result["wall_sec"] = time.time() - t_start
        atomic_write_json(args.out, result)
        summ = "  ".join(f"{k}: mean-layer x chance {by[k]['mean_layer_ratio_all']:.2f} (last pos {by[k]['mean_layer_ratio_answer']:.2f})"
                         for k, _ in table)
        print(f"[{SCRIPT}] DONE n={n} {elapsed / n:.2f} s/item peak {result['timing']['peak_mem_gb']:.1f} GB  {summ}", flush=True)
        log_line(f"{SCRIPT} END | ckpt={args.ckpt} n={n} runtime={elapsed:.0f}s ({elapsed / n:.2f} s/item, "
                 f"peak {result['timing']['peak_mem_gb']:.1f} GB) {summ} out={args.out}", out_dir)
        return result
    except Exception as ex:
        traceback.print_exc()
        log_line(f"{SCRIPT} FAILED | ckpt={args.ckpt} {type(ex).__name__}: {ex} | cmd: {cmdline()}", out_dir)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, choices=[k for k, v in LG.CKPTS.items() if v["kind"] == "prefix"])
    ap.add_argument("--n", type=int, default=100, help="first n GQA-rel items (J7 order)")
    ap.add_argument("--values", default="landed,0,1",
                    help="gate values; 'landed' (= learned tanh gate; scale 1 for rms_ungated), numbers")
    ap.add_argument("--out", default=None, help=f"output JSON (default {RESULTS_DIR}/j6_llava_<ckpt>.json)")
    ap.add_argument("--smoke", action="store_true", help="mark as smoke; default out under .../j6_llava/smoke/")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--chunk", type=int, default=3, help="gate values per batched forward (all by default)")
    ap.add_argument("--norm_items", type=int, default=64)
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()
    if args.out is None:
        args.out = (os.path.join(RESULTS_DIR, "smoke", f"j6_llava_{args.ckpt}_n{args.n}.json") if args.smoke
                    else os.path.join(RESULTS_DIR, f"j6_llava_{args.ckpt}.json"))
    run(args)


if __name__ == "__main__":
    main()
