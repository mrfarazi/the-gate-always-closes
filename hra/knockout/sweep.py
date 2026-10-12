#!/usr/bin/env python3
"""Generic paired sweep loop shared by J4 / J5 / J7.

A *model adapter* exposes:
    prepare_sc(image, caption)  -> ctx      (scoring; ctx carries merged template, labels)
    prepare_gqa(image, question) -> ctx     (generation)
    score(ctx, values)    -> [float]        one log-likelihood (= -mean caption CE) per value
    generate(ctx, values) -> [str]          one decoded answer per value
    norms(ctx)            -> dict           prefix / text token norms for this ctx
    value_name            -> 'gamma' | 'scale' | 'bias_mult'
The sweep multiplies the injected object (prefix tokens or attention bias) by each value
inside the adapter; every value of one item is evaluated from the same cached vision
features and in one batched LM call, so the comparison across values is exactly paired.
"""
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from hra.knockout.common import (ACCV_BASE, gqa_image_path, load_gqa_items, load_sugarcrepe_items,
                    normalize_gqa_pred, sc_image, summarise_binary)


def run_sweep(model, bench, items, values, data_base, dss=None, chunk=16,
              progress_every=100, norm_items=64, tag=""):
    """Returns (records, norm_summary, timing). records: list of per-item dicts."""
    records = []
    prefix_rms, prefix_l2, text_rms, text_l2 = [], [], [], []
    t0 = time.time()
    n_lm_calls = 0
    for i, it in enumerate(items):
        try:
            if bench == "sugarcrepe":
                img = sc_image(dss, it)
                ctx_p = model.prepare_sc(img, it["pos"])
                ctx_n = model.prepare_sc(img, it["neg"], reuse=ctx_p)
                s_pos = model.score(ctx_p, values, chunk=chunk)
                s_neg = model.score(ctx_n, values, chunk=chunk)
                n_lm_calls += 2 * math.ceil(len(values) / chunk)
                rec = {"id": it["id"], "s_pos": s_pos, "s_neg": s_neg,
                       "correct": [int(p > n) for p, n in zip(s_pos, s_neg)]}
                if getattr(model, "supports_absent", False):
                    ap_, an_ = model.score_absent(ctx_p), model.score_absent(ctx_n)
                    rec["absent"] = {"s_pos": ap_, "s_neg": an_, "correct": int(ap_ > an_)}
                    n_lm_calls += 2
                ctx_for_norm = ctx_p
            else:
                p = gqa_image_path(data_base, it)
                if not os.path.exists(p):
                    print(f"  [warn] missing image {p}; skipping item {it['id']}", flush=True)
                    continue
                img = Image.open(p).convert("RGB")
                ctx = model.prepare_gqa(img, it["question"])
                preds = model.generate(ctx, values, chunk=chunk)
                n_lm_calls += math.ceil(len(values) / chunk)
                preds = [normalize_gqa_pred(t) for t in preds]
                rec = {"id": it["id"], "answer": it["answer"], "pred": preds,
                       "correct": [int(pr == it["answer"]) for pr in preds]}
                if getattr(model, "supports_absent", False):
                    pa = normalize_gqa_pred(model.generate_absent(ctx))
                    rec["absent"] = {"pred": pa, "correct": int(pa == it["answer"])}
                    n_lm_calls += 1
                ctx_for_norm = ctx
            if len(prefix_rms) < norm_items:
                nm = model.norms(ctx_for_norm)
                prefix_rms.append(nm["prefix_rms"]); prefix_l2.append(nm["prefix_l2"])
                text_rms.append(nm["text_rms_median"]); text_l2.append(nm["text_l2_median"])
            records.append(rec)
        except Exception as ex:  # keep the sequence going on data errors only
            import traceback
            if isinstance(ex, torch.OutOfMemoryError) or "CUDA" in str(ex) or "device-side" in str(ex):
                raise   # a GPU failure must fail the job, not silently thin the sample
            print(f"  [warn] item {it.get('id')}: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
            continue
        if (i + 1) % progress_every == 0 or (i + 1) == len(items):
            el = time.time() - t0
            acc0 = 100.0 * np.mean([r["correct"][0] for r in records]) if records else float("nan")
            print(f"[{tag}] {i + 1}/{len(items)} items  {el:.0f}s  "
                  f"{(i + 1) / max(el, 1e-9):.2f} items/s  acc@{model.value_name}=0: {acc0:.1f}",
                  flush=True)
    elapsed = time.time() - t0
    norm_summary = {
        "n_items_measured": len(prefix_rms),
        "prefix_rms_at_1": float(np.median(prefix_rms)) if prefix_rms else float("nan"),
        "prefix_rms_mean": float(np.mean(prefix_rms)) if prefix_rms else float("nan"),
        "prefix_l2_at_1": float(np.median(prefix_l2)) if prefix_l2 else float("nan"),
        "text_rms_median": float(np.median(text_rms)) if text_rms else float("nan"),
        "text_l2_median": float(np.median(text_l2)) if text_l2 else float("nan"),
    }
    if prefix_l2 and text_l2:
        norm_summary["prefix_over_text_l2_ratio"] = norm_summary["prefix_l2_at_1"] / \
            max(norm_summary["text_l2_median"], 1e-12)
    timing = {"runtime_sec": elapsed, "n_items": len(records),
              "items_per_sec": len(records) / max(elapsed, 1e-9),
              "n_lm_calls": n_lm_calls}
    return records, norm_summary, timing


def build_rows(records, values, n_boot=5000, seed=0, baseline_index=0):
    by_val = {v: [r["correct"][k] for r in records] for k, v in enumerate(values)}
    base_v = values[baseline_index]
    rows = summarise_binary(values, by_val, baseline=base_v, n_boot=n_boot, seed=seed)
    if records and all("absent" in r for r in records):
        # prefix slot removed entirely (the published LoRA-only / prefix-dropped condition)
        by_val[None] = [r["absent"]["correct"] for r in records]
        extra = summarise_binary([None], by_val, baseline=base_v, n_boot=n_boot, seed=seed)[0]
        extra["label"] = "prefix_absent"
        rows.append(extra)
    return rows


def load_bench(bench, n, data_base):
    if bench == "sugarcrepe":
        items, dss = load_sugarcrepe_items(data_base, n)
        return items, dss
    if bench in ("gqa", "gqa_rel"):
        return load_gqa_items(data_base, bench, n), None
    raise ValueError(bench)
