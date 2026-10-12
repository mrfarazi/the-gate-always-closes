#!/usr/bin/env python3
"""One job = one (model, bench, value-list, eps) sweep -> one JSON. Shared by J4/J5/J7."""
import math
import os
import time
import traceback

from PIL import Image

from hra.knockout.common import (atomic_write_json, cmdline, gqa_image_path, log_line,
                    native_rmsnorm_eps, now, patch_rmsnorm_eps, sc_image, skip_if_exists)
from hra.knockout.sweep import build_rows, load_bench, run_sweep


def run_job(script, out, model_loader, values_fn, bench, n, data_base, eps_spec="native",
            value_key="gamma", meta=None, smoke=0, chunk=16, n_boot=5000, norm_items=64,
            unit_check_kwargs=None):
    if skip_if_exists(out, script):
        return None
    t_start = time.time()
    log_line(f"{script} START | n={n} bench={bench} eps={eps_spec} out={out} | cmd: {cmdline()}")
    result = {"script": script, "cmd": cmdline(), "started": now(), "bench": bench,
              "n_requested": n, "smoke": bool(smoke), "value_key": value_key,
              "bootstrap": {"n_resamples": n_boot, "seed": 0, "paired_vs": 0.0}}
    if meta:
        result.update(meta)
    try:
        model = model_loader()
        result["learned_gate"] = float(model.learned_gate) if math.isfinite(model.learned_gate) else None
        result["learned_gate_raw"] = float(model.learned_gate_raw) if math.isfinite(model.learned_gate_raw) else None
        result["trained_ungated"] = bool(model.trained_ungated)
        values = values_fn(model)
        result["values"] = values
        # ---- epsilon patch (always applied, so n_patched is reported even at native)
        native = native_rmsnorm_eps(model.lm)
        eps_val = native if eps_spec in (None, "native") else float(eps_spec)
        n_patched, restore, natives = patch_rmsnorm_eps(model.lm, eps_val)
        result.update({"eps": eps_val, "eps_native": native, "eps_native_values_seen": natives,
                       "n_patched": n_patched, "eps_is_native": abs(eps_val - native) < 1e-15})
        print(f"[{script}] eps={eps_val:g} (native {native:g}) patched {n_patched} RMSNorm modules; "
              f"values={values}", flush=True)
        try:
            items, dss = load_bench(bench, n, data_base)
            result["n_items_listed"] = len(items)
            result["item_ids"] = [it["id"] for it in items]
            # ---- unit check on item 0
            uc_kw = dict(unit_check_kwargs or {})
            if bench == "sugarcrepe":
                img0 = sc_image(dss, items[0])
                uc = model.unit_check(img0, prompt_kind="sugarcrepe", caption=items[0]["pos"], **uc_kw)
            else:
                img0 = Image.open(gqa_image_path(data_base, items[0])).convert("RGB")
                uc = model.unit_check(img0, prompt_kind="gqa", question=items[0]["question"], **uc_kw)
            uc["n_patched_gt_0"] = n_patched > 0
            result["unit_check"] = uc
            print(f"[{script}] unit_check: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in uc.items()), flush=True)
            # ---- sweep
            records, norms, timing = run_sweep(model, bench, items, values, data_base, dss=dss,
                                               chunk=chunk, norm_items=norm_items, tag=script)
        finally:
            restore()
        if len(records) == 0:
            raise RuntimeError("no items were evaluated; refusing to write an empty result")
        rows = build_rows(records, values, n_boot=n_boot, seed=0)
        for r in rows:
            r[value_key] = r.pop("value")
        result["n"] = len(records)
        result["rows"] = rows
        try:
            import torch
            timing["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            timing["chunk"] = chunk
        except Exception:
            pass
        norms["pred_gamma_c"] = math.sqrt(eps_val) / max(norms["prefix_rms_at_1"], 1e-12)
        norms["pred_gamma_c_0p4"] = 0.4 * norms["pred_gamma_c"]
        norms[f"prefix_rms_at_{value_key}1"] = norms["prefix_rms_at_1"]
        result["token_norms"] = norms
        result["timing"] = timing
        result["per_item"] = {
            "ids": [r["id"] for r in records],
            "correct": [r["correct"] for r in records],
        }
        if all("absent" in r for r in records):
            result["per_item"]["absent_correct"] = [r["absent"]["correct"] for r in records]
            if bench != "sugarcrepe":
                result["per_item"]["absent_pred"] = [r["absent"]["pred"] for r in records]
            result["rows_note"] = ("row with label 'prefix_absent' = prefix slot removed "
                                   "(published LoRA-only / prefix-dropped condition)")
        if bench == "sugarcrepe":
            result["per_item"]["s_pos"] = [r["s_pos"] for r in records]
            result["per_item"]["s_neg"] = [r["s_neg"] for r in records]
        else:
            result["per_item"]["pred"] = [r["pred"] for r in records]
            result["per_item"]["answer"] = [r["answer"] for r in records]
        result["finished"] = now()
        result["wall_sec"] = time.time() - t_start
        atomic_write_json(out, result)
        summary = "  ".join(
            (f"{value_key}={r[value_key]:g}:{r['acc']:.1f}" if r[value_key] is not None
             else f"absent:{r['acc']:.1f}") for r in rows)
        print(f"[{script}] DONE n={len(records)} {timing['items_per_sec']:.2f} items/s  {summary}", flush=True)
        log_line(f"{script} END | n={len(records)} runtime={timing['runtime_sec']:.0f}s "
                 f"({timing['items_per_sec']:.2f} items/s, wall {time.time() - t_start:.0f}s) out={out}")
        return result
    except Exception as ex:
        traceback.print_exc()
        log_line(f"{script} FAILED | {type(ex).__name__}: {ex} | cmd: {cmdline()}")
        raise
