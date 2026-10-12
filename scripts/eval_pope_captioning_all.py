#!/usr/bin/env python3
"""Overnight runner: POPE + COCO captioning for all 4 headline models on 1 GPU.

Models:
  vanilla, small_alpha (HRA, bias on), rms_ungated (prefix zeroed), soft_prompt (prefix on)

Outputs:
  outputs/final/pope_<model>.json
  outputs/final/captioning_<model>.json
  outputs/final/summary.json  (aggregated)

Estimated wall clock on 1 GPU: ~5-6h.
"""
import argparse
import json
import os
import sys
import time

import torch
from PIL import Image


from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH
OUT_DIR = os.environ.get("RUN_DIR", os.path.join(RUNS, "generation"))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"

# (name, model type, registry name). Model types: "prefix" = pathway on; "prefix_zeroed" = prefix
# tokens x 0 with the slots kept (the condition of the "RMS prefix, ungated" row of the paper's
# generation table); "lora_only" = pathway absent (docs/PROTOCOL.md).
MODELS = [
    ("vanilla",     "vanilla",   None),
    ("small_alpha", "hra",       "small_alpha"),
    ("rms_ungated", "prefix_zeroed", "rms_ungated"),
    ("soft_prompt", "prefix",    "soft_prompt"),
]


def make_grid_boxes(n=6):
    boxes = []
    step = 1.0 / n
    for i in range(n):
        for j in range(n):
            boxes.append([j*step, i*step, (j+1)*step, (i+1)*step])
    return (torch.tensor(boxes, dtype=torch.float32).unsqueeze(0).to(DEVICE),
            torch.ones(1, len(boxes)).to(DEVICE))


def load_model(model_type, ckpt, variant=None):
    """(llava_model, processor, wrapper); wrapper is None when the pathway is absent.

    `prefix_zeroed` loads the prefix pathway and multiplies its tokens by 0 (slots kept)."""
    from hra.checkpoints import load_for_eval, set_prefix_scale
    if model_type == "prefix_zeroed":
        vlm, processor, wrapper = load_for_eval("prefix", ckpt, DEVICE, variant=variant)
        return vlm, processor, set_prefix_scale(wrapper, 0.0)
    return load_for_eval(model_type, ckpt, DEVICE, variant=variant)


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

from hra.eval.pope import eval_pope_variant, compute_metrics
POPE_VARIANTS = ["random", "popular", "adversarial"]


def run_pope(name, vlm, processor, wrapper):
    print(f"\n[POPE] {name}...")
    out = {}
    t0 = time.time()
    for v in POPE_VARIANTS:
        print(f"  {v}...")
        result = eval_pope_variant(v, vlm, processor, DEVICE, wrapper)
        m = result.get("metrics") if isinstance(result, dict) else None
        if m is not None:
            out[v] = m
            print(f"    Acc={m['accuracy']}% P={m['precision']}% R={m['recall']}% "
                  f"F1={m['f1']}% Yes={m['yes_rate']}%")
        else:
            out[v] = result
    out["_elapsed_sec"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

COCO_IMG = os.path.join(DATA, "coco/val2017")
COCO_ANN = os.path.join(DATA, "coco/annotations/captions_val2017.json")


def generate_caption(vlm, processor, wrapper, image, boxes, box_mask):
    prompt = "USER: <image>\nProvide a brief caption for this image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    if wrapper is not None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = wrapper.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                boxes=boxes, box_mask=box_mask,
                max_new_tokens=50, do_sample=False)
    else:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = vlm.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50, do_sample=False)
    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]
    return text.strip()


def run_captioning(name, vlm, processor, wrapper, boxes, box_mask, limit=0):
    print(f"\n[Captioning] {name}...")
    with open(COCO_ANN) as f:
        ann = json.load(f)
    image_ids = sorted(set(img["id"] for img in ann["images"]))
    id_to_fname = {img["id"]: img["file_name"] for img in ann["images"]}
    if limit > 0:
        image_ids = image_ids[:limit]
    t0 = time.time()
    preds = []
    for i, iid in enumerate(image_ids):
        fname = id_to_fname.get(iid)
        if not fname:
            continue
        p = os.path.join(COCO_IMG, fname)
        if not os.path.exists(p):
            continue
        try:
            img = Image.open(p).convert("RGB")
            cap = generate_caption(vlm, processor, wrapper, img, boxes, box_mask)
            preds.append({"image_id": iid, "caption": cap})
        except Exception as e:
            if len(preds) < 5:
                print(f"    ERR on {fname}: {e}")
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(image_ids) - i - 1) / rate
            print(f"    {i+1}/{len(image_ids)} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    print(f"  Generated {len(preds)} captions in {time.time() - t0:.0f}s")
    # Save predictions
    with open(os.path.join(OUT_DIR, f"captioning_{name}_predictions.json"), "w") as f:
        json.dump(preds, f)
    # Compute metrics
    try:
        from eval_captioning import run_coco_eval
        metrics = run_coco_eval(preds, COCO_ANN)
    except Exception as e:
        print(f"  Metrics computation failed: {e}")
        metrics = {"error": str(e)}
    return {"metrics": metrics, "n_preds": len(preds),
            "_elapsed_sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_pope", action="store_true")
    p.add_argument("--skip_captioning", action="store_true")
    p.add_argument("--caption_limit", type=int, default=0,
                   help="0=all 5000 val2017 images; else use first N")
    p.add_argument("--models", default="",
                   help="Override MODELS: semicolon-separated name:mtype:ckpt. Added for "
                        "so that a new checkpoint can be scored through the SAME "
                        "path that produced the published numbers (apples-to-apples).")
    p.add_argument("--out_name", default="pope_captioning_summary.json")
    args = p.parse_args()

    models = MODELS
    if args.models:
        models = []
        for spec in args.models.split(";"):
            spec = spec.strip()
            if spec:
                name, mtype, ckpt = spec.split(":", 2)
                models.append((name, mtype, ckpt or None))

    all_results = {}
    for name, mtype, ckpt in models:
        print(f"\n{'='*60}\nLoading model: {name} ({mtype}) from {ckpt}\n{'='*60}")
        vlm, processor, wrapper = load_model(mtype, ckpt)
        boxes, box_mask = make_grid_boxes()

        results = {}
        if not args.skip_pope:
            try:
                results["pope"] = run_pope(name, vlm, processor, wrapper)
            except Exception as e:
                print(f"POPE failed for {name}: {e}")
                results["pope"] = {"error": str(e)}
        if not args.skip_captioning:
            try:
                results["captioning"] = run_captioning(
                    name, vlm, processor, wrapper, boxes, box_mask, args.caption_limit)
            except Exception as e:
                print(f"Captioning failed for {name}: {e}")
                results["captioning"] = {"error": str(e)}

        with open(os.path.join(OUT_DIR, f"{name}_pope_captioning.json"), "w") as f:
            json.dump(results, f, indent=2)
        all_results[name] = results

        # Free memory before next model
        del vlm, processor, wrapper
        torch.cuda.empty_cache()
        import gc; gc.collect()

    with open(os.path.join(OUT_DIR, args.out_name), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== FINAL SUMMARY ===")
    for name in all_results:
        r = all_results[name]
        pope = r.get("pope", {})
        cap = r.get("captioning", {})
        pope_line = "  ".join(f"{v}={pope.get(v, {}).get('accuracy', '?')}"
                              for v in POPE_VARIANTS if isinstance(pope.get(v), dict))
        cap_m = cap.get("metrics", {})
        cap_line = "  ".join(f"{k}={v:.4f}" for k, v in cap_m.items() if isinstance(v, (int, float)))
        print(f"  {name:14s}  POPE: {pope_line}   CAP: {cap_line}")


if __name__ == "__main__":
    main()
