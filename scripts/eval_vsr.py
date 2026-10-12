#!/usr/bin/env python3
"""VSR (Visual Spatial Reasoning) eval — binary classification of image + caption.

For each (image, caption, gold_label) triple, compute
    log P('Yes' | image, "Does the image show '{caption}'?")
    log P('No'  | image, "Does the image show '{caption}'?")
and predict 1 if P(Yes) > P(No) else 0. Compare to gold.

Usage:
    python eval_vsr.py --model vanilla                --output_dir outputs/vsr/vanilla
    python eval_vsr.py --model hra   --checkpoint DIR --output_dir outputs/vsr/hra
    python eval_vsr.py --model prefix --checkpoint DIR --output_dir outputs/vsr/prefix
"""
import argparse
import json
import os
import sys

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH
VSR_PATH = os.path.join(DATA, "vsr/test")
COCO_PATH = os.path.join(DATA, "coco/train2017")
MAX_PROPOSALS = 36


def make_grid_boxes(n=6, device="cuda"):
    boxes = []
    step = 1.0 / n
    for i in range(n):
        for j in range(n):
            boxes.append([j*step, i*step, (j+1)*step, (i+1)*step])
    return (torch.tensor(boxes, dtype=torch.float32).unsqueeze(0).to(device),
            torch.ones(1, len(boxes)).to(device))


def yes_no_log_probs(model, processor, wrapper, image, caption, device,
                     boxes=None, box_mask=None):
    """Return (log P(' Yes' | prompt), log P(' No' | prompt))."""
    tokenizer = processor.tokenizer
    prompt = f"USER: <image>\nDoes the image show: \"{caption}\"?\nASSISTANT:"

    log_probs = {}
    for answer in (" Yes", " No"):
        full = prompt + answer
        inputs = processor(text=full, images=image, return_tensors="pt").to(device)
        full_ids = inputs["input_ids"][0].tolist()
        asst_tokens = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        completion_start = None
        for pos in range(len(full_ids) - len(asst_tokens)):
            if full_ids[pos:pos + len(asst_tokens)] == asst_tokens:
                completion_start = pos + len(asst_tokens)
                break
        labels = inputs["input_ids"].clone()
        if completion_start is not None:
            labels[0, :completion_start] = -100

        if wrapper is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                score = wrapper.score(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    boxes=boxes, box_mask=box_mask,
                    labels=labels)
        else:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels)
            score = -outputs.loss.item()
        log_probs[answer.strip()] = score

    return log_probs["Yes"], log_probs["No"]


def load_model(model_type, checkpoint, device, variant=None):
    """Return (model, processor, wrapper). wrapper is None when the auxiliary pathway is absent
    (vanilla / lora_only); for hra / prefix the wrapper carries the pathway and model is None.
    `checkpoint` is a registry name or directory; prefix graphs are rebuilt from `variant` or
    the checkpoint's variant.json (never from the path)."""
    from hra.checkpoints import load_for_eval
    model, processor, wrapper = load_for_eval(model_type, checkpoint, device, variant=variant)
    return (None if wrapper is not None else model), processor, wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vanilla", "lora_only", "hra", "prefix"], required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="0=all, N=first N examples")
    p.add_argument("--variant", default=None,
                   help="VARIANT_CONFIGS key; REQUIRED for correctness on any "
                        "checkpoint trained with ungated/no_rms (see load_model).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print(f"Loading {args.model} (ckpt={args.checkpoint})...")
    model, processor, wrapper = load_model(args.model, args.checkpoint, device,
                                           variant=args.variant)

    boxes, box_mask = make_grid_boxes(device=device) if wrapper is not None else (None, None)

    ds = load_from_disk(VSR_PATH)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Evaluating on {len(ds)} VSR examples")

    predictions = []
    correct = 0
    errors = 0
    # per-relation tracking
    per_rel = {}

    for i, ex in enumerate(ds):
        img_path = os.path.join(COCO_PATH, ex["image"])
        if not os.path.exists(img_path):
            errors += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            log_yes, log_no = yes_no_log_probs(
                model, processor, wrapper, image, ex["caption"], device, boxes, box_mask)
            pred = 1 if log_yes > log_no else 0
            gold = int(ex["label"])
            hit = int(pred == gold)
            correct += hit

            rel = ex.get("relation", "unknown") or "unknown"
            per_rel.setdefault(rel, {"correct": 0, "total": 0})
            per_rel[rel]["correct"] += hit
            per_rel[rel]["total"] += 1

            predictions.append({
                "image": ex["image"], "caption": ex["caption"], "relation": rel,
                "label": gold, "pred": pred, "correct": hit,
                "log_yes": float(log_yes), "log_no": float(log_no),
            })
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  ERR on {ex['image']}: {e}")

        if (i + 1) % 200 == 0:
            acc = 100 * correct / (i + 1 - errors) if (i + 1 - errors) > 0 else 0
            print(f"  {i+1}/{len(ds)}: acc={acc:.2f}% (err={errors})")

    total_scored = len(predictions)
    acc = 100 * correct / max(total_scored, 1)
    print(f"\n=== VSR ({args.model}) ===")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total_scored})  errors={errors}")
    print("Per-relation (top 15 by count):")
    rel_sorted = sorted(per_rel.items(), key=lambda kv: -kv[1]["total"])
    for rel, s in rel_sorted[:15]:
        rel_acc = 100 * s["correct"] / max(s["total"], 1)
        print(f"  {rel:30s} {rel_acc:6.2f}% ({s['correct']}/{s['total']})")

    out = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "accuracy": round(acc, 2),
        "correct": correct,
        "total_scored": total_scored,
        "errors": errors,
        "per_relation": {r: {"correct": s["correct"], "total": s["total"],
                             "accuracy": round(100 * s["correct"] / max(s["total"], 1), 2)}
                         for r, s in per_rel.items()},
    }
    with open(os.path.join(args.output_dir, "vsr_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.output_dir, "vsr_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
