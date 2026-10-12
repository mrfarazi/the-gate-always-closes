#!/usr/bin/env python3
"""POPE hallucination evaluation — yes/no object probing on COCO images.

Three variants: random, popular, adversarial (3000 questions each, 500 images).
Metrics: accuracy, precision, recall, F1, yes-rate.
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
POPE_DIR = os.path.join(DATA, "pope")
COCO_IMG_DIR = os.path.join(DATA, "coco/val2014")


def make_grid_boxes(n_per_side=6):
    boxes = []
    step = 1.0 / n_per_side
    for i in range(n_per_side):
        for j in range(n_per_side):
            boxes.append([j * step, i * step, (j + 1) * step, (i + 1) * step])
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    box_mask = torch.ones(len(boxes))
    return box_tensor.unsqueeze(0), box_mask.unsqueeze(0)


def parse_yes_no(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    return text.split()[0] if text else "unknown"


def compute_metrics(predictions):
    tp = fp = tn = fn = 0
    for p in predictions:
        pred = p["pred_label"]
        gt = p["gt_label"]
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    yes_rate = (tp + fp) / max(total, 1) * 100

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "yes_rate": round(yes_rate, 2),
        "total": total,
    }


def eval_pope_variant(variant, model, processor, device, wrapper=None):
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{variant}.json")
    if not os.path.exists(pope_file):
        return {"error": f"File not found: {pope_file}"}

    with open(pope_file) as f:
        questions = [json.loads(line) for line in f]

    boxes, box_mask = make_grid_boxes()
    boxes, box_mask = boxes.to(device), box_mask.to(device)

    predictions = []
    t0 = time.time()

    for idx, q in enumerate(questions):
        img_name = q["image"]
        img_path = os.path.join(COCO_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        question_text = q["text"]

        prompt = f"USER: <image>\n{question_text}\nAnswer with yes or no.\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        if wrapper is not None:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output_ids = wrapper.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    boxes=boxes, box_mask=box_mask,
                    max_new_tokens=10, do_sample=False)
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10, do_sample=False)

        text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1]

        pred_label = parse_yes_no(text)
        predictions.append({
            "question_id": q["question_id"],
            "pred_label": pred_label,
            "gt_label": q["label"],
            "raw_output": text.strip(),
        })

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {variant}: {idx+1}/{len(questions)} ({elapsed:.1f}s)")

    metrics = compute_metrics(predictions)
    return {"metrics": metrics, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="registry name or checkpoint directory")
    parser.add_argument("--model", default="auto",
                        choices=["auto", "vanilla", "lora_only", "hra", "prefix"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="also evaluate the same checkpoint with the pathway absent (LoRA only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.environ.get("RUN_DIR", os.path.join(RUNS, "pope"))
    os.makedirs(output_dir, exist_ok=True)
    variants = ["random", "popular", "adversarial"]

    from hra.checkpoints import load_for_eval

    def run(model_type, tag):
        model, processor, wrapper = load_for_eval(model_type, args.checkpoint, device, variant=args.variant)
        print(f"\n=== POPE Evaluation ({tag}) ===")
        res = {}
        for variant in variants:
            print(f"  Evaluating {variant}...")
            result = eval_pope_variant(variant, model, processor, device, wrapper)
            res[variant] = result.get("metrics", result)
            if "metrics" in result:
                m = result["metrics"]
                print(f"    Acc={m['accuracy']}% P={m['precision']}% R={m['recall']}% "
                      f"F1={m['f1']}% Yes={m['yes_rate']}%")
        del model, wrapper
        torch.cuda.empty_cache()
        return res

    all_results = {"checkpoint": args.checkpoint, "model": args.model}
    all_results["pathway"] = run(args.model, args.model)
    if args.baseline and args.checkpoint:
        all_results["lora_only"] = run("lora_only", "LoRA only, pathway absent")

    with open(os.path.join(output_dir, "pope_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/pope_results.json")


if __name__ == "__main__":
    main()
