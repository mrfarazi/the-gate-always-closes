#!/usr/bin/env python3
"""COCO captioning evaluation (free-form generation quality; supplement Table S10).

Generates captions for COCO val2017 images and computes CIDEr, BLEU-4, METEOR, ROUGE-L with
pycocoevalcap (METEOR/SPICE need Java). `--checkpoint` is a registry name or directory; `--model`
selects whether the auxiliary pathway is on (hra/prefix) or absent (lora_only).
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
COCO_IMG_DIR = os.path.join(DATA, "coco/val2017")
COCO_ANN_PATH = os.path.join(DATA, "coco/annotations/captions_val2017.json")


def make_grid_boxes(n_per_side=6):
    boxes = []
    step = 1.0 / n_per_side
    for i in range(n_per_side):
        for j in range(n_per_side):
            boxes.append([j * step, i * step, (j + 1) * step, (i + 1) * step])
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    box_mask = torch.ones(len(boxes))
    return box_tensor.unsqueeze(0), box_mask.unsqueeze(0)


def generate_caption_hra(wrapper, processor, image, boxes, box_mask, device):
    prompt = "USER: <image>\nProvide a brief description of the image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = wrapper.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            boxes=boxes, box_mask=box_mask,
            max_new_tokens=128, do_sample=False)
    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def generate_caption_vanilla(model, processor, image, device):
    prompt = "USER: <image>\nProvide a brief description of the image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128, do_sample=False)
    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def run_coco_eval(predictions, ann_path):
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco_gt = COCO(ann_path)

    coco_res = coco_gt.loadRes(predictions)
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    coco_eval.params['image_id'] = [p["image_id"] for p in predictions]
    coco_eval.evaluate()

    return coco_eval.eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="registry name or checkpoint directory")
    parser.add_argument("--model", default="auto", choices=["auto", "vanilla", "lora_only", "hra", "prefix"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of images to evaluate (0 = all 5000)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run the same checkpoint with the pathway absent (LoRA only; vanilla if no checkpoint)")
    parser.add_argument("--prefix_scale", type=float, default=1.0,
                        help="multiplier on the prefix tokens for --model prefix (0 = zeroed, slots kept)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.environ.get("RUN_DIR", os.path.join(RUNS, "captioning"))
    os.makedirs(output_dir, exist_ok=True)

    with open(COCO_ANN_PATH) as f:
        ann_data = json.load(f)
    image_ids = sorted(set(img["id"] for img in ann_data["images"]))
    if args.num_samples > 0:
        image_ids = image_ids[:args.num_samples]

    id_to_filename = {img["id"]: img["file_name"] for img in ann_data["images"]}

    from hra.checkpoints import load_for_eval, set_prefix_scale
    model, processor, wrapper = load_for_eval(args.model, args.checkpoint, device, variant=args.variant)
    if wrapper is not None and hasattr(wrapper, "prefix_graph") and args.prefix_scale != 1.0:
        set_prefix_scale(wrapper, args.prefix_scale)
    mode = "absent" if wrapper is None else ("on" if args.prefix_scale == 1.0 else f"prefix x {args.prefix_scale:g}")
    print(f"Loaded model={args.model} checkpoint={args.checkpoint} pathway={mode}")

    boxes, box_mask = make_grid_boxes()
    boxes, box_mask = boxes.to(device), box_mask.to(device)

    hra_metrics = None
    if wrapper is None:
        print(f"\n=== Captioning, pathway absent ({len(image_ids)} images) ===")
        preds = []
        t0 = time.time()
        for idx, img_id in enumerate(image_ids):
            fname = id_to_filename.get(img_id)
            if fname is None or not os.path.exists(os.path.join(COCO_IMG_DIR, fname)):
                continue
            image = Image.open(os.path.join(COCO_IMG_DIR, fname)).convert("RGB")
            preds.append({"image_id": img_id, "caption": generate_caption_vanilla(model, processor, image, device)})
            if (idx + 1) % 100 == 0:
                print(f"  {idx+1}/{len(image_ids)} ({time.time()-t0:.1f}s)")
        with open(os.path.join(output_dir, "captioning_predictions.json"), "w") as f:
            json.dump(preds, f, indent=2)
        hra_metrics = run_coco_eval(preds, COCO_ANN_PATH)
        for k, v in hra_metrics.items():
            print(f"  {k}: {v:.4f}")
        with open(os.path.join(output_dir, "captioning_pathway_absent_results.json"), "w") as f:
            json.dump(hra_metrics, f, indent=2)
    if wrapper is not None:
        print(f"\n=== Captioning, pathway on ({len(image_ids)} images) ===")
        hra_predictions = []
        t0 = time.time()
        for idx, img_id in enumerate(image_ids):
            fname = id_to_filename.get(img_id)
            if fname is None:
                continue
            img_path = os.path.join(COCO_IMG_DIR, fname)
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            caption = generate_caption_hra(wrapper, processor, image, boxes, box_mask, device)
            hra_predictions.append({"image_id": img_id, "caption": caption})
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {idx+1}/{len(image_ids)} ({elapsed:.1f}s)")

        with open(os.path.join(output_dir, "captioning_hra_predictions.json"), "w") as f:
            json.dump(hra_predictions, f, indent=2)

        print("Computing HRA metrics...")
        hra_metrics = run_coco_eval(hra_predictions, COCO_ANN_PATH)
        print("HRA Captioning Results:")
        for k, v in hra_metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(os.path.join(output_dir, "captioning_hra_results.json"), "w") as f:
            json.dump(hra_metrics, f, indent=2)

        del wrapper
        torch.cuda.empty_cache()

    # --- Baseline ---
    if args.baseline:
        model_bl, processor_bl, _ = load_for_eval("lora_only" if args.checkpoint else "vanilla",
                                                  args.checkpoint, device)

        print(f"\n=== Baseline Captioning ({len(image_ids)} images) ===")
        bl_predictions = []
        t0 = time.time()
        for idx, img_id in enumerate(image_ids):
            fname = id_to_filename.get(img_id)
            if fname is None:
                continue
            img_path = os.path.join(COCO_IMG_DIR, fname)
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            caption = generate_caption_vanilla(model_bl, processor_bl, image, device)
            bl_predictions.append({"image_id": img_id, "caption": caption})
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {idx+1}/{len(image_ids)} ({elapsed:.1f}s)")

        with open(os.path.join(output_dir, "captioning_baseline_predictions.json"), "w") as f:
            json.dump(bl_predictions, f, indent=2)

        print("Computing baseline metrics...")
        bl_metrics = run_coco_eval(bl_predictions, COCO_ANN_PATH)
        print("Baseline Captioning Results:")
        for k, v in bl_metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(os.path.join(output_dir, "captioning_baseline_results.json"), "w") as f:
            json.dump(bl_metrics, f, indent=2)

    # --- Summary ---
    print("\n=== Summary ===")
    all_results = {"checkpoint": args.checkpoint, "model": args.model}
    if hra_metrics is not None:
        all_results["pathway_on" if wrapper is not None else "pathway_absent"] = hra_metrics
        print("Model:   " + "  ".join(f"{k}={v:.4f}" for k, v in hra_metrics.items()))
    if args.baseline:
        all_results["baseline"] = bl_metrics
        print("Baseline:" + "  ".join(f"{k}={v:.4f}" for k, v in bl_metrics.items()))

    with open(os.path.join(output_dir, "captioning_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}/captioning_results.json")


if __name__ == "__main__":
    main()
