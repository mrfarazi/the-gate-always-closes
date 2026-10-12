#!/usr/bin/env python3
"""Evaluate HRA on Winoground and SugarCrepe compositionality benchmarks."""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from datasets import load_from_disk
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel


from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH
MAX_PROPOSALS = 36


def make_grid_boxes(n_per_side=6):
    boxes = []
    step = 1.0 / n_per_side
    for i in range(n_per_side):
        for j in range(n_per_side):
            boxes.append([j * step, i * step, (j + 1) * step, (i + 1) * step])
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    box_mask = torch.ones(len(boxes))
    return box_tensor.unsqueeze(0), box_mask.unsqueeze(0)


def sequence_log_likelihood(model, processor, image, caption, device,
                            wrapper=None, boxes=None, box_mask=None):
    prompt = f"USER: <image>\nDescribe this image.\nASSISTANT: {caption}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    tokenizer = processor.tokenizer
    full_ids = inputs["input_ids"][0].tolist()
    asst_tokens = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    caption_start = None
    for pos in range(len(full_ids) - len(asst_tokens)):
        if full_ids[pos:pos + len(asst_tokens)] == asst_tokens:
            caption_start = pos + len(asst_tokens)
            break

    labels = inputs["input_ids"].clone()
    if caption_start is not None:
        labels[0, :caption_start] = -100

    if wrapper is not None and boxes is not None:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return wrapper.score(
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
        return -outputs.loss.item()


def eval_winoground(processor, model, device, wrapper=None):
    wg_path = os.path.join(DATA, "winoground")
    ds = load_from_disk(os.path.join(wg_path, "test"))
    boxes, box_mask = make_grid_boxes()
    boxes, box_mask = boxes.to(device), box_mask.to(device)

    text_correct = 0
    image_correct = 0
    group_correct = 0
    total = 0

    for row in ds:
        img_0 = row["image_0"].convert("RGB")
        img_1 = row["image_1"].convert("RGB")
        cap_0 = row["caption_0"]
        cap_1 = row["caption_1"]

        s00 = sequence_log_likelihood(model, processor, img_0, cap_0, device, wrapper, boxes, box_mask)
        s01 = sequence_log_likelihood(model, processor, img_0, cap_1, device, wrapper, boxes, box_mask)
        s10 = sequence_log_likelihood(model, processor, img_1, cap_0, device, wrapper, boxes, box_mask)
        s11 = sequence_log_likelihood(model, processor, img_1, cap_1, device, wrapper, boxes, box_mask)

        t_ok = s00 > s01 and s11 > s10
        i_ok = s00 > s10 and s11 > s01
        g_ok = t_ok and i_ok

        if t_ok:
            text_correct += 1
        if i_ok:
            image_correct += 1
        if g_ok:
            group_correct += 1
        total += 1

    return {
        "text_score": round(100 * text_correct / max(total, 1), 2),
        "image_score": round(100 * image_correct / max(total, 1), 2),
        "group_score": round(100 * group_correct / max(total, 1), 2),
        "total_pairs": total,
    }


def eval_sugarcrepe(processor, model, device, wrapper=None):
    sc_base = os.path.join(DATA, "sugarcrepe")
    subtasks = [
        "SugarCrepe_add_att", "SugarCrepe_add_obj",
        "SugarCrepe_replace_att", "SugarCrepe_replace_obj", "SugarCrepe_replace_rel",
        "SugarCrepe_swap_att", "SugarCrepe_swap_obj",
    ]
    boxes, box_mask = make_grid_boxes()
    boxes, box_mask = boxes.to(device), box_mask.to(device)

    results = {}
    for subtask in subtasks:
        ds_path = os.path.join(sc_base, subtask)
        if not os.path.exists(ds_path):
            results[subtask] = {"accuracy": 0, "total": 0, "error": "not found"}
            continue

        ds = load_from_disk(ds_path)
        correct = 0
        total = 0

        for sample in ds:
            img = sample["image"].convert("RGB")
            labels_list = sample["tested_labels"]
            true_label = ds.features["true_label"].int2str(sample["true_label"])
            pos_cap = true_label
            neg_cap = labels_list[1] if labels_list[0] == true_label else labels_list[0]

            s_pos = sequence_log_likelihood(model, processor, img, pos_cap, device, wrapper, boxes, box_mask)
            s_neg = sequence_log_likelihood(model, processor, img, neg_cap, device, wrapper, boxes, box_mask)

            if s_pos > s_neg:
                correct += 1
            total += 1

        acc = 100 * correct / max(total, 1)
        results[subtask] = {"accuracy": round(acc, 2), "correct": correct, "total": total}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="registry name or checkpoint directory")
    parser.add_argument("--model", default="auto",
                        choices=["auto", "vanilla", "lora_only", "hra", "prefix"],
                        help="auto = from the checkpoint contents (hra/prefix modules present -> pathway on)")
    parser.add_argument("--variant", default=None, help="VARIANT_CONFIGS key for --model prefix (else variant.json)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from hra.checkpoints import load_for_eval
    model, processor, wrapper = load_for_eval(args.model, args.checkpoint, device, variant=args.variant)
    print(f"Loaded model={args.model} checkpoint={args.checkpoint} pathway={'on' if wrapper is not None else 'absent'}")

    print("=== Winoground Evaluation ===")
    try:
        wg_results = eval_winoground(processor, model, device, wrapper)
        print(f"  Text score:  {wg_results['text_score']}%")
        print(f"  Image score: {wg_results['image_score']}%")
        print(f"  Group score: {wg_results['group_score']}%")
    except Exception as e:
        print(f"  Winoground FAILED: {e}")
        import traceback; traceback.print_exc()
        wg_results = {"error": str(e)}

    print("\n=== SugarCrepe Evaluation ===")
    try:
        sc_results = eval_sugarcrepe(processor, model, device, wrapper)
        for name, r in sc_results.items():
            short = name.replace("SugarCrepe_", "")
            print(f"  {short:15s}: {r['accuracy']:6.2f}% ({r.get('correct', 0)}/{r.get('total', 0)})")
    except Exception as e:
        print(f"  SugarCrepe FAILED: {e}")
        import traceback; traceback.print_exc()
        sc_results = {"error": str(e)}

    output_dir = args.output_dir or os.environ.get("RUN_DIR", RUNS)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "compositionality_results.json"), "w") as f:
        json.dump({"winoground": wg_results, "sugarcrepe": sc_results}, f, indent=2)
    print(f"\nResults saved to {output_dir}/compositionality_results.json")


if __name__ == "__main__":
    main()
