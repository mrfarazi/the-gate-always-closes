#!/usr/bin/env python3
"""GQA spatial/relational subset evaluation.

Filters GQA testdev to 'rel' semantic type (5308 questions) and evaluates
each model on this relational reasoning subset. Reports accuracy + bootstrap CI.

This is the diagnostic eval: if hyperbolic models outperform soft_prompt
specifically on relational questions, we have the positive result.
"""

import argparse
import json
import os
import sys
import random
import subprocess

import numpy as np


from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS, PYTHON  # noqa: E402
from hra import hub  # noqa: E402
MODEL_PATH = LLAVA_PATH


def bootstrap_ci(correct_arr, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(42)
    n = len(correct_arr)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(correct_arr, size=n, replace=True)
        boot_means.append(sample.mean() * 100)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return round(lower, 2), round(upper, 2)


def filter_gqa_by_type(semantic_type="rel"):
    """Filter GQA testdev to a specific semantic type."""
    with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
        all_qs = json.load(f)

    filtered = {}
    for qid, q in all_qs.items():
        sem = q.get("types", {}).get("semantic", "unknown")
        if sem == semantic_type:
            filtered[qid] = q

    return filtered


def eval_model_on_subset(model_type, checkpoint, qids, gpu_id, output_path, max_questions=0,
                         variant=None):
    """Evaluate a model on a GQA subset.

    `checkpoint` is a registry name or directory. For prefix checkpoints the graph is rebuilt
    from `variant` (a VARIANT_CONFIGS key) or the checkpoint's variant.json; it is never
    inferred from the path, because the `no_rms`/`ungated` flags are not in the state dict
    and getting `ungated` wrong multiplies the prefix by tanh(0)=0 at inference.
    """
    if checkpoint is not None:
        checkpoint = hub.resolve(checkpoint)
    # Write qids to temp file
    qids_file = output_path + "_qids.json"
    with open(qids_file, "w") as f:
        json.dump(qids if max_questions == 0 else qids[:max_questions], f)

    if model_type in ("vanilla", "lora_only"):
        # lora_only = the checkpoint's LoRA weights with the auxiliary pathway ABSENT.
        # It isolates the training-time regularisation from the inference-time pathway.
        lora_block = ""
        if model_type == "lora_only" and checkpoint:
            lora_block = (f'\nfrom peft import PeftModel\n'
                          f'model.language_model = PeftModel.from_pretrained('
                          f'model.language_model, "{checkpoint}/lora")\n'
                          f'model = model.to(device)\n')
        script = f'''
import json, os, sys, torch
from PIL import Image as PILImage
from transformers import LlavaForConditionalGeneration, AutoProcessor

DATA = "{DATA}"
LLAVA_PATH = "{LLAVA_PATH}"
device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(LLAVA_PATH)
model = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_PATH, torch_dtype=torch.bfloat16).to(device){lora_block}
model.eval()
tokenizer = processor.tokenizer

with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
    questions = json.load(f)
with open("{qids_file}") as f:
    qids = json.load(f)

predictions = {{}}
for i, qid in enumerate(qids):
    entry = questions[qid]
    gt = entry["answer"].strip().lower()
    question = entry["question"]
    image_id = entry.get("imageId", qid.split("_")[0])
    img = PILImage.open(os.path.join(DATA, "gqa/images", image_id + ".jpg")).convert("RGB")
    prompt = f"USER: <image>\\n{{question}}\\nAnswer the question using a single word or phrase.\\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             attention_mask=inputs["attention_mask"], max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    if "ASSISTANT:" in text: pred = text.split("ASSISTANT:")[-1].strip()
    else: pred = text.strip()
    for sep in [",",".",";","!","?"]:
        if sep in pred: pred = pred[:pred.index(sep)]
    pred = pred.strip().lower()
    predictions[qid] = {{"prediction": pred, "answer": gt, "correct": pred == gt}}
    if (i+1) % 500 == 0:
        print(f"  {{i+1}}/{{len(qids)}}", flush=True)

with open("{output_path}", "w") as f:
    json.dump(predictions, f)
os.remove("{qids_file}")
print(f"Done ({{len(predictions)}} questions)")
'''
    elif model_type == "hra":
        script = f'''
import json, os, sys, torch
from PIL import Image as PILImage

DATA = "{DATA}"
LLAVA_PATH = "{LLAVA_PATH}"
device = torch.device("cuda")
from hra.model import HRAWrapper
wrapper, processor = HRAWrapper.load_for_eval("{checkpoint}", device)
tokenizer = processor.tokenizer

from hra.boxes import boxes_or_grid
scene_graphs = {{}}
for sg_name in ["train_sceneGraphs.json", "val_sceneGraphs.json"]:
    sg_path = os.path.join(DATA, "gqa", sg_name)
    if os.path.exists(sg_path):
        with open(sg_path) as f:
            scene_graphs.update(json.load(f))

def get_boxes(image_id, img_w, img_h, max_n=36):
    bt, bm, _source = boxes_or_grid(scene_graphs.get(image_id), img_w, img_h, max_n)
    return bt, bm

with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
    questions = json.load(f)
with open("{qids_file}") as f:
    qids = json.load(f)

predictions = {{}}
for i, qid in enumerate(qids):
    entry = questions[qid]
    gt = entry["answer"].strip().lower()
    question = entry["question"]
    image_id = entry.get("imageId", qid.split("_")[0])
    img = PILImage.open(os.path.join(DATA, "gqa/images", image_id + ".jpg")).convert("RGB")
    img_w, img_h = img.size
    prompt = f"USER: <image>\\n{{question}}\\nAnswer the question using a single word or phrase.\\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
    boxes, box_mask = get_boxes(image_id, img_w, img_h)
    boxes = boxes.unsqueeze(0).to(device)
    box_mask = box_mask.unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        gen = wrapper.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"], boxes=boxes, box_mask=box_mask,
                               max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    if "ASSISTANT:" in text: pred = text.split("ASSISTANT:")[-1].strip()
    else: pred = text.strip()
    for sep in [",",".",";","!","?"]:
        if sep in pred: pred = pred[:pred.index(sep)]
    pred = pred.strip().lower()
    predictions[qid] = {{"prediction": pred, "answer": gt, "correct": pred == gt}}
    if (i+1) % 500 == 0:
        print(f"  {{i+1}}/{{len(qids)}}", flush=True)

with open("{output_path}", "w") as f:
    json.dump(predictions, f)
os.remove("{qids_file}")
print(f"Done ({{len(predictions)}} questions)")
'''
    elif model_type == "prefix":
        # The prefix graph is rebuilt from the variant name (or variant.json); see hra.checkpoints.
        build_graph = ('from hra.checkpoints import build_prefix_graph_from_dir\n'
                       f'pg, _variant = build_prefix_graph_from_dir("{checkpoint}", {variant!r})')
        graph_class = "FixedPrefixGraph"
        script = f'''
import json, os, sys, torch
from PIL import Image as PILImage
DATA = "{DATA}"
LLAVA_PATH = "{LLAVA_PATH}"
device = torch.device("cuda")
from hra.train_prefix_fixed import FixedPrefixWrapper
from hra.checkpoints import load_llava, attach_lora

vlm, processor = load_llava()
attach_lora(vlm, "{checkpoint}")
{build_graph}
print("[graph]", type(pg).__name__,
      "ungated=", getattr(getattr(pg, "prefix_proj", None), "ungated", "n/a"),
      "no_rms=", getattr(getattr(pg, "prefix_proj", None), "no_rms", "n/a"), flush=True)
wrapper = FixedPrefixWrapper(vlm, processor, pg).to(device).eval()
tokenizer = processor.tokenizer

def make_grid_boxes(n=6):
    boxes, step = [], 1.0/n
    for i in range(n):
        for j in range(n):
            boxes.append([j*step, i*step, (j+1)*step, (i+1)*step])
    return torch.tensor(boxes).unsqueeze(0).to(device), torch.ones(n*n).unsqueeze(0).to(device)
grid_boxes, grid_mask = make_grid_boxes()

scene_graphs = {{}}
for sg_name in ["train_sceneGraphs.json", "val_sceneGraphs.json"]:
    sg_path = os.path.join(DATA, "gqa", sg_name)
    if os.path.exists(sg_path):
        with open(sg_path) as f:
            scene_graphs.update(json.load(f))

def get_boxes(image_id, img_w, img_h, max_n=36):
    sg = scene_graphs.get(image_id, {{}})
    boxes = []
    for obj in sg.get("objects", {{}}).values():
        x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 1), obj.get("h", 1)
        boxes.append([x / max(img_w, 1), y / max(img_h, 1),
                      (x + w) / max(img_w, 1), (y + h) / max(img_h, 1)])
        if len(boxes) >= max_n:
            break
    if not boxes:
        return grid_boxes, grid_mask
    bt = torch.zeros(max_n, 4); bm = torch.zeros(max_n)
    for j, b in enumerate(boxes): bt[j] = torch.tensor(b); bm[j] = 1.0
    return bt.unsqueeze(0).to(device), bm.unsqueeze(0).to(device)

with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
    questions = json.load(f)
with open("{qids_file}") as f:
    qids = json.load(f)

predictions = {{}}
for i, qid in enumerate(qids):
    entry = questions[qid]
    gt = entry["answer"].strip().lower()
    question = entry["question"]
    image_id = entry.get("imageId", qid.split("_")[0])
    img = PILImage.open(os.path.join(DATA, "gqa/images", image_id + ".jpg")).convert("RGB")
    img_w, img_h = img.size
    prompt = f"USER: <image>\\n{{question}}\\nAnswer the question using a single word or phrase.\\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
    boxes, box_mask = get_boxes(image_id, img_w, img_h)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        gen = wrapper.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"], boxes=boxes, box_mask=box_mask,
                               max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    if "ASSISTANT:" in text: pred = text.split("ASSISTANT:")[-1].strip()
    else: pred = text.strip()
    for sep in [",",".",";","!","?"]:
        if sep in pred: pred = pred[:pred.index(sep)]
    pred = pred.strip().lower()
    predictions[qid] = {{"prediction": pred, "answer": gt, "correct": pred == gt}}
    if (i+1) % 500 == 0:
        print(f"  {{i+1}}/{{len(qids)}}", flush=True)

with open("{output_path}", "w") as f:
    json.dump(predictions, f)
os.remove("{qids_file}")
print(f"Done ({{len(predictions)}} questions)")
'''

    proc = subprocess.Popen(
        [PYTHON, "-c", script],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.join(RUNS, "gqa_spatial"))
    parser.add_argument("--semantic_type", default="rel", help="GQA semantic type to filter")
    parser.add_argument("--max_questions", type=int, default=0, help="0 = all")
    parser.add_argument("--smoke", action="store_true", help="Run on 20 questions only")
    parser.add_argument("--models", default="",
                        help="Override the model list. Semicolon-separated specs of the "
                             "form name:mtype:ckpt:variant:gpu  (ckpt = registry name or "
                             "directory; variant may be empty for vanilla/hra or when the "
                             "checkpoint has a variant.json).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter questions
    filtered = filter_gqa_by_type(args.semantic_type)
    qids = list(filtered.keys())
    if args.smoke:
        qids = qids[:20]
    elif args.max_questions > 0:
        qids = qids[:args.max_questions]
    print(f"GQA {args.semantic_type} subset: {len(qids)} questions")

    # Define models to evaluate
    models = [
        ("vanilla", "vanilla", None, 0),
        ("soft_prompt", "prefix", "soft_prompt", 1),
        ("small_alpha", "hra", "small_alpha", 2),
        ("euclidean", "prefix", "euclidean", 3),
        ("rms_ungated", "prefix", "rms_ungated", 4),
    ]

    # name, mtype, ckpt, gpu  ->  extended with `variant`
    models = [(n, m, c, g, None) for (n, m, c, g) in models]
    if args.models:
        models = []
        for spec in args.models.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            name, mtype, ckpt, variant, gpu = spec.split(":")
            models.append((name, mtype, ckpt or None, int(gpu), variant or None))

    # Launch all in parallel
    procs = []
    for name, mtype, ckpt, gpu, variant in models:
        output_path = os.path.join(args.output_dir, f"{name}_{args.semantic_type}.json")
        print(f"  Launching {name} on GPU {gpu} (variant={variant})...")
        proc = eval_model_on_subset(mtype, ckpt, qids, gpu, output_path,
                                    args.max_questions, variant=variant)
        procs.append((name, proc, output_path))

    # Wait and collect
    results = {}
    for name, proc, output_path in procs:
        stdout, _ = proc.communicate()
        print(f"  {name}: rc={proc.returncode}")
        if stdout:
            print(stdout.decode()[-200:])
        if os.path.exists(output_path):
            with open(output_path) as f:
                preds = json.load(f)
            correct_arr = np.array([1 if v["correct"] else 0 for v in preds.values()])
            acc = correct_arr.mean() * 100
            ci_low, ci_high = bootstrap_ci(correct_arr)
            results[name] = {
                "accuracy": round(acc, 2),
                "ci_95": [ci_low, ci_high],
                "total": len(preds),
                "correct": int(correct_arr.sum()),
            }

    # Print results
    print(f"\n=== GQA {args.semantic_type} subset results ===")
    for name, r in sorted(results.items()):
        print(f"  {name:15s}: {r['accuracy']:.2f}% (95% CI: [{r['ci_95'][0]:.2f}, {r['ci_95'][1]:.2f}]) "
              f"({r['correct']}/{r['total']})")

    with open(os.path.join(args.output_dir, f"results_{args.semantic_type}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
