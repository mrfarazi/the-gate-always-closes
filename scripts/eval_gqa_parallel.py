#!/usr/bin/env python3
"""Full GQA-testdev evaluation, sharded across GPUs.

    python scripts/eval_gqa_parallel.py --model vanilla                         --gpus 0,1,2,3
    python scripts/eval_gqa_parallel.py --model lora_only --checkpoint rms_ungated --gpus 0,1
    python scripts/eval_gqa_parallel.py --model hra       --checkpoint hra_v1      --gpus 0,1
    python scripts/eval_gqa_parallel.py --model prefix    --checkpoint raw_gated   --gpus 0,1

`--checkpoint` is a registry name (hra.hub) or a checkpoint directory. Modes: `vanilla` (no
LoRA), `lora_only` (the checkpoint's LoRA with the auxiliary pathway absent), `hra` (attention
bias on), `prefix` (prefix tokens on at the learned gate / scale 1 for ungated variants).

Region proposals: GQA testdev has no scene graphs, so `--boxes auto` (default) falls back to a
6x6 grid for every image without one and records `box_source` per question; `--boxes grid`
forces the grid. An empty proposal set is never evaluated (see docs/PROTOCOL.md).
"""
import argparse
import json
import os
import subprocess

from hra.paths import DATA, LLAVA_PATH, PYTHON, RUNS

MAX_PROPOSALS = 36

_COMMON = '''
import json, os, sys, torch
from PIL import Image as PILImage
DATA = "{DATA}"
LLAVA_PATH = "{LLAVA_PATH}"
device = torch.device("cuda")
with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
    questions = json.load(f)
with open("{qids_file}") as f:
    qids = json.load(f)
PROMPT = "USER: <image>\\n{{q}}\\nAnswer the question using a single word or phrase.\\nASSISTANT:"

def clean(text):
    pred = text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()
    for sep in [",", ".", ";", "!", "?"]:
        if sep in pred:
            pred = pred[:pred.index(sep)]
    return pred.strip().lower()
'''

_PLAIN = _COMMON + '''
from hra.checkpoints import load_vanilla, load_lora_only
model, processor = ({loader})
tokenizer = processor.tokenizer
predictions = {{}}
for i, qid in enumerate(qids):
    entry = questions[qid]
    gt = entry["answer"].strip().lower()
    image_id = entry.get("imageId", qid.split("_")[0])
    q_type = entry.get("types", {{}}).get("structural", "unknown")
    img = PILImage.open(os.path.join(DATA, "gqa/images", image_id + ".jpg")).convert("RGB")
    inputs = processor(text=PROMPT.format(q=entry["question"]), images=img, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             attention_mask=inputs["attention_mask"], max_new_tokens=20, do_sample=False)
    pred = clean(tokenizer.decode(gen[0], skip_special_tokens=True))
    predictions[qid] = {{"prediction": pred, "answer": gt, "correct": pred == gt, "type": q_type}}
    if (i + 1) % 200 == 0:
        print(f"  GPU {gpu_id}: {{i+1}}/{{len(qids)}}", flush=True)
with open("{shard_file}", "w") as f:
    json.dump(predictions, f)
os.remove("{qids_file}")
print(f"GPU {gpu_id}: done ({{len(predictions)}} questions)")
'''

_WRAPPED = _COMMON + '''
from hra.boxes import boxes_or_grid, grid_boxes_tensor
from hra.checkpoints import load_hra, load_prefix
wrapper, processor = ({loader})
tokenizer = processor.tokenizer
scene_graphs = {{}}
if "{boxes_mode}" == "auto":
    for sg_name in ["train_sceneGraphs.json", "val_sceneGraphs.json"]:
        sg_path = os.path.join(DATA, "gqa", sg_name)
        if os.path.exists(sg_path):
            with open(sg_path) as f:
                scene_graphs.update(json.load(f))
predictions = {{}}
for i, qid in enumerate(qids):
    entry = questions[qid]
    gt = entry["answer"].strip().lower()
    image_id = entry.get("imageId", qid.split("_")[0])
    q_type = entry.get("types", {{}}).get("structural", "unknown")
    img = PILImage.open(os.path.join(DATA, "gqa/images", image_id + ".jpg")).convert("RGB")
    img_w, img_h = img.size
    inputs = processor(text=PROMPT.format(q=entry["question"]), images=img, return_tensors="pt").to(device)
    boxes, box_mask, source = boxes_or_grid(scene_graphs.get(image_id), img_w, img_h, {max_proposals})
    boxes, box_mask = boxes.unsqueeze(0).to(device), box_mask.unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gen = wrapper.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                               attention_mask=inputs["attention_mask"], boxes=boxes, box_mask=box_mask,
                               max_new_tokens=20, do_sample=False)
    pred = clean(tokenizer.decode(gen[0], skip_special_tokens=True))
    predictions[qid] = {{"prediction": pred, "answer": gt, "correct": pred == gt, "type": q_type,
                        "box_source": source}}
    if (i + 1) % 200 == 0:
        print(f"  GPU {gpu_id}: {{i+1}}/{{len(qids)}}", flush=True)
with open("{shard_file}", "w") as f:
    json.dump(predictions, f)
os.remove("{qids_file}")
print(f"GPU {gpu_id}: done ({{len(predictions)}} questions)")
'''


def run_shard(gpu_id, shard_qids, model, checkpoint, variant, output_dir, boxes_mode):
    shard_file = os.path.join(output_dir, f"shard_{gpu_id}.json")
    qids_file = os.path.join(output_dir, f"shard_{gpu_id}_qids.json")
    with open(qids_file, "w") as f:
        json.dump(shard_qids, f)
    ck = json.dumps(checkpoint)
    if model == "vanilla":
        script = _PLAIN.format(loader="load_vanilla(device)", **_fmt(gpu_id, qids_file, shard_file))
    elif model == "lora_only":
        script = _PLAIN.format(loader=f"load_lora_only({ck}, device)", **_fmt(gpu_id, qids_file, shard_file))
    elif model == "hra":
        script = _WRAPPED.format(loader=f"load_hra({ck}, device)", boxes_mode=boxes_mode,
                                 max_proposals=MAX_PROPOSALS, **_fmt(gpu_id, qids_file, shard_file))
    elif model == "prefix":
        script = _WRAPPED.format(loader=f"load_prefix({ck}, device, variant={variant!r})",
                                 boxes_mode=boxes_mode, max_proposals=MAX_PROPOSALS,
                                 **_fmt(gpu_id, qids_file, shard_file))
    else:
        raise ValueError(model)
    proc = subprocess.Popen([PYTHON, "-c", script],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc, shard_file


def _fmt(gpu_id, qids_file, shard_file):
    return dict(DATA=DATA, LLAVA_PATH=LLAVA_PATH, qids_file=qids_file, shard_file=shard_file, gpu_id=gpu_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, choices=["vanilla", "lora_only", "hra", "prefix"])
    parser.add_argument("--checkpoint", default="none", help="registry name or checkpoint directory")
    parser.add_argument("--variant", default=None, help="VARIANT_CONFIGS key for --model prefix (else variant.json)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: 0..num_gpus-1)")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--boxes", default="auto", choices=["auto", "grid"])
    # legacy flags
    parser.add_argument("--baseline", action="store_true", help="same as --model vanilla")
    parser.add_argument("--lora_only", action="store_true", help="same as --model lora_only")
    args = parser.parse_args()

    model = args.model or ("vanilla" if args.baseline else "lora_only" if args.lora_only else "hra")
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(args.num_gpus))
    output_dir = args.output_dir or os.environ.get("RUN_DIR") or os.path.join(RUNS, "gqa_testdev", f"{model}_{os.path.basename(str(args.checkpoint)) or 'vanilla'}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
        questions = json.load(f)
    all_qids = list(questions.keys())
    if args.max_questions > 0:
        all_qids = all_qids[:args.max_questions]
    n = len(all_qids)
    shard_size = (n + len(gpus) - 1) // len(gpus)
    shards = [all_qids[i:i + shard_size] for i in range(0, n, shard_size)]
    print(f"GQA testdev eval (model={model}, checkpoint={args.checkpoint}, boxes={args.boxes}): "
          f"{n} questions across {len(shards)} GPUs {gpus[:len(shards)]}")

    procs = []
    for gpu_id, shard in zip(gpus, shards):
        proc, shard_file = run_shard(gpu_id, shard, model, args.checkpoint, args.variant, output_dir, args.boxes)
        procs.append((gpu_id, proc, shard_file))
        print(f"  GPU {gpu_id}: {len(shard)} questions")

    all_preds = {}
    for gpu_id, proc, sf in procs:
        stdout, _ = proc.communicate()
        print(stdout.decode() if stdout else "", end="")
        if proc.returncode != 0:
            print(f"  GPU {gpu_id}: FAILED (rc={proc.returncode})")
        if os.path.exists(sf):
            with open(sf) as f:
                all_preds.update(json.load(f))
            os.remove(sf)

    correct = sum(1 for v in all_preds.values() if v["correct"])
    total = len(all_preds)
    accuracy = 100.0 * correct / total if total > 0 else 0
    type_correct, type_total, box_sources = {}, {}, {}
    for v in all_preds.values():
        t = v.get("type", "unknown")
        type_total[t] = type_total.get(t, 0) + 1
        if v["correct"]:
            type_correct[t] = type_correct.get(t, 0) + 1
        if "box_source" in v:
            box_sources[v["box_source"]] = box_sources.get(v["box_source"], 0) + 1

    print(f"\nGQA testdev exact-match accuracy: {accuracy:.2f}% ({correct}/{total})")
    type_acc = {}
    for t in sorted(type_total):
        ta = 100.0 * type_correct.get(t, 0) / type_total[t]
        type_acc[t] = {"correct": type_correct.get(t, 0), "total": type_total[t], "accuracy": round(ta, 2)}
        print(f"  {t}: {ta:.2f}% ({type_correct.get(t, 0)}/{type_total[t]})")
    if box_sources:
        print(f"  box sources: {box_sources}")

    results = {"model": model, "checkpoint": str(args.checkpoint), "variant": args.variant,
               "boxes": args.boxes, "box_sources": box_sources,
               "accuracy": round(accuracy, 2), "correct": correct, "total": total,
               "accuracy_by_type": type_acc}
    with open(os.path.join(output_dir, "gqa_testdev_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "gqa_testdev_predictions.json"), "w") as f:
        json.dump(all_preds, f)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
