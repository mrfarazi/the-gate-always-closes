#!/usr/bin/env python3
"""Verify what the published S1 (hra_v1) GQA-testdev 'bias on' vs 'bias off' rows actually
compared.  READ-ONLY w.r.t. the repo: writes only into this directory.

Conditions on the first N testdev questions (dict order, as the published eval):
  A_pub_on   : exact published 'with bias' path  -> HRAWrapper.generate with get_boxes()
               (scene-graph boxes; testdev has none -> all-zero boxes and mask)
  D_wrap_off : same HRAWrapper forward/merge path but _hra_bias left None (bias absent)
  C_grid_on  : HRAWrapper.generate with 6x6 grid boxes (mask all ones)
  B_pub_off  : exact published 'without bias' path -> stock LlavaForConditionalGeneration
               + PeftModel LoRA, model.generate(input_ids, pixel_values, attention_mask)
For A and C we also record max|patch_bias| computed exactly as generate() computes it.
"""
import json
import os
import subprocess
import sys
import time

from hra.paths import DATA, RESULTS, RUNS  # noqa: E402
from hra import hub  # noqa: E402
OUT = os.environ.get("HRA_VERIFY_OUT", os.path.join(RUNS, "probes", "verify_zero_boxes"))
os.makedirs(OUT, exist_ok=True)

used = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"]
).decode().strip())
print(f"[pre] GPU0 memory used = {used} MiB", flush=True)
if used >= 55 * 1024:
    print("[abort] GPU0 has >= 55 GB in use", flush=True)
    sys.exit(2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402


N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
CKPT = hub.resolve("hra_v1")
MAX_PROPOSALS = 36
device = torch.device("cuda")

with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
    questions = json.load(f)
qids = list(questions.keys())[:N]

# ---- exact copy of the published box builder (src/eval_gqa_parallel.py:90-111) ----
scene_graphs = {}
for sg_name in ["train_sceneGraphs.json", "val_sceneGraphs.json"]:
    sg_path = os.path.join(DATA, "gqa", sg_name)
    if os.path.exists(sg_path):
        with open(sg_path) as f:
            scene_graphs.update(json.load(f))


def get_boxes(image_id, img_w, img_h, max_n=MAX_PROPOSALS):
    sg = scene_graphs.get(image_id, {})
    boxes = []
    for obj in sg.get("objects", {}).values():
        x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 1), obj.get("h", 1)
        boxes.append([x / max(img_w, 1), y / max(img_h, 1),
                      (x + w) / max(img_w, 1), (y + h) / max(img_h, 1)])
        if len(boxes) >= max_n:
            break
    box_tensor = torch.zeros(max_n, 4)
    box_mask = torch.zeros(max_n)
    for j, b in enumerate(boxes):
        box_tensor[j] = torch.tensor(b, dtype=torch.float32)
        box_mask[j] = 1.0
    return box_tensor, box_mask


def grid_boxes(image_id, img_w, img_h, n=6):
    s = 1.0 / n
    b = [[j * s, i * s, (j + 1) * s, (i + 1) * s] for i in range(n) for j in range(n)]
    return torch.tensor(b, dtype=torch.float32), torch.ones(n * n)


def clean(text):
    if "ASSISTANT:" in text:
        pred = text.split("ASSISTANT:")[-1].strip()
    else:
        pred = text.strip()
    for sep in [",", ".", ";", "!", "?"]:
        if sep in pred:
            pred = pred[: pred.index(sep)]
    return pred.strip().lower()


def prompt_for(q):
    return f"USER: <image>\n{q}\nAnswer the question using a single word or phrase.\nASSISTANT:"


n_sg = sum(1 for qid in qids if questions[qid]["imageId"] in scene_graphs)
print(f"[data] {len(qids)} questions; {n_sg} of them have a scene graph", flush=True)

results = {"n": len(qids), "ckpt": CKPT, "qids": qids, "n_with_scene_graph": n_sg,
           "versions": {"torch": torch.__version__}, "conditions": {}}
import transformers, peft  # noqa: E402
results["versions"]["transformers"] = transformers.__version__
results["versions"]["peft"] = peft.__version__

# =============================== HRA wrapper conditions ===============================
from hra.model import HRAWrapper  # noqa: E402

t0 = time.time()
wrapper, processor = HRAWrapper.load_for_eval(CKPT, device)
tok = processor.tokenizer
lm = wrapper.vlm.language_model
bm = lm.base_model.model if hasattr(lm, "base_model") else lm
results["wrapper_attn_impl"] = getattr(bm.config, "_attn_implementation", None)
results["bias_scale_beta"] = float(wrapper.hra_graph.cone_attn.bias_scale)
print(f"[load] HRAWrapper in {time.time()-t0:.0f}s attn={results['wrapper_attn_impl']} "
      f"beta={results['bias_scale_beta']:.4f}", flush=True)


@torch.no_grad()
def generate_no_bias(w, pixel_values, input_ids, attention_mask, **gen_kwargs):
    raw_vision, image_features = w._get_vision_features(pixel_values)
    inputs_embeds = w._get_embed_fn()(input_ids)
    merged_embeds, merged_attn, img_start = w._merge_with_vision(
        inputs_embeds, image_features, input_ids, attention_mask)
    w._hra_bias = None
    w._image_range = None
    pad_id = w.processor.tokenizer.pad_token_id or w.processor.tokenizer.eos_token_id
    return w.vlm.language_model.generate(
        inputs_embeds=merged_embeds, attention_mask=merged_attn,
        pad_token_id=pad_id, **gen_kwargs)


def run_wrapper(name, boxes_fn, no_bias=False):
    preds, maxbias, meanbias, nnan = {}, [], [], 0
    t0 = time.time()
    for i, qid in enumerate(qids):
        entry = questions[qid]
        gt = entry["answer"].strip().lower()
        image_id = entry.get("imageId", qid.split("_")[0])
        img = PILImage.open(f"{DATA}/gqa/images/{image_id}.jpg").convert("RGB")
        img_w, img_h = img.size
        inputs = processor(text=prompt_for(entry["question"]), images=img,
                           return_tensors="pt").to(device)
        boxes, box_mask = boxes_fn(image_id, img_w, img_h)
        boxes = boxes.unsqueeze(0).to(device)
        box_mask = box_mask.unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            raw_vision, _ = wrapper._get_vision_features(inputs["pixel_values"])
            pf = wrapper._extract_roi_features(raw_vision, boxes, box_mask)
            pb, _, _, _ = wrapper.hra_graph(pf, boxes, box_mask)
            pbf = pb.float()
            nnan += int(torch.isnan(pbf).sum())
            maxbias.append(float(pbf.abs().max()))
            meanbias.append(float(pbf.abs().mean()))
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if no_bias:
                gen = generate_no_bias(wrapper, inputs["pixel_values"], inputs["input_ids"],
                                       inputs["attention_mask"], max_new_tokens=20, do_sample=False)
            else:
                gen = wrapper.generate(
                    pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], boxes=boxes, box_mask=box_mask,
                    max_new_tokens=20, do_sample=False)
        pred = clean(tok.decode(gen[0], skip_special_tokens=True))
        preds[qid] = {"prediction": pred, "answer": gt, "correct": pred == gt}
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i+1}/{len(qids)} ({time.time()-t0:.0f}s)", flush=True)
    acc = 100.0 * sum(v["correct"] for v in preds.values()) / len(preds)
    res = {"preds": preds, "acc": acc, "max_abs_bias_over_items": max(maxbias),
           "mean_abs_bias_over_items": sum(meanbias) / len(meanbias),
           "n_items_with_nonzero_bias": sum(1 for m in maxbias if m > 0),
           "n_nan_bias_entries": nnan, "bias_used_in_generate": not no_bias,
           "seconds": time.time() - t0}
    print(f"[{name}] acc={acc:.2f} max|bias|={max(maxbias):.3e} nonzero_items="
          f"{res['n_items_with_nonzero_bias']} nan={nnan}", flush=True)
    return res


results["conditions"]["A_pub_on_zero_boxes"] = run_wrapper("A_pub_on", get_boxes)
results["conditions"]["D_wrapper_bias_absent"] = run_wrapper("D_wrap_off", get_boxes, no_bias=True)
results["conditions"]["C_grid_on"] = run_wrapper("C_grid_on", grid_boxes)

del wrapper
torch.cuda.empty_cache()

# =============================== published 'off' path ===============================
from transformers import LlavaForConditionalGeneration, AutoProcessor  # noqa: E402
from peft import PeftModel  # noqa: E402

t0 = time.time()
processor = AutoProcessor.from_pretrained(LLAVA_PATH)
model = LlavaForConditionalGeneration.from_pretrained(
    LLAVA_PATH, torch_dtype=torch.bfloat16).to(device)
model.language_model = PeftModel.from_pretrained(model.language_model, os.path.join(CKPT, "lora"))
model.eval()
tok = processor.tokenizer
results["lora_only_attn_impl"] = getattr(model.language_model.config, "_attn_implementation", None)
print(f"[load] lora_only in {time.time()-t0:.0f}s attn={results['lora_only_attn_impl']}", flush=True)

preds = {}
t0 = time.time()
for i, qid in enumerate(qids):
    entry = questions[qid]
    gt = entry["answer"].strip().lower()
    image_id = entry.get("imageId", qid.split("_")[0])
    img = PILImage.open(f"{DATA}/gqa/images/{image_id}.jpg").convert("RGB")
    inputs = processor(text=prompt_for(entry["question"]), images=img, return_tensors="pt").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             attention_mask=inputs["attention_mask"], max_new_tokens=20, do_sample=False)
    pred = clean(tok.decode(gen[0], skip_special_tokens=True))
    preds[qid] = {"prediction": pred, "answer": gt, "correct": pred == gt}
    if (i + 1) % 50 == 0:
        print(f"  [B_pub_off] {i+1}/{len(qids)} ({time.time()-t0:.0f}s)", flush=True)
acc = 100.0 * sum(v["correct"] for v in preds.values()) / len(preds)
results["conditions"]["B_pub_off_lora_only"] = {"preds": preds, "acc": acc, "seconds": time.time() - t0}
print(f"[B_pub_off] acc={acc:.2f}", flush=True)

# =============================== comparisons ===============================
pub_on = json.load(open(os.path.join(RESULTS, "published/hra_v1_sgpath_bias_on_gqa_testdev_predictions.json")))
pub_off = json.load(open(os.path.join(RESULTS, "published/hra_v1_lora_only_gqa_testdev_predictions.json")))
results["conditions"]["PUB_on_file"] = {"preds": {q: pub_on[q] for q in qids}}
results["conditions"]["PUB_off_file"] = {"preds": {q: pub_off[q] for q in qids}}


def flips(a, b):
    n = sum(1 for q in qids if a[q]["prediction"] != b[q]["prediction"])
    ex = [(q, a[q]["prediction"], b[q]["prediction"]) for q in qids
          if a[q]["prediction"] != b[q]["prediction"]][:8]
    return {"n_flips": n, "flip_pct": 100.0 * n / len(qids), "examples": ex}


C = results["conditions"]
pairs = [("A_pub_on_zero_boxes", "B_pub_off_lora_only"),
         ("A_pub_on_zero_boxes", "D_wrapper_bias_absent"),
         ("A_pub_on_zero_boxes", "C_grid_on"),
         ("D_wrapper_bias_absent", "B_pub_off_lora_only"),
         ("A_pub_on_zero_boxes", "PUB_on_file"),
         ("B_pub_off_lora_only", "PUB_off_file"),
         ("PUB_on_file", "PUB_off_file")]
results["flips"] = {f"{a} vs {b}": flips(C[a]["preds"], C[b]["preds"]) for a, b in pairs}
for k, v in results["flips"].items():
    print(f"[flips] {k}: {v['n_flips']}/{len(qids)} ({v['flip_pct']:.2f}%)", flush=True)

with open(os.path.join(OUT, f"verify_zero_boxes_n{len(qids)}.json"), "w") as f:
    json.dump(results, f, indent=1)
print("[done]", flush=True)
