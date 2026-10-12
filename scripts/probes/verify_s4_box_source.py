#!/usr/bin/env python3
"""Which box source produced the published S4 (prefix, no_dropout) 'with pathway' 59.40?
outputs/e2_ablations/s4_no_dropout_with_prefix.json has no launch script on disk. Re-generate
the first N of its predictions with (i) scene-graph boxes (testdev -> all-zero boxes/mask) and
(ii) 6x6 grid boxes, and see which reproduces the file. Also record the prefix tokens'
image-dependence under each protocol. Writes only into this directory.
"""
import json
import os
import subprocess
import sys
import time

from hra.paths import DATA, RESULTS, RUNS  # noqa: E402
from hra import hub  # noqa: E402
OUT = os.environ.get("HRA_VERIFY_OUT", os.path.join(RUNS, "probes", "verify_zero_boxes"))
used = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"]).decode().strip())
print(f"[pre] GPU0 memory used = {used} MiB", flush=True)
if used >= 55 * 1024:
    sys.exit(2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from hra.train_prefix_fixed import FixedPrefixWrapper  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CKPT = hub.resolve("no_dropout")
device = torch.device("cuda")
questions = json.load(open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")))
qids = list(questions.keys())[:N]
pub = json.load(open(os.path.join(RESULTS, "published/s4_no_dropout_with_prefix.json")))
off = json.load(open(os.path.join(RESULTS, "published/no_dropout_lora_only_gqa_testdev_predictions.json")))


def zero_boxes():
    return torch.zeros(1, 36, 4, device=device), torch.zeros(1, 36, device=device)


def grid_boxes(n=6):
    s = 1.0 / n
    b = [[j * s, i * s, (j + 1) * s, (i + 1) * s] for i in range(n) for j in range(n)]
    return torch.tensor(b).unsqueeze(0).to(device), torch.ones(1, n * n, device=device)


def clean(text):
    pred = text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()
    for sep in [",", ".", ";", "!", "?"]:
        if sep in pred:
            pred = pred[: pred.index(sep)]
    return pred.strip().lower()


wrapper, processor = FixedPrefixWrapper.load_for_eval(CKPT, device, k=4)
tok = processor.tokenizer
pg = wrapper.prefix_graph if hasattr(wrapper, "prefix_graph") else None
gate = None
for name, mod in wrapper.named_modules():
    if hasattr(mod, "gate") and isinstance(getattr(mod, "gate"), torch.nn.Parameter):
        gate = float(mod.gate)
        gate_name = name
        break
print(f"[load] gate={gate} ({gate_name if gate is not None else 'n/a'}) "
      f"ungated={getattr(getattr(pg, 'prefix_proj', None), 'ungated', 'n/a')}", flush=True)

res = {"n": N, "ckpt": CKPT, "gate": gate, "conditions": {}}
for cname, bfn in [("zero_boxes_sg_protocol", zero_boxes), ("grid_6x6", grid_boxes)]:
    preds, prefixes = {}, []
    t0 = time.time()
    for i, qid in enumerate(qids):
        e = questions[qid]
        img = PILImage.open(f"{DATA}/gqa/images/{e['imageId']}.jpg").convert("RGB")
        prompt = f"USER: <image>\n{e['question']}\nAnswer the question using a single word or phrase.\nASSISTANT:"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        boxes, mask = bfn()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            raw_vision, _ = wrapper._get_vision_features(inputs["pixel_values"])
            pf = wrapper._extract_roi_features(raw_vision, boxes, mask)
            ptoks = wrapper.prefix_graph(pf, boxes, mask)[0] if hasattr(wrapper, "prefix_graph") else None
            if ptoks is not None and len(prefixes) < 10:
                prefixes.append(ptoks.float().cpu())
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            gen = wrapper.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                                   attention_mask=inputs["attention_mask"], boxes=boxes, box_mask=mask,
                                   max_new_tokens=20, do_sample=False)
        pred = clean(tok.decode(gen[0], skip_special_tokens=True))
        preds[qid] = {"prediction": pred, "answer": e["answer"].strip().lower(),
                      "correct": pred == e["answer"].strip().lower()}
    match_pub = sum(preds[q]["prediction"] == pub[q]["prediction"] for q in qids)
    match_off = sum(preds[q]["prediction"] == off[q]["prediction"] for q in qids)
    pd = {}
    if prefixes:
        P = torch.stack(prefixes)  # (m, 1, K, D)
        pd = {"prefix_l2_mean": float(P.norm(dim=-1).mean()),
              "max_abs_diff_across_images": float((P - P[0:1]).abs().max()),
              "n_images_compared": int(P.shape[0])}
    res["conditions"][cname] = {"preds": preds, "n_match_published_s4_file": match_pub,
                                "n_match_lora_only_file": match_off, "prefix": pd,
                                "acc": 100.0 * sum(v["correct"] for v in preds.values()) / N,
                                "seconds": time.time() - t0}
    print(f"[{cname}] match_pub={match_pub}/{N} match_off={match_off}/{N} "
          f"acc={res['conditions'][cname]['acc']:.2f} prefix={pd}", flush=True)

res["published_s4_vs_lora_only_on_these"] = sum(pub[q]["prediction"] != off[q]["prediction"] for q in qids)
with open(os.path.join(OUT, f"verify_s4_box_source_n{N}.json"), "w") as f:
    json.dump(res, f, indent=1)
print("[done]", flush=True)
