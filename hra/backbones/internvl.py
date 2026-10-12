#!/usr/bin/env python3
"""Gate-closure run on InternVL3-8B (paper Sec. 3.3, Table 1).

Reproduces Mechanism B in its S5 form (raw, gated, caption-invariant hyperbolic prefix) on
InternVL3-8B and reports the final gate value, exactly as the LLaVA S1-S5 configurations and
the Qwen2-VL runs do.

Why the port is cheap: InternVL3-8B-hf has the LLaVA layout
(vision_tower / multi_modal_projector / language_model), standard image-token
substitution (image_token_id=151667), vision hidden 1024 (identical to LLaVA-1.5's CLIP
ViT-L, so FixedPrefixGraph's vis_dim is unchanged) and text hidden 3584 (identical to
Qwen2-VL, so only the prefix projector's output width changes).

Needs the backbone stack of requirements-backbones.txt (transformers >= 4.56).

  python scripts/train_internvl.py --seed 0 --steps 4000 --out runs/internvl3_seed0
  python scripts/train_internvl.py --seed 0 --steps 30 --smoke          # ~10 min incl. load
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS, LOCAL_FILES_ONLY  # noqa: E402
MAX_PROPOSALS = 36

# Backbones with the LLaVA layout that this wrapper supports. The paper reports internvl3-8b;
# the other two were smoke-tested only.
BACKBONES = {
    "internvl3-8b": dict(
        model_id="OpenGVLab/InternVL3-8B-hf",
        cls="InternVLForConditionalGeneration",
        llm_dim=3584, vis_dim=1024, vision_attr="vision_tower",
        nice="InternVL3-8B"),
    "llava-onevision-7b": dict(
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        cls="LlavaOnevisionForConditionalGeneration",
        llm_dim=3584, vis_dim=1152, vision_attr="vision_tower",
        nice="LLaVA-OneVision-7B (SigLIP-SO400M + Qwen2-7B)"),
    "qwen2.5-vl-7b": dict(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        cls="Qwen2_5_VLForConditionalGeneration",
        llm_dim=3584, vis_dim=1280, vision_attr="visual",
        nice="Qwen2.5-VL-7B"),
}


# --------------------------------------------------------------------- data
class GQARelDataset(Dataset):
    """GQA relational-subset QA, the same training distribution as the Qwen runs."""

    def __init__(self, processor, split="train", limit=0):
        with open(os.path.join(DATA, f"gqa/{split}_balanced_questions.json")) as f:
            q = json.load(f)
        self.qids = [k for k, v in q.items()
                     if v.get("types", {}).get("semantic") == "rel"]
        if limit:
            self.qids = self.qids[:limit]
        self.q = q
        self.processor = processor
        self.img_dir = os.path.join(DATA, "gqa/images")
        sg = {}
        for n in ("train_sceneGraphs.json", "val_sceneGraphs.json"):
            p = os.path.join(DATA, "gqa", n)
            if os.path.exists(p):
                sg.update(json.load(open(p)))
        self.sg = sg

    def __len__(self):
        return len(self.qids)

    def boxes_for(self, image_id, w, h):
        """Scene-graph boxes, normalised; grid fallback. Same convention as the paper."""
        objs = self.sg.get(image_id, {}).get("objects", {})
        bx = []
        for o in list(objs.values())[:MAX_PROPOSALS]:
            x, y, ow, oh = o.get("x", 0), o.get("y", 0), o.get("w", 1), o.get("h", 1)
            bx.append([x / max(w, 1), y / max(h, 1),
                       (x + ow) / max(w, 1), (y + oh) / max(h, 1)])
        if not bx:
            s = 1.0 / 6
            bx = [[j * s, i * s, (j + 1) * s, (i + 1) * s]
                  for i in range(6) for j in range(6)]
        b = torch.zeros(MAX_PROPOSALS, 4)
        m = torch.zeros(MAX_PROPOSALS)
        for i, v in enumerate(bx[:MAX_PROPOSALS]):
            b[i] = torch.tensor(v)
            m[i] = 1.0
        return b, m

    def __getitem__(self, i):
        qid = self.qids[i]
        e = self.q[qid]
        img = Image.open(os.path.join(self.img_dir, f"{e['imageId']}.jpg")).convert("RGB")
        b, m = self.boxes_for(e["imageId"], *img.size)
        return {"image": img, "question": e["question"],
                "answer": str(e["answer"]), "boxes": b, "box_mask": m}


def make_collate(processor):
    def collate(batch):
        msgs, answers = [], []
        for s in batch:
            msgs.append([{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": s["question"] +
                 "\nAnswer the question using a single word or phrase."}]}])
            answers.append(s["answer"])
        texts = [processor.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True) + a
                 for m, a in zip(msgs, answers)]
        enc = processor(images=[s["image"] for s in batch], text=texts,
                        return_tensors="pt", padding=True)
        enc["boxes"] = torch.stack([s["boxes"] for s in batch])
        enc["box_mask"] = torch.stack([s["box_mask"] for s in batch])
        enc["answers"] = answers
        return enc
    return collate


# -------------------------------------------------------------------- model
class InternVLGatedPrefix(torch.nn.Module):
    """InternVL3 + LoRA + gated caption-invariant hyperbolic prefix (Mechanism B)."""

    def __init__(self, model, processor, prefix_graph, k=4, vision_attr="vision_tower"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.prefix_graph = prefix_graph
        self.k = k
        self.vision_attr = vision_attr
        self.image_token_id = getattr(model.config, "image_token_id", None) \
            or getattr(model.config, "image_token_index")

    def _roi(self, pixel_values, boxes, box_mask):
        """Pre-projector vision features -> N proposal vectors (1024-d), as in the paper."""
        from torchvision.ops import roi_align
        vt = getattr(self.model.model, self.vision_attr)
        out = vt(pixel_values, output_hidden_states=True)
        h = out.hidden_states[-2]                    # (n_tiles, 1+P, 1024)
        h = h[:, 1:]                                 # drop CLS
        B = boxes.shape[0]
        if h.shape[0] != B:
            # InternVL tiles each image DYNAMICALLY (a 2-image batch produced 23 tiles,
            # i.e. 11+12), so the tile axis cannot be reshaped by batch size. We therefore
            # require batch_size=1 here and take the LAST tile, which is InternVL's global
            # thumbnail (use_thumbnail=True appends the whole downscaled image). That is
            # the only tile whose coordinate frame matches our normalised boxes -- an
            # arbitrary crop would misalign every RoI.
            assert B == 1, (f"InternVL dynamic tiling requires batch_size=1 "
                            f"(got B={B}, {h.shape[0]} tiles)")
            h = h[-1:]                               # global thumbnail, (1, P, 1024)
        n = h.shape[1]
        g = int(round(n ** 0.5))
        grid = h[:, :g * g].transpose(1, 2).reshape(B, -1, g, g).float()
        feats = []
        for i in range(B):
            bx = boxes[i].clone().float()
            bx[:, [0, 2]] *= g
            bx[:, [1, 3]] *= g
            rois = torch.cat([torch.zeros(bx.shape[0], 1, device=bx.device), bx], 1)
            p = roi_align(grid[i:i + 1], rois, output_size=(1, 1),
                          spatial_scale=1.0, aligned=True)
            feats.append(p.squeeze(-1).squeeze(-1))
        # The prefix graph runs in fp32 (same convention as the Qwen adapter),
        # so hand it fp32 and cast the prefix back to the embedding dtype later.
        return torch.stack(feats).float()

    def forward(self, pixel_values, input_ids, attention_mask, boxes, box_mask,
                labels=None, gamma_override=None):
        emb = self.model.get_input_embeddings()
        ie = emb(input_ids)
        img_feat = self.model.get_image_features(pixel_values=pixel_values)
        if isinstance(img_feat, (list, tuple)):
            img_feat = img_feat[0]
        # standard image-token substitution
        mask = (input_ids == self.image_token_id)
        ie = ie.clone()
        ie[mask] = img_feat.reshape(-1, ie.shape[-1]).to(ie.dtype)

        roi = self._roi(pixel_values, boxes, box_mask)
        pref, _, ent, ang = self.prefix_graph(roi, boxes.float(), box_mask.float(),
                                              ref_embeds=ie.float())
        pref = pref.to(ie.dtype)
        if gamma_override is not None:
            g = self.prefix_graph.prefix_proj
            base = pref / (torch.tanh(g.gate) + 1e-8) if not g.ungated else pref
            pref = base * gamma_override

        # splice K prefix embeddings immediately after the image span (LLaVA-style)
        B, S, D = ie.shape
        K = pref.shape[1]
        out_e = torch.zeros(B, S + K, D, device=ie.device, dtype=ie.dtype)
        out_a = torch.zeros(B, S + K, device=ie.device, dtype=attention_mask.dtype)
        out_l = None
        if labels is not None:
            out_l = torch.full((B, S + K), -100, device=ie.device, dtype=labels.dtype)
        for i in range(B):
            pos = mask[i].nonzero(as_tuple=True)[0]
            cut = int(pos[-1]) + 1 if len(pos) else 0
            out_e[i, :cut] = ie[i, :cut]
            out_a[i, :cut] = attention_mask[i, :cut]
            out_e[i, cut:cut + K] = pref[i]
            out_a[i, cut:cut + K] = 1
            out_e[i, cut + K:] = ie[i, cut:]
            out_a[i, cut + K:] = attention_mask[i, cut:]
            if labels is not None:
                out_l[i, :cut] = labels[i, :cut]
                out_l[i, cut + K:] = labels[i, cut:]
        out = self.model.model.language_model(inputs_embeds=out_e, attention_mask=out_a)
        hidden = out.last_hidden_state
        logits = self.model.lm_head(hidden)
        loss = None
        if out_l is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   out_l[:, 1:].reshape(-1), ignore_index=-100)
        return loss, logits, ent, ang


def build_labels(input_ids, answers, tok):
    labels = torch.full_like(input_ids, -100)
    for i, a in enumerate(answers):
        ids = tok.encode(a, add_special_tokens=False)
        seq = input_ids[i].tolist()
        for s in range(len(seq) - len(ids), -1, -1):
            if seq[s:s + len(ids)] == ids:
                labels[i, s:s + len(ids)] = input_ids[i, s:s + len(ids)]
                break
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr_lora", type=float, default=5e-5)
    ap.add_argument("--lr_graph", type=float, default=1e-4)
    ap.add_argument("--alpha_max", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--backbone", default="internvl3-8b", choices=list(BACKBONES))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")

    import transformers
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model
    from hra.train_prefix_fixed import FixedPrefixGraph
    import geoopt

    spec = BACKBONES[args.backbone]
    MODEL, LLM_DIM, VIS_DIM = spec["model_id"], spec["llm_dim"], spec["vis_dim"]
    print(f"[backbone] {spec['nice']}  ({MODEL})  "
          f"llm_dim={LLM_DIM} vis_dim={VIS_DIM}", flush=True)
    ModelCls = getattr(transformers, spec["cls"])
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=LOCAL_FILES_ONLY)
    model = ModelCls.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
    model.requires_grad_(False)
    model.model.language_model = get_peft_model(
        model.model.language_model,
        # NOTE: no task_type. We attach LoRA to the *inner* language model (a bare
        # Qwen2Model with no LM head) and compute logits ourselves through
        # model.lm_head, so PEFT's CAUSAL_LM wrapper is wrong here -- it expects
        # prepare_inputs_for_generation and raises AttributeError on a base model.
        LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                   lora_dropout=0.05, bias="none"))

    # Mechanism B, raw_gated recipe: no RMS, tanh gate init 0 -> matches Qwen g1b.
    pg = FixedPrefixGraph(vis_dim=VIS_DIM, k=4, prefix_dropout=0.0,
                          no_rms=True, ungated=False)
    pg.prefix_proj.llm_dim = LLM_DIM
    pg.prefix_proj.proj = torch.nn.Linear(pg.embed_dim, 4 * LLM_DIM)
    wrapper = InternVLGatedPrefix(model, processor, pg,
                                  vision_attr=spec["vision_attr"]).to(dev)

    lora_p = [p for n, p in wrapper.named_parameters()
              if p.requires_grad and "prefix_graph" not in n]
    graph_p = [p for n, p in wrapper.named_parameters()
               if p.requires_grad and "prefix_graph" in n and n != "prefix_graph.kappa"]
    kappa_p = [p for n, p in wrapper.named_parameters()
               if p.requires_grad and n == "prefix_graph.kappa"]
    opt_l = torch.optim.AdamW(lora_p, lr=args.lr_lora, weight_decay=0.01)
    opt_g = geoopt.optim.RiemannianAdam(graph_p, lr=args.lr_graph) if graph_p else None
    opt_k = torch.optim.Adam(kappa_p, lr=args.lr_graph) if kappa_p else None
    print(f"[internvl] trainable: lora={sum(p.numel() for p in lora_p):,} "
          f"graph={sum(p.numel() for p in graph_p):,}", flush=True)

    ds = GQARelDataset(processor, limit=2000 if args.smoke else 0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=make_collate(processor),
                    generator=torch.Generator().manual_seed(args.seed))
    print(f"[internvl] dataset={len(ds)} steps={args.steps}", flush=True)

    tel = open(os.path.join(args.out, "telemetry.jsonl"), "w")
    step = 0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            px = batch["pixel_values"].to(dev, dtype=torch.bfloat16)
            ids = batch["input_ids"].to(dev)
            am = batch["attention_mask"].to(dev)
            bx = batch["boxes"].to(dev)
            bm = batch["box_mask"].to(dev)
            lb = build_labels(batch["input_ids"], batch["answers"],
                              processor.tokenizer).to(dev)
            alpha = min(args.alpha_max, 0.001 + (args.alpha_max - 0.001) * step / 100)
            loss, _, ent, ang = wrapper(px, ids, am, bx, bm, labels=lb)
            total = loss + alpha * (1.0 * ang + 0.1 * ent)
            if not torch.isfinite(total):
                print(f"[internvl] non-finite at {step}, skipping", flush=True)
                opt_l.zero_grad(); step += 1; continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(lora_p, 1.0)
            if graph_p:
                torch.nn.utils.clip_grad_norm_(graph_p, 1.0)
            opt_l.step(); opt_l.zero_grad()
            if opt_g: opt_g.step(); opt_g.zero_grad()
            if opt_k:
                opt_k.step(); opt_k.zero_grad()
                with torch.no_grad():
                    pg.kappa.data.clamp_(0.1, 5.0)
            if step % 25 == 0:
                g = float(torch.tanh(pg.prefix_proj.gate).item())
                rec = {"step": step, "loss": float(loss), "ang": float(ang),
                       "ent": float(ent), "gate_tanh": g,
                       "kappa": float(pg.kappa)}
                tel.write(json.dumps(rec) + "\n"); tel.flush()
                print(f"[S{step}] loss={rec['loss']:.4f} gate={g:+.6f} "
                      f"kappa={rec['kappa']:.2f}", flush=True)
            step += 1
    tel.close()

    gate = float(torch.tanh(pg.prefix_proj.gate).item())
    json.dump({"backbone": spec["nice"], "model_id": MODEL,
               "seed": args.seed, "steps": args.steps,
               "final_gate_tanh": gate, "closed": abs(gate) < 0.05},
              open(os.path.join(args.out, "FINAL.json"), "w"), indent=2)
    torch.save({"prefix_graph": pg.state_dict()},
               os.path.join(args.out, "prefix_modules.pt"))
    model.model.language_model.save_pretrained(os.path.join(args.out, "lora"))
    from hra.checkpoints import write_variant_json
    write_variant_json(args.out, kind="prefix_internvl", variant="raw_gated", backbone=MODEL,
                       seed=args.seed, steps=args.steps, trainer="hra/backbones/internvl.py")
    print(f"[internvl] FINAL gate={gate:+.6f} closed={abs(gate) < 0.05}", flush=True)


if __name__ == "__main__":
    main()
