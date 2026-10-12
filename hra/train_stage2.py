#!/usr/bin/env python3
"""Stage 2: Compositional bias activation.

LoRA and HRA graph are frozen from Stage 1. Only trainable:
- bias_scale (β)
- per-head gates (32 sigmoid scalars)
- per-layer gates (32 sigmoid scalars)
- text→image bias projection

Objective: compositional margin loss + KL anchor + L1 gate sparsity.
"""

import argparse
import json
import math
import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel

from hra.model import HRAGraph, HRAWrapper
from hra.model import VIS_DIM, EMBED_DIM, NUM_HEADS, MAX_PROPOSALS, MODEL_PATH, BASE
from hra.hyperbolic import LorentzManifold

MARGIN = 2.0
LAMBDA_KL = 1.0
LAMBDA_L1 = 0.0  # Disabled — was killing gates before margin loss could use them
LR_STAGE2 = 5e-3  # Increased from 1e-3 — only 65 params, need faster learning
NUM_LAYERS = 32
NUM_HEADS_LLM = 32


class CompositionalDataset(Dataset):
    """Loads image + positive caption + hard negative caption."""
    def __init__(self, pairs_path, image_dir, processor, max_samples=0):
        with open(pairs_path) as f:
            self.pairs = json.load(f)
        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]
        self.image_dir = image_dir
        self.processor = processor
        # VG image IDs map to images in data/vg/ or data/gqa/images/
        # We'll try both paths
        self.gqa_img_dir = os.path.join(DATA, "gqa/images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image_id = pair["image_id"]
        positive = pair["positive"]
        # Randomly pick one type of negative
        neg_type = random.choice(list(pair["negatives"].keys()))
        negative = pair["negatives"][neg_type]

        # Try to find image
        img_path = os.path.join(self.gqa_img_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            # Return a dummy — will be filtered
            return None

        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "positive": positive,
            "negative": negative,
            "neg_type": neg_type,
            "image_id": image_id,
        }


def collate_compositional(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return batch


def make_grid_boxes(n_per_side=6, device="cpu"):
    boxes = []
    step = 1.0 / n_per_side
    for i in range(n_per_side):
        for j in range(n_per_side):
            boxes.append([j * step, i * step, (j + 1) * step, (i + 1) * step])
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    box_mask = torch.ones(len(boxes))
    return box_tensor.unsqueeze(0).to(device), box_mask.unsqueeze(0).to(device)


def compute_log_likelihood(wrapper, processor, image, caption, device, boxes, box_mask):
    """Compute log P(caption | image) scoring only caption tokens."""
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

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        score = wrapper.score(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            boxes=boxes, box_mask=box_mask,
            labels=labels)
    return score  # negative loss = log likelihood


class Stage2Wrapper(nn.Module):
    """Wraps HRAWrapper with per-head/per-layer gates for Stage 2."""
    def __init__(self, hra_wrapper):
        super().__init__()
        self.hra_wrapper = hra_wrapper
        # Per-head gates (initialized near 0 → sigmoid ≈ 0.5)
        self.head_gates = nn.Parameter(torch.zeros(NUM_HEADS_LLM))
        # Per-layer gates (initialized near 0)
        self.layer_gates = nn.Parameter(torch.zeros(NUM_LAYERS))
        # Override the bias_scale to be trainable
        self.hra_wrapper.hra_graph.cone_attn.bias_scale = nn.Parameter(torch.tensor(0.0))
        self.hra_wrapper.hra_graph.cone_attn.bias_scale.requires_grad_(True)

    def get_gated_bias(self, raw_bias):
        """Apply scalar gate to the raw (B, 1, P, P) bias. Keeps shape (B, 1, P, P) for SDPA compatibility."""
        head_weight = torch.sigmoid(self.head_gates).mean()
        layer_weight = torch.sigmoid(self.layer_gates).mean()
        return raw_bias * head_weight * layer_weight

    def forward_score(self, pixel_values, input_ids, attention_mask, boxes, box_mask, labels):
        """Score with gated bias — used for compositional margin loss."""
        # Get the raw bias from HRA graph
        raw_vision, image_features = self.hra_wrapper._get_vision_features(pixel_values)
        proposal_features = self.hra_wrapper._extract_roi_features(raw_vision.detach(), boxes, box_mask)
        patch_bias, _, _, _ = self.hra_wrapper.hra_graph(proposal_features, boxes, box_mask)

        # Apply gates
        gated_bias = self.get_gated_bias(patch_bias.unsqueeze(1))
        self.hra_wrapper._hra_bias = gated_bias

        # Merge vision + text
        inputs_embeds = self.hra_wrapper._get_embed_fn()(input_ids)
        merged_embeds, merged_attn, img_start = self.hra_wrapper._merge_with_vision(
            inputs_embeds, image_features, input_ids, attention_mask)
        num_patches = image_features.shape[1]
        self.hra_wrapper._image_range = (img_start, img_start + num_patches)

        expanded_labels = self.hra_wrapper._expand_labels(labels, input_ids, num_patches)

        try:
            outputs = self.hra_wrapper.vlm.language_model(
                inputs_embeds=merged_embeds, attention_mask=merged_attn,
                labels=expanded_labels)
        finally:
            self.hra_wrapper._hra_bias = None
            self.hra_wrapper._image_range = None

        return -outputs.loss


def safe_float(x):
    try:
        if isinstance(x, torch.Tensor):
            x = x.item()
        if x != x or abs(x) == float('inf'):
            return None
        return float(x)
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--phase", default="stage2")
    parser.add_argument("--seed", type=int, default=-1, help="training seed (-1 = unseeded, as in the paper)")
    args = parser.parse_args()
    if args.seed >= 0:
        import random
        import numpy as np
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    smoke_steps = int(os.environ.get("SMOKE_STEPS", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or os.environ.get("RUN_DIR", os.path.join(RUNS, "stage2"))
    os.makedirs(output_dir, exist_ok=True)

    # Generate hard negatives if not already done
    pairs_path = os.path.join(DATA, "compositional_pairs.json")
    if not os.path.exists(pairs_path):
        print("Generating hard negatives...")
        from generate_hard_negatives import load_vg_relationships, generate_negatives
        pairs, predicates, objects = load_vg_relationships()
        samples = generate_negatives(pairs, predicates, objects)
        random.shuffle(samples)
        with open(pairs_path, "w") as f:
            json.dump(samples[:50000], f)
        print(f"Generated {min(50000, len(samples))} pairs")

    # Disable flash/efficient SDPA backends — they can't handle modified attention masks
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Load Stage 1 checkpoint
    print(f"Loading Stage 1 checkpoint from {args.stage1_checkpoint}...")
    hra_wrapper, processor = HRAWrapper.load_for_eval(args.stage1_checkpoint, device)

    # Freeze everything from Stage 1
    for param in hra_wrapper.parameters():
        param.requires_grad_(False)

    # Create Stage 2 wrapper with gates
    stage2 = Stage2Wrapper(hra_wrapper).to(device)

    # Only train the new parameters
    trainable = [
        stage2.hra_wrapper.hra_graph.cone_attn.bias_scale,
        stage2.head_gates,
        stage2.layer_gates,
    ]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)}")

    optimizer = torch.optim.Adam(trainable, lr=LR_STAGE2)

    # Dataset
    dataset = CompositionalDataset(
        pairs_path=pairs_path,
        image_dir=os.path.join(DATA, "gqa/images"),
        processor=processor,
        max_samples=50000 if smoke_steps == 0 else 100,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=collate_compositional, num_workers=0)

    boxes, box_mask = make_grid_boxes(device=device)

    telemetry_path = os.path.join(output_dir, "stage2_telemetry.jsonl")
    telemetry_file = open(telemetry_path, "w")

    print(f"Stage 2 training: {len(dataset)} pairs, {args.epochs} epochs")
    if smoke_steps > 0:
        print(f"SMOKE TEST: {smoke_steps} steps only")

    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0
        epoch_count = 0

        for step, batch in enumerate(loader):
            if smoke_steps > 0 and global_step >= smoke_steps:
                break
            if batch is None:
                continue

            sample = batch[0]
            image = sample["image"]
            pos_caption = sample["positive"]
            neg_caption = sample["negative"]

            # Score positive
            pos_prompt = f"USER: <image>\nDescribe this image.\nASSISTANT: {pos_caption}"
            pos_inputs = processor(text=pos_prompt, images=image, return_tensors="pt").to(device)

            pos_tokenizer = processor.tokenizer
            pos_ids = pos_inputs["input_ids"][0].tolist()
            asst_tokens = pos_tokenizer.encode("ASSISTANT:", add_special_tokens=False)
            pos_start = None
            for p in range(len(pos_ids) - len(asst_tokens)):
                if pos_ids[p:p + len(asst_tokens)] == asst_tokens:
                    pos_start = p + len(asst_tokens)
                    break
            pos_labels = pos_inputs["input_ids"].clone()
            if pos_start:
                pos_labels[0, :pos_start] = -100

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pos_score = stage2.forward_score(
                    pos_inputs["pixel_values"], pos_inputs["input_ids"],
                    pos_inputs["attention_mask"], boxes, box_mask, pos_labels)

            # Score negative
            neg_prompt = f"USER: <image>\nDescribe this image.\nASSISTANT: {neg_caption}"
            neg_inputs = processor(text=neg_prompt, images=image, return_tensors="pt").to(device)

            neg_ids = neg_inputs["input_ids"][0].tolist()
            neg_start = None
            for p in range(len(neg_ids) - len(asst_tokens)):
                if neg_ids[p:p + len(asst_tokens)] == asst_tokens:
                    neg_start = p + len(asst_tokens)
                    break
            neg_labels = neg_inputs["input_ids"].clone()
            if neg_start:
                neg_labels[0, :neg_start] = -100

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                neg_score = stage2.forward_score(
                    neg_inputs["pixel_values"], neg_inputs["input_ids"],
                    neg_inputs["attention_mask"], boxes, box_mask, neg_labels)

            # Margin loss: want pos_score > neg_score by margin
            margin_loss = F.relu(MARGIN - (pos_score - neg_score))

            # L1 on gates
            l1_loss = LAMBDA_L1 * (
                torch.sigmoid(stage2.head_gates).sum() +
                torch.sigmoid(stage2.layer_gates).sum()
            )

            total_loss = margin_loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            loss_val = safe_float(total_loss)
            if loss_val:
                epoch_loss += loss_val
                epoch_count += 1

            record = {
                "step": global_step,
                "epoch": epoch,
                "margin_loss": safe_float(margin_loss),
                "l1_loss": safe_float(l1_loss),
                "total_loss": loss_val,
                "pos_score": safe_float(pos_score),
                "neg_score": safe_float(neg_score),
                "bias_scale": safe_float(stage2.hra_wrapper.hra_graph.cone_attn.bias_scale),
                "head_gates_mean": safe_float(torch.sigmoid(stage2.head_gates).mean()),
                "layer_gates_mean": safe_float(torch.sigmoid(stage2.layer_gates).mean()),
                "pos_gt_neg": 1 if pos_score > neg_score else 0,
            }
            telemetry_file.write(json.dumps(record) + "\n")
            telemetry_file.flush()

            if smoke_steps > 0 or global_step % 100 == 0:
                print(f"[E{epoch} S{global_step}] loss={loss_val} margin={safe_float(margin_loss)} "
                      f"pos={safe_float(pos_score):.3f} neg={safe_float(neg_score):.3f} "
                      f"β={safe_float(stage2.hra_wrapper.hra_graph.cone_attn.bias_scale):.4f} "
                      f"heads={safe_float(torch.sigmoid(stage2.head_gates).mean()):.3f}")

        if smoke_steps > 0 and global_step >= smoke_steps:
            break

        avg_loss = epoch_loss / max(epoch_count, 1)
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s, avg_loss={avg_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(output_dir, "stage2_checkpoint.pt")
        torch.save({
            "bias_scale": stage2.hra_wrapper.hra_graph.cone_attn.bias_scale.data,
            "head_gates": stage2.head_gates.data,
            "layer_gates": stage2.layer_gates.data,
            "epoch": epoch,
            "global_step": global_step,
        }, save_path)

        from hra.checkpoints import write_variant_json
        write_variant_json(os.path.dirname(save_path), kind="hra_stage2", variant=None,
                           stage1=args.stage1_checkpoint, seed=args.seed, epochs=args.epochs,
                           trainer="hra/train_stage2.py")
    telemetry_file.close()
    save_path = os.path.join(output_dir, "stage2_checkpoint.pt")
    torch.save({
        "bias_scale": stage2.hra_wrapper.hra_graph.cone_attn.bias_scale.data,
        "head_gates": stage2.head_gates.data,
        "layer_gates": stage2.layer_gates.data,
        "global_step": global_step,
    }, save_path)
    from hra.checkpoints import write_variant_json
    write_variant_json(os.path.dirname(save_path), kind="hra_stage2", variant=None,
                       stage1=args.stage1_checkpoint, seed=args.seed, epochs=args.epochs,
                       trainer="hra/train_stage2.py")
    print(f"Stage 2 complete. Checkpoint: {save_path}")
    print(f"Final β={stage2.hra_wrapper.hra_graph.cone_attn.bias_scale.item():.4f}")
    print(f"Head gates: {torch.sigmoid(stage2.head_gates).detach().cpu().tolist()}")
    print(f"Layer gates mean: {torch.sigmoid(stage2.layer_gates).mean().item():.4f}")


if __name__ == "__main__":
    main()
