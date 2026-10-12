#!/usr/bin/env python3
"""HRA training: attention bias injection via hyperbolic cone scores."""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from accelerate import Accelerator
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import geoopt

from hra.model import HRAGraph, HRAWrapper
from hra.model import VIS_DIM, EMBED_DIM, NUM_HEADS, MAX_PROPOSALS, LLM_DIM, MODEL_PATH, BASE
from hra.paths import RUNS
from hra.checkpoints import write_variant_json
from hra.hyperbolic import LorentzManifold
from hra.losses import KappaHingeLoss
from hra.dataset import GQAVisualDataset, collate_fn

BATCH_SIZE = 8
GRAD_ACCUM = 2
NUM_EPOCHS = 5
LR_VLM = 2e-5
LR_HYP = 1e-4
LORA_RANK = 16
LORA_TARGETS = ["q_proj", "v_proj"]
GRAD_SCALER_ALPHA_INIT = 0.001
GRAD_SCALER_ALPHA_MAX = 0.05
LAMBDA_ANGLE = 1.0
LAMBDA_ENT = 0.1
LAMBDA_KAPPA = 0.1
RUN_DIR = os.environ.get("RUN_DIR") or os.path.join(RUNS, "hra_" + time.strftime("%Y%m%d_%H%M%S"))
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
TELEMETRY_DIR = RUN_DIR


def build_label_ids(input_ids, answers, tokenizer):
    labels = input_ids.clone()
    labels[:] = -100
    for i, ans in enumerate(answers):
        ans_tokens = tokenizer.encode(" " + ans, add_special_tokens=False)
        seq = input_ids[i].tolist()
        seq_len = len(seq)
        ans_len = len(ans_tokens)
        for start in range(seq_len - ans_len, -1, -1):
            if seq[start:start + ans_len] == ans_tokens:
                labels[i, start:start + ans_len] = input_ids[i, start:start + ans_len]
                break
    return labels


def state_is_finite(state_dict):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                return False
    return True


def safe_float(x):
    try:
        if isinstance(x, torch.Tensor):
            x = x.item()
        if x != x or abs(x) == float('inf'):
            return None
        return float(x)
    except:
        return None


ABLATION_CHOICES = [
    "none", "euclidean", "symmetric", "lora_only",
    "no_ent", "no_angle", "k12", "k48",
    "random_bias", "spatial_bias",
    "detach_bias", "small_alpha", "frozen_bias",
    "eval_only_bias", "small_bias_init", "hyp_gradclip",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="8gpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--ablation", default="none", choices=ABLATION_CHOICES)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1,
                        help="training seed (-1 = unseeded, as in the paper's runs)")
    parser.add_argument("--smoke", action="store_true",
                        help="20 micro-steps at batch 4 on the current GPU (SMOKE_STEPS/SMOKE_BATCH env still honoured)")
    args = parser.parse_args()
    if args.smoke:
        os.environ.setdefault("SMOKE_STEPS", "20")
        os.environ.setdefault("SMOKE_BATCH", "4")
    if args.seed >= 0:
        import random
        import numpy as np
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ablation = args.ablation
    run_label = f"hra_{ablation}" if ablation != "none" else "hra"
    checkpoint_subdir = os.path.join(CHECKPOINT_DIR, run_label)
    smoke_steps = int(os.environ.get("SMOKE_STEPS", 0))
    smoke_batch = int(os.environ.get("SMOKE_BATCH", 0))
    num_epochs = args.epochs if args.epochs > 0 else NUM_EPOCHS

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM,
        mixed_precision="bf16",
    )

    telemetry_path = os.path.join(TELEMETRY_DIR, "hra_telemetry.jsonl")
    os.makedirs(os.path.dirname(telemetry_path), exist_ok=True)
    os.makedirs(checkpoint_subdir, exist_ok=True)

    if accelerator.is_main_process:
        print("Loading LLaVA model...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    for i in range(accelerator.num_processes):
        if accelerator.local_process_index == i:
            vlm = LlavaForConditionalGeneration.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if accelerator.num_processes > 1:
            torch.distributed.barrier()

    vlm.requires_grad_(False)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK, target_modules=LORA_TARGETS,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    vlm.language_model = get_peft_model(vlm.language_model, lora_config)

    if accelerator.is_main_process:
        vlm.language_model.print_trainable_parameters()

    max_proposals = MAX_PROPOSALS
    if ablation == "k12":
        max_proposals = 12
    elif ablation == "k48":
        max_proposals = 48

    hra_graph = HRAGraph(
        vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        max_proposals=max_proposals)

    if ablation == "euclidean":
        hra_graph.cone_attn.eta.requires_grad_(False)
        # Override manifold operations: use Euclidean L2 distance + cosine angle
        from hra.model import EuclideanConeAttention
        hra_graph.cone_attn = EuclideanConeAttention(embed_dim=EMBED_DIM)

    if ablation == "symmetric":
        hra_graph.symmetrize = True

    if ablation == "lora_only":
        hra_graph.requires_grad_(False)
        hra_graph.cone_attn.bias_scale = nn.Parameter(torch.tensor(0.0))
        hra_graph.cone_attn.bias_scale.requires_grad_(False)

    if ablation == "random_bias":
        from hra.model import RandomLearnedBias
        hra_graph.cone_attn = RandomLearnedBias(max_proposals=max_proposals)

    if ablation == "spatial_bias":
        from hra.model import SpatialProximityBias
        hra_graph.cone_attn = SpatialProximityBias()

    lambda_angle = LAMBDA_ANGLE
    lambda_ent = LAMBDA_ENT
    if ablation == "no_ent":
        lambda_ent = 0.0
    elif ablation == "no_angle":
        lambda_angle = 0.0

    # Tier 2 ablations: flags for training loop behavior
    detach_bias = (ablation == "detach_bias")
    frozen_bias_mode = (ablation == "frozen_bias")
    eval_only_bias = (ablation == "eval_only_bias")

    alpha_max = GRAD_SCALER_ALPHA_MAX
    if ablation == "small_alpha":
        alpha_max = 0.01

    if ablation == "small_bias_init":
        hra_graph.cone_attn.bias_scale = nn.Parameter(torch.tensor(-5.0))

    if frozen_bias_mode:
        hra_graph.cone_attn.bias_scale = nn.Parameter(torch.tensor(0.0))
        hra_graph.cone_attn.bias_scale.requires_grad_(False)

    if eval_only_bias:
        hra_graph.cone_attn.bias_scale = nn.Parameter(torch.tensor(0.0))
        hra_graph.cone_attn.bias_scale.requires_grad_(False)

    wrapper = HRAWrapper(vlm, processor, hra_graph)
    if detach_bias:
        wrapper.detach_bias = True
    if eval_only_bias:
        wrapper.disable_bias = True
    import gc; gc.collect()
    torch.cuda.empty_cache()

    hyp_params, vlm_params, kappa_params = [], [], []
    for name, param in wrapper.named_parameters():
        if not param.requires_grad:
            continue
        if name == "hra_graph.kappa":
            kappa_params.append(param)
        elif "hra_graph" in name:
            hyp_params.append(param)
        else:
            vlm_params.append(param)

    optimizer_vlm = torch.optim.AdamW(vlm_params, lr=LR_VLM, weight_decay=0.01)
    optimizer_hyp = geoopt.optim.RiemannianAdam(hyp_params, lr=LR_HYP) if hyp_params else None
    optimizer_kappa = torch.optim.Adam(kappa_params, lr=LR_HYP) if kappa_params else None

    kappa_hinge = KappaHingeLoss()

    bs = smoke_batch if smoke_batch > 0 else (args.batch_size if args.batch_size > 0 else BATCH_SIZE)
    train_dataset = GQAVisualDataset(split="train", processor=processor, max_proposals=max_proposals)
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False)

    wrapper = wrapper.to(accelerator.device)
    to_prepare = [optimizer_vlm, train_loader]
    if optimizer_hyp:
        to_prepare.append(optimizer_hyp)
    if optimizer_kappa:
        to_prepare.append(optimizer_kappa)
    prepared = accelerator.prepare(*to_prepare)
    optimizer_vlm = prepared[0]
    train_loader = prepared[1]
    if optimizer_hyp:
        optimizer_hyp = prepared[2]
    if optimizer_kappa:
        optimizer_kappa = prepared[2 + (1 if optimizer_hyp else 0)]
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]

    if args.resume:
        if accelerator.is_main_process:
            print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location="cpu")
        wrapper.hra_graph.load_state_dict(state["hra_graph"])

    grad_scaler_alpha = GRAD_SCALER_ALPHA_INIT
    global_step = 0
    best_loss = float('inf')

    if accelerator.is_main_process:
        print(f"Training HRA: phase={args.phase}, ablation={ablation}, batch={bs}, "
              f"grad_accum={GRAD_ACCUM}, epochs={num_epochs}")
        if smoke_steps > 0:
            print(f"SMOKE TEST MODE: {smoke_steps} steps only")

    telemetry_file = None
    if accelerator.is_main_process:
        telemetry_file = open(telemetry_path, "w")

    for epoch in range(num_epochs):
        wrapper.train()
        t0 = time.time()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        micro_step = 0
        bias_scale_val = None
        eta_val = None
        grad_norm_vlm = 0.0
        grad_norm_hyp = 0.0
        grad_norm_kappa = 0.0

        for step, batch in enumerate(train_loader):
            if smoke_steps > 0 and global_step >= smoke_steps:
                break

            labels = build_label_ids(batch["input_ids"], batch["answers"], tokenizer)
            labels = labels.to(batch["input_ids"].device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs, relation_points, ent_loss, angle_loss = wrapper(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    boxes=batch["boxes"],
                    box_mask=batch["box_mask"],
                    labels=labels,
                )

                loss_task = outputs.loss
                kappa = wrapper.hra_graph.kappa
                loss_kappa = kappa_hinge(kappa)
                loss_hyp = (angle_loss * lambda_angle +
                            ent_loss * lambda_ent +
                            loss_kappa * LAMBDA_KAPPA)
                total_loss = (loss_task + grad_scaler_alpha * loss_hyp) / GRAD_ACCUM

            total_loss.backward()

            if torch.isnan(total_loss).item() or torch.isinf(total_loss).item():
                if accelerator.is_main_process:
                    print(f"[NaN DETECTED] step={global_step}")
                raise RuntimeError(f"NaN loss at step {global_step}")

            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                if accelerator.num_processes > 1:
                    for p in trainable_params:
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

                has_nan_grad = False
                for p in trainable_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        p.grad.zero_()
                        has_nan_grad = True

                grad_norm_vlm = sum(p.grad.data.norm(2).item() ** 2 for p in vlm_params if p.grad is not None) ** 0.5
                grad_norm_hyp = sum(p.grad.data.norm(2).item() ** 2 for p in hyp_params if p.grad is not None) ** 0.5
                grad_norm_kappa = sum(p.grad.data.norm(2).item() ** 2 for p in kappa_params if p.grad is not None) ** 0.5

                bias_scale_val = safe_float(getattr(wrapper.hra_graph.cone_attn, 'bias_scale', 0))
                eta_val = safe_float(getattr(wrapper.hra_graph.cone_attn, 'eta', 0))

                if has_nan_grad:
                    if accelerator.is_main_process:
                        print(f"[GRAD SANITIZE] step={global_step}")
                    optimizer_vlm.zero_grad()
                    if optimizer_hyp:
                        optimizer_hyp.zero_grad()
                    if optimizer_kappa:
                        optimizer_kappa.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(vlm_params, max_norm=1.0)
                    hyp_clip = 0.1 if ablation == "hyp_gradclip" else 1.0
                    if hyp_params:
                        torch.nn.utils.clip_grad_norm_(hyp_params, max_norm=hyp_clip)
                    optimizer_vlm.step()
                    if optimizer_hyp:
                        optimizer_hyp.step()
                    if optimizer_kappa:
                        optimizer_kappa.step()

                    with torch.no_grad():
                        wrapper.hra_graph.kappa.data.clamp_(min=0.1, max=5.0)

                    optimizer_vlm.zero_grad()
                    if optimizer_hyp:
                        optimizer_hyp.zero_grad()
                    if optimizer_kappa:
                        optimizer_kappa.zero_grad()

                total_loss = total_loss * GRAD_ACCUM

            global_step += 1

            if grad_scaler_alpha < alpha_max:
                grad_scaler_alpha = min(alpha_max,
                    GRAD_SCALER_ALPHA_INIT + (alpha_max - GRAD_SCALER_ALPHA_INIT) * global_step / 100)

            if accelerator.is_main_process:
                loss_val = safe_float(total_loss)
                if loss_val is not None:
                    epoch_loss_sum += loss_val
                    epoch_loss_count += 1

                record = {
                    "step": global_step, "epoch": epoch,
                    "loss_total": safe_float(total_loss),
                    "loss_task": safe_float(loss_task),
                    "loss_angle": safe_float(angle_loss),
                    "loss_ent": safe_float(ent_loss),
                    "loss_kappa": safe_float(loss_kappa),
                    "kappa": safe_float(kappa),
                    "bias_scale": bias_scale_val if micro_step % GRAD_ACCUM == 0 else None,
                    "eta_cone": eta_val if micro_step % GRAD_ACCUM == 0 else None,
                    "grad_norm_vlm": safe_float(grad_norm_vlm) if micro_step % GRAD_ACCUM == 0 else None,
                    "grad_norm_hyp": safe_float(grad_norm_hyp) if micro_step % GRAD_ACCUM == 0 else None,
                    "grad_norm_kappa": safe_float(grad_norm_kappa) if micro_step % GRAD_ACCUM == 0 else None,
                    "grad_scaler_alpha": grad_scaler_alpha,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
                telemetry_file.write(json.dumps(record) + "\n")
                telemetry_file.flush()

                if smoke_steps > 0 or global_step % 50 == 0:
                    print(f"[E{epoch} S{global_step}] loss={loss_val} task={safe_float(loss_task)} "
                          f"angle={safe_float(angle_loss)} ent={safe_float(ent_loss)} "
                          f"kappa={safe_float(kappa)} bias_scale={bias_scale_val} "
                          f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        if smoke_steps > 0 and global_step >= smoke_steps and os.environ.get("SMOKE_SAVE") != "1":
            break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path = os.path.join(checkpoint_subdir, f"epoch{epoch}")
            os.makedirs(save_path, exist_ok=True)
            state = {
                "hra_graph": wrapper.hra_graph.state_dict(),
                "kappa": wrapper.hra_graph.kappa.data,
                "epoch": epoch, "global_step": global_step,
            }
            if state_is_finite(state):
                torch.save(state, os.path.join(save_path, "hra_modules.pt"))
                wrapper.vlm.language_model.save_pretrained(os.path.join(save_path, "lora"))
                write_variant_json(save_path, kind="hra", variant=None, ablation=ablation, epoch=epoch,
                                   seed=args.seed, backbone=MODEL_PATH, trainer="hra/train.py")
                torch.save(state, os.path.join(checkpoint_subdir, "latest_checkpoint.pt"))
                avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(checkpoint_subdir, "best")
                    os.makedirs(best_path, exist_ok=True)
                    torch.save(state, os.path.join(best_path, "hra_modules.pt"))
                    wrapper.vlm.language_model.save_pretrained(os.path.join(best_path, "lora"))
                    write_variant_json(best_path, kind="hra", variant=None, ablation=ablation, epoch=epoch,
                                       seed=args.seed, backbone=MODEL_PATH, trainer="hra/train.py")
                    print(f"Best checkpoint (loss={avg_loss:.4f})")
                print(f"Epoch {epoch} done in {time.time()-t0:.1f}s")
        if smoke_steps > 0:
            break

    if accelerator.is_main_process and telemetry_file:
        telemetry_file.close()
        print("HRA training complete.")


if __name__ == "__main__":
    main()
