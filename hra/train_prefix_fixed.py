#!/usr/bin/env python3
"""Option A: Fixed prefix injection — BMVC mechanism with proper scaling.

Fixes the two engineering defects that caused prefix collapse:
1. RMS-normalize prefix tokens to match text embedding distribution
2. Tanh-gated scalar (init 0) gives continuous control, not binary on/off
3. Prefix dropout (p=0.5) during training for exposure robustness
4. Optional KL anchor on GQA outputs to prevent generation drift

Brute-force sweep of variants via --variant flag.
"""

import argparse
import json
import math
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

from hra.hyperbolic import LorentzManifold
from hra.losses import KappaHingeLoss
from hra.dataset import GQAVisualDataset, collate_fn
from hra.model import (DenseVisualRelations, VisualEntailmentLoss, VisualAngleLoss,
                   SpatialEncoder, HRAGraph)

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH
EMBED_DIM = 256
NUM_HEADS = 4
MAX_PROPOSALS = 36
LLM_DIM = 4096
VIS_DIM = 1024

BATCH_SIZE = 8
GRAD_ACCUM = 2
LR_VLM = 2e-5
LR_HYP = 1e-4
LORA_RANK = 16
LORA_TARGETS = ["q_proj", "v_proj"]
GRAD_SCALER_ALPHA_INIT = 0.001
GRAD_SCALER_ALPHA_MAX = 0.05
LAMBDA_ANGLE = 1.0
LAMBDA_ENT = 0.1
LAMBDA_KAPPA = 0.1
CHECKPOINT_DIR = os.path.join(os.environ.get("RUN_DIR", BASE), "checkpoints")
TELEMETRY_DIR = os.environ.get("RUN_DIR", os.path.join(BASE, "outputs"))


class GatedPrefixProjector(nn.Module):
    """Project hyperbolic nodes to LLM prefix tokens with proper scaling.

    Key fixes vs BMVC:
    - RMS normalization matches text embedding scale
    - Tanh gate (init 0) → starts as no-op, learned activation
    - Prefix dropout for exposure robustness
    """
    def __init__(self, embed_dim=EMBED_DIM, k=4, llm_dim=LLM_DIM, prefix_dropout=0.5,
                 no_rms=False, ungated=False):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.llm_dim = llm_dim
        self.prefix_dropout = prefix_dropout
        self.no_rms = no_rms
        self.ungated = ungated

        self.query = nn.Parameter(torch.randn(embed_dim + 1) * 0.01)
        self.proj = nn.Linear(embed_dim, k * llm_dim)
        self.rms_norm = nn.RMSNorm(llm_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))  # tanh gate, init 0

    def forward(self, relation_points, manifold, mask=None, ref_embeds=None):
        """
        relation_points: (B, N*N, D+1) on Lorentz manifold
        ref_embeds: (B, S, llm_dim) — reference text embeddings for scale matching
        Returns: (B, K, llm_dim) gated prefix tokens
        """
        B, M, D = relation_points.shape

        # Top-K selection by geodesic distance to learned query
        query_point = manifold.project_onto_manifold(self.query.unsqueeze(0).expand(B, -1))
        dists = manifold.geodesic_distance(
            query_point.unsqueeze(1).expand(B, M, D), relation_points)
        if mask is not None:
            dists = dists.masked_fill(mask == 0, 1e9)
        k = min(self.k, M)
        _, topk_idx = torch.topk(dists, k, dim=-1, largest=False)
        selected = torch.gather(relation_points, 1, topk_idx.unsqueeze(-1).expand(B, k, D))

        # Project to tangent space, then to LLM dim
        tangent = manifold.log_map_origin(selected)
        spatial = tangent[..., 1:]  # (B, K, embed_dim)
        # Pool K selected into single vector, then project to K*llm_dim
        pooled = spatial.mean(dim=1)  # (B, embed_dim)
        tokens = self.proj(pooled).view(B, self.k, self.llm_dim)

        # RMS normalize to match text embedding scale (unless disabled for 2×2 factorization)
        if not self.no_rms:
            tokens = self.rms_norm(tokens)

        # Scale to match reference embedding magnitude if available
        if ref_embeds is not None and not self.no_rms:
            with torch.no_grad():
                ref_scale = ref_embeds.norm(dim=-1).mean()
                tok_scale = tokens.norm(dim=-1).mean().clamp(min=1e-6)
            tokens = tokens * (ref_scale / tok_scale)

        # Tanh gate (unless disabled for 2×2 factorization ungated corner)
        if not self.ungated:
            gate_val = torch.tanh(self.gate)
            tokens = tokens * gate_val

        # Prefix dropout: randomly zero out prefix tokens during training
        if self.training and self.prefix_dropout > 0:
            drop_mask = torch.rand(B, 1, 1, device=tokens.device) > self.prefix_dropout
            tokens = tokens * drop_mask

        return tokens


class SoftPromptGraph(nn.Module):
    """Control: learned prefix tokens with no geometric backbone. Same parameter budget."""
    def __init__(self, k=4, llm_dim=LLM_DIM, prefix_dropout=0.5):
        super().__init__()
        self.k = k
        self.kappa = nn.Parameter(torch.tensor(1.0))  # dummy for interface compat
        self.prefix_tokens = nn.Parameter(torch.randn(k, llm_dim) * 0.02)
        self.rms_norm = nn.RMSNorm(llm_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.prefix_dropout = prefix_dropout

    def forward(self, features, boxes, mask=None, ref_embeds=None):
        B = features.shape[0]
        tokens = self.rms_norm(self.prefix_tokens).unsqueeze(0).expand(B, -1, -1)
        if ref_embeds is not None:
            with torch.no_grad():
                ref_scale = ref_embeds.norm(dim=-1).mean()
                tok_scale = tokens.norm(dim=-1).mean().clamp(min=1e-6)
            tokens = tokens * (ref_scale / tok_scale)
        gate_val = torch.tanh(self.gate)
        tokens = tokens * gate_val
        if self.training and self.prefix_dropout > 0:
            drop_mask = torch.rand(B, 1, 1, device=tokens.device) > self.prefix_dropout
            tokens = tokens * drop_mask
        ent_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        ang_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        return tokens, None, ent_loss, ang_loss


class RandomFixedPrefixGraph(nn.Module):
    """G2 NEFTune control: fixed random prefix tokens, RMS-matched to text scale,
    no learnable gate, no geometric losses. Isolates whether *any* auxiliary
    forward perturbation regularizes LoRA vs. specifically the geometric one."""
    def __init__(self, k=4, llm_dim=None, prefix_dropout=0.5, **kwargs):
        super().__init__()
        if llm_dim is None:
            llm_dim = LLM_DIM
        self.k = k
        # Dummy kappa for interface compatibility with other graphs
        self.kappa = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        # Fixed random init as BUFFER — never trained
        self.register_buffer("prefix_tokens_fixed",
                             torch.randn(k, llm_dim) * 0.02)
        # Frozen RMSNorm (weight fixed at 1.0)
        self.rms_norm = nn.RMSNorm(llm_dim)
        self.rms_norm.weight.requires_grad = False
        # Dummy gate for interface compatibility (never applied)
        self.gate = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.prefix_dropout = prefix_dropout

    def forward(self, features, boxes, mask=None, ref_embeds=None):
        B = features.shape[0]
        tokens = self.rms_norm(self.prefix_tokens_fixed).unsqueeze(0).expand(B, -1, -1)
        if ref_embeds is not None:
            with torch.no_grad():
                ref_scale = ref_embeds.norm(dim=-1).mean()
                tok_scale = tokens.norm(dim=-1).mean().clamp(min=1e-6)
            tokens = tokens * (ref_scale / tok_scale)
        # No gate applied (ungated by design)
        if self.training and self.prefix_dropout > 0:
            drop_mask = torch.rand(B, 1, 1, device=tokens.device) > self.prefix_dropout
            tokens = tokens * drop_mask
        ent_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        ang_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        return tokens, None, ent_loss, ang_loss


class EuclideanPrefixGraph(nn.Module):
    """Control: Euclidean geometry feeding gated prefix (same pipeline, flat space).

    NOTE: this control carries NO geometric losses
    (ent_loss = ang_loss = 0, see end of forward). It therefore sits in the
    *no-geometric-loss* cluster and cannot separate "flat geometry" from
    "no containment objective". For the control that isolates geometry with the
    losses held on, use `EuclideanGeoPrefixGraph` / --variant euclidean_geo.
    """
    def __init__(self, vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 max_proposals=MAX_PROPOSALS, k=4, prefix_dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.kappa = nn.Parameter(torch.tensor(1.0))  # dummy
        self.dense_relations = DenseVisualRelations(
            vis_dim=vis_dim, embed_dim=embed_dim, num_heads=num_heads)
        self.relation_to_features = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.relation_to_features.weight, gain=0.1)
        nn.init.zeros_(self.relation_to_features.bias)
        self.proj = nn.Linear(embed_dim, k * LLM_DIM)
        self.rms_norm = nn.RMSNorm(LLM_DIM)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.k = k
        self.prefix_dropout = prefix_dropout

    def forward(self, features, boxes, mask=None, ref_embeds=None):
        B, N, _ = features.shape
        rel_features = self.dense_relations(features, boxes, mask)
        rel_flat = rel_features.view(B, N * N, self.embed_dim)
        rel_proj = self.relation_to_features(rel_flat)
        pooled = rel_proj.mean(dim=1)  # (B, embed_dim)
        tokens = self.proj(pooled).view(B, self.k, LLM_DIM)
        tokens = self.rms_norm(tokens)
        if ref_embeds is not None:
            with torch.no_grad():
                ref_scale = ref_embeds.norm(dim=-1).mean()
                tok_scale = tokens.norm(dim=-1).mean().clamp(min=1e-6)
            tokens = tokens * (ref_scale / tok_scale)
        gate_val = torch.tanh(self.gate)
        tokens = tokens * gate_val
        if self.training and self.prefix_dropout > 0:
            drop_mask = torch.rand(B, 1, 1, device=tokens.device) > self.prefix_dropout
            tokens = tokens * drop_mask
        # No geometric losses for Euclidean
        ent_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        ang_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        return tokens, None, ent_loss, ang_loss


class EuclideanGeometry(nn.Module):
    """Flat-space drop-in for LorentzManifold, exposing the same method names.

    Control: "does a spatial containment loss applied directly in Euclidean
    space yield the same regularizing effect?"

    Because this shim exposes exterior_angle / half_aperture / einstein_midpoint /
    geodesic_distance / log_map_origin / project_onto_manifold with the same
    signatures as LorentzManifold, `VisualEntailmentLoss`, `VisualAngleLoss` and
    `GatedPrefixProjector` are reused UNCHANGED. The IoA thresholds (0.8 / 0.05),
    the eta parameter, the masking, the relu/mean reductions and the loss weights
    are therefore bit-identical to the hyperbolic run. Geometry is the only variable.

    Point layout matches the Lorentz convention: (..., D+1) with [...,0] a dummy
    time coordinate held at 0 and [...,1:] the flat embedding, so every tensor
    shape downstream is unchanged.

    Substitutions (exactly as described in paper Sec. 2.4 "Controls"):
      exterior_angle    -> cosine angle between the flat vectors
      half_aperture     -> arcsin(K / ||p||), K=0.1
      einstein_midpoint -> arithmetic mean
    """

    K_CONE = 0.1

    def __init__(self):
        super().__init__()
        # kept for API parity with LorentzManifold; never used in flat space
        self.kappa = torch.tensor(1.0)

    def project_onto_manifold(self, x):
        # Flat space: no constraint surface. Hold the dummy time coord at 0.
        out = x.clone()
        out[..., 0] = 0.0
        return out

    def log_map_origin(self, x):
        # In flat space the point IS its own tangent vector at the origin.
        return self.project_onto_manifold(x)

    def exp_map_origin(self, v):
        return self.project_onto_manifold(v)

    def exterior_angle(self, p, q):
        """Cosine angle at the origin between the flat vectors (replaces phi)."""
        p_space = p[..., 1:]
        q_space = q[..., 1:]
        norm_p = torch.norm(p_space, dim=-1)
        norm_q = torch.norm(q_space, dim=-1)
        dot = torch.sum(p_space * q_space, dim=-1)
        cos_angle = dot / (norm_p * norm_q + 1e-6)
        cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-5, max=1.0 - 1e-5)
        return torch.acos(cos_angle)

    def half_aperture(self, q):
        """Euclidean cone half-aperture: arcsin(K / ||q||). Same form as hyperbolic."""
        q_space = q[..., 1:]
        q_space_norm = torch.clamp(torch.norm(q_space, dim=-1), min=1e-6)
        sin_arg = self.K_CONE / q_space_norm
        sin_arg = torch.clamp(sin_arg, min=-1.0 + 1e-5, max=1.0 - 1e-5)
        return torch.asin(sin_arg)

    def einstein_midpoint(self, points, weights=None):
        """Arithmetic mean over the aggregation axis (replaces Einstein midpoint)."""
        if weights is None:
            midpoint = points.mean(dim=-2)
        else:
            w = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
            midpoint = torch.sum(w.unsqueeze(-1) * points, dim=-2)
        return self.project_onto_manifold(midpoint)

    def geodesic_distance(self, p, q):
        """Euclidean L2 distance on the flat parts."""
        return torch.norm(p[..., 1:] - q[..., 1:], dim=-1)


class EuclideanGeoPrefixGraph(nn.Module):
    """Euclidean control WITH the geometric losses.

    Identical to FixedPrefixGraph in every respect except that the Lorentz
    manifold is replaced by EuclideanGeometry. Same DenseVisualRelations, same
    linear projection + norm clamp, same VisualEntailmentLoss / VisualAngleLoss
    instances, same GatedPrefixProjector, same RMS + scale-match + gate + dropout.
    There is no curvature parameter, so no kappa hinge applies.
    """
    def __init__(self, vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 max_proposals=MAX_PROPOSALS, k=4, prefix_dropout=0.5,
                 no_rms=False, ungated=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_proposals = max_proposals
        self.no_rms = no_rms
        self.ungated = ungated
        self.geometry = EuclideanGeometry()
        # Flat space has no curvature. Registered as a BUFFER (not a Parameter) so it
        # (a) satisfies the training loop's `prefix_graph.kappa` / `.data.clamp_()`
        # contract, (b) moves with .to(device), (c) is absent from named_parameters()
        # so no kappa optimizer group is created, and (d) makes the kappa hinge
        # max(0, 0.3 - 1.0)^2 == 0 exactly, contributing no loss and no gradient.
        self.register_buffer("kappa", torch.tensor(1.0))
        self.dense_relations = DenseVisualRelations(
            vis_dim=vis_dim, embed_dim=embed_dim, num_heads=num_heads)
        self.relation_to_manifold = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.relation_to_manifold.weight, gain=0.1)
        nn.init.zeros_(self.relation_to_manifold.bias)
        self.entailment_loss = VisualEntailmentLoss()
        self.angle_loss = VisualAngleLoss()
        self.prefix_proj = GatedPrefixProjector(
            embed_dim=embed_dim, k=k, prefix_dropout=prefix_dropout,
            no_rms=no_rms, ungated=ungated)

    def forward(self, features, boxes, mask=None, ref_embeds=None):
        B, N, _ = features.shape

        rel_features = self.dense_relations(features, boxes, mask)
        rel_flat = rel_features.view(B, N * N, self.embed_dim)
        rel_projected = self.relation_to_manifold(rel_flat)
        # Same norm clamp as the hyperbolic path, for parity.
        tnorm = rel_projected.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rel_projected = rel_projected * (torch.clamp(tnorm, max=5.0) / tnorm)
        # Flat "points": dummy time coord 0, spatial part = the embedding.
        relation_points = torch.zeros(
            B, N * N, self.embed_dim + 1,
            device=features.device, dtype=rel_projected.dtype)
        relation_points[..., 1:] = rel_projected

        rel_points_2d = relation_points.view(B, N, N, self.embed_dim + 1)
        ent_loss = self.entailment_loss(rel_points_2d, boxes, self.geometry)
        node_points = self.geometry.einstein_midpoint(rel_points_2d)
        ang_loss = self.angle_loss(node_points, boxes, self.geometry, mask)

        rel_mask = None
        if mask is not None:
            rel_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).view(B, N * N)

        prefix_tokens = self.prefix_proj(
            relation_points, self.geometry, rel_mask, ref_embeds)

        return prefix_tokens, relation_points, ent_loss, ang_loss


class FixedPrefixGraph(nn.Module):
    """HRA graph that outputs gated prefix tokens instead of attention bias."""
    def __init__(self, vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 max_proposals=MAX_PROPOSALS, k=4, prefix_dropout=0.5,
                 no_rms=False, ungated=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_proposals = max_proposals
        self.no_rms = no_rms
        self.ungated = ungated
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.manifold = LorentzManifold(self.kappa)
        self.dense_relations = DenseVisualRelations(
            vis_dim=vis_dim, embed_dim=embed_dim, num_heads=num_heads)
        self.relation_to_manifold = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.relation_to_manifold.weight, gain=0.1)
        nn.init.zeros_(self.relation_to_manifold.bias)
        self.entailment_loss = VisualEntailmentLoss()
        self.angle_loss = VisualAngleLoss()
        self.prefix_proj = GatedPrefixProjector(
            embed_dim=embed_dim, k=k, prefix_dropout=prefix_dropout,
            no_rms=no_rms, ungated=ungated)

    def forward(self, features, boxes, mask=None, ref_embeds=None):
        self.manifold.kappa = self.kappa
        B, N, _ = features.shape

        rel_features = self.dense_relations(features, boxes, mask)
        rel_flat = rel_features.view(B, N * N, self.embed_dim)
        rel_projected = self.relation_to_manifold(rel_flat)
        tnorm = rel_projected.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rel_projected = rel_projected * (torch.clamp(tnorm, max=5.0) / tnorm)
        v = torch.zeros(B, N * N, self.embed_dim + 1,
                        device=features.device, dtype=features.dtype)
        v[..., 1:] = rel_projected
        relation_points = self.manifold.project_onto_manifold(
            self.manifold.exp_map_origin(v))

        rel_points_2d = relation_points.view(B, N, N, self.embed_dim + 1)
        ent_loss = self.entailment_loss(rel_points_2d, boxes, self.manifold)
        node_points = self.manifold.einstein_midpoint(rel_points_2d)
        ang_loss = self.angle_loss(node_points, boxes, self.manifold, mask)

        rel_mask = None
        if mask is not None:
            rel_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).view(B, N * N)

        prefix_tokens = self.prefix_proj(
            relation_points, self.manifold, rel_mask, ref_embeds)

        return prefix_tokens, relation_points, ent_loss, ang_loss


class FixedPrefixWrapper(nn.Module):
    """VLM wrapper with gated prefix injection (Option A)."""

    def __init__(self, vlm, processor, prefix_graph):
        super().__init__()
        self.vlm = vlm
        self.processor = processor
        self.prefix_graph = prefix_graph

    def _get_embed_fn(self):
        lm = self.vlm.language_model
        if hasattr(lm, 'base_model'):
            return lm.base_model.model.model.embed_tokens
        elif hasattr(lm, 'model') and hasattr(lm.model, 'model'):
            return lm.model.model.embed_tokens
        return lm.model.embed_tokens

    def _extract_roi_features(self, image_features, boxes, mask):
        from torchvision.ops import roi_align
        B, P, D = image_features.shape
        if int(P ** 0.5) ** 2 != P:
            image_features = image_features[:, 1:, :]
            P = P - 1
        grid_size = int(P ** 0.5)
        feat_grid = image_features.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
        N = boxes.shape[1]
        proposal_features = torch.zeros(B, N, D, device=image_features.device, dtype=image_features.dtype)
        for i in range(B):
            valid = mask[i].sum().int().item()
            if valid == 0:
                continue
            bxs = boxes[i, :valid].clone()
            bxs[:, [0, 2]] *= grid_size
            bxs[:, [1, 3]] *= grid_size
            batch_idx = torch.zeros(valid, 1, device=bxs.device)
            rois = torch.cat([batch_idx, bxs], dim=1)
            pooled = roi_align(feat_grid[i:i+1], rois, output_size=(1, 1),
                               spatial_scale=1.0, aligned=True)
            proposal_features[i, :valid] = pooled.squeeze(-1).squeeze(-1)
        return proposal_features

    def _get_vision_features(self, pixel_values):
        vision_outputs = self.vlm.vision_tower(pixel_values, output_hidden_states=True)
        raw_vision = vision_outputs.hidden_states[self.vlm.config.vision_feature_layer]
        selected = raw_vision[:, 1:]
        image_features = self.vlm.multi_modal_projector(selected)
        return raw_vision, image_features

    def forward(self, pixel_values, input_ids, attention_mask, boxes, box_mask, labels=None):
        raw_vision, image_features = self._get_vision_features(pixel_values)
        proposal_features = self._extract_roi_features(raw_vision.detach(), boxes, box_mask)

        # Get reference text embedding scale
        inputs_embeds = self._get_embed_fn()(input_ids)

        prefix_tokens, relation_points, ent_loss, ang_loss = self.prefix_graph(
            proposal_features, boxes, box_mask, ref_embeds=inputs_embeds)

        # Merge: [text_before_image, image_patches, PREFIX, text_after_image]
        B, S, D = inputs_embeds.shape
        num_patches = image_features.shape[1]
        K = prefix_tokens.shape[1]
        image_token_id = self.vlm.config.image_token_index
        new_len = S - 1 + num_patches + K

        merged = torch.zeros(B, new_len, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        merged_attn = torch.zeros(B, new_len, device=inputs_embeds.device, dtype=attention_mask.dtype)

        for i in range(B):
            img_pos = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged[i, :S] = inputs_embeds[i]
                merged_attn[i, :S] = attention_mask[i]
                continue
            pos = img_pos[0].item()
            merged[i, :pos] = inputs_embeds[i, :pos]
            merged_attn[i, :pos] = attention_mask[i, :pos]
            merged[i, pos:pos + num_patches] = image_features[i]
            merged_attn[i, pos:pos + num_patches] = 1
            pstart = pos + num_patches
            merged[i, pstart:pstart + K] = prefix_tokens[i]
            merged_attn[i, pstart:pstart + K] = 1
            merged[i, pstart + K:] = inputs_embeds[i, pos + 1:]
            merged_attn[i, pstart + K:] = attention_mask[i, pos + 1:]

        if labels is not None:
            expanded = torch.full((B, new_len), -100, dtype=labels.dtype, device=labels.device)
            for i in range(B):
                img_pos = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
                if len(img_pos) == 0:
                    expanded[i, :S] = labels[i]
                    continue
                pos = img_pos[0].item()
                expanded[i, :pos] = labels[i, :pos]
                expanded[i, pos + num_patches + K:] = labels[i, pos + 1:]
            labels = expanded

        outputs = self.vlm.language_model(
            inputs_embeds=merged, attention_mask=merged_attn, labels=labels)

        return outputs, relation_points, ent_loss, ang_loss

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, boxes, box_mask, **gen_kwargs):
        raw_vision, image_features = self._get_vision_features(pixel_values)
        proposal_features = self._extract_roi_features(raw_vision, boxes, box_mask)
        inputs_embeds = self._get_embed_fn()(input_ids)
        prefix_tokens, _, _, _ = self.prefix_graph(
            proposal_features, boxes, box_mask, ref_embeds=inputs_embeds)

        B, S, D = inputs_embeds.shape
        num_patches = image_features.shape[1]
        K = prefix_tokens.shape[1]
        image_token_id = self.vlm.config.image_token_index
        new_len = S - 1 + num_patches + K

        merged = torch.zeros(B, new_len, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        merged_attn = torch.zeros(B, new_len, device=inputs_embeds.device, dtype=attention_mask.dtype)
        for i in range(B):
            img_pos = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                merged[i, :S] = inputs_embeds[i]
                merged_attn[i, :S] = attention_mask[i]
                continue
            pos = img_pos[0].item()
            merged[i, :pos] = inputs_embeds[i, :pos]
            merged_attn[i, :pos] = attention_mask[i, :pos]
            merged[i, pos:pos + num_patches] = image_features[i]
            merged_attn[i, pos:pos + num_patches] = 1
            pstart = pos + num_patches
            merged[i, pstart:pstart + K] = prefix_tokens[i]
            merged_attn[i, pstart:pstart + K] = 1
            merged[i, pstart + K:] = inputs_embeds[i, pos + 1:]
            merged_attn[i, pstart + K:] = attention_mask[i, pos + 1:]

        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        return self.vlm.language_model.generate(
            inputs_embeds=merged, attention_mask=merged_attn,
            pad_token_id=pad_id, **gen_kwargs)

    @torch.no_grad()
    def score(self, pixel_values, input_ids, attention_mask, boxes, box_mask, labels):
        outputs, _, _, _ = self.forward(
            pixel_values, input_ids, attention_mask, boxes, box_mask, labels)
        return -outputs.loss.item()

    @classmethod
    def load_for_eval(cls, checkpoint_dir, device, k=4, variant=None):
        """Load a prefix checkpoint (registry name or directory). The variant is read from the
        checkpoint's variant.json, or must be passed explicitly; it is never inferred from the
        path (see hra.checkpoints)."""
        from hra.checkpoints import load_prefix
        return load_prefix(checkpoint_dir, device, variant=variant)


VARIANT_CONFIGS = {
    # Option A base variants
    "base":        {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic"},
    "k2":          {"k": 2, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic"},
    "k8":          {"k": 8, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic"},
    "no_dropout":  {"k": 4, "dropout": 0.0, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic"},
    "low_dropout": {"k": 4, "dropout": 0.3, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic"},
    "no_scale":    {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": False, "mode": "hyperbolic"},
    "small_alpha": {"k": 4, "dropout": 0.5, "alpha_max": 0.01, "scale_match": True,  "mode": "hyperbolic"},
    # Tier 1 controls
    "soft_prompt":   {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,  "mode": "soft_prompt"},
    "euclidean":     {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,  "mode": "euclidean"},
    # Euclidean containment control: flat-space graph WITH the
    # geometric losses, mirroring rms_ungated exactly (ungated, dropout 0.0) so it is a
    # like-for-like row against the hyperbolic rms_ungated.
    "euclidean_geo": {"k": 4, "dropout": 0.0, "alpha_max": 0.05, "scale_match": True,  "mode": "euclidean_geo", "ungated": True},
    # Adapter-capacity control: identical to `base` (the gated RMS
    # cell, so there is a gate trajectory to observe) except that LoRA is applied to all
    # seven linear projections. Tests whether gate closure is a capacity artefact.
    "lora_all_linear": {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,
                        "mode": "hyperbolic",
                        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"]},
    # 2×2 factorization
    "raw_gated":     {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": False, "mode": "hyperbolic", "no_rms": True},
    "rms_ungated":   {"k": 4, "dropout": 0.0, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic", "ungated": True},
    # Row 5: RMS + None + prefix dropout (the CFG rescue)
    "rms_dropout":   {"k": 4, "dropout": 0.5, "alpha_max": 0.05, "scale_match": True,  "mode": "hyperbolic", "ungated": True},
    # Fixed random prefix (RMS-matched, ungated, no geometric losses); no shipped result uses it.
    # `g2_no_geo` is the configuration of the run reported as "G2" in the paper: a gated
    # FixedPrefixGraph with alpha_max=0, which holds the geometric-loss ramp at its 0.001
    # starting value.
    "random_fixed":  {"k": 4, "dropout": 0.0, "alpha_max": 0.0,  "scale_match": True,  "mode": "random_fixed"},
    "g2_no_geo":     {"k": 4, "dropout": 0.0, "alpha_max": 0.0,  "scale_match": True,  "mode": "hyperbolic"},
}


def build_prefix_graph_for_eval(variant, k=None, prefix_dropout=0.0):
    """Construct a prefix graph EXACTLY as training constructed it, from VARIANT_CONFIGS.

    Single source of truth for evaluation. Inferring the graph class from the checkpoint
    PATH is unsafe in two ways:
      1. 'euclidean' matches both `euclidean` and `euclidean_geo`, silently loading the
         wrong class;
      2. `no_rms` / `ungated` are constructor flags that do NOT appear in the state dict,
         so a checkpoint trained with ungated=True would be rebuilt with ungated=False and
         then multiplied by tanh(gate)=tanh(0)=0, zeroing the prefix at inference.
    Passing the variant name explicitly removes both failure modes.
    """
    cfg = VARIANT_CONFIGS[variant]
    mode = cfg.get("mode", "hyperbolic")
    kk = cfg["k"] if k is None else k
    common = dict(vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                  max_proposals=MAX_PROPOSALS, k=kk, prefix_dropout=prefix_dropout)
    if mode == "soft_prompt":
        return SoftPromptGraph(k=kk, prefix_dropout=prefix_dropout)
    if mode == "euclidean":
        return EuclideanPrefixGraph(**common)
    if mode == "euclidean_geo":
        return EuclideanGeoPrefixGraph(
            no_rms=cfg.get("no_rms", False), ungated=cfg.get("ungated", False), **common)
    if mode == "random_fixed":
        return RandomFixedPrefixGraph(k=kk, prefix_dropout=prefix_dropout)
    return FixedPrefixGraph(
        no_rms=cfg.get("no_rms", False), ungated=cfg.get("ungated", False), **common)


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


def safe_float(x):
    try:
        if isinstance(x, torch.Tensor):
            x = x.item()
        if x != x or abs(x) == float('inf'):
            return None
        return float(x)
    except:
        return None


def state_is_finite(state_dict):
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base", choices=list(VARIANT_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--phase", default="prefix")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Training seed (init + data order). -1 = unseeded "
                             "(the paper's single-run LLaVA configurations were unseeded).")
    parser.add_argument("--smoke", action="store_true",
                        help="20 micro-steps at batch 4 (SMOKE_STEPS/SMOKE_BATCH env still honoured)")
    parser.add_argument("--run_tag", default="",
                        help="Suffix appended to run_label so parallel seeds do not collide.")
    args = parser.parse_args()
    if args.smoke:
        os.environ.setdefault("SMOKE_STEPS", "20")
        os.environ.setdefault("SMOKE_BATCH", "4")

    cfg = VARIANT_CONFIGS[args.variant]
    smoke_steps = int(os.environ.get("SMOKE_STEPS", 0))
    smoke_batch = int(os.environ.get("SMOKE_BATCH", 0))
    num_epochs = args.epochs

    # --- Training seed (multi-seed runs) ---
    # The paper's single-run LLaVA configurations were launched without a fixed seed and are
    # reported as independent runs rather than "seed 0".
    seed_generator = None
    if args.seed >= 0:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        seed_generator = torch.Generator()
        seed_generator.manual_seed(args.seed)

    run_label = f"prefix_{args.variant}"
    if args.run_tag:
        run_label = f"{run_label}_{args.run_tag}"
    checkpoint_subdir = os.path.join(CHECKPOINT_DIR, run_label)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM, mixed_precision="bf16")

    telemetry_path = os.path.join(TELEMETRY_DIR, f"{run_label}_telemetry.jsonl")
    os.makedirs(os.path.dirname(telemetry_path), exist_ok=True)
    os.makedirs(checkpoint_subdir, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Option A: Fixed Prefix — variant={args.variant}, config={cfg}")
        print(f"  run_label={run_label}  seed={args.seed if args.seed >= 0 else 'unseeded'}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    for i in range(accelerator.num_processes):
        if accelerator.local_process_index == i:
            vlm = LlavaForConditionalGeneration.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if accelerator.num_processes > 1:
            torch.distributed.barrier()

    vlm.requires_grad_(False)
    lora_targets = cfg.get("lora_targets", LORA_TARGETS)
    if accelerator.is_main_process and lora_targets != LORA_TARGETS:
        print(f"  LoRA targets overridden: {lora_targets}")
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK, target_modules=lora_targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    vlm.language_model = get_peft_model(vlm.language_model, lora_config)

    if accelerator.is_main_process:
        vlm.language_model.print_trainable_parameters()

    mode = cfg.get("mode", "hyperbolic")
    if mode == "soft_prompt":
        prefix_graph = SoftPromptGraph(k=cfg["k"], prefix_dropout=cfg["dropout"])
    elif mode == "euclidean":
        prefix_graph = EuclideanPrefixGraph(
            vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
            max_proposals=MAX_PROPOSALS, k=cfg["k"], prefix_dropout=cfg["dropout"])
    elif mode == "euclidean_geo":
        prefix_graph = EuclideanGeoPrefixGraph(
            vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
            max_proposals=MAX_PROPOSALS, k=cfg["k"], prefix_dropout=cfg["dropout"],
            no_rms=cfg.get("no_rms", False), ungated=cfg.get("ungated", False))
    elif mode == "random_fixed":
        prefix_graph = RandomFixedPrefixGraph(k=cfg["k"], prefix_dropout=cfg["dropout"])
    else:
        prefix_graph = FixedPrefixGraph(
            vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
            max_proposals=MAX_PROPOSALS, k=cfg["k"], prefix_dropout=cfg["dropout"],
            no_rms=cfg.get("no_rms", False), ungated=cfg.get("ungated", False))

    wrapper = FixedPrefixWrapper(vlm, processor, prefix_graph)

    def _write_variant(path, epoch):
        from hra.checkpoints import write_variant_json
        write_variant_json(path, kind="prefix", variant=args.variant,
                           graph_class=type(prefix_graph).__name__, config=cfg, k=cfg["k"],
                           seed=args.seed, run_tag=args.run_tag, epoch=epoch,
                           epochs=num_epochs, backbone=MODEL_PATH, trainer="hra/train_prefix_fixed.py")

    import gc; gc.collect()
    torch.cuda.empty_cache()

    hyp_params, vlm_params, kappa_params = [], [], []
    for name, param in wrapper.named_parameters():
        if not param.requires_grad:
            continue
        if name == "prefix_graph.kappa":
            kappa_params.append(param)
        elif "prefix_graph" in name:
            hyp_params.append(param)
        else:
            vlm_params.append(param)

    optimizer_vlm = torch.optim.AdamW(vlm_params, lr=LR_VLM, weight_decay=0.01)
    optimizer_hyp = geoopt.optim.RiemannianAdam(hyp_params, lr=LR_HYP) if hyp_params else None
    optimizer_kappa = torch.optim.Adam(kappa_params, lr=LR_HYP) if kappa_params else None
    kappa_hinge = KappaHingeLoss()

    bs = smoke_batch if smoke_batch > 0 else (args.batch_size if args.batch_size > 0 else BATCH_SIZE)
    train_dataset = GQAVisualDataset(split="train", processor=processor, max_proposals=MAX_PROPOSALS)
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
        generator=seed_generator)

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
    alpha_max = cfg["alpha_max"]
    grad_scaler_alpha = GRAD_SCALER_ALPHA_INIT
    global_step = 0
    best_loss = float('inf')

    if accelerator.is_main_process:
        print(f"Training: epochs={num_epochs}, batch={bs}, k={cfg['k']}, "
              f"dropout={cfg['dropout']}, alpha_max={alpha_max}")
        if smoke_steps > 0:
            print(f"SMOKE TEST: {smoke_steps} steps")

    telemetry_file = None
    if accelerator.is_main_process:
        telemetry_file = open(telemetry_path, "w")

    for epoch in range(num_epochs):
        wrapper.train()
        t0 = time.time()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        micro_step = 0

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
                kappa = wrapper.prefix_graph.kappa
                loss_kappa = kappa_hinge(kappa)
                loss_hyp = (angle_loss * LAMBDA_ANGLE +
                            ent_loss * LAMBDA_ENT +
                            loss_kappa * LAMBDA_KAPPA)
                total_loss = (loss_task + grad_scaler_alpha * loss_hyp) / GRAD_ACCUM

            total_loss.backward()

            if torch.isnan(total_loss).item() or torch.isinf(total_loss).item():
                if accelerator.is_main_process:
                    print(f"[NaN] step={global_step}")
                raise RuntimeError(f"NaN at step {global_step}")

            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                if accelerator.num_processes > 1:
                    for p in trainable_params:
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

                has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                             for p in trainable_params)
                if has_nan:
                    if accelerator.is_main_process:
                        print(f"[GRAD SANITIZE] step={global_step}")
                    optimizer_vlm.zero_grad()
                    if optimizer_hyp: optimizer_hyp.zero_grad()
                    if optimizer_kappa: optimizer_kappa.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(vlm_params, 1.0)
                    if hyp_params:
                        torch.nn.utils.clip_grad_norm_(hyp_params, 1.0)
                    optimizer_vlm.step()
                    if optimizer_hyp: optimizer_hyp.step()
                    if optimizer_kappa: optimizer_kappa.step()
                    with torch.no_grad():
                        wrapper.prefix_graph.kappa.data.clamp_(0.1, 5.0)
                    optimizer_vlm.zero_grad()
                    if optimizer_hyp: optimizer_hyp.zero_grad()
                    if optimizer_kappa: optimizer_kappa.zero_grad()

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

                _gate_param = getattr(wrapper.prefix_graph, 'gate', None)
                if _gate_param is None:
                    _gate_param = wrapper.prefix_graph.prefix_proj.gate
                gate_val = safe_float(torch.tanh(_gate_param))

                record = {
                    "step": global_step, "epoch": epoch,
                    "loss_total": loss_val,
                    "loss_task": safe_float(loss_task),
                    "loss_angle": safe_float(angle_loss),
                    "loss_ent": safe_float(ent_loss),
                    "kappa": safe_float(kappa),
                    "gate": gate_val,
                    "grad_scaler_alpha": grad_scaler_alpha,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
                telemetry_file.write(json.dumps(record) + "\n")
                telemetry_file.flush()

                if smoke_steps > 0 or global_step % 50 == 0:
                    print(f"[E{epoch} S{global_step}] loss={loss_val} task={safe_float(loss_task)} "
                          f"angle={safe_float(angle_loss)} ent={safe_float(ent_loss)} "
                          f"kappa={safe_float(kappa)} gate={gate_val} "
                          f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        if smoke_steps > 0 and global_step >= smoke_steps and os.environ.get("SMOKE_SAVE") != "1":
            break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_path = os.path.join(checkpoint_subdir, f"epoch{epoch}")
            os.makedirs(save_path, exist_ok=True)
            state = {
                "prefix_graph": wrapper.prefix_graph.state_dict(),
                "kappa": wrapper.prefix_graph.kappa.data,
                "gate": (getattr(wrapper.prefix_graph, 'gate', None) if hasattr(wrapper.prefix_graph, 'gate') else wrapper.prefix_graph.prefix_proj.gate).data,
                "epoch": epoch, "global_step": global_step,
            }
            if state_is_finite(state):
                torch.save(state, os.path.join(save_path, "prefix_modules.pt"))
                wrapper.vlm.language_model.save_pretrained(os.path.join(save_path, "lora"))
                _write_variant(save_path, epoch)
                avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(checkpoint_subdir, "best")
                    os.makedirs(best_path, exist_ok=True)
                    torch.save(state, os.path.join(best_path, "prefix_modules.pt"))
                    wrapper.vlm.language_model.save_pretrained(os.path.join(best_path, "lora"))
                    _write_variant(best_path, epoch)
                    print(f"Best checkpoint (loss={avg_loss:.4f})")
                print(f"Epoch {epoch} done in {time.time()-t0:.1f}s, gate={gate_val}")
        if smoke_steps > 0:
            break

    if accelerator.is_main_process and telemetry_file:
        telemetry_file.close()
        print("Fixed prefix training complete.")
        _gp = getattr(wrapper.prefix_graph, 'gate', None) if hasattr(wrapper.prefix_graph, 'gate') else wrapper.prefix_graph.prefix_proj.gate
        print(f"Final gate: {torch.tanh(_gp).item():.4f}")
        print(f"Final kappa: {wrapper.prefix_graph.kappa.item():.4f}")


if __name__ == "__main__":
    main()
