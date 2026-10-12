"""HRA: Hyperbolic Relational Attention — attention bias injection for LLaVA-1.5-7B."""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision.ops import roi_align


from hra.hyperbolic import LorentzManifold, SafeArcCosh

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MODEL_PATH = LLAVA_PATH
EMBED_DIM = 256
NUM_HEADS = 4
MAX_PROPOSALS = 36
LLM_DIM = 4096
VIS_DIM = 1024
PATCH_GRID = 24


# ---------------------------------------------------------------------------
# Spatial encoder (pairwise bounding box geometry)
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def compute_deltas(self, boxes):
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        dx = (cx.unsqueeze(2) - cx.unsqueeze(1)) / w.unsqueeze(1)
        dy = (cy.unsqueeze(2) - cy.unsqueeze(1)) / h.unsqueeze(1)
        lw = torch.log(w.unsqueeze(2) / w.unsqueeze(1))
        lh = torch.log(h.unsqueeze(2) / h.unsqueeze(1))
        return torch.stack([dx, dy, lw, lh], dim=-1)

    def forward(self, boxes):
        return self.mlp(self.compute_deltas(boxes))


# ---------------------------------------------------------------------------
# Dense N×N visual relations with spatial bias
# ---------------------------------------------------------------------------

class DenseVisualRelations(nn.Module):
    def __init__(self, vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, spatial_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(vis_dim, embed_dim)
        self.k_proj = nn.Linear(vis_dim, embed_dim)
        self.v_proj = nn.Linear(vis_dim, embed_dim)
        self.spatial_encoder = SpatialEncoder(out_dim=spatial_dim)
        self.spatial_to_bias = nn.Linear(spatial_dim, num_heads)
        self.spatial_to_rel = nn.Linear(spatial_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.use_checkpoint = True

    def _attn_forward(self, features, boxes, mask):
        B, N, _ = features.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(features).view(B, N, H, D).permute(0, 2, 1, 3)
        K = self.k_proj(features).view(B, N, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(features).view(B, N, H, D).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

        spatial_feats = self.spatial_encoder(boxes)
        spatial_bias = self.spatial_to_bias(spatial_feats).permute(0, 3, 1, 2)
        attn_logits = attn_logits + spatial_bias

        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2).expand(B, H, N, N)
            attn_logits = attn_logits.masked_fill(mask_2d == 0, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn_weights, V)
        attended = attended.permute(0, 2, 1, 3).contiguous().view(B, N, self.embed_dim)
        attended = self.out_proj(attended)

        subj_features = attended.unsqueeze(2).expand(B, N, N, self.embed_dim)
        obj_features = attended.unsqueeze(1).expand(B, N, N, self.embed_dim)
        spatial_rel = self.spatial_to_rel(spatial_feats)
        rel_features = subj_features + obj_features + spatial_rel

        return rel_features

    def forward(self, features, boxes, mask=None):
        if self.training and self.use_checkpoint:
            features = features.requires_grad_(True)
            return grad_checkpoint(self._attn_forward, features, boxes, mask, use_reentrant=False)
        return self._attn_forward(features, boxes, mask)


# ---------------------------------------------------------------------------
# Geometric losses (visual)
# ---------------------------------------------------------------------------

class VisualEntailmentLoss(nn.Module):
    def __init__(self, ioa_threshold=0.8):
        super().__init__()
        self.ioa_threshold = ioa_threshold
        self.eta = nn.Parameter(torch.tensor(1.5))

    def compute_ioa_matrix(self, boxes):
        x1_i, x1_j = boxes[:, :, 0].unsqueeze(2), boxes[:, :, 0].unsqueeze(1)
        y1_i, y1_j = boxes[:, :, 1].unsqueeze(2), boxes[:, :, 1].unsqueeze(1)
        x2_i, x2_j = boxes[:, :, 2].unsqueeze(2), boxes[:, :, 2].unsqueeze(1)
        y2_i, y2_j = boxes[:, :, 3].unsqueeze(2), boxes[:, :, 3].unsqueeze(1)
        inter = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(min=0) * \
                (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(min=0)
        area_i = ((x2_i - x1_i) * (y2_i - y1_i)).clamp(min=1e-6)
        return inter / area_i

    def forward(self, points, boxes, manifold):
        B, N = boxes.shape[:2]
        ioa = self.compute_ioa_matrix(boxes)
        eye = torch.eye(N, device=boxes.device).unsqueeze(0)
        containment_mask = ((ioa > self.ioa_threshold).float() * (1.0 - eye))

        if containment_mask.sum() == 0:
            return torch.tensor(0.0, device=boxes.device, requires_grad=True)

        node_points = manifold.einstein_midpoint(points)
        child = node_points.unsqueeze(2).expand(B, N, N, -1)
        parent = node_points.unsqueeze(1).expand(B, N, N, -1)
        child_flat = child[containment_mask > 0]
        parent_flat = parent[containment_mask > 0]

        if child_flat.shape[0] == 0:
            return torch.tensor(0.0, device=boxes.device, requires_grad=True)

        angle = manifold.exterior_angle(child_flat, parent_flat)
        aperture = manifold.half_aperture(parent_flat)
        eta = torch.abs(self.eta) + 1e-6
        return F.relu(angle - eta * aperture).mean()


class VisualAngleLoss(nn.Module):
    def __init__(self, ioa_threshold=0.05):
        super().__init__()
        self.ioa_threshold = ioa_threshold

    def forward(self, node_points, boxes, manifold, mask=None):
        B, N = boxes.shape[:2]
        x1_i, x1_j = boxes[:, :, 0].unsqueeze(2), boxes[:, :, 0].unsqueeze(1)
        y1_i, y1_j = boxes[:, :, 1].unsqueeze(2), boxes[:, :, 1].unsqueeze(1)
        x2_i, x2_j = boxes[:, :, 2].unsqueeze(2), boxes[:, :, 2].unsqueeze(1)
        y2_i, y2_j = boxes[:, :, 3].unsqueeze(2), boxes[:, :, 3].unsqueeze(1)
        inter = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(min=0) * \
                (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(min=0)
        area_i = ((x2_i - x1_i) * (y2_i - y1_i)).clamp(min=1e-6)
        ioa = inter / area_i

        eye = torch.eye(N, device=boxes.device).unsqueeze(0)
        non_overlap = ((ioa < self.ioa_threshold).float() * (1.0 - eye))
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            non_overlap = non_overlap * pair_mask

        if non_overlap.sum() == 0:
            return torch.tensor(0.0, device=boxes.device, requires_grad=True)

        p = node_points.unsqueeze(2).expand(B, N, N, -1)
        q = node_points.unsqueeze(1).expand(B, N, N, -1)
        p_flat = p[non_overlap > 0]
        q_flat = q[non_overlap > 0]

        angles = manifold.exterior_angle(p_flat, q_flat)
        return F.relu(math.pi / 2 - angles).mean()


# ---------------------------------------------------------------------------
# HRA-specific: cone attention + proposal-to-patch mapper
# ---------------------------------------------------------------------------

class HyperbolicConeAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(1.5))
        self.bias_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_points, manifold, mask=None):
        """
        node_points: (B, N, D+1) on Lorentz manifold
        Returns: cone_scores (B, N, N) asymmetric
        """
        B, N, D = node_points.shape
        eta = torch.abs(self.eta) + 1e-6

        pi = node_points.unsqueeze(2).expand(B, N, N, D).reshape(B * N * N, D)
        pj = node_points.unsqueeze(1).expand(B, N, N, D).reshape(B * N * N, D)

        angle_ij = manifold.exterior_angle(pi, pj).view(B, N, N)
        aperture_j = manifold.half_aperture(pj).view(B, N, N)

        scores = eta * aperture_j - angle_ij

        scale = F.softplus(self.bias_scale)
        scores = torch.tanh(scores * scale)

        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            scores = scores * pair_mask

        return scores


class EuclideanConeAttention(nn.Module):
    """Ablation: replace Lorentz geometry with Euclidean L2 + cosine."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(1.5))
        self.bias_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_points, manifold, mask=None):
        B, N, D = node_points.shape
        eta = torch.abs(self.eta) + 1e-6
        spatial = node_points[..., 1:]  # strip time component

        pi = spatial.unsqueeze(2).expand(B, N, N, -1)
        pj = spatial.unsqueeze(1).expand(B, N, N, -1)
        diff = pi - pj
        dist = diff.norm(dim=-1).clamp(min=1e-6)
        norm_j = pj.norm(dim=-1).clamp(min=1e-6)
        aperture = 0.1 / norm_j  # analogous to half_aperture

        scores = eta * aperture - dist
        scale = F.softplus(self.bias_scale)
        scores = torch.tanh(scores * scale)

        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            scores = scores * pair_mask
        return scores


class RandomLearnedBias(nn.Module):
    """Ablation: learnable per-proposal bias matrix, no geometry."""
    def __init__(self, max_proposals=MAX_PROPOSALS):
        super().__init__()
        self.bias_matrix = nn.Parameter(torch.zeros(max_proposals, max_proposals))
        self.bias_scale = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.tensor(1.5))  # unused, for compat

    def forward(self, node_points, manifold, mask=None):
        B, N, D = node_points.shape
        scores = self.bias_matrix[:N, :N].unsqueeze(0).expand(B, N, N)
        scale = F.softplus(self.bias_scale)
        scores = torch.tanh(scores * scale)
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            scores = scores * pair_mask
        return scores


class SpatialProximityBias(nn.Module):
    """Ablation: IoU-based proximity bias, no manifold."""
    def __init__(self):
        super().__init__()
        self.bias_scale = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.tensor(1.5))  # unused, for compat

    def forward(self, node_points, manifold, mask=None, boxes=None):
        B, N, D = node_points.shape
        if boxes is None:
            return torch.zeros(B, N, N, device=node_points.device)
        x1_i, x1_j = boxes[:, :, 0].unsqueeze(2), boxes[:, :, 0].unsqueeze(1)
        y1_i, y1_j = boxes[:, :, 1].unsqueeze(2), boxes[:, :, 1].unsqueeze(1)
        x2_i, x2_j = boxes[:, :, 2].unsqueeze(2), boxes[:, :, 2].unsqueeze(1)
        y2_i, y2_j = boxes[:, :, 3].unsqueeze(2), boxes[:, :, 3].unsqueeze(1)
        inter = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(min=0) * \
                (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(min=0)
        area_i = ((x2_i - x1_i) * (y2_i - y1_i)).clamp(min=1e-6)
        area_j = ((x2_j - x1_j) * (y2_j - y1_j)).clamp(min=1e-6)
        union = area_i + area_j - inter
        iou = inter / union.clamp(min=1e-6)
        scale = F.softplus(self.bias_scale)
        scores = torch.tanh(iou * scale)
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            scores = scores * pair_mask
        return scores


class ProposalToPatchMapper(nn.Module):
    def __init__(self, grid_size=PATCH_GRID):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size

    def _compute_iou_assignment(self, boxes, mask, device):
        """Build soft assignment matrix A (B, P, N) from IoU between patches and proposals."""
        G = self.grid_size
        P = self.num_patches
        B, N, _ = boxes.shape

        step = 1.0 / G
        grid_y, grid_x = torch.meshgrid(
            torch.arange(G, device=device, dtype=torch.float32),
            torch.arange(G, device=device, dtype=torch.float32),
            indexing="ij",
        )
        patch_x1 = (grid_x * step).reshape(P)
        patch_y1 = (grid_y * step).reshape(P)
        patch_x2 = patch_x1 + step
        patch_y2 = patch_y1 + step

        bx1 = boxes[:, :, 0].unsqueeze(1)  # (B, 1, N)
        by1 = boxes[:, :, 1].unsqueeze(1)
        bx2 = boxes[:, :, 2].unsqueeze(1)
        by2 = boxes[:, :, 3].unsqueeze(1)

        px1 = patch_x1.view(1, P, 1)  # (1, P, 1)
        py1 = patch_y1.view(1, P, 1)
        px2 = patch_x2.view(1, P, 1)
        py2 = patch_y2.view(1, P, 1)

        inter_w = (torch.min(px2, bx2) - torch.max(px1, bx1)).clamp(min=0)
        inter_h = (torch.min(py2, by2) - torch.max(py1, by1)).clamp(min=0)
        inter = inter_w * inter_h

        area_p = step * step
        area_b = ((bx2 - bx1) * (by2 - by1)).clamp(min=1e-8)
        union = area_p + area_b - inter
        iou = inter / union.clamp(min=1e-8)  # (B, P, N)

        if mask is not None:
            iou = iou * mask.unsqueeze(1)

        row_sum = iou.sum(dim=2, keepdim=True).clamp(min=1e-8)
        A = iou / row_sum

        return A

    def forward(self, cone_scores, boxes, mask=None):
        """
        cone_scores: (B, N, N) asymmetric
        boxes: (B, N, 4)
        Returns: patch_bias (B, P, P) where P = grid_size^2
        """
        A = self._compute_iou_assignment(boxes, mask, cone_scores.device)
        patch_bias = torch.bmm(torch.bmm(A, cone_scores), A.transpose(1, 2))
        return patch_bias


# ---------------------------------------------------------------------------
# HRAGraph: visual proposals → dense relations → Lorentz → cone → patch bias
# ---------------------------------------------------------------------------

class HRAGraph(nn.Module):
    def __init__(self, vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 max_proposals=MAX_PROPOSALS, grid_size=PATCH_GRID):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_proposals = max_proposals
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.manifold = LorentzManifold(self.kappa)
        self.dense_relations = DenseVisualRelations(
            vis_dim=vis_dim, embed_dim=embed_dim, num_heads=num_heads)
        self.relation_to_manifold = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.relation_to_manifold.weight, gain=0.1)
        nn.init.zeros_(self.relation_to_manifold.bias)
        self.entailment_loss = VisualEntailmentLoss()
        self.angle_loss = VisualAngleLoss()
        self.cone_attn = HyperbolicConeAttention(embed_dim=embed_dim)
        self.patch_mapper = ProposalToPatchMapper(grid_size=grid_size)
        self.symmetrize = False

    def forward(self, features, boxes, mask=None):
        """
        features: (B, N, vis_dim) proposal RoI features
        boxes: (B, N, 4) normalized
        mask: (B, N)
        Returns: (patch_bias, relation_points, ent_loss, ang_loss)
        """
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

        if isinstance(self.cone_attn, SpatialProximityBias):
            cone_scores = self.cone_attn(node_points, self.manifold, mask, boxes=boxes)
        else:
            cone_scores = self.cone_attn(node_points, self.manifold, mask)
        if self.symmetrize:
            cone_scores = (cone_scores + cone_scores.transpose(1, 2)) / 2
        patch_bias = self.patch_mapper(cone_scores, boxes, mask)

        return patch_bias, relation_points, ent_loss, ang_loss


# ---------------------------------------------------------------------------
# HRAWrapper: LLaVA + LoRA + HRA with attention bias injection
# ---------------------------------------------------------------------------

class HRAWrapper(nn.Module):
    def __init__(self, vlm, processor, hra_graph):
        super().__init__()
        self.vlm = vlm
        self.processor = processor
        self.hra_graph = hra_graph
        self._hra_bias = None
        self._image_range = None
        self.detach_bias = False
        self.disable_bias = False
        self._patch_llama_model()

    def _get_llama_model(self):
        lm = self.vlm.language_model
        if hasattr(lm, 'base_model'):
            # PeftModel -> LoraModel -> LlamaForCausalLM -> LlamaModel
            return lm.base_model.model.model
        elif hasattr(lm, 'model') and hasattr(lm.model, 'model'):
            return lm.model.model
        elif hasattr(lm, 'model'):
            return lm.model
        return lm

    def _patch_llama_model(self):
        llama_model = self._get_llama_model()
        orig_fn = llama_model._update_causal_mask
        wrapper_ref = self

        # Force eager attention — SDPA's Flash Attention requires contiguous masks
        # but the KV-cache slicing in generation produces non-contiguous views.
        # Eager attention handles arbitrary mask shapes correctly.
        lm = self.vlm.language_model
        base_model = lm.base_model.model if hasattr(lm, 'base_model') else lm
        if hasattr(base_model, 'config'):
            base_model.config._attn_implementation = "eager"
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'config'):
            base_model.model.config._attn_implementation = "eager"

        def _patched_update_causal_mask(self_llama, attention_mask, input_tensor, cache_position,
                                        past_seen_tokens):
            mask = orig_fn(attention_mask, input_tensor, cache_position,
                           past_seen_tokens)
            if wrapper_ref._hra_bias is None:
                return mask

            if mask is None:
                B = input_tensor.shape[0]
                S = input_tensor.shape[1]
                dtype = input_tensor.dtype
                device = input_tensor.device
                min_val = torch.finfo(dtype).min
                mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
                mask.masked_fill_(torch.ones(S, S, device=device).triu(1).bool(), min_val)

            img_s, img_e = wrapper_ref._image_range
            Q = mask.shape[2]  # query length
            K = mask.shape[3]  # key length
            # Only inject bias during prefill (Q > 1 and image patches in query range)
            if Q > 1 and img_e <= K and img_e <= Q:
                bias = wrapper_ref._hra_bias
                # Expand mask to match bias head dimension if needed
                if bias.shape[1] > mask.shape[1]:
                    mask = mask.expand(-1, bias.shape[1], -1, -1).clone()
                else:
                    mask = mask.clone()
                mask[:, :, img_s:img_e, img_s:img_e] += bias
            return mask

        import types
        llama_model._update_causal_mask = types.MethodType(_patched_update_causal_mask, llama_model)

    def _get_embed_fn(self):
        lm = self.vlm.language_model
        if hasattr(lm, 'base_model'):
            return lm.base_model.model.model.embed_tokens
        elif hasattr(lm, 'model') and hasattr(lm.model, 'model'):
            return lm.model.model.embed_tokens
        elif hasattr(lm, 'model'):
            return lm.model.embed_tokens
        return lm.embed_tokens

    def _extract_roi_features(self, image_features, boxes, mask):
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
            pooled = roi_align(
                feat_grid[i:i+1], rois, output_size=(1, 1),
                spatial_scale=1.0, aligned=True
            )
            proposal_features[i, :valid] = pooled.squeeze(-1).squeeze(-1)

        return proposal_features

    def _merge_with_vision(self, inputs_embeds, image_features, input_ids, attention_mask):
        """Replace <image> token with image features — standard LLaVA merge, no prefix tokens."""
        B, S, D = inputs_embeds.shape
        num_patches = image_features.shape[1]
        image_token_id = self.vlm.config.image_token_index
        new_len = S - 1 + num_patches

        merged = torch.zeros(B, new_len, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        merged_attn = torch.zeros(B, new_len, device=inputs_embeds.device, dtype=attention_mask.dtype)

        img_start = 0
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
            merged[i, pos + num_patches:] = inputs_embeds[i, pos + 1:]
            merged_attn[i, pos + num_patches:] = attention_mask[i, pos + 1:]
            img_start = pos

        return merged, merged_attn, img_start

    def _expand_labels(self, labels, input_ids, num_patches):
        B, S = labels.shape
        image_token_id = self.vlm.config.image_token_index
        new_len = S - 1 + num_patches
        expanded = torch.full((B, new_len), -100, dtype=labels.dtype, device=labels.device)
        for i in range(B):
            img_pos = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) == 0:
                expanded[i, :S] = labels[i]
                continue
            pos = img_pos[0].item()
            expanded[i, :pos] = labels[i, :pos]
            expanded[i, pos + num_patches:] = labels[i, pos + 1:]
        return expanded

    def _get_vision_features(self, pixel_values):
        """Extract vision features matching LLaVA's default strategy: hidden_states[-2], strip CLS."""
        vision_outputs = self.vlm.vision_tower(pixel_values, output_hidden_states=True)
        raw_vision = vision_outputs.hidden_states[self.vlm.config.vision_feature_layer]
        selected = raw_vision[:, 1:]  # strip CLS (default strategy)
        image_features = self.vlm.multi_modal_projector(selected)
        return raw_vision, image_features

    def forward(self, pixel_values, input_ids, attention_mask, boxes, box_mask, labels=None):
        raw_vision, image_features = self._get_vision_features(pixel_values)

        proposal_features = self._extract_roi_features(raw_vision.detach(), boxes, box_mask)

        patch_bias, relation_points, ent_loss, ang_loss = self.hra_graph(
            proposal_features, boxes, box_mask)

        inputs_embeds = self._get_embed_fn()(input_ids)
        merged_embeds, merged_attn, img_start = self._merge_with_vision(
            inputs_embeds, image_features, input_ids, attention_mask)

        num_patches = image_features.shape[1]
        bias = patch_bias.unsqueeze(1)  # (B, 1, P, P)
        if self.detach_bias:
            bias = bias.detach()
        if self.disable_bias:
            bias = None
        self._hra_bias = bias
        self._image_range = (img_start, img_start + num_patches)

        if labels is not None:
            labels = self._expand_labels(labels, input_ids, num_patches)

        try:
            outputs = self.vlm.language_model(
                inputs_embeds=merged_embeds,
                attention_mask=merged_attn,
                labels=labels,
            )
        finally:
            self._hra_bias = None
            self._image_range = None

        return outputs, relation_points, ent_loss, ang_loss

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, boxes, box_mask, **gen_kwargs):
        raw_vision, image_features = self._get_vision_features(pixel_values)

        proposal_features = self._extract_roi_features(raw_vision, boxes, box_mask)
        patch_bias, _, _, _ = self.hra_graph(proposal_features, boxes, box_mask)

        inputs_embeds = self._get_embed_fn()(input_ids)
        merged_embeds, merged_attn, img_start = self._merge_with_vision(
            inputs_embeds, image_features, input_ids, attention_mask)

        num_patches = image_features.shape[1]
        self._hra_bias = patch_bias.unsqueeze(1)
        self._image_range = (img_start, img_start + num_patches)

        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        try:
            output_ids = self.vlm.language_model.generate(
                inputs_embeds=merged_embeds, attention_mask=merged_attn,
                pad_token_id=pad_id, **gen_kwargs)
        finally:
            self._hra_bias = None
            self._image_range = None

        return output_ids

    @torch.no_grad()
    def score(self, pixel_values, input_ids, attention_mask, boxes, box_mask, labels):
        raw_vision, image_features = self._get_vision_features(pixel_values)

        proposal_features = self._extract_roi_features(raw_vision, boxes, box_mask)
        patch_bias, _, _, _ = self.hra_graph(proposal_features, boxes, box_mask)

        inputs_embeds = self._get_embed_fn()(input_ids)
        merged_embeds, merged_attn, img_start = self._merge_with_vision(
            inputs_embeds, image_features, input_ids, attention_mask)

        num_patches = image_features.shape[1]
        expanded_labels = self._expand_labels(labels, input_ids, num_patches)

        self._hra_bias = patch_bias.unsqueeze(1)
        self._image_range = (img_start, img_start + num_patches)

        try:
            outputs = self.vlm.language_model(
                inputs_embeds=merged_embeds, attention_mask=merged_attn,
                labels=expanded_labels)
        finally:
            self._hra_bias = None
            self._image_range = None

        return -outputs.loss.item()

    @classmethod
    def load_for_eval(cls, checkpoint_dir, device):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        vlm = LlavaForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16)
        vlm.requires_grad_(False)

        lora_path = os.path.join(checkpoint_dir, "lora")
        if os.path.exists(lora_path):
            vlm.language_model = PeftModel.from_pretrained(vlm.language_model, lora_path)

        hra_graph = HRAGraph(vis_dim=VIS_DIM, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                             max_proposals=MAX_PROPOSALS)
        hyp_state = torch.load(
            os.path.join(checkpoint_dir, "hra_modules.pt"), map_location="cpu")
        hra_graph.load_state_dict(hyp_state["hra_graph"])

        wrapper = cls(vlm, processor, hra_graph)
        wrapper = wrapper.to(device).eval()
        return wrapper, processor
