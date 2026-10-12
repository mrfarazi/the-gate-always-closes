"""Lorentz hyperboloid modules for HyperTrip."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeArcCosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1.0 + 1e-6, max=1e8)
        ctx.save_for_backward(x)
        return torch.acosh(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Wider safety margin — diagnosis showed grads were 10000x too large
        x_clamped = torch.clamp(x, min=1.0 + 1e-2)
        return grad_output / torch.sqrt(x_clamped * x_clamped - 1.0)


class LorentzManifold:
    def __init__(self, kappa):
        self.kappa = kappa

    def _get_k(self):
        if isinstance(self.kappa, (nn.Parameter, torch.Tensor)):
            return torch.abs(self.kappa) + 1e-6
        return torch.tensor(float(self.kappa))

    def lorentz_inner(self, x, y):
        """Minkowski inner product: -x0*y0 + x1*y1 + ... + xn*yn."""
        time = -x[..., 0:1] * y[..., 0:1]
        space = torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        return time + space

    def project_onto_manifold(self, x):
        """Project x onto the Lorentz hyperboloid: x0 = sqrt(1/kappa + ||x_space||^2)."""
        k = self._get_k()
        x_space = x[..., 1:]
        x_0 = torch.sqrt(1.0 / k + torch.sum(x_space * x_space, dim=-1, keepdim=True))
        return torch.cat([x_0, x_space], dim=-1)

    def exp_map_origin(self, v):
        """Exponential map from the origin o=(1/sqrt(k), 0,...). v is tangent (v[...,0] = 0)."""
        k = self._get_k()
        v_space = v[..., 1:]
        v_norm = torch.norm(v_space, dim=-1, keepdim=True).clamp(min=1e-6)
        sqrt_k = torch.sqrt(k)
        arg = sqrt_k * v_norm
        x_0 = torch.cosh(arg) / sqrt_k
        x_space = torch.sinh(arg) / (arg + 1e-6) * v_space
        x = torch.cat([x_0, x_space], dim=-1)
        return self.project_onto_manifold(x)

    def log_map_origin(self, y):
        """Logarithmic map to the origin o=(1/sqrt(k), 0,...). y is a manifold point."""
        k = self._get_k()
        sqrt_k = torch.sqrt(k)
        y_0 = y[..., 0:1]
        y_space = y[..., 1:]
        alpha = sqrt_k * y_0
        alpha = torch.clamp(alpha, min=1.0 + 1e-2)
        theta = SafeArcCosh.apply(alpha)
        coeff = theta / (torch.sinh(theta) + 1e-6)
        v_space = coeff * y_space
        v_0 = torch.zeros_like(y_0)
        return torch.cat([v_0, v_space], dim=-1)

    def geodesic_distance(self, x, y):
        """Geodesic distance on Lorentz hyperboloid using SafeArcCosh."""
        k = self._get_k()
        inner = self.lorentz_inner(x, y)
        return SafeArcCosh.apply(-k * inner.squeeze(-1)) / torch.sqrt(k)

    def exterior_angle(self, p, q):
        """Angle at the origin between log_map(p) and log_map(q)."""
        v_p = self.log_map_origin(p)
        v_q = self.log_map_origin(q)
        v_p_space = v_p[..., 1:]
        v_q_space = v_q[..., 1:]
        norm_p = torch.norm(v_p_space, dim=-1, keepdim=False)
        norm_q = torch.norm(v_q_space, dim=-1, keepdim=False)
        dot = torch.sum(v_p_space * v_q_space, dim=-1)
        cos_angle = dot / (norm_p * norm_q + 1e-6)
        cos_angle = torch.clamp(cos_angle, min=-1.0 + 1e-5, max=1.0 - 1e-5)
        return torch.acos(cos_angle)

    def half_aperture(self, q):
        """Half-aperture of the entailment cone at q. K_cone=0.1."""
        K_cone = 0.1
        k = self._get_k()
        q_space = q[..., 1:]
        q_space_norm = torch.norm(q_space, dim=-1)
        q_space_norm = torch.clamp(q_space_norm, min=1e-6)
        sin_arg = K_cone / q_space_norm
        sin_arg = torch.clamp(sin_arg, min=-1.0 + 1e-5, max=1.0 - 1e-5)
        return torch.asin(sin_arg)

    def einstein_midpoint(self, points, weights=None):
        """Einstein midpoint: gamma_i = z_i[0] (time component), weighted Minkowski sum, project."""
        k = self._get_k()
        gamma = points[..., 0]
        if weights is None:
            weights = torch.ones_like(gamma)
        w = weights * gamma
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)
        x_sum = torch.sum(w.unsqueeze(-1) * points, dim=-2)
        inner_sum = self.lorentz_inner(x_sum, x_sum)
        denom = torch.sqrt(torch.clamp(-k * inner_sum, min=1e-6))
        midpoint = x_sum / denom
        return self.project_onto_manifold(midpoint)


class TripletEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_predicates=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.entity_embed = nn.Embedding(vocab_size, embed_dim)
        self.predicate_embed = nn.Embedding(num_predicates, embed_dim)
        n_pairs = embed_dim // 2
        self.givens_angles = nn.Parameter(torch.zeros(num_predicates, n_pairs))
        self.predicate_translation = nn.Embedding(num_predicates, embed_dim)
        self.kappa = nn.Parameter(torch.tensor(2.0))
        self.manifold = LorentzManifold(self.kappa)

    def _givens_rotation(self, x, angles):
        """Apply block-diagonal Givens rotation to spatial vector x."""
        n_pairs = x.shape[-1] // 2
        cos_a = torch.cos(angles[..., :n_pairs])
        sin_a = torch.sin(angles[..., :n_pairs])
        x_pairs = x.view(*x.shape[:-1], n_pairs, 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        r1 = cos_a * x1 - sin_a * x2
        r2 = sin_a * x1 + cos_a * x2
        rotated = torch.stack([r1, r2], dim=-1)
        return rotated.view(*x.shape)

    def forward(self, triplet_tokens, predicate_ids):
        """
        triplet_tokens: (B, K, max_seq_len) — tokenized triplet strings
        predicate_ids: (B, K) — predicate class indices
        Returns: (B, K, embed_dim+1) points on Lorentz manifold
        """
        self.manifold.kappa = self.kappa
        B, K, S = triplet_tokens.shape
        flat_tokens = triplet_tokens.view(B * K, S)
        mask = (flat_tokens != 0).float()
        token_embeds = self.entity_embed(flat_tokens)
        lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        pooled = (token_embeds * mask.unsqueeze(-1)).sum(dim=1) / lengths
        pooled = pooled.view(B, K, self.embed_dim)

        pred_ids_flat = predicate_ids.view(B * K)
        angles = self.givens_angles[pred_ids_flat].view(B, K, -1)
        t_p = self.predicate_translation(predicate_ids)

        rotated = self._givens_rotation(pooled, angles)
        composed = rotated + t_p

        v = torch.zeros(*composed.shape[:-1], composed.shape[-1] + 1,
                         device=composed.device, dtype=composed.dtype)
        v[..., 1:] = composed
        points = self.manifold.exp_map_origin(v)
        points = self.manifold.project_onto_manifold(points)
        return points


class LorentzEntailmentAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, max_triplets=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_triplets = max_triplets
        self.head_dim = embed_dim // num_heads
        self.kappas = nn.Parameter(torch.full((num_heads,), 2.0))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, triplet_points, triplet_mask=None):
        """
        query: (B, embed_dim) — query embedding (spatial part only)
        triplet_points: (B, N, embed_dim+1) — triplet manifold points
        triplet_mask: (B, N) — 1 for valid triplets, 0 for padding
        Returns: (B, embed_dim+1) aggregated point on manifold
        """
        B, N, D = triplet_points.shape
        spatial_dim = D - 1

        q = self.q_proj(query)
        k = self.k_proj(triplet_points[..., 1:])
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(self.head_dim)

        if triplet_mask is not None:
            mask_expanded = triplet_mask.unsqueeze(1).expand(B, self.num_heads, N)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.mean(dim=1)

        topk_k = min(self.max_triplets, N)
        _, topk_indices = torch.topk(attn_weights, topk_k, dim=-1)
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(B, topk_k, D)
        selected_points = torch.gather(triplet_points, 1, topk_indices_exp)
        selected_weights = torch.gather(attn_weights, 1, topk_indices)

        manifold = LorentzManifold(torch.abs(self.kappas[0]) + 1e-6)
        aggregated = manifold.einstein_midpoint(selected_points, selected_weights)
        return aggregated


class TokenProjector(nn.Module):
    """Project aggregated Lorentz point (dim=257) to 4 LLM prefix tokens (4*4096)."""
    def __init__(self, input_dim=257, num_tokens=4, llm_dim=4096):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.proj = nn.Linear(input_dim, num_tokens * llm_dim)

    def forward(self, x):
        """x: (B, 257) → (B, num_tokens, 4096)"""
        out = self.proj(x)
        return out.view(x.shape[0], self.num_tokens, self.llm_dim)


class AttentionBiasMLP(nn.Module):
    """Sidecar: projects hyperbolic aggregation into per-patch visual attention weights.
    Computes dot-product attention between the projected Lorentz point and each visual patch,
    then applies sigmoid to produce modulation weights in (0, 1).
    """
    def __init__(self, hyp_dim=257, vis_dim=4096):
        super().__init__()
        self.query_proj = nn.Linear(hyp_dim, vis_dim)

    def forward(self, aggregated, image_features):
        """
        aggregated: (B, hyp_dim) — aggregated Lorentz point from entailment attention
        image_features: (B, num_patches, vis_dim) — visual features from VLM
        Returns: (B, num_patches) — per-patch modulation weights in (0, 1)
        """
        query = self.query_proj(aggregated)
        scores = torch.einsum("bd,bnd->bn", query, image_features)
        return torch.sigmoid(scores)


class SpatialMLP(nn.Module):
    """Encodes relative bounding box geometry into attention bias.
    Input: (B, K, 4) — [dx, dy, log(w_sub/w_obj), log(h_sub/h_obj)]
    Output: (B, K) scalar bias per triplet, broadcastable into attention logits.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bbox_deltas):
        """bbox_deltas: (B, K, 4) → (B, K) spatial bias."""
        return self.mlp(bbox_deltas).squeeze(-1)


class VisioSpatialCrossAttention(nn.Module):
    """Cross-attention from subject to object visual features with spatial bias.
    Query = subject visual features, Key/Value = object visual features.
    Adds SpatialMLP bias to pre-softmax attention logits.
    """
    def __init__(self, vis_dim=1024, embed_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(vis_dim, embed_dim)
        self.k_proj = nn.Linear(vis_dim, embed_dim)
        self.v_proj = nn.Linear(vis_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.spatial_mlp = SpatialMLP()

    def forward(self, subj_features, obj_features, bbox_deltas, mask=None):
        """
        subj_features: (B, K, vis_dim) — subject visual features per triplet
        obj_features: (B, K, vis_dim) — object visual features per triplet
        bbox_deltas: (B, K, 4) — relative bounding box geometry
        mask: (B, K) — 1 for valid triplets, 0 for padding
        Returns: (B, K, embed_dim) — relation visual features
        """
        B, K, _ = subj_features.shape
        H = self.num_heads
        D = self.head_dim

        Q = self.q_proj(subj_features).view(B, K, H, D).permute(0, 2, 1, 3)
        KK = self.k_proj(obj_features).view(B, K, H, D).permute(0, 2, 1, 3)
        V = self.v_proj(obj_features).view(B, K, H, D).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(Q, KK.transpose(-2, -1)) / math.sqrt(D)

        spatial_bias = self.spatial_mlp(bbox_deltas)
        attn_logits = attn_logits + spatial_bias.unsqueeze(1).unsqueeze(-1)

        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2).expand(B, H, K, K)
            attn_logits = attn_logits.masked_fill(mask_2d == 0, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, K, self.embed_dim)
        return self.out_proj(out)


class SpatialTripletEncoder(nn.Module):
    """V7 TripletEncoder: replaces text-only pooling with visual cross-attention + spatial bias.
    Combines text embeddings, visual relation features, and Givens rotation + translation,
    then maps to the Lorentz manifold.
    """
    def __init__(self, vocab_size, embed_dim=256, num_predicates=512, vis_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.entity_embed = nn.Embedding(vocab_size, embed_dim)
        self.predicate_embed = nn.Embedding(num_predicates, embed_dim)
        n_pairs = embed_dim // 2
        self.givens_angles = nn.Parameter(torch.zeros(num_predicates, n_pairs))
        self.predicate_translation = nn.Embedding(num_predicates, embed_dim)
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.manifold = LorentzManifold(self.kappa)
        self.cross_attn = VisioSpatialCrossAttention(vis_dim=vis_dim, embed_dim=embed_dim)
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)

    def _givens_rotation(self, x, angles):
        n_pairs = x.shape[-1] // 2
        cos_a = torch.cos(angles[..., :n_pairs])
        sin_a = torch.sin(angles[..., :n_pairs])
        x_pairs = x.view(*x.shape[:-1], n_pairs, 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        r1 = cos_a * x1 - sin_a * x2
        r2 = sin_a * x1 + cos_a * x2
        rotated = torch.stack([r1, r2], dim=-1)
        return rotated.view(*x.shape)

    def forward(self, triplet_tokens, predicate_ids, subj_features=None, obj_features=None,
                bbox_deltas=None, triplet_mask=None):
        """
        triplet_tokens: (B, K, S) — tokenized triplet strings
        predicate_ids: (B, K) — predicate class indices
        subj_features: (B, K, vis_dim) — subject RoI features (optional)
        obj_features: (B, K, vis_dim) — object RoI features (optional)
        bbox_deltas: (B, K, 4) — relative bbox geometry (optional)
        triplet_mask: (B, K) — valid triplet mask
        Returns: (B, K, embed_dim+1) points on Lorentz manifold
        """
        self.manifold.kappa = self.kappa
        B, K, S = triplet_tokens.shape

        # Text pathway (same as V6)
        flat_tokens = triplet_tokens.view(B * K, S)
        mask = (flat_tokens != 0).float()
        token_embeds = self.entity_embed(flat_tokens)
        lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        pooled = (token_embeds * mask.unsqueeze(-1)).sum(dim=1) / lengths
        text_feat = pooled.view(B, K, self.embed_dim)

        # Visual pathway (V7: cross-attention with spatial bias)
        if subj_features is not None and obj_features is not None and bbox_deltas is not None:
            vis_feat = self.cross_attn(subj_features, obj_features, bbox_deltas, triplet_mask)
            composed_input = self.fuse(torch.cat([text_feat, vis_feat], dim=-1))
        else:
            composed_input = text_feat

        # Givens rotation + translation (same as V6)
        pred_ids_flat = predicate_ids.view(B * K)
        angles = self.givens_angles[pred_ids_flat].view(B, K, -1)
        t_p = self.predicate_translation(predicate_ids)
        rotated = self._givens_rotation(composed_input, angles)
        composed = rotated + t_p

        # Map to Lorentz manifold
        v = torch.zeros(*composed.shape[:-1], composed.shape[-1] + 1,
                         device=composed.device, dtype=composed.dtype)
        v[..., 1:] = composed
        points = self.manifold.exp_map_origin(v)
        points = self.manifold.project_onto_manifold(points)
        return points
