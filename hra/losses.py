"""Loss functions for HyperTrip: angle-based contrastive, entailment, centroid, kappa hinge."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hra.hyperbolic import LorentzManifold, SafeArcCosh


class AngleContrastiveLoss(nn.Module):
    """Angle-based contrastive loss (ATMG-style). Asymmetric: query → triplet only."""
    def __init__(self):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(0.07))

    def forward(self, query_points, triplet_points, manifold):
        """
        query_points: (B, D) manifold points from query
        triplet_points: (B, D) aggregated triplet manifold points
        Computes BxB pairwise angle matrix: logits[i,j] = angle(query_i, triplet_j)
        Diagonal = matching pairs.
        """
        B = query_points.shape[0]
        tau = torch.abs(self.tau) + 1e-6
        # Pairwise: query_i vs triplet_j for all i,j
        q_exp = query_points.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
        t_exp = triplet_points.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
        alpha_flat = manifold.exterior_angle(q_exp, t_exp)
        alpha = alpha_flat.view(B, B)
        beta = math.pi - alpha
        logits_alpha = -alpha / tau
        logits_beta = beta / tau
        labels = torch.arange(B, device=query_points.device)
        loss_alpha = F.cross_entropy(logits_alpha, labels)
        loss_beta = F.cross_entropy(logits_beta, labels)
        return loss_alpha + loss_beta


class EntailmentLoss(nn.Module):
    """Cone-based entailment loss."""
    def __init__(self):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(1.5))

    def forward(self, parent_points, child_points, manifold):
        """parent should entail child: angle(child, parent) < eta * half_aperture(parent)."""
        angle = manifold.exterior_angle(child_points, parent_points)
        aperture = manifold.half_aperture(parent_points)
        eta = torch.abs(self.eta) + 1e-6
        violation = F.relu(angle - eta * aperture)
        return violation.mean()


class CentroidLoss(nn.Module):
    """Predicate centroid should be closer to origin than entity centroid."""
    def forward(self, predicate_points, entity_points, manifold):
        safe_manifold = LorentzManifold(manifold.kappa.detach())
        pred_dists = safe_manifold.geodesic_distance(
            predicate_points,
            self._origin_like(predicate_points, safe_manifold)
        )
        ent_dists = safe_manifold.geodesic_distance(
            entity_points,
            self._origin_like(entity_points, safe_manifold)
        )
        return F.relu(pred_dists.mean() - ent_dists.mean())

    def _origin_like(self, x, manifold):
        k = manifold._get_k()
        with torch.no_grad():
            origin = torch.zeros_like(x)
            origin[..., 0] = 1.0 / torch.sqrt(k)
        return origin.detach()


class KappaHingeLoss(nn.Module):
    """Prevent kappa from collapsing below 0.3."""
    def forward(self, kappa):
        k = torch.abs(kappa) + 1e-6
        return torch.square(F.relu(0.3 - k))


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle_loss = AngleContrastiveLoss()
        self.entailment_loss = EntailmentLoss()
        self.centroid_loss = CentroidLoss()
        self.kappa_hinge_loss = KappaHingeLoss()
        self.lambda_angle = 1.0
        self.lambda_ent = 0.1
        self.lambda_cen = 0.01
        self.lambda_kappa = 0.1

    def forward(self, query_points, triplet_points, predicate_points, entity_points, kappa, manifold):
        l_angle = self.angle_loss(query_points, triplet_points, manifold)
        l_ent = self.entailment_loss(triplet_points, query_points, manifold)
        l_cen = self.centroid_loss(predicate_points, entity_points, manifold)
        l_kappa = self.kappa_hinge_loss(kappa)
        total = (self.lambda_angle * l_angle +
                 self.lambda_ent * l_ent +
                 self.lambda_cen * l_cen +
                 self.lambda_kappa * l_kappa)
        return total, {
            "loss_angle": l_angle.item(),
            "loss_ent": l_ent.item(),
            "loss_cen": l_cen.item(),
            "loss_kappa": l_kappa.item(),
        }
