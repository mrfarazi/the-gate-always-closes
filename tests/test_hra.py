"""CPU-only unit tests for HRA modules."""

import math
import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hra.hyperbolic import LorentzManifold, SafeArcCosh
from hra.model import (
    HyperbolicConeAttention,
    ProposalToPatchMapper,
    HRAGraph,
    DenseVisualRelations,
    SpatialEncoder,
)


@pytest.fixture
def manifold():
    kappa = nn.Parameter(torch.tensor(1.0))
    return LorentzManifold(kappa)


@pytest.fixture
def sample_boxes():
    B, N = 2, 6
    boxes = torch.rand(B, N, 4)
    boxes[..., 2] = boxes[..., 0] + 0.1 + torch.rand(B, N) * 0.2
    boxes[..., 3] = boxes[..., 1] + 0.1 + torch.rand(B, N) * 0.2
    boxes = boxes.clamp(0, 1)
    mask = torch.ones(B, N)
    return boxes, mask


class TestSafeArcCosh:
    def test_forward(self):
        x = torch.tensor([1.0001, 2.0, 5.0])
        result = SafeArcCosh.apply(x)
        assert result.shape == (3,)
        assert not torch.isnan(result).any()

    def test_gradient(self):
        x = torch.tensor([1.001, 2.0, 5.0], requires_grad=True)
        result = SafeArcCosh.apply(x)
        result.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_near_one(self):
        x = torch.tensor([1.0 + 1e-7, 1.0 + 1e-8, 1.0], requires_grad=True)
        result = SafeArcCosh.apply(x)
        result.sum().backward()
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestLorentzManifold:
    def test_project_onto_manifold(self, manifold):
        x = torch.randn(2, 5, 257)
        projected = manifold.project_onto_manifold(x)
        assert projected.shape == x.shape
        inner = manifold.lorentz_inner(projected, projected)
        k = manifold._get_k()
        expected = -1.0 / k
        assert torch.allclose(inner.squeeze(-1), expected.expand_as(inner.squeeze(-1)), atol=1e-4)

    def test_exp_map_origin(self, manifold):
        v = torch.zeros(3, 257)
        v[..., 1:] = torch.randn(3, 256) * 0.1
        result = manifold.exp_map_origin(v)
        assert result.shape == (3, 257)
        assert result[..., 0].min() > 0


class TestHyperbolicConeAttention:
    def test_output_shape(self, manifold):
        B, N = 2, 8
        cone = HyperbolicConeAttention()
        node_points = manifold.project_onto_manifold(torch.randn(B, N, 257))
        scores = cone(node_points, manifold)
        assert scores.shape == (B, N, N)

    def test_asymmetry(self, manifold):
        B, N = 1, 4
        cone = HyperbolicConeAttention()
        cone.bias_scale = nn.Parameter(torch.tensor(2.0))
        # Create points with very different spatial norms to get different apertures
        raw = torch.randn(B, N, 257)
        raw[0, 0, 1:] *= 0.1  # small spatial norm → large aperture
        raw[0, 1, 1:] *= 5.0  # large spatial norm → small aperture
        node_points = manifold.project_onto_manifold(raw)
        scores = cone(node_points, manifold)
        diff = (scores - scores.transpose(1, 2)).abs().max().item()
        assert diff > 1e-3, f"Cone scores should be asymmetric, max diff={diff}"

    def test_gradient_flow(self, manifold):
        B, N = 2, 4
        cone = HyperbolicConeAttention()
        raw = torch.randn(B, N, 257, requires_grad=True)
        node_points = manifold.project_onto_manifold(raw)
        scores = cone(node_points, manifold)
        scores.sum().backward()
        assert raw.grad is not None
        assert not torch.isnan(raw.grad).any()

    def test_masking(self, manifold):
        B, N = 2, 6
        cone = HyperbolicConeAttention()
        node_points = manifold.project_onto_manifold(torch.randn(B, N, 257))
        mask = torch.ones(B, N)
        mask[0, 4:] = 0
        scores = cone(node_points, manifold, mask=mask)
        assert (scores[0, 4:, :] == 0).all()
        assert (scores[0, :, 4:] == 0).all()

    def test_bias_scale_starts_at_zero(self):
        cone = HyperbolicConeAttention()
        assert cone.bias_scale.item() == 0.0


class TestProposalToPatchMapper:
    def test_output_shape(self, sample_boxes):
        boxes, mask = sample_boxes
        B, N = boxes.shape[:2]
        mapper = ProposalToPatchMapper(grid_size=24)
        cone_scores = torch.randn(B, N, N)
        patch_bias = mapper(cone_scores, boxes, mask)
        assert patch_bias.shape == (B, 576, 576)

    def test_assignment_normalization(self, sample_boxes):
        boxes, mask = sample_boxes
        mapper = ProposalToPatchMapper(grid_size=24)
        A = mapper._compute_iou_assignment(boxes, mask, boxes.device)
        row_sums = A.sum(dim=2)
        # Patches overlapping at least one proposal should sum to ~1
        overlapping = row_sums > 0.5
        assert overlapping.any(), "Some patches should overlap proposals"
        assert torch.allclose(row_sums[overlapping], torch.ones_like(row_sums[overlapping]), atol=1e-4)

    def test_small_grid(self):
        mapper = ProposalToPatchMapper(grid_size=4)
        B, N = 1, 2
        boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]])
        mask = torch.ones(B, N)
        cone_scores = torch.tensor([[[1.0, 0.5], [0.3, 1.0]]])
        patch_bias = mapper(cone_scores, boxes, mask)
        assert patch_bias.shape == (1, 16, 16)
        assert not torch.isnan(patch_bias).any()


class TestDenseVisualRelations:
    def test_output_shape(self, sample_boxes):
        boxes, mask = sample_boxes
        B, N = boxes.shape[:2]
        dvr = DenseVisualRelations(vis_dim=64, embed_dim=32, num_heads=2)
        features = torch.randn(B, N, 64)
        dvr.use_checkpoint = False
        rel = dvr(features, boxes, mask)
        assert rel.shape == (B, N, N, 32)


class TestHRAGraph:
    def test_end_to_end_shapes(self, sample_boxes):
        boxes, mask = sample_boxes
        B, N = boxes.shape[:2]
        graph = HRAGraph(vis_dim=64, embed_dim=32, num_heads=2, max_proposals=N, grid_size=4)
        graph.dense_relations.use_checkpoint = False
        features = torch.randn(B, N, 64)
        patch_bias, rel_points, ent_loss, ang_loss = graph(features, boxes, mask)
        assert patch_bias.shape == (B, 16, 16)
        assert rel_points.shape == (B, N * N, 33)  # embed_dim + 1

    def test_gradient_chain(self, sample_boxes):
        boxes, mask = sample_boxes
        B, N = boxes.shape[:2]
        graph = HRAGraph(vis_dim=64, embed_dim=32, num_heads=2, max_proposals=N, grid_size=4)
        graph.dense_relations.use_checkpoint = False
        features = torch.randn(B, N, 64)
        patch_bias, rel_points, ent_loss, ang_loss = graph(features, boxes, mask)
        loss = patch_bias.sum() + ent_loss + ang_loss
        loss.backward()
        assert graph.kappa.grad is not None or graph.cone_attn.eta.grad is not None
        for name, p in graph.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


class TestSpatialEncoder:
    def test_deltas_shape(self, sample_boxes):
        boxes, mask = sample_boxes
        enc = SpatialEncoder()
        deltas = enc.compute_deltas(boxes)
        B, N = boxes.shape[:2]
        assert deltas.shape == (B, N, N, 4)

    def test_self_deltas_zero(self):
        boxes = torch.tensor([[[0.1, 0.2, 0.5, 0.6]]])
        enc = SpatialEncoder()
        deltas = enc.compute_deltas(boxes)
        assert torch.allclose(deltas[0, 0, 0, :2], torch.zeros(2), atol=1e-5)
