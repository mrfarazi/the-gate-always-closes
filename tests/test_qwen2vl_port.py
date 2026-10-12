"""CPU tests for the Qwen2-VL port (hra/backbones/qwen2vl.py). No GPU, no downloads."""
import os

import pytest
import torch

from hra import hub
from hra.backbones import qwen2vl
from hra.boxes import grid_boxes_tensor

HIDDEN = 3584
EXPECTED_KEYS = 25


def _graph():
    torch.manual_seed(0)
    return qwen2vl.build_prefix_graph(HIDDEN, k=qwen2vl.K_PREFIX)


def test_import_is_lazy_and_graph_shape():
    # importing the module must not touch the hub / a GPU; the graph is the S5 form widened
    # to 4 x 3584, with the projector's unused RMSNorm left at its default width.
    assert qwen2vl.QWEN2VL_ID.endswith("Qwen2-VL-7B-Instruct")
    pg = _graph()
    assert pg.prefix_proj.no_rms is True and pg.prefix_proj.ungated is False
    assert tuple(pg.prefix_proj.proj.weight.shape) == (qwen2vl.K_PREFIX * HIDDEN, pg.embed_dim)
    assert tuple(pg.prefix_proj.rms_norm.weight.shape) == (4096,)
    state = pg.state_dict()
    assert len(state) == EXPECTED_KEYS
    assert tuple(state["dense_relations.q_proj.weight"].shape) == (pg.embed_dim, HIDDEN)


def test_graph_tokens_gate_and_override_semantics():
    pg = _graph()
    with torch.no_grad():
        pg.prefix_proj.gate.fill_(0.3)
    boxes, mask = grid_boxes_tensor(6, max_n=qwen2vl.MAX_PROPOSALS)
    feats = torch.randn(1, qwen2vl.MAX_PROPOSALS, HIDDEN)
    with torch.no_grad():
        live, ent, ang = qwen2vl.graph_tokens(pg, feats, boxes, mask)
        ref, _, _, _ = pg(feats, boxes, mask)                      # graph's own gated output
        forced0, _, _ = qwen2vl.graph_tokens(pg, feats, boxes, mask, gamma_override=0.0)
        forced1, _, _ = qwen2vl.graph_tokens(pg, feats, boxes, mask, gamma_override=1.0)
    assert live.shape == (1, qwen2vl.K_PREFIX, HIDDEN)
    assert torch.allclose(live, ref, atol=1e-6)                     # gate applied exactly once
    assert torch.count_nonzero(forced0) == 0
    assert torch.allclose(forced1 * torch.tanh(torch.tensor(0.3)), live, atol=1e-6)
    assert pg.prefix_proj.ungated is False                          # flag restored
    assert torch.isfinite(ent) and torch.isfinite(ang)


@pytest.mark.skipif(not os.environ.get("HRA_LOCAL_CHECKPOINTS"), reason="needs local checkpoints")
def test_seed0_prefix_modules_load_strict():
    d = hub.resolve("qwen2vl_seed0", download=False)
    state = torch.load(os.path.join(d, "prefix_modules.pt"), map_location="cpu")
    assert set(state) >= {"prefix_graph", "kappa", "gate"}
    assert len(state["prefix_graph"]) == EXPECTED_KEYS
    pg = _graph()
    pg.load_state_dict(state["prefix_graph"], strict=True)
    assert torch.equal(pg.prefix_proj.gate.data, state["gate"])
    assert abs(float(torch.tanh(pg.prefix_proj.gate))) < 0.05        # the gate closed
    assert 0.1 <= float(pg.kappa) <= 5.0
