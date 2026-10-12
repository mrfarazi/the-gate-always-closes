import json
import os

import pytest
import torch

from hra import hub
from hra.checkpoints import (build_prefix_graph_from_dir, read_variant, resolve_variant,
                             write_variant_json)
from hra.train_prefix_fixed import (VARIANT_CONFIGS, FixedPrefixGraph, RandomFixedPrefixGraph,
                                    build_prefix_graph_for_eval)


def test_variant_json_roundtrip(tmp_path):
    write_variant_json(str(tmp_path), kind="prefix", variant="raw_gated", seed=1)
    meta = read_variant(str(tmp_path))
    assert meta["schema"] == 1 and meta["variant"] == "raw_gated" and meta["seed"] == 1
    assert resolve_variant(str(tmp_path)) == "raw_gated"
    assert resolve_variant(str(tmp_path), "raw_gated") == "raw_gated"
    with pytest.raises(ValueError):
        resolve_variant(str(tmp_path), "base")


def test_missing_variant_json_requires_explicit_variant(tmp_path):
    with pytest.raises(ValueError):
        resolve_variant(str(tmp_path))
    assert resolve_variant(str(tmp_path), "base") == "base"


def test_g2_and_random_fixed_dispatch():
    assert "g2_no_geo" in VARIANT_CONFIGS and VARIANT_CONFIGS["g2_no_geo"]["alpha_max"] == 0.0
    assert isinstance(build_prefix_graph_for_eval("g2_no_geo"), FixedPrefixGraph)
    assert isinstance(build_prefix_graph_for_eval("random_fixed"), RandomFixedPrefixGraph)
    pg = build_prefix_graph_for_eval("rms_ungated")
    assert pg.prefix_proj.ungated is True
    pg = build_prefix_graph_for_eval("raw_gated")
    assert pg.prefix_proj.no_rms is True and pg.prefix_proj.ungated is False


def test_state_dict_roundtrip_through_variant_json(tmp_path):
    pg = build_prefix_graph_for_eval("raw_gated")
    torch.save({"prefix_graph": pg.state_dict()}, tmp_path / "prefix_modules.pt")
    write_variant_json(str(tmp_path), kind="prefix", variant="raw_gated")
    pg2, v = build_prefix_graph_from_dir(str(tmp_path))
    assert v == "raw_gated" and pg2.prefix_proj.no_rms is True


def test_registry_is_consistent():
    for name, spec in hub.REGISTRY.items():
        assert spec["repo"].startswith(hub.HF_OWNER + "/tgac-")
        if spec["kind"] == "prefix":
            assert spec["variant"] in VARIANT_CONFIGS, name
    assert hub.REGISTRY["g2"]["variant"] == "g2_no_geo"
    assert hub.resolve(os.getcwd()) == os.getcwd()   # a directory path resolves to itself
    with pytest.raises(KeyError):
        hub.resolve("no_such_checkpoint")


@pytest.mark.skipif(not os.environ.get("HRA_LOCAL_CHECKPOINTS"), reason="needs local checkpoints")
def test_published_g2_loads_as_g2_no_geo():
    d = hub.resolve("g2", download=False)
    pg, v = build_prefix_graph_from_dir(d, "g2_no_geo")
    assert v == "g2_no_geo" and isinstance(pg, FixedPrefixGraph)


def test_set_prefix_scale_zeroes_tokens_and_keeps_identity_at_one():
    import types
    import torch
    from hra.checkpoints import set_prefix_scale

    class G(torch.nn.Module):
        def forward(self, x):
            return x, None, torch.tensor(0.0), torch.tensor(0.0)

    w = types.SimpleNamespace(prefix_graph=G())
    set_prefix_scale(w, 0.0)
    out = w.prefix_graph(torch.ones(2, 3))
    assert torch.equal(out[0], torch.zeros(2, 3)) and out[1] is None and len(out) == 4
    w1 = types.SimpleNamespace(prefix_graph=G())
    set_prefix_scale(w1, 1.0)
    assert torch.equal(w1.prefix_graph(torch.ones(1))[0], torch.ones(1))
