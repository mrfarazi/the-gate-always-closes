import torch

from hra.boxes import boxes_or_grid, grid_boxes, grid_boxes_tensor, scene_graph_boxes


def test_grid_boxes_cover_unit_square():
    b = grid_boxes(6)
    assert len(b) == 36
    assert min(x[0] for x in b) == 0.0 and max(x[2] for x in b) == 1.0


def test_grid_tensor_padding():
    boxes, mask = grid_boxes_tensor(6, max_n=40, batch=False)
    assert boxes.shape == (40, 4) and mask.shape == (40,)
    assert int(mask.sum()) == 36


def test_scene_graph_boxes_normalised():
    sg = {"objects": {"1": {"x": 0, "y": 0, "w": 50, "h": 25}, "2": {"x": 50, "y": 50, "w": 50, "h": 50}}}
    b = scene_graph_boxes(sg, 100, 100)
    assert b == [[0.0, 0.0, 0.5, 0.25], [0.5, 0.5, 1.0, 1.0]]


def test_fallback_to_grid_when_no_scene_graph():
    bt, bm, src = boxes_or_grid(None, 640, 480, max_n=36)
    assert src == "grid" and int(bm.sum()) == 36 and bt.shape == (36, 4)
    bt, bm, src = boxes_or_grid({"objects": {}}, 640, 480, max_n=36)
    assert src == "grid"


def test_scene_graph_used_when_present():
    sg = {"objects": {"1": {"x": 0, "y": 0, "w": 10, "h": 10}}}
    bt, bm, src = boxes_or_grid(sg, 100, 100, max_n=36)
    assert src == "scene_graph" and int(bm.sum()) == 1
    assert torch.allclose(bt[0], torch.tensor([0.0, 0.0, 0.1, 0.1]))
