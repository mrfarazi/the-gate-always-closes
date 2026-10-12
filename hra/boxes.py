"""Region proposals for evaluation.

Training always used GQA scene-graph boxes. Most evaluation sets (GQA testdev, COCO-based
SugarCrepe / POPE / captioning, VSR) have no scene graphs, so evaluation uses a fixed 6x6 grid of
proposals; `boxes_or_grid` falls back to the grid whenever a scene graph is missing and asserts
that the proposal set is never empty (an empty set makes the attention bias identically zero and
the prefix image-independent; see docs/PROTOCOL.md).
"""
import torch


def grid_boxes(n=6):
    """n*n normalised [x1, y1, x2, y2] boxes covering the image, row-major."""
    step = 1.0 / n
    return [[j * step, i * step, (j + 1) * step, (i + 1) * step] for i in range(n) for j in range(n)]


def grid_boxes_tensor(n=6, max_n=None, batch=True):
    """Boxes (n*n, 4) and mask (n*n,), zero-padded to `max_n` if given; batched if `batch`."""
    b = torch.tensor(grid_boxes(n), dtype=torch.float32)
    m = torch.ones(b.shape[0])
    if max_n is not None and max_n > b.shape[0]:
        pad = max_n - b.shape[0]
        b = torch.cat([b, torch.zeros(pad, 4)])
        m = torch.cat([m, torch.zeros(pad)])
    return (b.unsqueeze(0), m.unsqueeze(0)) if batch else (b, m)


def scene_graph_boxes(sg, img_w, img_h, max_n=36):
    """Normalised object boxes from one GQA scene-graph entry (dict with 'objects')."""
    boxes = []
    for obj in (sg or {}).get("objects", {}).values():
        x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 1), obj.get("h", 1)
        boxes.append([x / max(img_w, 1), y / max(img_h, 1),
                      (x + w) / max(img_w, 1), (y + h) / max(img_h, 1)])
        if len(boxes) >= max_n:
            break
    return boxes


def boxes_or_grid(sg, img_w, img_h, max_n=36, grid_n=6):
    """(boxes (max_n, 4), mask (max_n,), source) with source in {'scene_graph', 'grid'}."""
    boxes = scene_graph_boxes(sg, img_w, img_h, max_n)
    source = "scene_graph"
    if not boxes:
        boxes, source = grid_boxes(grid_n), "grid"
    bt = torch.zeros(max_n, 4)
    bm = torch.zeros(max_n)
    for j, b in enumerate(boxes[:max_n]):
        bt[j] = torch.tensor(b, dtype=torch.float32)
        bm[j] = 1.0
    assert bm.sum() > 0, "empty proposal set"
    return bt, bm, source
