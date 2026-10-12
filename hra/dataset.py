"""GQA dataset: images with bounding boxes from scene graphs. RoI features extracted at forward time."""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402
MAX_PROPOSALS = 36


class GQAVisualDataset(Dataset):
    """Yields image, question, answer, and object bounding boxes (no text predicates)."""
    def __init__(self, split="train", processor=None, max_proposals=MAX_PROPOSALS):
        self.image_dir = os.path.join(DATA, "gqa/images")
        self.max_proposals = max_proposals
        self.processor = processor

        q_file = os.path.join(DATA, f"gqa/{split}_balanced_questions.json")
        with open(q_file) as f:
            self.questions = json.load(f)
        self.qids = list(self.questions.keys())

        # Load scene graphs for bounding boxes
        sg_path = os.path.join(DATA, "gqa/train_sceneGraphs.json")
        val_sg_path = os.path.join(DATA, "gqa/val_sceneGraphs.json")
        self.scene_graphs = {}
        for path in [sg_path, val_sg_path]:
            if os.path.exists(path):
                with open(path) as f:
                    self.scene_graphs.update(json.load(f))

    def __len__(self):
        return len(self.qids)

    def _get_boxes(self, image_id, img_w, img_h):
        """Extract normalized [x1, y1, x2, y2] boxes from scene graph."""
        sg = self.scene_graphs.get(image_id, {})
        objects = sg.get("objects", {})
        boxes = []
        for obj in objects.values():
            x, y, w, h = obj.get("x", 0), obj.get("y", 0), obj.get("w", 1), obj.get("h", 1)
            x1 = x / max(img_w, 1)
            y1 = y / max(img_h, 1)
            x2 = (x + w) / max(img_w, 1)
            y2 = (y + h) / max(img_h, 1)
            boxes.append([x1, y1, x2, y2])
            if len(boxes) >= self.max_proposals:
                break
        return boxes

    def __getitem__(self, idx):
        qid = self.qids[idx]
        entry = self.questions[qid]
        image_id = entry.get("imageId", qid.split("_")[0])
        question = entry.get("question", "")
        answer = entry.get("answer", "")

        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        boxes = self._get_boxes(image_id, img_w, img_h)

        if self.processor is not None:
            prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            input_ids = inputs["input_ids"].squeeze(0)
        else:
            pixel_values = torch.zeros(3, 336, 336)
            input_ids = torch.zeros(1, dtype=torch.long)

        # Pad/truncate boxes to max_proposals
        N = len(boxes)
        box_tensor = torch.zeros(self.max_proposals, 4)
        box_mask = torch.zeros(self.max_proposals)
        for i, b in enumerate(boxes[:self.max_proposals]):
            box_tensor[i] = torch.tensor(b, dtype=torch.float32)
            box_mask[i] = 1.0

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "question": question,
            "answer": answer,
            "image_id": image_id,
            "boxes": box_tensor,
            "box_mask": box_mask,
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    boxes = torch.stack([b["boxes"] for b in batch])
    box_masks = torch.stack([b["box_mask"] for b in batch])
    answers = [b["answer"] for b in batch]

    max_ids_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_ids_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_ids_len, dtype=torch.long)
    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = 1

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "boxes": boxes,
        "box_mask": box_masks,
        "answers": answers,
    }
