#!/usr/bin/env python3
"""Report which of the expected data files / directories are present under HRA_DATA (docs/DATA.md)."""
import os

from hra.paths import DATA, MODELS

EXPECTED = {
    "GQA questions (train/eval)": ["gqa/train_balanced_questions.json", "gqa/testdev_balanced_questions.json"],
    "GQA scene graphs (training boxes)": ["gqa/train_sceneGraphs.json", "gqa/val_sceneGraphs.json"],
    "GQA images": ["gqa/images"],
    "SugarCrepe (7 sub-tasks)": [f"sugarcrepe/SugarCrepe_{s}" for s in
                                  ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]],
    "Winoground": ["winoground/test"],
    "VSR + COCO train2017": ["vsr/test", "coco/train2017"],
    "POPE + COCO val2014": ["pope/coco_pope_random.json", "pope/coco_pope_popular.json", "pope/coco_pope_adversarial.json", "coco/val2014"],
    "COCO captioning (val2017)": ["coco/val2017", "coco/annotations/captions_val2017.json"],
    "S3 hard-negative pairs": ["compositional_pairs.json"],
}


def main():
    print(f"HRA_DATA = {DATA}")
    missing = 0
    for group, items in EXPECTED.items():
        got = [os.path.exists(os.path.join(DATA, i)) for i in items]
        status = "ok" if all(got) else ("partial" if any(got) else "MISSING")
        missing += not all(got)
        print(f"  [{status:7s}] {group}")
        for i, g in zip(items, got):
            if not g:
                print(f"            missing: {i}")
    llava = os.path.join(MODELS, "llava-1.5-7b")
    print(f"  [{'ok' if os.path.isdir(llava) else 'hub    '}] LLaVA-1.5-7B weights: "
          f"{llava if os.path.isdir(llava) else 'not local; llava-hf/llava-1.5-7b-hf will be downloaded'}")
    print("all present" if not missing else f"{missing} group(s) incomplete; see docs/DATA.md")


if __name__ == "__main__":
    main()
