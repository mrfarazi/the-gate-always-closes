#!/usr/bin/env python3
"""Download the Hugging Face datasets the evaluations read from disk (datasets.save_to_disk layout).

    python scripts/data/download_hf_datasets.py [--only sugarcrepe,vsr,winoground]

SugarCrepe (7 sub-tasks) -> data/sugarcrepe/SugarCrepe_<subtask>/ ; VSR test -> data/vsr/test/ ;
Winoground (gated: accept the terms on the Hub and `hf auth login` first) -> data/winoground/test/.
GQA, COCO and POPE are not on the Hub in this form: see docs/DATA.md.
"""
import argparse
import os

from hra.paths import DATA

SC = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="sugarcrepe,vsr,winoground")
    args = ap.parse_args()
    from datasets import load_dataset
    want = set(args.only.split(","))
    if "sugarcrepe" in want:
        for st in SC:
            dst = os.path.join(DATA, "sugarcrepe", f"SugarCrepe_{st}")
            if os.path.exists(dst):
                print("exists", dst); continue
            ds = load_dataset(f"HuggingFaceM4/SugarCrepe_{st}")
            ds.save_to_disk(dst); print("saved", dst)
    if "vsr" in want:
        dst = os.path.join(DATA, "vsr", "test")
        if not os.path.exists(dst):
            load_dataset("cambridgeltl/vsr_random", split="test").save_to_disk(dst); print("saved", dst)
    if "winoground" in want:
        dst = os.path.join(DATA, "winoground", "test")
        if not os.path.exists(dst):
            load_dataset("facebook/winoground", split="test").save_to_disk(dst); print("saved", dst)


if __name__ == "__main__":
    main()
