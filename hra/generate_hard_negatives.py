"""Generate hard negative caption pairs from Visual Genome relationships and COCO captions."""

import json
import os
import random

from hra.paths import ROOT as BASE, DATA, LLAVA_PATH, RUNS, RESULTS  # noqa: E402


def load_vg_relationships(max_images=50000):
    vg_path = os.path.join(DATA, "vg/relationships.json")
    with open(vg_path) as f:
        data = json.load(f)

    pairs = []
    predicates = set()
    objects = set()

    for img_data in data[:max_images]:
        image_id = img_data["image_id"]
        for rel in img_data.get("relationships", []):
            subj_name = rel.get("subject", {}).get("name", "")
            obj_name = rel.get("object", {}).get("names", [""])[0] if "names" in rel.get("object", {}) else rel.get("object", {}).get("name", "")
            pred = rel.get("predicate", "")
            if subj_name and obj_name and pred:
                pairs.append({
                    "image_id": image_id,
                    "subject": subj_name.lower().strip(),
                    "predicate": pred.lower().strip(),
                    "object": obj_name.lower().strip(),
                })
                predicates.add(pred.lower().strip())
                objects.add(obj_name.lower().strip())
                objects.add(subj_name.lower().strip())

    return pairs, list(predicates), list(objects)


def generate_negatives(pairs, predicates, objects, seed=42):
    rng = random.Random(seed)
    samples = []

    for p in pairs:
        positive = f"{p['subject']} {p['predicate']} {p['object']}"

        # Replace-Relation
        alt_pred = rng.choice(predicates)
        while alt_pred == p["predicate"]:
            alt_pred = rng.choice(predicates)
        neg_rel = f"{p['subject']} {alt_pred} {p['object']}"

        # Replace-Object
        alt_obj = rng.choice(objects)
        while alt_obj == p["object"]:
            alt_obj = rng.choice(objects)
        neg_obj = f"{p['subject']} {p['predicate']} {alt_obj}"

        # Replace-Subject
        alt_subj = rng.choice(objects)
        while alt_subj == p["subject"]:
            alt_subj = rng.choice(objects)
        neg_subj = f"{alt_subj} {p['predicate']} {p['object']}"

        # Swap subject/object
        neg_swap = f"{p['object']} {p['predicate']} {p['subject']}"

        samples.append({
            "image_id": p["image_id"],
            "positive": positive,
            "negatives": {
                "replace_rel": neg_rel,
                "replace_obj": neg_obj,
                "replace_subj": neg_subj,
                "swap": neg_swap,
            }
        })

    return samples


def main():
    print("Loading VG relationships...")
    pairs, predicates, objects = load_vg_relationships()
    print(f"  {len(pairs)} relationship pairs, {len(predicates)} predicates, {len(objects)} objects")

    print("Generating hard negatives...")
    samples = generate_negatives(pairs, predicates, objects)

    # Subsample for training
    rng = random.Random(42)
    rng.shuffle(samples)
    train_samples = samples[:50000]

    out_path = os.path.join(DATA, "compositional_pairs.json")
    with open(out_path, "w") as f:
        json.dump(train_samples, f)
    print(f"Saved {len(train_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
