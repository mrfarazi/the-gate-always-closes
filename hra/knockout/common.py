#!/usr/bin/env python3
"""Shared helpers for the knock-out and sweep scripts (J4 / J5 / J7 / InternVL).

Nothing here imports torch/transformers at module level, so `setup_env()` can be called
before those libraries read HF_HOME.
"""
import datetime
import json
import math
import os
import sys
import time

from hra.paths import ROOT as ACCV_BASE, RESULTS, DATA  # repository root; datasets under DATA

# The HyperVis (BMVC 2026) checkpoint used for the Raw/None cells lives in a separate repository;
# set HYPERVIS_ROOT to its checkout to re-run those rows (the shipped results do not need it).
HV_BASE = os.environ.get("HYPERVIS_ROOT")
LOG_MD = os.path.join(RESULTS, "knockouts", "LOG.md")

SC_SUBTASKS = [
    "SugarCrepe_add_att", "SugarCrepe_add_obj",
    "SugarCrepe_replace_att", "SugarCrepe_replace_obj", "SugarCrepe_replace_rel",
    "SugarCrepe_swap_att", "SugarCrepe_swap_obj",
]


# ------------------------------------------------------------------ environment / logging
def setup_env(need_tmpfs_hub=False):
    """Call BEFORE importing transformers. Respects the user's HF_HOME / HF_HUB_OFFLINE."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg):
    """Append one timestamped line to results/knockouts/LOG.md (only if HRA_KNOCKOUT_LOG=1)."""
    if os.environ.get("HRA_KNOCKOUT_LOG") != "1":
        print(f"[{now()}] {msg}")
        return
    os.makedirs(os.path.dirname(LOG_MD), exist_ok=True)
    new = not os.path.exists(LOG_MD)
    with open(LOG_MD, "a") as f:
        if new:
            f.write("# Journal eval log\n\nOne line per script start/end. "
                    "Format: `- [time] SCRIPT event | details`.\n\n")
        f.write(f"- [{now()}] {msg}\n")


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def skip_if_exists(out, script):
    if os.path.exists(out):
        msg = f"{script} SKIP | output exists: {out}"
        print(f"[{script}] {msg}", flush=True)
        log_line(msg)
        return True
    return False


def cmdline():
    return " ".join(sys.argv)


# ------------------------------------------------------------------ value lists
def parse_values(spec, learned=None):
    """Parse '0,1e-3,learned,-learned' -> list of floats, order kept, duplicates dropped."""
    vals = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "learned":
            if learned is None or not math.isfinite(learned):
                continue
            v = float(learned)
        elif tok == "-learned":
            if learned is None or not math.isfinite(learned):
                continue
            v = -float(learned)
        else:
            v = float(tok)
        if not any(abs(v - u) <= 1e-12 * max(1.0, abs(v)) for u in vals):
            vals.append(v)
    return vals


def ensure_baseline_first(vals, baseline=0.0):
    vals = [v for v in vals if v != baseline]
    return [baseline] + vals


# ------------------------------------------------------------------ statistics
def paired_bootstrap(a, b, n_boot=5000, seed=0):
    """Paired bootstrap of mean(a - b). a, b: equal-length 1-D arrays.
    Returns (delta, lo, hi) on the same scale as the inputs."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def bootstrap_mean_ci(x, n_boot=5000, seed=0):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def summarise_binary(values, correct_by_value, baseline=0.0, n_boot=5000, seed=0):
    """correct_by_value: {value: [0/1 per item]} -> rows with acc, delta vs baseline, CI."""
    import numpy as np
    base = np.asarray(correct_by_value[baseline], dtype=np.float64)
    rows = []
    for v in values:
        c = np.asarray(correct_by_value[v], dtype=np.float64)
        acc = 100.0 * c.mean() if len(c) else float("nan")
        d, lo, hi = paired_bootstrap(c, base, n_boot=n_boot, seed=seed)
        rows.append({"value": v, "acc": acc, "n_correct": int(c.sum()),
                     "delta_vs_0_pp": 100.0 * d, "ci95": [100.0 * lo, 100.0 * hi]})
    return rows


# ------------------------------------------------------------------ data: SugarCrepe
def load_sugarcrepe_items(base, n):
    """Fixed, paired item list: round-robin over the 7 subtasks (item i of each subtask
    in canonical order, then item i+1, ...), truncated to n. Mirrors the pos/neg
    convention of eval_compositionality.eval_sugarcrepe. Returns (items, datasets)."""
    from datasets import load_from_disk
    dss = {}
    for st in SC_SUBTASKS:
        p = os.path.join(DATA, "sugarcrepe", st)
        if os.path.exists(p):
            dss[st] = load_from_disk(p)
    items = []
    i = 0
    while len(items) < n:
        progressed = False
        for st in SC_SUBTASKS:
            ds = dss.get(st)
            if ds is None or i >= len(ds):
                continue
            progressed = True
            true_label = ds.features["true_label"].int2str(int(ds[i]["true_label"]))
            labels = ds[i]["tested_labels"]
            neg = labels[1] if labels[0] == true_label else labels[0]
            items.append({"id": f"{st}:{i}", "subtask": st, "idx": i,
                          "pos": true_label, "neg": neg})
            if len(items) >= n:
                break
        if not progressed:
            break
        i += 1
    return items, dss


def sc_image(dss, item):
    return dss[item["subtask"]][item["idx"]]["image"].convert("RGB")


# ------------------------------------------------------------------ data: GQA
def load_gqa_items(base, bench, n):
    """bench: 'gqa' (full testdev, dict order) or 'gqa_rel' (semantic == rel, dict order,
    same filter as eval_gqa_spatial.filter_gqa_by_type). First n."""
    with open(os.path.join(DATA, "gqa/testdev_balanced_questions.json")) as f:
        qs = json.load(f)
    items = []
    for qid, q in qs.items():
        if bench == "gqa_rel" and q.get("types", {}).get("semantic") != "rel":
            continue
        items.append({"id": qid, "question": q["question"],
                      "answer": str(q["answer"]).strip().lower(),
                      "imageId": q["imageId"]})
        if len(items) >= n:
            break
    return items


def gqa_image_path(base, item):
    return os.path.join(DATA, "gqa/images", f"{item['imageId']}.jpg")


def normalize_gqa_pred(text):
    """Answer normalisation of src/eval_gqa_spatial.py / eval_gqa_parallel.py."""
    if "ASSISTANT:" in text:
        pred = text.split("ASSISTANT:")[-1].strip()
    else:
        pred = text.strip()
    for sep in [",", ".", ";", "!", "?"]:
        if sep in pred:
            pred = pred[:pred.index(sep)]
    return pred.strip().lower()


GQA_PROMPT = ("USER: <image>\n{q}\nAnswer the question using a single word or phrase."
              "\nASSISTANT:")
SC_PROMPT = "USER: <image>\nDescribe this image.\nASSISTANT: {caption}"


def caption_labels(processor, input_ids):
    """labels = input_ids with everything up to and including 'ASSISTANT:' masked
    (identical to eval_compositionality.sequence_log_likelihood)."""
    tokenizer = processor.tokenizer
    full_ids = input_ids[0].tolist()
    asst = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    start = None
    for pos in range(len(full_ids) - len(asst)):
        if full_ids[pos:pos + len(asst)] == asst:
            start = pos + len(asst)
            break
    labels = input_ids.clone()
    if start is not None:
        labels[0, :start] = -100
    return labels


# ------------------------------------------------------------------ RMSNorm epsilon patch
def patch_rmsnorm_eps(root, eps, exclude_substrings=("vision",), class_suffix="RMSNorm"):
    """Set `variance_epsilon` on every *RMSNorm module under `root` whose qualified
    name contains none of `exclude_substrings`. Returns (n_patched, restore_fn,
    native_eps_values)."""
    patched = []
    natives = set()
    for name, m in root.named_modules():
        if not type(m).__name__.endswith(class_suffix):
            continue
        if not hasattr(m, "variance_epsilon"):
            continue
        if any(s in name for s in exclude_substrings):
            continue
        patched.append((m, float(m.variance_epsilon)))
        natives.add(float(m.variance_epsilon))
        m.variance_epsilon = float(eps)

    def restore():
        for m, old in patched:
            m.variance_epsilon = old
    return len(patched), restore, sorted(natives)


def native_rmsnorm_eps(root, exclude_substrings=("vision",), class_suffix="RMSNorm"):
    for name, m in root.named_modules():
        if type(m).__name__.endswith(class_suffix) and hasattr(m, "variance_epsilon") \
                and not any(s in name for s in exclude_substrings):
            return float(m.variance_epsilon)
    return None


# ------------------------------------------------------------------ misc
class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0


def grid_boxes(n=6):
    s = 1.0 / n
    return [[j * s, i * s, (j + 1) * s, (i + 1) * s] for i in range(n) for j in range(n)]
