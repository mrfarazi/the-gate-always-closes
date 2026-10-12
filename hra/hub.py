"""Checkpoint registry and Hugging Face Hub download helper.

Every configuration reported in the paper has a short registry name (the keys below). A name
resolves to a local directory containing `lora/`, `prefix_modules.pt` or `hra_modules.pt`, and a
`variant.json` written at training or upload time. Resolution order:

1. an existing directory path passed instead of a name;
2. `HRA_LOCAL_CHECKPOINTS` (JSON file mapping name -> local directory; used on the training
   machine before/without uploading);
3. `HRA_CKPT_DIR/<name>` if already downloaded;
4. `huggingface_hub.snapshot_download` of the model repo `<HRA_HF_ORG>/tgac-<backbone>-<name>`.
"""
import json
import os

from hra.paths import CKPT_DIR

HF_OWNER = os.environ.get("HRA_HF_ORG", "mrfarazi")
_BACKBONE_TAG = {"llava-1.5-7b": "llava15", "internvl3-8b": "internvl3", "qwen2-vl-7b": "qwen2vl"}


def _e(name, paper, kind, variant=None, backbone="llava-1.5-7b", stage2=False, notes=""):
    return dict(name=name, paper=paper, kind=kind, variant=variant, backbone=backbone,
                stage2=stage2, notes=notes,
                repo=f"{HF_OWNER}/tgac-{_BACKBONE_TAG[backbone]}-{name.replace('_', '-')}")


# name -> spec. `kind`: hra (attention bias), hra_stage2 (attention bias + Stage-2 gates),
# prefix (gated/ungated prefix on LLaVA-1.5), prefix_internvl, prefix_qwen2vl.
REGISTRY = {
    # attention-bias designs (Sec. 3.2)
    "hra_v1":      _e("hra_v1", "S1 (attention bias, alpha_max 0.05)", "hra"),
    "small_alpha": _e("small_alpha", "S2 (attention bias, alpha_max 0.01)", "hra"),
    "stage2":      _e("stage2", "S3 (margin loss, LoRA frozen)", "hra_stage2", stage2=True,
                      notes="Stage-1 checkpoint (detach_bias) plus stage2_checkpoint.pt (beta, head/layer gates)."),
    # gated-prefix designs and controls
    "no_dropout":  _e("no_dropout", "S4 (gated RMS prefix, no dropout)", "prefix", "no_dropout"),
    "raw_gated":   _e("raw_gated", "S5 (raw gated prefix)", "prefix", "raw_gated"),
    "rms_ungated": _e("rms_ungated", "RMS prefix, ungated (published)", "prefix", "rms_ungated"),
    "rms_ungated_r2": _e("rms_ungated_r2", "RMS prefix, ungated (run 2)", "prefix", "rms_ungated"),
    "rms_ungated_r3": _e("rms_ungated_r3", "RMS prefix, ungated (run 3)", "prefix", "rms_ungated"),
    "base_r1":     _e("base_r1", "RMS prefix, gated (2x2 cell)", "prefix", "base"),
    "soft_prompt": _e("soft_prompt", "soft prompt", "prefix", "soft_prompt"),
    "soft_prompt_r2": _e("soft_prompt_r2", "soft prompt (seed 2)", "prefix", "soft_prompt"),
    "euclidean":   _e("euclidean", "Euclidean gated, no geometric losses", "prefix", "euclidean"),
    "euclidean_geo_r1": _e("euclidean_geo_r1", "Euclidean containment control (seed 1)", "prefix", "euclidean_geo"),
    "euclidean_geo_s2": _e("euclidean_geo_s2", "Euclidean containment control (seed 2, 1 epoch)", "prefix", "euclidean_geo"),
    "euclidean_geo_s3": _e("euclidean_geo_s3", "Euclidean containment control (seed 3, 1 epoch)", "prefix", "euclidean_geo"),
    "g2":          _e("g2", "G2 (hyperbolic prefix, geometric losses removed)", "prefix", "g2_no_geo",
                      notes="Gated FixedPrefixGraph trained with alpha_max=0 (geometric-loss ramp held at 0.001); see docs/PROTOCOL.md."),
    "synthesis":   _e("synthesis", "synthesis (second soft-prompt run, effective batch 96)", "prefix", "soft_prompt"),
    "lora_all_linear_r1": _e("lora_all_linear_r1", "LoRA on all linear layers (seed 1, 1 epoch)", "prefix", "lora_all_linear"),
    "lora_all_linear_s2": _e("lora_all_linear_s2", "LoRA on all linear layers (seed 2, 1 epoch)", "prefix", "lora_all_linear"),
    "rms_dropout_r1": _e("rms_dropout_r1", "RMS prefix, ungated, 50% prefix dropout", "prefix", "rms_dropout"),
    # other backbones (S5 recipe)
    "internvl3_seed0": _e("internvl3_seed0", "InternVL3-8B, S5 recipe, seed 0", "prefix_internvl", "raw_gated", "internvl3-8b"),
    "internvl3_seed1": _e("internvl3_seed1", "InternVL3-8B, S5 recipe, seed 1", "prefix_internvl", "raw_gated", "internvl3-8b"),
    "internvl3_seed2": _e("internvl3_seed2", "InternVL3-8B, S5 recipe, seed 2", "prefix_internvl", "raw_gated", "internvl3-8b"),
    "qwen2vl_seed0": _e("qwen2vl_seed0", "Qwen2-VL-7B, S5 recipe, seed 0", "prefix_qwen2vl", "raw_gated", "qwen2-vl-7b"),
    "qwen2vl_seed1": _e("qwen2vl_seed1", "Qwen2-VL-7B, S5 recipe, seed 1", "prefix_qwen2vl", "raw_gated", "qwen2-vl-7b"),
    "qwen2vl_seed2": _e("qwen2vl_seed2", "Qwen2-VL-7B, S5 recipe, seed 2", "prefix_qwen2vl", "raw_gated", "qwen2-vl-7b"),
}

_MARKERS = ("variant.json", "prefix_modules.pt", "hra_modules.pt", "lora")


def names(kind=None):
    return [n for n, s in REGISTRY.items() if kind is None or s["kind"] == kind]


def repo_id(name):
    return REGISTRY[name]["repo"]


def local_dir(name):
    return os.path.join(CKPT_DIR, name)


def _local_override(name):
    manifest = os.environ.get("HRA_LOCAL_CHECKPOINTS")
    if not manifest or not os.path.exists(manifest):
        return None
    with open(manifest) as f:
        m = json.load(f)
    entry = m.get(name)
    if entry is None:
        return None
    return entry["src"] if isinstance(entry, dict) else entry


def _looks_like_checkpoint(d):
    return os.path.isdir(d) and any(os.path.exists(os.path.join(d, m)) for m in _MARKERS)


def resolve(name_or_path, download=True):
    """Return a local checkpoint directory for a registry name (downloading if needed) or an
    existing path."""
    if os.path.isdir(str(name_or_path)):
        return str(name_or_path)
    if name_or_path not in REGISTRY:
        raise KeyError(f"'{name_or_path}' is neither a directory nor a registry name; "
                       f"known names: {', '.join(REGISTRY)}")
    override = _local_override(name_or_path)
    if override and _looks_like_checkpoint(override):
        return override
    d = local_dir(name_or_path)
    if _looks_like_checkpoint(d):
        return d
    if not download:
        raise FileNotFoundError(f"checkpoint '{name_or_path}' not found under {CKPT_DIR}")
    return download_checkpoint(name_or_path)


def download_checkpoint(name, force=False):
    from huggingface_hub import snapshot_download
    d = local_dir(name)
    if _looks_like_checkpoint(d) and not force:
        return d
    os.makedirs(d, exist_ok=True)
    snapshot_download(repo_id(name), local_dir=d)
    return d


def stage2_path(name_or_path):
    return os.path.join(resolve(name_or_path), "stage2_checkpoint.pt")


if __name__ == "__main__":
    import sys
    for n in sys.argv[1:] or names():
        print(f"{n:22s} {REGISTRY[n]['kind']:16s} {REGISTRY[n]['variant'] or '-':16s} {repo_id(n)}")
