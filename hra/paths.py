"""Single source of every filesystem location used by the package.

Every location can be overridden with an environment variable; the defaults are relative to
the repository root, so a fresh clone works after `pip install -e .` once `data/` and the base
models are in place (see docs/DATA.md). Values are plain strings so legacy call sites using
os.path.join / f-strings keep working.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

ROOT = os.environ.get("HRA_ROOT", str(_REPO))                       # repository root
DATA = os.environ.get("HRA_DATA", os.path.join(ROOT, "data"))       # datasets (docs/DATA.md)
MODELS = os.environ.get("HRA_MODELS", os.path.join(ROOT, "models")) # optional local base models
CKPT_DIR = os.environ.get("HRA_CKPT_DIR", os.path.join(ROOT, "checkpoints"))  # hub downloads
RUNS = os.environ.get("HRA_RUNS", os.path.join(ROOT, "runs"))       # training outputs
RESULTS = os.path.join(ROOT, "results")                             # shipped result JSONs / tables

_local_llava = os.path.join(MODELS, "llava-1.5-7b")
LLAVA_PATH = os.environ.get("HRA_LLAVA_PATH") or (
    _local_llava if os.path.isdir(_local_llava) else "llava-hf/llava-1.5-7b-hf")
INTERNVL_ID = os.environ.get("HRA_INTERNVL_ID", "OpenGVLab/InternVL3-8B-hf")
QWEN2VL_ID = os.environ.get("HRA_QWEN2VL_ID", "Qwen/Qwen2-VL-7B-Instruct")

PYTHON = sys.executable                     # interpreter used to spawn evaluation shards
LOCAL_FILES_ONLY = os.environ.get("HF_HUB_OFFLINE") == "1"


def data_path(*parts):
    return os.path.join(DATA, *parts)
