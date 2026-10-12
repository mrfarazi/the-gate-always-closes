# Checkpoints

Every configuration reported in the paper has a short registry name in `hra/hub.py` and one model
repository on the Hugging Face Hub, `<HRA_HF_ORG>/tgac-<backbone tag>-<name with - for _>`
(`HRA_HF_ORG` defaults to `mrfarazi`). The repositories contain only what was trained: the LoRA
adapter and the graph-module weights. Base models are downloaded from their own repositories and
keep their own licences (LLaVA-1.5: Llama 2 community licence).

## Registry

`python hra/hub.py` prints this list. `kind` selects the loader; `variant` is the
`VARIANT_CONFIGS` key of `hra/train_prefix_fixed.py` the prefix graph must be rebuilt with.

| Registry name | Paper label | Kind | Variant | Hub repository |
|---|---|---|---|---|
| `hra_v1` | S1 (attention bias, alpha_max 0.05) | hra | - | `mrfarazi/tgac-llava15-hra-v1` |
| `small_alpha` | S2 (attention bias, alpha_max 0.01) | hra | - | `mrfarazi/tgac-llava15-small-alpha` |
| `stage2` | S3 (margin loss, LoRA frozen) | hra_stage2 | - | `mrfarazi/tgac-llava15-stage2` |
| `no_dropout` | S4 (gated RMS prefix, no dropout) | prefix | `no_dropout` | `mrfarazi/tgac-llava15-no-dropout` |
| `raw_gated` | S5 (raw gated prefix) | prefix | `raw_gated` | `mrfarazi/tgac-llava15-raw-gated` |
| `rms_ungated` | RMS prefix, ungated (published) | prefix | `rms_ungated` | `mrfarazi/tgac-llava15-rms-ungated` |
| `rms_ungated_r2` | RMS prefix, ungated (run 2) | prefix | `rms_ungated` | `mrfarazi/tgac-llava15-rms-ungated-r2` |
| `rms_ungated_r3` | RMS prefix, ungated (run 3) | prefix | `rms_ungated` | `mrfarazi/tgac-llava15-rms-ungated-r3` |
| `base_r1` | RMS prefix, gated (2x2 cell) | prefix | `base` | `mrfarazi/tgac-llava15-base-r1` |
| `soft_prompt` | soft prompt | prefix | `soft_prompt` | `mrfarazi/tgac-llava15-soft-prompt` |
| `soft_prompt_r2` | soft prompt (seed 2) | prefix | `soft_prompt` | `mrfarazi/tgac-llava15-soft-prompt-r2` |
| `euclidean` | Euclidean gated, no geometric losses | prefix | `euclidean` | `mrfarazi/tgac-llava15-euclidean` |
| `euclidean_geo_r1` | Euclidean containment control (seed 1) | prefix | `euclidean_geo` | `mrfarazi/tgac-llava15-euclidean-geo-r1` |
| `euclidean_geo_s2` | Euclidean containment control (seed 2, 1 epoch) | prefix | `euclidean_geo` | `mrfarazi/tgac-llava15-euclidean-geo-s2` |
| `euclidean_geo_s3` | Euclidean containment control (seed 3, 1 epoch) | prefix | `euclidean_geo` | `mrfarazi/tgac-llava15-euclidean-geo-s3` |
| `g2` | G2 (hyperbolic prefix, geometric losses removed) | prefix | `g2_no_geo` | `mrfarazi/tgac-llava15-g2` |
| `synthesis` | synthesis (second soft-prompt run, effective batch 96) | prefix | `soft_prompt` | `mrfarazi/tgac-llava15-synthesis` |
| `lora_all_linear_r1` | LoRA on all linear layers (seed 1, 1 epoch) | prefix | `lora_all_linear` | `mrfarazi/tgac-llava15-lora-all-linear-r1` |
| `lora_all_linear_s2` | LoRA on all linear layers (seed 2, 1 epoch) | prefix | `lora_all_linear` | `mrfarazi/tgac-llava15-lora-all-linear-s2` |
| `rms_dropout_r1` | RMS prefix, ungated, 50% prefix dropout | prefix | `rms_dropout` | `mrfarazi/tgac-llava15-rms-dropout-r1` |
| `internvl3_seed0` | InternVL3-8B, S5 recipe, seed 0 | prefix_internvl | `raw_gated` | `mrfarazi/tgac-internvl3-internvl3-seed0` |
| `internvl3_seed1` | InternVL3-8B, S5 recipe, seed 1 | prefix_internvl | `raw_gated` | `mrfarazi/tgac-internvl3-internvl3-seed1` |
| `internvl3_seed2` | InternVL3-8B, S5 recipe, seed 2 | prefix_internvl | `raw_gated` | `mrfarazi/tgac-internvl3-internvl3-seed2` |
| `qwen2vl_seed0` | Qwen2-VL-7B, S5 recipe, seed 0 | prefix_qwen2vl | `raw_gated` | `mrfarazi/tgac-qwen2vl-qwen2vl-seed0` |
| `qwen2vl_seed1` | Qwen2-VL-7B, S5 recipe, seed 1 | prefix_qwen2vl | `raw_gated` | `mrfarazi/tgac-qwen2vl-qwen2vl-seed1` |
| `qwen2vl_seed2` | Qwen2-VL-7B, S5 recipe, seed 2 | prefix_qwen2vl | `raw_gated` | `mrfarazi/tgac-qwen2vl-qwen2vl-seed2` |

Notes on two entries:

- `stage2` holds the S3 stage-1 checkpoint (trained with `attention detach_bias`) plus
  `stage2_checkpoint.pt` with the stage-2 parameters (`bias_scale` beta, `head_gates`,
  `layer_gates`). `hub.stage2_path(name)` returns that file; the knock-out code applies the gates.
- `g2` is registered as variant `g2_no_geo`: a gated `FixedPrefixGraph` trained with
  `alpha_max = 0`, the configuration of the run labelled G2 in the paper. See
  [PROTOCOL.md](PROTOCOL.md).

## What a repository contains

Files are staged by `scripts/release/upload_checkpoints.py`:

| Kind | Files |
|---|---|
| `hra` (S1, S2) | `lora/` (PEFT adapter for the LLaVA language model), `hra_modules.pt` (`hra_graph` state dict, `kappa`), `variant.json`, `README.md` (model card) |
| `hra_stage2` (S3) | as `hra`, plus `stage2_checkpoint.pt` |
| `prefix` (every LLaVA prefix design) | `lora/`, `prefix_modules.pt` (`prefix_graph` state dict), `variant.json`, `README.md` |
| `prefix_internvl` | `lora/`, `prefix_modules.pt`, `FINAL.json` (backbone, seed, steps, `final_gate_tanh`, `closed`), `variant.json`, `README.md` |
| `prefix_qwen2vl` | `lora/`, `prefix_modules.pt`, `FINAL.json`, `variant.json`, `README.md` |

A small training telemetry file (`telemetry.jsonl`) is included where one existed and was under
5 MB.

## Downloading

`hra.hub.resolve(name_or_path)` returns a local checkpoint directory. Resolution order:

1. an existing directory path passed instead of a name;
2. the directory mapped to the name in the JSON file named by `HRA_LOCAL_CHECKPOINTS`;
3. `HRA_CKPT_DIR/<name>` if already downloaded (default `checkpoints/<name>`);
4. `huggingface_hub.snapshot_download` of the repository into `HRA_CKPT_DIR/<name>`.

Every evaluator and knock-out script calls `resolve`, so passing a registry name to
`--checkpoint` or `--ckpt` downloads on first use. To download explicitly:

```bash
python -c "from hra import hub; hub.download_checkpoint('raw_gated')"
make download NAME=raw_gated            # same thing
python -c "from hra import hub; [hub.download_checkpoint(n) for n in hub.names()]"   # everything
```

`HRA_LOCAL_CHECKPOINTS` is for people who retrain: a JSON file mapping registry names to local
directories, either `{"raw_gated": "/path/to/runs/.../checkpoints/best"}` or
`{"raw_gated": {"src": "/path/..."}}` (the format of `scripts/release/manifest.example.json`).
A mapped directory is used in place of the Hub copy whenever it looks like a checkpoint (contains
`variant.json`, `prefix_modules.pt`, `hra_modules.pt` or `lora/`).

## Loading

`hra/checkpoints.py` is the single entry point used by every evaluator:

```python
from hra.checkpoints import load_for_eval, load_prefix, load_hra, load_lora_only

# Uniform: model in {"auto", "vanilla", "lora_only", "hra", "prefix"}; "auto" reads the checkpoint.
# Returns (llava_model, processor, wrapper); wrapper is None when the pathway is absent.
vlm, processor, wrapper = load_for_eval("auto", "raw_gated", "cuda")

wrapper, processor = load_prefix("rms_ungated", "cuda")        # FixedPrefixWrapper, pathway on
wrapper, processor = load_hra("small_alpha", "cuda")           # HRAWrapper, attention bias on
vlm, processor = load_lora_only("rms_ungated", "cuda")         # the checkpoint's LoRA, pathway absent
```

- `load_prefix` attaches `lora/`, rebuilds the prefix graph with
  `hra.train_prefix_fixed.build_prefix_graph_for_eval(variant)` and loads `prefix_modules.pt`
  with `strict=True`. The variant comes from `variant.json`, else from the registry (for registry
  names), else from an explicit `variant=` argument; otherwise it raises.
- `load_hra` calls `hra.model.HRAWrapper.load_for_eval(dir, device)`, which attaches `lora/` and
  loads `hra_modules.pt` into an `HRAGraph`. For `stage2` this loads the stage-1 graph; the stage-2
  beta and gates in `stage2_checkpoint.pt` are applied by the knock-out code
  (`hra/knockout/llava_gate_lib.py`).
- `load_lora_only` is the "pathway absent" arm of every table: LoRA weights, no prefix slots and no
  bias.
- Ported backbones: `scripts/knockout/internvl_knockout.py` and `scripts/knockout/j4_internvl_eps.py`
  load the InternVL3 checkpoints in the backbone environment (a `FixedPrefixGraph` with
  `no_rms=True`, gate on, prefix projector width 3584, `lora/` attached to the language model).
  `hra/backbones/qwen2vl.py` (`load_gated_qwen`) loads the Qwen2-VL-7B checkpoints the same way
  (prefix slots reserved after the image span); `scripts/knockout/qwen2vl_knockout.py` runs the
  forced-gate knock-out on GQA-rel. The 15-run closure table and the forced-gate sweep the paper
  reports are shipped under `results/external/gate-objective/`.

Inference-mode conventions (Sec. 3.1 of the paper): *on* = prefix at the learned gate, or scale 1
for ungated variants, or bias at its learned scale; *zeroed* = prefix multiplied by 0 with the
slots kept, or bias times 0; *absent* = LoRA-only. `hra.checkpoints.set_prefix_scale(wrapper, s)`
multiplies a loaded prefix pathway's tokens by `s` (`s = 0` is the zeroed condition; the knock-out
code has its own forced-gate sweep). Every evaluator uses 6x6 grid proposals when the image has no
scene graph (`hra/boxes.py`); [PROTOCOL.md](PROTOCOL.md) says which mode each shipped number is in.

## variant.json

The prefix module's `no_rms` and `ungated` switches are constructor flags that are not part of the
saved state dict, so a checkpoint directory must say which variant it is. Rebuilding an ungated
checkpoint as a gated one multiplies its prefix by `tanh(0) = 0` at inference, silently. The old
path-substring inference was removed for that reason; `hra.checkpoints.resolve_variant` reads
`variant.json` and refuses to guess.

Fields written at upload time (`schema: 1`): `name`, `paper_label`, `kind`, `variant`, `backbone`
(base-model id), `notes`, and the learned gate read from the weights on CPU: `gate_raw` and
`gate_tanh` for prefix kinds, `beta` and `softplus_beta` (plus `stage2_beta` for S3) for the
attention-bias kinds.

Fields written by the trainers for your own runs: `hra/train_prefix_fixed.py` writes `kind:
prefix`, `variant`, `graph_class`, `config` (the `VARIANT_CONFIGS` entry), `k`, `seed`,
`run_tag`, `epoch`, `epochs`, `backbone`, `trainer`; `hra/train.py` writes `kind: hra`,
`ablation`, `epoch`, `seed`, `backbone`, `trainer`; `hra/train_stage2.py` writes `kind:
hra_stage2`, `stage1`, `seed`, `epochs`, `trainer`; `hra/backbones/internvl.py` writes
`kind: prefix_internvl`, `variant: raw_gated`, `backbone`, `seed`, `steps`, `trainer`.

For the attention-bias kinds `variant` is null (the graph class is fixed) and the file is
informational. If a directory has no `variant.json`, pass `--variant <VARIANT_CONFIGS key>` to
the evaluator (or `variant=` to the loader); a mismatch between the file and the argument raises.

## Maintainer tool

`scripts/release/upload_checkpoints.py` stages each directory named in a private manifest,
writes `variant.json` and the model card, and uploads to the Hub (idempotent by sha256 of the
weights file). It is not needed to use the checkpoints.
