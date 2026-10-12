# Evaluation protocol notes

How the released code evaluates, labels and loads things, collected in one place so that every
number in [RESULTS.md](../RESULTS.md) can be matched to a script, an inference mode and a
checkpoint. Number-by-number provenance is in [results/NUMBERS.md](../results/NUMBERS.md); the
commands are in [REPRODUCE.md](REPRODUCE.md).

## 1. Region proposals

Training uses GQA scene-graph boxes (every `train_balanced` image has a scene graph). GQA testdev,
the COCO-based sets (SugarCrepe, POPE, captioning) and VSR have none. With an empty proposal set
the attention bias is identically zero for any beta (the pair mask is empty) and the prefix
degenerates to an image-independent constant, so neither pathway can be tested that way. Every
behavioural comparison therefore uses a fixed 6x6 grid of proposals for images without a scene
graph, passed through one wrapper for both arms. `hra.boxes.boxes_or_grid` performs the fallback,
records `box_source` (`scene_graph` or `grid`) per question and asserts that the proposal set is
never empty; `scripts/eval_gqa_parallel.py --boxes auto` (default) applies it per image and
`--boxes grid` forces the grid. On the S1 checkpoint the grid bias has mean magnitude 0.406 and
maximum 0.797 logits. Gate values and every LoRA-only number do not depend on this choice.

Two probes check the proposal paths against the per-question prediction files behind the paper's
Table 4 (`results/published/`): `scripts/probes/verify_zero_boxes.py` (S1: bias magnitude and
per-item agreement under the scene-graph path and under the grid) and
`scripts/probes/verify_s4_box_source.py` (S4: prefix image-dependence and per-item agreement).
Their output is the `tab_verify_supp` block of RESULTS.md, a protocol check that is not a table
of the paper.

`scripts/knockout/j5_prefix_scale.py --model hypervis --bench gqa` evaluates the HyperVis
checkpoint through its own scene-graph proposal path with the default `--boxes repo` (constant
prefix on testdev; 34.7 at scale 1, the raw-prefix number of Table 4) and with `--boxes grid`
(50.4, the Raw/None cell of Table 2).

## 2. Inference modes

A configuration is evaluated with the pathway **on** (bias at its learned scale; prefix at its
learned gate, or scale 1 for ungated prefixes), **zeroed** (bias x 0; prefix tokens multiplied by
zero with their slots kept) or **absent** (no bias; no prefix slots, i.e. LoRA-only). Keeping
zeroed slots costs 2-5pp of GQA-rel relative to absent slots for every prefix design (Table S2),
so the two controls are reported separately. Where the shipped files use each mode:

- **Table 4 (main results).** GQA is the pathway-absent model of every recipe on full testdev
  (`eval_gqa_parallel.py --model lora_only`; for the attention bias the scene-graph proposal path
  gives an identically zero bias, which is LoRA-only in effect; the raw-prefix row cites the
  HyperVis paper's own LoRA-only number). SugarCrepe is the pathway on with grid proposals; for
  the ungated RMS prefix that is scale 1
  (`results/published/rms_ungated_prefix_on_compositionality.json`, 84.90). The behavioural
  comparisons of these designs are Tables 2, 3 and 5.
- **Table 2 (2x2) and Table S6 (prefix scale).** RMS/None cells: the `rms_ungated` checkpoint
  with the prefix on at scale `s` (`s = 1` is the trained operating point, `s = 0` the zeroed
  slot, *absent* removes the slot), grid proposals
  (`scripts/knockout/j5_prefix_scale.py --model accv_rms_ungated`).
- **Table 3 and Tables S2-S3 (knock-outs).** *Off* is the zeroed condition (bias x 0, or prefix
  x 0 with slots kept); *absent* removes the slots; *on* is the learned bias scale or gate.
- **Table 5 (matched mode).** *Drop* = pathway absent; *on* = prefix at the learned gate or
  scale 1, grid proposals. The attention-bias *on* arm of `scripts/matched_mode.py` goes through
  the scene-graph proposal path and is LoRA-only in effect (marked in the paper); the grid
  knock-out of that design is Table 3.
- **Table S10 (generation).** The "RMS prefix, ungated" row is the zeroed condition (prefix x 0,
  slots kept): `scripts/eval_pope_prefix_on.py --checkpoint rms_ungated --scale 0` for POPE and
  `scripts/eval_captioning.py --checkpoint rms_ungated --model prefix --prefix_scale 0` for
  captioning. The prefix-dropout row (`rms_dropout_r1`) is the absent condition
  (`--model lora_only`). The last row is the same `rms_ungated` checkpoint with the prefix on at
  scale 1 with grid proposals (`eval_pope_prefix_on.py`, default scale; POPE 85.50 / 84.80 /
  83.37). The attention-bias row has its bias on with grid proposals; the soft prompt has its
  content-free prefix on.

## 3. Checkpoint variants and loaders

The prefix module's `no_rms` and `ungated` switches are constructor flags that do not appear in
the saved state dict, and `euclidean` and `euclidean_geo` share a state-dict layout, so the graph
class cannot be recovered from the weights or the checkpoint path. Trainers write `variant.json`;
`hra.checkpoints.resolve_variant` reads it, falls back to an explicit `--variant`, and refuses to
guess. `hra.train_prefix_fixed.build_prefix_graph_for_eval(variant)` is the single constructor
used for evaluation, and `prefix_modules.pt` is loaded with `strict=True`. Registered checkpoints
carry their variant in `hra/hub.py` ([CHECKPOINTS.md](CHECKPOINTS.md)).

- `g2` is registered as variant `g2_no_geo`: a gated `FixedPrefixGraph` trained with
  `alpha_max = 0`, which holds the geometric-loss ramp at its 0.001 starting value. Every G2
  number in the release was measured with that graph. `random_fixed` builds
  `RandomFixedPrefixGraph`, a fixed random prefix; no shipped result uses it.
- `synthesis` is a `SoftPromptGraph` (image-independent learned embeddings; the geometric losses
  are zero throughout), registered with `variant: soft_prompt`. It is a second run of the
  soft-prompt design at effective batch 96 on six GPUs, where `soft_prompt` used effective batch
  32 on two.
- `stage2` holds the S3 stage-1 checkpoint plus `stage2_checkpoint.pt` (beta, head and layer
  gates); the knock-out code applies the gates.
- `rms_ungated`, `rms_ungated_r2`, `rms_ungated_r3` are three training runs of the ungated RMS
  recipe (the first is the Table 4 checkpoint, *r1* in Table 5; the second is *r2*).

## 4. Seeds and batch sizes

`--seed` defaults to `-1` (unseeded) in `hra/train.py`, `hra/train_prefix_fixed.py` and
`hra/train_stage2.py`. The paper's single-run LLaVA-1.5 configurations were trained unseeded and
are reported as independent runs, not as "seed 0". The replicates (`rms_ungated_r2`,
`rms_ungated_r3`, `base_r1`, `soft_prompt_r2`, `euclidean_geo_r1/_s2/_s3`,
`lora_all_linear_r1/_s2`, `rms_dropout_r1`) used seeds 1-3; the InternVL3-8B and Qwen2-VL-7B
runs used seeds 0, 1, 2. Per-GPU batch 8 and gradient accumulation 2 are fixed in the trainers,
so the effective batch is 32 on two GPUs (most configurations) and 96 on six (S1 `hra_v1`, the
S3 stage-1 run and `synthesis`); `scripts/train_llava.sh --gpus` reproduces either. The paired
bootstraps are seeded (5,000 resamples, `numpy.random.default_rng(0)`, for the knock-outs;
10,000 resamples, `RandomState(42)`, for the matched-mode contrasts).

## 5. External dependency: the HyperVis checkpoint

The Raw/None cells (Tables 2 and S6) evaluate the HyperVis checkpoint (no prefix LayerNorm) from
the HyperVis (BMVC 2026) repository, which is not part of this release.
`hra/knockout/hypervis_adapter.py` imports it from `HYPERVIS_ROOT` (checkpoint at
`HYPERVIS_CKPT`, default `$HYPERVIS_ROOT/checkpoints/ckpt_v102_noln`). `reproduce_all.sh` skips
those jobs when the variable is unset; the shipped `results/knockouts/j5_hypervis_*.json` do not
need it.

## 6. Ported backbones

Only the S5 recipe (raw gated hyperbolic prefix: `no_rms=True`, tanh gate initialised at 0, LoRA
r16, 4,000 steps) was ported. `hra/backbones/internvl.py` is the InternVL3-8B port (prefix
projector width 3584, otherwise the LLaVA graph; batch size 1 with RoI features from the global
thumbnail tile); its knock-out (`scripts/knockout/internvl_knockout.py`) uses teacher-forced
cross-entropy on the gold answer. `hra/backbones/qwen2vl.py` is the Qwen2-VL-7B port
(`scripts/train_qwen2vl.py`: LoRA r16 on q/v, learning rate 5e-5, 4,000 optimiser steps at
effective batch 16; K=4 prefix slots are reserved after the image span because Qwen2-VL derives
its position ids from the input ids; `scripts/knockout/qwen2vl_knockout.py` for the forced-gate
knock-out on GQA-rel). The Qwen2-VL evidence of Table 1 (15 runs across four task families and
three seeds; forced-gamma sweep on GQA-rel) is shipped as `FINAL.json` files and one sweep JSON
under `results/external/gate-objective/`; the three GQA-rel checkpoints are on the Hub
(`qwen2vl_seed0..2`). The counting, attribute and presence task families of that table are
shipped as result files only and are not re-runnable from this repository. None of the
attention-bias or RMS-prefix designs were run on the other backbones.

## 7. Result-file conventions

Recorded command lines and checkpoint paths inside the shipped JSONs use repository-relative
paths, registry names, `hf://` ids and `$VAR` placeholders (`scripts/release/scrub_result_paths.py
--check` verifies this). Per-question prediction files behind the knock-outs and the matched-mode
tables are shipped in full; `make tables` regenerates every table, `results/NUMBERS.md` and
`RESULTS.md` from them.
