# Reproducing the tables

Three tiers: regenerate the tables from the shipped result files (no GPU), run one knock-out (one
GPU, about 10 minutes), or re-run every behavioural measurement (one to eight GPUs, a few days of
GPU time) and then regenerate the tables from your own files. Retraining the checkpoints is
covered by the recipe table in the [README](../README.md#retraining).

Conventions used throughout (Sec. 3.1 of the paper): 6x6 grid proposals whenever an image has no
scene graph (all evaluation sets); inference mode *on* / *zeroed* / *absent* as defined in
[CHECKPOINTS.md](CHECKPOINTS.md#loading); paired bootstrap on shared items (5,000 resamples,
`numpy.random.default_rng(0)`, for the knock-outs; 10,000 resamples, `RandomState(42)`, for the
matched-mode contrasts). Greedy decoding everywhere. GPU kernels are not bit-deterministic across
hardware, so a re-run can differ from the shipped files by a few predictions (the shipped protocol checks
under `results/knockouts/verify_zero_boxes/` were bit-exact against the Table 4 prediction files
on the same hardware).

## Tier 1: tables from the shipped results (no GPU)

```bash
make tables          # python scripts/make_tables.py
make check-tables    # regenerate into /tmp/tgac_tables and diff against tests/golden/tables
make test            # CPU unit tests, includes a golden-table comparison
```

`scripts/make_tables.py` reads the JSONs under `results/` (override with `HRA_RESULTS_DIR`) and
writes `results/tables/<name>.tex` (override with `HRA_TABLES_OUT`), `results/NUMBERS.md` (every
number with file and key path, plus cross-checks against the numbers quoted in the paper) and
`RESULTS.md` (the 12 paper tables and the protocol check in Markdown). It takes about a minute and needs only numpy and scipy.
Optional inputs (for example `results/knockouts/j7_stage2_gqa.json`) are skipped cleanly when
absent and listed as such in `NUMBERS.md`.

## Tier 2: one knock-out on one GPU

Needs the LLaVA-1.5 environment, `data/gqa/testdev_balanced_questions.json`, the testdev images,
and the `raw_gated` checkpoint (downloaded on first use; the LLaVA base model is downloaded from
the Hub unless `HRA_LLAVA_PATH` points to a local copy).

```bash
python scripts/knockout/j7_llava_knockout.py --ckpt raw_gated --bench gqa_rel --n 200 \
    --gammas 0,learned,1 --out runs/knockouts/j7_raw_gated_gqa_rel_n200.json
```

The JSON has one row per gamma with accuracy, delta versus gamma = 0 and its paired 95% CI. At
n = 200 the intervals are wide; the shipped Table 3 row used n = 1,000
(`results/knockouts/j7_raw_gated_gqa_rel.json`: 50.7 at gamma = 0, 44.0 at the learned gate,
12.2 at gamma = 1). About 10 minutes including model load.

`bash scripts/knockout/smoke_all.sh 0` runs every knock-out script on 16 items (a few minutes
each) to check an installation; `make smoke-knockout` is the same.

## Tier 3: full reproduction

Every script writes under `runs/` (`HRA_RUNS`) by default and never touches `results/`. When the
runs are complete, either copy the outputs into the same layout under `results/` or point
`HRA_RESULTS_DIR` at a directory with that layout, then `make tables`. The layout the table
generator expects is the one under `results/`: `knockouts/`, `matched_mode/{gqa_drop,gqa_active,vsr}`,
`generation/`, `probes/`, `probes/j6/`, `knockouts/verify_zero_boxes/`, `external/gate-objective/`,
`published/`.

Order of operations (one GPU unless stated):

```bash
# 1. Knock-outs and sweeps: J5 prefix-scale, J7 knock-outs (all designs, both benches, the
#    full-testdev and 2,000-question rows), J4 RMSNorm-eps sweeps, InternVL3 forced-gamma x eps.
#    Idempotent: a job is skipped if its JSON exists; failures are appended to <OUT>/FAILED.txt.
bash scripts/knockout/reproduce_all.sh 0 runs/knockouts
#    env: CHUNK (gammas per LM call, 16; use 4 on a shared GPU), N_J4 / N_J5 / N_J7 (1000 / 2000 / 1000),
#         SKIP_INTERNVL=1, BACKBONE_PYTHON=<python of the backbone env>, HYPERVIS_ROOT (see below)

# 2. Matched inference mode (Tables 5, S4, S5): GQA-rel, GQA-attr and VSR for every configuration,
#    pathway dropped and active. Internally parallel over the given GPUs.
python scripts/matched_mode.py --gpus 0,1,2,3,4,5,6,7 --out runs/matched_mode
bash scripts/matched_mode_seeds.sh runs/matched_mode 0,1,2,3      # multi-seed VSR (dropped) + G2 active cell
python scripts/analyze_matched_mode.py --dir runs/matched_mode    # paired-bootstrap contrasts (also done by make_tables)

# 3. Generation side (Table S10): POPE + COCO captioning for the four headline models, POPE for
#    rms_dropout_r1, and POPE with the RMS prefix genuinely on.
python scripts/eval_pope_captioning_all.py
python scripts/eval_pope_captioning_all.py --models rms_dropout_r1:lora_only:rms_dropout_r1 --skip_captioning
python scripts/eval_pope_prefix_on.py --checkpoint rms_ungated

# 4. Probes (Tables S8, S9, S4) and the J6 attention probe for every prefix checkpoint.
python scripts/probes/gate_gradient.py                 # rms_ungated, n=120, eps=0.01
python scripts/probes/layerwise_attention.py           # rms_ungated, soft_prompt, raw_gated; n=100
for c in base_r1 soft_prompt no_dropout euclidean synthesis raw_gated rms_ungated lora_all_linear_r1 lora_all_linear_s2; do
  python scripts/knockout/j6_llava_attention.py --ckpt $c --n 100
done
python scripts/probes/verify_zero_boxes.py 150         # S1 zero-box protocol check
python scripts/probes/verify_s4_box_source.py 100      # S4 box-source check

# 5. Tables.
HRA_RESULTS_DIR=<directory with the results/ layout> make tables
```

`reproduce_all.sh` runs 64 jobs sequentially on one GPU (5 J5, 26 J7, 18 J4, 15 InternVL3);
summing the per-job times recorded in the shipped JSONs gives roughly 14 hours. The jobs are
independent and can be spread over GPUs by running the underlying scripts directly.

### Table-by-table command map

Table numbers: paper (main) and supplement (S). Inputs are relative to `results/`. Runtimes are
wall-clock on one H100 80GB as recorded in the shipped JSONs, including model load.

| Table (file) | Script and arguments | Inputs | Approx. runtime |
|---|---|---|---|
| 3 `tab_knockout_main` | `scripts/knockout/j7_llava_knockout.py --ckpt {hra_v1,small_alpha,stage2,no_dropout,raw_gated} --bench {gqa_rel,sugarcrepe} --n 1000`; `--ckpt {hra_v1,small_alpha} --bench gqa --n 12578 --bias_mults 0,1`; `--ckpt {no_dropout,raw_gated} --bench gqa --n 2000 --gammas 0,learned` | `knockouts/j7_<ckpt>_{gqa_rel,sugarcrepe}.json`, `knockouts/j7_<ckpt>_gqa.json` | 5-19 min per checkpoint and bench at n = 1,000; 7-8 min for the 2,000-question rows; 25-28 min for full testdev |
| 1 `tab_backbones` | LLaVA rows: the J7 `gqa_rel` files above. InternVL3: `scripts/train_internvl.py --seed {0,1,2} --steps 4000 --out runs/internvl3_seed<s>` (gate; backbone env) and `scripts/knockout/internvl_knockout.py --ckpt internvl3_seed<s> --n 300 --gammas 0.0,1.0,2.0`. Qwen2-VL: `scripts/train_qwen2vl.py --seed {0,1,2}` (gate; backbone env) and `scripts/knockout/qwen2vl_knockout.py --ckpt qwen2vl_seed<s> --gammas 0,1.5,2`; the other task families are shipped only | `knockouts/j7_*_gqa_rel.json`, `probes/internvl_seed{0,1,2}.json` (the runs' `FINAL.json`), `probes/internvl_knockout{,_seed1,_seed2}.json`, `external/gate-objective/**/FINAL.json`, `external/gate-objective/gsweep/w4aud_g1b0_gqa.json` | InternVL knock-out about 6 min per seed; closure run 4,000 steps about 2 h per seed |
| 2 `tab_2x2` | `scripts/knockout/j5_prefix_scale.py --model hypervis --bench gqa --n 2000 --boxes grid`; `--model hypervis --bench sugarcrepe`; `--model accv_rms_ungated --bench {gqa,sugarcrepe} --n 2000`; plus J7 `raw_gated` and `base_r1` on `gqa_rel` and `sugarcrepe` | `knockouts/j5_hypervis_gqa_gridboxes.json`, `j5_hypervis_sugarcrepe.json`, `j5_accv_rms_ungated_{gqa,sugarcrepe}.json`, `j7_{raw_gated,base_r1}_{gqa_rel,sugarcrepe}.json` | J5: 20-31 min per job |
| 5 `tab_matched` | `scripts/matched_mode.py --gpus ... --out runs/matched_mode`; `scripts/matched_mode_seeds.sh` for the G2 active cell | `matched_mode/gqa_drop/*.json`, `gqa_active/*.json`, `vsr/*/vsr_predictions.json` | about 70 min on 8 GPUs |
| S4 `tab_matched_contrasts` | same runs; contrasts recomputed by `make_tables` (`hra/eval/matched_mode.py`) | `matched_mode/gqa_drop/*.json`, `vsr/*/vsr_predictions.json` | none beyond Table 5 |
| S5 `tab_vsr_seeds_supp` | VSR dropped for `rms_ungated`, `rms_ungated_r2`, `euclidean_geo_r1`, vanilla (`matched_mode.py`) and `rms_ungated_r3`, `euclidean_geo_s2`, `euclidean_geo_s3` (`matched_mode_seeds.sh`) | `matched_mode/vsr/{rmsPub,rmsR2,rmsR3,eucGeo,eucGeoS2,eucGeoS3}_DROP/vsr_predictions.json`, `vsr/vanilla/` | included above |
| S2 `tab_gated_prefix_supp` | `j7_llava_knockout.py --ckpt {base_r1,soft_prompt,no_dropout,euclidean,synthesis,lora_all_linear_r1,lora_all_linear_s2,raw_gated} --bench {gqa_rel,sugarcrepe} --n 1000` | `knockouts/j7_<ckpt>_{gqa_rel,sugarcrepe}.json` | 5-19 min per checkpoint and bench |
| S3 `tab_s5_sweep_supp` | `j7_llava_knockout.py --ckpt raw_gated --bench {gqa_rel,sugarcrepe} --n 1000` (default gamma list) | `knockouts/j7_raw_gated_{gqa_rel,sugarcrepe}.json` | included in Table 3 |
| S6 `tab_prefix_scale_supp` | the four J5 jobs of Table 2 plus `--model hypervis --bench gqa` with the default `--boxes repo` (reproduces the empty-proposal protocol) | `knockouts/j5_accv_rms_ungated_{gqa,sugarcrepe}.json`, `j5_hypervis_gqa_gridboxes.json`, `j5_hypervis_gqa.json`, `j5_hypervis_sugarcrepe.json` | 20-31 min per job |
| S7 `tab_gradient_supp` | `scripts/probes/gate_gradient.py` (`--checkpoint rms_ungated --n 120 --eps 0.01`) | `probes/gate_gradient.json` | minutes |
| S8 `tab_attention_supp` | (a) `scripts/probes/layerwise_attention.py --n_examples 100`; (b) `scripts/knockout/j6_llava_attention.py --ckpt <c> --n 100` for the nine prefix checkpoints listed above | `probes/layerwise_attention.json`, `probes/j6/j6_llava_<ckpt>.json` | (a) minutes; (b) about 1-1.5 min per checkpoint |
| S10 `tab_generation_supp` | `scripts/eval_pope_captioning_all.py` (vanilla; `small_alpha` bias on; `rms_ungated` prefix zeroed; `soft_prompt` prefix on); `--models rms_dropout_r1:lora_only:rms_dropout_r1 --skip_captioning` (pathway absent); prefix on: `scripts/eval_pope_prefix_on.py --checkpoint rms_ungated`. The zeroed condition alone: `scripts/eval_pope_prefix_on.py --checkpoint rms_ungated --scale 0` and `scripts/eval_captioning.py --checkpoint rms_ungated --model prefix --prefix_scale 0` | `generation/pope_captioning_summary.json`, `generation/captioning_metrics_final.json` (captioning metrics of the same runs), `generation/rms_dropout_r1_pope_captioning.json`, `generation/rms_ungated_pope_prefix_on.json` | about 40 min per model (POPE 10-13 min, captioning 25-30 min); prefix-on POPE about 18 min |
| check `tab_verify_supp` (not in the paper) | `scripts/probes/verify_zero_boxes.py 150`; `scripts/probes/verify_s4_box_source.py 100` (both compare against the shipped `results/published/*.json`) | `knockouts/verify_zero_boxes/verify_zero_boxes_n150.json`, `verify_s4_box_source_n100.json`, `published/*.json` | minutes each |

The J4 RMSNorm-epsilon sweeps (`scripts/knockout/j4_llava_eps_sweep.py`,
`scripts/knockout/j4_internvl_eps.py`; 13-19 min and about 8 min per job) are run by
`reproduce_all.sh` and shipped under `results/knockouts/j4_*.json` but are not used by any table.

### Default output locations

| Script | Default output | Table generator reads |
|---|---|---|
| `scripts/knockout/*.py` via `reproduce_all.sh` | `runs/knockouts/<job>.json` | `results/knockouts/` |
| `scripts/matched_mode.py`, `matched_mode_seeds.sh` | `runs/matched_mode/` | `results/matched_mode/` (`HRA_E1_DIR`) |
| `scripts/eval_pope_captioning_all.py`, `eval_pope_prefix_on.py` | `runs/generation/` (`RUN_DIR` overrides) | `results/generation/` |
| `scripts/probes/gate_gradient.py`, `layerwise_attention.py` | `runs/probes/*.json` | `results/probes/` |
| `scripts/knockout/j6_llava_attention.py` | `runs/probes/j6/` (`HRA_J6_DIR`) | `results/probes/j6/` |
| `scripts/probes/verify_*.py` | `runs/probes/verify_zero_boxes/` (`HRA_VERIFY_OUT`) | `results/knockouts/verify_zero_boxes/` |
| `scripts/knockout/internvl_knockout.py` | `runs/probes/internvl_knockout_<ckpt>.json` | `results/probes/internvl_knockout*.json` |

### HyperVis rows (Tables 2 and S7)

The Raw/None cells use the HyperVis checkpoint from the separate HyperVis repository
(BMVC 2026, <https://github.com/mrfarazi/hypertrip>). Set `HYPERVIS_ROOT` to that checkout
(and `HYPERVIS_CKPT` if the no-LN checkpoint is not at `$HYPERVIS_ROOT/checkpoints/ckpt_v102_noln`);
`reproduce_all.sh` skips the `hypervis` jobs when `HYPERVIS_ROOT` is unset. The shipped
`results/knockouts/j5_hypervis_*.json` do not need it.

### InternVL3 rows (Table 1)

Training and knock-outs run in the backbone environment (`requirements-backbones.txt`).
`reproduce_all.sh` and `smoke_all.sh` call it through `BACKBONE_PYTHON`; set `SKIP_INTERNVL=1` to
leave those jobs out. The InternVL knock-out uses teacher-forced cross-entropy on the gold answer
(n = 300, paired bootstrap of the CE change versus gamma = 0) rather than generation.

## Single-benchmark evaluators

All accept `--checkpoint <registry name or directory>`; `--model` selects the inference mode
(`vanilla`, `lora_only`, `hra`, `prefix`; `auto` where offered picks from the checkpoint
contents); `--variant` is needed only for a directory without `variant.json`.

```bash
# Full GQA testdev, sharded over GPUs; --boxes auto (default) falls back to the 6x6 grid per image
python scripts/eval_gqa_parallel.py --model vanilla --gpus 0,1,2,3
python scripts/eval_gqa_parallel.py --model prefix --checkpoint raw_gated --gpus 0,1
python scripts/eval_gqa_parallel.py --model hra --checkpoint hra_v1 --gpus 0,1
python scripts/eval_gqa_parallel.py --model lora_only --checkpoint rms_ungated --gpus 0,1

# GQA semantic subsets (rel: 5,308 questions; attr: 5,186), several models in parallel, one GPU each
python scripts/eval_gqa_spatial.py --semantic_type rel --output_dir runs/gqa_spatial \
    --models "vanilla:vanilla:::0;soft_ACT:prefix:soft_prompt:soft_prompt:1;smallA_ACT:hra:small_alpha::2"
#   spec = name:mtype:ckpt:variant:gpu (variant may be empty when the checkpoint has variant.json)

# SugarCrepe (7 sub-tasks) and Winoground log-likelihood scoring
python scripts/eval_compositionality.py --checkpoint raw_gated --model prefix

# POPE (random / popular / adversarial), COCO captioning (5,000 val2017 images), VSR
python scripts/eval_pope.py --checkpoint small_alpha --model hra
python scripts/eval_captioning.py --checkpoint small_alpha --model hra --num_samples 0
python scripts/eval_vsr.py --model lora_only --checkpoint rms_ungated --output_dir runs/vsr/rmsPub_DROP
```

## Training smoke tests

```bash
make smoke-train             # 20 micro-steps of `attention none` and `prefix raw_gated` on one GPU, ~2 min each
scripts/train_llava.sh prefix raw_gated --smoke --gpus 0
python scripts/train_internvl.py --seed 0 --steps 30 --smoke --out runs/internvl_smoke   # backbone env, ~10 min incl. load
```

Smoke runs write `runs/<label>_<timestamp>/` with `command.txt`, `accelerate.yaml`, `train.log`
and telemetry, and no checkpoint. Full training commands, GPU counts and approximate GPU-hours are
in the [README](../README.md#retraining).
