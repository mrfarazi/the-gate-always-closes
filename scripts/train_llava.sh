#!/usr/bin/env bash
# One launcher for every LLaVA-1.5-7B training configuration in the paper.
#
#   scripts/train_llava.sh attention <ablation>      # attention-bias designs (hra/train.py)
#   scripts/train_llava.sh prefix    <variant>       # gated / ungated prefix designs (hra/train_prefix_fixed.py)
#   scripts/train_llava.sh stage2    <stage1_ckpt>   # S3 stage 2: margin loss with LoRA frozen (hra/train_stage2.py)
#
# Options (after the two positionals):
#   --gpus 0,1        GPU ids (default 0,1: the paper's 2-GPU runs; S1/S3-stage-1/synthesis used 0-5)
#   --epochs N        default 3 (S1 and the S3 stage-1 run used 5)
#   --seed N          default -1 (unseeded, as the paper's single-run configurations; replicate runs used 1,2,3)
#   --tag TAG         run tag appended to the run directory / checkpoint name
#   --port P          accelerate main_process_port (default 29500)
#   --smoke           20 micro-steps at batch 4 on the first GPU; no checkpoint is written
#   --batch_size N    per-GPU batch (default 8; gradient accumulation 2 is fixed in the trainers)
#
# Paper configurations:
#   attention none          S1  (alpha_max 0.05, 5 epochs, 6 GPUs)      attention small_alpha   S2 (alpha_max 0.01)
#   attention detach_bias   S3 stage 1 (5 epochs, 6 GPUs) -> stage2 <that checkpoint>
#   prefix no_dropout       S4          prefix raw_gated        S5          prefix rms_ungated   RMS prefix (ungated)
#   prefix base             RMS prefix, gated                    prefix soft_prompt     soft prompt
#   prefix euclidean        Euclidean gated (no geo losses)      prefix euclidean_geo   Euclidean containment control
#   prefix g2_no_geo        G2 (see docs/PROTOCOL.md); prefix random_fixed = fixed random prefix
#   prefix lora_all_linear  LoRA on all linear layers            prefix rms_dropout     RMS prefix + 50% prefix dropout
# Outputs: runs/<label>_<timestamp>/{checkpoints/<name>/{epochN,best}/, *_telemetry.jsonl, accelerate.yaml, train.log}
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: train_llava.sh MODE TARGET [options]  (MODE = attention | prefix | stage2)}"; TARGET="${2:?usage: train_llava.sh MODE TARGET [options]}"; shift 2
GPUS="0,1"; EPOCHS=""; SEED="-1"; TAG=""; PORT="29500"; SMOKE=0; BS=""
while [ $# -gt 0 ]; do case "$1" in
  --gpus) GPUS="$2"; shift 2;; --epochs) EPOCHS="$2"; shift 2;; --seed) SEED="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;; --port) PORT="$2"; shift 2;; --smoke) SMOKE=1; shift;;
  --batch_size) BS="$2"; shift 2;; *) echo "unknown option $1"; exit 2;; esac; done
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
STAMP=$(date +%Y%m%d_%H%M%S)
LABEL="${MODE}_${TARGET//\//_}${TAG:+_$TAG}"
export RUN_DIR="${RUN_DIR:-$PWD/runs/${LABEL}_${STAMP}}"
mkdir -p "$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
if [ "$SMOKE" = 1 ]; then export SMOKE_STEPS="${SMOKE_STEPS:-20}" SMOKE_BATCH="${SMOKE_BATCH:-4}"; GPUS="${GPUS%%,*}"; NGPU=1; fi

cat > "$RUN_DIR/accelerate.yaml" <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: $([ "$NGPU" -gt 1 ] && echo MULTI_GPU || echo NO)
mixed_precision: bf16
num_processes: $NGPU
num_machines: 1
machine_rank: 0
gpu_ids: '$GPUS'
main_process_port: $PORT
same_network: true
use_cpu: false
downcast_bf16: 'no'
rdzv_backend: static
YAML

case "$MODE" in
  attention)
    CMD=(accelerate launch --config_file "$RUN_DIR/accelerate.yaml" -m hra.train --ablation "$TARGET" --phase "$LABEL" --seed "$SEED" --epochs "${EPOCHS:-3}");;
  prefix)
    CMD=(accelerate launch --config_file "$RUN_DIR/accelerate.yaml" -m hra.train_prefix_fixed --variant "$TARGET" --phase "$LABEL" --seed "$SEED" --epochs "${EPOCHS:-3}")
    [ -n "$TAG" ] && CMD+=(--run_tag "$TAG");;
  stage2)
    CMD=(python -m hra.train_stage2 --stage1_checkpoint "$TARGET" --output_dir "$RUN_DIR/stage2" --epochs "${EPOCHS:-1}" --seed "$SEED")
    export CUDA_VISIBLE_DEVICES="${GPUS%%,*}";;
  *) echo "mode must be attention | prefix | stage2"; exit 2;;
esac
[ -n "$BS" ] && CMD+=(--batch_size "$BS")
[ "$SMOKE" = 1 ] && export CUDA_VISIBLE_DEVICES="$GPUS"
echo "[train_llava] RUN_DIR=$RUN_DIR"; echo "[train_llava] ${CMD[*]}" | tee "$RUN_DIR/command.txt"
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
