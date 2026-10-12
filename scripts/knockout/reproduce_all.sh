#!/bin/bash
# Sequential, idempotent launcher for every knock-out / sweep job behind the paper's tables
# (J5 prefix-scale sweeps, J4 RMSNorm-eps sweeps, J7 knock-outs, InternVL3 forced-gamma).
#   usage: bash scripts/knockout/reproduce_all.sh [GPU=0] [OUT=runs/knockouts]
#   env:   CHUNK (values per LM call, default 16; use 4 on a shared GPU), N_J4 (1000), N_J5 (2000),
#          N_J7 (1000), SKIP_INTERNVL=1, BACKBONE_PYTHON (python of requirements-backbones.txt,
#          default: python), HYPERVIS_ROOT (needed only for the hypervis jobs; skipped if unset).
# Each job writes <OUT>/<job>.json and <OUT>/logs/<job>.log, is skipped if the JSON exists, and a
# failure is appended to <OUT>/FAILED.txt. The shipped reference files are results/knockouts/.
set -uo pipefail
GPU="${1:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${2:-$ROOT/runs/knockouts}"
S="$ROOT/scripts/knockout"
PY="${PYTHON:-python}"
GPY="${BACKBONE_PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"
CHUNK="${CHUNK:-16}"
N_J4="${N_J4:-1000}"; N_J5="${N_J5:-2000}"; N_J7="${N_J7:-1000}"
FAILED="$OUT/FAILED.txt"
MASTER="$OUT/logs/reproduce_all.log"
J4_GAMMAS="0,1e-4,3e-4,1e-3,2e-3,3e-3,5e-3,1e-2,3e-2,0.1,0.3,1,-1e-3,-1e-2,-1,learned"
J5_SCALES="1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0,-1"
J7_GAMMAS="0,learned,-learned,0.001,0.003,0.01,0.03,0.1,0.3,1,-1"
EPS_LIST="native 1e-7 1e-6 1e-4"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

run_job() {   # job_name cmd...
  local job="$1"; shift
  local out="$OUT/${job}.json"
  local logf="$OUT/logs/${job}.log"
  if [ -f "$out" ]; then log "skip  $job (exists)"; return 0; fi
  log "START $job"
  local t0=$SECONDS
  "$@" --out "$out" >> "$logf" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out" ]; then
    echo "[$(date '+%F %T')] $job rc=$rc log=$logf" >> "$FAILED"
    log "FAIL  $job rc=$rc ($((SECONDS - t0))s) -> $FAILED"
  else
    log "DONE  $job ($((SECONDS - t0))s)"
  fi
  return 0
}

j4() {  # ckpt bench eps
  run_job "j4_$1_$2_eps_$3" $PY "$S/j4_llava_eps_sweep.py" --ckpt "$1" --bench "$2" --n "$N_J4" \
          --eps "$3" --gammas "$J4_GAMMAS" --chunk "$CHUNK"
}
j5() {  # model bench [extra...]
  local m="$1" b="$2"; shift 2
  local tag=""; [ $# -gt 0 ] && tag="_gridboxes"
  if [ "$m" = "hypervis" ] && [ -z "${HYPERVIS_ROOT:-}" ]; then log "skip  j5_${m}_${b}${tag} (HYPERVIS_ROOT unset)"; return 0; fi
  run_job "j5_${m}_${b}${tag}" $PY "$S/j5_prefix_scale.py" --model "$m" --bench "$b" --n "$N_J5" \
          --scales "$J5_SCALES" --chunk "$CHUNK" "$@"
}
j7() {  # ckpt bench [n]
  local n="${3:-$N_J7}"
  run_job "j7_$1_$2" $PY "$S/j7_llava_knockout.py" --ckpt "$1" --bench "$2" --n "$n" \
          --gammas "$J7_GAMMAS" --bias_mults "0,0.1,0.3,1,3" --chunk "$CHUNK"
}

log "===== reproduce_all.sh start (GPU $GPU, CHUNK $CHUNK, OUT $OUT) ====="
# 1. J5 prefix-scale sweeps (HyperVis needs HYPERVIS_ROOT; the repo-box run reproduces the empty-proposal protocol)
j5 hypervis gqa
j5 hypervis gqa --boxes grid
j5 hypervis sugarcrepe
j5 accv_rms_ungated gqa
j5 accv_rms_ungated sugarcrepe
# 2. J7 knock-outs, every design, both benches (Table 3 / S2 / S3)
for c in hra_v1 small_alpha stage2 no_dropout raw_gated base_r1 soft_prompt euclidean synthesis lora_all_linear_r1 lora_all_linear_s2; do
  j7 "$c" gqa_rel; j7 "$c" sugarcrepe
done
# 3. J7 full-testdev / 2,000-question rows of Table 3
run_job j7_hra_v1_gqa      $PY "$S/j7_llava_knockout.py" --ckpt hra_v1      --bench gqa --n 12578 --bias_mults 0,1 --chunk "$CHUNK"
run_job j7_small_alpha_gqa $PY "$S/j7_llava_knockout.py" --ckpt small_alpha --bench gqa --n 12578 --bias_mults 0,1 --chunk "$CHUNK"
run_job j7_raw_gated_gqa   $PY "$S/j7_llava_knockout.py" --ckpt raw_gated   --bench gqa --n 2000  --gammas 0,learned --chunk "$CHUNK"
run_job j7_no_dropout_gqa  $PY "$S/j7_llava_knockout.py" --ckpt no_dropout  --bench gqa --n 2000  --gammas 0,learned --chunk "$CHUNK"
# 4. J4 RMSNorm-eps sweeps (not used by the paper's tables; shipped for completeness)
for e in $EPS_LIST; do j4 base_r1 sugarcrepe "$e"; j4 base_r1 gqa_rel "$e"; done
for e in $EPS_LIST; do j4 soft_prompt sugarcrepe "$e"; j4 raw_gated sugarcrepe "$e"; done
j4 rms_ungated sugarcrepe native
j4 rms_ungated gqa_rel native
# 5. InternVL3 forced-gamma x eps (backbone env)
if [ "${SKIP_INTERNVL:-0}" != "1" ]; then
  for s in 0 1 2; do for e in 1e-8 1e-7 native 1e-5 1e-4; do
    run_job "j4_internvl_seed${s}_eps_${e}" $GPY "$S/j4_internvl_eps.py" --ckpt "internvl3_seed${s}" --n 300 --eps "$e" --chunk 4
  done; done
fi
log "===== reproduce_all.sh finished ====="
if [ -f "$FAILED" ]; then log "failures recorded in $FAILED:"; cat "$FAILED" | tee -a "$MASTER"; fi
