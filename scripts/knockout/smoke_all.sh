#!/bin/bash
# Run every knock-out script on 16 items (a few minutes each) to check the installation.
#   usage: bash scripts/knockout/smoke_all.sh [GPU=0] [OUT=runs/knockouts/smoke]
#   env: SMOKE_N (16), SMOKE_CHUNK (2), BACKBONE_PYTHON (python of requirements-backbones.txt), HYPERVIS_ROOT
set -uo pipefail
GPU="${1:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${2:-$ROOT/runs/knockouts/smoke}"
S="$ROOT/scripts/knockout"
PY="${PYTHON:-python}"; GPY="${BACKBONE_PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
N="${SMOKE_N:-16}"; CHUNK="${SMOKE_CHUNK:-2}"
mkdir -p "$OUT"

run_smoke() {   # name python script args...
  local name="$1"; local py="$2"; shift 2
  local out="$OUT/${name}.json"; local log="$OUT/${name}.log"
  if [ -f "$out" ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date '+%T')] $name"
  if $py "$@" --out "$out" > "$log" 2>&1 && [ -f "$out" ]; then
    echo "[$(date '+%T')] $name OK: $(grep -o 'DONE.*' "$log" | tail -1 | cut -c1-200)"
  else
    echo "[$(date '+%T')] $name FAILED (see $log)"
  fi
}

run_smoke j7_raw_gated_gqa_rel   $PY "$S/j7_llava_knockout.py" --ckpt raw_gated --bench gqa_rel --n 1000 --smoke $N --chunk $CHUNK
run_smoke j7_hra_v1_sugarcrepe   $PY "$S/j7_llava_knockout.py" --ckpt hra_v1 --bench sugarcrepe --n 1000 --smoke $N --chunk $CHUNK
run_smoke j7_stage2_gqa_rel      $PY "$S/j7_llava_knockout.py" --ckpt stage2 --bench gqa_rel --n 1000 --smoke $N --chunk $CHUNK
run_smoke j5_accv_rms_ungated_gqa $PY "$S/j5_prefix_scale.py" --model accv_rms_ungated --bench gqa --n 2000 --smoke $N --chunk $CHUNK
run_smoke j4_base_r1_sugarcrepe_eps_native $PY "$S/j4_llava_eps_sweep.py" --ckpt base_r1 --bench sugarcrepe --n 1000 --eps native --smoke $N --chunk $CHUNK
if [ -n "${HYPERVIS_ROOT:-}" ]; then
  run_smoke j5_hypervis_gqa_gridboxes $PY "$S/j5_prefix_scale.py" --model hypervis --bench gqa --n 2000 --boxes grid --smoke $N --chunk $CHUNK
fi
if [ "${SKIP_INTERNVL:-0}" != "1" ]; then
  run_smoke j4_internvl_seed0_eps_native $GPY "$S/j4_internvl_eps.py" --ckpt internvl3_seed0 --n 300 --eps native --smoke 4 --chunk 2
fi
echo "[$(date '+%T')] smoke chain finished -> $OUT"
