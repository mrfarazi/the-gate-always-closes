#!/usr/bin/env bash
# Multi-seed VSR with the pathway dropped (supplement Table S5) and the G2 active cell.
#   bash scripts/matched_mode_seeds.sh [OUT=runs/matched_mode] [GPUS=0,1,2,3]
set -u
cd "$(dirname "$0")/.."
OUT="${1:-runs/matched_mode}"; IFS=',' read -r -a GPUS <<< "${2:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VSR="$OUT/vsr"; mkdir -p "$VSR" "$OUT/gqa_active"
g() { echo "${GPUS[$(( $1 % ${#GPUS[@]} ))]}"; }

# --- multi-seed VSR DROP (lora_only). matched_mode.py already covers rmsPub, rmsR2 (hyperbolic) and eucGeo r1 (flat)
run_vsr_drop () { # name registry_name gpu
  CUDA_VISIBLE_DEVICES="$3" python scripts/eval_vsr.py --model lora_only \
    --checkpoint "$2" --output_dir "$VSR/$1" > "$VSR/$1.log" 2>&1 &
}
run_vsr_drop rmsR3_DROP    rms_ungated_r3   "$(g 0)"
run_vsr_drop eucGeoS2_DROP euclidean_geo_s2 "$(g 1)"
run_vsr_drop eucGeoS3_DROP euclidean_geo_s3 "$(g 2)"

# --- g2 ACTIVE cell (variant g2_no_geo: the gated FixedPrefixGraph the G2 run actually trained)
python scripts/eval_gqa_spatial.py --semantic_type rel  --output_dir "$OUT/gqa_active" \
    --models "g2_ACT:prefix:g2:g2_no_geo:$(g 3)" > "$OUT/g2_ACT_rel.log" 2>&1 &
wait
CUDA_VISIBLE_DEVICES="$(g 0)" python scripts/eval_vsr.py --model prefix --checkpoint g2 --variant g2_no_geo \
    --output_dir "$VSR/g2_ACT" > "$VSR/g2_ACT.log" 2>&1
python scripts/eval_gqa_spatial.py --semantic_type attr --output_dir "$OUT/gqa_active" \
    --models "g2_ACT:prefix:g2:g2_no_geo:$(g 3)" > "$OUT/g2_ACT_attr.log" 2>&1
echo "matched_mode_seeds done -> $OUT"
