#!/usr/bin/env bash
# blend_sweep.sh — trace residual vs faithfulness along blur->white_noise.
set -u
OUT_LOG="blend_sweep.log"
ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
IMG_GLOB="benchmark_50/*.JPEG"
: > "$OUT_LOG"

shopt -s nullglob; images=( $IMG_GLOB ); shopt -u nullglob
for img in "${images[@]}"; do
  base=$(basename "$img")
  for a in "${ALPHAS[@]}"; do
    echo "===== RUN_BEGIN image=$base fill=blend alpha=$a =====" >> "$OUT_LOG"
    python LIMEScore.py --input "$img" --n-samples 2500 \
      --default-mask blend --blend-alpha "$a" --mask-prob 0.5 \
      --score residual \
      >> "$OUT_LOG" 2>&1
    echo "===== RUN_END image=$base fill=blend alpha=$a rc=$? =====" >> "$OUT_LOG"
  done
done