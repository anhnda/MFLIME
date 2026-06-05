#!/usr/bin/env bash
# maskprob_sweep.sh — test the kappa(Z,W) design-conditioning prediction:
# residual<->faithfulness agreement should WEAKEN as mask_prob -> high
# (design matrix degenerates: almost all cells off, Z ill-conditioned).
set -u
OUT_LOG="maskprob_sweep.log"
MASK_PROBS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
FILLS=(blur white_noise)
IMG_GLOB="benchmark_50/*.JPEG"
: > "$OUT_LOG"

shopt -s nullglob; images=( $IMG_GLOB ); shopt -u nullglob
if [ ${#images[@]} -eq 0 ]; then echo "no images matched $IMG_GLOB" >&2; exit 1; fi

for img in "${images[@]}"; do
  base=$(basename "$img")
  for f in "${FILLS[@]}"; do
    for mp in "${MASK_PROBS[@]}"; do
      echo "===== RUN_BEGIN image=$base fill=$f mask_prob=$mp =====" >> "$OUT_LOG"
      python LIMEScore.py --input "$img" --n-samples 2500 \
        --default-mask "$f" --mask-prob "$mp" --score residual \
        >> "$OUT_LOG" 2>&1
      echo "===== RUN_END image=$base fill=$f mask_prob=$mp rc=$? =====" >> "$OUT_LOG"
    done
  done
done
echo "[*] $(date -Is) done -> $OUT_LOG"