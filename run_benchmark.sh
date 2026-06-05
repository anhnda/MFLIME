#!/usr/bin/env bash
# run_benchmark.sh — sweep all benchmark images × fills into one log.
set -u

OUT_LOG="benchmark_50_all.log"
FILLS=(blur black white corner_mean white_noise)
N_SAMPLES=2500
MASK_PROB=0.5
SCORE=residual
IMG_GLOB="benchmark_50/*.JPEG"

: > "$OUT_LOG"   # truncate / create

shopt -s nullglob
images=( $IMG_GLOB )
shopt -u nullglob

if [ ${#images[@]} -eq 0 ]; then
  echo "No images matched $IMG_GLOB" >&2
  exit 1
fi

echo "[*] $(date -Is) starting sweep: ${#images[@]} images × ${#FILLS[@]} fills" | tee -a "$OUT_LOG"

for img in "${images[@]}"; do
  base=$(basename "$img")
  for f in "${FILLS[@]}"; do
    # Machine-parseable delimiter for downstream analysis.
    echo "===== RUN_BEGIN image=$base fill=$f =====" >> "$OUT_LOG"
    python LIMEScore.py --input "$img" --n-samples "$N_SAMPLES" \
      --default-mask "$f" --mask-prob "$MASK_PROB" --score "$SCORE" \
      >> "$OUT_LOG" 2>&1
    rc=$?
    echo "===== RUN_END image=$base fill=$f rc=$rc =====" >> "$OUT_LOG"
    if [ $rc -ne 0 ]; then
      echo "[!] FAILED image=$base fill=$f rc=$rc" | tee -a "$OUT_LOG" >&2
    fi
  done
done

echo "[*] $(date -Is) sweep complete -> $OUT_LOG" | tee -a "$OUT_LOG"