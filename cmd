# blend_sweep:
python sweep.py --img-glob 'benchmark_50/*.JPEG' --fills blend \
  --mask-probs 0.5 --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --out-csv blend_sweep.csv

# maskprob_sweep (the kappa(Z,W) conditioning test):
python sweep.py --img-glob 'benchmark_50/*.JPEG' --fills blur white_noise \
  --mask-probs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --out-csv maskprob_sweep.csv

# run_benchmark:
python sweep.py --img-glob 'benchmark_50/*.JPEG' \
  --fills blur black white corner_mean white_noise \
  --mask-probs 0.5 --out-csv benchmark_50_all.csv

# maskprob_sweep.sh and run_benchmark.sh, the python line:
python LIMEScore.py --input "$img" --n-samples 2500 \
  --default-mask "$f" --mask-prob "$mp" --score residual \
  --calib-cache calib_cache.npz \
  >> "$OUT_LOG" 2>&1