python lime.py --input img/frenchh.JPEG --n-sample 5000 --filter-modes white   --threshold-quantile 0.95 --pca-dim 64 --mask-prob 0.4
python lime.py --input img/frenchh.JPEG --n-sample 5000 --default-mask white  --mask-prob 0.4

python lime.py --input img/frenchh.JPEG --n-sample 5000 --filter-modes white   --threshold-quantile 0.95 --pca-dim 64 
python lime.py --input img/frenchh.JPEG --n-sample 5000 --filter-modes inpaint   --threshold-quantile 0.95 --pca-dim 64 