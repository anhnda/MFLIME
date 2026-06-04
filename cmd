# single mode
python off_manifold_filter.py --input img/frenchh.JPEG --score residual_delta --fill-modes white_noise

# subset
python off_manifold_filter.py --input img/frenchh.JPEG --score residual_delta --fill-modes black white white_noise