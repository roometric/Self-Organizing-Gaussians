"""
Standalone compression script: Load a raw PLY, prune, sort, and compress using SOGS.

Usage:
    source .venv_sogs_cpu/bin/activate
    python standalone_compress.py
"""
import os
import sys
import time
import yaml
import numpy as np
import torch
import pandas as pd

# Import directly from scene module (avoids gaussian_renderer which needs CUDA rasterizer)
from scene.gaussian_model import GaussianModel

# Import compression utilities directly (avoid compression_exp's top-level imports)
from compression.jpeg_xl import JpegXlCodec
from compression.npz import NpzCodec
from compression.exr import EXRCodec
from compression.png import PNGCodec

codecs = {
    "jpeg-xl": JpegXlCodec,
    "npz": NpzCodec,
    "exr": EXRCodec,
    "png": PNGCodec,
}

# ─── Configuration ───────────────────────────────────────────────────
PLY_PATH = "RAW PLY FILES/point_cloud.ply"
OUTPUT_DIR = "results/standalone_compression"
COMPRESSION_CONFIG = "config/compression/umbrella_sh.yaml"
SH_DEGREE = 3
DEVICE = "cpu"  # "cpu" for Mac, "cuda" for GPU machines


def log_transform(coords):
    positive = coords > 0
    negative = coords < 0
    transformed_coords = np.zeros_like(coords)
    transformed_coords[positive] = np.log1p(coords[positive])
    transformed_coords[negative] = -np.log1p(-coords[negative])
    return transformed_coords


def get_attr_numpy(gaussians, attr_name):
    attr_tensor = gaussians.attr_as_grid_img(attr_name)
    return attr_tensor.detach().cpu().numpy()


def compress_attr(attr_config, gaussians, out_folder):
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    attr_params = attr_config.get('params', {}) or {}

    codec = codecs[attr_method]()
    attr_np = get_attr_numpy(gaussians, attr_name)

    file_name = f"{attr_name}.{codec.file_ending()}"
    out_file = os.path.join(out_folder, file_name)

    if attr_config.get('contract', False):
        attr_np = log_transform(attr_np)

    if "quantize" in attr_config:
        quantization = attr_config["quantize"]
        min_val = attr_np.min()
        max_val = attr_np.max()
        val_range = max_val - min_val
        if val_range == 0:
            val_range = 1
        attr_np_norm = (attr_np - min_val) / val_range
        qpow = 2 ** quantization
        attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
        attr_np = (attr_np_quantized * val_range + min_val).astype(np.float32)

    if attr_config.get('normalize', False):
        min_val, max_val = codec.encode_with_normalization(attr_np, attr_name, out_file, **attr_params)
        return file_name, min_val, max_val
    else:
        codec.encode(attr_np, out_file, **attr_params)
        return file_name, None, None


def run_single_compression(gaussians, experiment_out_path, experiment_config):
    compressed_min_vals = {}
    compressed_max_vals = {}
    compressed_files = {}
    total_size_bytes = 0

    for attribute in experiment_config['attributes']:
        compressed_file, min_val, max_val = compress_attr(attribute, gaussians, experiment_out_path)
        attr_name = attribute['name']
        compressed_files[attr_name] = compressed_file
        compressed_min_vals[attr_name] = min_val
        compressed_max_vals[attr_name] = max_val
        total_size_bytes += os.path.getsize(os.path.join(experiment_out_path, compressed_file))

    compr_info = pd.DataFrame(
        [compressed_min_vals, compressed_max_vals, compressed_files],
        index=["min", "max", "file"]
    ).T
    compr_info.to_csv(os.path.join(experiment_out_path, "compression_info.csv"))

    experiment_config['max_sh_degree'] = gaussians.max_sh_degree
    experiment_config['active_sh_degree'] = gaussians.active_sh_degree
    experiment_config['disable_xyz_log_activation'] = gaussians.disable_xyz_log_activation
    with open(os.path.join(experiment_out_path, "compression_config.yml"), 'w') as stream:
        yaml.dump(experiment_config, stream)

    return total_size_bytes


def main():
    print(f"Device: {DEVICE}")
    print(f"PLY: {PLY_PATH}")
    original_size = os.path.getsize(PLY_PATH)
    print(f"Original PLY size: {original_size / 1e6:.1f} MB")

    # 1. Load PLY
    print("\n[1/4] Loading PLY...")
    t0 = time.time()
    gaussians = GaussianModel(SH_DEGREE, disable_xyz_log_activation=True, device=DEVICE)
    gaussians.load_ply(PLY_PATH)
    print(f"  Loaded {gaussians.get_xyz.shape[0]} gaussians in {time.time()-t0:.1f}s")

    # Detach tensors from autograd graph (not training, just compressing)
    with torch.no_grad():
        gaussians._xyz = torch.nn.Parameter(gaussians._xyz.detach(), requires_grad=False)
        gaussians._features_dc = torch.nn.Parameter(gaussians._features_dc.detach(), requires_grad=False)
        gaussians._features_rest = torch.nn.Parameter(gaussians._features_rest.detach(), requires_grad=False)
        gaussians._opacity = torch.nn.Parameter(gaussians._opacity.detach(), requires_grad=False)
        gaussians._scaling = torch.nn.Parameter(gaussians._scaling.detach(), requires_grad=False)
        gaussians._rotation = torch.nn.Parameter(gaussians._rotation.detach(), requires_grad=False)

    # 2. Prune to square grid
    print("\n[2/4] Pruning to square grid...")
    t0 = time.time()
    gaussians.prune_to_square_shape(sort_by_opacity=True, verbose=True)
    print(f"  Now {gaussians.get_xyz.shape[0]} gaussians (grid {gaussians.grid_sidelen}x{gaussians.grid_sidelen})")
    print(f"  Pruning took {time.time()-t0:.1f}s")

    # 3. PLAS sort
    print("\n[3/4] PLAS sorting...")
    t0 = time.time()

    class SortCfg:
        normalize = True
        activated = True
        shuffle = False
        improvement_break = 1e-4
        weights = {"xyz": 1.0, "features_dc": 1.0, "scaling": 1.0, "rotation": 10.0}

    gaussians.sort_into_grid(SortCfg(), verbose=True)
    print(f"  Sorting took {time.time()-t0:.1f}s")

    # 4. Compress
    print("\n[4/4] Compressing...")
    with open(COMPRESSION_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    experiment = config['experiments'][0]  # "exr_jxl_quant_5_norm"
    exp_name = experiment['name']
    exp_out_path = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_out_path, exist_ok=True)

    t0 = time.time()
    compressed_size = run_single_compression(gaussians, exp_out_path, experiment)
    print(f"  Compression took {time.time()-t0:.1f}s")

    # Summary
    print("\n" + "="*60)
    print(f"Original PLY:    {original_size / 1e6:.1f} MB")
    print(f"Compressed:      {compressed_size / 1e6:.1f} MB")
    print(f"Ratio:           {original_size / compressed_size:.1f}x")
    print(f"Output dir:      {exp_out_path}")
    print("="*60)

    # List output files
    print("\nCompressed files:")
    for f in sorted(os.listdir(exp_out_path)):
        fpath = os.path.join(exp_out_path, f)
        if os.path.isfile(fpath):
            print(f"  {f}: {os.path.getsize(fpath) / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
