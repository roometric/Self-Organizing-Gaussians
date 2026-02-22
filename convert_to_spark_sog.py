"""
Convert SOGS compressed output to Spark's SOGS ZIP (.sog) format.

Reads the compressed attribute images from the SOGS pipeline and re-packages
them into a ZIP file with meta.json + WebP images that Spark can load natively.

Usage:
    source .venv_sogs_cpu/bin/activate
    python convert_to_spark_sog.py
"""
import os
import io
import sys
import json
import zipfile
import numpy as np
import pandas as pd
import yaml
from PIL import Image

# ─── Configuration ───────────────────────────────────────────────────
COMPRESSED_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/standalone_compression/exr_jxl_quant_5_norm"
OUTPUT_SOG = sys.argv[2] if len(sys.argv) > 2 else "results/compressed.sog"


# ─── Codec Decoders (reuse SOGS codecs) ─────────────────────────────
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


def decompress_attr(attr_config, compressed_dir, compr_info):
    """Decompress a single attribute back to numpy array."""
    attr_name = attr_config['name']
    attr_method = attr_config['method']
    codec = codecs[attr_method]()
    compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])

    if attr_config.get('normalize', False):
        min_val = float(compr_info.loc[attr_name, "min"])
        max_val = float(compr_info.loc[attr_name, "max"])
        return codec.decode_with_normalization(compressed_file, min_val, max_val)
    else:
        return codec.decode(compressed_file)


def inverse_log_transform(transformed_coords):
    """Inverse of log1p transform: sign(x) * expm1(|x|)"""
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    return original_coords


# ─── Spark Encoding Functions ────────────────────────────────────────

def encode_positions(xyz, grid_sidelen):
    """
    Encode positions for Spark SOGS V1 format.
    Spark decodes: value = min + (max - min) * ((low + high*256) / 65535)
    Then applies: sign(x) * (exp(|x|) - 1)

    So we store: log1p(|x|) * sign(x) → normalize to [0,1] → quantize to 16-bit → split into low/high.
    """
    # Apply log1p transform (Spark will undo this with exp(|x|)-1)
    log_xyz = np.sign(xyz) * np.log1p(np.abs(xyz))

    mins = log_xyz.min(axis=0).tolist()  # [min_x, min_y, min_z]
    maxs = log_xyz.max(axis=0).tolist()  # [max_x, max_y, max_z]

    # Normalize to [0, 1]
    ranges = np.array(maxs) - np.array(mins)
    ranges[ranges == 0] = 1.0
    normalized = (log_xyz - np.array(mins)) / ranges

    # Quantize to 16-bit
    quant_16 = np.clip(np.round(normalized * 65535), 0, 65535).astype(np.uint16)

    low_bytes = (quant_16 & 0xFF).astype(np.uint8)
    high_bytes = ((quant_16 >> 8) & 0xFF).astype(np.uint8)

    N = xyz.shape[0]

    # Reshape to grid image (sidelen x sidelen x 4) RGBA
    low_rgba = np.zeros((N, 4), dtype=np.uint8)
    low_rgba[:, :3] = low_bytes
    low_rgba[:, 3] = 255
    low_img = low_rgba.reshape(grid_sidelen, grid_sidelen, 4)

    high_rgba = np.zeros((N, 4), dtype=np.uint8)
    high_rgba[:, :3] = high_bytes
    high_rgba[:, 3] = 255
    high_img = high_rgba.reshape(grid_sidelen, grid_sidelen, 4)

    return low_img, high_img, mins, maxs


def encode_scales(scaling, grid_sidelen):
    """
    Encode log-space scales for Spark SOGS V1.
    Spark decodes: lookup[i] = exp(min + (max - min) * (i / 255))
    So we store the raw log-space scale values, normalized to uint8.
    """
    mins = scaling.min(axis=0).tolist()
    maxs = scaling.max(axis=0).tolist()

    ranges = np.array(maxs) - np.array(mins)
    ranges[ranges == 0] = 1.0
    normalized = (scaling - np.array(mins)) / ranges

    quant_8 = np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)

    N = scaling.shape[0]
    rgba = np.zeros((N, 4), dtype=np.uint8)
    rgba[:, :3] = quant_8
    rgba[:, 3] = 255
    img = rgba.reshape(grid_sidelen, grid_sidelen, 4)

    return img, mins, maxs


def encode_quaternions(quats_wxyz, grid_sidelen):
    """
    Encode quaternions in Spark's packed octahedral format.

    Spark decodes (pcsogs.ts lines 113-133):
      lookup[i] = (i/255 - 0.5) * sqrt(2)   -- maps [0,255] to [-sqrt(2)/2, sqrt(2)/2]
      r0 = lookup[byte0], r1 = lookup[byte1], r2 = lookup[byte2]
      rr = sqrt(1 - r0^2 - r1^2 - r2^2)     -- reconstruct 4th component
      rOrder = byte3 - 252                     -- which component was largest (0=W, 1=X, 2=Y, 3=Z)
      rOrder=0: quatX=r0, quatY=r1, quatZ=r2, quatW=rr
      rOrder=1: quatX=rr, quatY=r1, quatZ=r2, quatW=r0
      rOrder=2: quatX=r1, quatY=rr, quatZ=r2, quatW=r0
      rOrder=3: quatX=r1, quatY=r2, quatZ=rr, quatW=r0
    """
    SQRT2_HALF = np.sqrt(2.0) / 2.0
    N = quats_wxyz.shape[0]
    result = np.zeros((N, 4), dtype=np.uint8)

    w = quats_wxyz[:, 0]
    x = quats_wxyz[:, 1]
    y = quats_wxyz[:, 2]
    z = quats_wxyz[:, 3]

    # Normalize quaternions
    norms = np.sqrt(w*w + x*x + y*y + z*z)
    norms[norms == 0] = 1.0
    w, x, y, z = w/norms, x/norms, y/norms, z/norms

    abs_q = np.abs(np.stack([w, x, y, z], axis=1))  # (N, 4) order: W=0, X=1, Y=2, Z=3
    largest = np.argmax(abs_q, axis=1)

    # Ensure largest component is positive
    largest_vals = np.choose(largest, [w, x, y, z])
    flip = largest_vals < 0
    w = np.where(flip, -w, w)
    x = np.where(flip, -x, x)
    y = np.where(flip, -y, y)
    z = np.where(flip, -z, z)

    # For each rOrder, determine which components go into r0, r1, r2
    # Reverse of Spark's decoding:
    # rOrder=0 (W largest): r0=X, r1=Y, r2=Z  (rr=W)
    # rOrder=1 (X largest): r0=W, r1=Y, r2=Z  (rr=X)
    # rOrder=2 (Y largest): r0=W, r1=X, r2=Z  (rr=Y)
    # rOrder=3 (Z largest): r0=W, r1=X, r2=Y  (rr=Z)
    #
    # Verifying with Spark's decode:
    # rOrder=0: quatX=r0=X ✓, quatY=r1=Y ✓, quatZ=r2=Z ✓, quatW=rr=W ✓
    # rOrder=1: quatX=rr=X ✓, quatY=r1=Y ✓, quatZ=r2=Z ✓, quatW=r0=W ✓
    # rOrder=2: quatX=r1=X ✓, quatY=rr=Y ✓, quatZ=r2=Z ✓, quatW=r0=W ✓
    # rOrder=3: quatX=r1=X ✓, quatY=r2=Y ✓, quatZ=rr=Z ✓, quatW=r0=W ✓

    r0 = np.where(largest == 0, x, w)
    r1 = np.where(largest <= 1, y, x)
    r2 = np.where(largest <= 2, z, y)

    # Map from [-sqrt(2)/2, sqrt(2)/2] to [0, 255]
    # Inverse of: lookup[i] = (i/255 - 0.5) * sqrt(2)
    # So: i = (value / sqrt(2) + 0.5) * 255
    result[:, 0] = np.clip(np.round((r0 / np.sqrt(2.0) + 0.5) * 255), 0, 255).astype(np.uint8)
    result[:, 1] = np.clip(np.round((r1 / np.sqrt(2.0) + 0.5) * 255), 0, 255).astype(np.uint8)
    result[:, 2] = np.clip(np.round((r2 / np.sqrt(2.0) + 0.5) * 255), 0, 255).astype(np.uint8)
    result[:, 3] = (252 + largest).astype(np.uint8)

    img = result.reshape(grid_sidelen, grid_sidelen, 4)
    return img


def encode_sh0_opacity(features_dc, opacity, grid_sidelen):
    """
    Encode DC color (SH band 0) + opacity into a single RGBA image.

    features_dc: (N, 1, 3) — raw SH DC coefficients
    opacity:     (N, 1)    — raw logit-space opacity

    Spark decodes (V1):
      color = SH_C0 * lookup[byte] + 0.5    where lookup maps [0..255] to [min..max]
      alpha = sigmoid(lookup[byte])
    So we store the raw values, normalized to [0, 255].
    """
    dc = features_dc.reshape(-1, 3)  # (N, 3)
    op = opacity.reshape(-1)  # (N,)

    # Combine into 4 channels: R_dc, G_dc, B_dc, opacity
    combined = np.stack([dc[:, 0], dc[:, 1], dc[:, 2], op], axis=1)  # (N, 4)

    mins = combined.min(axis=0).tolist()
    maxs = combined.max(axis=0).tolist()

    ranges = np.array(maxs) - np.array(mins)
    ranges[ranges == 0] = 1.0
    normalized = (combined - np.array(mins)) / ranges

    quant_8 = np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)
    img = quant_8.reshape(grid_sidelen, grid_sidelen, 4)

    return img, mins, maxs


def save_webp_lossless(img_array, buf):
    """Save RGBA numpy array as lossless WebP to a BytesIO buffer."""
    img = Image.fromarray(img_array, 'RGBA')
    img.save(buf, format='WebP', lossless=True)


def main():
    print(f"Input dir: {COMPRESSED_DIR}")
    print(f"Output:    {OUTPUT_SOG}")

    # Load compression metadata
    compr_info = pd.read_csv(os.path.join(COMPRESSED_DIR, "compression_info.csv"), index_col=0)
    with open(os.path.join(COMPRESSED_DIR, "compression_config.yml"), 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nCompression config: {config['name']}")
    print(f"SH degree: {config['max_sh_degree']}")

    # Decompress all attributes back to numpy
    print("\nDecompressing attributes...")
    attrs = {}
    for attr_cfg in config['attributes']:
        name = attr_cfg['name']
        attrs[name] = decompress_attr(attr_cfg, COMPRESSED_DIR, compr_info)
        print(f"  {name}: {attrs[name].shape}")

    # Determine grid dimensions
    # The compressed images are (sidelen, sidelen, channels)
    grid_sidelen = attrs['_xyz'].shape[0]
    N = grid_sidelen * grid_sidelen
    print(f"\nGrid: {grid_sidelen}x{grid_sidelen} = {N} gaussians")

    # Flatten grid images to (N, channels) for encoding
    xyz = attrs['_xyz'].reshape(N, 3)
    scaling = attrs['_scaling'].reshape(N, 3)
    rotation = attrs['_rotation'].reshape(N, 4)
    features_dc = attrs['_features_dc'].reshape(N, 1, 3)
    opacity = attrs['_opacity'].reshape(N, 1)

    # Check if log_activation was disabled (raw positions)
    disable_log = config.get('disable_xyz_log_activation', True)

    # Also check if contraction was applied during compression
    xyz_cfg = next(a for a in config['attributes'] if a['name'] == '_xyz')
    was_contracted = xyz_cfg.get('contract', False)

    if was_contracted:
        # Undo the log contraction that was applied during compression
        # (the codec already decompressed, but contract was applied before encoding)
        xyz = inverse_log_transform(xyz)

    # Encode for Spark
    print("\nEncoding for Spark format...")

    # 1. Positions
    print("  Encoding positions...")
    means_low, means_high, means_mins, means_maxs = encode_positions(xyz, grid_sidelen)

    # 2. Scales (already in log space from 3DGS)
    print("  Encoding scales...")
    scales_img, scales_mins, scales_maxs = encode_scales(scaling, grid_sidelen)

    # 3. Quaternions (WXYZ from 3DGS)
    print("  Encoding quaternions...")
    quats_img = encode_quaternions(rotation, grid_sidelen)

    # 4. SH0 + Opacity
    print("  Encoding SH0 + opacity...")
    sh0_img, sh0_mins, sh0_maxs = encode_sh0_opacity(features_dc, opacity, grid_sidelen)

    # Build meta.json (V1 format)
    meta = {
        "means": {
            "shape": [N, 3],
            "dtype": "uint8",
            "mins": means_mins,
            "maxs": means_maxs,
            "files": ["means_low.webp", "means_high.webp"]
        },
        "scales": {
            "shape": [N, 3],
            "dtype": "uint8",
            "mins": scales_mins,
            "maxs": scales_maxs,
            "files": ["scales.webp"]
        },
        "quats": {
            "shape": [N, 4],
            "dtype": "uint8",
            "encoding": "quaternion_packed",
            "files": ["quats.webp"]
        },
        "sh0": {
            "shape": [N, 4],
            "dtype": "uint8",
            "mins": sh0_mins,
            "maxs": sh0_maxs,
            "files": ["sh0.webp"]
        }
    }

    # Package into ZIP
    print("\nPackaging .sog file...")
    os.makedirs(os.path.dirname(OUTPUT_SOG) or ".", exist_ok=True)

    with zipfile.ZipFile(OUTPUT_SOG, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Write meta.json
        zf.writestr("meta.json", json.dumps(meta, indent=2))

        # Write images as WebP
        for name, img_array in [
            ("means_low.webp", means_low),
            ("means_high.webp", means_high),
            ("scales.webp", scales_img),
            ("quats.webp", quats_img),
            ("sh0.webp", sh0_img),
        ]:
            buf = io.BytesIO()
            save_webp_lossless(img_array, buf)
            zf.writestr(name, buf.getvalue())
            print(f"  {name}: {len(buf.getvalue()) / 1024:.1f} KB")

    sog_size = os.path.getsize(OUTPUT_SOG)
    print(f"\nDone! Output: {OUTPUT_SOG} ({sog_size / 1e6:.1f} MB)")

    print("Done!")


if __name__ == "__main__":
    main()
