"""Create the camera-specific image inserts for the BGDR overview figure.

Run on lme65, where the measured camera sinogram and the full 768^3
reconstructions are available.  The inserts are illustrative only; quantitative
camera comparisons remain in the dedicated results figure and table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


CAMERA_REPO = Path("/home/schneider/TrajectoryOptimization/RealWorldExample_ConOpt")
BGDR_ROOT = Path("/home/schneider/BGNR_Rekonstruktions")
OUT = BGDR_ROOT / "results" / "camera_overview_crops"
sys.path.insert(0, str(CAMERA_REPO))

import regularization_sweep as sweep  # noqa: E402


SART = BGDR_ROOT / "reconstructions" / "camera_bundle_k0400_sart.npy"
BGDR = (
    BGDR_ROOT
    / "reconstructions"
    / "camera_bundle_k0400_bgnr_hull_mseonly.npy"
)


def robust_normalize(array: np.ndarray, upper: float = 99.5) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = max(0.0, float(np.percentile(finite, 0.5)))
    hi = float(np.percentile(finite, upper))
    if hi <= lo:
        hi = float(finite.max())
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def square_crop(array: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 24) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    h, w = array.shape
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(h - side, cy - side // 2))
    x0 = max(0, min(w - side, cx - side // 2))
    return array[y0:y0 + side, x0:x0 + side]


def bbox_from_signal(array: np.ndarray) -> tuple[int, int, int, int]:
    normalized = robust_normalize(array)
    mask = normalized > 0.08
    yy, xx = np.nonzero(mask)
    if len(yy) == 0:
        return 0, array.shape[0], 0, array.shape[1]
    return int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1


def save_gray(array: np.ndarray, path: Path, bbox: tuple[int, int, int, int],
              size: int = 600, binary: bool = False) -> None:
    cropped = square_crop(array, bbox)
    if binary:
        pixels = np.where(cropped > 0, 225, 0).astype(np.uint8)
    else:
        pixels = np.round(255.0 * robust_normalize(cropped)).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").resize((size, size), Image.Resampling.LANCZOS)
    image.save(path, optimize=True)


def projection_montage(sinogram: np.ndarray, path: Path) -> None:
    indices = np.linspace(0, len(sinogram) - 1, 5, dtype=int)
    tile_size = 230
    canvas = Image.new("RGB", (360, 520), "white")
    for layer, index in enumerate(indices):
        pixels = np.round(255.0 * robust_normalize(sinogram[index], 99.7)).astype(np.uint8)
        tile = Image.fromarray(pixels, mode="L").resize(
            (tile_size, tile_size), Image.Resampling.LANCZOS
        ).convert("RGB")
        x = 18 + 25 * layer
        y = 250 - 48 * layer
        canvas.paste(tile, (x, y))
    canvas.save(path, optimize=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sart = np.load(SART, mmap_mode="r")
    bgdr = np.load(BGDR, mmap_mode="r")
    mid = sart.shape[0] // 2
    initial = np.asarray(sart[mid], dtype=np.float32)
    reconstruction = np.asarray(bgdr[mid], dtype=np.float32)
    support = reconstruction > 1e-8
    estimated = np.where(support, initial, 0.0)
    bbox = bbox_from_signal(initial)

    save_gray(initial, OUT / "backprojected.png", bbox)
    save_gray(support.astype(np.float32), OUT / "binary_top.png", bbox, binary=True)
    save_gray(support.astype(np.float32), OUT / "binary_small.png", bbox, binary=True)
    save_gray(estimated, OUT / "estimated.png", bbox)

    # A downsampled maximum-intensity projection conveys the reconstructed
    # three-dimensional camera assembly without loading the 1.8 GB array into
    # memory at full resolution.
    coarse = np.asarray(bgdr[::4, ::4, ::4], dtype=np.float32)
    mip = coarse.max(axis=0)
    mip_bbox = bbox_from_signal(mip)
    save_gray(mip, OUT / "volume3d.png", mip_bbox)

    _, _, _, _, sinogram, _, _ = sweep.load_arm("bundle", 400, sart.shape)
    projection_montage(np.asarray(sinogram, dtype=np.float32), OUT / "sinogram.png")
    projection_montage(np.asarray(sinogram, dtype=np.float32), OUT / "sino_right.png")
    print(f"wrote camera overview crops to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
