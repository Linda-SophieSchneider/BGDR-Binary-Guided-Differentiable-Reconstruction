"""The Chapter 7 metric protocol, on CUDA through torch.

A faithful port of ``Research/BGNR/eval/metrics.py``, which is written against
MLX and therefore runs only on Apple silicon.  The formulas are duplicated
rather than shared, because the MLX module cannot be imported here at all; what
guarantees they agree is not the code but the check in ``validate_port.py``,
which recomputes every metric of the engine table from the volumes the Mac
produced and compares digit for digit.

Only the two blurring kernels are genuinely different: MLX has no conv3d, so the
original composes conv2d passes with transposes, while here a separable conv3d
does the same thing more directly.  Both are 'valid' (no padding), so they
consume the same 10 voxels of border.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_reference(reference: np.ndarray) -> np.ndarray:
    """Map the reference to air = 0, material = 1 -- the Chapter 7 convention."""
    ref = np.asarray(reference, dtype=np.float32)
    air = float(np.median(ref))
    peak = float(np.percentile(ref, 99.9)) - air
    if peak <= 0:
        raise ValueError("reference has no dynamic range above its median")
    return ((ref - air) / peak).astype(np.float32)


def ls_scale(volume, reference) -> float:
    """``argmin_s ||s*v - ref||^2``, the least-squares amplitude match."""
    v = np.asarray(volume, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    denominator = float((v * v).sum())
    return float((v * r).sum() / denominator) if denominator > 0 else 0.0


def affine_match(volume, reference) -> tuple[float, float]:
    x = np.asarray(volume, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    mean_x, mean_r = x.mean(), r.mean()
    var_x = float(((x - mean_x) ** 2).mean())
    if var_x < 1e-20:
        return 0.0, float(mean_r)
    gain = float(((x - mean_x) * (r - mean_r)).mean() / var_x)
    return gain, float(mean_r - gain * mean_x)


def otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
    """Deterministic dense-reference threshold for method-independent ROIs."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0 or float(x.max()) <= float(x.min()):
        raise ValueError("reference has no usable range for ROI segmentation")
    hist, edges = np.histogram(x, bins=bins, range=(float(x.min()), float(x.max())))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_left = np.cumsum(hist, dtype=np.float64)
    weight_right = x.size - weight_left
    mean_left = np.cumsum(hist * centers, dtype=np.float64) \
        / np.maximum(weight_left, 1.0)
    reverse_sum = np.cumsum((hist * centers)[::-1], dtype=np.float64)[::-1]
    mean_right = reverse_sum / np.maximum(weight_right, 1.0)
    between = weight_left * weight_right * (mean_left - mean_right) ** 2
    between[(weight_left == 0) | (weight_right == 0)] = -1.0
    return float(centers[int(np.argmax(between))])


def _morphology6(mask: np.ndarray, iterations: int, *, dilate: bool) -> np.ndarray:
    """Dependency-free 6-connected binary dilation/erosion without wraparound."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        updated = out.copy()
        for axis in range(3):
            lower = np.zeros_like(out)
            upper = np.zeros_like(out)
            low_dst = [slice(None)] * 3
            low_src = [slice(None)] * 3
            high_dst = [slice(None)] * 3
            high_src = [slice(None)] * 3
            low_dst[axis], low_src[axis] = slice(1, None), slice(None, -1)
            high_dst[axis], high_src[axis] = slice(None, -1), slice(1, None)
            lower[tuple(low_dst)] = out[tuple(low_src)]
            upper[tuple(high_dst)] = out[tuple(high_src)]
            if dilate:
                updated |= lower | upper
            else:
                updated &= lower & upper
        out = updated
    return out


def reference_rois(reference: np.ndarray, *, threshold: float | None = None,
                   dilation_voxels: int = 12,
                   boundary_width_voxels: int = 3) -> dict[str, np.ndarray]:
    """Build foreground, dilated foreground, boundary, and far-background ROIs."""
    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim != 3:
        raise ValueError(f"expected a 3-D reference, got shape {ref.shape}")
    if dilation_voxels < 0 or boundary_width_voxels < 1:
        raise ValueError("ROI dilation must be non-negative and shell width positive")
    level = otsu_threshold(ref) if threshold is None else float(threshold)
    foreground = ref > level
    if not foreground.any() or foreground.all():
        raise ValueError("ROI threshold must separate foreground and background")
    dilated = _morphology6(foreground, dilation_voxels, dilate=True) \
        if dilation_voxels else foreground.copy()
    outer = _morphology6(foreground, boundary_width_voxels, dilate=True)
    inner = _morphology6(foreground, boundary_width_voxels, dilate=False)
    return {
        "foreground": foreground,
        "foreground_dilated": np.asarray(dilated, dtype=bool),
        "boundary_shell": np.asarray(outer & ~inner, dtype=bool),
        "far_background": np.asarray(~dilated, dtype=bool),
    }


def masked_ls_scale(volume, reference, mask) -> float:
    """Least-squares amplitude match within a fixed reference-derived mask."""
    v = np.asarray(volume, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if v.shape != r.shape or m.shape != r.shape:
        raise ValueError("volume, reference, and ROI mask must have equal shapes")
    denominator = float(np.square(v[m]).sum())
    return float((v[m] * r[m]).sum() / denominator) if denominator > 0 else 0.0


def _masked_error(scaled, reference, mask) -> tuple[float, float]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan"), float("nan")
    a = np.asarray(scaled, dtype=np.float64)[m]
    r = np.asarray(reference, dtype=np.float64)[m]
    mse = float(np.mean((a - r) ** 2))
    rms = float(np.sqrt(np.mean(r ** 2)))
    return mse, float(np.sqrt(mse) / rms) if rms > 0 else float("inf")


def _foreground_bbox(mask: np.ndarray,
                     minimum_size: int = SSIM_WINDOW) -> tuple[slice, slice, slice]:
    shape = np.asarray(mask.shape, dtype=int)
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("foreground ROI is empty")
    lo = points.min(axis=0)
    hi = points.max(axis=0) + 1
    for axis in range(3):
        missing = max(0, minimum_size - int(hi[axis] - lo[axis]))
        before = missing // 2
        after = missing - before
        lo[axis] = max(0, lo[axis] - before)
        hi[axis] = min(shape[axis], hi[axis] + after)
        if hi[axis] - lo[axis] < minimum_size:
            lo[axis] = max(0, hi[axis] - minimum_size)
            hi[axis] = min(shape[axis], lo[axis] + minimum_size)
        if hi[axis] - lo[axis] < minimum_size:
            raise ValueError("reference volume is too small for 11-voxel SSIM")
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _gaussian_kernel(window: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA) -> np.ndarray:
    coords = np.arange(window, dtype=np.float64) - (window - 1) / 2.0
    kernel = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


class Gaussian3D:
    """Separable isotropic Gaussian blur with 'valid' support, on the GPU."""

    def __init__(self, window: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA,
                 device: str = DEVICE):
        k = torch.as_tensor(_gaussian_kernel(window, sigma), device=device)
        self.kd = k.view(1, 1, window, 1, 1)
        self.kh = k.view(1, 1, 1, window, 1)
        self.kw = k.view(1, 1, 1, 1, window)

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        x = volume[None, None]
        x = F.conv3d(x, self.kd)
        x = F.conv3d(x, self.kh)
        x = F.conv3d(x, self.kw)
        return x[0, 0]


def _as_tensor(array, device: str = DEVICE) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32), device=device)


def ssim3d(volume, reference, data_range: float | None = None,
           device: str = DEVICE) -> float:
    """3-D SSIM with a Gaussian window, the Chapter 7 parameterization."""
    x = _as_tensor(volume, device)
    y = _as_tensor(reference, device)
    if data_range is None:
        data_range = float(y.max() - y.min())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    blur = Gaussian3D(device=device)
    mu_x, mu_y = blur(x), blur(y)
    xx = blur(x * x) - mu_x * mu_x
    yy = blur(y * y) - mu_y * mu_y
    xy = blur(x * y) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (xx + yy + c2)
    return float(torch.mean(numerator / denominator))


def _log_response(array, sigma: float = SSIM_SIGMA,
                  device: str = DEVICE) -> torch.Tensor:
    blur = Gaussian3D(sigma=sigma, device=device)
    s = blur(_as_tensor(array, device))
    return (-6.0 * s[1:-1, 1:-1, 1:-1]
            + s[2:, 1:-1, 1:-1] + s[:-2, 1:-1, 1:-1]
            + s[1:-1, 2:, 1:-1] + s[1:-1, :-2, 1:-1]
            + s[1:-1, 1:-1, 2:] + s[1:-1, 1:-1, :-2])


def hfen(volume, reference, sigma: float = SSIM_SIGMA, device: str = DEVICE,
         mask: np.ndarray | None = None) -> float:
    """Relative LoG error, optionally restricted to a fixed reference ROI."""
    a = _log_response(volume, sigma, device).double()
    b = _log_response(reference, sigma, device).double()
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != np.asarray(reference).shape:
            raise ValueError("HFEN mask must have the reference volume shape")
        margin = tuple((m.shape[i] - b.shape[i]) // 2 for i in range(3))
        interior = tuple(slice(n, n + b.shape[i]) for i, n in enumerate(margin))
        selected = torch.as_tensor(np.ascontiguousarray(m[interior]),
                                   device=device, dtype=torch.bool)
        if not bool(selected.any()):
            return float("nan")
        a, b = a[selected], b[selected]
    denominator = float(torch.linalg.vector_norm(b))
    if denominator <= 0:
        return float("inf")
    return float(torch.linalg.vector_norm(a - b) / denominator)


def nrmse(volume, reference) -> float:
    v = np.asarray(volume, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    rms = float(np.sqrt((r * r).mean()))
    return float(np.sqrt(((v - r) ** 2).mean()) / rms) if rms > 0 else float("inf")


def evaluate(volume, reference, device: str = DEVICE) -> dict:
    """Every metric of the engine table, on one documented intensity scale.

    The reconstruction is amplitude-matched to the reference by least squares
    first, so MSE and PSNR refer to the same quantity and
    ``PSNR = -10 log10 MSE`` holds by construction with ``peak = 1``.
    """
    v = np.asarray(volume, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    scale = ls_scale(v, r)
    scaled = (v * scale).astype(np.float32)
    mse = float(((scaled.astype(np.float64) - r.astype(np.float64)) ** 2).mean())
    peak = 1.0
    gain, offset = affine_match(v, r)
    return {"mse": mse,
            "psnr": float(20 * np.log10(peak) - 10 * np.log10(mse)) if mse > 0
                    else float("inf"),
            "ssim": ssim3d(scaled, r, data_range=peak, device=device),
            "nrmse": nrmse(scaled, r),
            "hfen": hfen(scaled, r, device=device),
            "ls_scale": scale,
            "affine_gain": gain,
            "affine_offset": offset,
            "reference_peak": peak}


def evaluate_rois(volume, reference, *, rois: dict[str, np.ndarray] | None = None,
                  threshold: float | None = None, dilation_voxels: int = 12,
                  boundary_width_voxels: int = 3,
                  device: str = DEVICE) -> dict:
    """Fixed-reference ROI metrics, matching ``eval/metrics.py``."""
    v = np.asarray(volume, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    if v.shape != r.shape:
        raise ValueError(f"shape mismatch: reconstruction {v.shape}, reference {r.shape}")
    masks = reference_rois(
        r, threshold=threshold, dilation_voxels=dilation_voxels,
        boundary_width_voxels=boundary_width_voxels
    ) if rois is None else {k: np.asarray(m, dtype=bool) for k, m in rois.items()}
    required = {"foreground", "foreground_dilated", "boundary_shell", "far_background"}
    missing = required.difference(masks)
    if missing:
        raise ValueError(f"missing ROI masks: {sorted(missing)}")
    if any(mask.shape != r.shape for mask in masks.values()):
        raise ValueError("all ROI masks must have the reference volume shape")

    scale = masked_ls_scale(v, r, masks["foreground_dilated"])
    scaled = (v * scale).astype(np.float32)
    regions: dict[str, dict[str, float | int]] = {}
    for name in ("foreground_dilated", "boundary_shell", "far_background"):
        mse, relative = _masked_error(scaled, r, masks[name])
        regions[name] = {
            "voxels": int(masks[name].sum()),
            "fraction": float(masks[name].mean()),
            "mse": mse, "nrmse": relative,
            "hfen": hfen(scaled, r, mask=masks[name], device=device),
        }
    whole_mse, whole_nrmse = _masked_error(scaled, r, np.ones_like(r, dtype=bool))
    regions["whole_volume"] = {
        "voxels": int(r.size), "fraction": 1.0,
        "mse": whole_mse, "nrmse": whole_nrmse,
        "hfen": hfen(scaled, r, device=device),
    }
    bbox = _foreground_bbox(masks["foreground_dilated"])
    return {
        "protocol": "fixed-reference-roi-v1", "ls_scale": scale,
        "roi_source": "dense_reference_only",
        "threshold": float(otsu_threshold(r) if threshold is None else threshold),
        "dilation_voxels": int(dilation_voxels),
        "boundary_width_voxels": int(boundary_width_voxels),
        "foreground_bbox": {
            "bounds": [[int(s.start), int(s.stop)] for s in bbox],
            "ssim": ssim3d(scaled[bbox], r[bbox], data_range=1.0, device=device),
        },
        "regions": regions,
    }


def fractional_data_residual(projected, measured) -> float:
    """``||s A x - y|| / ||y||`` after the least-squares projection-domain scale."""
    p = np.asarray(projected, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    denominator = float((p * p).sum())
    scale = float((p * y).sum() / denominator) if denominator > 0 else 0.0
    norm_y = float(np.linalg.norm(y))
    return float(np.linalg.norm(scale * p - y) / norm_y) if norm_y > 0 else float("inf")
