"""The engine pipeline on CUDA: geometry, operators and the library solvers.

A torch translation of the parts of ``Research/BGNR/eval/run_engine.py`` that
wrap the ``diffct_mlx`` solvers.  The solvers themselves are the same code on
both backends; what this module replaces is the MLX array handling around them.

Splitting it out from the run script keeps one place where the geometry and the
two projectors are defined, which is what every stage and every validation
shares.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

import diffct_mlx as dct
import engine_data as ed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent.parent / "reconstructions"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def as_tensor(array, device: str = DEVICE) -> torch.Tensor:
    """numpy arrays, tensors already on the device, and library outputs alike."""
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=torch.float32).contiguous()
    return torch.as_tensor(np.ascontiguousarray(np.asarray(array), dtype=np.float32),
                           device=device)


class Setup:
    """Geometry, operators and data for one detector binning."""

    def __init__(self, detector_bin: int, n_views: int, tag: str,
                 view_start: int = 0):
        self.bin = detector_bin
        self.n_views = n_views
        self.tag = tag
        self.view_start = int(view_start)
        self.pitch = ed.DETECTOR_PITCH_MM * detector_bin
        self.voxel = ed.VOXEL_MM * detector_bin
        self.volume_shape = tuple(s // detector_bin for s in ed.VOLUME_SHAPE)

        OUT.mkdir(parents=True, exist_ok=True)
        cache = OUT / f"sino_{tag}.npy"
        if cache.exists():
            self.sino_np = np.load(cache)
        else:
            log(f"loading {n_views} projections (detector bin {detector_bin}) ...")
            self.sino_np = ed.load_projections(
                n_views, view_start=self.view_start, detector_bin=detector_bin
            )
            np.save(cache, self.sino_np)
        self.sino = as_tensor(self.sino_np)
        self.sino_views = [self.sino[i] for i in range(self.sino.shape[0])]
        log(f"sinogram {tuple(self.sino.shape)} range "
            f"[{self.sino_np.min():.3f}, {self.sino_np.max():.3f}]")

        src, det_c, det_u, det_v = ed.geometry(
            n_views, view_start=self.view_start
        )
        self.src, self.det_c = as_tensor(src), as_tensor(det_c)
        self.det_u, self.det_v = as_tensor(det_u), as_tensor(det_v)
        detector_shape = tuple(int(s) for s in self.sino.shape[1:])
        self.fp_view, self.bp_view, self.bp_all = dct.make_cone_3d_operators(
            self.src, self.det_c, self.det_u, self.det_v,
            volume_shape=self.volume_shape, detector_shape=detector_shape,
            du=self.pitch, dv=self.pitch, voxel_spacing=self.voxel,
            projector_mode="footprint",
        )
        _, _, self.bp_all_siddon = dct.make_cone_3d_operators(
            self.src, self.det_c, self.det_u, self.det_v,
            volume_shape=self.volume_shape, detector_shape=detector_shape,
            du=self.pitch, dv=self.pitch, voxel_spacing=self.voxel,
            projector_mode="siddon",
        )

    def forward_all(self, volume: torch.Tensor) -> torch.Tensor:
        return dct.cone_forward_footprint(
            volume, self.src, self.det_c, self.det_u, self.det_v,
            int(self.sino.shape[1]), int(self.sino.shape[2]),
            self.pitch, self.pitch, self.voxel)

    def backward_all(self, sinogram: torch.Tensor) -> torch.Tensor:
        return dct.cone_backward_footprint(
            sinogram, self.src, self.det_c, self.det_u, self.det_v,
            self.volume_shape[0], self.volume_shape[1], self.volume_shape[2],
            self.pitch, self.pitch, self.voxel)

    def residual(self, volume) -> float:
        from metrics_torch import fractional_data_residual
        projected = self.forward_all(as_tensor(volume))
        return fractional_data_residual(projected.detach().cpu().numpy(), self.sino_np)

    def path(self, name: str) -> Path:
        return OUT / f"{name}_{self.tag}.npy"


def cached(setup: Setup, name: str, build):
    """Compute once, then reuse -- the same contract as the MLX pipeline."""
    path = setup.path(name)
    if path.exists():
        log(f"{name}: cached")
        return np.load(path)
    start = time.time()
    volume = np.asarray(build())
    np.save(path, volume)
    log(f"{name}: done in {time.time() - start:.0f}s  "
        f"range [{volume.min():.4f}, {volume.max():.4f}]")
    return volume


def run_fdk(setup: Setup) -> np.ndarray:
    n_u, n_v = int(setup.sino.shape[1]), int(setup.sino.shape[2])
    u = (torch.arange(n_u, device=DEVICE, dtype=torch.float32)
         - (n_u - 1) / 2) * setup.pitch
    v = (torch.arange(n_v, device=DEVICE, dtype=torch.float32)
         - (n_v - 1) / 2) * setup.pitch
    cone_w = ed.SDD_MM / torch.sqrt(
        ed.SDD_MM ** 2 + u.reshape(1, -1, 1) ** 2 + v.reshape(1, 1, -1) ** 2)
    params = dct.FDKParameters(
        detector_spacing=setup.pitch, voxel_spacing=setup.voxel,
        normalization_scale=(np.pi * ed.SID_MM) / (2.0 * ed.SDD_MM * setup.n_views),
        filter_axis=1)
    reco = dct.reconstruct_fdk(setup.sino, setup.bp_all_siddon, params,
                              weight_projections=lambda s: s * cone_w)
    return reco.detach().cpu().numpy()


def run_sart(setup: Setup, iterations: int) -> np.ndarray:
    params = dct.SARTParameters(
        volume_shape=setup.volume_shape, iteration_count=iterations,
        sart_iteration_count=1, iterative_update_method="sart",
        shuffle_projection_order=False, enforce_positivity=True)
    volume = dct.reconstruct_sart(setup.sino_views, setup.fp_view, setup.bp_view,
                                  params, show_progress=False)
    return volume.detach().cpu().numpy()


def run_asd_pocs(setup: Setup, alpha: float, epsilon: float,
                 iterations: int, reg_iterations: int) -> np.ndarray:
    reco_params = dct.ReconstructionParameters(
        volume_shape=setup.volume_shape, iteration_count=iterations,
        sart_iteration_count=2, iterative_update_method="sart",
        shuffle_projection_order=False, enforce_positivity=True)
    reg_params = dct.ASDPOCSParameters(
        reg_iteration_count=reg_iterations, alpha=alpha, tv_eps=1e-6,
        epsilon=epsilon, r_max=0.95, alpha_red=0.95, beta=1.0, beta_red=0.995)
    volume = dct.reconstruct_asd_pocs(setup.sino_views, setup.fp_view, setup.bp_view,
                                      reco_params, reg_params, show_progress=False)
    return volume.detach().cpu().numpy()


def run_dart(setup: Setup, sart_volume: np.ndarray, iterations: int) -> np.ndarray:
    material = float(np.percentile(sart_volume, 99.5))
    log(f"dart: gray levels (0, {material:.4f})")
    params = dct.DARTParameters(
        volume_shape=setup.volume_shape, iteration_count=iterations,
        sart_iteration_count=1, iterative_update_method="sart",
        shuffle_projection_order=False, enforce_positivity=True,
        gray_levels=(0.0, material), segmentation_threshold_method="otsu",
        otsu_percentile_window=(1.0, 99.9), free_pixel_probability=0.15,
        apply_smoothing=True, smoothing_beta=0.2, binary_fill_holes=True,
        initial_reconstruction_sweeps=2, random_seed=0)
    volume = dct.reconstruct_dart(setup.sino_views, setup.fp_view, setup.bp_view,
                                  params, show_progress=False)
    return volume.detach().cpu().numpy()


def dilate(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """6-connected binary dilation, ``iterations`` times."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        grown = out.copy()
        for axis in range(3):
            grown |= np.roll(out, 1, axis=axis)
            grown |= np.roll(out, -1, axis=axis)
        out = grown
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        shrunk = out.copy()
        for axis in range(3):
            shrunk &= np.roll(out, 1, axis=axis)
            shrunk &= np.roll(out, -1, axis=axis)
        out = shrunk
    return out


def close(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    return erode(dilate(mask, iterations), iterations)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(counts)
    weight_rev = weight[-1] - weight
    mean = np.cumsum(counts * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mean[-1] * weight - mean * weight[-1]) ** 2 / (
            weight * weight_rev * weight[-1] ** 2)
    return float(centers[int(np.argmax(np.nan_to_num(between)))])


def carve_hull(setup: Setup, sigma_factor: float = 5.0, tau: float = 0.05):
    """The support the measured rays certify, from the air-margin noise level."""
    margin = ed.AIR_MARGIN_PX
    background = np.concatenate([setup.sino_np[:, :margin, :].ravel(),
                                setup.sino_np[:, -margin:, :].ravel()])
    level = sigma_factor * float(np.std(background))
    sil = setup.sino_np > level
    outside = as_tensor((~sil).astype(np.float32))
    carved = setup.backward_all(outside).detach().cpu().numpy()
    total = setup.backward_all(torch.ones_like(outside)).detach().cpu().numpy()
    hull = (carved / np.maximum(total, 1e-8)) <= tau
    info = {"silhouette_level": level, "silhouette_share": float(sil.mean()),
            "tau": tau, "sigma_factor": sigma_factor,
            "occupancy": float(hull.mean())}
    log(f"hull: silhouette level {level:.4f} ({100 * sil.mean():.1f} % of pixels), "
        f"occupancy {hull.mean():.4f}")
    return hull, info
