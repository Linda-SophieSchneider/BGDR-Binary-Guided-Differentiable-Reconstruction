"""Ground-truth BGDR experiment on the Defrise flange phantom.

Reconstruction and evaluation follow the BGDR paper protocol. The primary
acquisition is the predeclared uniform 120-degree wedge with 60 views and
monochromatic Poisson transmission noise at I0 = 1e4.

The phantom is not distributed with this repository, see data/README.md for
how to obtain it and where to place it.

Run from the repository root, with a CUDA GPU available::

    CUDA_VISIBLE_DEVICES=0 python code/run_defrise_bgnr.py \
      --constraint wedge120 --views 60 --photon-count 10000 \
      --noise-seeds 0 1 2 --grid 384 --output results/defrise_bgnr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BGNR_ROOT = HERE.parent
PHANTOM = BGNR_ROOT / "data" / "lof_flange_v3.npy"
PHANTOM_METADATA = BGNR_ROOT / "data" / "lof_flange_v3_metadata.json"
sys.path.insert(0, str(HERE))

import diffct_mlx as dct  # noqa: E402
import engine_cuda as E  # noqa: E402
import metrics_torch as M  # noqa: E402
from bgnr_torch import (  # noqa: E402
    BGNRConfig,
    masked_backprojection_init,
    reconstruct_bgnr,
)
from run_engine_cuda import select_support_mask  # noqa: E402


DEVICE = "cuda"
SID_MM = 500.0
SDD_MM = 900.0
DETECTOR_PIXELS = 256
DETECTOR_PITCH_MM = 0.5
VOXEL_MM_NATIVE = 0.3
GRID_NATIVE = 384
CONSTRAINTS = {
    "wedge120": (-60.0, 60.0, -12.0, 12.0),
    "lamino": (-180.0, 180.0, 25.0, 35.0),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def as_tensor(array) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=DEVICE, dtype=torch.float32).contiguous()
    return torch.as_tensor(
        np.ascontiguousarray(np.asarray(array), dtype=np.float32), device=DEVICE
    )


def load_phantom(grid: int) -> np.ndarray:
    volume = np.load(PHANTOM).astype(np.float32)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"expected a cubic 3-D phantom, got {volume.shape}")
    source = int(volume.shape[0])
    if grid == source:
        return volume
    if grid < source and source % grid == 0:
        factor = source // grid
        return volume.reshape(
            grid, factor, grid, factor, grid, factor
        ).mean(axis=(1, 3, 5)).astype(np.float32)
    raise ValueError(f"cannot block-average phantom from {source}^3 to {grid}^3")


def uniform_sources(constraint: str, views: int) -> np.ndarray:
    theta_min, theta_max, phi_min, phi_max = CONSTRAINTS[constraint]
    endpoint = (theta_max - theta_min) < 359.0
    theta = np.deg2rad(np.linspace(theta_min, theta_max, views, endpoint=endpoint))
    phi = np.full(views, np.deg2rad(0.5 * (phi_min + phi_max)), dtype=np.float64)
    cos_phi = np.cos(phi)
    return np.stack(
        [
            -SID_MM * cos_phi * np.sin(theta),
            SID_MM * cos_phi * np.cos(theta),
            SID_MM * np.sin(phi),
        ],
        axis=-1,
    ).astype(np.float32)


def detector_geometry(sources: np.ndarray) -> tuple[np.ndarray, ...]:
    src = np.asarray(sources, dtype=np.float32)
    unit = src / np.maximum(np.linalg.norm(src, axis=1, keepdims=True), 1e-6)
    det_center = -unit * (SDD_MM - SID_MM)
    u_raw = np.stack([-unit[:, 1], unit[:, 0], np.zeros(len(unit))], axis=-1)
    u_norm = np.linalg.norm(u_raw, axis=1, keepdims=True)
    fallback = np.broadcast_to(np.array([[1.0, 0.0, 0.0]], np.float32), u_raw.shape)
    det_u = np.where(u_norm > 1e-6, u_raw / np.maximum(u_norm, 1e-6), fallback)
    v_raw = np.cross(unit, det_u)
    sign = np.sign(v_raw[:, 2:3] + 1e-12)
    det_v = np.where(np.abs(v_raw[:, 2:3]) > 1e-6, v_raw * sign, v_raw)
    det_v /= np.maximum(np.linalg.norm(det_v, axis=1, keepdims=True), 1e-6)
    return tuple(np.asarray(a, dtype=np.float32) for a in (src, det_center, det_u, det_v))


def add_poisson_noise(clean: np.ndarray, photon_count: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    expected = float(photon_count) * np.exp(-clean.astype(np.float64))
    measured = np.maximum(rng.poisson(expected).astype(np.float64), 1.0)
    return (-np.log(measured / float(photon_count))).astype(np.float32)


class Setup:
    def __init__(self, sino: np.ndarray, geometry: tuple[np.ndarray, ...],
                 volume_shape: tuple[int, int, int], voxel_mm: float):
        self.sino_np = np.asarray(sino, dtype=np.float32)
        self.sino = as_tensor(self.sino_np)
        self.sino_views = [self.sino[i] for i in range(self.sino.shape[0])]
        self.volume_shape = volume_shape
        self.voxel = float(voxel_mm)
        self.pitch = DETECTOR_PITCH_MM
        self.src, self.det_c, self.det_u, self.det_v = (
            as_tensor(a) for a in geometry
        )
        detector_shape = tuple(int(v) for v in self.sino.shape[1:])
        self.fp_view, self.bp_view, self.bp_all = dct.make_cone_3d_operators(
            self.src,
            self.det_c,
            self.det_u,
            self.det_v,
            volume_shape=volume_shape,
            detector_shape=detector_shape,
            du=self.pitch,
            dv=self.pitch,
            voxel_spacing=self.voxel,
            projector_mode="footprint",
        )

    def forward_all(self, volume: torch.Tensor) -> torch.Tensor:
        return dct.cone_forward_footprint(
            volume,
            self.src,
            self.det_c,
            self.det_u,
            self.det_v,
            DETECTOR_PIXELS,
            DETECTOR_PIXELS,
            self.pitch,
            self.pitch,
            self.voxel,
        )

    def backward_all(self, sinogram: torch.Tensor) -> torch.Tensor:
        return dct.cone_backward_footprint(
            sinogram,
            self.src,
            self.det_c,
            self.det_u,
            self.det_v,
            self.volume_shape[0],
            self.volume_shape[1],
            self.volume_shape[2],
            self.pitch,
            self.pitch,
            self.voxel,
        )

    def residual(self, volume: np.ndarray) -> float:
        projected = self.forward_all(as_tensor(volume)).detach().cpu().numpy()
        return M.fractional_data_residual(projected, self.sino_np)


def reconstruct_bgnr_volume(setup: Setup, support: np.ndarray, epochs: int,
                            ceiling: float = 0.0) -> tuple[np.ndarray, int, float]:
    support_t = as_tensor(support.astype(np.float32))
    initial = masked_backprojection_init(
        setup.sino, setup.forward_all, setup.backward_all, support_t,
        setup.volume_shape,
    )
    started = time.time()
    volume, history = reconstruct_bgnr(
        setup.sino,
        setup.forward_all,
        setup.backward_all,
        initial_volume=initial,
        support_mask=support_t,
        config=BGNRConfig(epochs=epochs, value_ceiling=ceiling),
        progress_callback=lambda epoch, loss: log(
            f"    epoch {epoch:4d}  loss {loss:.6e}"
        ),
    )
    return volume.detach().cpu().numpy(), len(history), time.time() - started


def carve_hull(setup: Setup, sigma_factor: float, tau: float) -> tuple[np.ndarray, dict]:
    sino = setup.sino_np
    border = 16
    background = np.concatenate(
        [
            sino[:, :border, :].ravel(),
            sino[:, -border:, :].ravel(),
            sino[:, :, :border].ravel(),
            sino[:, :, -border:].ravel(),
        ]
    )
    air = float(np.mean(background))
    sigma = float(np.std(background))
    level = air + sigma_factor * sigma
    silhouette = sino > level
    outside = as_tensor((~silhouette).astype(np.float32))
    carved = setup.backward_all(outside).detach().cpu().numpy()
    total = setup.backward_all(torch.ones_like(outside)).detach().cpu().numpy()
    hull = (carved / np.maximum(total, 1e-8)) <= tau
    return hull, {
        "air_level": air,
        "air_sigma": sigma,
        "silhouette_sigma_factor": float(sigma_factor),
        "silhouette_level": level,
        "silhouette_fraction": float(silhouette.mean()),
        "carving_tolerance": float(tau),
        "hull_occupancy": float(hull.mean()),
    }


def support_scores(support: np.ndarray, truth: np.ndarray) -> dict:
    support = np.asarray(support, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((support & truth).sum())
    fp = int((support & ~truth).sum())
    fn = int((~support & truth).sum())
    union = int((support | truth).sum())
    return {
        "occupancy": float(support.mean()),
        "truth_occupancy": float(truth.mean()),
        "containment": tp / max(int(truth.sum()), 1),
        "false_negative_fraction": fn / max(int(truth.sum()), 1),
        "precision": tp / max(tp + fp, 1),
        "iou": tp / max(union, 1),
        "false_positive_voxels": fp,
        "false_negative_voxels": fn,
    }


def artifact(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def score_volume(name: str, volume: np.ndarray, reference: np.ndarray,
                 rois: dict[str, np.ndarray], setup: Setup,
                 path: Path, *, support: np.ndarray | None = None,
                 epochs: int | None = None, runtime_seconds: float | None = None,
                 material_level: float | None = None) -> dict:
    row = {
        "method": name,
        "whole_volume": M.evaluate(volume, reference),
        "fixed_roi": M.evaluate_rois(volume, reference, rois=rois),
        "fractional_data_residual": setup.residual(volume),
        "artifact": artifact(path),
    }
    if support is not None:
        row["support"] = support_scores(support, rois["foreground"])
    if epochs is not None:
        row["epochs"] = int(epochs)
    if runtime_seconds is not None:
        row["runtime_seconds"] = float(runtime_seconds)
    if material_level is not None and material_level > 0:
        row["amplitude_excess"] = float(
            np.percentile(volume, 99.9) / material_level
        )
    log(
        f"  {name:24s} MSE {row['whole_volume']['mse']:.5e}  "
        f"SSIM {row['whole_volume']['ssim']:.4f}  "
        f"NRMSE {row['whole_volume']['nrmse']:.4f}  "
        f"HFEN {row['whole_volume']['hfen']:.4f}  "
        f"res {row['fractional_data_residual']:.4f}"
    )
    return row


def cached_volume(path: Path, build) -> tuple[np.ndarray, bool, dict]:
    if path.exists():
        log(f"{path.stem}: cached")
        return np.load(path), True, {}
    started = time.time()
    volume, metadata = build()
    np.save(path, np.asarray(volume, dtype=np.float32))
    return np.asarray(volume, dtype=np.float32), False, {
        "runtime_seconds": time.time() - started, **metadata
    }


def run_seed(args, seed: int, clean: np.ndarray, geometry: tuple[np.ndarray, ...],
             reference_physical: np.ndarray, reference: np.ndarray,
             rois: dict[str, np.ndarray], output: Path) -> dict:
    tag = f"{args.constraint}_k{args.views:04d}_i{int(args.photon_count):d}_n{seed}"
    recon_dir = output / "reconstructions" / tag
    recon_dir.mkdir(parents=True, exist_ok=True)
    sino_path = output / f"sinogram_{tag}.npy"
    if sino_path.exists():
        sino = np.load(sino_path)
        log(f"noise seed {seed}: sinogram cached")
    else:
        sino = add_poisson_noise(clean, args.photon_count, seed)
        np.save(sino_path, sino)
    setup = Setup(sino, geometry, tuple(reference.shape), args.voxel_mm)
    log(
        f"noise seed {seed}: sinogram {sino.shape}, range "
        f"[{sino.min():.4f}, {sino.max():.4f}]"
    )

    rows: list[dict] = []
    sart_path = recon_dir / "sart.npy"
    sart, _, sart_meta = cached_volume(
        sart_path,
        lambda: (E.run_sart(setup, args.sart_iters), {"iterations": args.sart_iters}),
    )
    rows.append(score_volume(
        "sart", sart, reference, rois, setup, sart_path,
        runtime_seconds=sart_meta.get("runtime_seconds"),
    ))

    dart_path = recon_dir / "dart.npy"
    dart, _, dart_meta = cached_volume(
        dart_path,
        lambda: (
            E.run_dart(setup, sart, args.dart_iters),
            {"iterations": args.dart_iters},
        ),
    )
    rows.append(score_volume(
        "dart", dart, reference, rois, setup, dart_path,
        runtime_seconds=dart_meta.get("runtime_seconds"),
    ))

    mask_path = recon_dir / "dart_support.npy"
    mask_info_path = output / f"dart_support_{tag}.json"
    if mask_path.exists() and mask_info_path.exists():
        dart_support = np.load(mask_path).astype(bool)
        mask_info = json.loads(mask_info_path.read_text())
        log("DART support: cached")
    else:
        log("selecting DART support without reference access")
        dart_support, mask_info = select_support_mask(setup, dart)
        np.save(mask_path, dart_support)
        mask_info_path.write_text(json.dumps(mask_info, indent=2) + "\n")
    dilated = E.dilate(dart_support, args.mask_dilation)

    hull_path = recon_dir / "carved_hull.npy"
    hull_info_path = output / f"carved_hull_{tag}.json"
    if hull_path.exists() and hull_info_path.exists():
        hull = np.load(hull_path).astype(bool)
        hull_info = json.loads(hull_info_path.read_text())
        log("carved hull: cached")
    else:
        hull, hull_info = carve_hull(setup, args.sigma_factor, args.tau)
        np.save(hull_path, hull)
        hull_info_path.write_text(json.dumps(hull_info, indent=2) + "\n")

    truth_support = rois["foreground"]
    support_report = {
        "dart": support_scores(dart_support, truth_support),
        "dilated": support_scores(dilated, truth_support),
        "carved_hull": support_scores(hull, truth_support),
        "oracle": support_scores(truth_support, truth_support),
    }
    log(
        "supports: "
        + ", ".join(
            f"{name} contains {100 * values['containment']:.2f}%"
            for name, values in support_report.items()
        )
    )

    first_view = setup.fp_view(as_tensor(sart), 0)
    epsilon = float(np.linalg.norm(
        first_view.detach().cpu().numpy() - setup.sino_np[0]
    ))
    sweep_path = output / f"asdpocs_sweep_{tag}.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
    else:
        sweep = []
        for alpha in args.alpha_sweep:
            log(f"ASD-POCS probe alpha={alpha:g}")
            probe = E.run_asd_pocs(
                setup, alpha, epsilon, max(10, args.asd_iters // 5),
                args.asd_reg_iters,
            )
            sweep.append({"alpha": alpha, "residual": setup.residual(probe)})
        sweep_path.write_text(json.dumps(sweep, indent=2) + "\n")
    best_alpha = float(min(sweep, key=lambda row: row["residual"])["alpha"])
    asd_path = recon_dir / "asdpocs.npy"
    asd, _, asd_meta = cached_volume(
        asd_path,
        lambda: (
            E.run_asd_pocs(
                setup, best_alpha, epsilon, args.asd_iters, args.asd_reg_iters
            ),
            {"iterations": args.asd_iters, "alpha": best_alpha},
        ),
    )
    rows.append(score_volume(
        "asdpocs", asd, reference, rois, setup, asd_path,
        runtime_seconds=asd_meta.get("runtime_seconds"),
    ))

    material_level = float(np.percentile(dart, 99.9))
    ceiling = args.ceiling_factor * material_level
    configurations = [
        ("bgdr_dart_support", dart_support, 0.0),
        ("bgdr_dilated_support", dilated, 0.0),
        ("bgdr_carved_hull", hull, 0.0),
        ("bgdr_carved_hull_box", hull, ceiling),
        ("bgdr_oracle_support", truth_support, 0.0),
    ]
    for name, support, bound in configurations:
        path = recon_dir / f"{name}.npy"

        def build(support=support, bound=bound):
            volume, epochs, runtime = reconstruct_bgnr_volume(
                setup, support, args.bgnr_epochs, ceiling=bound
            )
            return volume, {"epochs": epochs, "bgnr_runtime_seconds": runtime}

        log(
            f"{name}: support {support.mean():.4f}"
            + (f", ceiling {bound:.5f}" if bound else "")
        )
        volume, _, metadata = cached_volume(path, build)
        rows.append(score_volume(
            name,
            volume,
            reference,
            rois,
            setup,
            path,
            support=support,
            epochs=metadata.get("epochs"),
            runtime_seconds=metadata.get("bgnr_runtime_seconds"),
            material_level=material_level,
        ))
        del volume
        torch.cuda.empty_cache()

    seed_payload = {
        "schema_version": "bgnr-defrise-seed-v1",
        "dataset": "defrise_flange",
        "constraint": args.constraint,
        "view_count": args.views,
        "noise_seed": seed,
        "photon_count": args.photon_count,
        "grid": list(reference.shape),
        "voxel_mm": args.voxel_mm,
        "detector": {
            "shape": [DETECTOR_PIXELS, DETECTOR_PIXELS],
            "pitch_mm": DETECTOR_PITCH_MM,
            "sid_mm": SID_MM,
            "sdd_mm": SDD_MM,
        },
        "reference": {
            "source": str(PHANTOM),
            "physical_range_per_mm": [
                float(reference_physical.min()), float(reference_physical.max())
            ],
            "normalization": "divide by exact phantom maximum",
            "support_definition": "physical attenuation > 0",
        },
        "support_parameters": {
            "dart": mask_info,
            "dilation_voxels": args.mask_dilation,
            "hull": hull_info,
            "ground_truth_diagnostics": support_report,
        },
        "asdpocs": {
            "alpha": best_alpha,
            "epsilon": epsilon,
            "sweep": sweep,
            "iterations": args.asd_iters,
            "regularization_iterations": args.asd_reg_iters,
        },
        "amplitude_box": {
            "material_estimator": "same-seed DART p99.9",
            "material_level": material_level,
            "factor": args.ceiling_factor,
            "ceiling": ceiling,
        },
        "rows": rows,
        "inputs": {
            "phantom": artifact(PHANTOM),
            "phantom_metadata": artifact(PHANTOM_METADATA),
            "sinogram": artifact(sino_path),
        },
    }
    out = output / f"defrise_results_{tag}.json"
    out.write_text(json.dumps(seed_payload, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out}")
    del setup, sino, sart, dart, asd
    torch.cuda.empty_cache()
    return seed_payload


def nested(rows: list[dict], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def aggregate_runs(seed_runs: list[dict]) -> dict:
    methods = [row["method"] for row in seed_runs[0]["rows"]]
    summary = {}
    paths = {
        "mse": ("whole_volume", "mse"),
        "ssim": ("whole_volume", "ssim"),
        "nrmse": ("whole_volume", "nrmse"),
        "hfen": ("whole_volume", "hfen"),
        "foreground_mse": (
            "fixed_roi", "regions", "foreground_dilated", "mse"
        ),
        "bbox_ssim": ("fixed_roi", "foreground_bbox", "ssim"),
        "boundary_hfen": ("fixed_roi", "regions", "boundary_shell", "hfen"),
        "far_background_mse": (
            "fixed_roi", "regions", "far_background", "mse"
        ),
        "residual": ("fractional_data_residual",),
    }
    for method in methods:
        rows = [
            next(row for row in run["rows"] if row["method"] == method)
            for run in seed_runs
        ]
        summary[method] = {}
        for label, path in paths.items():
            values = nested(rows, path)
            summary[method][label] = {
                "mean": float(np.mean(values)),
                "population_std": float(np.std(values)),
                "values": values,
            }
    return summary


def sanity_check(args, phantom: np.ndarray, geometry: tuple[np.ndarray, ...]) -> int:
    small_geometry = tuple(a[:2] for a in geometry)
    dummy = np.zeros((2, DETECTOR_PIXELS, DETECTOR_PIXELS), dtype=np.float32)
    setup = Setup(dummy, small_geometry, tuple(phantom.shape), args.voxel_mm)
    projection = setup.forward_all(as_tensor(phantom))
    backprojection = setup.backward_all(torch.ones_like(projection))
    checks = {
        "source_norms_mm": np.linalg.norm(small_geometry[0], axis=1).tolist(),
        "projection_shape": list(projection.shape),
        "projection_range": [float(projection.min()), float(projection.max())],
        "projection_finite": bool(torch.isfinite(projection).all()),
        "backprojection_shape": list(backprojection.shape),
        "backprojection_finite": bool(torch.isfinite(backprojection).all()),
        "backprojection_nonzero": bool((backprojection > 0).any()),
    }
    print(json.dumps(checks, indent=2))
    return 0 if all(
        checks[key] for key in (
            "projection_finite", "backprojection_finite", "backprojection_nonzero"
        )
    ) else 1


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint", choices=sorted(CONSTRAINTS), required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--photon-count", type=float, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=int, default=GRID_NATIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sart-iters", type=int, default=30)
    parser.add_argument("--dart-iters", type=int, default=20)
    parser.add_argument("--asd-iters", type=int, default=100)
    parser.add_argument("--asd-reg-iters", type=int, default=10)
    parser.add_argument("--alpha-sweep", type=float, nargs="+",
                        default=[0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--bgnr-epochs", type=int, default=500)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument("--sigma-factor", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--ceiling-factor", type=float, default=1.5)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.voxel_mm = VOXEL_MM_NATIVE * GRID_NATIVE / args.grid

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 384^3 Defrise experiment")
    if not PHANTOM.exists():
        raise SystemExit(f"phantom not found at {PHANTOM}, see data/README.md")
    log(f"device: {torch.cuda.get_device_name(0)}")
    phantom = load_phantom(args.grid)
    reference = (phantom / float(phantom.max())).astype(np.float32)
    rois = M.reference_rois(
        reference, threshold=0.0, dilation_voxels=12, boundary_width_voxels=3
    )
    geometry = detector_geometry(uniform_sources(args.constraint, args.views))
    if args.sanity_only:
        return sanity_check(args, phantom, geometry)

    clean_path = args.output / (
        f"sinogram_clean_{args.constraint}_k{args.views:04d}_g{args.grid}.npy"
    )
    if clean_path.exists():
        clean = np.load(clean_path)
        log("clean ground-truth projection: cached")
    else:
        log("forward-projecting the exact ground truth")
        empty = np.zeros(
            (args.views, DETECTOR_PIXELS, DETECTOR_PIXELS), dtype=np.float32
        )
        clean_setup = Setup(empty, geometry, tuple(phantom.shape), args.voxel_mm)
        clean = clean_setup.forward_all(as_tensor(phantom)).detach().cpu().numpy()
        np.save(clean_path, clean)
        del clean_setup
        torch.cuda.empty_cache()
    log(f"clean sinogram range [{clean.min():.4f}, {clean.max():.4f}]")

    runs = []
    for seed in args.noise_seeds:
        log(f"starting noise seed {seed}")
        runs.append(run_seed(
            args, seed, clean, geometry, phantom, reference, rois, args.output
        ))

    payload = {
        "schema_version": "bgnr-defrise-experiment-v1",
        "status": "complete",
        "command_parameters": {
            "constraint": args.constraint,
            "views": args.views,
            "photon_count": args.photon_count,
            "noise_seeds": args.noise_seeds,
            "grid": args.grid,
        },
        "protocol": {
            "trajectory": "uniform feasible source positions",
            "measurement": "monochromatic Beer-Lambert plus Poisson counts",
            "reference": "exact digital phantom ground truth",
            "selection_without_reference": True,
            "metric_rois": "exact ground-truth support, fixed for all methods",
            "inverse_crime_caveat": (
                "simulation and reconstruction use the matched voxel-footprint "
                "operator; photon noise is included"
            ),
        },
        "aggregate": aggregate_runs(runs),
        "seed_result_files": [
            str(args.output / (
                f"defrise_results_{args.constraint}_k{args.views:04d}_"
                f"i{int(args.photon_count):d}_n{seed}.json"
            )) for seed in args.noise_seeds
        ],
        "provenance": {
            "script": artifact(Path(__file__).resolve()),
            "phantom": artifact(PHANTOM),
            "bgnr_revision": git_revision(BGNR_ROOT),
            "phantom_sha256": digest(PHANTOM) if PHANTOM.exists() else "unavailable",
            "diffct_mlx_version": getattr(dct, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "runtime_seconds": time.time() - started,
    }
    result_path = args.output / "defrise_results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "bgnr-run-manifest-v1",
        "dataset": "defrise_flange",
        "status": "complete",
        "result": artifact(result_path),
        "clean_sinogram": artifact(clean_path),
        "seed_outputs": [artifact(Path(path)) for path in payload["seed_result_files"]],
        "runtime_seconds": payload["runtime_seconds"],
    }
    manifest_path = args.output / "defrise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"wrote {result_path}")
    log(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
