"""Run the fixed-budget SART baseline for all measured camera trajectories.

The geometry and admissible field of view are loaded through the same camera
experiment code used by ``run_camera_bgnr.py``.  FDK is deliberately absent:
the planned arms are non-circular and the paper does not rank FDK as a baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CAMERA_REPO = Path("/home/schneider/TrajectoryOptimization/RealWorldExample_ConOpt")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMERA_REPO))

import reconstruct_ezrt_cuda as rec  # noqa: E402
import regularization_sweep as sweep  # noqa: E402
import diffct_mlx as dct  # noqa: E402
from diffct_mlx.backend import active as backend  # noqa: E402
from diffct_mlx.reconstruction_algorithms.cases import (  # noqa: E402
    _build_leap_style_circular_fov_mask,
    _build_sensitivity_support_mask,
)
from EZRT_Helpers.rek2py import rek2py  # noqa: E402

OUT = HERE.parent / "reconstructions"
RESULTS = HERE.parent / "results"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("circular", "bundle", "all3"))
    parser.add_argument("--k", required=True, type=int, choices=(100, 400))
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    _, reference = rek2py(str(sweep.REFERENCE), switch_order=True)
    shape = tuple(np.asarray(reference).shape)
    src, det_c, det_u_v, det_v_v, sino, du, dv = sweep.load_arm(
        args.arm, args.k, shape
    )
    n_views, det_u, det_v = sino.shape

    def tensor(value):
        return torch.as_tensor(
            np.ascontiguousarray(np.asarray(value), dtype=np.float32), device="cuda"
        )

    src_t, dc_t, du_t, dv_t = map(tensor, (src, det_c, det_u_v, det_v_v))
    fp_view, bp_view, bp_all = dct.make_cone_3d_operators(
        src_t, dc_t, du_t, dv_t,
        volume_shape=shape, detector_shape=(det_u, det_v),
        du=du, dv=dv, voxel_spacing=sweep.VOXEL_MM, projector_mode="footprint",
    )
    sensitivity = _build_sensitivity_support_mask(bp_all, (n_views, det_u, det_v))
    fov = _build_leap_style_circular_fov_mask(
        shape, sweep.VOXEL_MM, (det_u, det_v), du, src_t, dc_t
    )
    admissible = sensitivity & fov
    admissible_t = tensor(
        np.asarray(backend.to_numpy(admissible), dtype=np.float32)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"camera_{args.arm}_k{args.k:04d}_sart"
    npy_path = OUT / f"{stem}.npy"
    rek_path = OUT / f"{stem}.rek"

    started = time.time()
    if npy_path.exists():
        volume = np.load(npy_path)
        cached = True
    else:
        params = dct.SARTParameters(
            volume_shape=shape,
            iteration_count=args.iterations,
            sart_iteration_count=1,
            iterative_update_method="sart",
            shuffle_projection_order=False,
            enforce_positivity=True,
            volume_support_mask=admissible_t,
            volume_support_mask_mode="always",
        )
        views = [tensor(sino[i]) for i in range(n_views)]
        result = dct.reconstruct_sart(
            views, fp_view, bp_view, params, show_progress=True
        )
        volume = np.asarray(backend.to_numpy(result), dtype=np.float32)
        np.save(npy_path, volume)
        cached = False
    rec.save_rek(volume, rek_path, sweep.VOXEL_MM * 1000.0)
    runtime = time.time() - started

    payload = {
        "schema_version": "bgnr-run-manifest-v1",
        "dataset": "camera",
        "trajectory": args.arm,
        "view_count": int(n_views),
        "support": {"source": "geometry_fov_and_sensitivity_only"},
        "optimizer": {
            "name": "SART",
            "iterations": args.iterations,
            "projection_order": "fixed",
            "positivity": True,
            "objective": "mean_projection_mse",
            "projection_ssim_weight": 0.0,
        },
        "stopping": {"type": "fixed_budget", "iterations": args.iterations},
        "backend": {"package": "diffct_mlx", "version": dct.__version__,
                    "array_backend": "torch", "projector_mode": "footprint"},
        "hardware": {"device": str(torch.cuda.get_device_name(0))},
        "runtime_seconds": runtime,
        "cached_input": cached,
        "geometry_source": "regularization_sweep.load_arm",
        "volume_shape": list(shape),
        "detector_shape": [int(det_u), int(det_v)],
        "outputs": [
            {"path": str(npy_path), "sha256": digest(npy_path)},
            {"path": str(rek_path), "sha256": digest(rek_path)},
        ],
    }
    manifest = RESULTS / f"{stem}_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {rek_path} and {manifest}; runtime {runtime:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
