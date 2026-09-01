"""BGNR on the measured camera scan, on the grid the camera experiment already uses.

Chapter 9's reference-based evaluation rests on one object, the engine block. The
casting was meant to be the second, but its dense reference exists only on a
Windows path nobody here can reach, so it can carry no reference-based metric at
all. This dataset closes that gap and closes it better: a dense reference from
all 1200 circular views, the same bench, and -- the part the casting could never
have supplied -- few-view sets at matched budgets on three trajectory families,
so the acquisition chapters and this one can be measured together on one object.

Everything about the geometry comes from the camera experiment's own sweep
driver: ``regularization_sweep.load_arm`` is called unchanged, so the source and
detector vectors, the view subset, the voxel size and the volume shape are by
construction the ones behind ``results_final/*/reconstruction.rek``.

Calling that function rather than reimplementing it is not convenience. The
object was repositioned between the dense reference scan and this session, and
the planned arms are brought back into the reference frame by a rigid transform
applied to the *geometry* -- ``rot_x(-0.5) @ rot_z(-89.75)`` and a per-scan
translation -- before reconstruction. A pipeline that rebuilds the geometry from
the loader alone, as an earlier version of this script did, reconstructs a
perfectly good volume in the wrong pose: its carved hull covered 55 % of the
reference support and no axis permutation or volume-space registration repaired
it, because the deficit was never an axis convention. Reconstructing in the
reference frame removes the need for any registration at all, and makes these
volumes voxel-comparable with the baseline exactly as ``compute_final_metrics.py``
assumes.

What is added here is the support prior and the two repairs of Chapter 9:

*   the carved hull of the measured silhouettes, intersected with the sensitivity
    and field-of-view masks the ASD-POCS baseline already uses -- outside those
    the data say nothing and the hull would be vacuous;
*   the amplitude box, whose material level is read off the ASD-POCS
    reconstruction of the *same* cell, so no reference volume is consulted.

    python run_camera_bgnr.py --arm all3 --k 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CAMERA_REPO = Path("/home/schneider/TrajectoryOptimization/RealWorldExample_ConOpt")
KAMERA = Path("/ssd_data/diffct_scratch/TrajektorienOptimierung/Kamera")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMERA_REPO))

import numpy as np
import torch

import reconstruct_ezrt_cuda as rec                      # noqa: E402
import regularization_sweep as sweep                     # noqa: E402
import diffct_mlx as dct                                 # noqa: E402
from diffct_mlx.backend import active as _b              # noqa: E402
from diffct_mlx.reconstruction_algorithms.cases import (  # noqa: E402
    _build_leap_style_circular_fov_mask,
    _build_sensitivity_support_mask,
)
from EZRT_Helpers.rek2py import rek2py                   # noqa: E402
from differentiable_coverage.eval import metrics as M    # noqa: E402

from bgnr_torch import (BGNRConfig, masked_backprojection_init,  # noqa: E402
                        reconstruct_bgnr)

OUT = HERE.parent / "reconstructions"
RESULTS = HERE.parent / "results"
TAU = 0.05                 # carving tolerance, as on the engine
SIGMA_FACTOR = 5.0         # silhouette threshold in air-noise sigmas
CEILING_FACTOR = 1.5       # amplitude box, as on the engine
AIR_BORDER_PX = 24         # the border the camera loader calls air


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def carve_hull(sino: np.ndarray, backward, shape) -> tuple[np.ndarray, dict]:
    """The support the measured rays certify, from the air-border noise level.

    The noise scale is estimated robustly, by the median absolute deviation of
    the border rather than its standard deviation.  On this scan the border is
    not pure air -- the camera reaches into it in part of the orbit, so its upper
    tail is object and the plain standard deviation comes out 27 times too large.
    A threshold that high would carve away most of the object: it left a hull of
    0.4 % occupancy, against 24 % for the robust estimate.  The engine reader
    keeps the plain estimate on purpose, because there the air is clipped to
    exactly zero and the median absolute deviation degenerates instead.

    Which way the estimate errs matters more than its precision.  Too low a
    threshold enlarges the silhouette, hence the hull, which the error bound
    tolerates; too high a threshold carves real material away, which it does not.
    The containment reported against the dense reference is the check that this
    landed on the safe side -- a diagnostic, never a selection criterion.
    """
    b = AIR_BORDER_PX
    background = np.concatenate([sino[:, :b, :].ravel(), sino[:, -b:, :].ravel(),
                                 sino[:, :, :b].ravel(), sino[:, :, -b:].ravel()])
    air = float(np.median(background))
    sigma = 1.4826 * float(np.median(np.abs(background - air)))
    level = air + SIGMA_FACTOR * sigma
    sil = sino > level
    outside = torch.as_tensor(np.ascontiguousarray((~sil).astype(np.float32)),
                              device="cuda")
    carved = backward(outside).detach().cpu().numpy()
    total = backward(torch.ones_like(outside)).detach().cpu().numpy()
    hull = (carved / np.maximum(total, 1e-8)) <= TAU
    info = {"silhouette_level": level, "silhouette_share": float(sil.mean()),
            "air_level": air, "robust_sigma": sigma,
            "naive_sigma": float(np.std(background)),
            "tau": TAU, "sigma_factor": SIGMA_FACTOR,
            "hull_occupancy": float(hull.mean())}
    log(f"hull: silhouette level {level:.4f} ({100 * sil.mean():.1f} % of pixels), "
        f"occupancy {hull.mean():.4f}")
    return hull, info


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=["circular", "bundle", "all3"])
    parser.add_argument("--k", type=int, required=True, choices=[100, 400])
    parser.add_argument("--hull-only", action="store_true",
                        help="stop after the support diagnostics")
    parser.add_argument("--epochs", type=int, default=500,
                        help="cap; the framework's own rule stops at 212")
    parser.add_argument("--namespace", default="",
                        help="append a configuration stamp to every output so a "
                             "new protocol never reuses legacy reconstructions")
    args = parser.parse_args()
    cell = f"{args.arm}_k{args.k:04d}"
    suffix = f"_{args.namespace}" if args.namespace else ""
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # The reference defines the grid, exactly as in regularization_sweep.main.
    _, ref = rek2py(str(sweep.REFERENCE), switch_order=True)
    ref = np.asarray(ref, np.float32)
    shape = ref.shape
    voxel_mm = sweep.VOXEL_MM

    src, det_c, det_u_v, det_v_v, sino, du, dv = sweep.load_arm(args.arm, args.k,
                                                               shape)
    n_views, det_u, det_v = sino.shape
    log(f"{cell}: {n_views} views, detector {det_u}x{det_v} at {du:.4f} mm, "
        f"voxel {voxel_mm:.4f} mm, volume {shape}"
        + ("" if args.arm == "circular" else
           f", geometry registered (yaw {sweep.YAW}, roll {sweep.ROLL})"))

    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.asarray(a), np.float32),
                                  device="cuda")
    src_t, dc_t, du_t, dv_t = T(src), T(det_c), T(det_u_v), T(det_v_v)
    _, _, bwd_all = dct.make_cone_3d_operators(
        src_t, dc_t, du_t, dv_t, volume_shape=shape, detector_shape=(det_u, det_v),
        du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="footprint")

    def forward(volume):
        return dct.cone_forward_footprint(volume, src_t, dc_t, du_t, dv_t,
                                          det_u, det_v, du, dv, voxel_mm)

    def backward(sinogram):
        return dct.cone_backward_footprint(sinogram, src_t, dc_t, du_t, dv_t,
                                           shape[0], shape[1], shape[2],
                                           du, dv, voxel_mm)

    # The same admissible region the ASD-POCS baseline is given: voxels the rays
    # actually reach, inside the circular field of view.
    sens = _build_sensitivity_support_mask(bwd_all, (n_views, det_u, det_v))
    fov = _build_leap_style_circular_fov_mask(shape, voxel_mm, (det_u, det_v), du,
                                              src_t, dc_t)
    admissible = np.asarray(_b.to_numpy(sens & fov)).astype(bool)
    log(f"admissible region (sensitivity and FOV): {admissible.mean():.4f}")

    hull, hull_info = carve_hull(sino, backward, shape)
    support = hull & admissible
    hull_info["support_occupancy"] = float(support.mean())
    hull_info["admissible_occupancy"] = float(admissible.mean())
    log(f"carved support within the admissible region: {support.mean():.4f}")

    # Diagnostic only, as on the engine: how much of the object the carved
    # support keeps.  The reference is segmented at Otsu and enters no decision.
    from engine_cuda import otsu_threshold
    dense = ref > otsu_threshold(ref[::8].ravel())
    contained = float((dense & support).sum() / max(dense.sum(), 1))
    hull_info["reference_support_contained"] = contained
    hull_info["reference_support_occupancy"] = float(dense.mean())
    log(f"contains {100 * contained:.1f} % of the segmented reference support "
        f"(occupancy {dense.mean():.4f}) -- diagnostic, not a criterion")

    if args.hull_only:
        for mask, label in ((support, "carved support"), (dense, "reference")):
            i = np.argwhere(mask)
            log(f"  {label:18s} occupancy {mask.mean():.4f}  "
                f"bbox {i.min(0)}..{i.max(0)}  centroid {i.mean(0).round(1)}")
        np.save(OUT / f"camera_{cell}_support{suffix}.npy", support)
        return 0

    del dense

    # The material level for the amplitude box, from the ASD-POCS reconstruction
    # of this same cell -- reference-free, the role DART plays on the engine.
    asd_path = CAMERA_REPO / "results_final" / f"{args.arm}_final_k{args.k:04d}" \
        / "reconstruction.rek"
    _, asd = rek2py(str(asd_path), switch_order=True)
    material = float(np.percentile(np.asarray(asd, np.float32), 99.5))
    ceiling = CEILING_FACTOR * material
    log(f"material level {material:.5f} (99.5th percentile of {asd_path.parent.name})"
        f" -> amplitude box {ceiling:.5f}")
    del asd

    sino_t = T(sino)
    rows = []
    for name, box in (("bgnr_hull", 0.0), ("bgnr_hull_box", ceiling)):
        path = OUT / f"camera_{cell}_{name}{suffix}.npy"
        if path.exists():
            volume = np.load(path)
            log(f"{name}: cached")
        else:
            log(f"{name}: support {support.mean():.4f}"
                + (f", box {box:.5f}" if box else ""))
            support_t = T(support.astype(np.float32))
            init = masked_backprojection_init(sino_t, forward, backward,
                                              support_t, shape)
            start = time.time()
            vol, history = reconstruct_bgnr(
                sino_t, forward, backward, initial_volume=init,
                support_mask=support_t,
                config=BGNRConfig(epochs=args.epochs, value_ceiling=box),
                progress_callback=lambda e, l: log(f"    epoch {e:4d}  loss {l:.6e}"))
            volume = vol.detach().cpu().numpy()
            np.save(path, volume)
            log(f"    {len(history)} epochs, loss {history[-1]:.6e}, "
                f"{time.time() - start:.0f}s")
        rek = OUT / f"camera_{cell}_{name}{suffix}.rek"
        rec.save_rek(volume, rek, voxel_mm * 1000.0)
        projected = forward(T(volume)).detach().cpu().numpy()
        scale = float((projected * sino).sum() / max((projected * projected).sum(), 1e-30))
        residual = float(np.linalg.norm(scale * projected - sino)
                         / np.linalg.norm(sino))
        # Scored exactly as compute_final_metrics.py scores the baseline: least
        # squares amplitude match, then the same four metrics from the same
        # module.  MSE is added because Chapter 9 reports it.
        ls = float((volume * ref).sum() / max((volume * volume).sum(), 1e-30))
        matched = volume * ls
        rows.append({"arm": args.arm, "k": args.k, "config": name,
                     "rek": str(rek), "fractional_data_residual": residual,
                     "npy": str(path), "sha256": digest(path),
                     "ls_scale": ls,
                     "mse": float(np.mean((matched - ref) ** 2)),
                     "psnr": float(M.psnr(matched, ref)),
                     "ssim": float(M.ssim(matched, ref)),
                     "nrmse": float(M.nrmse(matched, ref)),
                     "hfen": float(M.hfen(matched, ref)),
                     "amplitude_excess": float(np.percentile(volume, 99.9) / material)})
        r = rows[-1]
        log(f"  {name:16s} residual {residual:.5f}  LS {ls:.3f}  "
            f"PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}  "
            f"NRMSE {r['nrmse']:.4f}  HFEN {r['hfen']:.2f}  "
            f"amplitude {r['amplitude_excess']:.2f} rho")

    payload = {"dataset": "camera", "cell": cell, "arm": args.arm, "k": args.k,
               "views": n_views, "detector_bin": rec.DETECTOR_BIN,
               "detector_shape": [det_u, det_v], "du_mm": du,
               "voxel_mm": voxel_mm, "volume_shape": list(shape),
               "geometry_frame": ("native" if args.arm == "circular"
                                  else {"yaw_deg": sweep.YAW, "roll_deg": sweep.ROLL,
                                        "translation_mm": sweep.SCAN_T[
                                            f"{args.arm}_N0{args.k:03d}"].tolist()}),
               "material_level": material, "ceiling_factor": CEILING_FACTOR,
               "ceiling": ceiling, "epochs_cap": args.epochs,
               "run_namespace": args.namespace,
               "objective": {"name": "mean_projection_mse",
                             "projection_ssim_weight": 0.0},
               "geometry_source": "regularization_sweep.load_arm",
               "hull": hull_info, "rows": rows}
    out = RESULTS / f"camera_bgnr_{cell}{suffix}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    artifacts = [out]
    for row in rows:
        artifacts.extend((Path(row["npy"]), Path(row["rek"])))
    manifest = {
        "schema_version": "bgnr-run-manifest-v1",
        "dataset": "camera",
        "trajectory": args.arm,
        "view_count": int(n_views),
        "support": {"source": "relaxed silhouette hull intersected with geometry FOV",
                    "silhouette_sigma_factor": SIGMA_FACTOR,
                    "carving_tolerance": TAU},
        "amplitude_bound": {"factor": CEILING_FACTOR,
                            "material_estimator": "same-cell ASD-POCS p99.5",
                            "conditional_single_material_prior": True},
        "optimizer": {"name": "sensitivity-preconditioned AdamW",
                      "objective": "mean_projection_mse",
                      "projection_ssim_weight": 0.0,
                      "learning_rate": 1e-2,
                      "weight_decay": 1e-2,
                      "betas": [0.9, 0.999],
                      "epochs_cap": args.epochs},
        "stopping": {"type": "early_stop_after_epoch", "after_epoch": 200,
                     "patience": 10, "improvement_ratio": 0.98},
        "backend": {"package": "diffct_mlx",
                    "version": getattr(dct, "__version__", "unknown"),
                    "array_backend": "torch", "torch_version": torch.__version__,
                    "projector_mode": "footprint"},
        "hardware": {"device": torch.cuda.get_device_name(0)},
        "runtime_seconds": time.time() - started,
        "run_namespace": args.namespace,
        "outputs": [{"path": str(path), "bytes": path.stat().st_size,
                     "sha256": digest(path)} for path in artifacts],
    }
    manifest_path = RESULTS / f"camera_bgnr_{cell}{suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out}")
    log(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
