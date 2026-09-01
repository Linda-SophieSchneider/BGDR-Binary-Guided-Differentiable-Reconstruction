"""Score every camera reconstruction -- BGNR and the ASD-POCS baseline -- on one
protocol, and record where that protocol differs from Chapter 9's.

The camera experiment's own ``compute_final_metrics.py`` is the authority for the
baseline column, so its four metrics are reproduced from the same module and must
agree with its committed CSV; this script checks that they do.  Two of them,
though, are on conventions Chapter 9 does not use, and reading the numbers
without knowing that invites a false comparison across objects:

*   ``psnr`` and ``ssim`` take their peak and data range from the reference's own
    span rather than from 1, so both are offset -- and SSIM is not merely offset,
    since the stabilising constants scale with the range.
*   ``hfen`` is the *unnormalised* norm of the Laplacian-of-Gaussian residual,
    which is why the camera baseline scores 12--33 where the engine scores 0.8.
    Chapter 9 divides by the reference's own high-frequency norm, so that 1 is
    the score of a reconstruction with no high-frequency content and anything
    above 1 means manufactured edges.  Both are reported here: ``hfen`` for
    continuity with the camera experiment, ``hfen_norm`` for continuity with
    Chapter 9.

MSE is added because Chapter 9 reports it, and the fractional data residual
because it is the only reference-free column and the one BGNR is selected on.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

CAMERA_REPO = Path("/home/schneider/TrajectoryOptimization/RealWorldExample_ConOpt")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMERA_REPO))
sys.path.insert(0, str(CAMERA_REPO.parent / "Differentiable-Coverage"))

from EZRT_Helpers.rek2py import rek2py                   # noqa: E402
from differentiable_coverage.eval import metrics as M    # noqa: E402
from scipy.ndimage import gaussian_laplace               # noqa: E402
import metrics_torch as RM                               # noqa: E402

OUT = HERE.parent / "reconstructions"
RESULTS = HERE.parent / "results"
LOGS = HERE.parent / "logs"
REFERENCE = (CAMERA_REPO / "reference_reconstructions" /
             "output_circular1200_fdk_quant" / "reconstruction_FDK.rek")
CELLS = [(arm, k) for arm in ("circular", "bundle", "all3") for k in (100, 400)]
CONFIGS = ("bgnr_hull", "bgnr_hull_box")


def material_spread(volume: np.ndarray, label: str, log=print) -> dict:
    """How far the attenuation reaches above its own bulk level.

    The amplitude box assumes a single material: it clips at 1.5 times the level
    the prior estimates, which is licensed on the engine because that reference's
    material spans only 1.12 times its own bulk value. This measures the same
    quantity here, and it is measurable on the algebraic reconstruction alone --
    no reference volume -- so it is a precondition the method can check before
    deciding to apply the box, rather than a hope.
    """
    from engine_cuda import otsu_threshold
    threshold = otsu_threshold(volume[::8].ravel())
    material = volume[volume > threshold]
    bulk = float(np.median(material))
    rho = float(np.percentile(volume, 99.5))
    ceiling = 1.5 * rho
    clipped = material > ceiling
    info = {"otsu": float(threshold), "material_share": float(material.size / volume.size),
            "bulk": bulk, "rho": rho, "ceiling": ceiling,
            "p999_over_bulk": float(np.percentile(material, 99.9) / bulk),
            "max_over_bulk": float(material.max() / bulk),
            "rho_over_bulk": rho / bulk,
            "clipped_voxel_share": float(clipped.sum() / material.size),
            "clipped_mass_share": float(material[clipped].sum() / material.sum())}
    log(f"{label}: material {100 * info['material_share']:.2f} %, "
        f"p99.9/bulk {info['p999_over_bulk']:.2f}, max/bulk {info['max_over_bulk']:.2f}; "
        f"the box would clip {100 * info['clipped_voxel_share']:.2f} % of material "
        f"voxels holding {100 * info['clipped_mass_share']:.2f} % of the mass")
    return info


def epochs_from_log(arm: str, k: int) -> dict[str, int]:
    """Epochs per configuration, recovered from the run log.

    The run script records the stopping epoch only in its log, and it was not
    edited while the reconstructions were in flight -- editing an imported module
    mid-run is how four engine rows once came to be reconstructed with the wrong
    projector.  The log lines appear in the order the configurations are run, so
    they pair with CONFIGS positionally.
    """
    path = LOGS / f"camera_{arm}_k{k}.log"
    if not path.exists():
        return {}
    found = re.findall(r"^\s+(\d+) epochs, loss", path.read_text(), re.M)
    return {config: int(n) for config, n in zip(CONFIGS, found)}


def load(path: Path) -> np.ndarray:
    _, v = rek2py(str(path), switch_order=True)
    return np.asarray(v, np.float32)


def score(volume: np.ndarray, ref: np.ndarray, log_ref_norm: float) -> dict:
    """The camera protocol, plus the two columns Chapter 9 needs."""
    ls = float((volume * ref).sum() / max((volume * volume).sum(), 1e-30))
    a = volume * ls
    hfen = float(M.hfen(a, ref))   # one Laplacian-of-Gaussian pass, reused below
    return {"ls_scale": ls,
            "mse": float(np.mean((a - ref) ** 2)),
            "psnr": float(M.psnr(a, ref)),
            "ssim": float(M.ssim(a, ref)),
            "nrmse": float(M.nrmse(a, ref)),
            "hfen": hfen,
            "hfen_norm": hfen / log_ref_norm}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="",
                        help="score BGDR outputs carrying this protocol stamp")
    args = parser.parse_args()
    suffix = f"_{args.namespace}" if args.namespace else ""
    ref = load(REFERENCE)
    roi_reference = RM.normalize_reference(ref)
    fixed_rois = RM.reference_rois(
        roi_reference, dilation_voxels=12, boundary_width_voxels=3
    )
    log_ref_norm = float(np.linalg.norm(gaussian_laplace(ref, sigma=1.5)))
    print(f"reference {ref.shape}, span {ref.min():.4f}..{ref.max():.4f}, "
          f"high-frequency norm {log_ref_norm:.4f}")

    audit = {"reference": material_spread(ref, "reference (1200-view FDK)")}

    rows = []
    for arm, k in CELLS:
        sart_path = OUT / f"camera_{arm}_k{k:04d}_sart.rek"
        if sart_path.exists():
            volume = load(sart_path)
            row = {"arm": arm, "k": k, "config": "sart",
                   "source": sart_path.name}
            row.update(score(volume, ref, log_ref_norm))
            row["fixed_roi"] = RM.evaluate_rois(
                volume, roi_reference, rois=fixed_rois,
                dilation_voxels=12, boundary_width_voxels=3
            )
            manifest = RESULTS / f"camera_{arm}_k{k:04d}_sart_manifest.json"
            if manifest.exists():
                row["manifest"] = manifest.name
            rows.append(row)
            del volume
        else:
            print(f"  missing: {sart_path.name}")

        base = CAMERA_REPO / "results_final" / f"{arm}_final_k{k:04d}" / "reconstruction.rek"
        if base.exists():
            volume = load(base)
            audit[f"asdpocs_{arm}_k{k:04d}"] = material_spread(
                volume, f"ASD-POCS {arm} k={k}")
            row = {"arm": arm, "k": k, "config": "asdpocs",
                   "source": str(base.relative_to(CAMERA_REPO))}
            row.update(score(volume, ref, log_ref_norm))
            row["fixed_roi"] = RM.evaluate_rois(
                volume, roi_reference, rois=fixed_rois,
                dilation_voxels=12, boundary_width_voxels=3
            )
            rows.append(row)
            del volume
        for config in CONFIGS:
            path = OUT / f"camera_{arm}_k{k:04d}_{config}{suffix}.rek"
            if not path.exists():
                print(f"  missing: {path.name}")
                continue
            row = {"arm": arm, "k": k, "config": config, "source": path.name}
            epochs = epochs_from_log(arm, k).get(config)
            if epochs:
                row["epochs"] = epochs
            volume = load(path)
            row.update(score(volume, ref, log_ref_norm))
            row["fixed_roi"] = RM.evaluate_rois(
                volume, roi_reference, rois=fixed_rois,
                dilation_voxels=12, boundary_width_voxels=3
            )
            del volume
            cell = RESULTS / f"camera_bgnr_{arm}_k{k:04d}{suffix}.json"
            if cell.exists():
                payload = json.loads(cell.read_text())
                match = [r for r in payload["rows"] if r["config"] == config]
                if match:
                    row["fractional_data_residual"] = match[0]["fractional_data_residual"]
                    row["amplitude_excess"] = match[0]["amplitude_excess"]
                row["reference_support_contained"] = \
                    payload["hull"]["reference_support_contained"]
                row["support_occupancy"] = payload["hull"]["support_occupancy"]
            rows.append(row)

    # The baseline column must reproduce the camera experiment's committed CSV.
    committed = CAMERA_REPO / "results_final" / "final_metrics_20260811.csv"
    if committed.exists():
        published = {(r["arm"], int(r["k"])): r
                     for r in csv.DictReader(committed.open())}
        print("\ncross-check against final_metrics_20260811.csv:")
        worst = 0.0
        for row in rows:
            if row["config"] != "asdpocs":
                continue
            want = published.get((row["arm"], row["k"]))
            if not want:
                continue
            for key in ("ls_scale", "psnr", "ssim", "nrmse", "hfen"):
                delta = abs(row[key] - float(want[key]))
                worst = max(worst, delta / max(abs(float(want[key])), 1e-9))
            print(f"  {row['arm']:9s} k={row['k']:3d}: "
                  f"PSNR {row['psnr']:.3f} vs {float(want['psnr']):.3f}, "
                  f"SSIM {row['ssim']:.4f} vs {float(want['ssim']):.4f}")
        print(f"  worst relative disagreement {worst:.2e}"
              f"  {'OK' if worst < 1e-3 else 'MISMATCH'}")

    print(f"\n{'cell':16s} {'config':15s} {'MSE':>9s} {'SSIM':>7s} {'PSNR':>7s} "
          f"{'NRMSE':>7s} {'HFENn':>7s} {'resid':>7s}")
    for row in rows:
        print(f"{row['arm'] + ' k=' + str(row['k']):16s} {row['config']:15s} "
              f"{row['mse']:9.3e} {row['ssim']:7.4f} {row['psnr']:7.2f} "
              f"{row['nrmse']:7.4f} {row['hfen_norm']:7.4f} "
              f"{row.get('fractional_data_residual', float('nan')):7.4f}")

    out = RESULTS / f"camera_results{suffix}.json"
    out.write_text(json.dumps(
        {"reference": str(REFERENCE), "reference_span": [float(ref.min()), float(ref.max())],
         "log_reference_norm": log_ref_norm,
         "protocol": {
             "primary": "fixed-reference-roi-v1",
             "roi_source": "dense_reference_only",
             "dilation_voxels": 12,
             "boundary_width_voxels": 3,
             "secondary_whole_volume":
                 "differentiable_coverage.eval.metrics, as compute_final_metrics.py",
         },
         "run_namespace": args.namespace,
         "material_audit": audit,
         "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
