"""Recompute the whole engine table from one code state, on CUDA.

Why this exists: the BGNR rows of the published engine table were produced before
an edit to ``bgnr.py`` on 2026-08-24 22:50 and the carved-support rows after it,
and the two states do not agree -- the stored dense-support row sits below the
minimum of the whole trajectory today's code produces, so no epoch count
reproduces it and the earlier state cannot be recovered.  The fix is not
archaeology but a single consistent recomputation, which is what this does.

The pipeline mirrors ``Research/BGNR/eval/run_engine.py`` stage for stage,
including the reuse of the coarse DART mask at full resolution, because that is
what both documents describe.  The three deterministic rows are checked against
the Mac table by ``check_deterministic.py``; they must not move.

    python run_engine_cuda.py --bin 1 --tag bin1 --reuse-tag bin2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import engine_cuda as E
import engine_data as ed
import metrics_torch as M
from bgnr_torch import BGNRConfig, masked_backprojection_init, reconstruct_bgnr

RESULTS = Path(__file__).resolve().parent.parent / "results"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()

LABELS = {
    "fdk": "FDK",
    "sart": "SART",
    "dart": "DART",
    "asdpocs": "ASD-POCS",
    "bgnr_init": "Prior-informed initialization",
    "bgnr_support": "Support-constrained only",
    "bgnr_full": "Full BGNR",
    "bgnr_full_dilated": "Full BGNR, dilated support",
    "bgnr_full_oracle": "Full BGNR, dense-data support",
    "bgnr_hull": "Full BGNR, data-carved hull",
    "bgnr_carved": "Full BGNR, carved margin",
    "bgnr_box": "Full BGNR, dilated support + amplitude box",
    "bgnr_hull_box": "Carved hull + amplitude box",
    "bgnr_carved_box": "Carved margin + amplitude box",
    "bgnr_two_level": "Hull hard / DART soft + amplitude box",
    "bgnr_oracle_box": "Full BGNR, dense-data support + box",
}


def run_bgnr(setup: E.Setup, support: np.ndarray, *, prior_init: bool,
             use_support: bool, epochs: int, ceiling: float = 0.0,
             soft: np.ndarray | None = None, soft_decay: float = 0.0) -> np.ndarray:
    support_t = E.as_tensor(support.astype(np.float32))
    soft_t = E.as_tensor(soft.astype(np.float32)) if soft is not None else None
    if prior_init:
        init = masked_backprojection_init(setup.sino, setup.forward_all,
                                          setup.backward_all, support_t,
                                          setup.volume_shape)
    else:
        init = torch.zeros(setup.volume_shape, dtype=torch.float32,
                           device=E.DEVICE)
    volume, history = reconstruct_bgnr(
        setup.sino, setup.forward_all, setup.backward_all,
        initial_volume=init, support_mask=support_t if use_support else None,
        config=BGNRConfig(epochs=epochs, value_ceiling=ceiling,
                          soft_decay=soft_decay),
        soft_mask=soft_t,
        progress_callback=lambda e, l: E.log(f"    epoch {e:4d}  loss {l:.6e}"))
    E.log(f"    stopped after {len(history)} epochs, loss {history[-1]:.6e}")
    return volume.detach().cpu().numpy()


def select_support_mask(setup: E.Setup, dart_volume: np.ndarray,
                        probe_epochs: int = 60) -> tuple[np.ndarray, dict]:
    """The reference-free delta rule: smallest support that costs no data consistency."""
    threshold = E.otsu_threshold(dart_volume.ravel())
    material = float(np.percentile(dart_volume, 99.9))
    footprint = setup.sino_np > 0.05 * setup.sino_np.max(axis=(1, 2), keepdims=True)
    report = []
    for ratio in (0.0, 0.02, 0.05, 0.10, 0.20):
        delta = ratio * material
        mask = dart_volume >= threshold - delta
        mask_t = E.as_tensor(mask.astype(np.float32))
        projected = setup.forward_all(mask_t).detach().cpu().numpy()
        coverage = float((projected[footprint] > 0).mean())
        init = masked_backprojection_init(setup.sino, setup.forward_all,
                                         setup.backward_all, mask_t,
                                         setup.volume_shape)
        probe, _ = reconstruct_bgnr(
            setup.sino, setup.forward_all, setup.backward_all,
            initial_volume=init, support_mask=mask_t,
            config=BGNRConfig(epochs=probe_epochs,
                              early_stop_after_epoch=10 ** 9))
        residual = setup.residual(probe)
        report.append({"ratio": ratio, "delta": delta,
                       "occupancy": float(mask.mean()), "coverage": coverage,
                       "residual": residual})
        E.log(f"  delta = {ratio:.2f} * rho: occupancy {mask.mean():.4f}  "
              f"coverage {coverage:.5f}  probe residual {residual:.5f}")
    best = min(r["residual"] for r in report)
    chosen = next(r for r in report if r["residual"] <= 1.01 * best)
    mask = dart_volume >= threshold - chosen["delta"]
    info = {"otsu_threshold": threshold, "material_level": material,
            "delta": chosen["delta"], "delta_ratio": chosen["ratio"],
            "coverage": chosen["coverage"], "occupancy": chosen["occupancy"],
            "probe_residual": chosen["residual"], "probe_epochs": probe_epochs,
            "sweep": report}
    return mask, info


def _crop_to(volume: np.ndarray, shape) -> np.ndarray:
    return volume[:shape[0], :shape[1], :shape[2]]


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=int, default=1)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--views", type=int, default=ed.N_VIEWS_FEW)
    parser.add_argument("--sart-iters", type=int, default=30)
    parser.add_argument("--dart-iters", type=int, default=20)
    parser.add_argument("--asd-iters", type=int, default=100)
    parser.add_argument("--asd-reg-iters", type=int, default=10)
    parser.add_argument("--bgnr-epochs", type=int, default=500)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument("--alpha-sweep", type=float, nargs="*",
                        default=[0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--ceiling-factor", type=float, default=1.5)
    parser.add_argument("--soft-decay", type=float, default=0.01)
    parser.add_argument("--reuse-tag", default=None,
                        help="take the DART mask and the alpha sweep from a "
                             "coarser run and upsample the mask")
    parser.add_argument("--reuse-bin", type=int, default=None)
    args = parser.parse_args()
    tag = args.tag or f"bin{args.bin}"
    RESULTS.mkdir(parents=True, exist_ok=True)

    setup = E.Setup(args.bin, args.views, tag)
    reference = M.normalize_reference(ed.load_reference(args.bin))
    E.log(f"reference {reference.shape}, angular range "
          f"{ed.angular_range_deg(args.views):.1f} deg")

    volumes: dict[str, np.ndarray] = {}
    volumes["fdk"] = E.cached(setup, "fdk", lambda: E.run_fdk(setup))
    volumes["sart"] = E.cached(setup, "sart",
                               lambda: E.run_sart(setup, args.sart_iters))
    volumes["dart"] = E.cached(
        setup, "dart", lambda: E.run_dart(setup, volumes["sart"], args.dart_iters)
    )

    # ---- the DART prior and the reference-free delta rule -------------------
    mask_path = setup.path("mask")
    info_path = RESULTS / f"mask_info_{tag}.json"
    if args.reuse_tag and not mask_path.exists():
        factor = (args.reuse_bin or 2 * args.bin) // args.bin
        coarse = np.load(E.OUT / f"mask_{args.reuse_tag}.npy")
        mask = _crop_to(np.kron(coarse, np.ones((factor,) * 3, dtype=bool)),
                        setup.volume_shape)
        mask_info = json.loads((RESULTS / f"mask_info_{args.reuse_tag}.json").read_text())
        mask_info.update({"reused_from": args.reuse_tag, "upsample_factor": factor})
        np.save(mask_path, mask)
        info_path.write_text(json.dumps(mask_info, indent=2))
        E.log(f"mask: upsampled {factor}x from {args.reuse_tag}")
    if mask_path.exists() and info_path.exists():
        mask = np.load(mask_path)
        mask_info = json.loads(info_path.read_text())
        E.log("mask: cached")
    else:
        dart = volumes["dart"]
        E.log("selecting the support mask (reference-free delta rule)")
        mask, mask_info = select_support_mask(setup, dart)
        np.save(mask_path, mask)
        info_path.write_text(json.dumps(mask_info, indent=2))
    E.log(f"mask: occupancy {mask.mean():.4f}, "
          f"delta = {mask_info['delta_ratio']:.2f} * rho")

    dense = ed.load_ezrt_binary_mask(args.bin)
    dice = 2 * (mask & dense).sum() / (mask.sum() + dense.sum())
    contained = (dense & mask).sum() / dense.sum()
    dilated = E.dilate(mask, args.mask_dilation)
    mask_info.update({
        "ezrt_occupancy": float(dense.mean()),
        "dice_vs_ezrt_binary": float(dice),
        "ezrt_support_contained": float(contained),
        "object_extent_mm": list(ed.object_extent_mm(dense, setup.voxel)),
        "dart_mask_bbox_mm": list(ed.object_extent_mm(mask, setup.voxel)),
        "dilated": {"iterations": args.mask_dilation,
                    "occupancy": float(dilated.mean()),
                    "ezrt_support_contained":
                        float((dense & dilated).sum() / dense.sum())}})
    E.log(f"mask vs uncoll_bin.rek: Dice {dice:.4f}, contained {contained:.4f}")

    # ---- ASD-POCS ----------------------------------------------------------
    first_view = setup.fp_view(E.as_tensor(volumes["sart"]), 0)
    epsilon = float(np.linalg.norm(first_view.detach().cpu().numpy()
                                  - setup.sino_np[0]))
    E.log(f"asdpocs: epsilon_min = {epsilon:.1f}")
    sweep_path = RESULTS / f"alpha_sweep_{tag}.json"
    if args.reuse_tag and not sweep_path.exists():
        reused = RESULTS / f"alpha_sweep_{args.reuse_tag}.json"
        if reused.exists():
            sweep_path.write_text(reused.read_text())
            E.log(f"asdpocs: reusing the alpha sweep of {args.reuse_tag}")
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
    else:
        sweep = []
        for alpha in args.alpha_sweep:
            probe = E.run_asd_pocs(setup, alpha, epsilon,
                                   max(10, args.asd_iters // 5), args.asd_reg_iters)
            sweep.append({"alpha": alpha, "residual": setup.residual(probe)})
            E.log(f"  alpha {alpha}: residual {sweep[-1]['residual']:.5f}")
        sweep_path.write_text(json.dumps(sweep, indent=2))
    best_alpha = min(sweep, key=lambda r: r["residual"])["alpha"]
    E.log(f"asdpocs: selected alpha = {best_alpha}")
    volumes["asdpocs"] = E.cached(
        setup, "asdpocs",
        lambda: E.run_asd_pocs(setup, best_alpha, epsilon, args.asd_iters,
                               args.asd_reg_iters))

    # ---- the published configurations --------------------------------------
    for name, prior_init, use_support, support in (
            ("bgnr_init", True, False, mask),
            ("bgnr_support", False, True, mask),
            ("bgnr_full", True, True, mask),
            ("bgnr_full_dilated", True, True, dilated),
            ("bgnr_full_oracle", True, True, dense)):
        E.log(f"{name}: prior_init={prior_init} support={use_support} "
              f"occupancy {support.mean():.4f}")
        volumes[name] = E.cached(setup, name, lambda s=support, p=prior_init,
                                 u=use_support: run_bgnr(
                                     setup, s, prior_init=p, use_support=u,
                                     epochs=args.bgnr_epochs))

    # ---- the repaired support ----------------------------------------------
    hull_path = setup.path("hull")
    if hull_path.exists():
        hull = np.load(hull_path)
        hull_info = json.loads((RESULTS / f"hull_info_{tag}.json").read_text())
        E.log("hull: cached")
    else:
        hull, hull_info = E.carve_hull(setup)
        np.save(hull_path, hull)
        (RESULTS / f"hull_info_{tag}.json").write_text(json.dumps(hull_info, indent=2))
    hull_info.update({
        "ezrt_support_contained": float((dense & hull).sum() / dense.sum()),
        "dart_mask_outside_hull": float((mask & ~hull).sum() / mask.sum())})
    carved = hull & E.dilate(E.close(mask, 2), 4)
    ceiling = args.ceiling_factor * float(mask_info["material_level"])
    E.log(f"amplitude box: {args.ceiling_factor:g} x material = {ceiling:.5f}")

    for name, support, kw in (
            ("bgnr_hull", hull, {}),
            ("bgnr_carved", carved, {}),
            ("bgnr_box", dilated, {"ceiling": ceiling}),
            ("bgnr_hull_box", hull, {"ceiling": ceiling}),
            ("bgnr_carved_box", carved, {"ceiling": ceiling}),
            ("bgnr_two_level", hull, {"ceiling": ceiling, "soft": dilated,
                                      "soft_decay": args.soft_decay}),
            ("bgnr_oracle_box", dense, {"ceiling": ceiling})):
        E.log(f"{name}: occupancy {support.mean():.4f}"
              + (f", box {kw['ceiling']:.5f}" if "ceiling" in kw else ""))
        volumes[name] = E.cached(setup, name, lambda s=support, k=kw: run_bgnr(
            setup, s, prior_init=True, use_support=True,
            epochs=args.bgnr_epochs, **k))

    # ---- metrics -----------------------------------------------------------
    E.log("computing metrics")
    supports = {"bgnr_full": mask, "bgnr_support": mask, "bgnr_init": mask,
                "bgnr_full_dilated": dilated, "bgnr_full_oracle": dense,
                "bgnr_hull": hull, "bgnr_carved": carved, "bgnr_box": dilated,
                "bgnr_hull_box": hull, "bgnr_carved_box": carved,
                "bgnr_two_level": hull, "bgnr_oracle_box": dense}
    rows = []
    for name, label in LABELS.items():
        scores = M.evaluate(volumes[name], reference)
        scores.update({"method": name, "label": label,
                       "residual": setup.residual(volumes[name])})
        if name in supports:
            s = supports[name]
            scores["support_occupancy"] = float(s.mean())
            scores["support_contained"] = float((s & dense).sum() / dense.sum())
        scores["amplitude_excess"] = float(
            np.percentile(volumes[name], 99.9) / mask_info["material_level"])
        rows.append(scores)
        E.log(f"  {label:42s} MSE {scores['mse']:.5f}  SSIM {scores['ssim']:.4f}  "
              f"PSNR {scores['psnr']:.2f}  NRMSE {scores['nrmse']:.4f}  "
              f"HFEN {scores['hfen']:.3f}  res {scores['residual']:.4f}")

    payload = {
        "dataset": "engine", "tag": tag, "detector_bin": args.bin,
        "views": args.views, "angular_range_deg": ed.angular_range_deg(args.views),
        "volume_shape": list(setup.volume_shape), "voxel_mm": setup.voxel,
        "detector_pitch_mm": setup.pitch,
        "platform": "CUDA / torch, one code state (see README)",
        "metric_protocol": (
            "Chapter 7 protocol: reference mapped to air = 0 / material = 1 by "
            "(ref - median) / (p99.9 - median); reconstruction amplitude matched "
            "by a least-squares scale; MSE over all voxels; "
            "PSNR = -10 log10(MSE); 3-D SSIM (Gaussian, sigma 1.5, window 11); "
            "NRMSE relative to the reference RMS"),
        "asdpocs": {"alpha": best_alpha, "epsilon": epsilon,
                    "iterations": args.asd_iters,
                    "reg_iterations": args.asd_reg_iters, "sweep": sweep},
        "mask": mask_info, "hull": hull_info,
        "bgnr": {"learning_rate": 1e-2, "max_epochs": args.bgnr_epochs,
                 "objective": "mean_projection_mse",
                 "projection_ssim_weight": 0.0,
                 "ceiling_factor": args.ceiling_factor,
                 "soft_decay": args.soft_decay},
        "rows": rows,
    }
    out = RESULTS / f"engine_results_{tag}_cuda.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    artifacts = [setup.path(name) for name in LABELS if setup.path(name).exists()]
    artifacts.extend(path for path in (mask_path, hull_path, out) if path.exists())
    manifest = {
        "schema_version": "bgnr-run-manifest-v1",
        "dataset": "engine",
        "trajectory": "limited_arc_circular_diagnostic_geometry",
        "view_count": int(args.views),
        "support": {
            "source": "DART, dense-reference oracle, or silhouette hull by row",
            "silhouette_sigma_factor": 5.0,
            "carving_tolerance": float(hull_info.get("tau", 0.05)),
            "dart_threshold_tolerance_ratio": float(mask_info["delta_ratio"]),
            "morphology": {"dilation_voxels": args.mask_dilation,
                           "carved_closing_voxels": 2,
                           "carved_dilation_voxels": 4},
        },
        "amplitude_bound": {"factor": args.ceiling_factor,
                            "material_estimator": "DART p99.9"},
        "optimizer": {
            "name": "sensitivity-preconditioned AdamW",
            "objective": "mean_projection_mse",
            "projection_ssim_weight": 0.0,
            "learning_rate": 1e-2,
            "weight_decay": 1e-2,
            "betas": [0.9, 0.999],
            "epochs_cap": args.bgnr_epochs,
        },
        "stopping": {"type": "early_stop_after_epoch",
                     "after_epoch": 200, "patience": 10,
                     "improvement_ratio": 0.98},
        "backend": {"package": "diffct_mlx",
                    "version": getattr(E.dct, "__version__", "unknown"),
                    "array_backend": "torch",
                    "torch_version": torch.__version__,
                    "projector_mode": "footprint"},
        "hardware": {"device": torch.cuda.get_device_name(0)},
        "runtime_seconds": time.time() - started,
        "result_file": out.name,
        "outputs": [{"path": str(path), "bytes": path.stat().st_size,
                     "sha256": digest(path)} for path in artifacts],
    }
    manifest_path = RESULTS / f"engine_{tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    E.log(f"wrote {out}")
    E.log(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
