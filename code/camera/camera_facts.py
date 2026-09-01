import sys, json
from pathlib import Path
R = Path("/home/schneider/TrajectoryOptimization/RealWorldExample_ConOpt")
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent / "Differentiable-Coverage"))
import numpy as np
import reconstruct_ezrt_cuda as rec
import regularization_sweep as sweep
from EZRT_Helpers.rek2py import rek2py

out = {}
for arm, k in [("circular", 100), ("circular", 400), ("bundle", 100),
               ("bundle", 400), ("all3", 100), ("all3", 400)]:
    src, det_c, du_v, dv_v, sino, du, dv = sweep.load_arm(arm, k, (768, 768, 768))
    fod = np.linalg.norm(src, axis=1)
    fdd = np.linalg.norm(det_c - src, axis=1)
    d = src / fod[:, None]                      # unit source directions
    polar = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
    azim = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    # widest pairwise angular gap on the unit sphere, as a coverage proxy
    cos = np.clip(d @ d.T, -1, 1)
    np.fill_diagonal(cos, -1.0)
    nn_gap = np.degrees(np.arccos(cos.max(axis=1)))
    out[f"{arm}_k{k}"] = {
        "views": int(sino.shape[0]),
        "fod_mm": [float(fod.mean()), float(fod.std())],
        "fdd_mm": [float(fdd.mean()), float(fdd.std())],
        "du_mm": float(du), "dv_mm": float(dv),
        "detector": [int(sino.shape[1]), int(sino.shape[2])],
        "polar_deg": [float(polar.min()), float(polar.max())],
        "azimuth_span_deg": float(azim.max() - azim.min()),
        "nn_gap_deg": [float(nn_gap.mean()), float(nn_gap.max())],
    }
    print(f"{arm} k={k}: {sino.shape[0]} views, FOD {fod.mean():.1f}+-{fod.std():.2f}, "
          f"FDD {fdd.mean():.1f}+-{fdd.std():.2f}, polar {polar.min():.1f}-{polar.max():.1f} deg, "
          f"azimuth span {azim.max()-azim.min():.1f} deg, "
          f"nn gap mean {nn_gap.mean():.2f} max {nn_gap.max():.2f}", flush=True)
    del sino

_, ref = rek2py(str(sweep.REFERENCE), switch_order=True)
ref = np.asarray(ref, np.float32)
sys.path.insert(0, "/home/schneider/BGNR_Rekonstruktions/code")
from engine_cuda import otsu_threshold
m = ref > otsu_threshold(ref[::8].ravel())
i = np.argwhere(m)
extent = (i.max(0) - i.min(0) + 1) * sweep.VOXEL_MM
out["reference"] = {"shape": list(ref.shape), "voxel_mm": sweep.VOXEL_MM,
                    "span": [float(ref.min()), float(ref.max())],
                    "occupancy": float(m.mean()),
                    "extent_mm": [float(v) for v in extent],
                    "fov_mm": float(768 * sweep.VOXEL_MM)}
print(f"reference: occupancy {m.mean():.4f}, object extent "
      f"{extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm, "
      f"field of view {768*sweep.VOXEL_MM:.1f} mm")
Path("/home/schneider/BGNR_Rekonstruktions/results/camera_facts.json").write_text(
    json.dumps(out, indent=2) + "\n")
print("wrote camera_facts.json")
