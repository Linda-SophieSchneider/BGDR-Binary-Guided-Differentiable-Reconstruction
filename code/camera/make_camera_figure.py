"""Render the camera comparison figure from the reconstructions of one cell.

The camera table reports four numbers per configuration and none of them says
where the difference sits in the volume. This shows the central axial slice of the
dense reference, the ASD-POCS baseline of the same cell, and the two BGDR
configurations, on one grey-value window taken from the reference, with a line
profile through all four underneath.

The window runs from zero to the reference's own 99.5th percentile, the same
convention the engine figure uses, so the panels here can be read next to it.
Each reconstruction is amplitude matched to the reference first, exactly as
the metrics are, otherwise the panels would differ by a global factor of about two
and the comparison would be about scaling rather than structure.

One row of panels per view budget, so the budget dependence the table reports is
visible rather than asserted: the shell the constrained reconstruction places
between the object's outline and the hull's is thinner at the larger budget.

    python make_camera_figure.py --arm bundle --k 100 400 --namespace mseonly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

R = Path("/home/schneider/TrajectoryOptimization/RealWorldExample_ConOpt")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from EZRT_Helpers.rek2py import rek2py

OUT = HERE.parent / "reconstructions"
RESULTS = HERE.parent / "results"
REFERENCE = (R / "reference_reconstructions" / "output_circular1200_fdk_quant"
             / "reconstruction_FDK.rek")
ARM_LABEL = {"circular": "circular subset", "bundle": "planned bundle",
             "all3": "planned full composite"}


def load(path: Path) -> np.ndarray:
    _, v = rek2py(str(path), switch_order=True)
    return np.asarray(v, np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=["circular", "bundle", "all3"])
    parser.add_argument("--k", type=int, nargs="+", required=True,
                        choices=[100, 400])
    parser.add_argument("--namespace", default="",
                        help="optional reconstruction protocol suffix")
    args = parser.parse_args()

    ref = load(REFERENCE)
    mid = ref.shape[0] // 2
    lo, hi = 0.0, float(np.percentile(ref[mid], 99.5))
    row = ref.shape[1] // 2

    budgets = []
    suffix = f"_{args.namespace}" if args.namespace else ""
    for k in args.k:
        cell = f"{args.arm}_k{k:04d}"
        panels = [(ref, "1200-view pseudo-reference")]
        for path, label in (
                (R / "results_final" / f"{args.arm}_final_k{k:04d}" / "reconstruction.rek",
                 "ASD-POCS"),
                (OUT / f"camera_{cell}_bgnr_hull{suffix}.rek", "BGDR, silhouette hull"),
                (OUT / f"camera_{cell}_bgnr_hull_box{suffix}.rek", "BGDR, hull + box")):
            if not path.exists():
                print(f"missing, skipped: {path.name}")
                continue
            volume = load(path)
            # Amplitude match, as the metrics do, so the panels compare structure.
            scale = float((volume * ref).sum() / max((volume * volume).sum(), 1e-30))
            panels.append((volume * scale, label))
        budgets.append((k, panels))

    columns = max(len(p) for _, p in budgets)
    rows_per_budget = 2
    fig = plt.figure(figsize=(3.3 * columns, 4.8 * len(budgets)))
    gs = fig.add_gridspec(rows_per_budget * len(budgets), columns,
                          height_ratios=[3.0, 1.25] * len(budgets),
                          hspace=0.55, wspace=0.04, left=0.06)
    for b, (k, panels) in enumerate(budgets):
        profile = fig.add_subplot(gs[2 * b + 1, :])
        row_axes = []
        for index, (volume, label) in enumerate(panels):
            ax = fig.add_subplot(gs[2 * b, index])
            ax.imshow(volume[mid], cmap="gray", vmin=lo, vmax=hi)
            ax.axhline(row, color="tab:red", lw=0.6, alpha=0.65)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            row_axes.append(ax)
            profile.plot(volume[mid, row], lw=0.9,
                         label=label,
                         color="black" if index == 0 else None,
                         ls="-" if index == 0 else "--" if index == 1 else "-")
        left_pos = row_axes[0].get_position()
        fig.text(left_pos.x0 - 0.025, (left_pos.y0 + left_pos.y1) / 2,
                  f"$k = {k}$", rotation=90, ha="center", va="center",
                  fontsize=12)
        profile.set_xlim(0, ref.shape[2] - 1)
        profile.set_xlabel("Position along profile (voxel)", fontsize=9)
        profile.set_ylabel("Attenuation (a.u.)", fontsize=9)
        profile.tick_params(labelsize=8)
        profile.legend(fontsize=8, ncol=columns, loc="upper center",
                       frameon=False)
    # No figure-level title: it sat far above the first row of panels and left a
    # band of white space that the caption's first words already cover.
    tag = "_".join(f"k{k:04d}" for k in args.k) + suffix
    path = RESULTS / f"camera_comparison_{args.arm}_{tag}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
