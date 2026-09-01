"""Create diagnostic and publication figures for the Defrise experiment.

Reads the seed-0 reconstruction volumes written by ``run_defrise_bgnr.py``
for the image panels; quantitative paper claims use the three-seed aggregate
JSON in ``results/defrise_bgnr/defrise_results.json`` instead, see
``make_defrise_table.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# The phantom is not distributed with this repository, see data/README.md.
DEFAULT_PHANTOM = REPO_ROOT / "data" / "lof_flange_v3.npy"
DEFAULT_RECON_DIR = (
    REPO_ROOT / "results" / "defrise_bgnr" / "reconstructions"
    / "wedge120_k0060_i10000_n0"
)


def load_volume(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)


def amplitude_match(volume: np.ndarray, reference: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float64)[mask]
    target = np.asarray(reference, dtype=np.float64)[mask]
    denominator = float(np.dot(values, values))
    scale = float(np.dot(values, target) / denominator) if denominator > 0 else 0.0
    return (np.asarray(volume, dtype=np.float32) * scale).astype(np.float32)


def central_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        image = volume[index, :, :]
    elif axis == 1:
        image = volume[:, index, :]
    elif axis == 2:
        image = volume[:, :, index]
    else:
        raise ValueError(axis)
    return np.flipud(np.asarray(image))


def save_diagnostic(volumes: dict[str, np.ndarray], output: Path) -> None:
    axes = (0, 1, 2)
    offsets = (-48, 0, 48)
    names = list(volumes)
    fig, axs = plt.subplots(
        len(axes) * len(offsets), len(names), figsize=(2.2 * len(names), 16.0)
    )
    for row, (axis, offset) in enumerate(
        (pair for axis in axes for pair in [(axis, off) for off in offsets])
    ):
        index = volumes[names[0]].shape[axis] // 2 + offset
        for col, name in enumerate(names):
            ax = axs[row, col]
            ax.imshow(
                central_slice(volumes[name], axis, index),
                cmap="gray",
                vmin=0.0,
                vmax=0.24,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"axis {axis}, slice {index}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_publication(volumes: dict[str, np.ndarray], output: Path) -> None:
    labels = [
        ("Reference", "reference"),
        ("ASD-POCS", "asdpocs"),
        ("BGDR\nDART support", "bgdr_dart_support"),
        ("BGDR\nsilhouette hull", "bgdr_carved_hull"),
        ("BGDR\nhull + box", "bgdr_carved_hull_box"),
        ("BGDR\noracle support", "bgdr_oracle_support"),
    ]
    axis = 1
    index = volumes["reference"].shape[axis] // 2
    ref = central_slice(volumes["reference"], axis, index)
    support = ref > 0
    yy, xx = np.nonzero(support)
    margin = 18
    y0, y1 = max(int(yy.min()) - margin, 0), min(int(yy.max()) + margin + 1, ref.shape[0])
    x0, x1 = max(int(xx.min()) - margin, 0), min(int(xx.max()) + margin + 1, ref.shape[1])
    crop = np.s_[y0:y1, x0:x1]
    ref = ref[crop]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    fig, axs = plt.subplots(2, len(labels), figsize=(6.9, 2.55))
    for col, (title, key) in enumerate(labels):
        image = central_slice(volumes[key], axis, index)
        image = image[crop]
        ax = axs[0, col]
        ax.imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, pad=3)
        err = np.abs(image - ref)
        axs[1, col].imshow(
            err,
            cmap="magma",
            vmin=0.0,
            vmax=0.5,
            interpolation="nearest",
        )
        for row in range(2):
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            for spine in axs[row, col].spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("0.35")
    scale_voxels = 20.0 / 0.3
    bar_y = ref.shape[0] - 9
    axs[0, 0].plot([8, 8 + scale_voxels], [bar_y, bar_y], color="white",
                   linewidth=2.0, solid_capstyle="butt")
    axs[0, 0].text(8 + scale_voxels / 2, bar_y - 5, "20 mm", color="white",
                   fontsize=6.5, ha="center", va="bottom")
    axs[0, 0].set_ylabel("attenuation")
    axs[1, 0].set_ylabel("absolute error")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.88, bottom=0.015,
                        wspace=0.035, hspace=0.06)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phantom", type=Path, default=DEFAULT_PHANTOM)
    parser.add_argument("--recon-dir", type=Path, default=DEFAULT_RECON_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.phantom.exists():
        raise SystemExit(
            f"phantom not found at {args.phantom}, see data/README.md"
        )

    reference_physical = load_volume(args.phantom)
    reference_max = float(reference_physical.max())
    reference = (reference_physical / reference_max).astype(np.float32)
    evaluation_mask = ndimage.binary_dilation(reference > 0, iterations=12)
    volumes = {"reference": reference}
    for name in (
        "asdpocs",
        "bgdr_dart_support",
        "bgdr_carved_hull",
        "bgdr_carved_hull_box",
        "bgdr_oracle_support",
    ):
        raw = load_volume(args.recon_dir / f"{name}.npy")
        volumes[name] = amplitude_match(raw, reference, evaluation_mask)

    if args.diagnostic:
        save_diagnostic(volumes, args.output_dir / "defrise_slice_diagnostic.png")
    save_publication(volumes, args.output_dir / "defrise_comparison")


if __name__ == "__main__":
    main()
