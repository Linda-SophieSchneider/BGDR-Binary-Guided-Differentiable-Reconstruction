"""Score all stored engine reconstructions in fixed reference-derived ROIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine_data as ed  # noqa: E402
import metrics_torch as M  # noqa: E402

RECONSTRUCTIONS = HERE.parent / "reconstructions"
RESULTS = HERE.parent / "results"


def result_file(name: str) -> Path:
    """Locate historical JSONs without assuming the early cluster layout."""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidates = (RESULTS / name, HERE / name,
                  RESULTS / f"{stem}_cuda{suffix}", HERE / f"{stem}_cuda{suffix}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"could not locate {name} in {[str(p) for p in candidates]}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="bin1")
    parser.add_argument("--bin", type=int, default=1)
    args = parser.parse_args()

    primary = json.loads(result_file(f"engine_results_{args.tag}.json").read_text())
    variants_path = next(
        (path for path in (RESULTS / f"support_variants_{args.tag}.json",
                           HERE / f"support_variants_{args.tag}.json") if path.exists()),
        RESULTS / f"support_variants_{args.tag}.json",
    )
    variants = json.loads(variants_path.read_text()) if variants_path.exists() else {"rows": []}
    methods = []
    for row in primary["rows"] + variants["rows"]:
        method = row["method"]
        if method not in methods:
            methods.append(method)
    if (RECONSTRUCTIONS / f"dart_{args.tag}.npy").exists() and "dart" not in methods:
        methods.insert(2, "dart")

    reference = M.normalize_reference(ed.load_reference(args.bin))
    rois = M.reference_rois(reference, dilation_voxels=12, boundary_width_voxels=3)
    rows = []
    for method in methods:
        path = RECONSTRUCTIONS / f"{method}_{args.tag}.npy"
        if not path.exists():
            print(f"missing {path}")
            continue
        print(f"scoring {method}: {path}", flush=True)
        volume = np.load(path)
        rows.append({
            "method": method,
            "source": str(path),
            "sha256": digest(path),
            "fixed_roi": M.evaluate_rois(
                volume, reference, rois=rois,
                dilation_voxels=12, boundary_width_voxels=3
            ),
        })
        del volume

    output = RESULTS / f"engine_roi_results_{args.tag}.json"
    payload = {
        "schema_version": "fixed-reference-roi-v1",
        "dataset": "engine",
        "reference_source": "engine_data.load_reference",
        "reference_normalization": "air median to zero; p99.9 minus air to one",
        "roi_source": "dense_reference_only",
        "dilation_voxels": 12,
        "boundary_width_voxels": 3,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
