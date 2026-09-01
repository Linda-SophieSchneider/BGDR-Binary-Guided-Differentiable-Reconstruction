"""The engine (Audi cylinder head) dataset: projections, geometry, reference, mask.

Everything here was verified against the EZRT in-house reconstruction: an FDK of
all 400 views on the geometry below correlates with ``uncoll.rek`` at
NCC = 0.98, and the 400 views were shown to span a full circle (view 200 is the
mirror of view 0, view 399 repeats view 0), i.e. 0.9 deg per view -- the
``angle step 0.45`` field of the EZRT header does not describe the stored set.

The few-view test set of the paper (133 views over 120 deg) is therefore the
first 133 consecutive views, spanning 119.7 deg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ezrt_io import read_projection, read_volume

ENGINE_DIR = Path("/hdd_data/schneider/schneider/Data_External_Harddrive/real_data/AudiZylinderkopf512")

# Geometry (EZRT header, cross-checked by the FDK/reference correlation).
SID_MM = 1489.786
SDD_MM = 1972.67
DETECTOR_PITCH_MM = 0.8
DETECTOR_SHAPE = (512, 512)          # (rows = v, cols = u)
VOXEL_MM = 0.6041704711914062
VOLUME_SHAPE = (460, 510, 510)       # (D, H, W) -- matches uncoll.rek
N_VIEWS_FULL = 400                   # full circle, 0.9 deg per view
N_VIEWS_FEW = 133                    # 119.7 deg  ~  the paper's "133 views / 120 deg"
TUBE_KV = 400.0
TUBE_UA = 2200.0

AIR_MARGIN_PX = 12                   # left/right detector columns that stay in air


def load_projections(n_views: int = N_VIEWS_FEW, *, view_step: int = 1,
                     view_start: int = 0,
                     detector_bin: int = 1) -> np.ndarray:
    """Load and log-transform ``n_views`` projections as a ``(views, u, v)`` sinogram.

    ``I0`` is estimated per view from the left and right detector margins, which
    stay in air for every view; the object is truncated at the top and bottom of
    the detector, so a full-frame percentile would be biased.
    """
    from ezrt_io import bin2d

    views = []
    indices = [(view_start + offset * view_step) % N_VIEWS_FULL
               for offset in range(n_views)]
    for index in indices:
        image = read_projection(ENGINE_DIR / f"uncoll_{index:04d}.raw")
        margins = np.concatenate(
            [image[:, :AIR_MARGIN_PX].ravel(), image[:, -AIR_MARGIN_PX:].ravel()]
        )
        i0 = float(np.median(margins))
        attenuation = -np.log(np.clip(image / i0, 1e-4, 1.0))
        views.append(bin2d(attenuation, detector_bin).astype(np.float32))
    stack = np.stack(views)                                  # (views, v, u)
    return np.ascontiguousarray(stack.transpose(0, 2, 1))    # (views, u, v)


def geometry(n_views: int = N_VIEWS_FEW, *, view_step: int = 1,
             view_start: int = 0):
    """Source/detector vectors of the first ``n_views`` views of the full circle."""
    import diffct_mlx as dct

    src, det_c, det_u, det_v = dct.circular_trajectory_3d(
        N_VIEWS_FULL, sid=SID_MM, sdd=SDD_MM
    )
    take = [(view_start + offset * view_step) % N_VIEWS_FULL
            for offset in range(n_views)]
    return src[take], det_c[take], det_u[take], det_v[take]


def angular_range_deg(n_views: int = N_VIEWS_FEW, *, view_step: int = 1) -> float:
    """Angular span covered by the selected views, in degrees."""
    return 360.0 / N_VIEWS_FULL * view_step * (n_views - 1)


def load_reference(detector_bin: int = 1) -> np.ndarray:
    """EZRT reference reconstruction (400 views, full circle) as ``(D, H, W)`` float32."""
    from ezrt_io import bin3d

    return bin3d(read_volume(ENGINE_DIR / "uncoll.rek").astype(np.float32), detector_bin)


def load_ezrt_binary_mask(detector_bin: int = 1) -> np.ndarray:
    """The binary volume shipped with the dataset (``uncoll_bin.rek``), as a bool mask.

    Kept only as an external cross-check: it is registered to the reference grid
    and therefore cannot be used as a prior without leaking the reference into
    the reconstruction.  The prior used in the experiments is the DART mask
    computed from the few-view data itself.
    """
    from ezrt_io import bin3d

    binary = read_volume(ENGINE_DIR / "uncoll_bin.rek").astype(np.float32)
    return bin3d(binary, detector_bin) > 0.5


def object_extent_mm(mask: np.ndarray, voxel_mm: float = VOXEL_MM) -> tuple[float, float, float]:
    """Bounding-box extent (W, H, D) in mm of a boolean support mask."""
    idx_d, idx_h, idx_w = np.nonzero(mask)
    span = lambda a: (a.max() - a.min() + 1) * voxel_mm
    return span(idx_w), span(idx_h), span(idx_d)
