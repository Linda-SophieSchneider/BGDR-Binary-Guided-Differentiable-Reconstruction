# BGDR — Binary-Guided Differentiable Reconstruction

Reconstruction and evaluation code for the paper "Binary-Guided
Differentiable Reconstruction for Few-View and Arbitrary-Trajectory
Industrial CT" (Schneider, Sun, Ye, Maier), submitted to NDT & E
International. A citation will be added once the paper is published.

BGDR restricts direct per-scan voxel optimization to a binary support
derived from the measured projections, without training data or a neural
network. This repository contains the code that produced the paper's
tables and figures. It does not contain the industrial projection data,
which remain subject to third-party approval, or the digital phantom, which
is available from the corresponding author on reasonable request.

## What reproduces out of the box

The Defrise flange is the paper's one fully synthetic, author-designed
experiment. Its derived results are committed under `results/defrise_bgnr/`,
so the table reproduces immediately with no GPU and no additional data.

```bash
pip install -r requirements.txt   # numpy alone is enough for this step
python code/make_defrise_table.py
```

This prints the exact rows of the paper's Table 1 and matches
`tab_defrise_rows.tex` byte for byte.

## Full reproduction from the phantom

Regenerating the sinograms, reconstructions, and the comparison figure needs
the phantom itself and a CUDA GPU. See `data/README.md` for where to place
it. With the phantom in `data/`:

```bash
CUDA_VISIBLE_DEVICES=0 python code/run_defrise_bgnr.py \
  --constraint wedge120 --views 60 --photon-count 10000 \
  --noise-seeds 0 1 2 --grid 384 --output results/defrise_bgnr

python code/make_defrise_table.py --write

python code/make_defrise_figure.py \
  --recon-dir results/defrise_bgnr/reconstructions/wedge120_k0060_i10000_n0 \
  --output-dir results/defrise_bgnr
```

`run_defrise_bgnr.py` caches every stage as a `.npy` file, so a re-run only
computes what is missing. Each of the three noise seeds reconstructs eight
configurations, SART, DART, ASD-POCS, and five BGDR support variants, which
takes on the order of hours on a single GPU.

## Layout

| Path | Contents |
|---|---|
| `code/run_defrise_bgnr.py` | the Defrise experiment end to end, sinogram simulation through all eight reconstructions and their metrics |
| `code/bgnr_torch.py` | the BGDR reconstruction algorithm, sensitivity-preconditioned support-projected AdamW |
| `code/engine_cuda.py` | the SART, DART, and ASD-POCS baselines, dataset-agnostic |
| `code/run_engine_cuda.py` | the DART support selection used by both the Defrise and engine pipelines |
| `code/metrics_torch.py` | the metric protocol, MSE, NRMSE, HFEN, SSIM, and the fractional data residual |
| `code/make_defrise_table.py` | standalone LaTeX table generator, needs only `results/defrise_bgnr/` |
| `code/make_defrise_figure.py` | the reconstruction comparison figure, needs the phantom and the reconstructed volumes |
| `code/engine_data.py`, `code/ezrt_io.py`, `code/score_engine_rois.py` | engine dataset reading and scoring, reference only, see below |
| `code/camera/` | camera dataset reconstruction and scoring, reference only, see below |
| `results/defrise_bgnr/` | the committed derived JSON results and diagnostics behind the paper's Defrise table |
| `data/` | where the phantom goes, not committed |

## Engine and camera code

`code/engine_data.py`, `code/ezrt_io.py`, `code/score_engine_rois.py`, and
everything under `code/camera/` are included for transparency and match the
paper's declared release of its evaluation code, but neither runs
standalone. The engine reader points at the authors' industrial scan
storage, and the camera scripts import `regularization_sweep` and
`differentiable_coverage.eval.metrics` from a companion repository that
carries the raw camera acquisition. Both raw datasets remain subject to
third-party approval and are available from the corresponding author on
reasonable request. `code/run_defrise_bgnr.py` also imports
`select_support_mask` from `run_engine_cuda.py`, which is why that file is
included even though its own driver code is engine-specific.

## Requirements

See `requirements.txt`. The reconstruction and figure scripts need a CUDA
GPU and `diffct-mlx`, the differentiable CT projector package cited in the
paper as version 2.1.0. `make_defrise_table.py` needs only Python and numpy.

## License

Apache License 2.0, see `LICENSE`.
