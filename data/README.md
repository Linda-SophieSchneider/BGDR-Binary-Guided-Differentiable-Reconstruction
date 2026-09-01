# Defrise phantom

This directory is where `code/run_defrise_bgnr.py` and `code/make_defrise_figure.py`
expect to find the digital phantom used in the paper's controlled experiment.
It is not distributed in this repository.

## Expected files

- `lof_flange_v3.npy` — a cubic 3-D float32 array, native resolution
  384x384x384 at 0.3 mm isotropic voxels, giving the physical attenuation in
  mm^-1. The flange body is aluminium (0.07 mm^-1) with steel inserts
  (0.24 mm^-1), as described in the paper's Experimental Setup section.
- `lof_flange_v3_metadata.json` — accompanying metadata for the array above.

Place both files directly in this directory before running the reconstruction
scripts.

## Obtaining the phantom

The phantom is available from the corresponding author on reasonable
request, the same terms as the industrial projection data described in the
paper's Data and code availability declaration.

## What still works without it

`code/make_defrise_table.py` reproduces the paper's Defrise table
(`tab_defrise_rows.tex`) directly from the derived JSON results already
committed under `results/defrise_bgnr/`, without needing the phantom, a GPU,
or any reconstruction step. The phantom is only required to regenerate the
sinograms and reconstructed volumes from scratch, or to regenerate the
comparison figure, which reads the reconstructed volumes rather than the
summary JSON.
