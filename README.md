# Colon straightening and unrolling for CT colonography

Code and result files from an MSc thesis on the geometric fidelity of colon
reconstruction for CT colonography (Bernardo Augusto, Instituto Superior
Técnico, 2026).

The repository contains:

- a direct ray-casting colon unrolling method and a vertex-based colon
  straightening method (`colon_core.py`);
- the scripts used to evaluate both methods, and an existing straightening
  method (SATO), on twelve real CT colonography datasets, with the result
  files they produced (`analysis/`);
- the scripts used to prepare inputs for SATO and to run and analyse it
  (`analysis/sato_batch_run/`). SATO's own code is not included.

It does not contain the CT scans, the segmentation masks or the centrelines.
See [Data](#data).

## What the methods do

**Unrolling.** A rotation-minimising frame is propagated along the colon
centreline. At every 1 mm of arc length, 360 rays are cast outward in the
plane perpendicular to the centreline, one per degree, against the original
mesh. The distance to the first wall hit at each (arc length, angle) position
forms a height map, which is exported as a flat mesh.

**Vertex-based straightening.** Every mesh vertex is assigned to a centreline
position by a flood fill over the mesh, then re-expressed in the local frame
of that position, giving a 3D tube with a straight centreline.

**Self-proximity.** Many scripts measure how much a colon passes close to
itself. A *bridge edge* is a short mesh edge whose two endpoints are nearest
to centreline positions more than 100 mm apart along the colon, where those
two centreline points are also within 1.15 times the sum of the local wall
radii. The per-dataset measure is bridge edges per metre of centreline.

**Jump cells.** Adjacent height-map cells whose wall distances differ by more
than 15 mm. They are the artefact measure used for the unrolling method.

## Requirements

Python 3.8. The results in this repository were produced with:

| Package | Version |
|---|---|
| numpy | 1.24.3 |
| scipy | 1.10.1 |
| scikit-image | 0.21.0 |
| nibabel | 5.2.1 |
| SimpleITK | 2.3.1 |
| trimesh | 4.11.5 |
| shapely | 2.0.7 |
| matplotlib | 3.7.5 |
| numpy-stl | any recent version |

## Data

The twelve scans come from the CT Colonography collection of The Cancer
Imaging Archive (TCIA):
<https://www.cancerimagingarchive.net/collection/ct-colonography/>.
They can be downloaded with the NBIA Data Retriever.

| Name in the code | TCIA ID | Position |
|---|---|---|
| `787` | CTC-8337000787 | Supine |
| `518` | 1.3.6.1.4.1.9328.50.4.0518 | Prone |
| `2320` | 1.3.6.1.4.1.9328.50.4.2320 | Supine |
| `3114` | 1.3.6.1.4.1.9328.50.4.3114 | Supine |
| `4371` | 1.3.6.1.4.1.9328.50.4.4371 | Supine |
| `563` | 1.3.6.1.4.1.9328.50.4.0563 | Supine |
| `5980` | 1.3.6.1.4.1.9328.50.4.5980 | Supine |
| `6651` | 1.3.6.1.4.1.9328.50.4.6651 | Prone |
| `7889` | 1.3.6.1.4.1.9328.50.4.7889 | Supine |
| `1840` | 1.3.6.1.4.1.9328.50.4.1840 | Supine |
| `3828` | 1.3.6.1.4.1.9328.50.4.3828 | Prone |
| `7198` | 1.3.6.1.4.1.9328.50.4.7198 | Supine |

Each script expects, per dataset, a segmentation mask (NIfTI, colon label 1)
and a centreline exported from 3D Slicer (`.mrk.json`, extracted with the VMTK
extension). The expected paths are listed in `DATASETS` at the top of
`analysis/validate_batch/validate_batch.py`, in the form:

```
Dataset/<name>/<mask>.nii
Dataset/<name>/<centreline>.mrk.json
```

The masks and centrelines used to produce the results are not distributed
here.

The masks were made in two ways, recorded in the `quality` field of the code
and result files: `curated` for `787` and `518` (AI segmentation with
manual correction), and `residue-fused` for the other ten (manual segmentation, with
a separately segmented residue mask merged in).

## Core code

| File | What it does |
|---|---|
| `colon_core.py` | Centreline smoothing and rotation-minimising frame, mesh extraction, bridge-edge detection, vertex-based straightening (`straighten_mesh`), ray-casting unrolling (`cast_rays_direct`, `cast_rays_adaptive`), gap filling and height-map mesh construction. |
| `generate_unrolls.py` | Writes the unrolled mesh (STL) for every dataset in `DATASETS`. |
| `generate_straightened_mesh.py` | Writes the vertex-based straightened mesh (STL) for the datasets listed in its `TARGETS`. |

## Evaluation scripts

Run every script from the repository root, for example
`python analysis/validate_batch/validate_batch.py`. Each script writes its
results as JSON next to itself, and those result files are included.

### Unrolling

| Script | What it measures | Result file |
|---|---|---|
| `analysis/validate_batch/validate_batch.py` | Per dataset: colon length, bridge edges per metre, ray-miss rate, jump cells. | `validation_results.json` |
| `analysis/bridge_edge_normalisation/bridge_edge_normalisation.py` | Bridge edges normalised by mesh edge count instead of length, to check the self-proximity ranking does not depend on mesh density. | `bridge_edge_normalisation_*.json` |
| `analysis/centreline_self_proximity/centreline_self_proximity.py` | Self-proximity from the centreline alone (share of centreline with another point more than 100 mm away along the colon but within 30, 40, 43 or 50 mm in space), less sensitive to where the mask places the wall, correlated with bridge edges/m and the distortion measures. | `centreline_self_proximity_results.json` |
| `analysis/polyp_roundtrip_validation/polyp_roundtrip_validation.py` | Position round trip of six marked structures on `787`: 3D position to grid cell and back. | `polyp_roundtrip_results.json` |
| `analysis/cross_section_validation/cross_section_validation.py` | Agreement between the ray-cast mean radius and the equivalent-area radius of a direct mesh-plane section, at 20 positions per dataset. | `cross_section_results.json` |
| `analysis/cross_section_validation/per_angle_validation.py` | The same sections compared angle by angle with the ray-cast distances. | `per_angle_results.json` |
| `analysis/containment_check/containment_check.py` | Distance from jump cells to the nearest bridge vertex, against a random baseline. | `containment_results.json` |
| `analysis/containment_check/permutation_tests.py` | Circular-shift permutation test of that distance per dataset, and a Wilcoxon test across datasets. | `containment_permutation_results.json`, `containment_wilcoxon_result.json` |
| `analysis/determinism/determinism_check.py` | Whether two from-scratch runs on `1840` give bit-identical output. Run twice with different tags (`A`, `B`) and compare the hashes. | `determinism_A.json`, `determinism_B.json` |
| `analysis/jump_mechanism_audit/jump_mechanism_audit.py` | Samples jump regions on three datasets and exports them as 3D Slicer markups. The regions were then classified by eye in 3D Slicer; the classifications are in `jump_mechanism_audit.xlsx`. | `jump_regions_*.json` |

### Straightening under self-proximity

| Script | What it does | Result file |
|---|---|---|
| `analysis/sato_batch_run/sato_prepare.py` | Converts a mask and centreline into SATO's input format (isotropic resample, voxel-space centreline and width). | |
| `analysis/sato_batch_run/sato_batch_run.py` | Runs SATO on all twelve datasets and flags slices whose foreground area exceeds 1.5 times the local median. Needs `straighten_seg_fast.py`, see [SATO](#sato). | `sato_batch_results.json` |
| `analysis/sato_batch_run/run_sato_original_518.sh` | Times the unmodified SATO script on `518`. | `sato_original_518_timing.log` |
| `analysis/sato_batch_run/verify_fast_matches_original.py` | Compares the output of the unmodified and the patched script on `518`, voxel by voxel. | |
| `analysis/sato_batch_run/test_strict_connectivity.py` | Re-runs SATO's object selection with face-only connectivity on `518` and `787`. Needs `straighten_seg_fast.py`. | `strict_connectivity_results.json` |
| `analysis/jump_mechanism_audit/sato_corrected_spike_check.py` | Compares the cross-sectional area of SATO's output on `518` with the area expected from the ray-cast radius. | `sato_corrected_spike_check_results.json` |
| `analysis/containment_check/curvature_permutation_current_def.py` | Centreline turning angle at bridge-edge locations on `787`, with a circular-shift permutation test. | `curvature_permutation_787_current_def.json` |
| `analysis/stretched_edge_validation/stretched_edge_validation.py` | Share of edges in the vertex-based straightened mesh longer than 8 times the median edge length. Needs the meshes from `generate_straightened_mesh.py`. | `stretched_edge_results.json` |

### Correlations and figures

| Script | What it does | Result file |
|---|---|---|
| `analysis/A1_correlations/correlations.py` | Pearson and Spearman correlations between self-proximity and each distortion measure (including the per-angle share of angles above 2 mm) across the twelve datasets, with leave-one-out ranges and bootstrap intervals. Reads `input.csv`. | `results.json` |
| `analysis/A1_correlations/make_scatter_figures.py` | Scatter plots of self-proximity against jump-cell density and against cross-sectional disagreement. | `figures/*.png` |

### Segmentation comparison

`analysis/ai_segmentation_comparison/ai_segmentation_comparison_results.json`
holds the `validate_batch.py` metrics for two further segmentations of `518`,
produced with the MONAI Auto3DSeg and TotalSegmentator extensions for
3D Slicer. Those masks are not distributed here.

### Order of execution

Some scripts read the output of others:

1. `validate_batch.py` first. Its `validation_results.json` is read by the
   cross-section, bridge-edge normalisation, SATO batch and correlation
   scripts.
2. `generate_straightened_mesh.py` before `stretched_edge_validation.py`.
3. `sato_prepare.py` and `sato_batch_run.py` before
   `sato_corrected_spike_check.py`, `test_strict_connectivity.py` and
   `correlations.py`.

## SATO

SATO (Zhou et al., *SATO: Straighten Any 3D Tubular Object*, IEEE Transactions
on Medical Imaging, 2025) is not included. Clone it into `SATO_repo/` at the
repository root:

```
git clone https://github.com/Yanfeng-Zhou/SATO SATO_repo
```

The results here were produced with the `main` branch at commit
`33928e3c13c9e1f9e06a80aab19064f9bcd95c98` (2 May 2025).

### The patched straightening script

SATO's repository has no licence, so no SATO code is distributed here,
including the patched script used for these results. `sato_batch_run.py` and
`test_strict_connectivity.py` import it as
`analysis/sato_batch_run/straighten_seg_fast.py`, so it has to be recreated
locally from SATO's `straighten/straighten_seg.py`:

1. Copy `SATO_repo/straighten/straighten_seg.py` to
   `analysis/sato_batch_run/straighten_seg_fast.py`.
2. Keep `save_center_objects(image, x_size, y_size, z_size)` as a module-level
   function.
3. Move the body of the script into a function
   `straighten_seg_fast(seg_path, centerline_path, save_straighten_path,
   crop_radius_ratio=5, if_smooth=True, remove_small_holes_thr=200)` that
   takes these values as arguments instead of reading them from `argparse`.
4. Replace the per-slice `np.concatenate` with a preallocated array. Before
   the loop, create
   `np.zeros((length + 2, crop_side, crop_side))` with
   `crop_side = 2 * max_radius + 1`, and inside the loop write slice `m` to
   index `m + 1`. This keeps the one empty slice at each end that the original
   produces.

The original script grows its output with `np.concatenate` on every slice,
which is quadratic in the number of slices: it took 292 s on `518` (about
1,500 slices), against 89 s after the change. The straightening computation
is otherwise unchanged, and on `518` the two scripts produced identical
output (`verify_fast_matches_original.py` performs that comparison).

## Limitations of this release

- The scripts cannot be rerun without the masks and centrelines, which are not
  distributed. The included result files are the outputs of the runs reported
  in the thesis.
- The classification of jump regions was done by eye and is recorded in a
  spreadsheet, not produced by a script.

## Citation

> Bernardo Augusto. *Geometric Fidelity in CT Colonography Reconstruction:
> Segmentation, Straightening and Unrolling*. MSc thesis, Instituto Superior
> Técnico, Universidade de Lisboa, 2026.
