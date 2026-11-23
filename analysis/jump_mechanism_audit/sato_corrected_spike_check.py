"""SATO excess-area check for C0518.

Compares each SATO-straightened slice's foreground cross-sectional area
against the area expected from the ray-cast radius at the matching
centreline arc position: expected_area = pi * row_radius(arc)^2. A slice is
flagged if actual/expected exceeds RATIO_THRESHOLD.

Run from the repo root: python analysis/jump_mechanism_audit/sato_corrected_spike_check.py
"""
import json
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy import interpolate

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS

RATIO_THRESHOLD = 1.5  # a slice is flagged when actual/expected area exceeds this
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def recompute_sato_centerline_positions(centerline_npy_path):
    centerline = np.load(centerline_npy_path, allow_pickle=True).tolist()
    length = int(centerline["edge_length"])
    coords = np.array([p["coordinate"] for p in centerline["point"]], dtype=float)
    x_coordinate, y_coordinate, z_coordinate = coords[:, 0], coords[:, 1], coords[:, 2]
    tck, myu = interpolate.splprep([x_coordinate, y_coordinate, z_coordinate])
    myu = np.linspace(myu.min(), myu.max(), length)
    zx, zy, zz = interpolate.splev(myu, tck)
    return np.stack([zx, zy, zz], axis=1)  # (length, 3) voxel-index (z,y,x)


def main():
    cache_path = os.path.join(OUT_DIR, "518_row_radius_cache.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached ray-cast radius from {cache_path} (skipping the expensive re-cast)...")
        cached = np.load(cache_path)
        arc_steps, row_radius = cached["arc_steps"], cached["row_radius"]
    else:
        entry = next(e for e in DATASETS if e["name"] == "518")
        print("Loading C0518 mesh/centerline and casting rays (RQ3 pipeline) to get the validated radius...")
        cf.MESH_UPSAMPLE = 1
        cl_pts = cf.load_centerline_slicer(entry["centerline"])
        cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
        verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
        arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)
        arc_steps, angle_steps, hit_distances, *_ = cf.cast_rays_adaptive(
            verts, faces, cl_pts, arc, normals, binormals, base_ray_dist=entry.get("max_ray_dist", 50.0))
        filled = cf.fill_gaps_smooth(hit_distances)
        row_radius = np.nanmean(filled, axis=1)  # mm, per arc_steps position
        np.savez(cache_path, arc_steps=arc_steps, row_radius=row_radius)
        print(f"  Cached to {cache_path} for future reruns")
    print(f"  {len(arc_steps)} arc positions, radius range {row_radius.min():.2f}-{row_radius.max():.2f}mm")

    # interpolator for the ray-cast radius at any arc position (mm)
    radius_of_arc = interpolate.interp1d(
        arc_steps, row_radius, bounds_error=False, fill_value=(row_radius[0], row_radius[-1]))

    print("\nLoading SATO straightened output...")
    straight_img = sitk.ReadImage("SATO_inputs/518_straightened.nii.gz")
    straight_arr = sitk.GetArrayFromImage(straight_img)
    spacing = straight_img.GetSpacing()  # (x,y,z), mm/voxel; in-plane area per pixel:
    px_area_mm2 = spacing[0] * spacing[1]
    fg_per_slice = straight_arr.reshape(straight_arr.shape[0], -1).sum(axis=1)
    fg_interior = fg_per_slice[1:-1]  # drop the zero-pad slices straighten_seg_fast.py adds
    actual_area_mm2 = fg_interior.astype(float) * px_area_mm2

    print("Recomputing per-slice centreline arc positions (voxel-index space)...")
    slice_coords_zyx = recompute_sato_centerline_positions("SATO_inputs/518_centerline.npy")
    assert len(slice_coords_zyx) == len(fg_interior), \
        f"length mismatch: {len(slice_coords_zyx)} centreline samples vs {len(fg_interior)} interior slices"

    # convert the SATO centreline's voxel-index coordinates to physical points
    # and take the cumulative 3D distance as each slice's arc position (mm)
    mask_img = sitk.ReadImage("SATO_inputs/518_mask.nii.gz")
    phys_pts = np.array([
        mask_img.TransformContinuousIndexToPhysicalPoint((p[2], p[1], p[0]))
        for p in slice_coords_zyx
    ])
    seg_dist = np.linalg.norm(np.diff(phys_pts, axis=0), axis=1)
    slice_arc_mm = np.concatenate([[0.0], np.cumsum(seg_dist)])

    expected_radius = radius_of_arc(slice_arc_mm)
    expected_area_mm2 = np.pi * expected_radius ** 2
    ratio = actual_area_mm2 / np.maximum(expected_area_mm2, 1.0)

    is_spike = ratio > RATIO_THRESHOLD
    pct_spike = 100.0 * is_spike.sum() / len(is_spike)
    max_ratio = float(ratio.max())

    # contiguous-region summary
    from itertools import groupby
    run_lengths = [len(list(g)) for k, g in groupby(is_spike) if k]
    frac_affected_len = 100.0 * is_spike.sum() / len(is_spike)
    longest_run = max(run_lengths) if run_lengths else 0
    longest_run_pct = 100.0 * longest_run / len(is_spike)

    print(f"\nSlices flagged (ratio > {RATIO_THRESHOLD}x expected area): "
          f"{is_spike.sum()}/{len(is_spike)} ({pct_spike:.6f}%)")
    print(f"Max ratio (actual/expected area): {max_ratio:.6f}x")
    print(f"Longest single contiguous spike run: {longest_run} slices "
          f"({longest_run_pct:.6f}% of straightened length)")
    print(f"Total fraction of length flagged (possibly split across runs): {frac_affected_len:.6f}%")

    flagged_idx = np.where(is_spike)[0].tolist()
    # contiguous run boundaries, for readability (start,end inclusive, slice count)
    runs = []
    start = None
    for i, flag in enumerate(is_spike):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(is_spike) - 1, len(is_spike) - start))
    runs.sort(key=lambda r: -r[2])
    print(f"\n{len(runs)} contiguous flagged runs (slice indices, 0-based, inclusive), longest first:")
    for s, e, n in runs[:15]:
        print(f"  slices {s}-{e} ({n} slices, arc {slice_arc_mm[s]:.1f}-{slice_arc_mm[e]:.1f}mm, "
              f"max ratio in run {ratio[s:e+1].max():.2f}x)")
    if len(runs) > 15:
        print(f"  ... and {len(runs) - 15} more shorter runs")

    result = {
        "dataset": "518 (C0518)",
        "method": "actual per-slice foreground area vs. pi*validated_ray_cast_radius^2, "
                  "ratio > 1.5x flagged (independent of SATO's own output, unlike the naive "
                  "rolling-median spike metric)",
        "n_slices": int(len(is_spike)),
        "n_spike_slices": int(is_spike.sum()),
        "pct_spike_slices": round(pct_spike, 6),
        "max_ratio": round(max_ratio, 6),
        "longest_contiguous_run_slices": int(longest_run),
        "longest_contiguous_run_pct": round(longest_run_pct, 6),
        "flagged_slice_indices": flagged_idx,
        "contiguous_runs": [{"start": s, "end": e, "n_slices": n,
                              "arc_start_mm": round(float(slice_arc_mm[s]), 2),
                              "arc_end_mm": round(float(slice_arc_mm[e]), 2),
                              "max_ratio_in_run": round(float(ratio[s:e+1].max()), 4)}
                             for s, e, n in runs],
    }
    with open(os.path.join(OUT_DIR, "sato_corrected_spike_check_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved sato_corrected_spike_check_results.json")


if __name__ == "__main__":
    main()
