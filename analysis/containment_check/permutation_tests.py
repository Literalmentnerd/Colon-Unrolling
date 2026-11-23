"""Circular-shift permutation test for the local containment check.

Runs the containment test on all 12 datasets, then a cross-dataset
Wilcoxon test on the containment log-ratios.

Step 1: within-dataset containment permutation test.
    For each dataset, take jump cells as (arc, angle) grid positions.
    For k=1..1000, draw a random arc offset s ~ Uniform(0, kept_length),
    shift every jump cell's arc by s modulo kept_length (angle and radius
    unchanged), and
    reconstruct 3D points to get a shifted median distance to the nearest
    bridge vertex. p = (1 + #shifts with shifted_median <= observed_median) / 1001.

Step 2: across-dataset containment test.
    L_i = log(random_median_i / jump_median_i) for the 12 datasets;
    scipy.stats.wilcoxon(L, alternative='greater').
"""
import json
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS, TOUCH_MULTIPLIER

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_SHIFTS = 1000
SEED = 42


def reconstruct_3d(arc_vals, angle_vals, radius_vals, arc, cl_pts, normals, binormals):
    C, N, B = cf.interpolate_centerline_frame(arc_vals, arc, cl_pts, normals, binormals)
    angle_rad = np.radians(angle_vals)
    return C + radius_vals[:, None] * (np.cos(angle_rad)[:, None] * N + np.sin(angle_rad)[:, None] * B)


def circular_shift_arc(arc_vals, arc_min, kept_length, s):
    return arc_min + ((arc_vals - arc_min + s) % kept_length)


def containment_one(entry, rng):
    name = entry["name"]
    print(f"\n{'='*70}\nContainment permutation test: {name}\n{'='*70}", flush=True)
    t0 = time.time()

    cf.MESH_UPSAMPLE = 1

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)

    arc_steps, angle_steps, hit_distances, _, _, _ = cf.cast_rays_adaptive(
        verts, faces, cl_pts, arc, normals, binormals,
        base_ray_dist=entry.get("max_ray_dist", 50.0))
    filled = cf.fill_gaps_smooth(hit_distances)
    row_radius = np.nanmean(filled, axis=1)

    bridge_edges = cf.find_bridge_edges(
        verts, faces, cl_pts, arc, radius_arc=arc_steps, radius_values=row_radius,
        touch_multiplier=TOUCH_MULTIPLIER)
    bridge_vertex_idx = sorted(set(i for e in bridge_edges for i in e))
    if len(bridge_vertex_idx) < 2:
        print(f"  Too few bridge vertices ({len(bridge_vertex_idx)}) for {name} -- skipping")
        return None
    bridge_pts = verts[bridge_vertex_idx]
    bridge_tree = cKDTree(bridge_pts)

    d_arc = np.abs(np.diff(filled, axis=0))
    d_angle = np.abs(np.diff(filled, axis=1))

    ia, ja = np.where(d_arc > 15)
    jump_arc_vals = (arc_steps[ia] + arc_steps[ia + 1]) / 2.0
    jump_angle_vals_a = angle_steps[ja]
    jump_radius_a = np.nanmean(np.stack([filled[ia, ja], filled[ia + 1, ja]]), axis=0)

    ib, jb = np.where(d_angle > 15)
    jump_arc_vals_b = arc_steps[ib]
    jump_angle_vals_b = (angle_steps[jb] + angle_steps[jb + 1]) / 2.0
    jump_radius_b = np.nanmean(np.stack([filled[ib, jb], filled[ib, jb + 1]]), axis=0)

    jump_arc = np.concatenate([jump_arc_vals, jump_arc_vals_b])
    jump_angle = np.concatenate([jump_angle_vals_a, jump_angle_vals_b])
    jump_radius = np.concatenate([jump_radius_a, jump_radius_b])
    valid = ~np.isnan(jump_radius)
    jump_arc, jump_angle, jump_radius = jump_arc[valid], jump_angle[valid], jump_radius[valid]
    n_jump = len(jump_arc)

    if n_jump == 0:
        print(f"  No jump cells for {name} -- skipping")
        return None

    jump_pts = reconstruct_3d(jump_arc, jump_angle, jump_radius, arc, cl_pts, normals, binormals)
    jump_dist, _ = bridge_tree.query(jump_pts)
    observed_median = float(np.median(jump_dist))

    rng2 = np.random.default_rng(0)  # same seed as containment_check.py, for the random-cell baseline
    valid_cells = np.argwhere(~np.isnan(filled))
    n_sample = min(n_jump, len(valid_cells))
    sel = rng2.choice(len(valid_cells), size=n_sample, replace=False)
    rand_i, rand_j = valid_cells[sel, 0], valid_cells[sel, 1]
    rand_pts = reconstruct_3d(arc_steps[rand_i], angle_steps[rand_j], filled[rand_i, rand_j],
                               arc, cl_pts, normals, binormals)
    rand_dist, _ = bridge_tree.query(rand_pts)
    random_median = float(np.median(rand_dist))

    arc_min = float(arc_steps.min())
    kept_length = float(arc_steps.max() - arc_steps.min())

    print(f"  {n_jump} jump cells, {len(bridge_pts)} bridge vertices, "
          f"kept_length={kept_length:.0f}mm -- running {N_SHIFTS} circular shifts...", flush=True)

    n_le = 0
    shifted_medians = np.empty(N_SHIFTS)
    for k in range(N_SHIFTS):
        s = rng.uniform(0, kept_length)
        shifted_arc = circular_shift_arc(jump_arc, arc_min, kept_length, s)
        shifted_pts = reconstruct_3d(shifted_arc, jump_angle, jump_radius, arc, cl_pts, normals, binormals)
        shifted_dist, _ = bridge_tree.query(shifted_pts)
        m = float(np.median(shifted_dist))
        shifted_medians[k] = m
        if m <= observed_median:
            n_le += 1

    perm_p = (1 + n_le) / (N_SHIFTS + 1)

    result = {
        "dataset": name,
        "quality": entry.get("quality", "unknown"),
        "n_bridge_edges": len(bridge_edges),
        "n_bridge_vertices": len(bridge_pts),
        "n_jump_cells": n_jump,
        "jump_dist_median": round(observed_median, 2),
        "random_dist_median": round(random_median, 2),
        "ratio_random_over_jump_median": round(random_median / max(observed_median, 0.01), 3),
        "kept_length_mm": round(kept_length, 1),
        "n_shifts": N_SHIFTS,
        "shifted_median_mean": round(float(shifted_medians.mean()), 2),
        "shifted_median_min": round(float(shifted_medians.min()), 2),
        "shifted_median_max": round(float(shifted_medians.max()), 2),
        "n_shifts_le_observed": n_le,
        "permutation_p_value": perm_p,
        "time_s": round(time.time() - t0, 1),
    }
    print(f"  observed median={observed_median:.2f}mm, shifted median range="
          f"[{result['shifted_median_min']:.2f}, {result['shifted_median_max']:.2f}]mm, "
          f"{n_le}/{N_SHIFTS} shifts <= observed -> perm_p={perm_p:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return result


if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    containment_results = []
    for entry in DATASETS:
        try:
            r = containment_one(entry, rng)
            if r is not None:
                containment_results.append(r)
        except Exception as e:
            print(f"FAILED on {entry['name']}: {e}", flush=True)

    with open(os.path.join(OUT_DIR, "containment_permutation_results.json"), "w") as f:
        json.dump(containment_results, f, indent=2)

    # Step 2: across-dataset Wilcoxon on log-ratios
    L = np.array([np.log(r["random_dist_median"] / r["jump_dist_median"]) for r in containment_results])
    wstat, wp = wilcoxon(L, alternative="greater")
    ratios = np.array([r["ratio_random_over_jump_median"] for r in containment_results])
    wilcoxon_result = {
        "n_datasets": len(L),
        "log_ratios": L.tolist(),
        "wilcoxon_statistic": float(wstat),
        "wilcoxon_p_value": float(wp),
        "median_ratio": round(float(np.median(ratios)), 3),
    }
    with open(os.path.join(OUT_DIR, "containment_wilcoxon_result.json"), "w") as f:
        json.dump(wilcoxon_result, f, indent=2)
    print(f"\n{'='*70}\nCross-dataset Wilcoxon (step 2)\n{'='*70}", flush=True)
    print(json.dumps(wilcoxon_result, indent=2), flush=True)

    print(f"\n{'='*70}\nALL DONE in {time.time()-t_start:.0f}s\n{'='*70}", flush=True)
