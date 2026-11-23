"""Local containment check, all 12 datasets: do jump-artifact cells
spatially cluster around self-proximity (bridge-edge) locations?

Method: reconstruct a 3D point for every jump-artifact grid cell, then find
its distance to the nearest bridge-edge vertex. Compare that distribution
against a random baseline (the same number of cells sampled uniformly from
the valid grid) with a one-sided Mann-Whitney U test.
"""
import json
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS, TOUCH_MULTIPLIER

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def reconstruct_3d(arc_vals, angle_vals, radius_vals, arc, cl_pts, normals, binormals):
    C, N, B = cf.interpolate_centerline_frame(arc_vals, arc, cl_pts, normals, binormals)
    angle_rad = np.radians(angle_vals)
    return C + radius_vals[:, None] * (np.cos(angle_rad)[:, None] * N + np.sin(angle_rad)[:, None] * B)


def check_one(entry):
    name = entry["name"]
    print(f"\n{'='*70}\nLocal containment check: {name}\n{'='*70}")
    t0 = time.time()

    cf.MESH_UPSAMPLE = 1

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)

    # adaptive ray-distance ceiling (colon_core.cast_rays_adaptive), as in validate_batch.py
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
    print(f"\n{len(bridge_edges)} bridge edges -> {len(bridge_pts)} unique bridge vertices")
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
    print(f"{n_jump} jump-artifact cell instances (>15mm) reconstructed to 3D")

    if n_jump == 0:
        print(f"  No jump cells for {name} -- skipping")
        return None

    jump_pts = reconstruct_3d(jump_arc, jump_angle, jump_radius, arc, cl_pts, normals, binormals)
    jump_dist, _ = bridge_tree.query(jump_pts)

    rng = np.random.default_rng(0)
    valid_cells = np.argwhere(~np.isnan(filled))
    n_sample = min(n_jump, len(valid_cells))
    sel = rng.choice(len(valid_cells), size=n_sample, replace=False)
    rand_i, rand_j = valid_cells[sel, 0], valid_cells[sel, 1]
    rand_pts = reconstruct_3d(arc_steps[rand_i], angle_steps[rand_j], filled[rand_i, rand_j],
                               arc, cl_pts, normals, binormals)
    rand_dist, _ = bridge_tree.query(rand_pts)

    u_stat, p_value = mannwhitneyu(jump_dist, rand_dist, alternative="less")

    result = {
        "dataset": name,
        "quality": entry.get("quality", "unknown"),
        "n_bridge_edges": len(bridge_edges),
        "n_bridge_vertices": len(bridge_pts),
        "n_jump_cells": n_jump,
        "jump_dist_mean": round(float(jump_dist.mean()), 2),
        "jump_dist_median": round(float(np.median(jump_dist)), 2),
        "random_dist_mean": round(float(rand_dist.mean()), 2),
        "random_dist_median": round(float(np.median(rand_dist)), 2),
        "ratio_random_over_jump_median": round(float(np.median(rand_dist)) / max(float(np.median(jump_dist)), 0.01), 3),
        "mannwhitney_p_value": float(p_value),
        "time_s": round(time.time() - t0, 1),
    }
    print(f"\nJump cells  -> dist to nearest bridge vertex: mean={result['jump_dist_mean']}mm "
          f"median={result['jump_dist_median']}mm")
    print(f"Random cells -> dist to nearest bridge vertex: mean={result['random_dist_mean']}mm "
          f"median={result['random_dist_median']}mm")
    print(f"Ratio (random median / jump median): {result['ratio_random_over_jump_median']:.2f}x  "
          f"Mann-Whitney p={p_value:.3e}")
    return result


if __name__ == "__main__":
    results = []
    for entry in DATASETS:
        try:
            r = check_one(entry)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"FAILED on {entry['name']}: {e}")

    with open(os.path.join(OUT_DIR, "containment_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    cols = ["dataset", "quality", "n_bridge_vertices", "n_jump_cells",
            "jump_dist_median", "random_dist_median", "ratio_random_over_jump_median",
            "mannwhitney_p_value"]
    header = " | ".join(c.ljust(15) for c in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        print(" | ".join(str(r[c]).ljust(15) for c in cols))

    ratios = np.array([r["ratio_random_over_jump_median"] for r in results])
    print(f"\nMean ratio across {len(results)} datasets: {ratios.mean():.2f}x")
    print(f"All p-values < 0.001: {all(r['mannwhitney_p_value'] < 0.001 for r in results)}")
    print("Saved containment_results.json")
