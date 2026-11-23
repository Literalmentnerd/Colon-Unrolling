"""Turning-angle (curvature) permutation test on C0787, using the
same bridge-edge definition as the batch validation harness (local wall
radius from the ray-cast output, touch_multiplier=TOUCH_MULTIPLIER).

15 mm turning-angle window, both endpoints of every bridge edge as arc
samples, 1,000 circular shifts,
p = (1 + #shifts with shifted_mean >= observed_mean) / 1001.

Also reports the random-position baseline (mean angle, % above 90 degrees)
and, as a stability check, the p-value at 10,000 shifts and across ten
other seeds.

Run from the repo root:
    python analysis/containment_check/curvature_permutation_current_def.py
"""
import json
import os
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS, TOUCH_MULTIPLIER

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_MM = 15.0
N_SHIFTS = 1000
SEED = 42
RADIUS_CACHE = os.path.join(OUT_DIR, "787_row_radius_cache.npz")


def permutation_p(bridge_arcs, turning_angle_at, arc_min, kept_length, observed_mean, n_shifts, seed):
    rng = np.random.default_rng(seed)
    shifted_means = np.empty(n_shifts)
    for k in range(n_shifts):
        s = rng.uniform(0, kept_length)
        shifted_arc = arc_min + ((bridge_arcs - arc_min + s) % kept_length)
        shifted_means[k] = turning_angle_at(shifted_arc).mean()
    n_ge = int((shifted_means >= observed_mean).sum())
    return (1 + n_ge) / (n_shifts + 1), n_ge, shifted_means


def main():
    t0 = time.time()
    entry = next(e for e in DATASETS if e["name"] == "787")
    cf.MESH_UPSAMPLE = 1

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])

    if os.path.exists(RADIUS_CACHE):
        cached = np.load(RADIUS_CACHE)
        arc_steps, row_radius = cached["arc_steps"], cached["row_radius"]
        print(f"Loaded cached ray-cast radius from {os.path.basename(RADIUS_CACHE)}", flush=True)
    else:
        print("Casting rays to get the local wall radius (same as the batch harness)...", flush=True)
        arc_steps, angle_steps, hit_distances, *_ = cf.cast_rays_adaptive(
            verts, faces, cl_pts, arc, normals, binormals,
            base_ray_dist=entry.get("max_ray_dist", 50.0))
        filled = cf.fill_gaps_smooth(hit_distances)
        row_radius = np.nanmean(filled, axis=1)
        np.savez(RADIUS_CACHE, arc_steps=arc_steps, row_radius=row_radius)

    bridge_edges = cf.find_bridge_edges(
        verts, faces, cl_pts, arc, radius_arc=arc_steps, radius_values=row_radius,
        touch_multiplier=TOUCH_MULTIPLIER)
    print(f"{len(bridge_edges)} bridge edges under the harness definition", flush=True)

    tree = cKDTree(cl_pts)
    _, nearest_idx = tree.query(verts)
    t_est = arc[nearest_idx]

    def turning_angle_at(arc_pos):
        i_before = np.clip(np.searchsorted(arc, arc_pos - WINDOW_MM), 0, len(arc) - 1)
        i_after = np.clip(np.searchsorted(arc, arc_pos + WINDOW_MM), 0, len(arc) - 1)
        t1, t2 = tangents[i_before], tangents[i_after]
        cos_a = np.clip(np.sum(t1 * t2, axis=-1) /
                        (np.linalg.norm(t1, axis=-1) * np.linalg.norm(t2, axis=-1) + 1e-12), -1, 1)
        return np.degrees(np.arccos(cos_a))

    bridge_arcs = np.array([t_est[k] for e in bridge_edges for k in e])
    bridge_angles = turning_angle_at(bridge_arcs)
    observed_mean = float(bridge_angles.mean())

    # random-position baseline (fixed seed)
    random_arcs = np.random.default_rng(0).uniform(arc.min(), arc.max(), len(bridge_arcs))
    random_angles = turning_angle_at(random_arcs)

    arc_min = float(arc.min())
    kept_length = float(arc.max() - arc.min())

    perm_p, n_ge, shifted_means = permutation_p(
        bridge_arcs, turning_angle_at, arc_min, kept_length, observed_mean, N_SHIFTS, SEED)

    p_10k, n_ge_10k, _ = permutation_p(
        bridge_arcs, turning_angle_at, arc_min, kept_length, observed_mean, 10000, SEED)
    other_seed_p = [permutation_p(bridge_arcs, turning_angle_at, arc_min, kept_length,
                                  observed_mean, N_SHIFTS, s)[0] for s in range(10)]

    result = {
        "dataset": "787 (C0787)",
        "bridge_edge_definition": "local wall radius, touch_multiplier=%.2f (batch harness definition)" % TOUCH_MULTIPLIER,
        "n_bridge_edges": len(bridge_edges),
        "n_bridge_arc_samples": int(len(bridge_arcs)),
        "window_mm": WINDOW_MM,
        "bridge_turning_angle_mean": observed_mean,
        "random_turning_angle_mean": float(random_angles.mean()),
        "ratio_bridge_over_random_mean": observed_mean / float(random_angles.mean()),
        "pct_bridge_angle_gt_90deg": float(100 * np.mean(bridge_angles > 90)),
        "pct_random_angle_gt_90deg": float(100 * np.mean(random_angles > 90)),
        "pct_bridge_angle_lt_90deg": float(100 * np.mean(bridge_angles < 90)),
        "kept_length_mm": round(kept_length, 1),
        "n_shifts": N_SHIFTS,
        "seed": SEED,
        "shifted_mean_mean": round(float(shifted_means.mean()), 2),
        "n_shifts_ge_observed": n_ge,
        "permutation_p_value": perm_p,
        "stability_check": {
            "p_at_10000_shifts_same_seed": p_10k,
            "n_shifts_ge_observed_of_10000": n_ge_10k,
            "p_at_1000_shifts_seeds_0_to_9": other_seed_p,
        },
        "time_s": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2), flush=True)
    with open(os.path.join(OUT_DIR, "curvature_permutation_787_current_def.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("Saved curvature_permutation_787_current_def.json")


if __name__ == "__main__":
    main()
