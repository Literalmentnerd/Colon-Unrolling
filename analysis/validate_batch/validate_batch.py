"""Batch validation harness: runs the direct ray-cast pipeline over a list
of (mask, centerline) dataset pairs and collects comparable metrics --
colon size, self-proximity/coiling, ray-miss rate, adaptive edge trim, and
residual jump artifacts -- into one summary table and a JSON file, instead
of reading through separate console logs per dataset.

Run with CWD at the repo root (DATASETS entries below are root-relative
paths), e.g. `python analysis/validate_batch/validate_batch.py` from
Straightening/, not `cd`'d into this folder.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import colon_core as cf

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# "max_ray_dist" is optional per entry (defaults to 50.0mm).
#
# "quality" is "curated" (AI + manual segmentation, no residue mask merged
# in) or "residue-fused" (manual segmentation with a separately segmented residue
# mask merged in by voxel-wise union).
DATASETS = [
    {"name": "787", "nifti": "Dataset/787/colon_finetuned_fixedl.nii",
     "centerline": "Dataset/787/Centerline curve 787.mrk.json", "label": 1, "quality": "curated"},
    {"name": "518", "nifti": "Dataset/518/colon_lumen_mask.nii",
     "centerline": "Dataset/518/Centerline curve_1 (0).mrk.json", "label": 1, "quality": "curated"},
    {"name": "2320", "nifti": "Dataset/2320/1_3_6_1_4_1_9328_50_4_2320.nii",
     "centerline": "Dataset/2320/Centerline curve_1 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "3114", "nifti": "Dataset/3114/1_3_6_1_4_1_9328_50_4_3114.nii",
     "centerline": "Dataset/3114/Centerline curve_2 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "4371", "nifti": "Dataset/4371/1_3_6_1_4_1_9328_50_4_4371.nii",
     "centerline": "Dataset/4371/Centerline curve_3 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "563", "nifti": "Dataset/563/1_3_6_1_4_1_9328_50_4_563.nii",
     "centerline": "Dataset/563/Centerline curve (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "5980", "nifti": "Dataset/5980/1_3_6_1_4_1_9328_50_4_5980.nii",
     "centerline": "Dataset/5980/Centerline curve_8 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "6651", "nifti": "Dataset/6651/1_3_6_1_4_1_9328_50_4_6651.nii",
     "centerline": "Dataset/6651/Centerline curve_4 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "7889", "nifti": "Dataset/7889/1_3_6_1_4_1_9328_50_4_7889.nii",
     "centerline": "Dataset/7889/Centerline curve_5 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "1840", "nifti": "Dataset/1840/1_3_6_1_4_1_9328_50_4_1840.nii",
     "centerline": "Dataset/1840/Centerline curve_7 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "3828", "nifti": "Dataset/3828/1_3_6_1_4_1_9328_50_4_3828.nii",
     "centerline": "Dataset/3828/Centerline curve_10 (0).mrk.json", "label": 1, "quality": "residue-fused"},
    {"name": "7198", "nifti": "Dataset/7198/1_3_6_1_4_1_9328_50_4_7198.nii",
     "centerline": "Dataset/7198/Centerline curve_11 (0).mrk.json", "label": 1, "quality": "residue-fused"},
]


# A mesh edge only counts as self-proximity if the two centerline points its
# endpoints are nearest to are within TOUCH_MULTIPLIER x
# (local_radius_a + local_radius_b) of each other, with the local radius
# interpolated from the ray-cast data at each side's arc position.
TOUCH_MULTIPLIER = 1.15


def validate_dataset(entry, mesh_upsample=1):
    name = entry["name"]
    print(f"\n{'='*70}\nValidating: {name}\n{'='*70}")
    cf.MESH_UPSAMPLE = mesh_upsample
    t0 = time.time()

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)
    setup_time = time.time() - t0
    centerline_length_mm = float(arc[-1])

    # ray-casting runs first because bridge-edge detection needs the local wall
    # radius; cf.cast_rays_adaptive re-casts at a higher ray-distance ceiling if
    # the base ceiling truncates hits
    t2 = time.time()
    arc_steps, angle_steps, hit_distances, final_max_ray_dist, near_cap_probe_pct, recast = \
        cf.cast_rays_adaptive(verts, faces, cl_pts, arc, normals, binormals,
                               base_ray_dist=entry.get("max_ray_dist", 50.0))
    raycast_time = time.time() - t2
    recast_ray_dist = final_max_ray_dist if recast else None

    filled = cf.fill_gaps_smooth(hit_distances)
    mean_radius = float(np.nanmean(filled))
    row_radius = np.nanmean(filled, axis=1)  # true LOCAL radius per arc position
    d_arc = np.abs(np.diff(filled, axis=0))
    d_angle = np.abs(np.diff(filled, axis=1))
    # each discontinuity is counted once per adjacent cell pair: np.diff on a
    # length-N axis yields N-1 differences, one per (i, i+1) pair. Angular
    # wrap-around (359deg -> 0deg) is not included.
    big_jumps = int((d_arc > 15).sum() + (d_angle > 15).sum())
    jump_cells_10mm = int((d_arc > 10).sum() + (d_angle > 10).sum())
    jump_cells_20mm = int((d_arc > 20).sum() + (d_angle > 20).sum())
    # valid_cell_pairs: number of non-NaN adjacent-cell pairs in the diff arrays
    valid_cell_pairs = int((~np.isnan(d_arc)).sum() + (~np.isnan(d_angle)).sum())
    total_cell_pairs = int(d_arc.size + d_angle.size)
    # valid_cells: number of non-NaN cells in the (already trimmed) ray-cast
    # grid; jump_pct's denominator
    valid_cells = int((~np.isnan(hit_distances)).sum())
    kept_length_m = float(arc_steps[-1] - arc_steps[0]) / 1000.0
    jump_pct = 100.0 * big_jumps / valid_cells if valid_cells else None
    jump_per_m = big_jumps / kept_length_m if kept_length_m else None

    # self-proximity: mesh edges that bridge two loops of the colon. The raw
    # (uncapped) count is reported alongside the local-radius-checked one.
    t1 = time.time()
    bridge_edges_raw = cf.find_bridge_edges(verts, faces, cl_pts, arc)
    bridge_edges = cf.find_bridge_edges(verts, faces, cl_pts, arc,
                                         radius_arc=arc_steps, radius_values=row_radius,
                                         touch_multiplier=TOUCH_MULTIPLIER)
    bridge_time = time.time() - t1
    n_edge_instances = faces.shape[0] * 3
    bridge_edge_pct = 100.0 * len(bridge_edges) / n_edge_instances
    bridge_per_meter = len(bridge_edges) / (centerline_length_mm / 1000.0)
    bridge_per_meter_uncapped = len(bridge_edges_raw) / (centerline_length_mm / 1000.0)

    raw_miss_pct = 100.0 * np.isnan(hit_distances).sum() / hit_distances.size
    trim_start_mm = float(arc_steps[0])
    trim_end_mm = float(arc[-1] - arc_steps[-1])

    # share of hits landing within 2mm of the MAX_RAY_DIST ceiling (truncation risk)
    valid_hits = hit_distances[~np.isnan(hit_distances)]
    near_cap = valid_hits > (cf.MAX_RAY_DIST - 2.0)
    near_cap_pct = 100.0 * near_cap.sum() / max(1, hit_distances.size)

    total_time = time.time() - t0

    metrics = {
        "name": name,
        "quality": entry.get("quality", "unknown"),
        "mesh_upsample": mesh_upsample,
        "n_verts": len(verts),
        "n_faces": len(faces),
        "centerline_length_mm": round(centerline_length_mm, 1),
        "mean_radius_mm": round(mean_radius, 2),
        "bridge_edge_count": len(bridge_edges),
        "bridge_edge_pct_of_edges": round(bridge_edge_pct, 4),
        "bridge_edges_per_meter": round(bridge_per_meter, 2),
        "touch_multiplier": TOUCH_MULTIPLIER,
        "bridge_edge_count_uncapped": len(bridge_edges_raw),
        "bridge_edges_per_meter_uncapped": round(bridge_per_meter_uncapped, 2),
        "ray_miss_pct": round(raw_miss_pct, 2),
        "max_ray_dist_mm": cf.MAX_RAY_DIST,
        "adaptive_recast": recast_ray_dist is not None,
        "near_cap_pct_before_recast": round(near_cap_probe_pct, 3) if recast_ray_dist else None,
        "near_cap_hit_pct": round(near_cap_pct, 3),
        "trim_start_mm": round(trim_start_mm, 1),
        "trim_end_mm": round(trim_end_mm, 1),
        "big_jump_count_gt15mm": big_jumps,
        "jump_cells_10mm": jump_cells_10mm,
        "jump_cells_20mm": jump_cells_20mm,
        "valid_cells": valid_cells,
        "kept_length_m": round(kept_length_m, 4),
        "jump_pct": round(jump_pct, 4) if jump_pct is not None else None,
        "jump_per_m": round(jump_per_m, 2) if jump_per_m is not None else None,
        "valid_cell_pairs": valid_cell_pairs,
        "total_cell_pairs": total_cell_pairs,
        "setup_time_s": round(setup_time, 1),
        "bridge_detect_time_s": round(bridge_time, 1),
        "raycast_time_s": round(raycast_time, 1),
        "total_time_s": round(total_time, 1),
    }
    if near_cap_pct > 0.5:
        print(f"  WARNING: {near_cap_pct:.2f}% of hits are within 2mm of the "
              f"{cf.MAX_RAY_DIST}mm ray-distance ceiling -- likely truncating "
              f"real geometry (a wide merged-in residue mask?). Consider raising max_ray_dist "
              f"for this dataset and re-running.")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def print_summary_table(results):
    if not results:
        print("\nNo successful results to summarize.")
        return
    cols = ["name", "quality", "centerline_length_mm", "mean_radius_mm", "n_verts",
            "bridge_edge_count", "bridge_edges_per_meter", "ray_miss_pct",
            "near_cap_hit_pct", "trim_start_mm", "trim_end_mm",
            "big_jump_count_gt15mm", "total_time_s"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in results)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-" * len(header)

    # group the summary by quality (curated vs residue-fused)
    for group_name in sorted(set(r["quality"] for r in results)):
        group = [r for r in results if r["quality"] == group_name]
        print("\n" + "=" * len(header))
        print(f"SUMMARY -- {group_name} ({len(group)} dataset{'s' if len(group)!=1 else ''})")
        print("=" * len(header))
        print(header)
        print(sep)
        for r in group:
            print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
        avg_cols = ["bridge_edges_per_meter", "ray_miss_pct", "near_cap_hit_pct",
                    "big_jump_count_gt15mm"]
        avg = {c: round(sum(r[c] for r in group) / len(group), 3) for c in avg_cols}
        print(sep)
        print("  group average: " + ", ".join(f"{c}={avg[c]}" for c in avg_cols))


if __name__ == "__main__":
    results = []
    for entry in DATASETS:
        try:
            results.append(validate_dataset(entry))
        except Exception as e:
            print(f"FAILED on {entry['name']}: {e}")

    print_summary_table(results)

    with open(os.path.join(OUT_DIR, "validation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved validation_results.json")
