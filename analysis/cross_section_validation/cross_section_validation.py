"""Cross-sectional reference for the ray-cast output.

Slices the mesh with a plane perpendicular to the centerline tangent at N
sampled positions, computes the cross-sectional area of that slice
(mesh-plane polygon intersection, not ray-casting), derives an equivalent
radius (sqrt(area/pi)), and compares it against the ray-cast's mean radial
distance at the same position.

Sampling is restricted to each dataset's own trimmed/kept arc range (from
validation_results.json).
"""
import json
import os
import sys
import time
import numpy as np
import trimesh
from shapely.geometry import Point

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATE_BATCH_RESULTS = os.path.join(_ROOT, "analysis", "validate_batch", "validation_results.json")

N_SAMPLES_PER_DATASET = 20


def cross_section_radius(mesh, origin, normal):
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return None
    planar, to_3D = section.to_2D()
    polys = planar.polygons_full
    if not polys:
        return None
    origin_2d = trimesh.transform_points(origin.reshape(1, 3), np.linalg.inv(to_3D))[0][:2]
    pt = Point(origin_2d)
    containing = [p for p in polys if p.contains(pt)]
    best_poly = min(containing, key=lambda p: p.area) if containing else \
        min(polys, key=lambda p: p.centroid.distance(pt))
    if best_poly.area <= 0:
        return None
    return float(np.sqrt(best_poly.area / np.pi))


def validate_cross_sections(entry, trim_start_mm, trim_end_mm, max_ray_dist=50.0):
    name = entry["name"]
    print(f"\n{'='*70}\nCross-section validation: {name}\n{'='*70}")
    t0 = time.time()

    # use the ray-distance ceiling validate_batch.py determined for this dataset
    cf.MAX_RAY_DIST = max_ray_dist

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    lo = arc[0] + trim_start_mm
    hi = arc[-1] - trim_end_mm
    sample_ts = np.linspace(lo, hi, N_SAMPLES_PER_DATASET)

    angle_positions = np.arange(0, 360, 1.0)
    hit_distances, *_ = cf._cast_rays_at_positions(
        verts, faces, cl_pts, arc, normals, binormals, sample_ts, angle_positions)
    raycast_radius = np.nanmean(hit_distances, axis=1)

    rows = []
    for i, t in enumerate(sample_ts):
        C, N, B = cf.interpolate_centerline_frame(np.array([t]), arc, cl_pts, normals, binormals)
        T = np.cross(N[0], B[0])
        true_r = cross_section_radius(mesh, C[0], T)
        rc_r = float(raycast_radius[i])
        if true_r is None or np.isnan(rc_r):
            continue
        diff_mm = rc_r - true_r
        rows.append({
            "arc_mm": round(float(t), 1),
            "true_radius_mm": round(true_r, 2),
            "raycast_radius_mm": round(rc_r, 2),
            "diff_mm": round(diff_mm, 2),
            "diff_pct": round(100 * diff_mm / true_r, 1),
        })
        print(f"  t={t:7.1f}mm  true={true_r:6.2f}mm  raycast={rc_r:6.2f}mm  "
              f"diff={diff_mm:+6.2f}mm ({100*diff_mm/true_r:+5.1f}%)")

    diffs_mm = np.array([r["diff_mm"] for r in rows])
    diffs_pct = np.array([r["diff_pct"] for r in rows])
    summary = {
        "name": name,
        "quality": entry.get("quality", "unknown"),
        "n_samples": len(rows),
        "mean_abs_diff_mm": round(float(np.abs(diffs_mm).mean()), 2),
        "median_abs_diff_mm": round(float(np.median(np.abs(diffs_mm))), 2),
        "max_abs_diff_mm": round(float(np.abs(diffs_mm).max()), 2),
        "mean_abs_diff_pct": round(float(np.abs(diffs_pct).mean()), 1),
        "median_abs_diff_pct": round(float(np.median(np.abs(diffs_pct))), 1),
        "max_abs_diff_pct": round(float(np.abs(diffs_pct).max()), 1),
        "samples": rows,
        "time_s": round(time.time() - t0, 1),
    }
    print(f"  -- {name}: mean_abs_diff={summary['mean_abs_diff_mm']}mm "
          f"({summary['mean_abs_diff_pct']}%), max={summary['max_abs_diff_mm']}mm "
          f"({summary['max_abs_diff_pct']}%), n={summary['n_samples']}")
    return summary


if __name__ == "__main__":
    with open(VALIDATE_BATCH_RESULTS) as f:
        prior = {r["name"]: r for r in json.load(f)}

    results = []
    for entry in DATASETS:
        prev = prior.get(entry["name"])
        trim_start = prev["trim_start_mm"] if prev else 1.0
        trim_end = prev["trim_end_mm"] if prev else 1.0
        max_ray_dist = prev["max_ray_dist_mm"] if prev else 50.0
        try:
            results.append(validate_cross_sections(entry, trim_start, trim_end, max_ray_dist))
        except Exception as e:
            print(f"FAILED on {entry['name']}: {e}")

    with open(os.path.join(OUT_DIR, "cross_section_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    cols = ["name", "quality", "n_samples", "mean_abs_diff_mm", "mean_abs_diff_pct",
            "max_abs_diff_mm", "max_abs_diff_pct"]
    header = " | ".join(c.ljust(16) for c in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        print(" | ".join(str(r[c]).ljust(16) for c in cols))

    all_pct = np.array([r["mean_abs_diff_pct"] for r in results])
    print(f"\nOverall mean of per-dataset mean_abs_diff_pct: {all_pct.mean():.1f}%")
    print("Saved cross_section_results.json")
