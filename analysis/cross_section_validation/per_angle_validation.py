"""Per-angle cross-sectional comparison.

At each of the 20 sampled arc positions per dataset (same
trim_start_mm/trim_end_mm as validate_batch.py, so the same planes as
cross_section_validation.py), the mesh cross-section polygon is intersected
with the same 360 radial directions the ray-caster uses, in the same local
(N, B) frame, giving a reference distance at every angle that is compared
with the ray-cast's per-angle hit distance.

d_ray(theta) is read from the raw ray-cast grid (before gap-filling).

Run from the repo root: python analysis/cross_section_validation/per_angle_validation.py
"""
import json
import os
import sys
import time
import numpy as np
import trimesh
from shapely.geometry import Point, LineString

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATE_BATCH_RESULTS = os.path.join(_ROOT, "analysis", "validate_batch", "validation_results.json")

N_SAMPLES_PER_DATASET = 20
DIFF_THRESHOLD_MM = 2.0
RAY_FAR_MM = 300.0  # far enough past any realistic colon cross-section radius


def section_polygon_in_frame(mesh, origin, tangent, N, B):
    """True mesh cross-section at `origin`, re-expressed as a 2D polygon in
    the ray-caster's own (N, B) local frame (not trimesh's arbitrary 2D
    parameterisation of the section plane), so angle theta=0 here means the
    same direction as angle theta=0 to the ray-caster.
    """
    section = mesh.section(plane_origin=origin, plane_normal=tangent)
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

    ext_2d = np.asarray(best_poly.exterior.coords)  # trimesh's local 2D frame
    ext_2d_padded = np.column_stack([ext_2d, np.zeros(len(ext_2d))])
    ext_3d = trimesh.transform_points(ext_2d_padded, to_3D)  # back to world 3D
    rel = ext_3d - origin.reshape(1, 3)
    local_n = rel @ N
    local_b = rel @ B
    from shapely.geometry import Polygon
    poly_local = Polygon(np.column_stack([local_n, local_b]))
    if not poly_local.is_valid:
        poly_local = poly_local.buffer(0)
    if poly_local.is_empty or poly_local.area <= 0:
        return None
    return poly_local


def per_angle_ref_distances(poly_local, angle_positions):
    """d_ref(theta) for every angle: distance from the local origin (0,0) to
    the nearest polygon-boundary crossing along that ray direction, using
    shapely LineString intersection, matching how the ray-caster itself
    returns the nearest hit along a ray.
    """
    boundary = poly_local.exterior
    n = len(angle_positions)
    d_ref = np.full(n, np.nan)
    angle_rads = np.deg2rad(angle_positions)
    cos_a = np.cos(angle_rads)
    sin_a = np.sin(angle_rads)
    for i in range(n):
        end = (cos_a[i] * RAY_FAR_MM, sin_a[i] * RAY_FAR_MM)
        ray = LineString([(0.0, 0.0), end])
        inter = ray.intersection(boundary)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            pts = [inter]
        elif inter.geom_type == "MultiPoint":
            pts = list(inter.geoms)
        else:
            # LineString/GeometryCollection (ray runs briefly along an edge) --
            # take all point-like coordinates available
            pts = [Point(c) for g in getattr(inter, "geoms", [inter])
                   for c in (g.coords if hasattr(g, "coords") else [])]
            if not pts:
                continue
        dists = [np.hypot(p.x, p.y) for p in pts]
        d_ref[i] = min(dists)
    return d_ref


def validate_one(entry, trim_start_mm, trim_end_mm, max_ray_dist_mm):
    name = entry["name"]
    print(f"\n{'='*70}\nPer-angle cross-section validation: {name}\n{'='*70}")
    t0 = time.time()

    cf.MAX_RAY_DIST = max_ray_dist_mm

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    lo = arc[0] + trim_start_mm
    hi = arc[-1] - trim_end_mm
    sample_ts = np.linspace(lo, hi, N_SAMPLES_PER_DATASET)  # same 20 planes as cross_section_validation.py

    angle_positions = np.arange(0, 360, cf.ANGLE_STEP_DEG)
    # raw (pre-gap-fill) ray-cast grid at these 20 planes, all 360 angles
    hit_distances, *_ = cf._cast_rays_at_positions(
        verts, faces, cl_pts, arc, normals, binormals, sample_ts, angle_positions)

    plane_summaries = []
    for i, t in enumerate(sample_ts):
        C, Nv, Bv = cf.interpolate_centerline_frame(np.array([t]), arc, cl_pts, normals, binormals)
        C, Nv, Bv = C[0], Nv[0], Bv[0]
        T = np.cross(Nv, Bv)

        poly_local = section_polygon_in_frame(mesh, C, T, Nv, Bv)
        if poly_local is None:
            print(f"  t={t:7.1f}mm  no valid section polygon, skipped")
            continue

        d_ref = per_angle_ref_distances(poly_local, angle_positions)
        d_ray = hit_distances[i].copy()

        both_valid = ~np.isnan(d_ref) & ~np.isnan(d_ray)
        one_only = (~np.isnan(d_ref)) ^ (~np.isnan(d_ray))
        n_both = int(both_valid.sum())
        n_one_only = int(one_only.sum())

        if n_both == 0:
            print(f"  t={t:7.1f}mm  no angle with both a ray hit and a polygon hit, skipped")
            continue

        diffs = np.abs(d_ray[both_valid] - d_ref[both_valid])
        median_diff = float(np.median(diffs))
        p95_diff = float(np.percentile(diffs, 95))
        pct_gt_thresh = float(100.0 * (diffs > DIFF_THRESHOLD_MM).mean())

        plane_summaries.append({
            "arc_mm": round(float(t), 6),
            "n_angles_both": n_both,
            "n_angles_one_only": n_one_only,
            "median_abs_diff_mm": round(median_diff, 6),
            "p95_abs_diff_mm": round(p95_diff, 6),
            "pct_angles_gt_2mm": round(pct_gt_thresh, 6),
        })
        print(f"  t={t:7.1f}mm  n_both={n_both:3d}  one_only={n_one_only:3d}  "
              f"median|diff|={median_diff:9.6f}mm  P95|diff|={p95_diff:9.6f}mm  "
              f">2mm={pct_gt_thresh:8.4f}%")

    if not plane_summaries:
        return {"name": name, "quality": entry.get("quality", "unknown"), "n_planes": 0,
                "time_s": round(time.time() - t0, 1)}

    plane_medians = np.array([p["median_abs_diff_mm"] for p in plane_summaries])
    plane_p95s = np.array([p["p95_abs_diff_mm"] for p in plane_summaries])
    plane_pct_gt = np.array([p["pct_angles_gt_2mm"] for p in plane_summaries])
    total_one_only = sum(p["n_angles_one_only"] for p in plane_summaries)
    total_both = sum(p["n_angles_both"] for p in plane_summaries)

    summary = {
        "name": name,
        "quality": entry.get("quality", "unknown"),
        "n_planes": len(plane_summaries),
        "dataset_median_of_plane_medians_mm": round(float(np.median(plane_medians)), 6),
        "dataset_max_of_plane_p95_mm": round(float(np.max(plane_p95s)), 6),
        "dataset_mean_pct_angles_gt_2mm": round(float(plane_pct_gt.mean()), 6),
        "total_angles_compared": int(total_both),
        "total_angles_one_hit_only": int(total_one_only),
        "planes": plane_summaries,
        "time_s": round(time.time() - t0, 1),
    }
    print(f"  -- {name}: median-of-plane-medians={summary['dataset_median_of_plane_medians_mm']}mm, "
          f"max-of-plane-P95={summary['dataset_max_of_plane_p95_mm']}mm, "
          f"mean %>2mm={summary['dataset_mean_pct_angles_gt_2mm']}%, "
          f"one-hit-only={total_one_only}/{total_both + total_one_only} ({time.time()-t0:.1f}s)")
    return summary


if __name__ == "__main__":
    with open(VALIDATE_BATCH_RESULTS) as f:
        prior = {r["name"]: r for r in json.load(f)}

    old_path = os.path.join(OUT_DIR, "cross_section_results.json")
    old_results = {}
    if os.path.exists(old_path):
        with open(old_path) as f:
            old_results = {r["name"]: r for r in json.load(f)}

    results = []
    for entry in DATASETS:
        prev = prior.get(entry["name"])
        trim_start = prev["trim_start_mm"] if prev else 1.0
        trim_end = prev["trim_end_mm"] if prev else 1.0
        max_ray_dist = prev["max_ray_dist_mm"] if prev else 50.0
        try:
            results.append(validate_one(entry, trim_start, trim_end, max_ray_dist))
        except Exception as e:
            print(f"FAILED on {entry['name']}: {e}")
            import traceback
            traceback.print_exc()

    with open(os.path.join(OUT_DIR, "per_angle_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY -- per-angle vs. existing equivalent-radius comparison\n{'='*70}")
    cols = ["name", "quality", "n_planes", "dataset_median_of_plane_medians_mm",
            "dataset_max_of_plane_p95_mm", "dataset_mean_pct_angles_gt_2mm"]
    header = " | ".join(c.ljust(20) for c in cols) + " | old_mean_abs_diff_pct"
    print(header)
    print("-" * len(header))
    for r in results:
        row = " | ".join(str(r.get(c, "n/a")).ljust(20) for c in cols)
        old_pct = old_results.get(r["name"], {}).get("mean_abs_diff_pct", "n/a")
        print(f"{row} | {old_pct}")

    valid = [r for r in results if r.get("n_planes", 0) > 0]
    if valid:
        overall_median = np.median([r["dataset_median_of_plane_medians_mm"] for r in valid])
        overall_max_p95 = np.max([r["dataset_max_of_plane_p95_mm"] for r in valid])
        print(f"\nOverall median of dataset medians: {overall_median:.2f}mm")
        print(f"Overall max of dataset P95s: {overall_max_p95:.2f}mm")
    print("Saved per_angle_results.json")
