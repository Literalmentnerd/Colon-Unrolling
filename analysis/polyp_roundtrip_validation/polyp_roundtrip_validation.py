"""Polyp position round-trip validation.

For each marked polyp (its label's connected-component region):
  1. Project the centroid onto the centerline to get a reference (arc, angle,
     radial distance) position.
  2. Snap to the nearest (arc_step, angle_step) cell of the ray-cast grid and
     read the wall's ray-cast hit distance there.
  3. Invert that grid position back to a 3D point via the same
     interpolate_centerline_frame + radial-ray reconstruction the ray-caster
     uses.
  4. Position error = 3D distance from the recovered point to the nearest
     voxel of that polyp's labeled region. Centroid distance is also
     recorded, for reference.

Size preservation is not computed here.
"""
import json
import numpy as np
import os
import sys
import nibabel as nib
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import colon_core as cf

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NIFTI_PATH = "787_polyp_placement_verification.nii"
CENTERLINE_PATH = "Dataset/787/Centerline curve 787.mrk.json"
COLON_LABEL = 1
POLYP_LABEL = 2


def load_polyp_components(path, label_value=2):
    """Like cf.load_polyp_centroids, but also keeps each connected
    component's own voxel world-coordinates (needed for nearest-voxel
    ground-truth distance, not just the centroid)."""
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    mask = (np.round(data) == label_value).astype(np.uint8)
    del data
    nonzero = np.argwhere(mask)
    lo, hi = nonzero.min(axis=0), nonzero.max(axis=0) + 1
    mask_crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    del mask
    labeled, n = cc_label(mask_crop)
    del mask_crop
    components = []
    for i in range(1, n + 1):
        vox = np.argwhere(labeled == i) + lo
        ones = np.ones((len(vox), 1))
        world = (affine @ np.hstack([vox, ones]).T).T[:, :3]
        components.append(world)
    return components


def true_polyp_position(pt, cl_pts, seg_lens, arc, tangents, normals, binormals):
    """Continuous, ray-cast-independent ground truth: nearest point on the
    centerline (any segment, not snapped to a ray-cast grid step)."""
    n_cl = len(cl_pts)
    best_dist, best_t, best_n, best_b = np.inf, 0.0, 0.0, 0.0
    for i in range(n_cl - 1):
        dist, t_coord, n_coord, b_coord = cf.project_onto_segment(
            pt, i, cl_pts, seg_lens, arc, tangents, normals, binormals)
        if dist < best_dist:
            best_dist, best_t, best_n, best_b = dist, t_coord, n_coord, b_coord
    angle_deg = np.degrees(np.arctan2(best_b, best_n)) % 360.0
    radial = float(np.hypot(best_n, best_b))
    return best_t, angle_deg, radial


def main(mesh_upsample=1, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUT_DIR, "polyp_roundtrip_results.json")
    # mesh_upsample=1 by default, as in validate_batch.py; overridable
    cf.MESH_UPSAMPLE = mesh_upsample
    cl_pts = cf.load_centerline_slicer(CENTERLINE_PATH)
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(NIFTI_PATH, label_value=COLON_LABEL)
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)

    polyp_pts = cf.load_polyp_centroids(NIFTI_PATH, label_value=POLYP_LABEL)
    polyp_components = load_polyp_components(NIFTI_PATH, label_value=POLYP_LABEL)
    component_trees = [cKDTree(comp) for comp in polyp_components]

    p_angles = cf.polyp_angles_in_frame(
        polyp_pts, cl_pts, arc, seg_lens, tangents, normals, binormals)
    cut_angle = cf.find_best_cut_angle(p_angles)

    arc_steps, angle_steps, hit_distances, final_max_ray_dist, near_cap_probe_pct, recast = \
        cf.cast_rays_adaptive(verts, faces, cl_pts, arc, normals, binormals,
                               cut_angle_deg=cut_angle)
    filled = cf.fill_gaps_smooth(hit_distances)
    mean_radius = float(np.nanmean(filled))

    print(f"\n{'='*70}\nPolyp round-trip POSITION validation\n{'='*70}")
    print(f"mean_radius={mean_radius:.2f}mm  cut_angle={cut_angle:.1f}deg  "
          f"n_arc={len(arc_steps)}  n_angle={len(angle_steps)}\n")

    errors = []
    per_polyp = []
    for i, pt in enumerate(polyp_pts):
        true_t, true_angle_deg, true_radial = true_polyp_position(
            pt, cl_pts, seg_lens, arc, tangents, normals, binormals)

        # snap to the nearest ray-cast grid cell
        circ_diff = (angle_steps - true_angle_deg + 180) % 360 - 180
        angle_idx = int(np.argmin(np.abs(circ_diff)))
        arc_idx = int(np.argmin(np.abs(arc_steps - true_t)))

        wall_hit = filled[arc_idx, angle_idx]
        if np.isnan(wall_hit):
            wall_hit = mean_radius

        # invert: grid position -> 3D, using the same frame + radial-ray
        # construction the ray-caster uses
        C, N, B = cf.interpolate_centerline_frame(
            np.array([arc_steps[arc_idx]]), arc, cl_pts, normals, binormals)
        angle_rad = np.radians(angle_steps[angle_idx])
        recovered = C[0] + wall_hit * (np.cos(angle_rad) * N[0] + np.sin(angle_rad) * B[0])

        centroid_error = float(np.linalg.norm(recovered - pt))
        blob_dist, _ = component_trees[i].query(recovered)
        pos_error = float(blob_dist)
        arc_snap_err = float(abs(true_t - arc_steps[arc_idx]))
        angle_snap_err = float(abs(circ_diff[angle_idx]))
        errors.append(pos_error)
        per_polyp.append({
            "polyp_index": i + 1,
            "true_position": np.round(pt, 3).tolist(),
            "recovered_position": np.round(recovered, 3).tolist(),
            "pos_error_mm": round(pos_error, 4),
            "centroid_error_mm": round(centroid_error, 4),
            "true_radial_mm": round(true_radial, 3),
            "wall_hit_used_mm": round(float(wall_hit), 3),
            "arc_snap_err_mm": round(arc_snap_err, 4),
            "angle_snap_err_deg": round(angle_snap_err, 4),
        })
        print(f"Polyp {i+1}: true={np.round(pt,1).tolist()}  "
              f"recovered={np.round(recovered,1).tolist()}  "
              f"pos_error(nearest-voxel-of-blob)={pos_error:.2f}mm  "
              f"(centroid_error={centroid_error:.2f}mm, true_radial={true_radial:.1f}mm, "
              f"wall_hit_used={wall_hit:.1f}mm, "
              f"arc_snap_err={arc_snap_err:.2f}mm, "
              f"angle_snap_err={angle_snap_err:.2f}deg)")

    errors = np.array(errors)
    centroid_errors = np.array([p["centroid_error_mm"] for p in per_polyp])
    print(f"\nPosition error, nearest-voxel-of-blob (mm) across {len(errors)} polyps: "
          f"mean={errors.mean():.2f}  median={np.median(errors):.2f}  "
          f"min={errors.min():.2f}  max={errors.max():.2f}  std={errors.std():.2f}")
    print(f"(for reference, centroid-distance: mean={centroid_errors.mean():.2f}  "
          f"min={centroid_errors.min():.2f}  max={centroid_errors.max():.2f})")

    result = {
        "dataset": "787",
        "mesh_upsample": mesh_upsample,
        "pos_error_metric": "distance from the recovered 3D point to the nearest voxel "
                             "of that polyp's own labeled connected component (not the "
                             "centroid) -- see module docstring for rationale",
        "mean_radius_mm": round(mean_radius, 3),
        "cut_angle_deg": round(float(cut_angle), 2),
        "n_arc_steps": len(arc_steps),
        "n_angle_steps": len(angle_steps),
        "final_max_ray_dist_mm": final_max_ray_dist,
        "adaptive_recast": recast,
        "n_polyps": len(errors),
        "per_polyp": per_polyp,
        "pos_error_mm_summary": {
            "mean": round(float(errors.mean()), 4),
            "median": round(float(np.median(errors)), 4),
            "min": round(float(errors.min()), 4),
            "max": round(float(errors.max()), 4),
            "std": round(float(errors.std()), 4),
        },
        "centroid_error_mm_summary": {
            "mean": round(float(centroid_errors.mean()), 4),
            "median": round(float(np.median(centroid_errors)), 4),
            "min": round(float(centroid_errors.min()), 4),
            "max": round(float(centroid_errors.max()), 4),
            "std": round(float(centroid_errors.std()), 4),
        },
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
