"""Jump-artefact mechanism audit: region sampling and export.

Clusters adjacent jump cells (>15mm, same threshold as the batch harness)
into connected (arc, angle) regions, with wrap-around at the 359deg/0deg
seam. Randomly samples 20 regions per dataset (all if fewer), and for each
sampled region reconstructs the 3D ray origin and the two hit points on
either side of the region's largest-magnitude jump, using the same
(arc, angle, radius) -> 3D formula as the polyp position round-trip. Also
records each sampled region's distance to the nearest bridge vertex.

Classifying each sampled region is a manual visual step done in 3D Slicer
and is not done here. This script exports, per dataset, a Slicer-loadable
Markups fiducial file (one point per sampled region's ray origin and two
hit points, labeled with the region id) so the region locations can be
navigated to directly in 3D Slicer.

Run from the repo root: python analysis/jump_mechanism_audit/jump_mechanism_audit.py
"""
import json
import os
import sys
import time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS, TOUCH_MULTIPLIER

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_SAMPLE = 20
JUMP_THRESHOLD_MM = 15.0

# datasets audited: 1840, 518 and 6651
TARGET_NAMES = ["1840", "518", "6651"]


def reconstruct_point(arc_val, angle_val, radius_val, arc, cl_pts, normals, binormals):
    """Same formula as the polyp position round-trip / containment check,
    wrapped for a single scalar sample. Returns (hit_point, ray_origin)."""
    C, N, B = cf.interpolate_centerline_frame(np.array([arc_val]), arc, cl_pts, normals, binormals)
    angle_rad = np.radians(angle_val)
    hit = C[0] + radius_val * (np.cos(angle_rad) * N[0] + np.sin(angle_rad) * B[0])
    return hit, C[0]


def cluster_jump_regions(d_arc, d_angle, n_arc, n_angle, threshold=JUMP_THRESHOLD_MM):
    """Step 2: cell-level jump mask -> connected regions, with angle
    wrap-around (region touching column 0 and column n_angle-1 merged)."""
    jump_mask = np.zeros((n_arc, n_angle), dtype=bool)
    ia, ja = np.where(d_arc > threshold)
    jump_mask[ia, ja] = True
    jump_mask[ia + 1, ja] = True
    ib, jb = np.where(d_angle > threshold)
    jump_mask[ib, jb] = True
    jump_mask[ib, jb + 1] = True

    labeled, n_labels = ndimage.label(jump_mask)

    if n_labels > 0:
        left, right = labeled[:, 0], labeled[:, -1]
        parent = list(range(n_labels + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n_arc):
            l, r = int(left[i]), int(right[i])
            if l > 0 and r > 0 and l != r:
                union(l, r)

        remap = {old: find(old) for old in range(1, n_labels + 1)}
        merged = np.zeros_like(labeled)
        for old_label, new_label in remap.items():
            merged[labeled == old_label] = new_label
        labeled = merged
        unique_labels = sorted(set(remap.values()))
    else:
        unique_labels = []

    return labeled, unique_labels, jump_mask, (ia, ja, ib, jb)


def audit_one(entry, rng):
    name = entry["name"]
    print(f"\n{'='*70}\nJump-mechanism audit: {name}\n{'='*70}", flush=True)
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
    n_arc, n_angle = filled.shape

    bridge_edges = cf.find_bridge_edges(
        verts, faces, cl_pts, arc, radius_arc=arc_steps, radius_values=row_radius,
        touch_multiplier=TOUCH_MULTIPLIER)
    bridge_vertex_idx = sorted(set(i for e in bridge_edges for i in e))
    bridge_tree = cKDTree(verts[bridge_vertex_idx]) if len(bridge_vertex_idx) >= 1 else None

    d_arc = np.abs(np.diff(filled, axis=0))
    d_angle = np.abs(np.diff(filled, axis=1))

    labeled, unique_labels, jump_mask, (ia, ja, ib, jb) = cluster_jump_regions(
        d_arc, d_angle, n_arc, n_angle)

    # region -> list of cells, and list of jump edges (both directions) within it
    region_cells = {}
    for lab in unique_labels:
        ii, jj = np.where(labeled == lab)
        region_cells[lab] = list(zip(ii.tolist(), jj.tolist()))

    cell_to_region = {}
    for lab, cells in region_cells.items():
        for c in cells:
            cell_to_region[c] = lab

    # every jump edge, tagged with the region its two cells belong to (should
    # always be the same region by construction) and its magnitude
    edges_by_region = {lab: [] for lab in unique_labels}
    for i, j in zip(ia.tolist(), ja.tolist()):
        lab = cell_to_region.get((i, j)) or cell_to_region.get((i + 1, j))
        if lab is not None:
            edges_by_region[lab].append(("arc", i, j, i + 1, j, float(d_arc[i, j])))
    for i, j in zip(ib.tolist(), jb.tolist()):
        lab = cell_to_region.get((i, j)) or cell_to_region.get((i, j + 1))
        if lab is not None:
            edges_by_region[lab].append(("angle", i, j, i, j + 1, float(d_angle[i, j])))

    region_sizes = {lab: len(region_cells[lab]) for lab in unique_labels}
    n_regions = len(unique_labels)
    print(f"  {n_regions} jump regions, sizes: min={min(region_sizes.values()) if n_regions else 0} "
          f"max={max(region_sizes.values()) if n_regions else 0} "
          f"median={float(np.median(list(region_sizes.values()))) if n_regions else 0}", flush=True)

    # Step 3: random sample
    labels_list = unique_labels
    n_sample = min(N_SAMPLE, n_regions)
    sampled_labels = rng.choice(labels_list, size=n_sample, replace=False).tolist() if n_regions else []
    sampled_labels = sorted(int(x) for x in sampled_labels)

    # Step 4 + 6: reconstruct 3D positions + bridge-vertex distance for sampled regions
    sampled_regions = []
    for lab in sampled_labels:
        edges = edges_by_region[lab]
        edges.sort(key=lambda e: e[-1], reverse=True)
        direction, i1, j1, i2, j2, magnitude = edges[0]

        arc1, angle1, r1 = float(arc_steps[i1]), float(angle_steps[j1]), float(filled[i1, j1])
        arc2, angle2, r2 = float(arc_steps[i2]), float(angle_steps[j2]), float(filled[i2, j2])

        hit1, origin1 = reconstruct_point(arc1, angle1, r1, arc, cl_pts, normals, binormals)
        hit2, origin2 = reconstruct_point(arc2, angle2, r2, arc, cl_pts, normals, binormals)

        if bridge_tree is not None:
            d1, _ = bridge_tree.query(hit1)
            d2, _ = bridge_tree.query(hit2)
            bridge_dist = float(min(d1, d2))
        else:
            bridge_dist = None

        sampled_regions.append({
            "region_id": lab,
            "size_cells": region_sizes[lab],
            "n_edges_in_region": len(edges),
            "representative_jump": {
                "direction": direction,
                "magnitude_mm": round(magnitude, 2),
                "cell1": {"arc_mm": round(arc1, 1), "angle_deg": round(angle1, 1),
                          "radius_mm": round(r1, 2), "grid_ij": [i1, j1]},
                "cell2": {"arc_mm": round(arc2, 1), "angle_deg": round(angle2, 1),
                          "radius_mm": round(r2, 2), "grid_ij": [i2, j2]},
            },
            "hit_point_1": [round(float(x), 2) for x in hit1],
            "hit_point_2": [round(float(x), 2) for x in hit2],
            "ray_origin_1": [round(float(x), 2) for x in origin1],
            "ray_origin_2": [round(float(x), 2) for x in origin2],
            "dist_to_nearest_bridge_vertex_mm": round(bridge_dist, 2) if bridge_dist is not None else None,
            "category": None,  # to be filled in manually (step 5)
        })

    result = {
        "dataset": name,
        "quality": entry.get("quality", "unknown"),
        "grid_shape": [n_arc, n_angle],
        "n_bridge_vertices": len(bridge_vertex_idx),
        "n_regions_total": n_regions,
        "region_sizes_all": [region_sizes[lab] for lab in unique_labels],
        "n_sampled": len(sampled_regions),
        "random_seed": SEED,
        "jump_threshold_mm": JUMP_THRESHOLD_MM,
        "sampled_regions": sampled_regions,
        "time_s": round(time.time() - t0, 1),
    }
    print(f"  sampled {len(sampled_regions)}/{n_regions} regions in {result['time_s']}s", flush=True)
    return result


def write_slicer_fiducials(result, out_path):
    """Step 8 support: a Slicer Markups fiducial file with one point per
    sampled region's ray origins and hit points, for direct navigation."""
    control_points = []
    idx = 0
    for reg in result["sampled_regions"]:
        rid = reg["region_id"]
        for label_suffix, key in [("origin1", "ray_origin_1"), ("origin2", "ray_origin_2"),
                                   ("hit1", "hit_point_1"), ("hit2", "hit_point_2")]:
            idx += 1
            control_points.append({
                "id": str(idx),
                "label": f"R{rid}_{label_suffix}",
                "position": reg[key],
                "selected": True,
                "visibility": True,
            })
    doc = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": "RAS",
            "coordinateUnits": "mm",
            "controlPoints": control_points,
        }],
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)


if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    print(f"Random seed: {SEED}", flush=True)

    all_results = []
    for name in TARGET_NAMES:
        entry = next(e for e in DATASETS if e["name"] == name)
        try:
            r = audit_one(entry, rng)
            all_results.append(r)
            with open(os.path.join(OUT_DIR, f"jump_regions_{name}.json"), "w") as f:
                json.dump(r, f, indent=2)
            fid_path = os.path.join(OUT_DIR, f"jump_regions_{name}_fiducials.mrk.json")
            write_slicer_fiducials(r, fid_path)
            print(f"  Saved jump_regions_{name}.json and fiducials file", flush=True)
        except Exception as e:
            print(f"FAILED on {name}: {e}", flush=True)

    with open(os.path.join(OUT_DIR, "jump_regions_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in all_results:
        sizes = r["region_sizes_all"]
        print(f"{r['dataset']:6s}  regions={r['n_regions_total']:5d}  "
              f"sampled={r['n_sampled']:3d}  "
              f"size(min/median/max)={min(sizes) if sizes else 0}/"
              f"{np.median(sizes) if sizes else 0}/{max(sizes) if sizes else 0}")

    print(f"\nALL DONE in {time.time()-t_start:.0f}s", flush=True)
