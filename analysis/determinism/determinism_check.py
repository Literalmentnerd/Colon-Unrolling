"""Determinism check: does a full from-scratch re-run of the pipeline on the
same dataset produce bit-identical output? Run this script twice (as two
separate process invocations) with a different tag each time, then diff the
two saved results.

Uses 1840. Mirrors validate_batch.py's validate_dataset() ray-casting and
bridge-edge calls (cast_rays_adaptive, then the local-radius-calibrated
find_bridge_edges call).

Run from the repo root:
    python analysis/determinism/determinism_check.py A
    python analysis/determinism/determinism_check.py B
Output JSONs are written next to this script regardless of cwd.
"""
import hashlib
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS, TOUCH_MULTIPLIER

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

entry = next(e for e in DATASETS if e["name"] == "1840")

t0 = time.time()
cf.MESH_UPSAMPLE = 1

cl_pts = cf.load_centerline_slicer(entry["centerline"])
cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)

# adaptive ray-distance ceiling, as validate_batch.py calls it
arc_steps, angle_steps, hit_distances, final_max_ray_dist, near_cap_probe_pct, recast = \
    cf.cast_rays_adaptive(verts, faces, cl_pts, arc, normals, binormals,
                           base_ray_dist=entry.get("max_ray_dist", 50.0))

# local-radius-calibrated bridge-edge count, as validate_batch.py computes it
filled = cf.fill_gaps_smooth(hit_distances)
row_radius = np.nanmean(filled, axis=1)
bridge_edges_raw = cf.find_bridge_edges(verts, faces, cl_pts, arc)
bridge_edges = cf.find_bridge_edges(verts, faces, cl_pts, arc,
                                     radius_arc=arc_steps, radius_values=row_radius,
                                     touch_multiplier=TOUCH_MULTIPLIER)

# hash the raw arrays (NaNs included, via a fixed-bitpattern view) so
# "identical" means bit-identical, not just matching rounded summary stats
hd_bytes = np.nan_to_num(hit_distances, nan=-1.0).tobytes()
hd_hash = hashlib.sha256(hd_bytes).hexdigest()
arc_steps_hash = hashlib.sha256(arc_steps.tobytes()).hexdigest()

result = {
    "tag": TAG,
    "n_verts": len(verts),
    "n_faces": len(faces),
    "n_arc_steps": len(arc_steps),
    "arc_steps_hash": arc_steps_hash,
    "hit_distances_shape": list(hit_distances.shape),
    "hit_distances_hash": hd_hash,
    "hit_distances_nanmean": float(np.nanmean(hit_distances)),
    "hit_distances_nansum": float(np.nansum(hit_distances)),
    "n_nan": int(np.isnan(hit_distances).sum()),
    "final_max_ray_dist_mm": final_max_ray_dist,
    "adaptive_recast": recast,
    "bridge_edge_count": len(bridge_edges),
    "bridge_edge_count_uncapped": len(bridge_edges_raw),
    "time_s": round(time.time() - t0, 1),
}

with open(os.path.join(OUT_DIR, f"determinism_{TAG}.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
