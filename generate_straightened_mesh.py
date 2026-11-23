"""Generate the vertex-based straightened mesh (colon_core.straighten_mesh)
for the datasets listed in TARGETS, run with blocked_edges=None, and save
each STL into that dataset's own folder.
"""
import os
import sys
import time
import numpy as np
import colon_core as cf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "validate_batch"))
from validate_batch import DATASETS

TARGETS = ["2320", "3114", "4371", "563", "5980", "6651", "7889", "1840", "3828", "7198"]


def generate_one(entry):
    name = entry["name"]
    print(f"\n{'='*70}\nStraightening (vertex-based, straighten_mesh): {name}\n{'='*70}")
    t0 = time.time()
    cf.MESH_UPSAMPLE = 1

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)

    print(f"Mesh: {len(verts)} vertices, {len(faces)} faces. Starting flood-fill straighten...")
    new_verts = cf.straighten_mesh(verts, faces, cl_pts, arc, seg_lens, tangents, normals, binormals,
                                    blocked_edges=None)

    out_dir = os.path.dirname(entry["nifti"])
    out_path = os.path.join(out_dir, f"colon_straightened_{name}.stl")
    cf.save_stl(new_verts, faces, out_path)
    print(f"Saved: {out_path}  ({time.time()-t0:.1f}s)")
    return out_path


if __name__ == "__main__":
    for entry in DATASETS:
        if entry["name"] in TARGETS:
            generate_one(entry)
