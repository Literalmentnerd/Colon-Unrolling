"""Generate the unrolled 2.5D mesh (STL) for every dataset in
validate_batch.DATASETS and save it into that dataset's own folder,
alongside its mask and centerline.
"""
import os
import sys
import time
import numpy as np
import colon_core as cf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "validate_batch"))
from validate_batch import DATASETS


def generate_dataset_stl(entry, mesh_upsample=1):
    name = entry["name"]
    print(f"\n{'='*70}\nGenerating: {name}\n{'='*70}")
    cf.MESH_UPSAMPLE = mesh_upsample
    t0 = time.time()

    cl_pts = cf.load_centerline_slicer(entry["centerline"])
    cl_pts = cf.resample_centerline(cl_pts, step_mm=cf.ARC_STEP_MM)
    verts, faces = cf.load_nifti_mesh(entry["nifti"], label_value=entry["label"])
    arc, seg_lens, tangents, normals, binormals = cf.build_centerline_frame(cl_pts)

    # adaptive ray-distance ceiling (colon_core.cast_rays_adaptive), as in validate_batch.py
    arc_steps, angle_steps, hit_distances, *_ = cf.cast_rays_adaptive(
        verts, faces, cl_pts, arc, normals, binormals,
        base_ray_dist=entry.get("max_ray_dist", 50.0))
    unrolled_verts, unrolled_faces = cf.build_25d_mesh(arc_steps, angle_steps, hit_distances)

    out_dir = os.path.dirname(entry["nifti"])
    out_path = os.path.join(out_dir, f"colon_unrolled_{name}.stl")
    cf.save_stl(unrolled_verts, unrolled_faces, out_path)
    print(f"Total time: {time.time()-t0:.1f}s")
    return out_path


if __name__ == "__main__":
    saved = []
    for entry in DATASETS:
        try:
            saved.append(generate_dataset_stl(entry))
        except Exception as e:
            print(f"FAILED on {entry['name']}: {e}")

    print(f"\n{'='*70}\nDone. Saved {len(saved)}/{len(DATASETS)} STL files:")
    for p in saved:
        print(f"  {p}")
