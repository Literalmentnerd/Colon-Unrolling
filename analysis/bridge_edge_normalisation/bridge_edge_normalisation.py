"""Bridge-edge density normalisation.

Computes bridge_per_10k_edges = 10000 * bridge_edges / total_edges for the
12 main datasets plus the two AI segmentations of case 518, and compares
the resulting ranking of the 12 main datasets with the
bridge_edges_per_meter ranking.

total_edges is the number of unique edges in each mesh's face list.
surface_area_m2 is the sum of triangle areas, converted from mm^2.

Run from the repo root: python analysis/bridge_edge_normalisation/bridge_edge_normalisation.py
"""
import json
import os
import sys
import time
import numpy as np
from scipy.stats import spearmanr

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))
import colon_core as cf
from validate_batch import DATASETS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

VALIDATION_RESULTS = os.path.join(_ROOT, "analysis", "validate_batch", "validation_results.json")
AI_COMPARISON_RESULTS = os.path.join(_ROOT, "analysis", "ai_segmentation_comparison",
                                      "ai_segmentation_comparison_results.json")

# extra sources beyond the 12 main DATASETS: the two AI segmentations of case 518
EXTRA_SOURCES = [
    {"name": "TotalSegmentator", "nifti": "Dataset/TotalSegmentator/TotalSegmentator-colon-label.nii",
     "label": 1},
    {"name": "MONAI", "nifti": "Dataset/MONAI/MONAI-colon-label.nii", "label": 1},
]


def mesh_edges_and_area(verts, faces):
    """Unique edge count (not edge-instances) and total surface area (m^2)."""
    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.concatenate([e0, e1, e2], axis=0)
    unique_edges = np.unique(all_edges, axis=0)
    total_edges = len(unique_edges)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    tri_areas_mm2 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    surface_area_m2 = float(tri_areas_mm2.sum()) / 1e6
    return total_edges, surface_area_m2


def compute_one(name, nifti_path, label, bridge_edges, bridge_edges_per_m):
    print(f"\nLoading mesh: {name} ...", flush=True)
    t0 = time.time()
    cf.MESH_UPSAMPLE = 1
    verts, faces = cf.load_nifti_mesh(nifti_path, label_value=label)
    total_edges, surface_area_m2 = mesh_edges_and_area(verts, faces)
    bridge_per_10k = 10000.0 * bridge_edges / total_edges
    print(f"  {name}: total_edges={total_edges}, surface_area_m2={surface_area_m2:.4f}, "
          f"bridge_edges={bridge_edges}, bridge_per_10k_edges={bridge_per_10k:.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    return {
        "name": name,
        "n_verts": len(verts),
        "n_faces": len(faces),
        "total_edges": total_edges,
        "surface_area_m2": round(surface_area_m2, 4),
        "bridge_edge_count": bridge_edges,
        "bridge_edges_per_meter": bridge_edges_per_m,
        "bridge_per_10k_edges": round(bridge_per_10k, 3),
    }


if __name__ == "__main__":
    with open(VALIDATION_RESULTS) as f:
        main_results = {r["name"]: r for r in json.load(f)}
    with open(AI_COMPARISON_RESULTS) as f:
        ai_results = {r["name"]: r for r in json.load(f)}

    results_main = []
    for entry in DATASETS:
        name = entry["name"]
        r = main_results[name]
        results_main.append(compute_one(
            name, entry["nifti"], entry["label"],
            r["bridge_edge_count"], r["bridge_edges_per_meter"]))

    results_extra = []
    for entry in EXTRA_SOURCES:
        name = entry["name"]
        r = ai_results[name]
        results_extra.append(compute_one(
            name, entry["nifti"], entry["label"],
            r["bridge_edge_count"], r["bridge_edges_per_meter"]))

    with open(os.path.join(OUT_DIR, "bridge_edge_normalisation_main12.json"), "w") as f:
        json.dump(results_main, f, indent=2)
    with open(os.path.join(OUT_DIR, "bridge_edge_normalisation_extra.json"), "w") as f:
        json.dump(results_extra, f, indent=2)

    # Step 3: ranking comparison, 12 main datasets, raw per-metre vs normalised per-10k-edges
    per_m = [r["bridge_edges_per_meter"] for r in results_main]
    per_10k = [r["bridge_per_10k_edges"] for r in results_main]
    rho, pval = spearmanr(per_m, per_10k)

    ranking_result = {
        "n_datasets": len(results_main),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(pval),
        "ranking_unchanged": bool(rho > 0.9),
    }
    with open(os.path.join(OUT_DIR, "bridge_edge_ranking_check.json"), "w") as f:
        json.dump(ranking_result, f, indent=2)

    print(f"\n{'='*70}\nSpearman (bridge_edges_per_meter vs bridge_per_10k_edges), n=12\n{'='*70}")
    print(json.dumps(ranking_result, indent=2))

    print(f"\n{'='*70}\nSUMMARY -- 12 main datasets\n{'='*70}")
    print(f"{'name':8s} {'verts':>9s} {'edges':>9s} {'area_m2':>9s} {'bridge':>7s} "
          f"{'per_m':>9s} {'per_10k':>9s}")
    for r in results_main:
        print(f"{r['name']:8s} {r['n_verts']:9d} {r['total_edges']:9d} "
              f"{r['surface_area_m2']:9.4f} {r['bridge_edge_count']:7d} "
              f"{r['bridge_edges_per_meter']:9.2f} {r['bridge_per_10k_edges']:9.3f}")

    print(f"\n{'='*70}\nSUMMARY -- 518 segmentation-source comparison (Table 6.1 extension)\n{'='*70}")
    # reuse the already-computed "518" entry from results_main instead of reloading its mesh
    curated_row = dict(next(r for r in results_main if r["name"] == "518"))
    curated_row["name"] = "Curated (518)"
    all_518 = [curated_row] + results_extra
    with open(os.path.join(OUT_DIR, "table_6_1_normalised_row.json"), "w") as f:
        json.dump(all_518, f, indent=2)
    for r in all_518:
        print(f"{r['name']:18s} verts={r['n_verts']:7d} edges={r['total_edges']:7d} "
              f"bridge={r['bridge_edge_count']:5d} per_m={r['bridge_edges_per_meter']:8.2f} "
              f"per_10k={r['bridge_per_10k_edges']:7.3f}")

    print("\nALL DONE.")
