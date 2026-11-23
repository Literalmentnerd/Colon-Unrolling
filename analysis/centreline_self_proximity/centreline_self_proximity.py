"""
Self-proximity from the centreline alone, independent of the mask boundary.
For each centreline point (1 mm spacing), checks whether another centreline
point more than 100 mm away along the colon lies within d mm in space.
Reports, per dataset and per d, the share of centreline length that has such
a neighbour, and correlates it with bridge edges/m and with each distortion
measure in input.csv.

Run: python analysis/centreline_self_proximity/centreline_self_proximity.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "validate_batch"))
import colon_core as cc
from validate_batch import DATASETS

ARC_SEP_MM = 100.0
THRESHOLDS_MM = [30.0, 40.0, 43.0, 50.0]
OUTCOMES = ["bridge_edges_per_m", "stretched_edges_pct", "jump_pct", "xsec_mean_diff_pct"]


def close_fraction(cl, d):
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
    tree = cKDTree(cl)
    hit = np.zeros(len(cl), bool)
    for i, nbrs in enumerate(tree.query_ball_point(cl, d)):
        hit[i] = np.any(np.abs(arc[nbrs] - arc[i]) > ARC_SEP_MM)
    return 100.0 * hit.mean()


with open(ROOT / "analysis" / "A1_correlations" / "input.csv", newline="") as f:
    table = {r["label"]: r for r in csv.DictReader(f)}

rows = []
for e in DATASETS:
    cl = cc.resample_centerline(cc.load_centerline_slicer(str(ROOT / e["centerline"])))
    r = {"name": e["name"], "quality": e["quality"]}
    for d in THRESHOLDS_MM:
        r[f"close_pct_{int(d)}mm"] = close_fraction(cl, d)
    r.update({k: float(table[e["name"]][k]) for k in OUTCOMES})
    rows.append(r)

out = {"arc_separation_mm": ARC_SEP_MM, "datasets": rows, "correlations": {}}
for d in THRESHOLDS_MM:
    x = [r[f"close_pct_{int(d)}mm"] for r in rows]
    for k in OUTCOMES:
        y = [r[k] for r in rows]
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        out["correlations"][f"close_pct_{int(d)}mm__vs__{k}"] = {
            "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp}

with open(Path(__file__).with_name("centreline_self_proximity_results.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"{'dataset':8s} {'group':14s} " + " ".join(f"{int(d)}mm%".rjust(7) for d in THRESHOLDS_MM) + "  bridge/m")
for r in rows:
    print(f"{r['name']:8s} {r['quality']:14s} " + " ".join(f"{r[f'close_pct_{int(d)}mm']:7.1f}" for d in THRESHOLDS_MM)
          + f"  {r['bridge_edges_per_m']:8.1f}")
for key, v in out["correlations"].items():
    print(f"{key:48s} r={v['pearson_r']:6.3f} p={v['pearson_p']:.3f}  rho={v['spearman_rho']:6.3f} p={v['spearman_p']:.3f}")
