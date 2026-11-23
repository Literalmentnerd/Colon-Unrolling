"""
Recomputes every Pearson/Spearman correlation reported in the thesis from the
per-dataset values in input.csv (assembled from validation_results.json,
straighten_mesh_distortion_results.json and cross_section_results.json) and
from sato_batch_results.json and per_angle_results.json, with leave-one-out ranges and a bootstrap
confidence interval for each pair. input.csv holds full-precision values.

Run: python analysis/A1_correlations/correlations.py
"""

import csv
import itertools
import json
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

CSV_PATH = Path(__file__).resolve().parent / "input.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

PAIRS = [
    ("bridge_edges_per_m", "stretched_edges_pct", "bridge vs stretched (vertex-based method distortion)"),
    ("bridge_edges_per_m", "jump_cells", "bridge vs jump count (RQ3 batch validation)"),
    ("bridge_edges_per_m", "jump_pct", "bridge vs jump % (A2: normalised by valid_cells)"),
    ("bridge_edges_per_m", "xsec_mean_diff_pct", "bridge vs cross-sectional disagreement"),
    ("ray_miss_pct", "xsec_mean_diff_pct", "ray-completeness vs cross-sectional disagreement"),
]

# Same pairs restricted to the ten residue-fused datasets, which share one
# segmentation process, so self-proximity can be read without the curated/residue-fused split.
SUBGROUP = "residue-fused"
SUBGROUP_PAIRS = [
    ("bridge_edges_per_m", "xsec_mean_diff_pct", "bridge vs cross-sectional disagreement, residue-fused only"),
    ("bridge_edges_per_m", "jump_pct", "bridge vs jump %, residue-fused only"),
    ("bridge_edges_per_m", "per_angle_gt2mm_pct", "bridge vs per-angle % of angles > 2 mm, residue-fused only"),
]

# SATO spike-slice % vs self-proximity, joined by dataset name from the SATO batch output.
SATO_RESULTS_PATH = Path(__file__).resolve().parent.parent / "sato_batch_run" / "sato_batch_results.json"


def add_sato_spike_pct(rows):
    with open(SATO_RESULTS_PATH) as f:
        sato = {d["name"]: d for d in json.load(f)}
    for r in rows:
        s = sato[r["label"]]
        r["sato_spike_pct"] = 100.0 * s["n_spike_slices"] / s["n_slices"]
    return rows


# Per-angle cross-section share of angles differing by more than 2 mm, from per_angle_validation.py.
PER_ANGLE_RESULTS_PATH = Path(__file__).resolve().parent.parent / "cross_section_validation" / "per_angle_results.json"


def add_per_angle_gt2mm_pct(rows):
    with open(PER_ANGLE_RESULTS_PATH) as f:
        data = json.load(f)
    data = data if isinstance(data, list) else data["datasets"]
    per_angle = {d["name"]: d for d in data}
    for r in rows:
        r["per_angle_gt2mm_pct"] = per_angle[r["label"]]["dataset_mean_pct_angles_gt_2mm"]
    return rows


def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return [float(r[key]) for r in rows]


def report_pair(rows, xkey, ykey, description):
    x, y = col(rows, xkey), col(rows, ykey)
    n = len(x)
    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    print(f"\n{description}  ({xkey} vs {ykey}, n={n})")
    print(f"  Pearson  r   = {r:.3f}  p = {p:.3f}")
    print(f"  Spearman rho = {rho:.3f}  p = {p_rho:.3f}")

    # Leave-one-out: drop each dataset once, report min/max r (and rho) and flag sign/significance flips.
    loo = []
    for i in range(n):
        xi = x[:i] + x[i + 1:]
        yi = y[:i] + y[i + 1:]
        ri, pi = pearsonr(xi, yi)
        rho_i, prho_i = spearmanr(xi, yi)
        loo.append((rows[i]["label"], ri, pi, rho_i, prho_i))
    rs = [v[1] for v in loo]
    rhos = [v[3] for v in loo]
    print(f"  Leave-one-out r range:   [{min(rs):.3f}, {max(rs):.3f}]")
    print(f"  Leave-one-out rho range: [{min(rhos):.3f}, {max(rhos):.3f}]")
    base_sig = p < 0.05
    base_sig_rho = p_rho < 0.05
    loo_flags = []
    for label, ri, pi, rho_i, prho_i in loo:
        sig, sig_rho = pi < 0.05, prho_i < 0.05
        pearson_flip = bool((ri < 0) != (r < 0) or sig != base_sig)
        spearman_flip = bool((rho_i < 0) != (rho < 0) or sig_rho != base_sig_rho)
        if pearson_flip:
            print(f"    -> dropping {label} changes Pearson sign/significance: r={ri:.3f}, p={pi:.3f}")
        if spearman_flip:
            print(f"    -> dropping {label} changes Spearman sign/significance: rho={rho_i:.3f}, p={prho_i:.3f}")
        loo_flags.append({
            "dropped": label, "r": ri, "p": pi, "rho": rho_i, "p_rho": prho_i,
            "pearson_sign_or_sig_flip": pearson_flip, "spearman_sign_or_sig_flip": spearman_flip,
        })

    return {
        "n": n, "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_rho,
        "leave_one_out": {
            "pearson_r_range": [min(rs), max(rs)],
            "spearman_rho_range": [min(rhos), max(rhos)],
            "per_dataset": loo_flags,
        },
    }


def bootstrap_ci(rows, xkey, ykey, n_resamples=10000, seed=0):
    import random

    x, y = col(rows, xkey), col(rows, ykey)
    n = len(x)
    rng = random.Random(seed)
    rs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        r, _ = pearsonr(xs, ys)
        rs.append(r)
    rs.sort()
    lo = rs[int(0.025 * len(rs))]
    hi = rs[int(0.975 * len(rs))]
    print(f"  Bootstrap 95% CI for r ({len(rs)} valid resamples): [{lo:.3f}, {hi:.3f}]")
    return {"n_valid_resamples": len(rs), "ci_95_low": lo, "ci_95_high": hi, "n_resamples_requested": n_resamples}


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} datasets from {CSV_PATH.name}")

    output = {"n_datasets": len(rows), "input_csv": CSV_PATH.name, "pairs": {}}
    for xkey, ykey, desc in PAIRS:
        pair_result = report_pair(rows, xkey, ykey, desc)
        pair_result["bootstrap"] = bootstrap_ci(rows, xkey, ykey)
        pair_result["description"] = desc
        pair_result["x_key"] = xkey
        pair_result["y_key"] = ykey
        output["pairs"][f"{xkey}__vs__{ykey}"] = pair_result

    rows = add_sato_spike_pct(rows)
    sato_result = report_pair(rows, "bridge_edges_per_m", "sato_spike_pct", "bridge vs SATO spike-slice %")
    output["pairs"]["bridge_edges_per_m__vs__sato_spike_pct"] = sato_result

    rows = add_per_angle_gt2mm_pct(rows)
    per_angle_result = report_pair(rows, "bridge_edges_per_m", "per_angle_gt2mm_pct",
                                   "bridge vs per-angle % of angles > 2 mm")
    output["pairs"]["bridge_edges_per_m__vs__per_angle_gt2mm_pct"] = per_angle_result

    sub_rows = [r for r in rows if r["group"] == SUBGROUP]
    output["subgroup"] = {"group": SUBGROUP, "n": len(sub_rows), "pairs": {}}
    for xkey, ykey, desc in SUBGROUP_PAIRS:
        output["subgroup"]["pairs"][f"{xkey}__vs__{ykey}"] = report_pair(sub_rows, xkey, ykey, desc)

    print("\n--- Summary for thesis text ---")
    for (xkey, ykey, desc) in PAIRS:
        pr = output["pairs"][f"{xkey}__vs__{ykey}"]
        print(f"{desc}: r = {pr['pearson_r']:.3f} (p = {pr['pearson_p']:.3f}), "
              f"rho = {pr['spearman_rho']:.3f} (p = {pr['spearman_p']:.3f})")
    pr = output["pairs"]["bridge_edges_per_m__vs__sato_spike_pct"]
    print(f"bridge vs SATO spike-slice %: r = {pr['pearson_r']:.3f} (p = {pr['pearson_p']:.3f}), "
          f"rho = {pr['spearman_rho']:.3f} (p = {pr['spearman_p']:.3f})")
    pr = output["pairs"]["bridge_edges_per_m__vs__per_angle_gt2mm_pct"]
    print(f"bridge vs per-angle % > 2 mm: r = {pr['pearson_r']:.3f} (p = {pr['pearson_p']:.3f}), "
          f"rho = {pr['spearman_rho']:.3f} (p = {pr['spearman_p']:.3f})")
    for (xkey, ykey, desc) in SUBGROUP_PAIRS:
        pr = output["subgroup"]["pairs"][f"{xkey}__vs__{ykey}"]
        print(f"{desc} (n = {pr['n']}): r = {pr['pearson_r']:.3f} (p = {pr['pearson_p']:.3f}), "
              f"rho = {pr['spearman_rho']:.3f} (p = {pr['spearman_p']:.3f})")

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {RESULTS_PATH.name}")
