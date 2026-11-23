"""SATO batch run and spike-severity metric.

Runs, for every dataset in validate_batch.DATASETS, at SATO's default
crop_radius_ratio=5:
  1. sato_prepare.prepare_dataset: isotropic resample + centerline/width
     conversion into SATO's own format.
  2. straighten_seg_fast.straighten_seg_fast: SATO's reference straightening
     script with a preallocated output array (same math).
  3. Spike-severity metric: per-slice foreground-pixel count vs. a rolling
     local median.

Writes sato_batch_results.json, then reports the correlation between spike
severity and validation_results.json's bridge_edges_per_meter.
"""
import json
import os
import sys
import time
import numpy as np
import SimpleITK as sitk
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "validate_batch"))

from sato_prepare import prepare_dataset
from straighten_seg_fast import straighten_seg_fast
from validate_batch import DATASETS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATE_BATCH_RESULTS = os.path.join(_ROOT, "analysis", "validate_batch", "validation_results.json")

SPIKE_ROLL_WINDOW = 15
SPIKE_RATIO_THRESHOLD = 1.5


def spike_metrics(straightened_path):
    img = sitk.ReadImage(straightened_path)
    arr = sitk.GetArrayFromImage(img)
    fg = arr.reshape(arr.shape[0], -1).sum(axis=1)
    fg_interior = fg[1:-1]  # drop the leading/trailing zero-pad slices
    n = len(fg_interior)
    roll = np.array([
        np.median(fg_interior[max(0, i - SPIKE_ROLL_WINDOW):i + SPIKE_ROLL_WINDOW + 1])
        for i in range(n)
    ])
    roll = np.maximum(roll, 1.0)
    ratio = fg_interior / roll
    is_spike = ratio > SPIKE_RATIO_THRESHOLD
    return {
        "n_slices": int(n),
        "n_spike_slices": int(is_spike.sum()),
        "pct_spike_slices": round(100 * float(is_spike.sum()) / n, 2),
        "max_ratio": round(float(ratio.max()), 2),
        "mean_ratio": round(float(ratio.mean()), 2),
        "median_fg": float(np.median(fg_interior)),
        "max_fg": float(fg_interior.max()),
    }


def run_one(entry):
    name = entry["name"]
    print(f"\n{'='*70}\nSATO batch: {name}\n{'='*70}")
    t0 = time.time()
    try:
        mask_path, cl_path, iso = prepare_dataset(entry)
        straight_path = f"SATO_inputs/{name}_straightened.nii.gz"
        straighten_seg_fast(mask_path, cl_path, straight_path,
                             crop_radius_ratio=5, if_smooth=True, remove_small_holes_thr=200)
        metrics = spike_metrics(straight_path)
        metrics.update({"name": name, "quality": entry.get("quality", "unknown"),
                         "iso_spacing_mm": iso, "time_s": round(time.time() - t0, 1),
                         "status": "ok"})
        print(f"  -> {name}: {metrics['pct_spike_slices']}% spike slices, "
              f"max_ratio={metrics['max_ratio']}, time={metrics['time_s']}s")
        return metrics
    except Exception as e:
        print(f"  FAILED on {name}: {e}")
        return {"name": name, "quality": entry.get("quality", "unknown"), "status": "failed",
                "error": str(e), "time_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    results = [run_one(entry) for entry in DATASETS]

    with open(os.path.join(OUT_DIR, "sato_batch_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{'='*70}\nSUMMARY: {len(ok)}/{len(results)} succeeded\n{'='*70}")
    cols = ["name", "quality", "pct_spike_slices", "max_ratio", "mean_ratio", "time_s"]
    header = " | ".join(c.ljust(16) for c in cols)
    print(header)
    print("-" * len(header))
    for r in ok:
        print(" | ".join(str(r.get(c, "")).ljust(16) for c in cols))

    try:
        with open(VALIDATE_BATCH_RESULTS) as f:
            rq3 = {r["name"]: r for r in json.load(f)}
        names = [r["name"] for r in ok if r["name"] in rq3]
        spike_pct = np.array([r["pct_spike_slices"] for r in ok if r["name"] in rq3])
        coiling = np.array([rq3[n]["bridge_edges_per_meter"] for n in names])
        pear_r, pear_p = pearsonr(coiling, spike_pct)
        spear_r, spear_p = spearmanr(coiling, spike_pct)
        print(f"\nCorrelation vs bridge_edges_per_meter (n={len(names)}):")
        print(f"  Pearson  r={pear_r:.3f} p={pear_p:.3e}")
        print(f"  Spearman r={spear_r:.3f} p={spear_p:.3e}")
    except Exception as e:
        print(f"\nCorrelation step failed: {e}")

    print("\nSaved sato_batch_results.json")
