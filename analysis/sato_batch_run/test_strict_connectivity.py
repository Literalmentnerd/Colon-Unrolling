"""Re-run SATO's object selection with strict (face-only, connectivity=1)
component labeling instead of the default full 26-connectivity.

Monkeypatches the skimage.measure.label(image) call inside
straighten_seg_fast.py's save_center_objects to pass connectivity=1, since
SATO exposes no connectivity option.
"""
import json
import numpy as np
import skimage.measure
import SimpleITK as sitk

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sato_batch_run"))
import straighten_seg_fast as ssf

_orig_label = skimage.measure.label


def strict_label(image, *args, **kwargs):
    kwargs.setdefault("connectivity", 1)
    return _orig_label(image, *args, **kwargs)


def save_center_objects_strict(image, x_size, y_size, z_size):
    labeled_image = strict_label(image)
    labeled_list = skimage.measure.regionprops(labeled_image)
    label_num = None
    for i in range(len(labeled_list)):
        coords = list(labeled_list[i].coords)
        if (coords == np.array([int(z_size / 2), int((x_size - 1) / 2), int((y_size - 1) / 2)])).all(1).any():
            label_num = i + 1
    if label_num is None:
        raise ValueError("save_center_objects_strict: no labeled component contains the center voxel")
    labeled_image[labeled_image != label_num] = 0
    labeled_image[labeled_image == label_num] = 1
    return labeled_image


ssf.save_center_objects = save_center_objects_strict


def spike_metrics(straightened_path):
    img = sitk.ReadImage(straightened_path)
    arr = sitk.GetArrayFromImage(img)
    fg = arr.reshape(arr.shape[0], -1).sum(axis=1)
    fg_interior = fg[1:-1]
    n = len(fg_interior)
    roll_window = 15
    roll = np.array([
        np.median(fg_interior[max(0, i - roll_window):i + roll_window + 1])
        for i in range(n)
    ])
    roll = np.maximum(roll, 1.0)
    ratio = fg_interior / roll
    is_spike = ratio > 1.5
    return {
        "n_slices": int(n),
        "n_spike_slices": int(is_spike.sum()),
        "pct_spike_slices": round(100 * float(is_spike.sum()) / n, 2),
        "max_ratio": round(float(ratio.max()), 2),
        "mean_ratio": round(float(ratio.mean()), 2),
    }


results = {}
for name, mask_path, cl_path in [
    ("518", "SATO_inputs/518_mask.nii.gz", "SATO_inputs/518_centerline.npy"),
    ("787", "SATO_inputs/787_mask.nii.gz", "SATO_inputs/787_centerline.npy"),
]:
    out_path = f"SATO_inputs/{name}_straightened_strictconn.nii.gz"
    print(f"\n{'='*60}\nRunning strict-connectivity variant on {name}\n{'='*60}")
    ssf.straighten_seg_fast(mask_path, cl_path, out_path, crop_radius_ratio=5,
                             if_smooth=True, remove_small_holes_thr=200)
    metrics = spike_metrics(out_path)
    metrics["name"] = name
    results[name] = metrics
    print(json.dumps(metrics, indent=2))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strict_connectivity_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved strict_connectivity_results.json")
