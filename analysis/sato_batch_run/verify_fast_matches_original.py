"""
Confirms the preallocation-only fix in straighten_seg_fast.py produces the same
output as SATO's unmodified straighten_seg.py, by comparing both scripts'
straightened volumes for C0518 voxel by voxel.

Run: python analysis/sato_batch_run/verify_fast_matches_original.py
"""
import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk

INPUTS = Path(__file__).resolve().parent.parent.parent / "SATO_inputs"
ORIGINAL = INPUTS / "518_straightened_ORIGINAL_unfixed.nii.gz"
FAST = INPUTS / "518_straightened.nii.gz"


def load(path):
    img = sitk.ReadImage(str(path))
    return img, sitk.GetArrayFromImage(img)


if __name__ == "__main__":
    img_a, a = load(ORIGINAL)
    img_b, b = load(FAST)
    print(f"shape:   {a.shape} vs {b.shape}")
    print(f"spacing: {img_a.GetSpacing()} vs {img_b.GetSpacing()}")
    print(f"origin:  {img_a.GetOrigin()} vs {img_b.GetOrigin()}")
    if a.shape != b.shape:
        raise SystemExit("shapes differ")
    n_diff = int((a != b).sum())
    print(f"differing voxels: {n_diff} of {a.size}")
    print(f"foreground voxels: {int((a > 0).sum())} vs {int((b > 0).sum())}")
    for name, arr in (("original", a), ("fast", b)):
        print(f"sha256 {name}: {hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()}")
    print("IDENTICAL" if n_diff == 0 else "DIFFERENT")
