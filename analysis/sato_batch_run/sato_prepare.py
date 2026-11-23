"""Prepare per-dataset inputs for SATO's straighten_seg.py.

SATO's straightening script (SATO_repo/straighten/straighten_seg.py) operates
in raw voxel-index space with no spacing correction: 'coordinate' is (Z,Y,X)
array-index order matching sitk.GetArrayFromImage(image).shape, 'width' is a
radius in voxels, and 'edge_length' is the cumulative Euclidean length of the
polyline in voxel units. Every dataset is therefore resampled to isotropic
spacing first, and the centerline and local radius are rebuilt on that grid.

Coordinate convention: 3D Slicer's markups .mrk.json stores coordinates in
LPS, and SimpleITK represents images in LPS, so this script uses the raw
markup positions with no flip. colon_core.py's load_centerline_slicer flips
(x*=-1, y*=-1) to convert into nibabel's RAS convention instead.
"""
import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import colon_core as cf

OUT_DIR = "SATO_inputs"


def load_centerline_lps(path):
    with open(path, "r") as f:
        data = json.load(f)
    markup = data["markups"][0]
    coord_sys = markup.get("coordinateSystem", "unspecified")
    if coord_sys != "LPS":
        raise ValueError(f"{path}: expected coordinateSystem LPS, got {coord_sys!r} "
                          f"-- do not assume, re-verify the flip before proceeding")
    pts = np.array([cp["position"] for cp in markup["controlPoints"]], dtype=np.float64)
    return pts


def resample_isotropic(sitk_img, interpolator=sitk.sitkNearestNeighbor):
    orig_spacing = np.array(sitk_img.GetSpacing())
    iso = float(orig_spacing.min())
    orig_size = np.array(sitk_img.GetSize())
    new_size = [int(round(s)) for s in orig_size * orig_spacing / iso]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((iso, iso, iso))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(sitk_img), iso


def trim_unreliable_ends(coords_zyx, widths, floor_frac=0.3, window=5):
    """Trim both ends of the centerline where the local EDT width collapses:
    walks in from each end while a rolling-average width stays below a floor
    fraction of the whole centerline's median width.
    """
    median_w = np.median(widths)
    floor = floor_frac * median_w
    kernel = np.ones(window) / window
    roll = np.convolve(widths, kernel, mode="same")
    reliable = roll >= floor

    start = 0
    while start < len(reliable) and not reliable[start]:
        start += 1
    end = len(reliable) - 1
    while end >= 0 and not reliable[end]:
        end -= 1
    if start > end:
        raise ValueError("trim_unreliable_ends: no reliable interior region found")

    if start > 0 or end < len(widths) - 1:
        print(f"  edge trim: dropped {start} points from start, "
              f"{len(widths)-1-end} from end (width floor={floor:.2f}vox, "
              f"median={median_w:.2f}vox)")
    return coords_zyx[start:end + 1], widths[start:end + 1]


def build_sato_centerline(cl_pts_lps_mm, resampled_img, mask_arr_zyx):
    """cl_pts_lps_mm: Nx3 raw LPS mm points (already smoothed/resampled).
    Returns a dict in SATO's expected format: {'edge_length', 'point': [...]}."""
    edt = distance_transform_edt(mask_arr_zyx)  # isotropic grid -> voxel units directly

    coords_zyx = np.empty((len(cl_pts_lps_mm), 3), dtype=np.float64)
    for i, p in enumerate(cl_pts_lps_mm):
        idx_xyz = resampled_img.TransformPhysicalPointToContinuousIndex(tuple(p))
        coords_zyx[i] = [idx_xyz[2], idx_xyz[1], idx_xyz[0]]

    widths = map_coordinates(edt, coords_zyx.T, order=1, mode="nearest")
    coords_zyx, widths = trim_unreliable_ends(coords_zyx, widths)

    edge_length = float(np.linalg.norm(np.diff(coords_zyx, axis=0), axis=1).sum())
    points = [{"coordinate": coords_zyx[i], "width": float(widths[i])}
              for i in range(len(coords_zyx))]
    return {"edge_length": edge_length, "point": points}


def prepare_dataset(entry, label_value=1, tag=None):
    """entry: one of validate_batch.DATASETS' dicts (or a similar dict with
    'name', 'nifti', 'centerline'). label_value lets the same function build
    inputs for a polyp-only channel (label_value=2) on the same grid.
    """
    name = entry["name"] + (f"_{tag}" if tag else "")
    print(f"\n{'='*70}\nPreparing SATO input: {name}\n{'='*70}")
    os.makedirs(OUT_DIR, exist_ok=True)

    sitk_img = sitk.ReadImage(entry["nifti"])
    resampled, iso = resample_isotropic(sitk_img)
    print(f"  original spacing {sitk_img.GetSpacing()} -> isotropic {iso:.4f}mm, "
          f"size {sitk_img.GetSize()} -> {resampled.GetSize()}")

    arr = sitk.GetArrayFromImage(resampled)
    mask_arr = (arr == label_value).astype(np.uint8)
    n_fg = int(mask_arr.sum())
    print(f"  label {label_value}: {n_fg} foreground voxels after resample")
    if n_fg == 0:
        raise ValueError(f"{name}: no foreground voxels for label {label_value} "
                          f"after resampling -- check label_value / nifti path")

    mask_path = os.path.join(OUT_DIR, f"{name}_mask.nii.gz")
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(resampled)
    sitk.WriteImage(mask_img, mask_path)

    cl_raw = load_centerline_lps(entry["centerline"])
    cl_smooth = cf.resample_centerline(cl_raw, step_mm=1.0)

    sato_cl = build_sato_centerline(cl_smooth, resampled, mask_arr)
    cl_path = os.path.join(OUT_DIR, f"{name}_centerline.npy")
    np.save(cl_path, sato_cl, allow_pickle=True)
    print(f"  edge_length={sato_cl['edge_length']:.1f} voxels, "
          f"{len(sato_cl['point'])} centerline points, "
          f"width range {min(p['width'] for p in sato_cl['point']):.2f}-"
          f"{max(p['width'] for p in sato_cl['point']):.2f} voxels")
    print(f"  saved: {mask_path}, {cl_path}")
    return mask_path, cl_path, iso


if __name__ == "__main__":
    from validate_batch import DATASETS
    ds = next(e for e in DATASETS if e["name"] == "787")
    prepare_dataset(ds)
