"""Shared functions for colon unrolling and straightening.

Unrolling: build_centerline_frame, then cast_rays_adaptive, then build_25d_mesh.
Straightening: build_centerline_frame, then straighten_mesh.

Run generate_unrolls.py or generate_straightened_mesh.py to produce the meshes.
"""
import numpy as np
import json
import heapq
from skimage import measure
from scipy.ndimage import label, zoom as ndimage_zoom
from scipy.interpolate import CubicSpline, griddata
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import nibabel as nib
from stl import mesh as stl_mesh
from skimage.filters import gaussian

# ── CONFIG ─────────────────────────────────────────────────────────────────
ARC_STEP_MM      = 1.0
ANGLE_STEP_DEG   = 1.0
MAX_RAY_DIST     = 50.0
Z_SCALE          = 1.0   # amplify radial displacement in unrolled view

# ── 1. Load centerline from 3D Slicer .mrk.json ───────────────────────────
def load_centerline_slicer(path):
    with open(path, "r") as f:
        data = json.load(f)
    markups = data["markups"][0]
    control_points = markups["controlPoints"]
    pts = np.array([cp["position"] for cp in control_points], dtype=np.float64)
    pts[:, 0] *= -1
    pts[:, 1] *= -1
    print(f"Loaded {len(pts)} centerline points from {path}")
    return pts

# ── 1b. Smooth + resample centerline to fine, even spacing ─────────────────
SMOOTH_WINDOW_MM = 21   # Savitzky-Golay window (mm)
SMOOTH_POLYORDER = 3

def resample_centerline(pts, step_mm=ARC_STEP_MM):
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)

    keep = np.concatenate([[True], seg_lens > 1e-6])
    pts = pts[keep]
    seg_lens = seg_lens[seg_lens > 1e-6]

    window = min(SMOOTH_WINDOW_MM, len(pts) - (1 - len(pts) % 2))
    if window >= SMOOTH_POLYORDER + 2:
        if window % 2 == 0:
            window += 1
        pts_smooth = savgol_filter(pts, window_length=window, polyorder=SMOOTH_POLYORDER, axis=0)
        disp = np.linalg.norm(pts_smooth - pts, axis=1)
        print(f"Smoothed centerline (window={window}mm): mean shift {disp.mean():.3f}mm, max {disp.max():.3f}mm")
        pts = pts_smooth
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        keep = np.concatenate([[True], seg_lens > 1e-6])
        pts = pts[keep]
        seg_lens = seg_lens[seg_lens > 1e-6]

    arc = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = arc[-1]

    n_samples = max(int(np.ceil(total_len / step_mm)) + 1, 2)
    new_arc = np.linspace(0, total_len, n_samples)

    spline = CubicSpline(arc, pts, axis=0)
    new_pts = spline(new_arc)

    print(f"Resampled centerline: {len(pts)} -> {len(new_pts)} points at ~{step_mm}mm spacing")
    return new_pts

# ── 2. Load NIFTI and extract mesh via marching cubes ─────────────────────
MESH_UPSAMPLE = 2  # upsample factor before marching cubes (higher = smaller triangles)

def load_nifti_mesh(path, label_value=1):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine

    mask = (np.round(data) == label_value).astype(np.uint8)
    del data

    labeled, n = label(mask)
    if n > 1:
        sizes = [np.sum(labeled == i) for i in range(1, n+1)]
        mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
        print(f"Found {n} components, keeping largest")
    del labeled

    # crop to bounding box (+ padding) so we don't smooth/upsample empty background
    pad = 3
    nonzero = np.argwhere(mask)
    lo = np.maximum(nonzero.min(axis=0) - pad, 0)
    hi = np.minimum(nonzero.max(axis=0) + pad + 1, mask.shape)
    mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    print(f"Cropped mask to bounding box: {mask.shape}")

    # smooth at native resolution first, then upsample the smooth field —
    # keeps only one large buffer in memory instead of two
    mask_smooth = gaussian(mask, sigma=1.3)
    del mask

    if MESH_UPSAMPLE > 1:
        print(f"Upsampling smoothed mask {MESH_UPSAMPLE}x before marching cubes...")
        mask_smooth = ndimage_zoom(mask_smooth, zoom=MESH_UPSAMPLE, order=1)

    verts_vox, faces, _, _ = measure.marching_cubes(mask_smooth, level=0.5)
    del mask_smooth

    # scale voxel coords back to original voxel space, then shift back
    # from crop-local coords into full-volume voxel coords before applying affine
    verts_vox = verts_vox / MESH_UPSAMPLE
    verts_vox += lo

    ones = np.ones((len(verts_vox), 1))
    verts_world = (affine @ np.hstack([verts_vox, ones]).T).T[:, :3]

    print(f"Extracted mesh: {len(verts_world)} vertices, {len(faces)} faces")
    return verts_world, faces

# ── 2b. Load polyp centroids from NIfTI ───────────────────────────────────
def load_polyp_centroids(path, label_value=2):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    mask = (np.round(data) == label_value).astype(np.uint8)
    del data
    if mask.sum() == 0:
        print(f"  No voxels with label {label_value} found")
        return np.zeros((0, 3))
    nonzero = np.argwhere(mask)
    lo, hi = nonzero.min(axis=0), nonzero.max(axis=0) + 1
    mask_crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    del mask
    labeled, n = label(mask_crop)
    del mask_crop
    centroids = []
    for i in range(1, n + 1):
        vox = np.argwhere(labeled == i).mean(axis=0) + lo
        centroids.append((affine @ np.append(vox, 1.0))[:3])
    print(f"  Found {n} polyp component(s):")
    for i, c in enumerate(centroids):
        print(f"    Polyp {i+1}: [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] mm")
    return np.array(centroids)

# ── 3. Build arc length and rotation-minimising frame ──────────────────────
def build_centerline_frame(cl_pts):
    diffs = np.diff(cl_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_lens)])

    tangents = np.zeros_like(cl_pts)
    tangents[:-1] = diffs / seg_lens[:, None]
    tangents[-1] = tangents[-2]

    n = len(tangents)
    normals   = np.zeros((n, 3))
    binormals = np.zeros((n, 3))

    up = np.array([0, 0, 1.0])
    if abs(np.dot(tangents[0], up)) > 0.9:
        up = np.array([0, 1, 0.0])

    normals[0]  = np.cross(tangents[0], up)
    normals[0] /= np.linalg.norm(normals[0])
    binormals[0] = np.cross(tangents[0], normals[0])

    def rotate(v, axis, angle):
        return (v * np.cos(angle) +
                np.cross(axis, v) * np.sin(angle) +
                axis * np.dot(axis, v) * (1 - np.cos(angle)))

    for i in range(1, n):
        b = np.cross(tangents[i-1], tangents[i])
        nb = np.linalg.norm(b)
        if nb < 1e-6:
            normals[i]   = normals[i-1]
            binormals[i] = binormals[i-1]
        else:
            b /= nb
            angle = np.arccos(np.clip(np.dot(tangents[i-1], tangents[i]), -1, 1))
            normals[i]   = rotate(normals[i-1], b, angle)
            binormals[i] = np.cross(tangents[i], normals[i])

    def norm_rows(a):
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        return np.where(norms > 1e-10, a / norms, a)

    return arc, seg_lens, norm_rows(tangents), norm_rows(normals), norm_rows(binormals)

# ── 3b. Interpolate centerline position + frame at arbitrary arc-length
# values, vectorized. Used by the direct ray-caster to get a world-space
# ray origin/frame at each arc step without needing a straightened mesh at
# all.
def interpolate_centerline_frame(t_query, arc, cl_pts, normals, binormals):
    t_clamped = np.clip(t_query, arc[0], arc[-1])
    idx = np.clip(np.searchsorted(arc, t_clamped) - 1, 0, len(arc) - 2)
    seg_len = arc[idx + 1] - arc[idx]
    seg_len_safe = np.where(seg_len > 1e-10, seg_len, 1.0)
    alpha = np.where(seg_len > 1e-10, (t_clamped - arc[idx]) / seg_len_safe, 0.0)

    C = (1 - alpha)[:, None] * cl_pts[idx] + alpha[:, None] * cl_pts[idx + 1]

    def interp_normalize(a, b):
        v = (1 - alpha)[:, None] * a[idx] + alpha[:, None] * b[idx + 1]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(n > 1e-10, v / n, v)

    N = interp_normalize(normals, normals)
    B = interp_normalize(binormals, binormals)
    return C, N, B

# ── 4. Project vertex onto segment i ──────────────────────────────────────
def project_onto_segment(v, i, cl_pts, seg_lens, arc, tangents, normals, binormals):
    seg_len = seg_lens[i]
    if seg_len < 1e-10:
        return np.inf, arc[i], 0.0, 0.0

    seg_dir = (cl_pts[i+1] - cl_pts[i]) / seg_len
    offset_i = v - cl_pts[i]
    t_local = float(np.clip(np.dot(offset_i, seg_dir), 0.0, seg_len))
    closest_pt = cl_pts[i] + t_local * seg_dir
    dist = float(np.linalg.norm(v - closest_pt))

    alpha = t_local / seg_len

    def interp_normalize(a, b, alpha):
        v = (1 - alpha) * a + alpha * b
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    n_frame = interp_normalize(normals[i],   normals[i+1],   alpha)
    b_frame = interp_normalize(binormals[i], binormals[i+1], alpha)

    offset = v - closest_pt
    t_coord = arc[i] + t_local
    n_coord = float(np.dot(offset, n_frame))
    b_coord = float(np.dot(offset, b_frame))

    return dist, t_coord, n_coord, b_coord

# ── 4b. Project world points to centerline frame angles ───────────────────
def polyp_angles_in_frame(world_pts, cl_pts, arc, seg_lens, tangents, normals, binormals):
    n_cl = len(cl_pts)
    angles = []
    for pt in world_pts:
        best_dist, best_n, best_b = np.inf, 0.0, 0.0
        for i in range(n_cl - 1):
            dist, _, n_coord, b_coord = project_onto_segment(
                pt, i, cl_pts, seg_lens, arc, tangents, normals, binormals)
            if dist < best_dist:
                best_dist, best_n, best_b = dist, n_coord, b_coord
        angles.append(np.degrees(np.arctan2(best_b, best_n)))
    return np.array(angles)

def find_best_cut_angle(polyp_angles_deg):
    """Return the angle (degrees) of the largest circular gap between polyp angles."""
    if len(polyp_angles_deg) == 0:
        return 0.0
    angles = np.sort(polyp_angles_deg % 360)
    gaps = np.diff(angles, append=angles[0] + 360)
    best = np.argmax(gaps)
    cut = (angles[best] + gaps[best] / 2.0) % 360
    print(f"  Polyp angles (deg): {np.round(angles, 1).tolist()}")
    print(f"  Largest gap: {gaps[best]:.1f}° — placing seam at {cut:.1f}°")
    return float(cut)

# ── 4c. Self-proximity metric: mesh edges that bridge two loops of the colon
def find_bridge_edges(verts, faces, cl_pts, arc, t_diff_threshold=100.0, dist_threshold=2.0,
                       max_loop_sep_mm=None, radius_arc=None, radius_values=None,
                       touch_multiplier=1.15):
    """Find mesh edges that connect two different loops of a coiled colon
    where they pass close to each other in 3D. Each vertex's arc position is
    estimated from its nearest centerline point; an edge shorter than
    dist_threshold whose two endpoints' estimates differ by more than
    t_diff_threshold is flagged.

    Flagged edges are then kept only if the two centerline points involved
    are close enough for the loops to be touching, using one of:

    max_loop_sep_mm: a single flat distance cap.

    radius_arc + radius_values (e.g. arc_steps and the per-row mean radius
    from ray-casting, np.nanmean(filled, axis=1)): interpolates the local
    wall radius at each endpoint's arc position and requires
    loop_sep < touch_multiplier * (radius_a + radius_b). Takes precedence
    over max_loop_sep_mm when both are given.
    """
    tree = cKDTree(cl_pts)
    _, nearest_idx = tree.query(verts)
    t_est = arc[nearest_idx]

    edges_a = faces[:, [0, 1, 2]].ravel()
    edges_b = faces[:, [1, 2, 0]].ravel()
    t_diff = np.abs(t_est[edges_a] - t_est[edges_b])
    d3 = np.linalg.norm(verts[edges_a] - verts[edges_b], axis=1)
    bad = (t_diff > t_diff_threshold) & (d3 < dist_threshold)

    if radius_arc is not None and radius_values is not None:
        cl_a = cl_pts[nearest_idx[edges_a]]
        cl_b = cl_pts[nearest_idx[edges_b]]
        loop_sep = np.linalg.norm(cl_a - cl_b, axis=1)
        ra = np.interp(t_est[edges_a], radius_arc, radius_values)
        rb = np.interp(t_est[edges_b], radius_arc, radius_values)
        bad &= (loop_sep < touch_multiplier * (ra + rb))
    elif max_loop_sep_mm is not None:
        cl_a = cl_pts[nearest_idx[edges_a]]
        cl_b = cl_pts[nearest_idx[edges_b]]
        loop_sep = np.linalg.norm(cl_a - cl_b, axis=1)
        bad &= (loop_sep < max_loop_sep_mm)

    bridge_edges = set(zip(np.minimum(edges_a[bad], edges_b[bad]).tolist(),
                            np.maximum(edges_a[bad], edges_b[bad]).tolist()))
    return bridge_edges


# ── 5. Topology-aware flood fill (vertex-based straightening). Heap priority
# is each vertex's own raw distance to the centerline (best_dist), not a
# cumulative mesh-walk distance. Called by generate_straightened_mesh.py.
def straighten_mesh(verts, faces, cl_pts, arc, seg_lens, tangents, normals, binormals, blocked_edges=None):
    n_verts = len(verts)
    n_cl    = len(cl_pts)

    print("Building adjacency...")
    adjacency = [set() for _ in range(n_verts)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adjacency[a].update([b, c])
        adjacency[b].update([a, c])
        adjacency[c].update([a, b])
    if blocked_edges:
        for (u, v) in blocked_edges:
            adjacency[u].discard(v)
            adjacency[v].discard(u)
        print(f"  Blocked {len(blocked_edges)} bridge edges from adjacency")

    assigned = np.full(n_verts, -1, dtype=int)
    new_verts = np.zeros((n_verts, 3))

    print("Seeding...")
    SEED_CL_POINTS = 10
    seed_dists = np.full(n_verts, np.inf)
    for ci in range(min(SEED_CL_POINTS, n_cl)):
        d = np.linalg.norm(verts - cl_pts[ci], axis=1)
        seed_dists = np.minimum(seed_dists, d)

    threshold = np.percentile(seed_dists, 2)
    seed_indices = np.where(seed_dists < threshold)[0]

    heap = []
    for vi in seed_indices:
        best_dist = np.inf
        best_seg = 0
        best_coords = None
        for ci in range(min(SEED_CL_POINTS, n_cl - 1)):
            dist, t_coord, n_coord, b_coord = project_onto_segment(
                verts[vi], ci, cl_pts, seg_lens, arc, tangents, normals, binormals)
            if dist < best_dist:
                best_dist = dist
                best_seg = ci
                best_coords = (n_coord, t_coord, b_coord)
        assigned[vi] = best_seg
        new_verts[vi] = best_coords
        for nb in adjacency[vi]:
            if assigned[nb] == -1:
                heapq.heappush(heap, (best_dist, int(vi), int(nb), best_seg))

    SLACK = 3
    # how many segments ahead of the parent's assigned segment the search covers
    FORWARD_SLACK = 15
    processed = 0
    log_every = max(1, n_verts // 20)

    print("Flood filling...")
    while heap:
        _, parent_vi, vi, min_cl_idx = heapq.heappop(heap)

        if assigned[vi] != -1:
            continue

        search_start = max(0, min_cl_idx - SLACK)
        search_end   = min(n_cl - 1, min_cl_idx + FORWARD_SLACK)
        best_dist   = np.inf
        best_result = None
        best_seg    = search_start

        for i in range(search_start, search_end):
            dist, t_coord, n_coord, b_coord = project_onto_segment(
                verts[vi], i, cl_pts, seg_lens, arc, tangents, normals, binormals)
            if dist < best_dist:
                best_dist   = dist
                best_result = (n_coord, t_coord, b_coord)
                best_seg    = i
            elif dist > best_dist * 2 and i > search_start + 5:
                break

        if best_result is None:
            new_verts[vi] = new_verts[parent_vi]
            assigned[vi]  = min_cl_idx
        else:
            assigned[vi]  = best_seg
            new_verts[vi] = best_result

        for nb in adjacency[vi]:
            if assigned[nb] == -1:
                heapq.heappush(heap, (best_dist, int(vi), int(nb), best_seg))

        processed += 1
        if processed % log_every == 0:
            print(f"  {processed}/{n_verts} ({100*processed//n_verts}%)...")

    unassigned = np.where(assigned == -1)[0]
    if len(unassigned):
        print(f"Fallback for {len(unassigned)} disconnected vertices...")
        for vi in unassigned:
            dists = np.linalg.norm(cl_pts - verts[vi], axis=1)
            i = min(int(np.argmin(dists)), n_cl - 2)
            _, t_coord, n_coord, b_coord = project_onto_segment(
                verts[vi], i, cl_pts, seg_lens, arc, tangents, normals, binormals)
            new_verts[vi] = [n_coord, t_coord, b_coord]

    return new_verts

# ── 6. Build triangle arrays ───────────────────────────────────────────────
def build_triangles(verts, faces):
    return verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

# ── 7. Vectorized ray-triangle intersection ────────────────────────────────
def ray_triangle_intersect_batch(ray_origin, ray_dir, v0, v1, v2):
    eps = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_dir, edge2)
    a = np.einsum('ij,ij->i', edge1, h)

    valid = np.abs(a) > eps
    f = np.where(valid, 1.0 / a, 0.0)

    s = ray_origin - v0
    u = f * np.einsum('ij,ij->i', s, h)
    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, edge1)
    v = f * np.einsum('j,ij->i', ray_dir, q)
    valid &= (v >= 0.0) & (u + v <= 1.0)

    t = f * np.einsum('ij,ij->i', edge2, q)
    valid &= t > eps

    return np.where(valid, t, np.inf)

# ── 8. Direct ray casting against the original (unstraightened) mesh. For
# every (arc, angle) output pixel, the ray origin/direction come from the
# world-space centerline frame (interpolate_centerline_frame). A KD-tree on
# triangle centroids limits which triangles are tested near each ray origin;
# the ray-triangle intersection test decides hits.
def _cast_rays_at_positions(verts, faces, cl_pts, arc, normals, binormals,
                             arc_positions, angle_positions, progress=False,
                             tree=None, v0=None, v1=None, v2=None, centroids=None):
    """Core ray-cast: one arc/angle grid, no diagnostics."""
    n_arc   = len(arc_positions)
    n_angle = len(angle_positions)

    if tree is None:
        v0, v1, v2 = build_triangles(verts, faces)
        centroids = (v0 + v1 + v2) / 3.0
        tree = cKDTree(centroids)

    C_all, N_all, B_all = interpolate_centerline_frame(arc_positions, arc, cl_pts, normals, binormals)
    T_all = np.cross(N_all, B_all)  # N,B,T are a right-handed frame by construction

    angle_rads = np.deg2rad(angle_positions)
    cos_a = np.cos(angle_rads)
    sin_a = np.sin(angle_rads)

    search_radius = MAX_RAY_DIST + 5.0  # small margin past the accepted hit range
    # rays are purely radial (zero component along the tangent), so only
    # triangles within axial_tol of the plane perpendicular to the tangent
    # at C are kept as candidates
    axial_tol = max(3.0, ARC_STEP_MM)

    hit_distances = np.full((n_arc, n_angle), np.nan)
    log_every = max(1, n_arc // 20)

    for ai in range(n_arc):
        C = C_all[ai]
        candidate_idx = np.asarray(tree.query_ball_point(C, search_radius))
        if len(candidate_idx) == 0:
            continue
        T = T_all[ai]
        axial_offset = np.abs((centroids[candidate_idx] - C) @ T)
        candidate_idx = candidate_idx[axial_offset < axial_tol]
        if len(candidate_idx) == 0:
            continue
        v0n, v1n, v2n = v0[candidate_idx], v1[candidate_idx], v2[candidate_idx]

        N = N_all[ai]
        B = B_all[ai]
        for bi in range(n_angle):
            ray_dir = cos_a[bi] * N + sin_a[bi] * B
            t_vals = ray_triangle_intersect_batch(C, ray_dir, v0n, v1n, v2n)
            t_valid = t_vals[(t_vals > 1.0) & (t_vals < MAX_RAY_DIST)]
            if len(t_valid) > 0:
                hit_distances[ai, bi] = t_valid.min()

        if progress and ai % log_every == 0:
            print(f"  Arc step {ai}/{n_arc} ({100*ai//n_arc}%)...")

    return hit_distances, tree, v0, v1, v2, centroids


def cast_rays_direct(verts, faces, cl_pts, arc, normals, binormals, cut_angle_deg=0.0,
                      edge_trim_mm=None, trim_miss_threshold=0.10, trim_window_mm=5.0,
                      max_trim_mm=200.0):
    # cast almost the full arc range, then trim each end from the measured
    # per-row miss rate. Pass edge_trim_mm to use a fixed trim instead.
    margin = 1.0
    arc_steps   = np.arange(margin, max(margin, arc[-1] - margin), ARC_STEP_MM)
    angle_steps = np.arange(cut_angle_deg, cut_angle_deg + 360, ANGLE_STEP_DEG)
    n_arc   = len(arc_steps)
    n_angle = len(angle_steps)

    print(f"Ray grid: {n_arc} arc steps x {n_angle} angle steps = {n_arc * n_angle} rays")
    print("Building spatial index (KD-tree on triangle centroids)...")

    hit_distances, *_ = _cast_rays_at_positions(
        verts, faces, cl_pts, arc, normals, binormals, arc_steps, angle_steps, progress=True)

    nan_frac_per_row = np.isnan(hit_distances).mean(axis=1)

    if edge_trim_mm is not None:
        lo = int(np.searchsorted(arc_steps, arc_steps[0] + edge_trim_mm))
        hi = int(np.searchsorted(arc_steps, arc_steps[-1] - edge_trim_mm))
        print(f"  Fixed edge trim: {edge_trim_mm}mm at each end")
    else:
        window = max(1, int(round(trim_window_mm / ARC_STEP_MM)))
        roll = np.convolve(nan_frac_per_row, np.ones(window) / window, mode='same')
        max_trim_rows = min(int(round(max_trim_mm / ARC_STEP_MM)), n_arc // 4)

        lo = 0
        while lo < max_trim_rows and roll[lo] > trim_miss_threshold:
            lo += 1
        hi = n_arc
        while (n_arc - hi) < max_trim_rows and hi > lo and roll[hi - 1] > trim_miss_threshold:
            hi -= 1
        print(f"  Adaptive edge trim: {arc_steps[lo] - arc_steps[0]:.1f}mm at start, "
              f"{arc_steps[-1] - arc_steps[hi - 1]:.1f}mm at end "
              f"(rolling {trim_window_mm:.0f}mm window, {trim_miss_threshold*100:.0f}% miss threshold)")

    arc_steps = arc_steps[lo:hi]
    hit_distances = hit_distances[lo:hi]
    nan_frac_per_row = nan_frac_per_row[lo:hi]
    n_arc = len(arc_steps)

    print(f"  Hit distance range: {np.nanmin(hit_distances):.2f} to {np.nanmax(hit_distances):.2f} mm")
    print(f"  Mean hit distance: {np.nanmean(hit_distances):.2f} mm")

    total_missed = np.isnan(hit_distances).sum()
    print(f"  Ray misses: {total_missed}/{hit_distances.size} ({100*total_missed/hit_distances.size:.2f}%)")
    print(f"  Arc steps with >10% rays missed: {(nan_frac_per_row > 0.10).sum()}/{n_arc}")
    print(f"  Arc steps with >50% rays missed: {(nan_frac_per_row > 0.50).sum()}/{n_arc}")

    bin_size = 20.0  # mm
    bin_edges = np.arange(arc_steps[0], arc_steps[-1] + bin_size, bin_size)
    bin_idx = np.clip(np.digitize(arc_steps, bin_edges) - 1, 0, len(bin_edges) - 2)
    print("  Ray-miss profile by arc-length bin (only bins >5% shown):")
    any_flagged = False
    for b in range(len(bin_edges) - 1):
        rows = np.where(bin_idx == b)[0]
        if len(rows) == 0:
            continue
        frac = nan_frac_per_row[rows].mean()
        if frac > 0.05:
            any_flagged = True
            marker = "  <-- high miss rate" if frac > 0.30 else ""
            print(f"    arc {bin_edges[b]:.0f}-{bin_edges[b+1]:.0f}mm: {frac*100:.1f}% rays missed{marker}")
    if not any_flagged:
        print("    (none — misses are low and evenly spread)")

    return arc_steps, angle_steps, hit_distances

# ── 8b. Adaptive ray-distance ceiling: cast once at a base ceiling, and if
# truncation (hits landing within 2mm of the ceiling) exceeds a threshold,
# re-cast at a higher one. MAX_RAY_DIST also sets the KD-tree
# candidate-triangle search radius.
NEAR_CAP_RECAST_THRESHOLD_PCT = 0.3
RAISED_MAX_RAY_DIST = 90.0

def cast_rays_adaptive(verts, faces, cl_pts, arc, normals, binormals,
                        base_ray_dist=50.0,
                        near_cap_threshold_pct=NEAR_CAP_RECAST_THRESHOLD_PCT,
                        raised_ray_dist=RAISED_MAX_RAY_DIST, **kwargs):
    """Like cast_rays_direct, but automatically re-casts at raised_ray_dist
    if truncation at base_ray_dist is too high. Returns (arc_steps,
    angle_steps, hit_distances, final_max_ray_dist, near_cap_pct_at_base,
    was_recast). Leaves the module-level MAX_RAY_DIST set to whichever
    ceiling was actually used, matching cast_rays_direct's existing
    behavior of reading MAX_RAY_DIST as a global."""
    global MAX_RAY_DIST
    MAX_RAY_DIST = base_ray_dist
    arc_steps, angle_steps, hit_distances = cast_rays_direct(
        verts, faces, cl_pts, arc, normals, binormals, **kwargs)

    valid_hits = hit_distances[~np.isnan(hit_distances)]
    near_cap_pct = 100.0 * (valid_hits > base_ray_dist - 2.0).sum() / max(1, hit_distances.size)

    was_recast = False
    if near_cap_pct > near_cap_threshold_pct and base_ray_dist < raised_ray_dist:
        print(f"  Truncation at {base_ray_dist:.0f}mm ceiling is {near_cap_pct:.2f}% "
              f"(> {near_cap_threshold_pct}% threshold) -- re-casting at "
              f"{raised_ray_dist:.0f}mm instead of reporting truncated numbers")
        MAX_RAY_DIST = raised_ray_dist
        arc_steps, angle_steps, hit_distances = cast_rays_direct(
            verts, faces, cl_pts, arc, normals, binormals, **kwargs)
        was_recast = True

    return arc_steps, angle_steps, hit_distances, MAX_RAY_DIST, near_cap_pct, was_recast

# ── 9. Build 2.5D mesh from ray hit grid ──────────────────────────────────
def fill_gaps_smooth(hit_distances):
    """Fill NaN gaps by linear interpolation within each gap region from the
    surrounding valid values; nearest-neighbor is used only as a fallback
    for gap cells where linear interpolation returns no value."""
    n_angle = hit_distances.shape[1]
    pad = min(20, n_angle // 2)
    padded = np.concatenate(
        [hit_distances[:, -pad:], hit_distances, hit_distances[:, :pad]], axis=1)
    missing = np.isnan(padded)
    if missing.any() and not missing.all():
        lbl, n_regions = label(missing)
        valid_mask = ~missing
        for region_id in range(1, n_regions + 1):
            ys, xs = np.where(lbl == region_id)
            y0, y1 = max(0, ys.min() - 3), min(padded.shape[0], ys.max() + 4)
            x0, x1 = max(0, xs.min() - 3), min(padded.shape[1], xs.max() + 4)
            local = padded[y0:y1, x0:x1]
            local_valid = valid_mask[y0:y1, x0:x1]
            if local_valid.sum() < 3:
                continue
            pts = np.argwhere(local_valid)
            vals = local[local_valid]
            gap_pts = np.column_stack([ys - y0, xs - x0])
            filled = griddata(pts, vals, gap_pts, method='linear')
            still_nan = np.isnan(filled)
            if still_nan.any():
                filled[still_nan] = griddata(pts, vals, gap_pts[still_nan], method='nearest')
            padded[ys, xs] = filled
    hit_distances = padded[:, pad:pad + n_angle]
    return hit_distances


def build_25d_mesh(arc_steps, angle_steps, hit_distances):
    n_arc   = len(arc_steps)
    n_angle = len(angle_steps)

    hit_distances = fill_gaps_smooth(hit_distances)

    mean_radius = np.nanmean(hit_distances)
    print(f"  Mean inner radius: {mean_radius:.2f} mm")
    print(f"  Unrolled circumference: {2 * np.pi * mean_radius:.2f} mm")
    print(f"  Unrolled length: {arc_steps[-1] - arc_steps[0]:.2f} mm")

    angle_rads = np.deg2rad(angle_steps)
    x_coords = angle_rads * mean_radius
    y_coords = arc_steps
    r_grid = np.where(np.isnan(hit_distances), mean_radius, hit_distances)
    z_grid = (r_grid - mean_radius) * Z_SCALE

    xx, yy = np.meshgrid(x_coords, y_coords)
    verts = np.column_stack([xx.ravel(), yy.ravel(), z_grid.ravel()])

    faces = []
    for ai in range(n_arc - 1):
        for bi in range(n_angle-1):
            bi_next = (bi + 1) % n_angle
            i00 = ai * n_angle + bi
            i10 = (ai + 1) * n_angle + bi
            i01 = ai * n_angle + bi_next
            i11 = (ai + 1) * n_angle + bi_next
            faces.append([i00, i10, i11])
            faces.append([i00, i11, i01])

    faces = np.array(faces)
    print(f"Built 2.5D mesh: {len(verts)} vertices, {len(faces)} faces")
    return verts, faces

# ── 10. Save STL ──────────────────────────────────────────────────────────
def save_stl(verts, faces, path):
    mesh_data = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = verts[f[j]]
    mesh_data.save(path)
    print(f"Saved STL: {path}")

# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        "colon_core.py contains the shared functions and is not meant to be run directly.\n"
        "Run one of these instead:\n"
        "  generate_unrolls.py                           unrolled meshes for all datasets\n"
        "  generate_straightened_mesh.py                 straightened meshes\n"
        "  analysis/validate_batch/validate_batch.py     unrolling validation\n"
    )
