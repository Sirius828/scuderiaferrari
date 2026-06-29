#!/usr/bin/env python3
"""Probe model-free blue-arrow segmentation from shm_ar_video.

The script reads the same shared-memory frame format used by
track_perception_cpp:
  uint64 fid, uint32 width, uint32 height, followed by RGB uint8 image data.

It does not publish ROS topics. It is intended for fast visual/debug checks.
"""

import argparse
import mmap
import struct
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


HEADER = struct.Struct("QII")


def morph_kernel(args, size):
    shape = cv2.MORPH_ELLIPSE if args.morph_shape == "ellipse" else cv2.MORPH_RECT
    return cv2.getStructuringElement(shape, (size, size))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, n):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 255
    return out


def skeletonize(mask: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask)

    work = mask.copy()
    skeleton = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, kernel)
    return skeleton


def skeletonize_scaled(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.99:
        return skeletonize(mask)
    h, w = mask.shape
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    small_skel = skeletonize(small)
    skel = cv2.resize(small_skel, (w, h), interpolation=cv2.INTER_NEAREST)
    if cv2.countNonZero(skel) > 0:
        skel = cv2.morphologyEx(
            skel,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        )
    return skel


def keep_seed_components(mask: np.ndarray, args) -> np.ndarray:
    h, w = mask.shape
    y0 = int(h * args.corridor_seed_y_ratio)
    x0 = int(w * args.corridor_seed_x0)
    x1 = int(w * args.corridor_seed_x1)
    seed_roi = np.zeros_like(mask)
    seed_roi[max(0, y0):h, max(0, x0):min(w, x1)] = 255

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < args.corridor_seed_min_area:
            continue
        component = (labels == label)
        if np.any(component & (seed_roi > 0)):
            out[component] = 255
    return out


def build_corridor_mask(arrow_mask: np.ndarray, args) -> np.ndarray:
    corridor = arrow_mask.copy()
    if args.corridor_seed_close_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.corridor_seed_close_kernel, args.corridor_seed_close_kernel),
        )
        corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, k)
    if args.corridor_seed_dilate_x > 0 and args.corridor_seed_dilate_y > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.corridor_seed_dilate_x, args.corridor_seed_dilate_y),
        )
        corridor = cv2.dilate(corridor, k)
    if args.corridor_use_seed_filter:
        corridor = keep_seed_components(corridor, args)
    if args.corridor_close_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.corridor_close_kernel, args.corridor_close_kernel),
        )
        corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, k, iterations=args.corridor_close_iters)
    if args.corridor_dilate_x > 0 and args.corridor_dilate_y > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.corridor_dilate_x, args.corridor_dilate_y),
        )
        corridor = cv2.dilate(corridor, k, iterations=args.corridor_dilate_iters)
    if args.corridor_fill_holes:
        corridor = fill_holes(corridor)
    if args.corridor_open_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.corridor_open_kernel, args.corridor_open_kernel),
        )
        corridor = cv2.morphologyEx(corridor, cv2.MORPH_OPEN, k)
    corridor = remove_small_components(corridor, args.corridor_min_area)
    return corridor


def extract_arrow_components(mask: np.ndarray, args):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    h, w = mask.shape
    y_min = h * args.roi_y0
    y_max = h * args.roi_y1
    for label in range(1, n):
        x, y, bw, bh, area = stats[label]
        if area < args.path_min_area:
            continue
        cx, cy = centroids[label]
        if cy < y_min or cy > y_max:
            continue
        components.append(
            {
                "label": label,
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "area": int(area),
                "center": (float(cx), float(cy)),
            }
        )
    components.sort(key=lambda c: c["center"][1], reverse=True)
    return components


def build_component_edges(components, args):
    edges = []
    if len(components) < 2:
        return edges
    max_dx = float(args.path_max_dx)
    min_dy = float(args.path_min_dy)
    max_dy = float(args.path_max_dy)
    max_edges = max(1, int(args.path_max_edges_per_node))
    max_turn_cos = np.cos(np.deg2rad(float(args.path_max_turn_deg)))

    predecessors = [None] * len(components)
    for i, cur in enumerate(components):
        cx, cy = cur["center"]
        candidates = []
        for j, prev in enumerate(components):
            if i == j:
                continue
            dx = abs(cx - prev["center"][0])
            dy = prev["center"][1] - cy
            if dy < min_dy or dy > max_dy or dx > max_dx:
                continue
            score = dx + args.path_dy_weight * abs(dy - args.path_preferred_dy)
            candidates.append((score, j))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            predecessors[i] = candidates[0][1]

    for i, src in enumerate(components):
        sx, sy = src["center"]
        candidates = []
        for j, dst in enumerate(components):
            if i == j:
                continue
            dx = abs(dst["center"][0] - sx)
            dy = sy - dst["center"][1]
            if dy < min_dy or dy > max_dy or dx > max_dx:
                continue
            pred = predecessors[i]
            if pred is not None and args.path_max_turn_deg < 180:
                px, py = components[pred]["center"]
                vin = np.array([sx - px, sy - py], dtype=np.float64)
                vout = np.array([dst["center"][0] - sx, dst["center"][1] - sy], dtype=np.float64)
                denom = np.linalg.norm(vin) * np.linalg.norm(vout)
                if denom > 1e-6 and float(np.dot(vin, vout) / denom) < max_turn_cos:
                    continue
            score = dx + args.path_dy_weight * abs(dy - args.path_preferred_dy)
            candidates.append((score, j))
        candidates.sort(key=lambda item: item[0])
        for _score, j in candidates[:max_edges]:
            edges.append((i, j))
    return edges


def build_topology_edges(components, args):
    pairs = []
    for i, src in enumerate(components):
        sx, sy = src["center"]
        for j in range(i + 1, len(components)):
            tx, ty = components[j]["center"]
            dx = abs(tx - sx)
            dy = abs(ty - sy)
            dist = float(np.hypot(dx, dy))
            if dist > args.topology_max_dist:
                continue
            if dx > args.topology_max_dx or dy > args.topology_max_abs_dy:
                continue
            pairs.append((dist, i, j))

    degree = [0] * len(components)
    edges = []
    for _dist, i, j in sorted(pairs, key=lambda item: item[0]):
        if degree[i] >= args.topology_max_degree or degree[j] >= args.topology_max_degree:
            continue
        edges.append((i, j))
        degree[i] += 1
        degree[j] += 1
    return edges


def component_successors(components, args, index, prev_index=None, used=None, prefer_intermediate=True):
    if used is None:
        used = set()
    sx, sy = components[index]["center"]
    max_dx = float(args.path_max_dx)
    min_dy = float(args.path_min_dy)
    max_dy = float(args.path_max_dy)
    max_turn_cos = np.cos(np.deg2rad(float(args.path_max_turn_deg)))
    candidates = []

    for j, dst in enumerate(components):
        if j == index or j in used:
            continue
        dx_signed = dst["center"][0] - sx
        dx = abs(dx_signed)
        dy = sy - dst["center"][1]
        if dy < min_dy or dy > max_dy or dx > max_dx:
            continue
        if prev_index is not None and args.path_max_turn_deg < 180:
            px, py = components[prev_index]["center"]
            vin = np.array([sx - px, sy - py], dtype=np.float64)
            vout = np.array([dx_signed, dst["center"][1] - sy], dtype=np.float64)
            denom = np.linalg.norm(vin) * np.linalg.norm(vout)
            if denom > 1e-6 and float(np.dot(vin, vout) / denom) < max_turn_cos:
                continue
        score = dx + args.path_dy_weight * abs(dy - args.path_preferred_dy)
        candidates.append((score, j))

    candidates.sort(key=lambda item: item[0])
    ordered = [j for _score, j in candidates]
    if prefer_intermediate and args.path_prefer_intermediate and len(ordered) >= 2:
        for candidate in ordered:
            candidate_successors = component_successors(
                components, args, candidate, index, used | {candidate}, prefer_intermediate=False
            )
            if any(other in candidate_successors for other in ordered if other != candidate):
                ordered.remove(candidate)
                ordered.insert(0, candidate)
                break
    return ordered


def trace_path(components, args, start_index, prev_index=None, used=None):
    if used is None:
        used = set()
    path = []
    current = start_index
    previous = prev_index
    local_used = set(used)

    while current is not None and current not in local_used:
        path.append(current)
        local_used.add(current)
        successors = component_successors(components, args, current, previous, local_used)
        if not successors:
            break
        previous, current = current, successors[0]
    return path


def extract_paths(components, args):
    empty = {
        "trunk": [],
        "left": [],
        "right": [],
        "branch_index": None,
        "candidate_edges": [],
        "topology_edges": [],
    }
    if not components:
        return empty

    candidate_edges = build_component_edges(components, args)
    topology_edges = build_topology_edges(components, args)
    trunk = [0]
    previous = None
    current = 0
    used_trunk = {0}
    branch_successors = []

    while True:
        successors = component_successors(components, args, current, previous, used_trunk - {current})
        if not successors:
            break
        if len(successors) >= 2:
            s0, s1 = successors[0], successors[1]
            sep = abs(components[s0]["center"][0] - components[s1]["center"][0])
            s0_next = component_successors(components, args, s0, current, used_trunk | {s0})
            s1_next = component_successors(components, args, s1, current, used_trunk | {s1})
            same_chain = s1 in s0_next or s0 in s1_next
            s0_path = trace_path(components, args, s0, current, used_trunk)
            s1_path = trace_path(components, args, s1, current, used_trunk | set(s0_path))
            has_branch_length = (
                len(s0_path) >= args.path_branch_min_len and
                len(s1_path) >= args.path_branch_min_len
            )
            if sep >= args.path_branch_min_sep_px and not same_chain and has_branch_length:
                branch_successors = [s0, s1]
                break
        next_index = successors[0]
        if next_index in used_trunk:
            break
        previous, current = current, next_index
        trunk.append(current)
        used_trunk.add(current)

    left = []
    right = []
    branch_index = current if branch_successors else None
    if branch_successors:
        branch_successors.sort(key=lambda idx: components[idx]["center"][0])
        left_start, right_start = branch_successors[0], branch_successors[1]
        shared_used = set(trunk)
        left = trace_path(components, args, left_start, branch_index, shared_used)
        used_for_right = shared_used | set(left)
        right = trace_path(components, args, right_start, branch_index, used_for_right)

    return {
        "trunk": trunk,
        "left": left,
        "right": right,
        "branch_index": branch_index,
        "candidate_edges": candidate_edges,
        "topology_edges": topology_edges,
    }


class ShmFrameReader:
    def __init__(self, name: str):
        self.path = Path("/dev/shm") / name
        self.file = None
        self.map = None
        self.size = 0
        self.last_fid = 0

    def open(self):
        self.file = self.path.open("rb")
        self.size = self.path.stat().st_size
        self.map = mmap.mmap(self.file.fileno(), self.size, access=mmap.ACCESS_READ)

    def close(self):
        if self.map is not None:
            self.map.close()
            self.map = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def read_latest(self, wait_new: bool = True, timeout: float = 1.0):
        if self.map is None:
            self.open()

        deadline = time.time() + timeout
        while True:
            self.map.seek(0)
            header = self.map.read(HEADER.size)
            fid, width, height = HEADER.unpack(header)
            image_size = int(width) * int(height) * 3
            if fid and width and height and HEADER.size + image_size <= self.size:
                if not wait_new or fid != self.last_fid:
                    self.map.seek(HEADER.size)
                    image = np.frombuffer(self.map.read(image_size), dtype=np.uint8)
                    image = image.reshape((height, width, 3)).copy()
                    self.last_fid = fid
                    return fid, image
            if time.time() >= deadline:
                return None, None
            time.sleep(0.002)


def make_blue_mask(rgb: np.ndarray, args) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        (args.h_min, args.s_min, args.v_min),
        (args.h_max, args.s_max, args.v_max),
    )

    if args.blue_dominance > 0:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        dominant = (b > r + args.blue_dominance) & (b > g + args.blue_dominance // 2)
        mask = cv2.bitwise_and(mask, dominant.astype(np.uint8) * 255)

    h, w = mask.shape
    roi = np.zeros_like(mask)
    y0 = int(h * args.roi_y0)
    y1 = int(h * args.roi_y1)
    x0 = int(w * args.roi_x0)
    x1 = int(w * args.roi_x1)
    roi[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255
    mask = cv2.bitwise_and(mask, roi)

    if args.open_kernel > 0:
        k = morph_kernel(args, args.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if args.close_kernel > 0:
        k = morph_kernel(args, args.close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if args.fill_holes:
        mask = fill_holes(mask)
    if args.dilate_kernel > 0:
        k = morph_kernel(args, args.dilate_kernel)
        mask = cv2.dilate(mask, k, iterations=args.dilate_iters)
    return mask


def make_path_mask(rgb: np.ndarray, args) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        (args.h_min, args.s_min, args.v_min),
        (args.h_max, args.s_max, args.v_max),
    )
    if args.blue_dominance > 0:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        dominant = (b > r + args.blue_dominance) & (b > g + args.blue_dominance // 2)
        mask = cv2.bitwise_and(mask, dominant.astype(np.uint8) * 255)

    h, w = mask.shape
    roi = np.zeros_like(mask)
    roi[int(h * args.roi_y0):int(h * args.roi_y1), int(w * args.roi_x0):int(w * args.roi_x1)] = 255
    mask = cv2.bitwise_and(mask, roi)

    if args.path_close_kernel <= 0 and args.path_dilate_kernel <= 0:
        return mask

    morph = mask.copy()
    if args.path_close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.path_close_kernel, args.path_close_kernel))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, k)
    if args.path_dilate_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.path_dilate_kernel, args.path_dilate_kernel))
        morph = cv2.dilate(morph, k, iterations=args.path_dilate_iters)

    y_split = int(h * args.path_morph_y_ratio)
    out = mask.copy()
    out[max(0, y_split):h, :] = morph[max(0, y_split):h, :]
    return out


def merge_temporal_masks(history):
    merged = None
    for item in history:
        merged = item.copy() if merged is None else cv2.bitwise_or(merged, item)
    return merged


def normalize_frame_orientation(rgb: np.ndarray, args) -> np.ndarray:
    if args.input_format == "BGR":
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if args.flip_code == 0:
        rgb = cv2.flip(rgb, 0)
    elif args.flip_code == 1:
        rgb = cv2.flip(rgb, 1)
    elif args.flip_code == -1:
        rgb = cv2.flip(rgb, -1)
    if args.mirror:
        rgb = cv2.flip(rgb, 1)
    return rgb


def extract_band_points(mask: np.ndarray, args):
    h, w = mask.shape
    y0 = int(h * args.roi_y0)
    y1 = int(h * args.roi_y1)
    band_h = max(2, int(h * args.band_height_ratio))
    step = max(1, (y1 - y0) // max(1, args.band_count))
    points = []
    bands = []

    for i in range(args.band_count):
        by0 = max(0, y0 + i * step)
        by1 = min(h, by0 + band_h)
        if by1 <= by0:
            continue
        band = mask[by0:by1, :]
        ys, xs = np.nonzero(band)
        pix = int(xs.size)
        if pix < args.min_pixels_per_band:
            bands.append((by0, by1, None, pix, None, None))
            continue
        edge_pct = float(np.clip(args.edge_percentile, 0.0, 20.0))
        x_left = float(np.percentile(xs, edge_pct))
        x_right = float(np.percentile(xs, 100.0 - edge_pct))
        x_mean = float(np.mean(xs))
        x_midspan = 0.5 * (x_left + x_right)
        if args.center_method == "mean":
            x_center = x_mean
        elif args.center_method == "hybrid":
            x_center = args.hybrid_edge_weight * x_midspan + (1.0 - args.hybrid_edge_weight) * x_mean
        else:
            x_center = x_midspan
        y_center = float((by0 + by1 - 1) * 0.5)
        points.append((x_center, y_center, min(3.0, 1.0 + pix / 1500.0)))
        bands.append((by0, by1, x_center, pix, x_left, x_right))
    return points, bands


def band_to_point(band):
    y0, y1, x_center, pix, _x_left, _x_right = band
    if x_center is None:
        return None
    y_center = float((y0 + y1 - 1) * 0.5)
    return (float(x_center), y_center, min(3.0, 1.0 + float(pix) / 1500.0))


def select_main_track_bands(bands, image_width, args):
    selected_reversed = []
    last_x = None
    max_jump = float(args.max_center_jump_px)
    max_span = float(args.max_band_span_ratio) * float(image_width) if args.max_band_span_ratio > 0 else None

    for band in reversed(bands):
        _y0, _y1, x_center, pix, x_left, x_right = band
        if x_center is None:
            continue
        span = float(x_right - x_left)
        if max_span is not None and span > max_span:
            continue
        if last_x is not None and max_jump > 0 and abs(float(x_center) - last_x) > max_jump:
            continue
        selected_reversed.append(band)
        last_x = float(x_center)

    selected = list(reversed(selected_reversed))
    points = [band_to_point(b) for b in selected]
    points = [p for p in points if p is not None]
    return points, selected


def dump_band_debug(fid, bands, selected_bands):
    selected_ids = {id(b) for b in selected_bands}
    print(f"band_debug fid={fid}")
    print("idx keep y0-y1 cy left center right span pix")
    for i, band in enumerate(bands):
        y0, y1, x_center, pix, x_left, x_right = band
        keep = "Y" if id(band) in selected_ids else "N"
        cy = (y0 + y1) // 2
        if x_center is None:
            print(f"{i:02d} {keep} {y0:3d}-{y1:3d} {cy:3d} none pix={pix}")
            continue
        print(
            f"{i:02d} {keep} {y0:3d}-{y1:3d} {cy:3d} "
            f"{x_left:7.1f} {x_center:7.1f} {x_right:7.1f} "
            f"span={x_right - x_left:7.1f} pix={pix}"
        )


def fit_offset(points, image_shape, args):
    h, w = image_shape[:2]
    if len(points) < args.fit_min_points:
        return None
    pts = np.asarray(points, dtype=np.float64)
    order = min(args.fit_order, len(points) - 1)
    coeff = np.polyfit(pts[:, 1], pts[:, 0], order, w=pts[:, 2])
    max_y = float(np.max(pts[:, 1]))
    target_y = min(max_y, h * args.lookahead_y_ratio)
    target_x = float(np.polyval(coeff, target_y))
    offset = (target_x - w * 0.5) / (w * 0.5)
    offset = float(np.clip(offset, -1.0, 1.0))
    confidence = min(1.0, len(points) / max(1, args.band_count * 0.65))
    return offset, target_x, target_y, coeff, confidence


def build_stitched_track(points, image_shape, args):
    h, w = image_shape[:2]
    if len(points) < args.fit_min_points:
        return None

    pts = np.asarray(points, dtype=np.float64)
    order = min(args.fit_order, len(points) - 1)
    coeff = np.polyfit(pts[:, 1], pts[:, 0], order, w=pts[:, 2])
    deriv = np.polyder(coeff)

    y_min = max(h * args.roi_y0, float(np.min(pts[:, 1])))
    y_max = min(h * args.roi_y1, float(np.max(pts[:, 1])))
    if y_max <= y_min:
        return None

    samples = max(12, int(args.stitch_samples))
    ys = np.linspace(y_min, y_max, samples)
    xs = np.polyval(coeff, ys)
    dx_dy = np.polyval(deriv, ys) if deriv.size else np.zeros_like(ys)

    denom = max(1e-6, h * (args.roi_y1 - args.roi_y0))
    t = np.clip((ys - h * args.roi_y0) / denom, 0.0, 1.0)
    half_width = args.boundary_half_width_far * (1.0 - t) + args.boundary_half_width_near * t

    # Tangent is (dx/dy, 1); image-space normal is (-1, dx/dy).
    nx = -np.ones_like(dx_dy)
    ny = dx_dy
    norm = np.sqrt(nx * nx + ny * ny)
    nx /= norm
    ny /= norm

    center = np.column_stack((xs, ys))
    left = np.column_stack((xs + nx * half_width, ys + ny * half_width))
    right = np.column_stack((xs - nx * half_width, ys - ny * half_width))

    center[:, 0] = np.clip(center[:, 0], 0, w - 1)
    center[:, 1] = np.clip(center[:, 1], 0, h - 1)
    left[:, 0] = np.clip(left[:, 0], 0, w - 1)
    left[:, 1] = np.clip(left[:, 1], 0, h - 1)
    right[:, 0] = np.clip(right[:, 0], 0, w - 1)
    right[:, 1] = np.clip(right[:, 1], 0, h - 1)

    corridor = np.vstack((left, right[::-1])).astype(np.int32)
    return {
        "coeff": coeff,
        "center": center.astype(np.int32),
        "left": left.astype(np.int32),
        "right": right.astype(np.int32),
        "corridor": corridor,
        "half_width": half_width,
        "normal": np.column_stack((nx, ny)),
        "point_count": len(points),
    }


def dump_stitch_debug(fid, stitched):
    if stitched is None:
        print(f"stitch_debug fid={fid} invalid")
        return
    center = stitched["center"]
    normal = stitched["normal"]
    half_width = stitched["half_width"]
    print(f"stitch_debug fid={fid} samples={len(center)}")
    print("idx center_x center_y normal_x normal_y half_width")
    for i in np.linspace(0, len(center) - 1, min(12, len(center))).astype(int):
        print(
            f"{i:02d} {center[i,0]:8d} {center[i,1]:8d} "
            f"{normal[i,0]:8.3f} {normal[i,1]:8.3f} {half_width[i]:10.1f}"
        )


def render_debug(rgb, mask, bands, fit):
    overlay = rgb.copy()
    overlay[mask > 0] = (0.35 * overlay[mask > 0] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    h, w = mask.shape
    cv2.line(overlay, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1, cv2.LINE_AA)

    for y0, y1, x_center, pix, x_left, x_right in bands:
        color = (0, 255, 255) if x_center is not None else (80, 80, 80)
        cv2.line(overlay, (0, y0), (w - 1, y0), color, 1)
        if x_center is not None:
            cy = (y0 + y1) // 2
            cv2.circle(overlay, (int(round(x_center)), (y0 + y1) // 2), 4, (0, 255, 0), -1)
            cv2.circle(overlay, (int(round(x_left)), cy), 3, (255, 0, 255), -1)
            cv2.circle(overlay, (int(round(x_right)), cy), 3, (255, 255, 0), -1)

    if fit is not None:
        offset, target_x, target_y, coeff, confidence = fit
        ys = np.linspace(max(0, min(y0 for y0, *_rest in bands)), h - 1, 80)
        xs = np.polyval(coeff, ys)
        prev = None
        for x, y in zip(xs, ys):
            pt = (int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1))))
            if prev is not None:
                cv2.line(overlay, prev, pt, (0, 0, 255), 2, cv2.LINE_AA)
            prev = pt
        cv2.circle(overlay, (int(round(target_x)), int(round(target_y))), 7, (255, 255, 0), -1)
        cv2.putText(
            overlay,
            f"offset={offset:+.3f} conf={confidence:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def render_panel(rgb, mask, overlay, fid, dt_ms, fit):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]
    top = np.hstack((rgb, overlay))
    mask_panel = np.zeros_like(top)
    mask_resized = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_panel[:, :w] = mask_resized
    mask_panel[:, w:] = mask_resized
    panel = np.vstack((top, mask_panel))

    status = f"fid={fid} dt={dt_ms:.1f}ms"
    if fit is not None:
        offset, _target_x, _target_y, _coeff, confidence = fit
        status += f" offset={offset:+.3f} conf={confidence:.2f}"
    else:
        status += " invalid"
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1] - 1, 620), 34), (0, 0, 0), -1)
    cv2.putText(
        panel,
        status,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def render_binary(mask, fid, process_ms, mem_fps, proc_fps, fit):
    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    status = f"fid={fid} mem_fps={mem_fps:.1f} proc_fps={proc_fps:.1f} proc={process_ms:.1f}ms"
    if fit is not None:
        offset, _target_x, _target_y, _coeff, confidence = fit
        status += f" offset={offset:+.3f} conf={confidence:.2f}"
    else:
        status += " invalid"
    cv2.rectangle(view, (0, 0), (view.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        view,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return view


def render_stitched_track(rgb, mask, bands, stitched, fid, process_ms, mem_fps, proc_fps):
    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    view[mask > 0] = (80, 80, 80)

    if stitched is not None:
        fill = np.zeros_like(view)
        cv2.fillPoly(fill, [stitched["corridor"]], (35, 80, 35))
        view = cv2.addWeighted(view, 1.0, fill, 0.75, 0.0)

        cv2.polylines(view, [stitched["left"]], False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(view, [stitched["right"]], False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(view, [stitched["center"]], False, (0, 255, 0), 2, cv2.LINE_AA)

    left_edge_points = []
    right_edge_points = []
    for y0, y1, x_center, pix, x_left, x_right in bands:
        if x_center is None:
            continue
        cy = (y0 + y1) // 2
        cv2.circle(view, (int(round(x_center)), cy), 3, (0, 255, 255), -1)
        cv2.circle(view, (int(round(x_left)), cy), 3, (255, 0, 255), -1)
        cv2.circle(view, (int(round(x_right)), cy), 3, (255, 255, 0), -1)
        left_edge_points.append((int(round(x_left)), cy))
        right_edge_points.append((int(round(x_right)), cy))

    if len(left_edge_points) >= 2:
        cv2.polylines(view, [np.asarray(left_edge_points, dtype=np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    if len(right_edge_points) >= 2:
        cv2.polylines(view, [np.asarray(right_edge_points, dtype=np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)

    status = f"fid={fid} mem_fps={mem_fps:.1f} proc_fps={proc_fps:.1f} proc={process_ms:.1f}ms"
    if stitched is not None:
        status += f" stitch_points={stitched['point_count']}"
    else:
        status += " stitch_invalid"
    cv2.rectangle(view, (0, 0), (view.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        view,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return view


def render_corridor(mask, corridor, skeleton, fid, process_ms, mem_fps, proc_fps):
    view = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    view[mask > 0] = (90, 90, 90)
    view[corridor > 0] = (25, 90, 35)

    contours, _hierarchy = cv2.findContours(corridor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    view[skeleton > 0] = (0, 255, 255)

    # Mark likely branch points on the skeleton: pixels with at least 3 skeleton neighbors.
    sk = (skeleton > 0).astype(np.uint8)
    if cv2.countNonZero(skeleton) > 0:
        neighbor_count = cv2.filter2D(sk, -1, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT)
        branch = ((sk > 0) & (neighbor_count >= 4)).astype(np.uint8) * 255
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(branch, connectivity=8)
        for label in range(1, n):
            if int(stats[label, cv2.CC_STAT_AREA]) < 2:
                continue
            cx, cy = centroids[label]
            cv2.circle(view, (int(round(cx)), int(round(cy))), 6, (255, 0, 255), 2, cv2.LINE_AA)

    status = (
        f"fid={fid} mem_fps={mem_fps:.1f} proc_fps={proc_fps:.1f} proc={process_ms:.1f}ms "
        f"arrow_px={cv2.countNonZero(mask)} corridor_px={cv2.countNonZero(corridor)} "
        f"skel_px={cv2.countNonZero(skeleton)} contours={len(contours)}"
    )
    cv2.rectangle(view, (0, 0), (view.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        view,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return view


def draw_component_path(view, components, path, color, thickness=3):
    if not path:
        return
    pts = np.asarray(
        [tuple(int(round(v)) for v in components[idx]["center"]) for idx in path],
        dtype=np.int32,
    )
    if len(pts) >= 2:
        cv2.polylines(view, [pts], False, color, thickness, cv2.LINE_AA)
    for idx in path:
        cx, cy = components[idx]["center"]
        cv2.circle(view, (int(round(cx)), int(round(cy))), thickness + 2, color, -1)


def render_path_graph(mask, components, path_result, fid, process_ms, mem_fps, proc_fps, args):
    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    view[mask > 0] = (80, 80, 80)

    if args.show_topology_edges:
        for i, j in path_result["topology_edges"]:
            p0 = tuple(int(round(v)) for v in components[i]["center"])
            p1 = tuple(int(round(v)) for v in components[j]["center"])
            cv2.line(view, p0, p1, (255, 0, 0), 2, cv2.LINE_AA)

    if args.show_candidate_edges:
        for i, j in path_result["candidate_edges"]:
            p0 = tuple(int(round(v)) for v in components[i]["center"])
            p1 = tuple(int(round(v)) for v in components[j]["center"])
            cv2.line(view, p0, p1, (80, 120, 120), 1, cv2.LINE_AA)

    draw_component_path(view, components, path_result["trunk"], (0, 255, 0), 3)
    draw_component_path(view, components, path_result["left"], (255, 0, 255), 3)
    draw_component_path(view, components, path_result["right"], (255, 255, 0), 3)
    if path_result["branch_index"] is not None:
        cx, cy = components[path_result["branch_index"]]["center"]
        cv2.circle(view, (int(round(cx)), int(round(cy))), 9, (0, 0, 255), 2, cv2.LINE_AA)

    trunk_set = set(path_result["trunk"])
    left_set = set(path_result["left"])
    right_set = set(path_result["right"])
    for idx, comp in enumerate(components):
        x, y, bw, bh = comp["bbox"]
        cx, cy = comp["center"]
        color = (255, 0, 0)
        if idx in trunk_set:
            color = (0, 255, 0)
        elif idx in left_set:
            color = (255, 0, 255)
        elif idx in right_set:
            color = (255, 255, 0)
        cv2.rectangle(view, (x, y), (x + bw, y + bh), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(view, (int(round(cx)), int(round(cy))), 4, color, -1)
        cv2.putText(
            view,
            str(idx),
            (int(round(cx)) + 4, int(round(cy)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    status = (
        f"fid={fid} mem_fps={mem_fps:.1f} proc_fps={proc_fps:.1f} proc={process_ms:.1f}ms "
        f"components={len(components)} trunk={len(path_result['trunk'])} "
        f"left={len(path_result['left'])} right={len(path_result['right'])} "
        f"topo={len(path_result['topology_edges'])}"
    )
    cv2.rectangle(view, (0, 0), (view.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        view,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return view


def create_hsv_trackbars(window_name, args):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", window_name, int(args.h_min), 179, lambda _v: None)
    cv2.createTrackbar("H max", window_name, int(args.h_max), 179, lambda _v: None)
    cv2.createTrackbar("S min", window_name, int(args.s_min), 255, lambda _v: None)
    cv2.createTrackbar("S max", window_name, int(args.s_max), 255, lambda _v: None)
    cv2.createTrackbar("V min", window_name, int(args.v_min), 255, lambda _v: None)
    cv2.createTrackbar("V max", window_name, int(args.v_max), 255, lambda _v: None)


def read_hsv_trackbars(window_name, args):
    args.h_min = cv2.getTrackbarPos("H min", window_name)
    args.h_max = cv2.getTrackbarPos("H max", window_name)
    args.s_min = cv2.getTrackbarPos("S min", window_name)
    args.s_max = cv2.getTrackbarPos("S max", window_name)
    args.v_min = cv2.getTrackbarPos("V min", window_name)
    args.v_max = cv2.getTrackbarPos("V max", window_name)
    if args.h_min > args.h_max:
        args.h_min, args.h_max = args.h_max, args.h_min
    if args.s_min > args.s_max:
        args.s_min, args.s_max = args.s_max, args.s_min
    if args.v_min > args.v_max:
        args.v_min, args.v_max = args.v_max, args.v_min


def format_hsv_args(args):
    return (
        f"--h-min {args.h_min} --h-max {args.h_max} "
        f"--s-min {args.s_min} --s-max {args.s_max} "
        f"--v-min {args.v_min} --v-max {args.v_max}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-name", default="shm_ar_video")
    parser.add_argument("--input-format", choices=("RGB", "BGR"), default="RGB")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 means run forever unless --once is set")
    parser.add_argument("--save-dir", default="/tmp/blue_arrow_probe")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--panel", action="store_true", help="show original/overlay/mask panel instead of overlay only")
    parser.add_argument("--overlay", action="store_true", help="show overlay instead of binary mask")
    parser.add_argument("--stitch", action="store_true", help="show model-free track reconstruction debug view")
    parser.add_argument("--stitch-mode", choices=("paths", "corridor"), default="paths")
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--hsv-trackbars", action="store_true", help="show HSV tuning trackbars")
    parser.add_argument(
        "--flip-code",
        type=int,
        default=-1,
        choices=(-2, -1, 0, 1),
        help="-1 rotates 180 deg, 0 vertical flip, 1 horizontal flip, -2 disables flipping",
    )
    parser.add_argument("--mirror", dest="mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")

    parser.add_argument("--h-min", type=int, default=100)
    parser.add_argument("--h-max", type=int, default=109)
    parser.add_argument("--s-min", type=int, default=136)
    parser.add_argument("--s-max", type=int, default=255)
    parser.add_argument("--v-min", type=int, default=121)
    parser.add_argument("--v-max", type=int, default=255)
    parser.add_argument("--blue-dominance", type=int, default=15)

    parser.add_argument("--roi-x0", type=float, default=0.0)
    parser.add_argument("--roi-x1", type=float, default=1.0)
    parser.add_argument("--roi-y0", type=float, default=0.62)
    parser.add_argument("--roi-y1", type=float, default=0.96)
    parser.add_argument("--morph-shape", choices=("ellipse", "rect"), default="ellipse")
    parser.add_argument("--open-kernel", type=int, default=0)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--fill-holes", dest="fill_holes", action="store_true", default=True)
    parser.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")
    parser.add_argument("--dilate-kernel", type=int, default=5)
    parser.add_argument("--dilate-iters", type=int, default=1)

    parser.add_argument("--band-count", type=int, default=14)
    parser.add_argument("--band-height-ratio", type=float, default=0.025)
    parser.add_argument("--min-pixels-per-band", type=int, default=30)
    parser.add_argument(
        "--center-method",
        choices=("span", "mean", "hybrid"),
        default="span",
        help="span uses left/right edge midpoint; mean uses pixel centroid; hybrid blends both",
    )
    parser.add_argument("--edge-percentile", type=float, default=2.0)
    parser.add_argument("--hybrid-edge-weight", type=float, default=0.75)
    parser.add_argument("--max-center-jump-px", type=float, default=85.0)
    parser.add_argument("--max-band-span-ratio", type=float, default=0.45)
    parser.add_argument("--dump-bands", action="store_true")
    parser.add_argument("--dump-stitch", action="store_true")
    parser.add_argument("--fit-min-points", type=int, default=4)
    parser.add_argument("--fit-order", type=int, default=2)
    parser.add_argument("--lookahead-y-ratio", type=float, default=0.80)
    parser.add_argument("--stitch-samples", type=int, default=40)
    parser.add_argument("--boundary-half-width-far", type=float, default=38.0)
    parser.add_argument("--boundary-half-width-near", type=float, default=105.0)
    parser.add_argument("--corridor-close-kernel", type=int, default=21)
    parser.add_argument("--corridor-close-iters", type=int, default=1)
    parser.add_argument("--corridor-dilate-x", type=int, default=31)
    parser.add_argument("--corridor-dilate-y", type=int, default=19)
    parser.add_argument("--corridor-dilate-iters", type=int, default=1)
    parser.add_argument("--corridor-open-kernel", type=int, default=0)
    parser.add_argument("--corridor-fill-holes", dest="corridor_fill_holes", action="store_true", default=False)
    parser.add_argument("--no-corridor-fill-holes", dest="corridor_fill_holes", action="store_false")
    parser.add_argument("--corridor-min-area", type=int, default=800)
    parser.add_argument("--corridor-use-seed-filter", dest="corridor_use_seed_filter", action="store_true", default=True)
    parser.add_argument("--no-corridor-seed-filter", dest="corridor_use_seed_filter", action="store_false")
    parser.add_argument("--corridor-seed-y-ratio", type=float, default=0.78)
    parser.add_argument("--corridor-seed-x0", type=float, default=0.0)
    parser.add_argument("--corridor-seed-x1", type=float, default=1.0)
    parser.add_argument("--corridor-seed-min-area", type=int, default=300)
    parser.add_argument("--corridor-seed-close-kernel", type=int, default=17)
    parser.add_argument("--corridor-seed-dilate-x", type=int, default=29)
    parser.add_argument("--corridor-seed-dilate-y", type=int, default=17)
    parser.add_argument("--skeleton-scale", type=float, default=0.5)
    parser.add_argument("--path-min-area", type=int, default=120)
    parser.add_argument("--path-close-kernel", type=int, default=5)
    parser.add_argument("--path-dilate-kernel", type=int, default=1)
    parser.add_argument("--path-dilate-iters", type=int, default=1)
    parser.add_argument("--path-morph-y-ratio", type=float, default=0.72)
    parser.add_argument("--path-temporal-frames", type=int, default=1)
    parser.add_argument("--path-max-dx", type=float, default=140.0)
    parser.add_argument("--path-min-dy", type=float, default=8.0)
    parser.add_argument("--path-max-dy", type=float, default=70.0)
    parser.add_argument("--path-preferred-dy", type=float, default=38.0)
    parser.add_argument("--path-dy-weight", type=float, default=1.0)
    parser.add_argument("--path-max-edges-per-node", type=int, default=2)
    parser.add_argument("--path-max-turn-deg", type=float, default=85.0)
    parser.add_argument("--path-prefer-intermediate", dest="path_prefer_intermediate", action="store_true", default=True)
    parser.add_argument("--no-path-prefer-intermediate", dest="path_prefer_intermediate", action="store_false")
    parser.add_argument("--path-branch-min-sep-px", type=float, default=30.0)
    parser.add_argument("--path-branch-min-len", type=int, default=2)
    parser.add_argument("--show-candidate-edges", action="store_true")
    parser.add_argument("--show-topology-edges", dest="show_topology_edges", action="store_true", default=True)
    parser.add_argument("--no-topology-edges", dest="show_topology_edges", action="store_false")
    parser.add_argument("--topology-max-dist", type=float, default=170.0)
    parser.add_argument("--topology-max-dx", type=float, default=165.0)
    parser.add_argument("--topology-max-abs-dy", type=float, default=55.0)
    parser.add_argument("--topology-max-degree", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reader = ShmFrameReader(args.shm_name)
    count = 0
    prev_fid = None
    prev_frame_time = None
    mem_fps = 0.0
    proc_fps = 0.0
    path_mask_history = deque(maxlen=max(1, int(args.path_temporal_frames)))
    window_name = "blue_arrow_probe"
    if args.show and args.hsv_trackbars:
        create_hsv_trackbars(window_name, args)
    try:
        while True:
            t_read0 = time.perf_counter()
            fid, rgb = reader.read_latest(wait_new=not args.once)
            if rgb is None:
                print("no new frame")
                continue
            t_proc0 = time.perf_counter()
            if prev_fid is not None and prev_frame_time is not None:
                elapsed = max(1e-6, t_proc0 - prev_frame_time)
                inst_mem_fps = max(0, fid - prev_fid) / elapsed
                mem_fps = inst_mem_fps if mem_fps <= 0.0 else 0.85 * mem_fps + 0.15 * inst_mem_fps
            prev_fid = fid
            prev_frame_time = t_proc0

            rgb = normalize_frame_orientation(rgb, args)
            if args.show and args.hsv_trackbars:
                read_hsv_trackbars(window_name, args)
            if args.stitch and args.stitch_mode == "paths":
                mask = make_path_mask(rgb, args)
                if args.path_temporal_frames > 1:
                    path_mask_history.append(mask)
                    mask = merge_temporal_masks(path_mask_history)
            else:
                mask = make_blue_mask(rgb, args)
            raw_points, raw_bands = extract_band_points(mask, args)
            points, bands = raw_points, raw_bands
            fit = fit_offset(points, rgb.shape, args)
            corridor = None
            skel = None
            path_components = []
            path_result = {
                "trunk": [],
                "left": [],
                "right": [],
                "branch_index": None,
                "candidate_edges": [],
                "topology_edges": [],
            }
            if args.stitch and args.stitch_mode == "corridor":
                corridor = build_corridor_mask(mask, args)
                skel = skeletonize_scaled(corridor, args.skeleton_scale)
            elif args.stitch:
                path_components = extract_arrow_components(mask, args)
                path_result = extract_paths(path_components, args)
            if args.dump_bands:
                dump_band_debug(fid, raw_bands, bands)
            if args.dump_stitch:
                if args.stitch_mode == "corridor":
                    print(
                        f"corridor_debug fid={fid} arrow_px={cv2.countNonZero(mask)} "
                        f"corridor_px={cv2.countNonZero(corridor) if corridor is not None else 0} "
                        f"skel_px={cv2.countNonZero(skel) if skel is not None else 0}"
                    )
                else:
                    print(
                        f"path_debug fid={fid} components={len(path_components)} "
                        f"trunk={path_result['trunk']} left={path_result['left']} "
                        f"right={path_result['right']} candidates={len(path_result['candidate_edges'])} "
                        f"topology={path_result['topology_edges']} "
                        f"mask_px={cv2.countNonZero(mask)}"
                    )
            process_ms = (time.perf_counter() - t_proc0) * 1000.0
            total_ms = (time.perf_counter() - t_read0) * 1000.0
            inst_proc_fps = 1000.0 / max(1e-6, process_ms)
            proc_fps = inst_proc_fps if proc_fps <= 0.0 else 0.85 * proc_fps + 0.15 * inst_proc_fps

            if args.stitch:
                if args.stitch_mode == "corridor":
                    print(
                        f"fid={fid} arrow_px={int(cv2.countNonZero(mask))} "
                        f"corridor_px={int(cv2.countNonZero(corridor))} "
                        f"skel_px={int(cv2.countNonZero(skel))} mem_fps={mem_fps:.1f} "
                        f"proc_fps={proc_fps:.1f} proc={process_ms:.2f}ms total={total_ms:.2f}ms"
                    )
                else:
                    print(
                        f"fid={fid} components={len(path_components)} trunk={len(path_result['trunk'])} "
                        f"left={len(path_result['left'])} right={len(path_result['right'])} "
                        f"mask_px={int(cv2.countNonZero(mask))} mem_fps={mem_fps:.1f} "
                        f"proc_fps={proc_fps:.1f} proc={process_ms:.2f}ms total={total_ms:.2f}ms"
                    )
            elif fit is None:
                print(
                    f"fid={fid} valid=0 points={len(points)} mask_px={int(cv2.countNonZero(mask))} "
                    f"mem_fps={mem_fps:.1f} proc_fps={proc_fps:.1f} proc={process_ms:.2f}ms total={total_ms:.2f}ms"
                )
            else:
                offset, target_x, target_y, _coeff, confidence = fit
                print(
                    f"fid={fid} valid=1 offset={offset:+.3f} "
                    f"target=({target_x:.1f},{target_y:.1f}) points={len(points)} "
                    f"conf={confidence:.2f} mask_px={int(cv2.countNonZero(mask))} "
                    f"mem_fps={mem_fps:.1f} proc_fps={proc_fps:.1f} proc={process_ms:.2f}ms total={total_ms:.2f}ms"
                )

            if count % max(1, args.save_every) == 0:
                overlay = render_debug(rgb, mask, bands, fit)
                cv2.imwrite(str(save_dir / "frame_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_dir / "mask.png"), mask)
                cv2.imwrite(str(save_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if args.stitch:
                    if args.stitch_mode == "corridor":
                        stitch_view = render_corridor(mask, corridor, skel, fid, process_ms, mem_fps, proc_fps)
                    else:
                        stitch_view = render_path_graph(
                            mask, path_components, path_result, fid, process_ms, mem_fps, proc_fps, args
                        )
                    cv2.imwrite(str(save_dir / "stitch.png"), cv2.cvtColor(stitch_view, cv2.COLOR_RGB2BGR))

            if args.show:
                overlay = None
                if args.panel or args.overlay:
                    overlay = render_debug(rgb, mask, bands, fit)
                if args.stitch:
                    if args.stitch_mode == "corridor":
                        view = render_corridor(mask, corridor, skel, fid, process_ms, mem_fps, proc_fps)
                    else:
                        view = render_path_graph(
                            mask, path_components, path_result, fid, process_ms, mem_fps, proc_fps, args
                        )
                elif args.panel:
                    view = render_panel(rgb, mask, overlay, fid, process_ms, fit)
                elif args.overlay:
                    view = overlay
                else:
                    view = render_binary(mask, fid, process_ms, mem_fps, proc_fps, fit)
                if args.display_scale > 0 and abs(args.display_scale - 1.0) > 1e-3:
                    view = cv2.resize(
                        view,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(window_name, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("p"):
                    print(format_hsv_args(args))
                if key == ord("s"):
                    if overlay is None:
                        overlay = render_debug(rgb, mask, bands, fit)
                    overlay_path = save_dir / f"overlay_fid_{fid}.png"
                    mask_path = save_dir / f"mask_fid_{fid}.png"
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(mask_path), mask)
                    saved = f"saved {overlay_path} {mask_path}"
                    if args.stitch:
                        if args.stitch_mode == "corridor":
                            stitch_view = render_corridor(mask, corridor, skel, fid, process_ms, mem_fps, proc_fps)
                        else:
                            stitch_view = render_path_graph(
                                mask, path_components, path_result, fid, process_ms, mem_fps, proc_fps, args
                            )
                        stitch_path = save_dir / f"stitch_fid_{fid}.png"
                        cv2.imwrite(str(stitch_path), cv2.cvtColor(stitch_view, cv2.COLOR_RGB2BGR))
                        saved += f" {stitch_path}"
                    print(saved)
                    print(format_hsv_args(args))

            count += 1
            if args.once or (args.frames > 0 and count >= args.frames):
                break
    finally:
        reader.close()


if __name__ == "__main__":
    main()
