#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def read_param(config_text, name, default):
    pattern = rf"^\s*{re.escape(name)}\s*:\s*([^#\n]+)"
    match = re.search(pattern, config_text, re.MULTILINE)
    if not match:
        return default
    value = match.group(1).strip().strip('"').strip("'")
    try:
        if isinstance(default, int):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
    except ValueError:
        return default
    return value


def load_band_params(config_path):
    text = Path(config_path).read_text(encoding="utf-8")
    return {
        "band_count": read_param(text, "band_count", 20),
        "band_y_min_ratio": read_param(text, "band_y_min_ratio", 0.60),
        "band_y_max_ratio": read_param(text, "band_y_max_ratio", 1.0),
        "band_height_ratio": read_param(text, "band_height_ratio", 0.015),
        "min_segment_width_px": read_param(text, "min_segment_width_px", 60),
        "min_pixels_per_band": read_param(text, "min_pixels_per_band", 80),
    }


def make_mask(image, mode, threshold):
    if len(image.shape) == 2:
        gray = image
        return (gray >= threshold).astype(np.uint8)

    b, g, r = cv2.split(image)
    if mode == "blue":
        return ((b > threshold) & (b > g * 1.25) & (b > r * 1.25)).astype(np.uint8)
    if mode == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (gray >= threshold).astype(np.uint8)

    blue = ((b > threshold) & (b > g * 1.25) & (b > r * 1.25)).astype(np.uint8)
    if int(np.count_nonzero(blue)) > image.shape[0] * image.shape[1] * 0.01:
        return blue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray >= threshold).astype(np.uint8)


def extract_segments(band_mask, min_width, min_pixels):
    h, w = band_mask.shape
    col_counts = np.count_nonzero(band_mask == 1, axis=0)
    segments = []
    start = -1
    pixel_count = 0
    for x in range(w + 1):
        active = x < w and col_counts[x] > 0
        if active:
            if start < 0:
                start = x
                pixel_count = 0
            pixel_count += int(col_counts[x])
        elif start >= 0:
            end = x
            width = end - start
            if width >= min_width and pixel_count >= min_pixels:
                segments.append({
                    "x0": start,
                    "x1": end - 1,
                    "center": (start + end - 1) / 2.0,
                    "width": width,
                    "pixels": pixel_count,
                })
            start = -1
    return segments


def export_offsets(mask, params):
    h, w = mask.shape
    y_min = max(0, min(h - 1, int(h * params["band_y_min_ratio"])))
    y_max = max(y_min + 1, min(h, int(h * params["band_y_max_ratio"])))
    band_height = max(1, int(h * params["band_height_ratio"]))
    step = max(1, (y_max - y_min) // params["band_count"])

    offsets = []
    rows = []
    for i in range(params["band_count"]):
        y0 = y_min + i * step
        y1 = min(y0 + band_height, y_max)
        if y1 <= y0 or y0 >= h:
            offsets.append(-1.0)
            rows.append((i, y0, y1, 0, None, None, -1.0))
            continue
        segments = extract_segments(mask[y0:y1, :],
                                    params["min_segment_width_px"],
                                    params["min_pixels_per_band"])
        if not segments:
            offsets.append(-1.0)
            rows.append((i, y0, y1, 0, None, None, -1.0))
            continue
        left = min(segments, key=lambda s: s["x0"])
        offset = left["center"] - left["x0"]
        offsets.append(offset)
        rows.append((i, y0, y1, len(segments), left["x0"], left["center"], offset))
    return offsets, rows


def draw_debug(image, rows, out_path):
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for i, y0, y1, seg_count, left_x, center_x, offset in rows:
        cv2.rectangle(vis, (0, max(0, y0)), (vis.shape[1] - 1, min(vis.shape[0] - 1, y1 - 1)),
                      (255, 200, 100), 1)
        if left_x is not None and center_x is not None:
            cy = (y0 + y1) // 2
            cv2.circle(vis, (int(round(left_x)), cy), 3, (255, 0, 0), -1)
            cv2.circle(vis, (int(round(center_x)), cy), 3, (0, 255, 0), -1)
            cv2.line(vis, (int(round(left_x)), cy), (int(round(center_x)), cy), (0, 255, 255), 1)
    cv2.imwrite(str(out_path), vis)


def main():
    parser = argparse.ArgumentParser(
        description="Export left_boundary_template_offsets from a straight-road mask/screenshot.")
    parser.add_argument("image", help="Binary mask, grayscale mask, or fused_perception screenshot.")
    parser.add_argument("--config", default="src/track_perception_cpp/config/fused_perception.yaml",
                        help="Config file to read band parameters from.")
    parser.add_argument("--mask-mode", choices=["auto", "blue", "gray"], default="auto",
                        help="auto: prefer blue overlay mask, fallback to grayscale threshold.")
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--debug-image", default="")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"failed to read image: {args.image}")
    params = load_band_params(args.config)
    mask = make_mask(image, args.mask_mode, args.threshold)
    offsets, rows = export_offsets(mask, params)

    offset_text = ",".join(f"{v:.1f}" for v in offsets)
    valid = sum(1 for v in offsets if v > 0)
    print(f"# valid_offsets: {valid}/{len(offsets)}")
    print(f'left_boundary_template_offsets: "{offset_text}"')
    print()
    print("# band,index_y0_y1,segments,left_x,center_x,offset")
    for i, y0, y1, seg_count, left_x, center_x, offset in rows:
        lx = "None" if left_x is None else f"{left_x:.1f}"
        cx = "None" if center_x is None else f"{center_x:.1f}"
        print(f"# {i}: y={y0}-{y1} segments={seg_count} left={lx} center={cx} offset={offset:.1f}")

    if args.debug_image:
      draw_debug(image, rows, Path(args.debug_image))


if __name__ == "__main__":
    main()
