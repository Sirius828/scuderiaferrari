#!/usr/bin/env python3
"""Convert an Ultralytics YOLO11-seg PT checkpoint to RK3588 raw4 INT8 RKNN.

The direct ``model.export(format='rknn')`` path exposes decoded predictions
(``[1, 4 + nc + nm, N]``).  The C++ runtime in this repository is optimized
for the faster raw4 interface instead:

    raw_head_s8, raw_head_s16, raw_head_s32: [1, 4*reg_max + nc + nm, H, W]
    proto:                                     [1, nm,  H_proto, W_proto]

This script exports the pre-decoder YOLO heads, appends mask coefficients to
each detection scale, stabilizes the DFL/class/coeff ranges for RKNN INT8
calibration, and builds the RKNN model for RK3588.

Run this on a conversion machine (Colab/x86 is fine), not necessarily on the
Orange Pi runtime.  The Orange Pi only needs the resulting .rknn file.

Example:

  python tools/convert_yolo11_seg_pt_to_rknn.py \
      --pt src/track_perception/model/best-5.pt \
      --calib /path/to/letterboxed_320_images \
      --output src/track_perception/model/best-5-raw4-int8-rk3588.rknn

The calibration directory should contain representative 320x320 RGB/BGR
images produced with the same crop/letterbox preprocessing used by the node.
For a full-frame dataset, use --crop-y0/--crop-y1 to generate such images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt",
        type=Path,
        default=Path("src/track_perception/model/best-5.pt"),
        help="Ultralytics YOLO11-seg checkpoint",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help="Calibration image directory or a text file with one image path per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/track_perception/model/best-5-raw4-int8-rk3588.rknn"),
        help="Output RKNN path",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Optional raw4 ONNX path; defaults next to --output",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--max-calib-images", type=int, default=256)
    parser.add_argument("--bbox-clip", type=float, default=16.0)
    parser.add_argument("--cls-clip", type=float, default=16.0)
    parser.add_argument("--coeff-clip", type=float, default=16.0)
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Build a float RKNN for debugging; not the performance target",
    )
    parser.add_argument(
        "--keep-calib-list",
        action="store_true",
        help="Keep the generated RKNN calibration list",
    )
    parser.add_argument(
        "--crop-y0",
        type=float,
        default=None,
        help="Optional source-image crop start ratio before 320x320 letterbox",
    )
    parser.add_argument(
        "--crop-y1",
        type=float,
        default=1.0,
        help="Optional source-image crop end ratio before 320x320 letterbox",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def import_conversion_dependencies():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        fail(
            "missing PT export dependency; install torch, ultralytics, numpy and "
            f"opencv-python before running conversion ({exc})"
        )

    return cv2, np, torch, nn, YOLO


def image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in {".txt", ".list"}:
        paths = []
        for line in source.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            paths.append(path)
        return paths
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    fail(f"calibration source does not exist or is not an image/list/directory: {source}")
    return []


def letterbox_images(
    paths: Sequence[Path],
    output_dir: Path,
    imgsz: int,
    crop_y0: float,
    crop_y1: float,
    cv2,
    np,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"warning: cannot read calibration image, skipping: {path}", file=sys.stderr)
            continue
        height, width = image.shape[:2]
        y0 = max(0, min(height - 1, round(height * crop_y0)))
        y1 = max(y0 + 1, min(height, round(height * crop_y1)))
        crop = image[y0:y1, :]
        scale = min(imgsz / max(1, crop.shape[1]), imgsz / max(1, crop.shape[0]))
        resize_w = max(1, min(imgsz, round(crop.shape[1] * scale)))
        resize_h = max(1, min(imgsz, round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_x = (imgsz - resize_w) // 2
        pad_y = (imgsz - resize_h) // 2
        canvas[pad_y : pad_y + resize_h, pad_x : pad_x + resize_w] = resized
        target = output_dir / f"calib_{index:05d}.jpg"
        if not cv2.imwrite(str(target), canvas):
            fail(f"failed to write calibration image: {target}")
        result.append(target)
    return result


def make_calibration_list(args: argparse.Namespace, cv2, np) -> tuple[Path | None, bool]:
    if args.no_quantization:
        return None, False
    if args.calib is None:
        fail("INT8 build requires --calib; provide representative images or a list file")

    paths = image_paths(args.calib)
    paths = [p for p in paths if p.exists()]
    if not paths:
        fail(f"no usable calibration images found in: {args.calib}")
    if args.max_calib_images > 0:
        paths = paths[: args.max_calib_images]

    generated = False
    if args.crop_y0 is not None:
        if not 0.0 <= args.crop_y0 < args.crop_y1 <= 1.0:
            fail("crop ratios must satisfy 0 <= crop-y0 < crop-y1 <= 1")
        calib_dir = args.output.parent / f"{args.output.stem}_calib_320"
        paths = letterbox_images(paths, calib_dir, args.imgsz, args.crop_y0, args.crop_y1, cv2, np)
        generated = True
    if not paths:
        fail("calibration preprocessing produced no images")

    list_path = args.output.parent / f"{args.output.stem}.calib.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")
    print(f"calibration images: {len(paths)}")
    print(f"calibration list:   {list_path}")
    return list_path, generated


def raw4_wrapper(torch, nn, model, bbox_clip: float, cls_clip: float, coeff_clip: float):
    class Raw4SegExportWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = model.model
            self.head = self.core.model[-1]
            if not hasattr(self.head, "nm") or not hasattr(self.head, "reg_max"):
                raise RuntimeError("checkpoint is not an Ultralytics segmentation model")
            self.head.export = False

        @staticmethod
        def _stabilize(head_tensor, reg_channels, nc, nm):
            batch = head_tensor.shape[0]
            height = head_tensor.shape[2]
            width = head_tensor.shape[3]
            bbox = head_tensor[:, :reg_channels].reshape(batch, 4, reg_channels // 4, height, width)
            bbox_max = torch.max(bbox, dim=2, keepdim=True).values
            bbox = torch.clamp(bbox - bbox_max, min=-bbox_clip, max=0.0)
            bbox = bbox.reshape(batch, reg_channels, height, width)
            cls = torch.clamp(
                head_tensor[:, reg_channels : reg_channels + nc],
                min=-cls_clip,
                max=cls_clip,
            )
            coeff = torch.clamp(
                head_tensor[:, reg_channels + nc : reg_channels + nc + nm],
                min=-coeff_clip,
                max=coeff_clip,
            )
            return torch.cat((bbox, cls, coeff), dim=1)

        def forward(self, images):
            x = images
            saved = []
            for layer in self.core.model:
                if layer.f != -1:
                    x = saved[layer.f] if isinstance(layer.f, int) else [
                        x if j == -1 else saved[j] for j in layer.f
                    ]
                x = layer(x)
                saved.append(x if layer.i in self.core.save else None)

            # Segment.forward(export=False) returns:
            # (decoded_predictions, (raw_heads, mask_coeffs, proto)).
            if not isinstance(x, tuple) or len(x) != 2:
                raise RuntimeError("unexpected Ultralytics Segment output")
            extras = x[1]
            if not isinstance(extras, tuple) or len(extras) != 3:
                raise RuntimeError("unexpected Ultralytics raw segmentation output")
            raw_heads, mask_coeffs, proto = extras
            if len(raw_heads) != 3:
                raise RuntimeError(f"expected 3 detection scales, got {len(raw_heads)}")

            nc = int(self.head.nc)
            nm = int(self.head.nm)
            reg_channels = 4 * int(self.head.reg_max)
            scale_sizes = [int(t.shape[2] * t.shape[3]) for t in raw_heads]
            coeff_scales = torch.split(mask_coeffs, scale_sizes, dim=2)
            outputs = []
            for raw, coeff in zip(raw_heads, coeff_scales):
                coeff = coeff.reshape(raw.shape[0], nm, raw.shape[2], raw.shape[3])
                combined = torch.cat((raw, coeff), dim=1)
                combined = self._stabilize(combined, reg_channels, nc, nm)
                outputs.append(combined)
            return outputs[0], outputs[1], outputs[2], proto

    return Raw4SegExportWrapper().eval()


def export_raw4_onnx(args: argparse.Namespace, torch, YOLO) -> Path:
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        fail(f"ONNX export requires onnx: {exc}")

    if not args.pt.is_file():
        fail(f"PT checkpoint not found: {args.pt}")
    onnx_path = args.onnx_output or args.output.with_suffix(".raw4.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(args.pt))
    model = yolo.model
    model.eval()
    wrapper = raw4_wrapper(torch, torch.nn, model, args.bbox_clip, args.cls_clip, args.coeff_clip)
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    print(f"exporting raw4 ONNX: {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["raw_head_s8", "raw_head_s16", "raw_head_s32", "proto"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    checked = onnx.load(str(onnx_path))
    onnx.checker.check_model(checked)
    print("ONNX outputs:")
    for output in checked.graph.output:
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")
    return onnx_path


def build_rknn(args: argparse.Namespace, onnx_path: Path, calib_list: Path | None) -> None:
    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError as exc:
        fail(f"RKNN conversion requires rknn-toolkit2: {exc}")

    rknn = RKNN(verbose=True)
    try:
        print(f"configuring RKNN for {args.target_platform}")
        ret = rknn.config(
            target_platform=args.target_platform,
            mean_values=[[0.0, 0.0, 0.0]],
            std_values=[[255.0, 255.0, 255.0]],
            quantized_dtype="asymmetric_quantized-8",
            optimization_level=3,
        )
        if ret != 0:
            fail(f"rknn.config failed: {ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            fail(f"rknn.load_onnx failed: {ret}")

        print("building RKNN; this can take several minutes")
        ret = rknn.build(
            do_quantization=not args.no_quantization,
            dataset=str(calib_list) if calib_list is not None else None,
        )
        if ret != 0:
            fail(f"rknn.build failed: {ret}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            fail(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()
    print(f"saved RKNN: {args.output}")


def main() -> None:
    args = parse_args()
    cv2, np, torch, nn, YOLO = import_conversion_dependencies()
    calib_list, generated_calib = make_calibration_list(args, cv2, np)
    onnx_path = export_raw4_onnx(args, torch, YOLO)
    build_rknn(args, onnx_path, calib_list)

    if calib_list is not None and not args.keep_calib_list:
        calib_list.unlink(missing_ok=True)
    if generated_calib:
        print("generated calibration images kept at:", calib_list.parent / f"{args.output.stem}_calib_320")


if __name__ == "__main__":
    main()
