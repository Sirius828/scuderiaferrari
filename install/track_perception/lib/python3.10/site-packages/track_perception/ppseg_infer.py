import cv2
import sys
import os
import glob
import argparse
import time
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rknnlite.api import RKNNLite
from rknnpool import rknnPoolExecutor
import cv2
import numpy as np

# 分割可视化颜色表 (RGB)。控制逻辑只把 class 1 当作赛道，
# 因此可视化也只绘制 class 1，避免其他类别被误画成赛道蓝色。
SEG_COLORS = np.zeros((256, 3), dtype=np.uint8)
SEG_COLORS[1] = [0, 128, 255]

# 默认语义分割输入尺寸；可由 ROS 参数覆盖以适配低分辨率 RKNN。
DEFAULT_IMG_SIZE = (640, 480)  # (width, height)
IMG_SIZE = DEFAULT_IMG_SIZE

def preprocess_image(
    img,
    input_format="RGB",
    model_input_format="RGB",
    input_size=DEFAULT_IMG_SIZE,
    crop_y0_ratio=0.0,
    crop_y1_ratio=1.0,
    pad_value=0,
    return_meta=False,
):
    """
    预处理图像：Resize, 颜色转换（如果需要）, 归一化
    Args:
        img: 输入图像，可以是RGB或BGR格式
        input_format: 输入图像格式，"RGB" 或 "BGR"
        model_input_format: 模型需要的输入格式，"RGB" 或 "BGR"
    Returns:
        input_data: 预处理后的数据 (1, H, W, C)
    """
    # ⭐ 关键优化：只在需要时才进行颜色转换
    if input_format != model_input_format:
        if input_format == "RGB" and model_input_format == "BGR":
            img_converted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif input_format == "BGR" and model_input_format == "RGB":
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_converted = img
    else:
        # 格式一致，不需要转换
        img_converted = img
    
    input_w, input_h = int(input_size[0]), int(input_size[1])
    orig_h, orig_w = img_converted.shape[:2]

    y0_ratio = max(0.0, min(1.0, float(crop_y0_ratio)))
    y1_ratio = max(y0_ratio + 1e-6, min(1.0, float(crop_y1_ratio)))
    crop_y0 = int(round(orig_h * y0_ratio))
    crop_y1 = int(round(orig_h * y1_ratio))
    crop_y0 = max(0, min(crop_y0, orig_h - 1))
    crop_y1 = max(crop_y0 + 1, min(crop_y1, orig_h))

    crop = img_converted[crop_y0:crop_y1, :]
    crop_h, crop_w = crop.shape[:2]
    scale = min(input_w / max(1.0, float(crop_w)), input_h / max(1.0, float(crop_h)))
    resize_w = max(1, min(input_w, int(round(crop_w * scale))))
    resize_h = max(1, min(input_h, int(round(crop_h * scale))))

    resized = cv2.resize(crop, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_w - resize_w) // 2
    pad_y = (input_h - resize_h) // 2
    canvas = np.full((input_h, input_w, 3), int(pad_value), dtype=img_converted.dtype)
    canvas[pad_y:pad_y + resize_h, pad_x:pad_x + resize_w] = resized

    img_normalized = np.ascontiguousarray(canvas)
    # 增加 batch 维度: (H, W, C) -> (1, H, W, C)
    input_data = np.expand_dims(img_normalized, axis=0)
    if not return_meta:
        return input_data

    meta = {
        'orig_w': orig_w,
        'orig_h': orig_h,
        'crop_y0': crop_y0,
        'crop_y1': crop_y1,
        'crop_w': crop_w,
        'crop_h': crop_h,
        'input_w': input_w,
        'input_h': input_h,
        'resize_w': resize_w,
        'resize_h': resize_h,
        'pad_x': pad_x,
        'pad_y': pad_y,
    }
    return input_data, meta

def restore_mask_to_original(mask, original_size, preprocess_meta=None):
    """Undo crop + letterbox preprocessing and return a full-frame class mask."""
    orig_h, orig_w = original_size
    mask = np.asarray(mask, dtype=np.uint8)

    if preprocess_meta is None:
        if mask.shape[0] == orig_h and mask.shape[1] == orig_w:
            return mask.astype(np.uint8)
        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    input_w = int(preprocess_meta['input_w'])
    input_h = int(preprocess_meta['input_h'])
    if mask.shape[1] != input_w or mask.shape[0] != input_h:
        mask = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

    pad_x = int(preprocess_meta['pad_x'])
    pad_y = int(preprocess_meta['pad_y'])
    resize_w = int(preprocess_meta['resize_w'])
    resize_h = int(preprocess_meta['resize_h'])
    crop_w = int(preprocess_meta['crop_w'])
    crop_h = int(preprocess_meta['crop_h'])
    crop_y0 = int(preprocess_meta['crop_y0'])
    crop_y1 = int(preprocess_meta['crop_y1'])

    unpadded = mask[pad_y:pad_y + resize_h, pad_x:pad_x + resize_w]
    crop_mask = cv2.resize(unpadded, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full_mask[crop_y0:crop_y1, :crop_w] = crop_mask[:crop_y1 - crop_y0, :orig_w]
    return full_mask

def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def split_yolov8_seg_outputs(outputs):
    """Return (proto, pred) for YOLOv8-seg style outputs, otherwise (None, None)."""
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        return None, None

    arrays = [np.asarray(output) for output in outputs]
    proto = None
    pred = None

    for array in arrays:
        if array.ndim == 4:
            candidate = array[0] if array.shape[0] == 1 else array
            if candidate.ndim == 3:
                if candidate.shape[0] <= 64:
                    proto = candidate
                elif candidate.shape[-1] <= 64:
                    proto = np.transpose(candidate, (2, 0, 1))
        elif array.ndim in (2, 3):
            pred = array

    if proto is None or pred is None:
        return None, None
    return proto.astype(np.float32, copy=False), pred.astype(np.float32, copy=False)

def normalize_yolov8_seg_predictions(pred):
    if pred.ndim == 3:
        if pred.shape[0] == 1:
            pred = pred[0]
        elif pred.shape[-1] == 1:
            pred = pred[:, :, 0]
    if pred.ndim != 2:
        return None

    # Exports may be (N, attrs) or (attrs, N).
    if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
        pred = pred.T
    return pred

def normalize_yolov8_split_tensor(output):
    """Normalize a split YOLOv8 head tensor to (num_anchors, channels)."""
    tensor = np.asarray(output)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        return None

    # RKNN split heads may be (N, C) or (C, N).
    if tensor.shape[0] <= 128 and tensor.shape[1] > tensor.shape[0]:
        tensor = tensor.T
    return tensor.astype(np.float32, copy=False)

def normalize_yolov8_proto(output):
    """Normalize a YOLOv8-seg proto tensor to (mask_dim, proto_h, proto_w)."""
    proto = np.asarray(output)
    if proto.ndim != 4:
        return None

    if proto.shape[0] == 1:
        proto = proto[0]
    if proto.ndim != 3:
        return None

    if proto.shape[0] <= 64:
        return proto.astype(np.float32, copy=False)
    if proto.shape[-1] <= 64:
        return np.transpose(proto, (2, 0, 1)).astype(np.float32, copy=False)
    return None

def split_yolov8_seg_head_outputs(outputs):
    """
    Return split YOLOv8-seg outputs as (proto, boxes, class_scores, coeffs).

    The split RKNN export avoids score quantization loss by emitting:
      proto=(1, mask_dim, proto_h, proto_w)
      boxes=(1, anchors, 4)
      scores=(1, anchors, classes)
      coeffs=(1, anchors, mask_dim)
    """
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 4:
        return None, None, None, None

    proto = None
    tensors = []
    for output in outputs:
        proto_candidate = normalize_yolov8_proto(output)
        if proto_candidate is not None:
            proto = proto_candidate
            continue

        tensor = normalize_yolov8_split_tensor(output)
        if tensor is not None:
            tensors.append(tensor)

    if proto is None:
        return None, None, None, None

    mask_dim = proto.shape[0]
    boxes = None
    scores = None
    coeffs = None

    for tensor in tensors:
        channels = tensor.shape[1]
        if channels == 4 and boxes is None:
            boxes = tensor
        elif channels == mask_dim and coeffs is None:
            coeffs = tensor
        elif scores is None:
            scores = tensor

    if boxes is None or scores is None or coeffs is None:
        return None, None, None, None
    if not (boxes.shape[0] == scores.shape[0] == coeffs.shape[0]):
        return None, None, None, None

    return (
        proto.astype(np.float32, copy=False),
        boxes.astype(np.float32, copy=False),
        scores.astype(np.float32, copy=False),
        coeffs.astype(np.float32, copy=False),
    )

def build_yolov8_seg_mask(
    proto,
    boxes,
    class_scores,
    coeffs,
    original_size,
    input_size=DEFAULT_IMG_SIZE,
    preprocess_meta=None,
    conf_threshold=0.25,
    mask_threshold=0.5,
    max_detections=30,
):
    """Build the class-1 semantic road mask from YOLOv8-seg parts."""
    orig_h, orig_w = original_size
    empty_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if proto is None or boxes is None or class_scores is None or coeffs is None:
        return None
    if boxes.size == 0 or class_scores.size == 0 or coeffs.size == 0:
        return empty_mask

    proto = proto.astype(np.float32, copy=False)
    boxes = boxes.astype(np.float32, copy=False)
    class_scores = class_scores.astype(np.float32, copy=False)
    coeffs = coeffs.astype(np.float32, copy=False)

    if class_scores.ndim == 1:
        class_scores = class_scores.reshape(-1, 1)
    if boxes.ndim != 2 or class_scores.ndim != 2 or coeffs.ndim != 2:
        return None
    if boxes.shape[0] != class_scores.shape[0] or boxes.shape[0] != coeffs.shape[0]:
        return None

    if np.max(class_scores) > 1.0 or np.min(class_scores) < 0.0:
        class_scores = sigmoid(class_scores)

    scores = np.max(class_scores, axis=1)
    keep = np.flatnonzero(scores >= float(conf_threshold))
    if keep.size == 0:
        return empty_mask

    max_detections = int(max_detections)
    if max_detections > 0 and keep.size > max_detections:
        local_scores = scores[keep]
        top_local = np.argpartition(local_scores, -max_detections)[-max_detections:]
        keep = keep[top_local]
    keep = keep[np.argsort(scores[keep])[::-1]]

    boxes = boxes_to_xyxy(boxes[keep, :4])
    coeffs = coeffs[keep, :proto.shape[0]]

    input_w, input_h = int(input_size[0]), int(input_size[1])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(input_w))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(input_h))

    mask_dim, proto_h, proto_w = proto.shape
    proto_flat = proto.reshape(mask_dim, -1)
    masks = sigmoid(np.matmul(coeffs, proto_flat)).reshape(-1, proto_h, proto_w)

    road_mask_low = np.zeros((proto_h, proto_w), dtype=bool)
    scale_x = proto_w / max(float(input_w), 1.0)
    scale_y = proto_h / max(float(input_h), 1.0)

    for mask, box in zip(masks, boxes):
        x0 = int(np.floor(box[0] * scale_x))
        y0 = int(np.floor(box[1] * scale_y))
        x1 = int(np.ceil(box[2] * scale_x))
        y1 = int(np.ceil(box[3] * scale_y))

        x0 = max(0, min(x0, proto_w - 1))
        y0 = max(0, min(y0, proto_h - 1))
        x1 = max(x0 + 1, min(x1, proto_w))
        y1 = max(y0 + 1, min(y1, proto_h))

        instance_mask = mask > float(mask_threshold)
        road_mask_low[y0:y1, x0:x1] |= instance_mask[y0:y1, x0:x1]

    if not np.any(road_mask_low):
        road_mask_low = np.any(masks > float(mask_threshold), axis=0)

    road_mask_model = cv2.resize(
        road_mask_low.astype(np.uint8),
        (input_w, input_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return restore_mask_to_original(road_mask_model, original_size, preprocess_meta)

def summarize_yolov8_seg_outputs(outputs, conf_threshold=0.25):
    summary = {}
    if outputs is None:
        summary['seg_output_shapes'] = 'None'
        return summary

    output_shapes = []
    for output in outputs:
        array = np.asarray(output)
        output_shapes.append(str(array.shape))
    summary['seg_output_shapes'] = ','.join(output_shapes)

    proto, boxes, scores_tensor, coeffs = split_yolov8_seg_head_outputs(outputs)
    if proto is not None:
        class_scores = scores_tensor.astype(np.float32, copy=False)
        if np.max(class_scores) > 1.0 or np.min(class_scores) < 0.0:
            class_scores = sigmoid(class_scores)
        scores = np.max(class_scores, axis=1)
        summary.update({
            'seg_post_format': 'split_yolov8_seg',
            'seg_pred_rows': int(boxes.shape[0]),
            'seg_pred_attrs': int(4 + scores_tensor.shape[1] + coeffs.shape[1]),
            'seg_mask_dim': int(proto.shape[0]),
            'seg_class_count': int(scores_tensor.shape[1]),
            'seg_nonbox_min': float(min(np.min(scores_tensor), np.min(coeffs))),
            'seg_nonbox_max': float(max(np.max(scores_tensor), np.max(coeffs))),
            'seg_score_min': float(np.min(scores)),
            'seg_score_max': float(np.max(scores)),
            'seg_score_p99': float(np.percentile(scores, 99)),
            'seg_score_keep': int(np.count_nonzero(scores >= float(conf_threshold))),
        })
        return summary

    proto, pred = split_yolov8_seg_outputs(outputs)
    if proto is None or pred is None:
        summary['seg_post_format'] = 'semantic_or_unknown'
        return summary

    pred = normalize_yolov8_seg_predictions(pred)
    if pred is None:
        summary['seg_post_format'] = 'yolov8_seg_invalid_pred'
        return summary

    mask_dim = int(proto.shape[0])
    attrs = int(pred.shape[1])
    class_count = attrs - 4 - mask_dim
    summary.update({
        'seg_post_format': 'yolov8_seg',
        'seg_pred_rows': int(pred.shape[0]),
        'seg_pred_attrs': attrs,
        'seg_mask_dim': mask_dim,
        'seg_class_count': int(class_count),
        'seg_nonbox_min': float(np.min(pred[:, 4:])) if attrs > 4 else 0.0,
        'seg_nonbox_max': float(np.max(pred[:, 4:])) if attrs > 4 else 0.0,
    })

    if class_count <= 0:
        summary['seg_score_max'] = 0.0
        summary['seg_score_keep'] = 0
        return summary

    class_scores = pred[:, 4:4 + class_count].astype(np.float32, copy=False)
    if class_scores.size == 0:
        summary['seg_score_max'] = 0.0
        summary['seg_score_keep'] = 0
        return summary

    if np.max(class_scores) > 1.0 or np.min(class_scores) < 0.0:
        class_scores = sigmoid(class_scores)

    scores = np.max(class_scores, axis=1)
    summary.update({
        'seg_score_min': float(np.min(scores)),
        'seg_score_max': float(np.max(scores)),
        'seg_score_p99': float(np.percentile(scores, 99)),
        'seg_score_keep': int(np.count_nonzero(scores >= float(conf_threshold))),
    })
    return summary

def boxes_to_xyxy(boxes):
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes

    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy_ratio = np.mean((x1 > x0) & (y1 > y0))
    if xyxy_ratio >= 0.5:
        return boxes

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes[:, 0] = cx - bw / 2.0
    boxes[:, 1] = cy - bh / 2.0
    boxes[:, 2] = cx + bw / 2.0
    boxes[:, 3] = cy + bh / 2.0
    return boxes

def postprocess_yolov8_segmentation(
    outputs,
    original_size,
    input_size=DEFAULT_IMG_SIZE,
    preprocess_meta=None,
    conf_threshold=0.25,
    mask_threshold=0.5,
    max_detections=30,
):
    """Convert YOLOv8-seg instance output into the semantic road mask expected downstream."""
    proto, boxes, class_scores, coeffs = split_yolov8_seg_head_outputs(outputs)
    if proto is not None:
        return build_yolov8_seg_mask(
            proto,
            boxes,
            class_scores,
            coeffs,
            original_size,
            input_size=input_size,
            preprocess_meta=preprocess_meta,
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
            max_detections=max_detections,
        )

    proto, pred = split_yolov8_seg_outputs(outputs)
    if proto is None or pred is None:
        return None

    pred = normalize_yolov8_seg_predictions(pred)
    if pred is None:
        return None

    mask_dim, proto_h, proto_w = proto.shape
    class_count = pred.shape[1] - 4 - mask_dim
    if class_count <= 0:
        return None

    class_scores = pred[:, 4:4 + class_count].astype(np.float32, copy=False)
    coeffs = pred[:, 4 + class_count:4 + class_count + mask_dim].astype(np.float32, copy=False)
    return build_yolov8_seg_mask(
        proto,
        pred[:, :4],
        class_scores,
        coeffs,
        original_size,
        input_size=input_size,
        preprocess_meta=preprocess_meta,
        conf_threshold=conf_threshold,
        mask_threshold=mask_threshold,
        max_detections=max_detections,
    )

def postprocess_segmentation(
    output,
    original_size,
    input_size=DEFAULT_IMG_SIZE,
    preprocess_meta=None,
    conf_threshold=0.25,
    mask_threshold=0.5,
    max_detections=30,
):
    """
    后处理分割结果
    Args:
        output: 模型输出，形状通常为 (1, num_classes, H, W) 或 (1, H, W)
        original_size: 原始图像尺寸 (height, width)
    Returns:
        seg_map: 分割掩码 (H, W)，值为类别索引
    """
    try:
        yolo_mask = postprocess_yolov8_segmentation(
            output,
            original_size,
            input_size=input_size,
            preprocess_meta=preprocess_meta,
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
            max_detections=max_detections,
        )
        if yolo_mask is not None:
            return yolo_mask
        if isinstance(output, (list, tuple)) and len(output) > 2:
            print(f"[ERROR] Unsupported multi-output segmentation shape: {[np.asarray(o).shape for o in output]}")
            return np.zeros(original_size, dtype=np.uint8)

        # 获取输出数据
        seg_output = output[0] if isinstance(output, list) else output
        
        if seg_output is None:
            print("[ERROR] seg_output is None")
            return None
     
        # 处理不同维度的输出，最终统一为 (H, W) class-id mask。
        if len(seg_output.shape) == 4:
            # (1, 1, H, W) 或 (1, C, H, W)
            batch = seg_output[0]
            if batch.shape[0] == 1:
                seg_map = batch[0]
            else:
                seg_map = np.argmax(batch, axis=0)
        elif len(seg_output.shape) == 3:
            # (1, H, W) 或 (num_classes, H, W) 或 (H, W, num_classes)
            if seg_output.shape[0] == 1:
                seg_map = seg_output[0]
            elif seg_output.shape[0] <= 32:
                # (C, H, W)
                seg_map = np.argmax(seg_output, axis=0)
            elif seg_output.shape[-1] <= 32:
                # (H, W, C)
                seg_map = np.argmax(seg_output, axis=-1)
            else:
                print(f"[ERROR] Ambiguous segmentation output shape: {seg_output.shape}")
                return None
        elif len(seg_output.shape) == 2:
            # (H, W)
            seg_map = seg_output
        else:
            print(f"[ERROR] Unexpected output shape: {seg_output.shape}")
            return None

        return restore_mask_to_original(seg_map, original_size, preprocess_meta)
    except Exception as e:
        print(f"[ERROR] Error in postprocess_segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None

def colorize_segmentation(seg_map):
    """
    将分割掩码转换为彩色图像
    Args:
        seg_map: 分割掩码 (H, W)，值为类别索引
    Returns:
        colored_seg: 彩色分割图像 (H, W, 3)
    """
    seg_map = np.asarray(seg_map, dtype=np.uint8)
    return SEG_COLORS[seg_map]

def convert_image_to_bgr(img, image_format):
    """Return an image in BGR order for OpenCV display/blending."""
    fmt = str(image_format).upper()
    if fmt == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def blend_images(original_img, colored_seg, alpha=0.5):
    """
    将原始图像和分割结果混合
    Args:
        original_img: 原始图像 (H, W, 3) BGR
        colored_seg: 彩色分割图像 (H, W, 3) RGB
        alpha: 混合比例
    Returns:
        blended: 混合后的图像 (H, W, 3) BGR
    """
    # 确保尺寸一致
    if original_img.shape[:2] != colored_seg.shape[:2]:
        colored_seg = cv2.resize(colored_seg, (original_img.shape[1], original_img.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # 转换 colored_seg 从 RGB 到 BGR
    colored_seg_bgr = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
    
    # 创建掩码：找出非黑色（有颜色）的区域
    # 如果像素不是纯黑 [0,0,0]，则认为是有分割颜色的区域
    mask = np.any(colored_seg_bgr > 0, axis=2).astype(np.float32)
    
    # 扩展掩码维度以匹配图像形状 (H, W) -> (H, W, 1)
    mask = mask[:, :, np.newaxis]
    
    # Alpha 混合
    blended_full = cv2.addWeighted(original_img, 1 - alpha, colored_seg_bgr, alpha, 0)
    
    # 只在有颜色的区域应用混合，黑色区域保持原图
    blended = original_img * (1 - mask) + blended_full * mask
    blended = blended.astype(np.uint8)
    
    return blended

def seg_visualization(seg_map, img_bgr=None, blend_alpha=None):
    """
    PP-Seg 推理函数
    Args:
        seg_map: 分割结果 (H, W)
        blend_alpha: 混合透明度 (0-1)，None 表示只显示分割结果，不混合
    Returns:
        result_img: 分割结果图像 (BGR 格式)
    """
    # 可视化：生成彩色分割图
    colored_seg = colorize_segmentation(seg_map)
    # print(f"[DEBUG] Colored segmentation size: {colored_seg.shape[1]}x{colored_seg.shape[0]} (WxH)")
    
    # 根据 blend_alpha 决定是否混合
    if blend_alpha is not None and 0 < blend_alpha < 1 and img_bgr is not None:
        # 与原图混合
        result_img = blend_images(img_bgr, colored_seg, alpha=blend_alpha)
    else:
        # 直接返回彩色分割图
        result_img = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
    return result_img
def myFunc(
    rknn_lite,
    img_bgr,
    blend_alpha=0.5,
    show_visualization=True,
    input_format="RGB",
    model_input_format="RGB",
    input_size=DEFAULT_IMG_SIZE,
    crop_y0_ratio=0.0,
    crop_y1_ratio=1.0,
    pad_value=0,
    conf_threshold=0.25,
    mask_threshold=0.5,
    max_detections=30,
):
    """
    PP-Seg 推理函数（用于 rknnpool）
    Args:
        rknn_lite: RKNNLite 实例
        img_bgr: 输入图像 (BGR 格式)
        blend_alpha: 混合透明度 (0-1)，None 表示只显示分割结果，不混合
        show_visualization: 是否生成可视化结果（False时只返回seg_map）
        input_format: 输入图像格式，"RGB" 或 "BGR"
        model_input_format: 模型需要的输入格式，"RGB" 或 "BGR"
    Returns:
        result_img: 分割结果图像 (BGR 格式)，如果show_visualization=False则为None
        seg_map: 原始分割掩码 (H, W)，值为类别索引
        flag: 成功标志
    """
    try:
        t_total_start = time.perf_counter()
        t_preprocess_start = t_total_start

        # 保存原始尺寸
        original_size = img_bgr.shape[:2]  # (height, width)
        
        # 预处理 - ⭐ 传递颜色格式参数
        input_data, preprocess_meta = preprocess_image(
            img_bgr,
            input_format=input_format,
            model_input_format=model_input_format,
            input_size=input_size,
            crop_y0_ratio=crop_y0_ratio,
            crop_y1_ratio=crop_y1_ratio,
            pad_value=pad_value,
            return_meta=True,
        )

        t_rknn_start = time.perf_counter()
        
        # 推理
        outputs = rknn_lite.inference(inputs=[input_data])

        t_post_start = time.perf_counter()
        
        # 检查推理是否成功
        if outputs is None or len(outputs) == 0:
            print("[ERROR] Model inference returned None or empty output")
            return None, None, False, {}
    
        post_summary = summarize_yolov8_seg_outputs(outputs, conf_threshold=conf_threshold)

        # 后处理 - 获取分割掩码
        seg_map = postprocess_segmentation(
            outputs,
            original_size,
            input_size=input_size,
            preprocess_meta=preprocess_meta,
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
            max_detections=max_detections,
        )

        t_vis_start = time.perf_counter()
        
        if seg_map is None:
            print("[ERROR] Postprocessing returned None")
            return None, None, False, {}
        
        # ⭐ 关键优化：只在需要时才生成可视化
        if show_visualization:
            # 可视化：生成彩色分割图
            colored_seg = colorize_segmentation(seg_map)
            
            # 根据 blend_alpha 决定是否混合
            if blend_alpha is not None and 0 < blend_alpha < 1:
                # 与原图混合
                display_base = convert_image_to_bgr(img_bgr, input_format)
                result_img = blend_images(display_base, colored_seg, alpha=blend_alpha)
            else:
                # 直接返回彩色分割图
                result_img = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
        else:
            # 不需要可视化，返回None
            result_img = None

        t_end = time.perf_counter()
        worker_preprocess_ms = (t_rknn_start - t_preprocess_start) * 1000.0
        worker_rknn_ms = (t_post_start - t_rknn_start) * 1000.0
        worker_postprocess_ms = (t_vis_start - t_post_start) * 1000.0
        worker_visualization_ms = (t_end - t_vis_start) * 1000.0
        image_to_mask_ms = worker_preprocess_ms + worker_rknn_ms + worker_postprocess_ms
        model_to_mask_ms = worker_rknn_ms + worker_postprocess_ms
        profile = {
            'worker_total_ms': (t_end - t_total_start) * 1000.0,
            'worker_preprocess_ms': worker_preprocess_ms,
            'worker_rknn_ms': worker_rknn_ms,
            'worker_postprocess_ms': worker_postprocess_ms,
            'worker_visualization_ms': worker_visualization_ms,
            'worker_image_to_mask_ms': image_to_mask_ms,
            'worker_image_to_mask_fps': 1000.0 / image_to_mask_ms if image_to_mask_ms > 0 else 0.0,
            'worker_model_to_mask_ms': model_to_mask_ms,
            'worker_model_to_mask_fps': 1000.0 / model_to_mask_ms if model_to_mask_ms > 0 else 0.0,
            'worker_show_visualization': show_visualization,
        }
        profile.update(post_summary)
        if seg_map is not None:
            profile['seg_mask_sum'] = int(np.count_nonzero(seg_map == 1))
            profile['seg_mask_shape'] = f'{seg_map.shape[1]}x{seg_map.shape[0]}'
        
        # 返回3个值：可视化结果（可能为None）、原始seg_map、成功标志
        return result_img, seg_map, True, profile
    
    except Exception as e:
        print(f"Error in myFunc: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False, {}


def get_current_dir():
    """获取当前脚本所在目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir


def resolve_model_dir(model_dir):
    """Resolve relative model directories from the installed package share path."""
    if os.path.isabs(model_dir):
        return model_dir

    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('track_perception'), model_dir)
    except Exception:
        return os.path.join(os.path.dirname(get_current_dir()), model_dir)


class PPSegInfer:
    """PP-Seg 推理封装类"""
    
    def __init__(
        self,
        model_dir="model",
        model_filename=None,
        TPEs=1,
        blend_alpha=None,
        show_visualization=True,
        input_format="RGB",
        model_input_format="RGB",
        core_ids=None,
        input_size=DEFAULT_IMG_SIZE,
        crop_y0_ratio=0.0,
        crop_y1_ratio=1.0,
        pad_value=0,
        conf_threshold=0.25,
        mask_threshold=0.5,
        max_detections=30,
    ):
        """
        初始化 PP-Seg 推理器
        Args:
            model_dir: 模型目录（相对于当前脚本）
            model_filename: 模型文件名（例如 "pp_liteseg_v2.rknn"），None则自动查找
            TPEs: 线程池执行器数量
            blend_alpha: 混合透明度 (0-1)，None 表示只显示分割结果
            show_visualization: 是否生成可视化结果
            input_format: 输入图像格式，"RGB" 或 "BGR"
            model_input_format: 模型需要的输入格式，"RGB" 或 "BGR"
            core_ids: NPU 核心 ID 列表，例如 [2]；None 表示按 0/1/2 轮询
        """
        model_dir = resolve_model_dir(model_dir)
        model_path = self.get_model_path(model_dir, model_filename)
        
        self.TPEs = TPEs
        self.blend_alpha = blend_alpha
        self.show_visualization = show_visualization
        self.input_format = input_format
        self.model_input_format = model_input_format
        self.core_ids = [int(core_id) for core_id in core_ids] if core_ids else None
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.crop_y0_ratio = float(crop_y0_ratio)
        self.crop_y1_ratio = float(crop_y1_ratio)
        self.pad_value = int(pad_value)
        self.conf_threshold = float(conf_threshold)
        self.mask_threshold = float(mask_threshold)
        self.max_detections = int(max_detections)
        self.last_profile = {}
        
        # 创建带有参数的推理函数
        from functools import partial
        infer_func = partial(myFunc, blend_alpha=blend_alpha, show_visualization=show_visualization, 
                            input_format=input_format, model_input_format=model_input_format,
                            input_size=self.input_size, crop_y0_ratio=self.crop_y0_ratio,
                            crop_y1_ratio=self.crop_y1_ratio, pad_value=self.pad_value,
                            conf_threshold=self.conf_threshold,
                            mask_threshold=self.mask_threshold, max_detections=self.max_detections)
        
        self.rknn_pool = rknnPoolExecutor(
            rknnModel=model_path,
            TPEs=self.TPEs,
            func=infer_func,
            core_ids=self.core_ids
        )
        self.pool_flag = False
        
        mode = "blended" if blend_alpha is not None else "segmentation_only"
        vis_mode = "with_visualization" if show_visualization else "mask_only"
        print(f"PP-Seg model loaded: {model_path}")
        print(f"Thread pool size: {TPEs}")
        print(f"NPU Core IDs: {self.core_ids if self.core_ids else 'auto 0/1/2'}")
        print(f"Output mode: {mode} (blend_alpha={blend_alpha})")
        print(f"Visualization: {vis_mode}")
        print(f"Input size: {self.input_size[0]}x{self.input_size[1]}")
        print(f"Crop Y ratio: {self.crop_y0_ratio:.3f}-{self.crop_y1_ratio:.3f}, pad={self.pad_value}")
        print(
            f"YOLOv8-seg thresholds: conf={self.conf_threshold}, "
            f"mask={self.mask_threshold}, max_det={self.max_detections}"
        )

    def get_model_path(self, model_dir, model_filename=None):
        """获取模型路径 - 支持指定模型文件名"""
        # ⭐ 如果指定了模型文件名，优先使用
        if model_filename:
            seg_model = os.path.join(model_dir, model_filename)
            if os.path.exists(seg_model):
                print(f"Using semantic segmentation model: {seg_model}")
                return seg_model
            else:
                raise FileNotFoundError(f"Specified model not found: {seg_model}")
        
        # 否则使用默认查找逻辑
        seg_model = os.path.join(model_dir, "pp_liteseg.rknn")
        if os.path.exists(seg_model):
            print(f"Using semantic segmentation model: {seg_model}")
            return seg_model
        
        # 如果没有找到 pp_liteseg.rknn，尝试其他 .rknn 文件
        model_files = glob.glob(os.path.join(model_dir, "*.rknn"))
        if not model_files:
            raise FileNotFoundError(f"No .rknn model found in {model_dir}")
        model_path = model_files[0]
        print(f"WARNING: Using fallback model: {model_path}")
        return model_path

    def pool_init(self, img):
        """初始化线程池，预加载数据"""
        for i in range(self.TPEs + 1):
            self.rknn_pool.put(img)

    def infer(self, img):
        """
        执行推理
        Args:
            img: 输入图像 (BGR 格式，numpy array)
        Returns:
            result_img: 带有分割结果的图像 (numpy array)
            seg_map: 原始分割掩码 (H, W)，值为类别索引 (0=背景, 1=赛道)
            binary_mask: 二值化mask (H, W)，赛道=255, 背景=0
            flag: 推理是否成功 (bool)
        """
        if not self.pool_flag:
            self.pool_init(img)
            self.pool_flag = True
        
        self.rknn_pool.put(img)
        # rknnpool.get() 返回 (func_result, success_flag)
        # 而 func_result 是 myFunc 返回的 (result_img, seg_map, flag)
        func_result, pool_success = self.rknn_pool.get()
        
        if pool_success and func_result is not None:
            # 解包 myFunc 的返回值
            if isinstance(func_result, tuple) and len(func_result) == 4:
                result_img, seg_map, myfunc_flag, profile = func_result
                self.last_profile = profile if isinstance(profile, dict) else {}
                return result_img, seg_map, myfunc_flag
            elif isinstance(func_result, tuple) and len(func_result) == 3:
                result_img, seg_map, myfunc_flag = func_result
                self.last_profile = {}
                return result_img, seg_map, myfunc_flag
            else:
                self.last_profile = {}
                return func_result, None, True
        else:
            return None, None, False
    
    def __call__(self, *args, **kwargs):
        """使实例可调用"""
        return self.infer(*args, **kwargs)

    def get_last_profile(self):
        """返回最近一次完成推理任务的 worker 内部分段耗时。"""
        return self.last_profile
    
    def release(self):
        """释放资源"""
        self.rknn_pool.release()
        print("PP-Seg resources released")


def main():
    parser = argparse.ArgumentParser(description='PP-Seg Inference Demo')
    parser.add_argument('--model_dir', type=str, default='model', 
                        help='Directory containing the .rknn model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_path', type=str, default='./result.png',
                        help='Path to save result image')
    parser.add_argument('--TPEs', type=int, default=1,
                        help='Number of thread pool executors')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark test')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of iterations for benchmark')
    parser.add_argument('--blend', type=float, default=None,
                        help='Blend alpha (0-1) with original image. None means segmentation mask only')
    
    args = parser.parse_args()
    
    # 检查输入图像
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return -1
    
    # 读取图像
    img = cv2.imread(args.image_path)
    # print(img, img.size())
    if img is None:
        print(f"Error: Failed to read image: {args.image_path}")
        return -1
    
    print(f"Input image size: {img.shape[1]}x{img.shape[0]}", img.shape)
    
    # 初始化推理器
    try:
        infer = PPSegInfer(model_dir=args.model_dir, TPEs=args.TPEs, blend_alpha=args.blend)
    except Exception as e:
        print(f"Error initializing PPSegInfer: {e}")
        return -1
    
    # 推理
    if args.benchmark:
        # 性能测试
        print(f"\nRunning benchmark with {args.num_iterations} iterations...")
        
        # 预热
        for _ in range(10):
            result, flag = infer.infer(img)
        
        # 计时
        start_time = time.time()
        for i in range(args.num_iterations):
            seg_map, flag = infer.infer(img)
            if not flag:
                print(f"Iteration {i+1} failed!")
                break
        end_time = time.time()
        
        img_visual = seg_visualization(seg_map)
        elapsed_time = end_time - start_time
        fps = args.num_iterations / elapsed_time
        avg_time = (elapsed_time / args.num_iterations) * 1000
        
        print(f"\nBenchmark Results:")
        print(f"  Total time: {elapsed_time:.2f}s")
        print(f"  Average time: {avg_time:.2f}ms per frame")
        print(f"  FPS: {fps:.2f}")
        
        # 保存最后一次结果
        if flag:
            cv2.imwrite(args.output_path, img_visual)
            print(f"\nResult saved to: {args.output_path}")
    else:
        # 单次推理
        print("\nRunning single inference...")
        start_time = time.time()
        result, flag = infer.infer(img)
        end_time = time.time()
        
        if flag:
            elapsed_time = (end_time - start_time) * 1000
            print(f"Inference completed in {elapsed_time:.2f}ms")
            print(f"Result type: {type(result)}")
            print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            print(f"Flag value: {flag}")
            
            # 确保 result 是 numpy 数组
            if not isinstance(result, np.ndarray):
                print(f"Error: Result is not a numpy array, got {type(result)}")
                infer.release()
                return -1
            img_visual = seg_visualization(result)
            # 保存结果
            cv2.imwrite(args.output_path, img_visual)
            print(f"Result saved to: {args.output_path}")
        else:
            print("Inference failed!")
            infer.release()
            return -1
    
    # 释放资源
    infer.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
