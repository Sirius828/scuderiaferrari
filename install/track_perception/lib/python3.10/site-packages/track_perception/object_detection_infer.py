#!/usr/bin/env python3
"""
RKNN 目标检测推理封装（异步线程池版本）
基于官方 setupUI 的 InferWrap 重构，支持多 NPU 核心并行推理
"""

import os
import sys
import glob
import time
import traceback
from functools import partial
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ⭐ 导入 RKNN 线程池执行器
from .rknn_pool import RKNNPoolExecutor


# 目标检测参数
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
POST_TOPK_PER_CLASS = 50
POST_KEEP_TOPK = 30
ENABLE_CLASS_AGNOSTIC_NMS = True
CLASS_AGNOSTIC_NMS_THRESH = 0.60
ENABLE_FAST_POSTPROCESS = True
DUPLICATE_IOU_THRESH = 0.30
DUPLICATE_CONTAINMENT_THRESH = 0.75

_DFL_PROJ_CACHE = {}


def letterbox_image(img, new_shape=IMG_SIZE, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio, then pad on right/bottom.

    PaddleYOLO training letterbox keeps the resized image at the top-left corner
    and pads the remaining area on the right/bottom side.
    """
    src_h, src_w = img.shape[:2]
    dst_w, dst_h = new_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((dst_h, dst_w, 3), color, dtype=img.dtype)
    pad_x = 0
    pad_y = 0
    padded[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    return padded, scale, pad_x, pad_y


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def as_xyxy(x):
    """PaddleYOLO exported YOLOv8 output is already decoded as [x1, y1, x2, y2]."""
    return x.astype(np.float32, copy=True)


def dfl(position):
    """分布焦点损失(DFL)解码（纯NumPy实现）"""
    n, c, h, w = position.shape
    p_num = 4  # x,y,w,h四个参数
    mc = c // p_num  # 每个参数的分布数

    # 重塑为 (n, p_num, mc, h, w)
    y = position.reshape(n, p_num, mc, h, w)
    
    # NumPy实现softmax
    exp_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # 数值稳定版softmax
    y_softmax = exp_y / np.sum(exp_y, axis=2, keepdims=True)
    
    # 计算分布加权和
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y_softmax * acc_metrix, axis=2)
    return y


def box_process(position, size_im=IMG_SIZE):
    """边界框解码（纯NumPy实现）"""
    grid_h, grid_w = position.shape[2:4]
    # 生成网格坐标
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    grid = np.concatenate((col, row), axis=1)
    
    # 计算步长
    stride = np.array([size_im[1] // grid_h, size_im[0] // grid_w], dtype=np.float32).reshape(1, 2, 1, 1)

    # DFL解码
    position = dfl(position)
    
    # 计算边界框坐标
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def filter_boxes(boxes, box_confidences, box_class_probs):
    """过滤低置信度框（纯NumPy实现）"""
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores, nms_thresh=NMS_THRESH):
    """NMS算法（纯NumPy实现）"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def _intersection_with_one(box, boxes):
    """Return intersection area between one box and a set of boxes."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    return w * h


def suppress_duplicate_boxes(boxes, classes, scores):
    """
    Final same-class duplicate guard after NMS.

    NMS handles high-IoU boxes. Some RKNN/YOLO outputs still leave nested boxes
    around the same object, especially large signs/gates. This keeps the highest
    confidence box when same-class boxes either overlap enough or one mostly
    contains the other. Different classes are never compared.
    """
    if boxes is None or len(boxes) <= 1:
        return boxes, classes, scores, 0

    final_indices = []
    removed = 0

    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        class_boxes = boxes[class_indices].astype(np.float32, copy=False)
        class_scores = scores[class_indices]
        order = np.argsort(class_scores)[::-1]
        kept_local = []

        areas = np.maximum(0.0, class_boxes[:, 2] - class_boxes[:, 0]) * \
                np.maximum(0.0, class_boxes[:, 3] - class_boxes[:, 1])

        for local_idx in order:
            if not kept_local:
                kept_local.append(local_idx)
                continue

            kept_boxes = class_boxes[np.asarray(kept_local, dtype=np.int64)]
            kept_areas = areas[np.asarray(kept_local, dtype=np.int64)]
            cur_box = class_boxes[local_idx]
            cur_area = max(float(areas[local_idx]), 1e-6)

            inter = _intersection_with_one(cur_box, kept_boxes)
            union = cur_area + kept_areas - inter
            iou = inter / np.maximum(union, 1e-6)
            containment = inter / np.maximum(np.minimum(cur_area, kept_areas), 1e-6)

            duplicate = (
                np.any(iou > DUPLICATE_IOU_THRESH) or
                np.any(containment > DUPLICATE_CONTAINMENT_THRESH)
            )
            if duplicate:
                removed += 1
            else:
                kept_local.append(local_idx)

        final_indices.append(class_indices[np.asarray(kept_local, dtype=np.int64)])

    if not final_indices:
        return None, None, None, removed

    final_indices = np.concatenate(final_indices)
    return boxes[final_indices], classes[final_indices], scores[final_indices], removed


def _get_dfl_projection(mc):
    """缓存 DFL projection vector，避免每帧重复创建。"""
    if mc not in _DFL_PROJ_CACHE:
        _DFL_PROJ_CACHE[mc] = np.arange(mc, dtype=np.float32).reshape(1, mc)
    return _DFL_PROJ_CACHE[mc]


def _decode_selected_boxes(position, anchor_indices, img_shape):
    """
    只对选中的 anchor 做 DFL box decode。

    Args:
        position: RKNN box 输出，形状 (1, C, H, W)
        anchor_indices: flatten 后的 anchor 下标，顺序为 row-major H*W
        img_shape: 原始图像尺寸 (h, w)
    """
    if len(anchor_indices) == 0:
        return np.empty((0, 4), dtype=np.float32)

    _, channels, grid_h, grid_w = position.shape
    p_num = 4
    mc = channels // p_num

    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    rows = anchor_indices // grid_w
    cols = anchor_indices % grid_w

    selected = position[0, :, rows, cols].T.reshape(-1, p_num, mc).astype(np.float32, copy=False)
    selected = selected - np.max(selected, axis=2, keepdims=True)
    exp_selected = np.exp(selected)
    prob = exp_selected / np.sum(exp_selected, axis=2, keepdims=True)
    distances = np.sum(prob * _get_dfl_projection(mc), axis=2)

    stride_x = img_shape[1] // grid_w
    stride_y = img_shape[0] // grid_h

    x1 = (cols.astype(np.float32) + 0.5 - distances[:, 0]) * stride_x
    y1 = (rows.astype(np.float32) + 0.5 - distances[:, 1]) * stride_y
    x2 = (cols.astype(np.float32) + 0.5 + distances[:, 2]) * stride_x
    y2 = (rows.astype(np.float32) + 0.5 + distances[:, 3]) * stride_y

    return np.stack((x1, y1, x2, y2), axis=1)


def fast_post_process(input_data, img_shape=(640, 640)):
    """
    快速后处理：先筛选候选 anchor，再只对候选做 DFL decode。
    保留所有类别；per-class top-k 只限制每类进入 NMS 的候选数量。
    """
    t_start = time.perf_counter()
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    candidate_records = []
    candidates_before_filter = 0

    # score filter: 只根据 class confidence 先筛 anchor，不做全量 DFL。
    for i in range(default_branch):
        class_output = input_data[pair_per_branch * i + 1]
        _, class_count, grid_h, grid_w = class_output.shape
        class_probs = class_output.transpose(0, 2, 3, 1).reshape(-1, class_count)
        candidates_before_filter += class_probs.shape[0]

        class_max_score = np.max(class_probs, axis=1)
        classes = np.argmax(class_probs, axis=1)
        keep = np.flatnonzero(class_max_score >= OBJ_THRESH)
        if keep.size == 0:
            continue

        candidate_records.append({
            'branch': i,
            'anchor_indices': keep.astype(np.int64, copy=False),
            'classes': classes[keep].astype(np.int32, copy=False),
            'scores': class_max_score[keep].astype(np.float32, copy=False),
        })

    t_filter_done = time.perf_counter()

    if not candidate_records:
        profile = {
            'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
            'post_topk_ms': 0.0,
            'post_dfl_decode_ms': 0.0,
            'post_nms_ms': 0.0,
            'post_dedupe_ms': 0.0,
            'post_duplicates_removed': 0,
            'post_candidates_before_filter': candidates_before_filter,
            'post_candidates_after_filter': 0,
            'post_candidates_after_topk': 0,
            'post_nms_input_max_per_class': 0,
            'post_fast_path': True,
        }
        return None, None, None, profile

    branch_ids = np.concatenate([
        np.full(record['anchor_indices'].shape, record['branch'], dtype=np.int32)
        for record in candidate_records
    ])
    anchor_indices = np.concatenate([record['anchor_indices'] for record in candidate_records])
    classes = np.concatenate([record['classes'] for record in candidate_records])
    scores = np.concatenate([record['scores'] for record in candidate_records])
    candidates_after_filter = int(scores.size)

    # per-class top-k，保留全部类别，但限制每类进入 DFL/NMS 的候选数量。
    selected_global_indices = []
    nms_input_max_per_class = 0
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        if class_indices.size > POST_TOPK_PER_CLASS:
            local_scores = scores[class_indices]
            top_local = np.argpartition(local_scores, -POST_TOPK_PER_CLASS)[-POST_TOPK_PER_CLASS:]
            class_indices = class_indices[top_local]
        nms_input_max_per_class = max(nms_input_max_per_class, int(class_indices.size))
        selected_global_indices.append(class_indices)

    selected_global_indices = np.concatenate(selected_global_indices)
    branch_ids = branch_ids[selected_global_indices]
    anchor_indices = anchor_indices[selected_global_indices]
    classes = classes[selected_global_indices]
    scores = scores[selected_global_indices]
    candidates_after_topk = int(scores.size)

    t_topk_done = time.perf_counter()

    decoded_boxes = []
    decoded_classes = []
    decoded_scores = []
    for branch in np.unique(branch_ids):
        branch_mask = branch_ids == branch
        branch_anchor_indices = anchor_indices[branch_mask]
        branch_boxes = _decode_selected_boxes(
            input_data[pair_per_branch * int(branch)],
            branch_anchor_indices,
            img_shape
        )
        decoded_boxes.append(branch_boxes)
        decoded_classes.append(classes[branch_mask])
        decoded_scores.append(scores[branch_mask])

    boxes = np.concatenate(decoded_boxes) if decoded_boxes else np.empty((0, 4), dtype=np.float32)
    classes = np.concatenate(decoded_classes) if decoded_classes else np.empty((0,), dtype=np.int32)
    scores = np.concatenate(decoded_scores) if decoded_scores else np.empty((0,), dtype=np.float32)

    t_decode_done = time.perf_counter()

    if boxes.size == 0:
        profile = {
            'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
            'post_topk_ms': (t_topk_done - t_filter_done) * 1000.0,
            'post_dfl_decode_ms': (t_decode_done - t_topk_done) * 1000.0,
            'post_nms_ms': 0.0,
            'post_dedupe_ms': 0.0,
            'post_duplicates_removed': 0,
            'post_candidates_before_filter': candidates_before_filter,
            'post_candidates_after_filter': candidates_after_filter,
            'post_candidates_after_topk': candidates_after_topk,
            'post_nms_input_max_per_class': nms_input_max_per_class,
            'post_fast_path': True,
        }
        return None, None, None, profile

    nboxes, nclasses, nscores = [], [], []
    for class_id in np.unique(classes):
        inds = np.where(classes == class_id)
        b = boxes[inds]
        c_cls = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_cls[keep])
            nscores.append(s[keep])

    t_nms_done = time.perf_counter()

    if not nboxes:
        profile = {
            'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
            'post_topk_ms': (t_topk_done - t_filter_done) * 1000.0,
            'post_dfl_decode_ms': (t_decode_done - t_topk_done) * 1000.0,
            'post_nms_ms': (t_nms_done - t_decode_done) * 1000.0,
            'post_dedupe_ms': 0.0,
            'post_duplicates_removed': 0,
            'post_candidates_before_filter': candidates_before_filter,
            'post_candidates_after_filter': candidates_after_filter,
            'post_candidates_after_topk': candidates_after_topk,
            'post_nms_input_max_per_class': nms_input_max_per_class,
            'post_fast_path': True,
        }
        return None, None, None, profile

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    boxes, classes, scores, duplicates_removed = suppress_duplicate_boxes(boxes, classes, scores)
    t_dedupe_done = time.perf_counter()

    profile = {
        'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
        'post_topk_ms': (t_topk_done - t_filter_done) * 1000.0,
        'post_dfl_decode_ms': (t_decode_done - t_topk_done) * 1000.0,
        'post_nms_ms': (t_nms_done - t_decode_done) * 1000.0,
        'post_dedupe_ms': (t_dedupe_done - t_nms_done) * 1000.0,
        'post_duplicates_removed': duplicates_removed,
        'post_candidates_before_filter': candidates_before_filter,
        'post_candidates_after_filter': candidates_after_filter,
        'post_candidates_after_topk': candidates_after_topk,
        'post_nms_input_max_per_class': nms_input_max_per_class,
        'post_fast_path': True,
    }

    if boxes is None or len(boxes) == 0:
        return None, None, None, profile

    return boxes, classes, scores, profile


def _as_yolov8_flat_output(output):
    """Normalize a single YOLOv8 flat output to (num_anchors, 4 + num_classes)."""
    output = np.asarray(output)
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise ValueError(f'flat YOLO output batch must be 1, got shape={output.shape}')
        output = output[0]
    elif output.ndim != 2:
        raise ValueError(f'flat YOLO output must be 2D/3D, got shape={output.shape}')

    # RKNN exports may be (8400, 17) or (17, 8400).
    if output.shape[0] <= 256 and output.shape[1] > output.shape[0]:
        output = output.T

    if output.shape[1] < 6:
        raise ValueError(f'flat YOLO output attrs must be >= 6, got shape={output.shape}')
    return output.astype(np.float32, copy=False)


def flat_yolov8_post_process(input_data, img_shape=(640, 640), letterbox_meta=None):
    """
    后处理单输出 YOLOv8 RKNN: (1, 8400, 4 + num_classes).

    PaddleYOLO 导出的 flat 输出 bbox 已经是 xyxy 坐标，坐标位于 640x640
    letterbox 输入空间，再映射回当前画面坐标。
    """
    t_start = time.perf_counter()
    if input_data is None or len(input_data) != 1:
        raise ValueError(f'flat YOLOv8 postprocess expects 1 output, got {0 if input_data is None else len(input_data)}')

    output = _as_yolov8_flat_output(input_data[0])
    candidates_before_filter = int(output.shape[0])

    boxes_xyxy = output[:, :4]
    class_probs = output[:, 4:]
    raw_score_min = float(np.min(class_probs)) if class_probs.size else 0.0
    raw_score_max = float(np.max(class_probs)) if class_probs.size else 0.0

    # 有些导出会保留 logits；当前 RKNN 通常已是 0..1 概率。这里做保护性 sigmoid。
    if class_probs.size > 0 and (np.max(class_probs) > 1.0 or np.min(class_probs) < 0.0):
        class_probs = 1.0 / (1.0 + np.exp(-class_probs))

    anchor_max_scores = np.max(class_probs, axis=1)
    score_max = float(np.max(anchor_max_scores)) if anchor_max_scores.size else 0.0

    # PaddleYOLO multiclass_nms 风格：每个类别独立筛选候选，而不是每个
    # anchor 只取 argmax 类别。模型导出时去掉 NMS 后需要在这里补回来。
    filtered_per_class = []
    candidates_after_filter = 0
    for class_id in range(class_probs.shape[1]):
        class_scores = class_probs[:, class_id]
        keep = np.flatnonzero(class_scores >= OBJ_THRESH)
        if keep.size == 0:
            continue
        candidates_after_filter += int(keep.size)
        filtered_per_class.append((class_id, keep, class_scores[keep].astype(np.float32, copy=False)))
    t_filter_done = time.perf_counter()

    if not filtered_per_class:
        profile = {
            'post_format': 'flat_yolov8',
            'post_multiclass_nms': True,
            'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
            'post_topk_ms': 0.0,
            'post_dfl_decode_ms': 0.0,
            'post_nms_ms': 0.0,
            'post_dedupe_ms': 0.0,
            'post_agnostic_removed': 0,
            'post_keep_topk_removed': 0,
            'post_duplicates_removed': 0,
            'post_candidates_before_filter': candidates_before_filter,
            'post_candidates_after_filter': 0,
            'post_candidates_after_topk': 0,
            'post_nms_input_max_per_class': 0,
            'post_flat_raw_score_min': raw_score_min,
            'post_flat_raw_score_max': raw_score_max,
            'post_flat_score_max': score_max,
            'post_fast_path': False,
        }
        return None, None, None, profile

    selected_anchor_indices = []
    selected_classes = []
    selected_scores = []
    nms_input_max_per_class = 0
    for class_id, anchor_indices, class_scores in filtered_per_class:
        if anchor_indices.size > POST_TOPK_PER_CLASS:
            top_local = np.argpartition(class_scores, -POST_TOPK_PER_CLASS)[-POST_TOPK_PER_CLASS:]
            anchor_indices = anchor_indices[top_local]
            class_scores = class_scores[top_local]
        nms_input_max_per_class = max(nms_input_max_per_class, int(anchor_indices.size))
        selected_anchor_indices.append(anchor_indices)
        selected_classes.append(np.full(anchor_indices.shape, class_id, dtype=np.int32))
        selected_scores.append(class_scores)

    selected_anchor_indices = np.concatenate(selected_anchor_indices)
    classes = np.concatenate(selected_classes)
    scores = np.concatenate(selected_scores).astype(np.float32, copy=False)
    candidates_after_topk = int(scores.size)
    t_topk_done = time.perf_counter()

    boxes = as_xyxy(boxes_xyxy[selected_anchor_indices])
    if letterbox_meta is not None:
        scale, pad_x, pad_y = letterbox_meta
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale, 1e-6)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale, 1e-6)
    else:
        scale_x = float(img_shape[1]) / float(IMG_SIZE[0])
        scale_y = float(img_shape[0]) / float(IMG_SIZE[1])
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    t_decode_done = time.perf_counter()

    nboxes, nclasses, nscores = [], [], []
    for class_id in np.unique(classes):
        inds = np.flatnonzero(classes == class_id)
        b = boxes[inds]
        c_cls = classes[inds]
        s = scores[inds]
        keep_nms = nms_boxes(b, s)
        if len(keep_nms) != 0:
            nboxes.append(b[keep_nms])
            nclasses.append(c_cls[keep_nms])
            nscores.append(s[keep_nms])

    t_nms_done = time.perf_counter()

    if not nboxes:
        profile = {
            'post_format': 'flat_yolov8',
            'post_multiclass_nms': True,
            'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
            'post_topk_ms': (t_topk_done - t_filter_done) * 1000.0,
            'post_dfl_decode_ms': (t_decode_done - t_topk_done) * 1000.0,
            'post_nms_ms': (t_nms_done - t_decode_done) * 1000.0,
            'post_dedupe_ms': 0.0,
            'post_agnostic_removed': 0,
            'post_keep_topk_removed': 0,
            'post_duplicates_removed': 0,
            'post_candidates_before_filter': candidates_before_filter,
            'post_candidates_after_filter': candidates_after_filter,
            'post_candidates_after_topk': candidates_after_topk,
            'post_nms_input_max_per_class': nms_input_max_per_class,
            'post_flat_raw_score_min': raw_score_min,
            'post_flat_raw_score_max': raw_score_max,
            'post_flat_score_max': score_max,
            'post_fast_path': False,
        }
        return None, None, None, profile

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    agnostic_removed = 0
    if ENABLE_CLASS_AGNOSTIC_NMS and len(boxes) > 1:
        keep_agnostic = nms_boxes(boxes, scores, nms_thresh=CLASS_AGNOSTIC_NMS_THRESH)
        agnostic_removed = int(len(boxes) - len(keep_agnostic))
        boxes = boxes[keep_agnostic]
        classes = classes[keep_agnostic]
        scores = scores[keep_agnostic]

    boxes, classes, scores, duplicates_removed = suppress_duplicate_boxes(boxes, classes, scores)
    keep_topk_removed = 0
    if POST_KEEP_TOPK > 0 and len(scores) > POST_KEEP_TOPK:
        order = np.argsort(scores)[::-1][:POST_KEEP_TOPK]
        keep_topk_removed = int(len(scores) - len(order))
        boxes = boxes[order]
        classes = classes[order]
        scores = scores[order]
    t_dedupe_done = time.perf_counter()

    profile = {
        'post_format': 'flat_yolov8',
        'post_multiclass_nms': True,
        'post_score_filter_ms': (t_filter_done - t_start) * 1000.0,
        'post_topk_ms': (t_topk_done - t_filter_done) * 1000.0,
        'post_dfl_decode_ms': (t_decode_done - t_topk_done) * 1000.0,
        'post_nms_ms': (t_nms_done - t_decode_done) * 1000.0,
        'post_dedupe_ms': (t_dedupe_done - t_nms_done) * 1000.0,
        'post_agnostic_removed': agnostic_removed,
        'post_keep_topk_removed': keep_topk_removed,
        'post_duplicates_removed': duplicates_removed,
        'post_candidates_before_filter': candidates_before_filter,
        'post_candidates_after_filter': candidates_after_filter,
        'post_candidates_after_topk': candidates_after_topk,
        'post_nms_input_max_per_class': nms_input_max_per_class,
        'post_flat_raw_score_min': raw_score_min,
        'post_flat_raw_score_max': raw_score_max,
        'post_flat_score_max': score_max,
        'post_fast_path': False,
    }

    if boxes is None or len(boxes) == 0:
        return None, None, None, profile
    return boxes, classes, scores, profile


def split_yolov8_post_process(input_data, img_shape=(640, 640), letterbox_meta=None):
    """
    后处理拆分输出 YOLOv8 RKNN: boxes=(1,8400,4), scores=(1,8400,num_classes).
    拆分输出用于避免 bbox 大数值范围把 score 量化成 0。
    """
    if input_data is None or len(input_data) != 2:
        raise ValueError(f'split YOLOv8 postprocess expects 2 outputs, got {0 if input_data is None else len(input_data)}')

    first = np.asarray(input_data[0])
    second = np.asarray(input_data[1])
    if first.ndim == 3 and first.shape[0] == 1:
        first = first[0]
    if second.ndim == 3 and second.shape[0] == 1:
        second = second[0]

    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f'split YOLOv8 outputs must be 2D/3D, got {first.shape}, {second.shape}')

    if first.shape[1] == 4 and second.shape[0] == first.shape[0]:
        boxes_xyxy = first
        class_probs = second
    elif second.shape[1] == 4 and first.shape[0] == second.shape[0]:
        boxes_xyxy = second
        class_probs = first
    else:
        raise ValueError(f'cannot identify split YOLOv8 boxes/scores shapes: {first.shape}, {second.shape}')

    combined = np.concatenate(
        [
            boxes_xyxy.astype(np.float32, copy=False),
            class_probs.astype(np.float32, copy=False),
        ],
        axis=1
    )
    boxes, classes, scores, profile = flat_yolov8_post_process(
        [combined],
        img_shape,
        letterbox_meta=letterbox_meta
    )
    profile['post_format'] = 'split_yolov8'
    return boxes, classes, scores, profile


def post_process(input_data, img_shape=(640, 640)):
    """后处理（纯NumPy实现，适配RK3588输出格式）"""
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    # 解析每个分支的输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i], img_shape))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    # 展平特征图
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    # 合并所有尺度的结果
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 过滤和NMS
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    if boxes.size == 0:
        return None, None, None

    # 按类别进行NMS
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_cls = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_cls[keep])
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None
        
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def _output_shapes(outputs):
    if outputs is None:
        return 'None'
    return ', '.join(str(getattr(output, 'shape', None)) for output in outputs)


def run_detection_postprocess(outputs, img_shape, use_fast_postprocess=True, letterbox_meta=None):
    """根据 RKNN 输出格式自动选择后处理。"""
    if outputs is None or len(outputs) == 0:
        raise ValueError('RKNN inference returned no outputs')

    if len(outputs) == 1:
        return flat_yolov8_post_process(outputs, img_shape, letterbox_meta=letterbox_meta)

    if len(outputs) == 2:
        return split_yolov8_post_process(outputs, img_shape, letterbox_meta=letterbox_meta)

    if use_fast_postprocess and ENABLE_FAST_POSTPROCESS:
        return fast_post_process(outputs, img_shape)

    boxes, classes, scores = post_process(outputs, img_shape)
    return boxes, classes, scores, {
        'post_fast_path': False,
        'post_format': 'dfl_multi_output',
    }


def detection_inference_func(rknn_instance, img_bgr, use_fast_postprocess=True, input_size=IMG_SIZE):
    """
    目标检测推理回调函数（在线程池中执行）
    
    Args:
        rknn_instance: RKNN 实例
        img_bgr: BGR 格式图像 (H, W, 3)
    
    Returns:
        detections: 检测结果列表
        flag: 是否成功
    """
    try:
        t_total_start = time.perf_counter()
        t_preprocess_start = t_total_start

        # 保存原始尺寸
        h_orig, w_orig = img_bgr.shape[:2]
        
        # RKNN 输入尺寸必须与转换模型时的 input_size 完全一致。
        img_resized, lb_scale, lb_pad_x, lb_pad_y = letterbox_image(img_bgr, input_size)
        img_input = np.expand_dims(img_resized, 0)

        t_rknn_start = time.perf_counter()
        
        # 推理
        outputs = rknn_instance.inference(inputs=[img_input])

        t_post_start = time.perf_counter()
        
        # 后处理
        post_profile = {}
        try:
            boxes, classes, scores, post_profile = run_detection_postprocess(
                outputs,
                (h_orig, w_orig),
                use_fast_postprocess=use_fast_postprocess,
                letterbox_meta=(lb_scale, lb_pad_x, lb_pad_y)
            )
        except Exception as e:
            print(
                f'⚠️ Detection postprocess failed: {e}; '
                f'outputs=[{_output_shapes(outputs)}]'
            )
            if len(outputs) == 1:
                raise
            boxes, classes, scores = post_process(outputs, (h_orig, w_orig))
            post_profile = {
                'post_fast_path': False,
                'post_fallback': True,
                'post_format': 'dfl_multi_output',
            }

        t_build_start = time.perf_counter()
        
        if boxes is None or len(boxes) == 0:
            t_end = time.perf_counter()
            profile = {
                'worker_total_ms': (t_end - t_total_start) * 1000.0,
                'worker_preprocess_ms': (t_rknn_start - t_preprocess_start) * 1000.0,
                'worker_rknn_ms': (t_post_start - t_rknn_start) * 1000.0,
                'worker_postprocess_ms': (t_build_start - t_post_start) * 1000.0,
                'worker_build_result_ms': (t_end - t_build_start) * 1000.0,
                'worker_num_detections': 0,
            }
            profile.update(post_profile)
            return [], True, profile
        
        # 构建检测结果
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            
            # 限制在图像范围内
            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            x2 = max(0, min(x2, w_orig))
            y2 = max(0, min(y2, h_orig))
            
            class_id = int(classes[i])
            confidence = float(scores[i])
            
            # 计算中心点
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            detection = {
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'center': [float(cx), float(cy)]
            }
            detections.append(detection)
        
        t_end = time.perf_counter()
        profile = {
            'worker_total_ms': (t_end - t_total_start) * 1000.0,
            'worker_preprocess_ms': (t_rknn_start - t_preprocess_start) * 1000.0,
            'worker_rknn_ms': (t_post_start - t_rknn_start) * 1000.0,
            'worker_postprocess_ms': (t_build_start - t_post_start) * 1000.0,
            'worker_build_result_ms': (t_end - t_build_start) * 1000.0,
            'worker_num_detections': len(detections),
        }
        profile.update(post_profile)
        return detections, True, profile
        
    except Exception as e:
        print(f'❌ Inference error: {e}')
        traceback.print_exc()
        return [], False, {}


class ObjectDetectionInfer:
    """
    目标检测推理器（异步线程池版本）
    
    使用多个 RKNN 实例并行推理，显著提升吞吐量。
    理论 FPS 提升：从 ~10 FPS 提升到 ~30 FPS（3个 NPU 核心）
    """
    
    def __init__(
        self,
        model_path,
        label_list_path,
        TPEs=3,
        core_ids=None,
        use_fast_postprocess=False,
        input_size=IMG_SIZE,
    ):
        """
        初始化目标检测推理器
        
        Args:
            model_path: RKNN 模型路径
            label_list_path: 标签列表文件路径
            TPEs: 线程池执行器数量（建议设置为 NPU 核心数，通常为 3）
            core_ids: NPU 核心 ID 列表，例如 [0, 1]；None 表示按 0/1/2 轮询
        """
        self.TPEs = TPEs
        self.core_ids = [int(core_id) for core_id in core_ids] if core_ids else None
        self.use_fast_postprocess = bool(use_fast_postprocess)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.pool_initialized = False
        self.last_profile = {}
        
        # 加载标签列表
        with open(label_list_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]
        
        # ⭐ 创建 RKNN 线程池执行器（内部会创建 TPEs 个 RKNN 实例）
        self.rknn_pool = RKNNPoolExecutor(
            rknn_model=model_path,
            tpes=self.TPEs,
            func=partial(
                detection_inference_func,
                use_fast_postprocess=self.use_fast_postprocess,
                input_size=self.input_size,
            ),
            core_ids=self.core_ids
        )
        
        print(f'✅ Object Detection Model loaded: {len(self.classes)} classes')
        print(f'   Classes: {", ".join(self.classes)}')
        print(f'   TPEs: {self.TPEs} (parallel inference enabled)')
        print(f'   NPU Core IDs: {self.core_ids if self.core_ids else "auto 0/1/2"}')
        print(f'   Fast Postprocess: {self.use_fast_postprocess}')
        print(f'   Input Size: {self.input_size[0]}x{self.input_size[1]}')
    
    def _pool_init(self, img_bgr):
        """
        初始化线程池（预填充数据）
        
        Args:
            img_bgr: 示例图像
        """
        for i in range(self.TPEs + 1):
            self.rknn_pool.put(img_bgr)
        self.pool_initialized = True
    
    def infer(self, img_bgr):
        """
        执行推理（异步流水线）
        
        工作流程：
        1. 首次调用时初始化线程池（预填充 TPEs+1 帧）
        2. 提交当前帧到线程池（非阻塞）
        3. 从队列获取上一帧的结果（阻塞等待）
        
        Args:
            img_bgr: BGR 格式图像 (H, W, 3)
            
        Returns:
            detections: 检测结果列表，每个元素为 dict:
                {
                    'class_id': int,
                    'class_name': str,
                    'confidence': float,
                    'bbox': [x1, y1, x2, y2],  # 像素坐标
                    'center': [cx, cy]  # 中心点坐标
                }
            flag: 是否成功
        """
        try:
            # 首次调用时初始化线程池
            if not self.pool_initialized:
                self._pool_init(img_bgr)
            
            # 提交当前帧到线程池（非阻塞，立即返回）
            self.rknn_pool.put(img_bgr)
            
            # 从队列获取结果（阻塞等待上一帧的推理完成）
            result_tuple = self.rknn_pool.get()
            
            if result_tuple is None:
                return [], False
            
            # ⭐ 解包：result_tuple = ((detections, flag_from_func), flag_from_pool)
            inner_result, flag_from_pool = result_tuple
            if not flag_from_pool or inner_result is None:
                return [], False
            
            if isinstance(inner_result, tuple) and len(inner_result) == 3:
                detections, flag_from_func, profile = inner_result
                self.last_profile = profile if isinstance(profile, dict) else {}
            else:
                detections, flag_from_func = inner_result
                self.last_profile = {}
            
            # 添加类别名称
            if detections:  # 确保 detections 不是空列表
                for det in detections:
                    if isinstance(det, dict) and 'class_id' in det:
                        class_id = det['class_id']
                        det['class_name'] = self.classes[class_id] if class_id < len(self.classes) else 'unknown'
            
            return detections, flag_from_func
            
        except Exception as e:
            print(f'❌ Inference error: {e}')
            traceback.print_exc()
            return [], False

    def get_last_profile(self):
        """返回最近一次完成推理任务的 worker 内部分段耗时。"""
        return self.last_profile
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'rknn_pool'):
            self.rknn_pool.release()
