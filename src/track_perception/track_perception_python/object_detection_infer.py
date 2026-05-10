#!/usr/bin/env python3
"""
RKNN 目标检测推理封装（异步线程池版本）
基于官方 setupUI 的 InferWrap 重构，支持多 NPU 核心并行推理
"""

import os
import sys
import glob
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ⭐ 导入 RKNN 线程池执行器
from .rknn_pool import RKNNPoolExecutor


# 目标检测参数
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


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


def nms_boxes(boxes, scores):
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
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


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


def detection_inference_func(rknn_instance, img_bgr):
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
        # 保存原始尺寸
        h_orig, w_orig = img_bgr.shape[:2]
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # resize 到模型输入尺寸
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_input = np.expand_dims(img_resized, 0)
        
        # 推理
        outputs = rknn_instance.inference(inputs=[img_input])
        
        # 后处理
        boxes, classes, scores = post_process(outputs, (h_orig, w_orig))
        
        if boxes is None or len(boxes) == 0:
            return [], True
        
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
        
        return detections, True
        
    except Exception as e:
        print(f'❌ Inference error: {e}')
        return [], False


class ObjectDetectionInfer:
    """
    目标检测推理器（异步线程池版本）
    
    使用多个 RKNN 实例并行推理，显著提升吞吐量。
    理论 FPS 提升：从 ~10 FPS 提升到 ~30 FPS（3个 NPU 核心）
    """
    
    def __init__(self, model_path, label_list_path, TPEs=3):
        """
        初始化目标检测推理器
        
        Args:
            model_path: RKNN 模型路径
            label_list_path: 标签列表文件路径
            TPEs: 线程池执行器数量（建议设置为 NPU 核心数，通常为 3）
        """
        self.TPEs = TPEs
        self.pool_initialized = False
        
        # 加载标签列表
        with open(label_list_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]
        
        # ⭐ 创建 RKNN 线程池执行器（内部会创建 TPEs 个 RKNN 实例）
        self.rknn_pool = RKNNPoolExecutor(
            rknn_model=model_path,
            tpes=self.TPEs,
            func=detection_inference_func
        )
        
        print(f'✅ Object Detection Model loaded: {len(self.classes)} classes')
        print(f'   Classes: {", ".join(self.classes)}')
        print(f'   TPEs: {self.TPEs} (parallel inference enabled)')
    
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
            
            detections, flag_from_func = inner_result
            
            # 添加类别名称
            if detections:  # 确保 detections 不是空列表
                for det in detections:
                    if isinstance(det, dict) and 'class_id' in det:
                        class_id = det['class_id']
                        det['class_name'] = self.classes[class_id] if class_id < len(self.classes) else 'unknown'
            
            return detections, flag_from_func
            
        except Exception as e:
            print(f'❌ Inference error: {e}')
            return [], False
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'rknn_pool'):
            self.rknn_pool.release()
