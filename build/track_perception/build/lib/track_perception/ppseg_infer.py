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

def preprocess_image(img, input_format="RGB", model_input_format="RGB", input_size=DEFAULT_IMG_SIZE):
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
    
    input_size = (int(input_size[0]), int(input_size[1]))

    # Resize to model input size
    if img_converted.shape[1] == input_size[0] and img_converted.shape[0] == input_size[1]:
        img_resized = img_converted
    else:
        img_resized = cv2.resize(img_converted, input_size, interpolation=cv2.INTER_LINEAR)

    img_normalized = np.ascontiguousarray(img_resized)
    # 增加 batch 维度: (H, W, C) -> (1, H, W, C)
    input_data = np.expand_dims(img_normalized, axis=0)
    return input_data

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
    conf_threshold=0.25,
    mask_threshold=0.5,
    max_detections=30,
):
    """Convert YOLOv8-seg instance output into the semantic road mask expected downstream."""
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

    orig_h, orig_w = original_size
    empty_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    class_scores = pred[:, 4:4 + class_count].astype(np.float32, copy=False)
    if class_scores.size == 0:
        return empty_mask
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

    boxes = boxes_to_xyxy(pred[keep, :4])
    coeffs = pred[keep, 4 + class_count:4 + class_count + mask_dim].astype(np.float32, copy=False)

    input_w, input_h = int(input_size[0]), int(input_size[1])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(input_w))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(input_h))

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

    road_mask = cv2.resize(
        road_mask_low.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return road_mask.astype(np.uint8)

def postprocess_segmentation(
    output,
    original_size,
    input_size=DEFAULT_IMG_SIZE,
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
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
            max_detections=max_detections,
        )
        if yolo_mask is not None:
            return yolo_mask

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

        # ⭐ 关键优化：只在尺寸不一致时才resize
        orig_h, orig_w = original_size
        if seg_map.shape[0] == orig_h and seg_map.shape[1] == orig_w:
            # 尺寸已经一致，直接返回
            return seg_map.astype(np.uint8)
        else:
            # 需要resize
            seg_map_resized = cv2.resize(
                seg_map.astype(np.float32), 
                (orig_w, orig_h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            return seg_map_resized
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
        input_data = preprocess_image(
            img_bgr,
            input_format=input_format,
            model_input_format=model_input_format,
            input_size=input_size,
        )

        t_rknn_start = time.perf_counter()
        
        # 推理
        outputs = rknn_lite.inference(inputs=[input_data])

        t_post_start = time.perf_counter()
        
        # 检查推理是否成功
        if outputs is None or len(outputs) == 0:
            print("[ERROR] Model inference returned None or empty output")
            return None, None, False, {}
    
        # 后处理 - 获取分割掩码
        seg_map = postprocess_segmentation(
            outputs,
            original_size,
            input_size=input_size,
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
                result_img = blend_images(img_bgr, colored_seg, alpha=blend_alpha)
            else:
                # 直接返回彩色分割图
                result_img = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
        else:
            # 不需要可视化，返回None
            result_img = None

        t_end = time.perf_counter()
        profile = {
            'worker_total_ms': (t_end - t_total_start) * 1000.0,
            'worker_preprocess_ms': (t_rknn_start - t_preprocess_start) * 1000.0,
            'worker_rknn_ms': (t_post_start - t_rknn_start) * 1000.0,
            'worker_postprocess_ms': (t_vis_start - t_post_start) * 1000.0,
            'worker_visualization_ms': (t_end - t_vis_start) * 1000.0,
            'worker_show_visualization': show_visualization,
        }
        
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
        self.conf_threshold = float(conf_threshold)
        self.mask_threshold = float(mask_threshold)
        self.max_detections = int(max_detections)
        self.last_profile = {}
        
        # 创建带有参数的推理函数
        from functools import partial
        infer_func = partial(myFunc, blend_alpha=blend_alpha, show_visualization=show_visualization, 
                            input_format=input_format, model_input_format=model_input_format,
                            input_size=self.input_size, conf_threshold=self.conf_threshold,
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
