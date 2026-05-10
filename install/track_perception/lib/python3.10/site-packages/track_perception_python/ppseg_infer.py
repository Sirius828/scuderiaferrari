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

# Cityscapes 颜色映射表 (RGB)
# 可以自定义每个类别的显示颜色
# 设置为 [0, 0, 0] 的类别将不会显示（保持原图）
SEG_COLORS = np.array([
    [0, 0, 0],          # 0: 不显示（原认为是道路，实际可能是背景）
    [0, 128, 255],      # 1: sidewalk - 亮蓝色
    [0, 128, 255],      # 2: building - 亮蓝色
    [0, 128, 255],      # 3: wall - 亮蓝色
    [0, 128, 255],      # 4: fence - 亮蓝色
    [0, 128, 255],      # 5: pole - 亮蓝色
    [0, 128, 255],      # 6: traffic light - 亮蓝色
    [0, 128, 255],      # 7: traffic sign - 亮蓝色
    [0, 128, 255],      # 8: vegetation - 亮蓝色
    [0, 128, 255],      # 9: terrain - 亮蓝色
    [0, 128, 255],      # 10: sky - 亮蓝色
    [0, 128, 255],      # 11: person - 亮蓝色
    [0, 128, 255],      # 12: rider - 亮蓝色
    [0, 128, 255],      # 13: car - 亮蓝色
    [0, 128, 255],      # 14: truck - 亮蓝色
    [0, 128, 255],      # 15: bus - 亮蓝色
    [0, 128, 255],      # 16: train - 亮蓝色
    [0, 128, 255],      # 17: motorcycle - 亮蓝色
    [0, 128, 255]       # 18: bicycle - 亮蓝色
], dtype=np.uint8)

# 模型输入尺寸 (必须与 pp_liteseg.rknn 模型匹配)
# 921600 bytes = 640 * 480 * 3
IMG_SIZE = (640, 480)  # (width, height)

def preprocess_image(img, input_format="RGB", model_input_format="RGB"):
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
    
    # Resize to model input size
    img_resized = cv2.resize(img_converted, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    img_normalized = img_resized
    # 增加 batch 维度: (H, W, C) -> (1, H, W, C)
    input_data = np.expand_dims(img_normalized, axis=0)
    return input_data

def postprocess_segmentation(output, original_size):
    """
    后处理分割结果
    Args:
        output: 模型输出，形状通常为 (1, num_classes, H, W) 或 (1, H, W)
        original_size: 原始图像尺寸 (height, width)
    Returns:
        seg_map: 分割掩码 (H, W)，值为类别索引
    """
    try:
        # 获取输出数据
        seg_output = output[0] if isinstance(output, list) else output
        
        if seg_output is None:
            print("[ERROR] seg_output is None")
            return None
     
        # 处理不同维度的输出
        if len(seg_output.shape) == 4:
            # (1, num_classes, H, W) -> 取第一个batch
            seg_map = seg_output[0]
        elif len(seg_output.shape) == 3:
            # (1, H, W) 或 (num_classes, H, W) 或 (H, W, num_classes)
            if seg_output.shape[0] == 1:
                seg_map = seg_output[0]
            elif seg_output.shape[-1] > 10:  # 最后一个维度是类别数
                # (H, W, num_classes) -> argmax on last axis
                seg_map = np.argmax(seg_output, axis=-1)
                if postprocess_segmentation._debug_count <= 3:
                    print(f"[DEBUG] Detected (H, W, C) format, applying argmax on axis=-1")
            else:
                # 多类别输出，取 argmax
                seg_map = np.argmax(seg_output, axis=0)
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
    h, w = seg_map.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个类别应用颜色
    for class_idx, color in enumerate(SEG_COLORS):
        mask = seg_map == class_idx
        colored_seg[mask] = color
    
    return colored_seg

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
def myFunc(rknn_lite, img_bgr, blend_alpha=0.5, show_visualization=True, input_format="RGB", model_input_format="RGB"):
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
        # 保存原始尺寸
        original_size = img_bgr.shape[:2]  # (height, width)
        
        # 预处理 - ⭐ 传递颜色格式参数
        input_data = preprocess_image(img_bgr, input_format=input_format, model_input_format=model_input_format)
        
        # 推理
        outputs = rknn_lite.inference(inputs=[input_data])
        
        # 检查推理是否成功
        if outputs is None or len(outputs) == 0:
            print("[ERROR] Model inference returned None or empty output")
            return None, None, False
    
        # 后处理 - 获取分割掩码
        seg_map = postprocess_segmentation(outputs, original_size)
        
        if seg_map is None:
            print("[ERROR] Postprocessing returned None")
            return None, None, False
        
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
        
        # 返回3个值：可视化结果（可能为None）、原始seg_map、成功标志
        return result_img, seg_map, True
    
    except Exception as e:
        print(f"Error in myFunc: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def get_current_dir():
    """获取当前脚本所在目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir


class PPSegInfer:
    """PP-Seg 推理封装类"""
    
    def __init__(self, model_dir="model", model_filename=None, TPEs=1, blend_alpha=None, show_visualization=True, input_format="RGB", model_input_format="RGB"):
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
        """
        model_dir = os.path.join(get_current_dir(), model_dir)
        model_path = self.get_model_path(model_dir, model_filename)
        
        self.TPEs = TPEs
        self.blend_alpha = blend_alpha
        self.show_visualization = show_visualization
        self.input_format = input_format
        self.model_input_format = model_input_format
        
        # 创建带有参数的推理函数
        from functools import partial
        infer_func = partial(myFunc, blend_alpha=blend_alpha, show_visualization=show_visualization, 
                            input_format=input_format, model_input_format=model_input_format)
        
        self.rknn_pool = rknnPoolExecutor(
            rknnModel=model_path,
            TPEs=self.TPEs,
            func=infer_func
        )
        self.pool_flag = False
        
        mode = "blended" if blend_alpha is not None else "segmentation_only"
        vis_mode = "with_visualization" if show_visualization else "mask_only"
        print(f"PP-Seg model loaded: {model_path}")
        print(f"Thread pool size: {TPEs}")
        print(f"Output mode: {mode} (blend_alpha={blend_alpha})")
        print(f"Visualization: {vis_mode}")

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
            if isinstance(func_result, tuple) and len(func_result) == 3:
                result_img, seg_map, myfunc_flag = func_result
                return result_img, seg_map, myfunc_flag
            else:
                return func_result, None, True
        else:
            return None, None, False
    
    def __call__(self, *args, **kwargs):
        """使实例可调用"""
        return self.infer(*args, **kwargs)
    
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
