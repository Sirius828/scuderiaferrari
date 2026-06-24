#!/usr/bin/env python3
"""
目标检测 ROS2 节点
从共享内存读取视频流，进行 RKNN 目标检测推理，发布检测结果
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
import time
import struct
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker
from .object_detection_infer import ObjectDetectionInfer
from std_msgs.msg import Header, Float32MultiArray, String


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # ==================== 参数声明 ====================
        self.declare_parameter('shm_name', 'shm_ar_video')
        self.declare_parameter('model_path', 'model/yolov8_n_det_split_int8_v2.rknn')
        self.declare_parameter('label_list_path', 'model/label_list.txt')
        self.declare_parameter('tpes', 1)
        self.declare_parameter('core_ids', [0], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('enable_flip', True)
        self.declare_parameter('flip_code', 0)
        self.declare_parameter('input_format', 'RGB')
        self.declare_parameter('publish_detections', True)  # 是否发布检测结果
        self.declare_parameter('publish_rate', 0)  # <=0 表示不限速，发布每个完成结果
        self.declare_parameter('enable_perf_stats', True)
        self.declare_parameter('perf_interval', 2.0)
        self.declare_parameter('use_fast_postprocess', False)
        self.declare_parameter('model_input_width', 384)
        self.declare_parameter('model_input_height', 288)
        
        # 获取参数
        self.shm_name = self.get_parameter('shm_name').get_parameter_value().string_value
        
        # 模型路径处理（支持相对路径和绝对路径）
        model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path_param.startswith('/'):
            # 相对路径，转换为包内路径
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('track_perception')
            model_path_param = f'{pkg_dir}/{model_path_param}'
        
        label_list_path_param = self.get_parameter('label_list_path').get_parameter_value().string_value
        if not label_list_path_param.startswith('/'):
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('track_perception')
            label_list_path_param = f'{pkg_dir}/{label_list_path_param}'
        
        tpes = self.get_parameter('tpes').get_parameter_value().integer_value
        core_ids = list(self.get_parameter('core_ids').get_parameter_value().integer_array_value)
        self.enable_flip = self.get_parameter('enable_flip').get_parameter_value().bool_value
        self.flip_code = self.get_parameter('flip_code').get_parameter_value().integer_value
        self.input_format = self.get_parameter('input_format').get_parameter_value().string_value
        self.publish_detections = self.get_parameter('publish_detections').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().integer_value
        self.enable_perf_stats = self.get_parameter('enable_perf_stats').get_parameter_value().bool_value
        self.perf_interval = self.get_parameter('perf_interval').get_parameter_value().double_value
        self.use_fast_postprocess = self.get_parameter('use_fast_postprocess').get_parameter_value().bool_value
        self.model_input_width = self.get_parameter('model_input_width').get_parameter_value().integer_value
        self.model_input_height = self.get_parameter('model_input_height').get_parameter_value().integer_value
        
        self.SHM_HEADER_SIZE = 16
        self.last_fid = 0
        
        # FPS 统计
        self.fps_t = time.time()
        self.fps_n = 0
        self.cur_fps = 0.0
        self.publish_fps_t = time.time()
        self.processed_frames_window = 0
        self.published_msgs_window = 0
        self.rate_limited_skips_window = 0
        self.cur_publish_fps = 0.0
        self.perf_window_start_time = None
        self.perf_window_start_fid = None
        
        # ⭐ 性能统计时间控制
        self.last_perf_time = time.time()
        
        # ⭐ 发布时间控制（基于时间戳，而不是帧计数）
        self.last_publish_time = 0.0
        # publish_rate <= 0 表示每个完成的检测结果都发布，不做时间门限。
        self.publish_period = 1.0 / self.publish_rate if self.publish_rate > 0 else None
        
        # 初始化目标检测推理器
        try:
            self.det_infer = ObjectDetectionInfer(
                model_path=model_path_param,
                label_list_path=label_list_path_param,
                TPEs=tpes,
                core_ids=core_ids,
                use_fast_postprocess=self.use_fast_postprocess,
                input_size=(self.model_input_width, self.model_input_height)
            )
            self.get_logger().info('✅ Object Detection model initialized successfully')
            self.get_logger().info(f'   📦 Model: {model_path_param}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to initialize detection model: {e}')
            raise
        
        # ==================== 发布者 ====================
        # 发布检测结果：[class_id, confidence, x1, y1, x2, y2, cx, cy] * N
        self.detections_publisher = self.create_publisher(
            Float32MultiArray,
            '/detection/results',
            10
        )
        
        # 发布检测到的类别标签（逗号分隔）
        self.labels_publisher = self.create_publisher(
            String,
            '/detection/labels',
            10
        )
        
        # 共享内存对象
        self.shm = None
        
        self.get_logger().info(f'📡 Object Detection Node Ready')
        self.get_logger().info(f'   SHM Name: {self.shm_name}')
        self.get_logger().info(f'   Model Path: {model_path_param}')
        self.get_logger().info(f'   Label List: {label_list_path_param}')
        self.get_logger().info(f'   TPEs: {tpes}')
        self.get_logger().info(f'   NPU Core IDs: {core_ids if core_ids else "auto 0/1/2"}')
        self.get_logger().info(f'   Enable Flip: {self.enable_flip}')
        publish_mode = f'{self.publish_rate} Hz' if self.publish_period is not None else 'unlimited'
        self.get_logger().info(f'   Publish Rate: {publish_mode}')
        self.get_logger().info(f'   Perf Stats: {self.enable_perf_stats} (interval={self.perf_interval:.1f}s)')
        self.get_logger().info(f'   Fast Postprocess: {self.use_fast_postprocess}')
        self.get_logger().info(f'   Model Input Size: {self.model_input_width}x{self.model_input_height}')
        
        # 启动主循环定时器
        self.timer = self.create_timer(0.001, self.main_loop)  # 1ms 间隔，尽可能快
    
    def remove_shm_from_resource_tracker(self):
        """防止客户端退出时误删服务端的 SHM"""
        try:
            resource_tracker.unregister('/' + self.shm_name, 'shared_memory')
        except:
            pass
    
    def connect_to_shm(self):
        """连接到共享内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.remove_shm_from_resource_tracker()
            self.get_logger().info('✅ Connected to shared memory!')
            return True
        except FileNotFoundError:
            return False
    
    def process_frame(self):
        """处理一帧图像"""
        if self.shm is None:
            return
        
        try:
            # ⭐ 性能统计：低频采样，避免日志本身影响帧率
            current_time = time.time()
            need_perf_stats = (
                self.enable_perf_stats and
                current_time - self.last_perf_time >= self.perf_interval
            )
            t_start = None
            t_header_done = None
            t_copy_done = None
            t_preprocess_done = None
            t_inference_start = None
            t_inference_done = None
            t_publish_start = None
            t_publish_done = None
            if need_perf_stats:
                t_start = time.perf_counter()
            
            # 1. 读取头部
            header = bytes(self.shm.buf[:self.SHM_HEADER_SIZE])
            fid, w, h = struct.unpack('QII', header)
            if need_perf_stats:
                t_header_done = time.perf_counter()
            
            # 优化：帧号没变就不读
            if fid == self.last_fid:
                return
            
            self.last_fid = fid
            if self.perf_window_start_fid is None:
                self.perf_window_start_fid = fid
                self.perf_window_start_time = current_time
            
            # 2. 读取数据
            size = w * h * 3
            img_view = np.ndarray((h, w, 3), dtype=np.uint8, 
                                 buffer=self.shm.buf[self.SHM_HEADER_SIZE : self.SHM_HEADER_SIZE+size])
            frame = img_view.copy()  # Deep Copy
            del img_view  # 立即释放视图
            if need_perf_stats:
                t_copy_done = time.perf_counter()
            
            # 3. 预处理（flip + cvtColor）
            if self.enable_flip:
                frame = cv2.flip(frame, self.flip_code)
            
            # 颜色转换
            if self.input_format == "BGR":
                # 如果输入是 BGR，需要转换为 RGB 用于推理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            if need_perf_stats:
                t_preprocess_done = time.perf_counter()
            
            # FPS 统计
            self.fps_n += 1
            self.processed_frames_window += 1
            if time.time() - self.fps_t >= 1.0:
                self.cur_fps = self.fps_n / (time.time() - self.fps_t)
                self.fps_n = 0
                self.fps_t = time.time()
            
            # 4. 目标检测推理
            if need_perf_stats:
                t_inference_start = time.perf_counter()
            detections, flag = self.det_infer.infer(frame_rgb)
            if need_perf_stats:
                t_inference_done = time.perf_counter()
            
            if flag and detections:
                # ⭐ 移除检测目标的日志输出，减少刷屏
                
                # 5. ⭐ 基于时间的发布控制
                if self.publish_detections:
                    current_time = time.time()
                    if self.should_publish(current_time):
                        if need_perf_stats:
                            t_publish_start = time.perf_counter()
                        self.publish_detections_msg(detections)
                        self.published_msgs_window += 1
                        if need_perf_stats:
                            t_publish_done = time.perf_counter()
                        self.last_publish_time = current_time
                    else:
                        self.rate_limited_skips_window += 1
            else:
                # 没有检测到物体
                if self.publish_detections:
                    current_time = time.time()
                    if self.should_publish(current_time):
                        if need_perf_stats:
                            t_publish_start = time.perf_counter()
                        self.publish_empty_detections()
                        self.published_msgs_window += 1
                        if need_perf_stats:
                            t_publish_done = time.perf_counter()
                        self.last_publish_time = current_time
                    else:
                        self.rate_limited_skips_window += 1

            if need_perf_stats:
                t_end = time.perf_counter()
                publish_window_elapsed = max(current_time - self.publish_fps_t, 1e-6)
                upstream_window_elapsed = max(
                    current_time - (self.perf_window_start_time or self.publish_fps_t),
                    1e-6
                )
                upstream_frames_window = max(
                    0,
                    int(fid - self.perf_window_start_fid + 1)
                    if self.perf_window_start_fid is not None else self.processed_frames_window
                )
                missed_upstream_frames = max(0, upstream_frames_window - self.processed_frames_window)
                upstream_fps = upstream_frames_window / upstream_window_elapsed
                self.cur_publish_fps = self.published_msgs_window / publish_window_elapsed
                self.log_perf_stats(
                    t_start=t_start,
                    t_header_done=t_header_done,
                    t_copy_done=t_copy_done,
                    t_preprocess_done=t_preprocess_done,
                    t_inference_start=t_inference_start,
                    t_inference_done=t_inference_done,
                    t_publish_start=t_publish_start,
                    t_publish_done=t_publish_done,
                    t_end=t_end,
                    fid=fid,
                    frame_shape=(w, h),
                    detections_count=len(detections) if detections else 0,
                    flag=flag,
                    processed_window=self.processed_frames_window,
                    published_window=self.published_msgs_window,
                    rate_limited_skips=self.rate_limited_skips_window,
                    publish_window_elapsed=publish_window_elapsed,
                    upstream_frames_window=upstream_frames_window,
                    missed_upstream_frames=missed_upstream_frames,
                    upstream_fps=upstream_fps
                )
                self.last_perf_time = current_time
                self.publish_fps_t = current_time
                self.perf_window_start_time = current_time
                self.perf_window_start_fid = fid + 1
                self.processed_frames_window = 0
                self.published_msgs_window = 0
                self.rate_limited_skips_window = 0
            
        except (ValueError, struct.error, BufferError) as e:
            self.get_logger().error(f'Error processing frame: {e}')
            self.disconnect_shm()
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')

    def should_publish(self, current_time):
        """判断当前完成的检测结果是否需要发布。"""
        if self.publish_period is None:
            return True
        return current_time - self.last_publish_time >= self.publish_period

    def log_perf_stats(self, t_start, t_header_done, t_copy_done, t_preprocess_done,
                       t_inference_start, t_inference_done, t_publish_start,
                       t_publish_done, t_end, fid, frame_shape, detections_count, flag,
                       processed_window, published_window, rate_limited_skips,
                       publish_window_elapsed, upstream_frames_window,
                       missed_upstream_frames, upstream_fps):
        """打印 detection 单帧端到端耗时和 worker 内部分段耗时。"""
        def ms(a, b):
            if a is None or b is None:
                return 0.0
            return (b - a) * 1000.0

        publish_ms = ms(t_publish_start, t_publish_done) if t_publish_done is not None else 0.0
        total_ms = ms(t_start, t_end)
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        worker = self.det_infer.get_last_profile() if hasattr(self.det_infer, 'get_last_profile') else {}

        self.get_logger().info(
            '⏱️ Detection frame profile\n'
            f'   fid={fid}, size={frame_shape[0]}x{frame_shape[1]}, flag={flag}, detections={detections_count}, loop_fps≈{self.cur_fps:.1f}\n'
            f'   shm_stream: upstream_frames={upstream_frames_window}, '
            f'shm_fps≈{upstream_fps:.1f}, missed_by_node={missed_upstream_frames}\n'
            f'   window:     processed={processed_window}, published={published_window}, '
            f'skipped_by_rate={rate_limited_skips}, publish_fps≈{self.cur_publish_fps:.1f} over {publish_window_elapsed:.2f}s\n'
            f'   pipeline:   node_total={total_ms:7.2f} ms ({fps:5.1f} FPS), '
            f'shm_header={ms(t_start, t_header_done):.2f}, '
            f'shm_copy={ms(t_header_done, t_copy_done):.2f}, '
            f'node_pre={ms(t_copy_done, t_preprocess_done):.2f}, '
            f'infer_wait={ms(t_inference_start, t_inference_done):.2f}, '
            f'publish={publish_ms:.2f}\n'
            f'   det_model:  worker_total={worker.get("worker_total_ms", 0.0):7.2f} ms, '
            f'pre={worker.get("worker_preprocess_ms", 0.0):.2f}, '
            f'rknn={worker.get("worker_rknn_ms", 0.0):.2f}, '
            f'post={worker.get("worker_postprocess_ms", 0.0):.2f}, '
            f'build={worker.get("worker_build_result_ms", 0.0):.2f}, '
            f'num_det={worker.get("worker_num_detections", 0)}\n'
            f'   det_post:   format={worker.get("post_format", "unknown")}, '
            f'fast_path={worker.get("post_fast_path", False)}, '
            f'score_filter={worker.get("post_score_filter_ms", 0.0):.2f}, '
            f'topk={worker.get("post_topk_ms", 0.0):.2f}, '
            f'decode={worker.get("post_dfl_decode_ms", 0.0):.2f}, '
            f'nms={worker.get("post_nms_ms", 0.0):.2f}, '
            f'dedupe={worker.get("post_dedupe_ms", 0.0):.2f}, '
            f'removed={worker.get("post_duplicates_removed", 0)}\n'
            f'              multiclass_nms={worker.get("post_multiclass_nms", False)}, '
            f'agnostic_removed={worker.get("post_agnostic_removed", 0)}, '
            f'keep_topk_removed={worker.get("post_keep_topk_removed", 0)}\n'
            f'              candidates={worker.get("post_candidates_before_filter", 0)} -> '
            f'{worker.get("post_candidates_after_filter", 0)} -> '
            f'{worker.get("post_candidates_after_topk", 0)}, '
            f'max_per_class={worker.get("post_nms_input_max_per_class", 0)}\n'
            f'              flat_scores raw=[{worker.get("post_flat_raw_score_min", 0.0):.3f}, '
            f'{worker.get("post_flat_raw_score_max", 0.0):.3f}], '
            f'max={worker.get("post_flat_score_max", 0.0):.3f}'
        )
    
    def publish_detections_msg(self, detections):
        """发布检测结果消息"""
        try:
            # 构建 Float32MultiArray 消息
            # 格式: [class_id, confidence, x1, y1, x2, y2, cx, cy] * N
            data = []
            labels = []
            
            for det in detections:
                data.extend([
                    float(det['class_id']),
                    det['confidence'],
                    float(det['bbox'][0]),  # x1
                    float(det['bbox'][1]),  # y1
                    float(det['bbox'][2]),  # x2
                    float(det['bbox'][3]),  # y2
                    det['center'][0],  # cx
                    det['center'][1]   # cy
                ])
                labels.append(det['class_name'])
            
            # 发布检测结果
            msg = Float32MultiArray()
            msg.data = data
            self.detections_publisher.publish(msg)
            
            # 发布标签字符串
            label_msg = String()
            label_msg.data = ','.join(labels)
            self.labels_publisher.publish(label_msg)
            
            self.get_logger().debug(f'Published {len(detections)} detections')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing detections: {e}')
    
    def publish_empty_detections(self):
        """发布空检测结果"""
        try:
            msg = Float32MultiArray()
            msg.data = []
            self.detections_publisher.publish(msg)
            
            label_msg = String()
            label_msg.data = ''
            self.labels_publisher.publish(label_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing empty detections: {e}')
    
    def disconnect_shm(self):
        """断开共享内存连接"""
        if self.shm:
            try:
                self.shm.close()
            except:
                pass
            self.shm = None
            self.get_logger().warn('❌ Connection lost...')
    
    def main_loop(self):
        """主循环"""
        if self.shm is None:
            # 尝试连接
            if not self.connect_to_shm():
                # 降低日志频率
                if not hasattr(self, '_wait_log_counter'):
                    self._wait_log_counter = 0
                self._wait_log_counter += 1
                if self._wait_log_counter % 1000 == 0:  # 每1000次打印一次
                    self.get_logger().info('Waiting for server...')
        else:
            # 处理帧
            self.process_frame()
    
    def destroy_node(self):
        """清理资源"""
        self.get_logger().info('🛑 Shutting down object detection node...')
        
        # 关闭共享内存
        self.disconnect_shm()
        
        # 释放模型资源
        if hasattr(self, 'det_infer'):
            self.det_infer.release()
            self.get_logger().info('🔒 Detection model resources released')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = ObjectDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n🛑 Keyboard interrupt received')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
