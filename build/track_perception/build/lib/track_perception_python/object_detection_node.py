#!/usr/bin/env python3
"""
目标检测 ROS2 节点
从共享内存读取视频流，进行 RKNN 目标检测推理，发布检测结果
"""

import rclpy
from rclpy.node import Node
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
        self.declare_parameter('model_path', 'model/rknn_lt.rknn')
        self.declare_parameter('label_list_path', 'model/label_list.txt')
        self.declare_parameter('tpes', 3)
        self.declare_parameter('enable_flip', True)
        self.declare_parameter('flip_code', 0)
        self.declare_parameter('input_format', 'RGB')
        self.declare_parameter('publish_detections', True)  # 是否发布检测结果
        self.declare_parameter('publish_rate', 30)  # ⭐ 发布频率 (Hz)，提高到 30Hz 匹配异步推理的高吞吐量
        
        # 获取参数
        self.shm_name = self.get_parameter('shm_name').get_parameter_value().string_value
        
        # 模型路径处理（支持相对路径和绝对路径）
        model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path_param.startswith('/'):
            # 相对路径，转换为包内路径
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('track_perception')
            model_path_param = f'{pkg_dir}/track_perception_python/{model_path_param}'
        
        label_list_path_param = self.get_parameter('label_list_path').get_parameter_value().string_value
        if not label_list_path_param.startswith('/'):
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('track_perception')
            label_list_path_param = f'{pkg_dir}/track_perception_python/{label_list_path_param}'
        
        tpes = self.get_parameter('tpes').get_parameter_value().integer_value
        self.enable_flip = self.get_parameter('enable_flip').get_parameter_value().bool_value
        self.flip_code = self.get_parameter('flip_code').get_parameter_value().integer_value
        self.input_format = self.get_parameter('input_format').get_parameter_value().string_value
        self.publish_detections = self.get_parameter('publish_detections').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().integer_value
        
        self.SHM_HEADER_SIZE = 16
        self.last_fid = 0
        
        # FPS 统计
        self.fps_t = time.time()
        self.fps_n = 0
        self.cur_fps = 0.0
        
        # ⭐ 性能统计时间控制（每5秒打印一次）
        self.last_perf_time = time.time()
        self.perf_interval = 5.0  # 5秒间隔
        
        # ⭐ 发布时间控制（基于时间戳，而不是帧计数）
        self.last_publish_time = 0.0
        self.publish_period = 1.0 / self.publish_rate if self.publish_rate > 0 else 0.1
        
        # 初始化目标检测推理器
        try:
            self.det_infer = ObjectDetectionInfer(
                model_path=model_path_param,
                label_list_path=label_list_path_param,
                TPEs=tpes
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
        self.get_logger().info(f'   Enable Flip: {self.enable_flip}')
        self.get_logger().info(f'   Publish Rate: {self.publish_rate} Hz')
        
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
            # ⭐ 性能统计：先判断是否需要采样（每5秒一次）
            current_time = time.time()
            need_perf_stats = (current_time - self.last_perf_time >= self.perf_interval)
            t_start = None
            t_preprocess_start = None
            t_inference_start = None
            t_publish_start = None
            if need_perf_stats:
                t_start = time.perf_counter()
            
            # 1. 读取头部
            header = bytes(self.shm.buf[:self.SHM_HEADER_SIZE])
            fid, w, h = struct.unpack('QII', header)
            
            # 优化：帧号没变就不读
            if fid == self.last_fid:
                return
            
            self.last_fid = fid
            
            # 2. 读取数据
            size = w * h * 3
            img_view = np.ndarray((h, w, 3), dtype=np.uint8, 
                                 buffer=self.shm.buf[self.SHM_HEADER_SIZE : self.SHM_HEADER_SIZE+size])
            frame = img_view.copy()  # Deep Copy
            del img_view  # 立即释放视图
            
            # 3. 预处理（flip + cvtColor）
            if need_perf_stats:
                t_preprocess_start = time.perf_counter()
            if self.enable_flip:
                frame = cv2.flip(frame, self.flip_code)
            
            # 颜色转换
            if self.input_format == "BGR":
                # 如果输入是 BGR，需要转换为 RGB 用于推理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # FPS 统计
            self.fps_n += 1
            if time.time() - self.fps_t >= 1.0:
                self.cur_fps = self.fps_n / (time.time() - self.fps_t)
                self.fps_n = 0
                self.fps_t = time.time()
            
            # 4. 目标检测推理
            if need_perf_stats:
                t_inference_start = time.perf_counter()
            detections, flag = self.det_infer.infer(frame_rgb)
            
            if flag and detections:
                # ⭐ 移除检测目标的日志输出，减少刷屏
                
                # 5. ⭐ 基于时间的发布控制
                if self.publish_detections:
                    current_time = time.time()
                    if current_time - self.last_publish_time >= self.publish_period:
                        if need_perf_stats:
                            t_publish_start = time.perf_counter()
                        self.publish_detections_msg(detections)
                        self.last_publish_time = current_time
                        
                        # ⭐ 性能统计已禁用，如需启用请取消下面的注释
                        # if need_perf_stats:
                        #     t_end = time.perf_counter()
                        #     t_preprocess = t_inference_start - t_preprocess_start if t_inference_start and t_preprocess_start else 0
                        #     t_model_inference = t_publish_start - t_inference_start if t_publish_start and t_inference_start else 0
                        #     t_publish = t_end - t_publish_start if t_publish_start else 0
                        #     t_total = t_end - t_start
                        #     self.get_logger().info(
                        #         f'⏱️ Performance Stats:\n'
                        #         f'   Preprocess:    {t_preprocess*1000:6.2f} ms\n'
                        #         f'   Inference:     {t_model_inference*1000:6.2f} ms\n'
                        #         f'   Publish:       {t_publish*1000:6.2f} ms\n'
                        #         f'   Total:         {t_total*1000:6.2f} ms ({1000/t_total:6.1f} FPS)'
                        #     )
                        #     self.last_perf_time = current_time
            else:
                # 没有检测到物体
                if self.publish_detections:
                    current_time = time.time()
                    if current_time - self.last_publish_time >= self.publish_period:
                        if need_perf_stats:
                            t_publish_start = time.perf_counter()
                        self.publish_empty_detections()
                        self.last_publish_time = current_time
                        
                        # ⭐ 性能统计已禁用，如需启用请取消下面的注释
                        # if need_perf_stats:
                        #     t_end = time.perf_counter()
                        #     t_preprocess = t_inference_start - t_preprocess_start if t_inference_start and t_preprocess_start else 0
                        #     t_model_inference = t_publish_start - t_inference_start if t_publish_start and t_inference_start else 0
                        #     t_publish = t_end - t_publish_start if t_publish_start else 0
                        #     t_total = t_end - t_start
                        #     self.get_logger().info(
                        #         f'⏱️ Performance Stats (no detections):\n'
                        #         f'   Preprocess:    {t_preprocess*1000:6.2f} ms\n'
                        #         f'   Inference:     {t_model_inference*1000:6.2f} ms\n'
                        #         f'   Publish:       {t_publish*1000:6.2f} ms\n'
                        #         f'   Total:         {t_total*1000:6.2f} ms ({1000/t_total:6.1f} FPS)'
                        #     )
                        #     self.last_perf_time = current_time
            
        except (ValueError, struct.error, BufferError) as e:
            self.get_logger().error(f'Error processing frame: {e}')
            self.disconnect_shm()
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')
    
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
