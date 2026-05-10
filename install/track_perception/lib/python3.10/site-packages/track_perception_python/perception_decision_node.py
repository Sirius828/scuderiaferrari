#!/usr/bin/env python3
"""
感知决策节点
整合语义分割和目标检测，进行融合决策并发布控制指令
- 订阅: /detection/results (目标检测结果，仅用于信息记录)
- 订阅: 共享内存视频流 (用于语义分割)
- 发布: /segmentation/center_offset (赛道中心偏移)
- 发布: /segmentation/is_valid (是否有有效赛道)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import struct
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker
from .ppseg_infer import PPSegInfer
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32, Bool, Float32MultiArray, String


class PerceptionDecisionNode(Node):
    def __init__(self):
        super().__init__('perception_decision_node')
        
        # ==================== 参数声明 ====================
        # 共享内存参数
        self.declare_parameter('shm_name', 'shm_ar_video')
        
        # 语义分割模型参数
        self.declare_parameter('seg_model_dir', 'model')
        self.declare_parameter('seg_model_filename', 'pp_liteseg.rknn')  # ⭐ 默认使用旧版本
        self.declare_parameter('seg_tpes', 3)
        self.declare_parameter('blend_alpha', -1.0)
        self.declare_parameter('enable_flip', True)
        self.declare_parameter('flip_code', 0)
        self.declare_parameter('input_format', 'RGB')
        self.declare_parameter('model_input_format', 'RGB')
        
        # 显示参数
        self.declare_parameter('show_window', False)
        self.declare_parameter('show_debug_window', False)
        self.declare_parameter('enable_perf_stats', False)
        
        # 岔路口检测参数（旧逻辑，已弃用）
        self.declare_parameter('enable_intersection_logic', True)
        # ⭐ outer_side 已在高级逻辑中声明，此处不再重复
        self.declare_parameter('far_roi_y0_ratio', 0.35)
        self.declare_parameter('far_roi_y1_ratio', 0.75)
        self.declare_parameter('near_roi_y0_ratio', 0.75)
        self.declare_parameter('near_roi_y1_ratio', 0.95)
        self.declare_parameter('far_width_threshold', 0.65)
        self.declare_parameter('near_width_threshold', 0.60)
        self.declare_parameter('exit_width_threshold', 0.45)
        self.declare_parameter('exit_confirm_frames', 5)
        self.declare_parameter('max_lock_time', 2.0)
        self.declare_parameter('min_road_pixels', 300)
        
        # ⭐ GuideBoard 岔路选择参数
        self.declare_parameter('enable_guideboard_branch_selection', True)
        self.declare_parameter('guideboard_branch', 'right')
        self.declare_parameter('guideboard_detect_y0_ratio', 0.2)  # GuideBoard检测起始y比例
        self.declare_parameter('guideboard_detect_y1_ratio', 0.7)  # GuideBoard检测结束y比例
        
        # ⭐ 分支掩码屏蔽比例参数
        self.declare_parameter('branch_mask_ratio', 0.60)  # 默认屏蔽60%区域
        
        # ⭐ 高级岔路检测与中心线拟合参数
        self.declare_parameter('enable_segment_branch_logic', True)
        # Band 扫描参数
        self.declare_parameter('band_count', 8)
        self.declare_parameter('band_y_min_ratio', 0.25)
        self.declare_parameter('band_y_max_ratio', 0.95)
        self.declare_parameter('band_height_ratio', 0.04)
        # Segment 提取参数
        self.declare_parameter('min_segment_width_px', 25)
        self.declare_parameter('min_segment_gap_px', 40)
        self.declare_parameter('min_pixels_per_band', 80)
        # 岔路确认参数
        self.declare_parameter('branch_detect_min_bands', 3)
        self.declare_parameter('branch_detect_far_band_ratio', 0.6)
        # 分支选择参数
        self.declare_parameter('outer_side', 'left')
        self.declare_parameter('branch_lock_time', 2.0)
        self.declare_parameter('min_branch_lock_time', 0.8)
        self.declare_parameter('exit_single_path_confirm_frames', 5)
        self.declare_parameter('exit_single_path_min_ratio', 0.8)
        # 中心线拟合参数
        self.declare_parameter('fit_min_points', 4)
        self.declare_parameter('fit_order', 1)
        self.declare_parameter('use_heading_term', True)
        self.declare_parameter('heading_weight', 0.35)
        self.declare_parameter('near_offset_weight', 0.65)
        # 安全参数
        self.declare_parameter('max_offset_jump', 0.6)
        self.declare_parameter('offset_smoothing_alpha', 0.4)
        # 调试参数
        self.declare_parameter('publish_debug_info', True)
        self.declare_parameter('show_branch_debug', False)
        
        # 获取参数
        self.shm_name = self.get_parameter('shm_name').get_parameter_value().string_value
        seg_model_dir = self.get_parameter('seg_model_dir').get_parameter_value().string_value
        seg_model_filename = self.get_parameter('seg_model_filename').get_parameter_value().string_value
        seg_tpes = self.get_parameter('seg_tpes').get_parameter_value().integer_value
        blend_alpha_param = self.get_parameter('blend_alpha').get_parameter_value().double_value
        self.enable_flip = self.get_parameter('enable_flip').get_parameter_value().bool_value
        self.flip_code = self.get_parameter('flip_code').get_parameter_value().integer_value
        self.input_format = self.get_parameter('input_format').get_parameter_value().string_value
        self.model_input_format = self.get_parameter('model_input_format').get_parameter_value().string_value
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self.show_debug_window = self.get_parameter('show_debug_window').get_parameter_value().bool_value
        self.enable_perf_stats = self.get_parameter('enable_perf_stats').get_parameter_value().bool_value
        
        self.get_logger().info(f'📺 show_window={self.show_window}, show_debug_window={self.show_debug_window}')
        
        # 岔路口参数（旧逻辑，已弃用）
        self.enable_intersection_logic = self.get_parameter('enable_intersection_logic').get_parameter_value().bool_value
        # ⭐ outer_side 已在高级逻辑中获取，此处不再重复
        self.far_roi_y0_ratio = self.get_parameter('far_roi_y0_ratio').get_parameter_value().double_value
        self.far_roi_y1_ratio = self.get_parameter('far_roi_y1_ratio').get_parameter_value().double_value
        self.near_roi_y0_ratio = self.get_parameter('near_roi_y0_ratio').get_parameter_value().double_value
        self.near_roi_y1_ratio = self.get_parameter('near_roi_y1_ratio').get_parameter_value().double_value
        self.far_width_threshold = self.get_parameter('far_width_threshold').get_parameter_value().double_value
        self.near_width_threshold = self.get_parameter('near_width_threshold').get_parameter_value().double_value
        self.exit_width_threshold = self.get_parameter('exit_width_threshold').get_parameter_value().double_value
        self.exit_confirm_frames = self.get_parameter('exit_confirm_frames').get_parameter_value().integer_value
        self.max_lock_time = self.get_parameter('max_lock_time').get_parameter_value().double_value
        self.min_road_pixels = self.get_parameter('min_road_pixels').get_parameter_value().integer_value
        
        # ⭐ GuideBoard 岔路选择参数
        self.enable_guideboard_branch_selection = self.get_parameter('enable_guideboard_branch_selection').get_parameter_value().bool_value
        self.guideboard_branch = self.get_parameter('guideboard_branch').get_parameter_value().string_value
        self.guideboard_detect_y0_ratio = self.get_parameter('guideboard_detect_y0_ratio').get_parameter_value().double_value
        self.guideboard_detect_y1_ratio = self.get_parameter('guideboard_detect_y1_ratio').get_parameter_value().double_value
        
        # ⭐ 分支掩码屏蔽比例参数
        self.branch_mask_ratio = self.get_parameter('branch_mask_ratio').get_parameter_value().double_value
        
        # ⭐ 高级岔路检测与中心线拟合参数获取
        self.enable_segment_branch_logic = self.get_parameter('enable_segment_branch_logic').get_parameter_value().bool_value
        self.band_count = self.get_parameter('band_count').get_parameter_value().integer_value
        self.band_y_min_ratio = self.get_parameter('band_y_min_ratio').get_parameter_value().double_value
        self.band_y_max_ratio = self.get_parameter('band_y_max_ratio').get_parameter_value().double_value
        self.band_height_ratio = self.get_parameter('band_height_ratio').get_parameter_value().double_value
        self.min_segment_width_px = self.get_parameter('min_segment_width_px').get_parameter_value().integer_value
        self.min_segment_gap_px = self.get_parameter('min_segment_gap_px').get_parameter_value().integer_value
        self.min_pixels_per_band = self.get_parameter('min_pixels_per_band').get_parameter_value().integer_value
        self.branch_detect_min_bands = self.get_parameter('branch_detect_min_bands').get_parameter_value().integer_value
        self.branch_detect_far_band_ratio = self.get_parameter('branch_detect_far_band_ratio').get_parameter_value().double_value
        self.outer_side = self.get_parameter('outer_side').get_parameter_value().string_value
        self.branch_lock_time = self.get_parameter('branch_lock_time').get_parameter_value().double_value
        self.min_branch_lock_time = self.get_parameter('min_branch_lock_time').get_parameter_value().double_value
        self.exit_single_path_confirm_frames = self.get_parameter('exit_single_path_confirm_frames').get_parameter_value().integer_value
        self.exit_single_path_min_ratio = self.get_parameter('exit_single_path_min_ratio').get_parameter_value().double_value
        self.fit_min_points = self.get_parameter('fit_min_points').get_parameter_value().integer_value
        self.fit_order = self.get_parameter('fit_order').get_parameter_value().integer_value
        self.use_heading_term = self.get_parameter('use_heading_term').get_parameter_value().bool_value
        self.heading_weight = self.get_parameter('heading_weight').get_parameter_value().double_value
        self.near_offset_weight = self.get_parameter('near_offset_weight').get_parameter_value().double_value
        self.max_offset_jump = self.get_parameter('max_offset_jump').get_parameter_value().double_value
        self.offset_smoothing_alpha = self.get_parameter('offset_smoothing_alpha').get_parameter_value().double_value
        self.publish_debug_info = self.get_parameter('publish_debug_info').get_parameter_value().bool_value
        self.show_branch_debug = self.get_parameter('show_branch_debug').get_parameter_value().bool_value
        
        # 处理 blend_alpha
        self.blend_alpha = None if blend_alpha_param < 0 else blend_alpha_param
        self.show_visualization = self.show_window or self.show_debug_window
        
        self.SHM_HEADER_SIZE = 16
        self.last_fid = 0
        
        # FPS 统计
        self.fps_t = time.time()
        self.fps_n = 0
        self.cur_fps = 0.0
        
        # ==================== 初始化语义分割推理器 ====================
        try:
            self.seg_infer = PPSegInfer(
                model_dir=seg_model_dir,
                model_filename=seg_model_filename,  # ⭐ 从YAML读取的模型文件名
                TPEs=seg_tpes, 
                blend_alpha=self.blend_alpha,
                show_visualization=self.show_visualization,
                input_format=self.input_format,
                model_input_format=self.model_input_format
            )
            self.get_logger().info('✅ Semantic Segmentation model initialized')
            self.get_logger().info(f'   📦 Model: {seg_model_dir}/{seg_model_filename}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to initialize segmentation model: {e}')
            raise
        
        # ==================== 订阅者 ====================
        # 订阅目标检测结果
        self.detections_subscription = self.create_subscription(
            Float32MultiArray,
            '/detection/results',
            self.detection_callback,
            10  # ⭐ 使用默认 QoS (RELIABLE)，与 object_detection_node 保持一致
        )
        
        # ==================== 发布者 ====================
        offset_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.offset_publisher = self.create_publisher(
            Float32,
            '/segmentation/center_offset',
            qos_profile=offset_qos_profile
        )
        
        valid_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.is_valid_publisher = self.create_publisher(
            Bool,
            '/segmentation/is_valid',
            qos_profile=valid_qos_profile
        )
        
        # 共享内存对象
        self.shm = None
        
        # 目标检测结果缓存
        self.latest_detections = []
        self.detections_timestamp = 0
        
        # ⭐ 类别标签映射（与 object_detection_node 的 label_list.txt 保持一致）
        self.class_names = [
            'Car',           # 0
            'Human',         # 1
            'Gold',          # 2
            'Go',            # 3
            'Gate',          # 4
            'GuideBoard',    # 5
            'Speed_Limit',   # 6
            'red_light',     # 7
            'yellow_light',  # 8
            'green_light',   # 9
            'Stop'           # 10
        ]
        
        self.get_logger().info(f'📡 Perception Decision Node Ready')
        self.get_logger().info(f'   SHM Name: {self.shm_name}')
        
        # ⭐ 显示岔路口参数配置
        self.get_logger().info(f'   🛣️ Intersection Logic: {self.enable_intersection_logic}')
        self.get_logger().info(f'   🧭 Outer Side: {self.outer_side}')
        self.get_logger().info(f'   🚩 GuideBoard Selection: {self.enable_guideboard_branch_selection} (branch={self.guideboard_branch})')
        self.get_logger().info(f'   🔍 GuideBoard Detect Range: y={self.guideboard_detect_y0_ratio:.1f}-{self.guideboard_detect_y1_ratio:.1f}')
        self.get_logger().info(f'   🛡️ Branch Mask Ratio: {self.branch_mask_ratio:.2f} (屏蔽{int(self.branch_mask_ratio*100)}%区域)')
        self.get_logger().info(f'   🎯 Segment Branch Logic: {self.enable_segment_branch_logic}')
        self.get_logger().info(f'   🔒 Branch Lock: min={self.min_branch_lock_time:.2f}s, max={self.branch_lock_time:.2f}s, exit_ratio={self.exit_single_path_min_ratio:.2f}')
        
        # 岔路口状态机（旧逻辑，已弃用）
        self.intersection_state = 'NORMAL'
        self.decision = 'none'
        
        # ⭐ 当前偏移量（用于可视化）
        self.current_offset = 0.0
        
        # ⭐ 当前应用了掩码的分割图（用于可视化）
        self.masked_seg_map = None
        
        # ⭐ 高级逻辑状态变量（新逻辑）
        self.last_offset = 0.0
        self.branch_locked = False
        self.locked_branch_side = self.outer_side
        self.lock_start_time = None
        self.exit_confirm_count = 0
        self.current_segments = []  # 用于调试绘制
        self.fit_coeffs = None  # ⭐ 保存拟合系数用于可视化
        self.fit_points = []  # ⭐ 保存拟合用的点
        
        # ⭐ 心跳日志参数
        self.heartbeat_interval = 0.5  # 心跳间隔（秒）
        self.last_heartbeat_time = time.time()
        
        # ⭐ 道路情况描述（直行/右转）
        self.driving_direction = "直行"  # 默认直行
        
        # ⭐ far/near ROI 宽度检测结果（用于日志显示）
        self.far_width_ratio = 0.0  # far_roi 道路宽度比例
        self.near_width_ratio = 0.0  # near_roi 道路宽度比例
        
        # 启动主循环定时器
        self.timer = self.create_timer(0.01, self.main_loop)  # 10ms = 100Hz
    
    def detection_callback(self, msg: Float32MultiArray):
        """接收目标检测结果"""
        try:
            data = msg.data
            detections = []
            
            # 解析检测结果 [class_id, confidence, x1, y1, x2, y2, cx, cy] * N
            i = 0
            while i + 7 < len(data):
                class_id = int(data[i])
                confidence = data[i+1]
                x1 = data[i+2]
                y1 = data[i+3]
                x2 = data[i+4]
                y2 = data[i+5]
                cx = data[i+6]
                cy = data[i+7]
                
                # ⭐ 获取类别名称
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'Unknown'
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'cx': cx,
                    'cy': cy,
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx, cy]
                }
                detections.append(detection)
                i += 8
            
            self.latest_detections = detections
            self.detections_timestamp = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Error parsing detections: {e}')
    
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
            # 1. 读取头部
            header = bytes(self.shm.buf[:self.SHM_HEADER_SIZE])
            fid, w, h = struct.unpack('QII', header)
            
            if fid == self.last_fid:
                return
            
            self.last_fid = fid
            
            # 2. 读取数据
            size = w * h * 3
            img_view = np.ndarray((h, w, 3), dtype=np.uint8, 
                                 buffer=self.shm.buf[self.SHM_HEADER_SIZE : self.SHM_HEADER_SIZE+size])
            frame = img_view.copy()
            del img_view
            
            # 3. 预处理
            if self.enable_flip:
                frame = cv2.flip(frame, self.flip_code)
            
            if self.input_format != self.model_input_format:
                if self.input_format == "RGB" and self.model_input_format == "BGR":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif self.input_format == "BGR" and self.model_input_format == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # FPS 统计
            self.fps_n += 1
            if time.time() - self.fps_t >= 1.0:
                self.cur_fps = self.fps_n / (time.time() - self.fps_t)
                self.fps_n = 0
                self.fps_t = time.time()
            
            # 4. 语义分割推理
            seg_frame, seg_map, flag = self.seg_infer.infer(frame)
            
            if flag and seg_map is not None:
                # 5. 决策逻辑
                center_offset, is_valid = self.make_decision(seg_map, h, w)
                
                # ⭐ 保存当前偏移量（用于可视化）
                self.current_offset = center_offset
                
                # 6. 发布结果
                offset_msg = Float32()
                offset_msg.data = center_offset
                self.offset_publisher.publish(offset_msg)
                
                is_valid_msg = Bool()
                is_valid_msg.data = is_valid
                self.is_valid_publisher.publish(is_valid_msg)
                
                # ⭐ 7. 心跳日志（每秒输出一次）
                current_time = time.time()
                if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                    # ⭐ 根据新逻辑的状态显示
                    if self.enable_segment_branch_logic:
                        # 新逻辑：使用 branch_locked 状态
                        road_status = "LOCK" if self.branch_locked else "NORMAL"
                        # 显示分支选择信息
                        branch_info = f"branch={self.locked_branch_side}" if self.branch_locked else "-"
                    else:
                        # 旧逻辑：使用 intersection_state
                        if self.intersection_state == 'NORMAL':
                            road_status = "NORMAL"
                        elif self.intersection_state == 'APPROACH_INTERSECTION':
                            road_status = "APPROACH"
                        elif self.intersection_state == 'LOCK_OUTER_BRANCH':
                            road_status = "LOCK"
                        else:
                            road_status = self.intersection_state
                        branch_info = "-"
                    
                    self.get_logger().info(
                        f'{self.driving_direction} | {road_status} ({branch_info}) | Offset: {center_offset:.3f}'
                    )
                    self.last_heartbeat_time = current_time
                
                # 8. ⭐ 可视化显示（如果启用）
                if self.show_window and seg_frame is not None:
                    # ⭐ 如果有应用了掩码的分割图，使用它进行可视化
                    if self.masked_seg_map is not None:
                        # 将掩码转换为彩色图像用于显示
                        masked_seg_frame = self._create_colored_seg_from_mask(self.masked_seg_map)
                        self.display_visualization(masked_seg_frame, frame, h, w, show_masked=True)
                    else:
                        self.display_visualization(seg_frame, frame, h, w, show_masked=False)
            
        except (ValueError, struct.error, BufferError) as e:
            import traceback
            self.get_logger().error(f'Error processing frame: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')
            self.disconnect_shm()
        except Exception as e:
            import traceback
            self.get_logger().error(f'Unexpected error: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')
    
    def make_decision(self, seg_map, img_h, img_w):
        """
        融合决策逻辑：多层 Band 扫描 + Segment 提取 + 中心线拟合
        """
        if len(seg_map.shape) == 3:
            seg_map = seg_map[:, :, 0]
        
        h, w = seg_map.shape
        current_time = time.time()
        center_offset = 0.0
        is_valid = False
        self.masked_seg_map = None
        
        # ==================== 步骤1: Band 扫描与岔路检测 ====================
        if self.enable_segment_branch_logic:
            bands = self.build_bands(seg_map)
            branch_detected, branch_score = self.detect_branch_from_bands(bands)
            
            # 保存当前 segments 用于可视化
            self.current_segments = []
            for b in bands:
                self.current_segments.append({
                    'y0': b['y0'],
                    'y1': b['y1'],
                    'segments': b['segments']
                })
            
            # 状态机转换
            if not self.branch_locked:
                if branch_detected:
                    target_branch = self.outer_side
                    if self.enable_guideboard_branch_selection and self.check_guideboard_in_far_roi(h, w):
                        target_branch = self.guideboard_branch
                        self.get_logger().info(f'🚩 GuideBoard detected, selecting branch: {target_branch}')
                    
                    if target_branch not in ('left', 'right'):
                        self.get_logger().warn(f'Invalid branch side "{target_branch}", falling back to outer_side={self.outer_side}')
                        target_branch = self.outer_side
                    
                    self.branch_locked = True
                    self.locked_branch_side = target_branch
                    self.lock_start_time = current_time
                    self.exit_confirm_count = 0
                    self.get_logger().info(f'🚩 Branch detected (score={branch_score}), locking to {self.locked_branch_side}')
            else:
                # LOCK_BRANCH 状态下检查退出条件
                single_path_count = 0
                far_bands_count = max(1, int(len(bands) * self.branch_detect_far_band_ratio))
                for b in bands[:far_bands_count]:
                    if len(b['segments']) <= 1:
                        single_path_count += 1
                
                lock_duration = current_time - self.lock_start_time
                single_path_required = max(
                    self.branch_detect_min_bands,
                    int(np.ceil(far_bands_count * self.exit_single_path_min_ratio))
                )
                
                # 先保证最短锁定时间；之后只有多数远端 band 回到单路径才开始计数退出
                if lock_duration >= self.min_branch_lock_time and single_path_count >= single_path_required:
                    self.exit_confirm_count += 1
                else:
                    self.exit_confirm_count = 0
                
                # 退出条件：连续确认帧数达到阈值 或 锁定时间超时
                if (lock_duration >= self.min_branch_lock_time and
                    self.exit_confirm_count >= self.exit_single_path_confirm_frames) or \
                   (lock_duration > self.branch_lock_time):
                    self.branch_locked = False
                    self.locked_branch_side = self.outer_side
                    self.exit_confirm_count = 0
                    self.get_logger().info('✅ Branch lock released')
            
            # ==================== 步骤2: 收集点并拟合 ====================
            target_side = self.locked_branch_side if self.branch_locked else self.outer_side
            points = self.collect_centerline_points(bands, self.branch_locked, target_side, 
                                                    last_center_x=(self.last_offset * w/2 + w/2))
            raw_offset, coeffs = self.fit_centerline_and_compute_offset(points, h, w)
            
            # ⭐ 保存拟合结果用于可视化
            self.fit_coeffs = coeffs
            self.fit_points = points
            
            if raw_offset is not None:
                center_offset = self.smooth_offset(raw_offset)
                is_valid = True
            else:
                # 保底逻辑：回退到简单底部 ROI 计算
                bottom_seg = seg_map[int(h*0.8):h, :]
                center_offset = self._calculate_center_offset(bottom_seg)
                is_valid = bool(np.any(bottom_seg == 1))
                
                # 如果保底也无效，保持上一帧的 offset
                if not is_valid and abs(self.last_offset) > 0.01:
                    center_offset = self.last_offset
                    self.get_logger().debug('⚠️ Fit failed, keeping last offset')
                
                # 保底时清空拟合数据
                self.fit_coeffs = None
                self.fit_points = []
                
        else:
            # 旧逻辑兼容（如果关闭高级功能）
            half_h = h // 2
            bottom_seg = seg_map[half_h:h, :]
            center_offset = self._calculate_center_offset(bottom_seg)
            is_valid = bool(np.any(bottom_seg == 1))
            
        return center_offset, is_valid
    
    def _calculate_center_offset(self, bottom_seg):
        """计算赛道中心偏移量"""
        try:
            h, w = bottom_seg.shape
            track_pixels = np.where(bottom_seg == 1)
            
            if len(track_pixels[1]) == 0:
                return 0.0
            
            track_cols = track_pixels[1]
            center_col = np.mean(track_cols)
            center_offset = (center_col - w / 2.0) / (w / 2.0)
            center_offset = np.clip(center_offset, -1.0, 1.0)
            
            return float(center_offset)
        except Exception as e:
            self.get_logger().error(f'Error calculating center offset: {e}')
            return 0.0
    
    def detect_intersection_far(self, road_mask):
        """检测远处岔路"""
        try:
            h, w = road_mask.shape
            y0 = int(h * self.far_roi_y0_ratio)
            y1 = int(h * self.far_roi_y1_ratio)
            roi = road_mask[y0:y1, :]
            
            road_pixels = np.sum(roi == 1)
            if road_pixels < self.min_road_pixels:
                return False, 0.0
            
            cols = np.where(np.any(roi == 1, axis=0))[0]
            if len(cols) == 0:
                return False, 0.0
            
            road_width = (cols.max() - cols.min() + 1) / w
            far_intersection = road_width > self.far_width_threshold
            
            return far_intersection, road_width
        except Exception as e:
            self.get_logger().error(f'Error in detect_intersection_far: {e}')
            return False, 0.0
    
    def detect_intersection_near(self, road_mask):
        """检测近处岔路"""
        try:
            h, w = road_mask.shape
            y0 = int(h * self.near_roi_y0_ratio)
            y1 = int(h * self.near_roi_y1_ratio)
            roi = road_mask[y0:y1, :]
            
            road_pixels = np.sum(roi == 1)
            if road_pixels < self.min_road_pixels:
                return False, 0.0
            
            cols = np.where(np.any(roi == 1, axis=0))[0]
            if len(cols) == 0:
                return False, 0.0
            
            road_width = (cols.max() - cols.min() + 1) / w
            near_intersection = road_width > self.near_width_threshold
            
            return near_intersection, road_width
        except Exception as e:
            self.get_logger().error(f'Error in detect_intersection_near: {e}')
            return False, 0.0
    
    def check_guideboard_in_far_roi(self, img_h, img_w):
        """
        检查在指定区域是否检测到 GuideBoard 目标
        Args:
            img_h: 图像高度
            img_w: 图像宽度
        Returns:
            bool: 是否在指定区域检测到 GuideBoard
        """
        if not self.latest_detections:
            return False
        
        # ⭐ 使用专门的 GuideBoard 检测范围（默认 0.2-0.7）
        y0 = int(img_h * self.guideboard_detect_y0_ratio)
        y1 = int(img_h * self.guideboard_detect_y1_ratio)
        
        # 检查每个检测结果
        for det in self.latest_detections:
            class_name = det.get('class_name', '')
            cy = det.get('cy', 0)  # 目标中心 y 坐标
            
            # 如果是 GuideBoard 且在检测区域内
            if class_name == 'GuideBoard' and y0 <= cy <= y1:
                confidence = det.get('confidence', 0.0)
                self.get_logger().debug(f'🚩 GuideBoard detected at ({det.get("cx", 0)}, {cy}), confidence={confidence:.2f}')
                return True
        
        return False
    
    def apply_branch_mask_for_offset(self, road_mask, target_side):
        """应用分支掩码（增强版 - 覆盖 offset 计算区域）"""
        try:
            h, w = road_mask.shape
            masked = road_mask.copy()
            
            # ⭐ 从图像 80% 高度开始屏蔽，与 offset 计算区域一致
            y0 = int(h * 0.8)  # 从 80% 高度开始
            y1 = h
            
            # ⭐ 使用可配置的屏蔽比例，增强对未分开岔路的屏蔽效果
            if target_side == 'left':
                # 屏蔽右侧区域，保留左侧 (1 - branch_mask_ratio) 的部分
                mask_start = int(w * (1 - self.branch_mask_ratio))
                masked[y0:y1, mask_start:w] = 0
                self.get_logger().debug(f'Branch mask: left side, masking x={mask_start}-{w}, y={y0}-{y1} ({self.branch_mask_ratio*100:.0f}%)')
            elif target_side == 'right':
                # 屏蔽左侧区域，保留右侧 (1 - branch_mask_ratio) 的部分
                mask_end = int(w * self.branch_mask_ratio)
                masked[y0:y1, 0:mask_end] = 0
                self.get_logger().debug(f'Branch mask: right side, masking x=0-{mask_end}, y={y0}-{y1} ({self.branch_mask_ratio*100:.0f}%)')
            
            return masked
        except Exception as e:
            self.get_logger().error(f'Error in apply_branch_mask_for_offset: {e}')
            return road_mask
    
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
            if not self.connect_to_shm():
                if not hasattr(self, '_wait_log_counter'):
                    self._wait_log_counter = 0
                self._wait_log_counter += 1
                if self._wait_log_counter % 1000 == 0:
                    self.get_logger().info('Waiting for server...')
        else:
            self.process_frame()
    
    def destroy_node(self):
        """清理资源"""
        self.get_logger().info('🛑 Shutting down perception decision node...')
        
        # 关闭共享内存
        self.disconnect_shm()
        
        # 释放模型资源
        if hasattr(self, 'seg_infer'):
            self.seg_infer.release()
            self.get_logger().info('🔒 Segmentation model resources released')
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        super().destroy_node()
    
    def _create_colored_seg_from_mask(self, seg_map):
        """
        将分割掩码转换为彩色图像
        Args:
            seg_map: 分割掩码 (H, W)，值为类别索引 (0=背景, 1=赛道)
        Returns:
            colored_seg: 彩色分割图像 (H, W, 3) BGR格式
        """
        try:
            h, w = seg_map.shape
            colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 为每个类别应用颜色（RGB格式）
            SEG_COLORS_RGB = [
                [0, 0, 0],          # 0: 背景 - 黑色
                [0, 128, 255],      # 1: 赛道 - 亮蓝色
            ]
            
            for class_idx, color in enumerate(SEG_COLORS_RGB):
                mask = seg_map == class_idx
                colored_seg[mask] = color
            
            # 转换从 RGB 到 BGR
            colored_seg_bgr = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
            return colored_seg_bgr
        except Exception as e:
            self.get_logger().error(f'Error creating colored seg from mask: {e}')
            return np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)

    def build_bands(self, road_mask):
        """构建扫描 Band"""
        h, w = road_mask.shape
        y_min = int(h * self.band_y_min_ratio)
        y_max = int(h * self.band_y_max_ratio)
        band_height = max(1, int(h * self.band_height_ratio))
        
        bands = []
        step = max(1, (y_max - y_min) // self.band_count)
        
        for i in range(self.band_count):
            y0 = y_min + i * step
            y1 = min(y0 + band_height, y_max)
            if y1 <= y0: continue
            
            band_mask = road_mask[y0:y1, :]
            segments = self.extract_segments_in_band(band_mask)
            
            bands.append({
                'y0': y0,
                'y1': y1,
                'y_center': (y0 + y1) / 2.0,
                'segments': segments
            })
        return bands

    def extract_segments_in_band(self, band_mask):
        """提取单个 Band 内的赛道 Segment"""
        col_has_road = np.any(band_mask == 1, axis=0)
        segments = []
        in_segment = False
        start_x = 0
        pixel_count = 0
        
        for x in range(len(col_has_road)):
            if col_has_road[x]:
                if not in_segment:
                    start_x = x
                    in_segment = True
                    pixel_count = 0
                pixel_count += np.sum(band_mask[:, x])
            else:
                if in_segment:
                    width = x - start_x
                    if width >= self.min_segment_width_px and pixel_count >= self.min_pixels_per_band:
                        segments.append({
                            'x0': start_x,
                            'x1': x - 1,
                            'width': width,
                            'center_x': (start_x + x - 1) / 2.0,
                            'pixel_count': int(pixel_count)
                        })
                    in_segment = False
        
        # 处理末尾的 segment
        if in_segment:
            width = len(col_has_road) - start_x
            if width >= self.min_segment_width_px and pixel_count >= self.min_pixels_per_band:
                segments.append({
                    'x0': start_x,
                    'x1': len(col_has_road) - 1,
                    'width': width,
                    'center_x': (start_x + len(col_has_road) - 1) / 2.0,
                    'pixel_count': int(pixel_count)
                })
        return segments

    def detect_branch_from_bands(self, bands):
        """检测是否存在岔路"""
        far_bands_count = int(len(bands) * self.branch_detect_far_band_ratio)
        far_bands = bands[:far_bands_count] if far_bands_count > 0 else bands
        
        branch_bands = 0
        for b in far_bands:
            valid_segs = []
            for s in b['segments']:
                is_isolated = True
                for other_s in b['segments']:
                    if s is other_s: continue
                    gap = abs(s['center_x'] - other_s['center_x']) - (s['width']/2 + other_s['width']/2)
                    if gap < self.min_segment_gap_px:
                        is_isolated = False
                        break
                if is_isolated:
                    valid_segs.append(s)
            
            if len(valid_segs) >= 2:
                branch_bands += 1
        
        return branch_bands >= self.branch_detect_min_bands, branch_bands

    def choose_target_segment(self, band, outer_side):
        """选择目标分支 Segment"""
        if not band['segments']: return None
        
        if len(band['segments']) == 1:
            # 岔路入口常会被分割成一个连续宽 Segment。LOCK 后不改 mask，
            # 只把拟合目标点推向目标侧，让车辆更早贴向外圈/目标分支。
            seg = band['segments'][0]
            seg_width = seg['x1'] - seg['x0']
            
            if outer_side == 'left':
                # 选择左侧 20% 位置作为中心点
                target_x = seg['x0'] + seg_width * 0.20
            else:
                # 选择右侧 80% 位置作为中心点
                target_x = seg['x0'] + seg_width * 0.80
            
            # 返回一个虚拟的 segment 对象
            return {
                'x0': seg['x0'] if outer_side == 'left' else int(target_x),
                'x1': int(target_x) if outer_side == 'left' else seg['x1'],
                'width': seg_width * 0.4,
                'center_x': target_x,
                'pixel_count': seg.get('pixel_count', 100) // 2
            }
        
        # 多个 Segment 时，选择最左或最右的
        if outer_side == 'left':
            return min(band['segments'], key=lambda s: s['center_x'])
        else:
            return max(band['segments'], key=lambda s: s['center_x'])

    def collect_centerline_points(self, bands, branch_locked, outer_side, last_center_x=None):
        """收集用于拟合的中心点"""
        points = []
        for b in bands:
            target_seg = None
            
            # 如果当前 band 没有 segment，跳过
            if not b['segments']:
                continue
            
            if branch_locked:
                target_seg = self.choose_target_segment(b, outer_side)
            else:
                if len(b['segments']) == 1:
                    target_seg = b['segments'][0]
                elif last_center_x is not None:
                    # ⭐ 安全检查：确保 segments 不为空
                    if b['segments']:
                        target_seg = min(b['segments'], key=lambda s: abs(s['center_x'] - last_center_x))
                else:
                    target_seg = b['segments'][0]
            
            if target_seg:
                points.append((target_seg['center_x'], b['y_center']))
        return points

    def fit_centerline_and_compute_offset(self, points, h, w):
        """拟合中心线并计算 Offset"""
        if len(points) < self.fit_min_points:
            return None, None
        
        ys = np.array([p[1] for p in points])
        xs = np.array([p[0] for p in points])
        
        try:
            coeffs = np.polyfit(ys, xs, self.fit_order)
            near_y = int(h * 0.7)
            near_x = np.polyval(coeffs, near_y)
            
            near_offset = (near_x - w / 2.0) / (w / 2.0)
            heading_error = 0.0
            
            if self.use_heading_term and len(coeffs) > 1:
                # ⭐ 修复：np.polyder 返回的是系数数组，需要用 np.polyval 计算
                deriv_coeffs = np.polyder(coeffs)
                dx_dy = np.polyval(deriv_coeffs, near_y)
                heading_error = np.arctan(dx_dy) / (np.pi / 2) # Normalize to [-1, 1]
            
            final_offset = self.near_offset_weight * near_offset + self.heading_weight * heading_error
            return float(np.clip(final_offset, -1.0, 1.0)), coeffs
        except:
            return None, None

    def smooth_offset(self, raw_offset):
        """Offset 平滑与限幅"""
        if raw_offset is None:
            return self.last_offset
        
        diff = raw_offset - self.last_offset
        if abs(diff) > self.max_offset_jump:
            raw_offset = self.last_offset + np.sign(diff) * self.max_offset_jump
        
        smoothed = self.offset_smoothing_alpha * raw_offset + (1 - self.offset_smoothing_alpha) * self.last_offset
        self.last_offset = smoothed
        return smoothed
    
    def display_visualization(self, seg_frame, original_frame, h, w, show_masked=False):
        """
        显示可视化窗口：语义分割结果 + 目标检测框
        Args:
            seg_frame: 语义分割结果图像（BGR格式）
            original_frame: 原始帧（用于获取尺寸）
            h, w: 图像高度和宽度
            show_masked: 是否显示应用了掩码的分割结果
        """
        try:
            import cv2
            
            # ⭐ 直接在分割结果上绘制（不复制，提高性能）
            display_frame = seg_frame
            
            # ⭐ 绘制 Band 调试信息（如果启用）
            if self.show_branch_debug and hasattr(self, 'current_segments'):
                for i, band_info in enumerate(self.current_segments):
                    y0, y1 = band_info['y0'], band_info['y1']
                    
                    # 绘制 band 矩形框（淡蓝色）
                    cv2.rectangle(display_frame, (0, y0), (w-1, y1), (255, 200, 100), 1)
                    
                    # 绘制每个 segment
                    for seg in band_info['segments']:
                        x0, x1 = int(seg['x0']), int(seg['x1'])
                        center_x = int(seg['center_x'])
                        
                        # Segment 边界（绿色竖线）
                        cv2.line(display_frame, (x0, y0), (x0, y1), (0, 255, 0), 1)
                        cv2.line(display_frame, (x1, y0), (x1, y1), (0, 255, 0), 1)
                        
                        # Segment 中心点（黄色圆点，增大直径）
                        cv2.circle(display_frame, (center_x, int((y0+y1)/2)), 5, (0, 255, 255), -1)
                
                # ⭐ 绘制拟合中心线（红色曲线）
                if hasattr(self, 'fit_coeffs') and self.fit_coeffs is not None and len(self.fit_points) >= 2:
                    # 生成拟合曲线的点
                    ys_fit = np.linspace(self.band_y_min_ratio * h, self.band_y_max_ratio * h, 100)
                    xs_fit = np.polyval(self.fit_coeffs, ys_fit)
                    
                    # 绘制红色曲线，宽度为 3（与增大的中心点直径匹配）
                    for i in range(len(ys_fit) - 1):
                        pt1 = (int(xs_fit[i]), int(ys_fit[i]))
                        pt2 = (int(xs_fit[i+1]), int(ys_fit[i+1]))
                        # 确保点在图像范围内
                        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                            cv2.line(display_frame, pt1, pt2, (0, 0, 255), 3)
            
            # 绘制目标检测结果
            if self.latest_detections:
                for det in self.latest_detections:
                    class_id = det.get('class_id', -1)
                    confidence = det.get('confidence', 0.0)
                    x1 = int(det.get('x1', 0))
                    y1 = int(det.get('y1', 0))
                    x2 = int(det.get('x2', 0))
                    y2 = int(det.get('y2', 0))
                    cx = int(det.get('cx', 0))
                    cy = int(det.get('cy', 0))
                    class_name = det.get('class_name', 'Unknown')
                    
                    # 根据类别选择颜色
                    color_map = {
                        'Human': (0, 255, 0),      # 绿色
                        'Car': (0, 0, 255),         # 红色
                        'Stop': (0, 165, 255),      # 橙色
                        'Gold': (0, 255, 255),      # 黄色
                        'Go': (255, 255, 0),        # 青色
                        'Gate': (255, 0, 255),      # 紫色
                        'red_light': (0, 0, 255),   # 红色
                        'yellow_light': (0, 255, 255), # 黄色
                        'green_light': (0, 255, 0), # 绿色
                    }
                    color = color_map.get(class_name, (255, 255, 255))  # 默认白色
                    
                    # 绘制边界框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label = f"{class_name} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 2
                    
                    # 计算文本大小
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # 绘制文本背景
                    cv2.rectangle(
                        display_frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        color,
                        -1  # 填充
                    )
                    
                    # 绘制文本（黑色）
                    cv2.putText(
                        display_frame,
                        label,
                        (x1, y1 - 5),
                        font,
                        font_scale,
                        (0, 0, 0),  # 黑色文本
                        thickness
                    )
            
            # ⭐ 使用固定的窗口名称，避免创建多个窗口
            window_title = 'Perception Result'
            cv2.imshow(window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF  # 1ms 延迟，允许窗口更新
            
            # 按 'q' 或 ESC 关闭窗口
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                self.show_window = False
                self.get_logger().info('Visualization window closed')
        
        except Exception as e:
            import traceback
            self.get_logger().error(f'Error in display_visualization: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')


def main(args=None):
    rclpy.init(args=args)
    
    node = PerceptionDecisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n🛑 Keyboard interrupt received')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
