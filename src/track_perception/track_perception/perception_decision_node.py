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
from rcl_interfaces.msg import ParameterDescriptor
from dataclasses import asdict, dataclass
import json
import time
import struct
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker
from .ppseg_infer import PPSegInfer
from std_msgs.msg import Float32, Bool, Float32MultiArray, String


@dataclass
class LaneState:
    """Internal lane state; old center_offset/is_valid topics remain the public contract."""
    control_offset: float = 0.0
    lateral_offset: float = 0.0
    heading_error: float = 0.0
    curvature: float = 0.0
    confidence: float = 0.0
    is_valid: bool = False
    road_state: str = 'LOW_CONFIDENCE'
    branch_side: str = ''
    task_state: str = 'CLEAR'
    task_bias: float = 0.0
    timestamp: float = 0.0


class PerceptionDecisionNode(Node):
    def __init__(self):
        super().__init__('perception_decision_node')
        
        # ==================== 参数声明 ====================
        # 共享内存参数
        self.declare_parameter('shm_name', 'shm_ar_video')
        
        # 语义分割模型参数
        self.declare_parameter('seg_model_dir', 'model')
        self.declare_parameter('seg_model_filename', 'pp_liteseg.rknn')  # ⭐ 默认使用旧版本
        self.declare_parameter('seg_model_input_width', 640)
        self.declare_parameter('seg_model_input_height', 480)
        self.declare_parameter('seg_conf_threshold', 0.25)
        self.declare_parameter('seg_mask_threshold', 0.5)
        self.declare_parameter('seg_max_detections', 30)
        self.declare_parameter('seg_tpes', 3)
        self.declare_parameter('seg_core_ids', [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('blend_alpha', -1.0)
        self.declare_parameter('enable_flip', True)
        self.declare_parameter('flip_code', 0)
        self.declare_parameter('input_format', 'RGB')
        self.declare_parameter('model_input_format', 'RGB')
        
        # 显示参数
        self.declare_parameter('show_window', False)
        self.declare_parameter('enable_perf_stats', False)
        
        # ⭐ GuideBoard 岔路选择参数
        self.declare_parameter('enable_guideboard_branch_selection', True)
        self.declare_parameter('guideboard_branch', 'right')
        self.declare_parameter('guideboard_detect_y0_ratio', 0.2)  # GuideBoard检测起始y比例
        self.declare_parameter('guideboard_detect_y1_ratio', 0.7)  # GuideBoard检测结束y比例
        
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
        self.declare_parameter('enable_continuity_branch_selection', False)
        self.declare_parameter('branch_continuity_max_dx_ratio', 0.35)
        self.declare_parameter('branch_continuity_near_band_ratio', 0.5)
        self.declare_parameter('enable_locked_path_continuity', True)
        self.declare_parameter('locked_path_continuity_after_time', 0.5)
        self.declare_parameter('locked_path_continuity_max_dx_ratio', 0.28)
        self.declare_parameter('branch_lock_time', 2.0)
        self.declare_parameter('min_branch_lock_time', 0.8)
        self.declare_parameter('exit_single_path_confirm_frames', 5)
        self.declare_parameter('exit_single_path_min_ratio', 0.8)
        # 合流宽路处理：两条同向道路合成一个宽 segment 时，不取宽块中心。
        self.declare_parameter('enable_merge_wide_segment_logic', True)
        self.declare_parameter('merge_wide_segment_ratio', 1.35)
        self.declare_parameter('merge_wide_min_bands', 2)
        self.declare_parameter('merge_wide_confirm_frames', 2)
        self.declare_parameter('merge_wide_release_frames', 4)
        self.declare_parameter('merge_wide_lane_width_alpha', 0.2)
        # 中心线拟合参数
        self.declare_parameter('fit_min_points', 4)
        self.declare_parameter('fit_order', 1)
        self.declare_parameter('branch_fit_order', 2)
        self.declare_parameter('enable_fit_point_jump_filter', True)
        self.declare_parameter('max_fit_point_dx_ratio', 0.22)
        self.declare_parameter('max_fit_point_dx_px', 140.0)
        self.declare_parameter('enable_obstacle_avoidance', True)
        self.declare_parameter('obstacle_labels', 'Human,Car')
        self.declare_parameter('obstacle_min_confidence', 0.45)
        self.declare_parameter('obstacle_x_margin_px', 45.0)
        self.declare_parameter('obstacle_y_margin_px', 20.0)
        self.declare_parameter('obstacle_max_age', 0.3)
        self.declare_parameter('obstacle_min_bottom_y_ratio', 0.30)
        self.declare_parameter('enable_traffic_light_stop', True)
        self.declare_parameter('traffic_light_min_confidence', 0.45)
        self.declare_parameter('zebra_min_confidence', 0.45)
        self.declare_parameter('zebra_stop_y_ratio', 0.70)
        self.declare_parameter('traffic_light_max_age', 0.5)
        self.declare_parameter('green_light_confirm_frames', 1)
        self.declare_parameter('red_light_confirm_frames', 1)
        self.declare_parameter('enable_finish_stop', True)
        self.declare_parameter('finish_stop_min_confidence', 0.45)
        self.declare_parameter('finish_stop_arm_y_ratio', 0.70)
        self.declare_parameter('finish_stop_lost_frames', 3)
        self.declare_parameter('finish_stop_max_age', 0.5)
        self.declare_parameter('enable_branch_bottom_anchor', True)
        self.declare_parameter('branch_bottom_anchor_x_ratio', 0.5)
        self.declare_parameter('branch_bottom_anchor_y_ratio', 0.98)
        self.declare_parameter('branch_bottom_anchor_weight', 0.6)
        self.declare_parameter('lookahead_y_ratio', 0.7)
        self.declare_parameter('use_heading_term', True)
        self.declare_parameter('heading_weight', 0.35)
        self.declare_parameter('near_offset_weight', 0.65)
        # 安全参数
        self.declare_parameter('max_offset_jump', 0.6)
        self.declare_parameter('offset_smoothing_alpha', 0.4)
        # 调试参数
        self.declare_parameter('show_branch_debug', False)
        self.declare_parameter('enable_status_log', False)
        self.declare_parameter('enable_branch_event_log', False)
        self.declare_parameter('publish_lane_state', True)
        
        # 获取参数
        self.shm_name = self.get_parameter('shm_name').get_parameter_value().string_value
        seg_model_dir = self.get_parameter('seg_model_dir').get_parameter_value().string_value
        seg_model_filename = self.get_parameter('seg_model_filename').get_parameter_value().string_value
        seg_model_input_width = self.get_parameter('seg_model_input_width').get_parameter_value().integer_value
        seg_model_input_height = self.get_parameter('seg_model_input_height').get_parameter_value().integer_value
        seg_conf_threshold = self.get_parameter('seg_conf_threshold').get_parameter_value().double_value
        seg_mask_threshold = self.get_parameter('seg_mask_threshold').get_parameter_value().double_value
        seg_max_detections = self.get_parameter('seg_max_detections').get_parameter_value().integer_value
        seg_tpes = self.get_parameter('seg_tpes').get_parameter_value().integer_value
        seg_core_ids = list(self.get_parameter('seg_core_ids').get_parameter_value().integer_array_value)
        blend_alpha_param = self.get_parameter('blend_alpha').get_parameter_value().double_value
        self.enable_flip = self.get_parameter('enable_flip').get_parameter_value().bool_value
        self.flip_code = self.get_parameter('flip_code').get_parameter_value().integer_value
        self.input_format = self.get_parameter('input_format').get_parameter_value().string_value
        self.model_input_format = self.get_parameter('model_input_format').get_parameter_value().string_value
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self.enable_perf_stats = self.get_parameter('enable_perf_stats').get_parameter_value().bool_value
        
        self.get_logger().info(f'📺 show_window={self.show_window}')
        
        # ⭐ GuideBoard 岔路选择参数
        self.enable_guideboard_branch_selection = self.get_parameter('enable_guideboard_branch_selection').get_parameter_value().bool_value
        self.guideboard_branch = self.get_parameter('guideboard_branch').get_parameter_value().string_value
        self.guideboard_detect_y0_ratio = self.get_parameter('guideboard_detect_y0_ratio').get_parameter_value().double_value
        self.guideboard_detect_y1_ratio = self.get_parameter('guideboard_detect_y1_ratio').get_parameter_value().double_value
        
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
        self.enable_continuity_branch_selection = self.get_parameter('enable_continuity_branch_selection').get_parameter_value().bool_value
        self.branch_continuity_max_dx_ratio = self.get_parameter('branch_continuity_max_dx_ratio').get_parameter_value().double_value
        self.branch_continuity_near_band_ratio = self.get_parameter('branch_continuity_near_band_ratio').get_parameter_value().double_value
        self.enable_locked_path_continuity = self.get_parameter('enable_locked_path_continuity').get_parameter_value().bool_value
        self.locked_path_continuity_after_time = self.get_parameter('locked_path_continuity_after_time').get_parameter_value().double_value
        self.locked_path_continuity_max_dx_ratio = self.get_parameter('locked_path_continuity_max_dx_ratio').get_parameter_value().double_value
        self.branch_lock_time = self.get_parameter('branch_lock_time').get_parameter_value().double_value
        self.min_branch_lock_time = self.get_parameter('min_branch_lock_time').get_parameter_value().double_value
        self.exit_single_path_confirm_frames = self.get_parameter('exit_single_path_confirm_frames').get_parameter_value().integer_value
        self.exit_single_path_min_ratio = self.get_parameter('exit_single_path_min_ratio').get_parameter_value().double_value
        self.enable_merge_wide_segment_logic = self.get_parameter('enable_merge_wide_segment_logic').get_parameter_value().bool_value
        self.merge_wide_segment_ratio = self.get_parameter('merge_wide_segment_ratio').get_parameter_value().double_value
        self.merge_wide_min_bands = self.get_parameter('merge_wide_min_bands').get_parameter_value().integer_value
        self.merge_wide_confirm_frames = self.get_parameter('merge_wide_confirm_frames').get_parameter_value().integer_value
        self.merge_wide_release_frames = self.get_parameter('merge_wide_release_frames').get_parameter_value().integer_value
        self.merge_wide_lane_width_alpha = self.get_parameter('merge_wide_lane_width_alpha').get_parameter_value().double_value
        self.fit_min_points = self.get_parameter('fit_min_points').get_parameter_value().integer_value
        self.fit_order = self.get_parameter('fit_order').get_parameter_value().integer_value
        self.branch_fit_order = self.get_parameter('branch_fit_order').get_parameter_value().integer_value
        self.enable_fit_point_jump_filter = self.get_parameter('enable_fit_point_jump_filter').get_parameter_value().bool_value
        self.max_fit_point_dx_ratio = self.get_parameter('max_fit_point_dx_ratio').get_parameter_value().double_value
        self.max_fit_point_dx_px = self.get_parameter('max_fit_point_dx_px').get_parameter_value().double_value
        self.enable_obstacle_avoidance = self.get_parameter('enable_obstacle_avoidance').get_parameter_value().bool_value
        obstacle_labels_param = self.get_parameter('obstacle_labels').get_parameter_value().string_value
        self.obstacle_labels = {
            label.strip() for label in obstacle_labels_param.split(',') if label.strip()
        }
        self.obstacle_min_confidence = self.get_parameter('obstacle_min_confidence').get_parameter_value().double_value
        self.obstacle_x_margin_px = self.get_parameter('obstacle_x_margin_px').get_parameter_value().double_value
        self.obstacle_y_margin_px = self.get_parameter('obstacle_y_margin_px').get_parameter_value().double_value
        self.obstacle_max_age = self.get_parameter('obstacle_max_age').get_parameter_value().double_value
        self.obstacle_min_bottom_y_ratio = self.get_parameter('obstacle_min_bottom_y_ratio').get_parameter_value().double_value
        self.enable_traffic_light_stop = self.get_parameter('enable_traffic_light_stop').get_parameter_value().bool_value
        self.traffic_light_min_confidence = self.get_parameter('traffic_light_min_confidence').get_parameter_value().double_value
        self.zebra_min_confidence = self.get_parameter('zebra_min_confidence').get_parameter_value().double_value
        self.zebra_stop_y_ratio = self.get_parameter('zebra_stop_y_ratio').get_parameter_value().double_value
        self.traffic_light_max_age = self.get_parameter('traffic_light_max_age').get_parameter_value().double_value
        self.green_light_confirm_frames = self.get_parameter('green_light_confirm_frames').get_parameter_value().integer_value
        self.red_light_confirm_frames = self.get_parameter('red_light_confirm_frames').get_parameter_value().integer_value
        self.enable_finish_stop = self.get_parameter('enable_finish_stop').get_parameter_value().bool_value
        self.finish_stop_min_confidence = self.get_parameter('finish_stop_min_confidence').get_parameter_value().double_value
        self.finish_stop_arm_y_ratio = self.get_parameter('finish_stop_arm_y_ratio').get_parameter_value().double_value
        self.finish_stop_lost_frames = self.get_parameter('finish_stop_lost_frames').get_parameter_value().integer_value
        self.finish_stop_max_age = self.get_parameter('finish_stop_max_age').get_parameter_value().double_value
        self.enable_branch_bottom_anchor = self.get_parameter('enable_branch_bottom_anchor').get_parameter_value().bool_value
        self.branch_bottom_anchor_x_ratio = self.get_parameter('branch_bottom_anchor_x_ratio').get_parameter_value().double_value
        self.branch_bottom_anchor_y_ratio = self.get_parameter('branch_bottom_anchor_y_ratio').get_parameter_value().double_value
        self.branch_bottom_anchor_weight = self.get_parameter('branch_bottom_anchor_weight').get_parameter_value().double_value
        self.lookahead_y_ratio = self.get_parameter('lookahead_y_ratio').get_parameter_value().double_value
        self.use_heading_term = self.get_parameter('use_heading_term').get_parameter_value().bool_value
        self.heading_weight = self.get_parameter('heading_weight').get_parameter_value().double_value
        self.near_offset_weight = self.get_parameter('near_offset_weight').get_parameter_value().double_value
        self.max_offset_jump = self.get_parameter('max_offset_jump').get_parameter_value().double_value
        self.offset_smoothing_alpha = self.get_parameter('offset_smoothing_alpha').get_parameter_value().double_value
        self.show_branch_debug = self.get_parameter('show_branch_debug').get_parameter_value().bool_value
        self.enable_status_log = self.get_parameter('enable_status_log').get_parameter_value().bool_value
        self.enable_branch_event_log = self.get_parameter('enable_branch_event_log').get_parameter_value().bool_value
        self.publish_lane_state_enabled = self.get_parameter('publish_lane_state').get_parameter_value().bool_value
        
        # 处理 blend_alpha
        self.blend_alpha = None if blend_alpha_param < 0 else blend_alpha_param
        self.show_visualization = self.show_window
        
        self.SHM_HEADER_SIZE = 16
        self.last_fid = 0
        
        # FPS 统计
        self.fps_t = time.time()
        self.fps_n = 0
        self.cur_fps = 0.0
        self.perf_interval = 2.0
        self.last_perf_time = time.time()
        self.perf_window_t = time.time()
        self.processed_frames_window = 0
        self.published_msgs_window = 0
        self.cur_publish_fps = 0.0
        self.last_decision_profile = {}
        
        # ==================== 初始化语义分割推理器 ====================
        try:
            self.seg_infer = PPSegInfer(
                model_dir=seg_model_dir,
                model_filename=seg_model_filename,  # ⭐ 从YAML读取的模型文件名
                TPEs=seg_tpes, 
                blend_alpha=self.blend_alpha,
                show_visualization=self.show_visualization,
                input_format=self.input_format,
                model_input_format=self.model_input_format,
                core_ids=seg_core_ids,
                input_size=(seg_model_input_width, seg_model_input_height),
                conf_threshold=seg_conf_threshold,
                mask_threshold=seg_mask_threshold,
                max_detections=seg_max_detections,
            )
            self.get_logger().info('✅ Semantic Segmentation model initialized')
            self.get_logger().info(f'   📦 Model: {seg_model_dir}/{seg_model_filename}')
            self.get_logger().info(f'   Input Size: {seg_model_input_width}x{seg_model_input_height}')
            self.get_logger().info(
                f'   Seg Thresholds: conf={seg_conf_threshold:.2f}, '
                f'mask={seg_mask_threshold:.2f}, max_det={seg_max_detections}'
            )
            self.get_logger().info(f'   TPEs: {seg_tpes}')
            self.get_logger().info(f'   NPU Core IDs: {seg_core_ids if seg_core_ids else "auto 0/1/2"}')
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
        self.stop_request_publisher = self.create_publisher(
            Bool,
            '/perception/stop_request',
            10
        )
        self.lane_state_publisher = self.create_publisher(
            String,
            '/perception/lane_state',
            10
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
            'Stop',          # 10
            'Rock',          # 11
            'Zebra'          # 12
        ]
        
        self.get_logger().info(f'📡 Perception Decision Node Ready')
        self.get_logger().info(f'   SHM Name: {self.shm_name}')
        
        # ⭐ 显示岔路口参数配置
        self.get_logger().info(f'   🧭 Outer Side: {self.outer_side}')
        self.get_logger().info(f'   🚩 GuideBoard Selection: {self.enable_guideboard_branch_selection} (branch={self.guideboard_branch})')
        self.get_logger().info(f'   🔍 GuideBoard Detect Range: y={self.guideboard_detect_y0_ratio:.1f}-{self.guideboard_detect_y1_ratio:.1f}')
        self.get_logger().info(f'   🎯 Segment Branch Logic: {self.enable_segment_branch_logic}')
        self.get_logger().info(f'   🔒 Branch Lock: min={self.min_branch_lock_time:.2f}s, max={self.branch_lock_time:.2f}s, exit_ratio={self.exit_single_path_min_ratio:.2f}')
        self.get_logger().info(
            f'   🔀 Merge Wide Logic: {self.enable_merge_wide_segment_logic} '
            f'(ratio={self.merge_wide_segment_ratio:.2f}, min_bands={self.merge_wide_min_bands})'
        )
        self.get_logger().info(f'   📈 Fit: normal_order={self.fit_order}, branch_order={self.branch_fit_order}, branch_anchor={self.enable_branch_bottom_anchor}')
        self.get_logger().info(f'   🎯 Lookahead Y Ratio: {self.lookahead_y_ratio:.2f}')
        self.get_logger().info(f'   Perf Stats: {self.enable_perf_stats} (interval={self.perf_interval:.1f}s)')
        
        self.decision = 'none'
        
        # ⭐ 当前偏移量（用于可视化）
        self.current_offset = 0.0
        
        # ⭐ 高级逻辑状态变量（新逻辑）
        self.last_offset = 0.0
        self.branch_locked = False
        self.locked_branch_side = self.outer_side
        self.lock_start_time = None
        self.exit_confirm_count = 0
        self.merge_wide_locked = False
        self.merge_wide_side = None
        self.merge_wide_confirm_count = 0
        self.merge_wide_release_count = 0
        self.band_lane_widths = [None] * max(1, self.band_count)
        self.current_segments = []  # 用于调试绘制
        self.current_obstacle_zones = []  # 用于调试绘制
        self.fit_coeffs = None  # ⭐ 保存拟合系数用于可视化
        self.fit_points = []  # ⭐ 保存拟合用的点
        self.last_lane_state = LaneState()
        self.traffic_light_state = 'CLEAR'
        self.stop_request_active = False
        self.traffic_stop_active = False
        self.finish_stop_active = False
        self.finish_stop_state = 'CLEAR'
        self.finish_stop_lost_count = 0
        self.red_light_confirm_count = 0
        self.green_light_confirm_count = 0
        
        # ⭐ 心跳日志参数
        self.heartbeat_interval = 0.5  # 心跳间隔（秒）
        self.last_heartbeat_time = time.time()
        
        # ⭐ 道路情况描述（直行/右转）
        self.driving_direction = "直行"  # 默认直行
        
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

    def get_recent_detections_for_traffic_light(self):
        """返回未过期的检测结果；过期时不更新交通灯状态。"""
        if not self.latest_detections:
            return []
        if self.traffic_light_max_age > 0 and time.time() - self.detections_timestamp > self.traffic_light_max_age:
            return []
        return self.latest_detections

    def update_traffic_light_stop_state(self, image_height):
        """红灯+斑马线停车状态机。"""
        if not self.enable_traffic_light_stop:
            self.traffic_stop_active = False
            self.stop_request_active = self.finish_stop_active
            self.traffic_light_state = 'CLEAR'
            self.red_light_confirm_count = 0
            self.green_light_confirm_count = 0
            return

        detections = self.get_recent_detections_for_traffic_light()
        has_red = False
        has_green = False
        zebra_reached = False
        zebra_stop_y = float(image_height) * max(0.0, min(1.0, self.zebra_stop_y_ratio))

        for det in detections:
            class_name = det.get('class_name', '')
            confidence = float(det.get('confidence', 0.0))

            if class_name == 'green_light' and confidence >= self.traffic_light_min_confidence:
                has_green = True
            elif class_name == 'red_light' and confidence >= self.traffic_light_min_confidence:
                has_red = True
            elif class_name == 'Zebra' and confidence >= self.zebra_min_confidence:
                zebra_bottom_y = max(float(det.get('y1', 0.0)), float(det.get('y2', 0.0)))
                if zebra_bottom_y >= zebra_stop_y:
                    zebra_reached = True

        if has_green:
            self.green_light_confirm_count += 1
        else:
            self.green_light_confirm_count = 0

        if self.green_light_confirm_count >= max(1, self.green_light_confirm_frames):
            if self.traffic_stop_active and self.enable_branch_event_log:
                self.get_logger().info('🚦 Green light confirmed, releasing stop request')
            self.traffic_light_state = 'CLEAR'
            self.traffic_stop_active = False
            self.stop_request_active = self.finish_stop_active
            self.red_light_confirm_count = 0
            return

        if has_red:
            self.red_light_confirm_count += 1
        else:
            self.red_light_confirm_count = 0

        if self.traffic_light_state == 'WAIT_GREEN':
            self.traffic_stop_active = True
            self.stop_request_active = True
            return

        if self.red_light_confirm_count >= max(1, self.red_light_confirm_frames):
            self.traffic_light_state = 'RED_SEEN'

        if self.traffic_light_state == 'RED_SEEN' and zebra_reached:
            self.traffic_light_state = 'WAIT_GREEN'
            self.traffic_stop_active = True
            self.stop_request_active = True
            if self.enable_branch_event_log:
                self.get_logger().info('🚦 Red light + Zebra reached, requesting stop')
        else:
            self.traffic_stop_active = False
            self.stop_request_active = self.finish_stop_active

    def get_recent_detections_for_finish_stop(self):
        """返回未过期的检测结果；过期时按未看到Stop处理。"""
        if not self.latest_detections:
            return []
        if self.finish_stop_max_age > 0 and time.time() - self.detections_timestamp > self.finish_stop_max_age:
            return []
        return self.latest_detections

    def update_finish_stop_state(self, image_height):
        """终点Stop：接近后允许压过，Stop消失后永久停车。"""
        if not self.enable_finish_stop:
            self.finish_stop_active = False
            self.finish_stop_state = 'CLEAR'
            self.finish_stop_lost_count = 0
            self.stop_request_active = self.traffic_stop_active
            return

        if self.finish_stop_active:
            self.stop_request_active = True
            return

        detections = self.get_recent_detections_for_finish_stop()
        stop_seen = False
        stop_reached = False
        arm_y = float(image_height) * max(0.0, min(1.0, self.finish_stop_arm_y_ratio))

        for det in detections:
            if det.get('class_name') != 'Stop':
                continue
            if float(det.get('confidence', 0.0)) < self.finish_stop_min_confidence:
                continue

            stop_seen = True
            stop_bottom_y = max(float(det.get('y1', 0.0)), float(det.get('y2', 0.0)))
            if stop_bottom_y >= arm_y:
                stop_reached = True

        if self.finish_stop_state == 'CLEAR':
            if stop_seen:
                self.finish_stop_state = 'STOP_SEEN'
                self.finish_stop_lost_count = 0

        if self.finish_stop_state == 'STOP_SEEN':
            if stop_reached:
                self.finish_stop_state = 'STOP_ARMED'
                self.finish_stop_lost_count = 0
            elif not stop_seen:
                self.finish_stop_state = 'CLEAR'
                self.finish_stop_lost_count = 0

        elif self.finish_stop_state == 'STOP_ARMED':
            if stop_seen:
                self.finish_stop_lost_count = 0
            else:
                self.finish_stop_lost_count += 1
                if self.finish_stop_lost_count >= max(1, self.finish_stop_lost_frames):
                    self.finish_stop_state = 'FINISH_STOP'
                    self.finish_stop_active = True
                    if self.enable_branch_event_log:
                        self.get_logger().info('🏁 Finish Stop passed, requesting final stop')

        self.stop_request_active = self.traffic_stop_active or self.finish_stop_active

    def get_task_state(self):
        """Keep task state compact; future obstacle/coin logic should extend this."""
        if self.finish_stop_active:
            return 'FINISH_STOP'
        if self.traffic_stop_active:
            return 'TRAFFIC_STOP'
        return 'CLEAR'

    def publish_lane_state(self, lane_state: LaneState):
        """Publish rich lane diagnostics without changing existing control/UI topics."""
        if not self.publish_lane_state_enabled:
            return
        msg = String()
        msg.data = json.dumps(asdict(lane_state), ensure_ascii=False)
        self.lane_state_publisher.publish(msg)
    
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
            current_time = time.time()
            need_perf_stats = (
                self.enable_perf_stats and
                current_time - self.last_perf_time >= self.perf_interval
            )
            t_start = None
            t_header_done = None
            t_copy_done = None
            t_preprocess_done = None
            t_infer_start = None
            t_infer_done = None
            t_decision_done = None
            t_publish_done = None
            t_visualize_done = None
            if need_perf_stats:
                t_start = time.perf_counter()

            # 1. 读取头部
            header = bytes(self.shm.buf[:self.SHM_HEADER_SIZE])
            fid, w, h = struct.unpack('QII', header)
            if need_perf_stats:
                t_header_done = time.perf_counter()
            
            if fid == self.last_fid:
                return
            
            self.last_fid = fid
            
            # 2. 读取数据
            size = w * h * 3
            img_view = np.ndarray((h, w, 3), dtype=np.uint8, 
                                 buffer=self.shm.buf[self.SHM_HEADER_SIZE : self.SHM_HEADER_SIZE+size])
            frame = img_view.copy()
            del img_view
            if need_perf_stats:
                t_copy_done = time.perf_counter()
            
            # 3. 预处理
            if self.enable_flip:
                frame = cv2.flip(frame, self.flip_code)
            
            if self.input_format != self.model_input_format:
                if self.input_format == "RGB" and self.model_input_format == "BGR":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif self.input_format == "BGR" and self.model_input_format == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if need_perf_stats:
                t_preprocess_done = time.perf_counter()
            
            # FPS 统计
            self.fps_n += 1
            self.processed_frames_window += 1
            if time.time() - self.fps_t >= 1.0:
                self.cur_fps = self.fps_n / (time.time() - self.fps_t)
                self.fps_n = 0
                self.fps_t = time.time()
            
            # 4. 语义分割推理
            if need_perf_stats:
                t_infer_start = time.perf_counter()
            seg_frame, seg_map, flag = self.seg_infer.infer(frame)
            if need_perf_stats:
                t_infer_done = time.perf_counter()
            
            if flag and seg_map is not None:
                # 5. 决策逻辑
                lane_state = self.make_decision(seg_map, h, w)
                if need_perf_stats:
                    t_decision_done = time.perf_counter()
                
                # ⭐ 保存当前偏移量（用于可视化）
                self.current_offset = lane_state.control_offset
                self.update_traffic_light_stop_state(h)
                self.update_finish_stop_state(h)
                lane_state.task_state = self.get_task_state()
                lane_state.timestamp = time.time()
                self.last_lane_state = lane_state
                
                # 6. 发布结果
                offset_msg = Float32()
                offset_msg.data = lane_state.control_offset
                self.offset_publisher.publish(offset_msg)
                
                is_valid_msg = Bool()
                is_valid_msg.data = lane_state.is_valid
                self.is_valid_publisher.publish(is_valid_msg)

                stop_request_msg = Bool()
                stop_request_msg.data = self.stop_request_active
                self.stop_request_publisher.publish(stop_request_msg)
                self.publish_lane_state(lane_state)
                self.published_msgs_window += 1
                if need_perf_stats:
                    t_publish_done = time.perf_counter()
                
                # ⭐ 7. 心跳日志（每秒输出一次）
                current_time = time.time()
                if self.enable_status_log and current_time - self.last_heartbeat_time >= self.heartbeat_interval:
                    road_status = lane_state.road_state
                    branch_info = f"branch={self.locked_branch_side}" if self.branch_locked else "-"
                    
                    self.get_logger().info(
                        f'{self.driving_direction} | {road_status} ({branch_info}) | Offset: {lane_state.control_offset:.3f}'
                    )
                    self.last_heartbeat_time = current_time
                
                # 8. ⭐ 可视化显示（如果启用）
                if self.show_window and seg_frame is not None:
                    self.display_visualization(seg_frame, frame, h, w)
                if need_perf_stats:
                    t_visualize_done = time.perf_counter()

            if need_perf_stats:
                t_end = time.perf_counter()
                window_elapsed = max(current_time - self.perf_window_t, 1e-6)
                self.cur_publish_fps = self.published_msgs_window / window_elapsed
                self.log_perf_stats(
                    t_start=t_start,
                    t_header_done=t_header_done,
                    t_copy_done=t_copy_done,
                    t_preprocess_done=t_preprocess_done,
                    t_infer_start=t_infer_start,
                    t_infer_done=t_infer_done,
                    t_decision_done=t_decision_done,
                    t_publish_done=t_publish_done,
                    t_visualize_done=t_visualize_done,
                    t_end=t_end,
                    fid=fid,
                    frame_shape=(w, h),
                    flag=flag,
                    processed_window=self.processed_frames_window,
                    published_window=self.published_msgs_window,
                    window_elapsed=window_elapsed
                )
                self.last_perf_time = current_time
                self.perf_window_t = current_time
                self.processed_frames_window = 0
                self.published_msgs_window = 0
            
        except (ValueError, struct.error, BufferError) as e:
            import traceback
            self.get_logger().error(f'Error processing frame: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')
            self.disconnect_shm()
        except Exception as e:
            import traceback
            self.get_logger().error(f'Unexpected error: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')

    def log_perf_stats(self, t_start, t_header_done, t_copy_done, t_preprocess_done,
                       t_infer_start, t_infer_done, t_decision_done, t_publish_done,
                       t_visualize_done, t_end, fid, frame_shape, flag,
                       processed_window, published_window, window_elapsed):
        """打印 segmentation/decision 单帧端到端耗时和 worker 内部分段耗时。"""
        def ms(a, b):
            if a is None or b is None:
                return 0.0
            return (b - a) * 1000.0

        total_ms = ms(t_start, t_end)
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        worker = self.seg_infer.get_last_profile() if hasattr(self.seg_infer, 'get_last_profile') else {}
        decision_profile = self.last_decision_profile if hasattr(self, 'last_decision_profile') else {}

        self.get_logger().info(
            '⏱️ Segmentation frame profile\n'
            f'   fid={fid}, size={frame_shape[0]}x{frame_shape[1]}, flag={flag}, loop_fps≈{self.cur_fps:.1f}\n'
            f'   window: processed={processed_window}, published={published_window}, publish_fps≈{self.cur_publish_fps:.1f} over {window_elapsed:.2f}s\n'
            f'   node_total:       {total_ms:7.2f} ms ({fps:5.1f} FPS)\n'
            f'   shm_header:       {ms(t_start, t_header_done):7.2f} ms\n'
            f'   shm_copy:         {ms(t_header_done, t_copy_done):7.2f} ms\n'
            f'   node_preprocess:  {ms(t_copy_done, t_preprocess_done):7.2f} ms\n'
            f'   infer_wait_total: {ms(t_infer_start, t_infer_done):7.2f} ms\n'
            f'   decision:         {ms(t_infer_done, t_decision_done):7.2f} ms\n'
            f'   publish:          {ms(t_decision_done, t_publish_done):7.2f} ms\n'
            f'   visualize:        {ms(t_publish_done, t_visualize_done):7.2f} ms\n'
            f'   worker_total:     {worker.get("worker_total_ms", 0.0):7.2f} ms\n'
            f'     worker_pre:     {worker.get("worker_preprocess_ms", 0.0):7.2f} ms\n'
            f'     worker_rknn:    {worker.get("worker_rknn_ms", 0.0):7.2f} ms\n'
            f'     worker_post:    {worker.get("worker_postprocess_ms", 0.0):7.2f} ms\n'
            f'     worker_vis:     {worker.get("worker_visualization_ms", 0.0):7.2f} ms\n'
            f'   decision_total:   {decision_profile.get("total_ms", 0.0):7.2f} ms\n'
            f'     build_bands:    {decision_profile.get("build_bands_ms", 0.0):7.2f} ms\n'
            f'     branch_detect:  {decision_profile.get("branch_detect_ms", 0.0):7.2f} ms\n'
            f'     state_update:   {decision_profile.get("state_update_ms", 0.0):7.2f} ms\n'
            f'     collect_fit:    {decision_profile.get("collect_fit_ms", 0.0):7.2f} ms\n'
            f'     smooth_fallback:{decision_profile.get("smooth_fallback_ms", 0.0):7.2f} ms'
        )
    
    def make_decision(self, seg_map, img_h, img_w):
        """
        融合决策逻辑：多层 Band 扫描 + Segment 提取 + 中心线拟合
        """
        t_total_start = time.perf_counter()
        t_bands_done = t_total_start
        t_branch_done = t_total_start
        t_state_done = t_total_start
        t_fit_done = t_total_start

        if len(seg_map.shape) == 3:
            seg_map = seg_map[:, :, 0]
        
        h, w = seg_map.shape
        current_time = time.time()
        center_offset = 0.0
        lateral_offset = 0.0
        heading_error = 0.0
        curvature = 0.0
        confidence = 0.0
        road_state = 'LOW_CONFIDENCE'
        is_valid = False
        
        # ==================== 步骤1: Band 扫描与岔路检测 ====================
        if self.enable_segment_branch_logic:
            bands = self.build_bands(seg_map)
            t_bands_done = time.perf_counter()
            branch_detected, branch_score = self.detect_branch_from_bands(bands)
            t_branch_done = time.perf_counter()
            
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
                    last_center_x = self.last_offset * w / 2.0 + w / 2.0
                    target_branch = self.outer_side
                    if self.enable_guideboard_branch_selection and self.check_guideboard_in_far_roi(h, w):
                        target_branch = self.guideboard_branch
                        if self.enable_branch_event_log:
                            self.get_logger().info(f'🚩 GuideBoard detected, selecting branch: {target_branch}')
                    elif self.enable_continuity_branch_selection:
                        continuity_branch = self.choose_branch_side_by_continuity(bands, w, last_center_x)
                        if continuity_branch is not None:
                            target_branch = continuity_branch
                            if self.enable_branch_event_log:
                                self.get_logger().info(
                                    f'🚩 Branch detected, selecting continuous branch: {target_branch}'
                                )
                    
                    if target_branch not in ('left', 'right'):
                        self.get_logger().warn(f'Invalid branch side "{target_branch}", falling back to outer_side={self.outer_side}')
                        target_branch = self.outer_side
                    
                    self.branch_locked = True
                    self.locked_branch_side = target_branch
                    self.lock_start_time = current_time
                    self.exit_confirm_count = 0
                    self.merge_wide_locked = False
                    self.merge_wide_side = None
                    self.merge_wide_confirm_count = 0
                    self.merge_wide_release_count = 0
                    if self.enable_branch_event_log:
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
                    if self.enable_branch_event_log:
                        self.get_logger().info('✅ Branch lock released')
            t_state_done = time.perf_counter()
            
            # ==================== 步骤2: 收集点并拟合 ====================
            target_side = self.locked_branch_side if self.branch_locked else self.outer_side
            last_center_x = self.last_offset * w / 2.0 + w / 2.0
            if not self.branch_locked:
                self.update_merge_wide_state(bands, last_center_x)
            road_state = self.get_current_road_state()
            points = self.collect_centerline_points(bands, self.branch_locked, target_side, 
                                                    last_center_x=last_center_x,
                                                    image_width=w)
            fit_order = self.branch_fit_order if self.branch_locked else self.fit_order
            fit_points = self.filter_centerline_points(points, w, last_center_x=last_center_x)
            if self.branch_locked and self.enable_branch_bottom_anchor and fit_order >= 2:
                fit_points.append((
                    w * self.branch_bottom_anchor_x_ratio,
                    h * self.branch_bottom_anchor_y_ratio,
                    self.branch_bottom_anchor_weight
                ))
            raw_offset, coeffs, lateral_offset, heading_error, curvature = self.fit_centerline_and_compute_offset(
                fit_points,
                h,
                w,
                fit_order
            )
            t_fit_done = time.perf_counter()
            
            # ⭐ 保存拟合结果用于可视化
            self.fit_coeffs = coeffs
            self.fit_points = fit_points
            
            if raw_offset is not None:
                center_offset = self.smooth_offset(raw_offset)
                is_valid = True
                confidence = self.calculate_lane_confidence(fit_points, bands)
            else:
                # 保底逻辑：回退到简单底部 ROI 计算
                bottom_seg = seg_map[int(h*0.8):h, :]
                center_offset = self._calculate_center_offset(bottom_seg)
                lateral_offset = center_offset
                is_valid = bool(np.any(bottom_seg == 1))
                confidence = 0.2 if is_valid else 0.0
                
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
            lateral_offset = center_offset
            is_valid = bool(np.any(bottom_seg == 1))
            confidence = 0.2 if is_valid else 0.0
            road_state = 'NORMAL' if is_valid else 'LOW_CONFIDENCE'

        t_end = time.perf_counter()
        self.last_decision_profile = {
            'total_ms': (t_end - t_total_start) * 1000.0,
            'build_bands_ms': (t_bands_done - t_total_start) * 1000.0,
            'branch_detect_ms': (t_branch_done - t_bands_done) * 1000.0,
            'state_update_ms': (t_state_done - t_branch_done) * 1000.0,
            'collect_fit_ms': (t_fit_done - t_state_done) * 1000.0,
            'smooth_fallback_ms': (t_end - t_fit_done) * 1000.0,
        }
        lane_state = LaneState(
            control_offset=float(center_offset),
            lateral_offset=float(lateral_offset),
            heading_error=float(heading_error),
            curvature=float(curvature),
            confidence=float(max(0.0, min(1.0, confidence))),
            is_valid=bool(is_valid),
            road_state=road_state if is_valid else 'LOW_CONFIDENCE',
            branch_side=self.locked_branch_side if self.branch_locked else '',
            task_state='CLEAR',
            task_bias=0.0,
            timestamp=current_time
        )
        return self.apply_task_bias(lane_state)
    
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
    
    def build_bands(self, road_mask):
        """构建扫描 Band"""
        h, w = road_mask.shape
        obstacle_zones = self.get_active_obstacle_zones(w, h)
        self.current_obstacle_zones = obstacle_zones
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
            segments = self.apply_obstacle_exclusion_to_segments(segments, y0, y1, obstacle_zones)
            
            bands.append({
                'index': i,
                'y0': y0,
                'y1': y1,
                'y_center': (y0 + y1) / 2.0,
                'segments': segments
            })
        return bands

    def get_active_obstacle_zones(self, image_width, image_height):
        """将 Human/Car 检测框转换为当前帧的不可通行区间。"""
        if not self.enable_obstacle_avoidance:
            return []
        if not self.latest_detections:
            return []
        if self.obstacle_max_age > 0 and time.time() - self.detections_timestamp > self.obstacle_max_age:
            return []

        zones = []
        min_bottom_y = image_height * max(0.0, min(1.0, self.obstacle_min_bottom_y_ratio))
        x_margin = max(0.0, self.obstacle_x_margin_px)
        y_margin = max(0.0, self.obstacle_y_margin_px)

        for det in self.latest_detections:
            if det.get('class_name') not in self.obstacle_labels:
                continue
            if float(det.get('confidence', 0.0)) < self.obstacle_min_confidence:
                continue

            x1 = float(det.get('x1', 0.0))
            y1 = float(det.get('y1', 0.0))
            x2 = float(det.get('x2', 0.0))
            y2 = float(det.get('y2', 0.0))
            if max(y1, y2) < min_bottom_y:
                continue

            zone = {
                'x0': max(0.0, min(x1, x2) - x_margin),
                'x1': min(float(image_width - 1), max(x1, x2) + x_margin),
                'y0': max(0.0, min(y1, y2) - y_margin),
                'y1': min(float(image_height - 1), max(y1, y2) + y_margin),
                'label': det.get('class_name', 'Obstacle'),
                'confidence': float(det.get('confidence', 0.0)),
            }
            if zone['x1'] > zone['x0'] and zone['y1'] > zone['y0']:
                zones.append(zone)

        return zones

    def apply_obstacle_exclusion_to_segments(self, segments, band_y0, band_y1, obstacle_zones):
        """从赛道 segment 中扣除与障碍物重叠的横向区间。"""
        if not segments or not obstacle_zones:
            return segments

        split_segments = []
        for seg in segments:
            intervals = [(float(seg['x0']), float(seg['x1']))]
            for zone in obstacle_zones:
                if zone['y1'] < band_y0 or zone['y0'] > band_y1:
                    continue

                next_intervals = []
                for x0, x1 in intervals:
                    cut_x0 = max(x0, float(zone['x0']))
                    cut_x1 = min(x1, float(zone['x1']))
                    if cut_x1 < x0 or cut_x0 > x1:
                        next_intervals.append((x0, x1))
                        continue

                    if cut_x0 - x0 >= self.min_segment_width_px:
                        next_intervals.append((x0, cut_x0))
                    if x1 - cut_x1 >= self.min_segment_width_px:
                        next_intervals.append((cut_x1, x1))
                intervals = next_intervals
                if not intervals:
                    break

            for x0, x1 in intervals:
                width = int(round(x1 - x0))
                if width < self.min_segment_width_px:
                    continue
                split_segments.append({
                    'x0': int(round(x0)),
                    'x1': int(round(x1)),
                    'width': width,
                    'center_x': (x0 + x1) / 2.0,
                    'pixel_count': max(self.min_pixels_per_band, int(seg.get('pixel_count', self.min_pixels_per_band) * width / max(1, seg['width']))),
                    'obstacle_cut': True,
                })

        return split_segments

    def extract_segments_in_band(self, band_mask):
        """提取单个 Band 内的赛道 Segment"""
        road_pixel_counts = np.count_nonzero(band_mask == 1, axis=0)
        col_has_road = road_pixel_counts > 0
        if not np.any(col_has_road):
            return []

        segments = []

        padded = np.pad(col_has_road.astype(np.int8), (1, 1), constant_values=0)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)

        for start_x, end_x in zip(starts, ends):
            width = int(end_x - start_x)
            pixel_count = int(road_pixel_counts[start_x:end_x].sum())
            if width >= self.min_segment_width_px and pixel_count >= self.min_pixels_per_band:
                x1 = int(end_x - 1)
                segments.append({
                    'x0': int(start_x),
                    'x1': x1,
                    'width': width,
                    'center_x': (start_x + x1) / 2.0,
                    'pixel_count': pixel_count
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

    def choose_branch_side_by_continuity(self, bands, image_width, last_center_x):
        """没有明确路牌时，优先锁定与上一帧中心线连续的分支。"""
        if last_center_x is None or not bands:
            return None

        near_ratio = max(0.1, min(1.0, self.branch_continuity_near_band_ratio))
        near_count = max(1, int(np.ceil(len(bands) * near_ratio)))
        near_bands = bands[-near_count:]
        max_dx = max(1.0, self.branch_continuity_max_dx_ratio * float(image_width))

        best_seg = None
        best_segments = None
        best_dist = None
        for band in reversed(near_bands):
            segments = band.get('segments', [])
            if len(segments) < 2:
                continue

            candidate = min(segments, key=lambda s: abs(float(s['center_x']) - last_center_x))
            dist = abs(float(candidate['center_x']) - last_center_x)
            if dist > max_dx:
                continue

            if best_dist is None or dist < best_dist:
                best_seg = candidate
                best_segments = segments
                best_dist = dist

        if best_seg is None or not best_segments:
            return None

        sorted_segments = sorted(best_segments, key=lambda s: float(s['center_x']))
        best_index = min(
            range(len(sorted_segments)),
            key=lambda i: abs(float(sorted_segments[i]['center_x']) - float(best_seg['center_x']))
        )
        return 'left' if best_index < len(sorted_segments) / 2.0 else 'right'

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

    def choose_locked_segment_by_continuity(self, band, image_width, last_center_x):
        """LOCK 后期优先选择与当前行驶路径连续的 segment，而不是固定左/右侧。"""
        if last_center_x is None or not band['segments']:
            return None

        max_dx = max(1.0, self.locked_path_continuity_max_dx_ratio * float(image_width))
        if len(band['segments']) == 1:
            seg = band['segments'][0]
            if last_center_x < seg['x0'] - max_dx or last_center_x > seg['x1'] + max_dx:
                return None

            target_x = max(float(seg['x0']), min(float(seg['x1']), float(last_center_x)))
            return {
                'x0': int(max(seg['x0'], target_x - seg['width'] * 0.2)),
                'x1': int(min(seg['x1'], target_x + seg['width'] * 0.2)),
                'width': max(1.0, seg['width'] * 0.4),
                'center_x': target_x,
                'pixel_count': seg.get('pixel_count', 100)
            }

        target_seg = min(band['segments'], key=lambda s: abs(float(s['center_x']) - float(last_center_x)))
        if abs(float(target_seg['center_x']) - float(last_center_x)) > max_dx:
            return None
        return target_seg

    def should_use_locked_path_continuity(self):
        if not self.enable_locked_path_continuity or not self.branch_locked or self.lock_start_time is None:
            return False
        return (time.time() - self.lock_start_time) >= self.locked_path_continuity_after_time

    def get_current_road_state(self):
        if self.branch_locked:
            return 'BRANCH_LOCKED'
        if self.merge_wide_locked:
            return 'MERGE_WIDE'
        return 'NORMAL'

    def calculate_lane_confidence(self, fit_points, bands):
        valid_band_count = sum(1 for band in bands if band.get('segments'))
        if valid_band_count <= 0:
            return 0.0
        point_ratio = len(fit_points) / max(1.0, float(valid_band_count))
        return max(0.0, min(1.0, point_ratio))

    def apply_task_bias(self, lane_state: LaneState):
        """Future hook for obstacle repulsion and coin attraction; no bias is applied now."""
        lane_state.task_bias = 0.0
        return lane_state

    def get_band_lane_width(self, band_index):
        if band_index < 0 or band_index >= len(self.band_lane_widths):
            return None
        return self.band_lane_widths[band_index]

    def update_band_lane_width(self, band_index, width):
        if band_index < 0:
            return
        while band_index >= len(self.band_lane_widths):
            self.band_lane_widths.append(None)

        width = float(width)
        if width <= 0.0:
            return

        old_width = self.band_lane_widths[band_index]
        if old_width is None:
            self.band_lane_widths[band_index] = width
            return

        alpha = max(0.0, min(1.0, float(self.merge_wide_lane_width_alpha)))
        self.band_lane_widths[band_index] = (1.0 - alpha) * float(old_width) + alpha * width

    def is_merge_wide_segment(self, band):
        if not self.enable_merge_wide_segment_logic:
            return False
        segments = band.get('segments', [])
        if len(segments) != 1:
            return False

        lane_width = self.get_band_lane_width(int(band.get('index', -1)))
        if lane_width is None or lane_width <= 0.0:
            return False

        seg_width = float(segments[0].get('width', 0.0))
        ratio = max(1.05, float(self.merge_wide_segment_ratio))
        return seg_width >= lane_width * ratio

    def update_merge_wide_state(self, bands, last_center_x):
        """Detect a same-direction merge as a single road segment becoming abnormally wide."""
        if not self.enable_merge_wide_segment_logic:
            self.merge_wide_locked = False
            self.merge_wide_side = None
            return

        wide_bands = []
        for band in bands:
            segments = band.get('segments', [])
            if len(segments) != 1:
                continue

            seg = segments[0]
            band_index = int(band.get('index', -1))
            seg_width = float(seg.get('width', 0.0))
            lane_width = self.get_band_lane_width(band_index)

            if lane_width is None:
                self.update_band_lane_width(band_index, seg_width)
                continue

            if self.is_merge_wide_segment(band):
                wide_bands.append(band)
            elif not self.merge_wide_locked:
                self.update_band_lane_width(band_index, seg_width)

        if len(wide_bands) >= max(1, int(self.merge_wide_min_bands)):
            self.merge_wide_confirm_count += 1
            self.merge_wide_release_count = 0
            if self.merge_wide_confirm_count >= max(1, int(self.merge_wide_confirm_frames)):
                if not self.merge_wide_locked:
                    ref_seg = wide_bands[-1]['segments'][0]
                    self.merge_wide_side = 'left' if last_center_x <= float(ref_seg['center_x']) else 'right'
                    if self.enable_branch_event_log:
                        self.get_logger().info(f'🔀 Merge wide road detected, keeping {self.merge_wide_side} lane')
                self.merge_wide_locked = True
            return

        self.merge_wide_confirm_count = 0
        if self.merge_wide_locked:
            self.merge_wide_release_count += 1
            if self.merge_wide_release_count >= max(1, int(self.merge_wide_release_frames)):
                if self.enable_branch_event_log:
                    self.get_logger().info('🔀 Merge wide road released')
                self.merge_wide_locked = False
                self.merge_wide_side = None
                self.merge_wide_release_count = 0

    def choose_merge_wide_segment(self, band, last_center_x=None):
        if not self.merge_wide_locked or not self.is_merge_wide_segment(band):
            return None

        seg = band['segments'][0]
        lane_width = self.get_band_lane_width(int(band.get('index', -1)))
        if lane_width is None or lane_width <= 0.0:
            return None

        side = self.merge_wide_side
        if side not in ('left', 'right'):
            if last_center_x is None:
                side = self.outer_side if self.outer_side in ('left', 'right') else 'left'
            else:
                side = 'left' if last_center_x <= float(seg['center_x']) else 'right'
            self.merge_wide_side = side

        if side == 'left':
            x0 = float(seg['x0'])
            x1 = min(float(seg['x1']), x0 + lane_width)
            target_x = x0 + lane_width * 0.5
        else:
            x1 = float(seg['x1'])
            x0 = max(float(seg['x0']), x1 - lane_width)
            target_x = x1 - lane_width * 0.5

        target_x = max(float(seg['x0']), min(float(seg['x1']), target_x))
        return {
            'x0': int(round(x0)),
            'x1': int(round(x1)),
            'width': max(1.0, x1 - x0),
            'center_x': target_x,
            'pixel_count': seg.get('pixel_count', 100),
            'merge_wide_virtual': True,
        }

    def collect_centerline_points(self, bands, branch_locked, outer_side, last_center_x=None, image_width=None):
        """收集用于拟合的中心点"""
        points = []
        use_locked_continuity = (
            branch_locked and
            image_width is not None and
            self.should_use_locked_path_continuity()
        )
        for b in bands:
            target_seg = None
            
            # 如果当前 band 没有 segment，跳过
            if not b['segments']:
                continue
            
            if branch_locked:
                if use_locked_continuity:
                    target_seg = self.choose_locked_segment_by_continuity(b, image_width, last_center_x)
                if target_seg is None:
                    target_seg = self.choose_target_segment(b, outer_side)
            else:
                target_seg = self.choose_merge_wide_segment(b, last_center_x=last_center_x)
                if target_seg is None:
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

    def filter_centerline_points(self, points, image_width, last_center_x=None):
        """过滤连续 band 中 x 跳变过大的中心点，避免误分割区域参与拟合。"""
        points = list(points)
        if not self.enable_fit_point_jump_filter or len(points) < 3:
            return points

        ratio_limit = max(0.0, self.max_fit_point_dx_ratio) * float(image_width)
        px_limit = self.max_fit_point_dx_px if self.max_fit_point_dx_px > 0 else ratio_limit
        max_dx = max(1.0, min(ratio_limit, px_limit))

        keep = [True] * len(points)

        # 先剔除夹在两个连续点之间的孤立横向跳点。
        for i in range(1, len(points) - 1):
            prev_x = float(points[i - 1][0])
            cur_x = float(points[i][0])
            next_x = float(points[i + 1][0])
            if (abs(cur_x - prev_x) > max_dx and
                abs(cur_x - next_x) > max_dx and
                abs(next_x - prev_x) <= max_dx):
                keep[i] = False

        # 端点也可能来自画面边缘误分割；只在后续点彼此连续时剔除端点。
        if len(points) >= 3:
            x0 = float(points[0][0])
            x1 = float(points[1][0])
            x2 = float(points[2][0])
            if abs(x0 - x1) > max_dx and abs(x1 - x2) <= max_dx:
                keep[0] = False

            xn0 = float(points[-1][0])
            xn1 = float(points[-2][0])
            xn2 = float(points[-3][0])
            if abs(xn0 - xn1) > max_dx and abs(xn1 - xn2) <= max_dx:
                keep[-1] = False

        filtered = [p for p, should_keep in zip(points, keep) if should_keep]
        if len(filtered) < 2:
            return filtered

        # 如果过滤后仍被大跳变分成多段，只保留最连续的一段。
        chains = []
        current_chain = [filtered[0]]
        for point in filtered[1:]:
            if abs(float(point[0]) - float(current_chain[-1][0])) <= max_dx:
                current_chain.append(point)
            else:
                chains.append(current_chain)
                current_chain = [point]
        chains.append(current_chain)

        if len(chains) == 1:
            return filtered

        def chain_score(chain):
            length_score = len(chain)
            bottom_score = max(float(p[1]) for p in chain)
            if last_center_x is None:
                continuity_score = 0.0
            else:
                continuity_score = -min(abs(float(p[0]) - float(last_center_x)) for p in chain)
            return (length_score, bottom_score, continuity_score)

        return list(max(chains, key=chain_score))

    def fit_centerline_and_compute_offset(self, points, h, w, fit_order=None):
        """拟合中心线并计算 Offset"""
        if len(points) < self.fit_min_points:
            return None, None, 0.0, 0.0, 0.0
        
        order = self.fit_order if fit_order is None else fit_order
        if len(points) <= order:
            return None, None, 0.0, 0.0, 0.0
        
        ys = np.array([p[1] for p in points])
        xs = np.array([p[0] for p in points])
        weights = np.array([p[2] if len(p) > 2 else 1.0 for p in points])
        
        try:
            coeffs = np.polyfit(ys, xs, order, w=weights)
            near_y = int(h * self.lookahead_y_ratio)
            near_x = np.polyval(coeffs, near_y)
            
            near_offset = (near_x - w / 2.0) / (w / 2.0)
            heading_error = 0.0
            curvature = 0.0
            
            if self.use_heading_term and len(coeffs) > 1:
                # ⭐ 修复：np.polyder 返回的是系数数组，需要用 np.polyval 计算
                deriv_coeffs = np.polyder(coeffs)
                dx_dy = np.polyval(deriv_coeffs, near_y)
                heading_error = np.arctan(dx_dy) / (np.pi / 2) # Normalize to [-1, 1]
                if len(coeffs) > 2:
                    second_deriv_coeffs = np.polyder(coeffs, 2)
                    d2x_dy2 = np.polyval(second_deriv_coeffs, near_y)
                    curvature = np.clip(float(d2x_dy2) * float(h), -1.0, 1.0)
            
            final_offset = self.near_offset_weight * near_offset + self.heading_weight * heading_error
            return (
                float(np.clip(final_offset, -1.0, 1.0)),
                coeffs,
                float(np.clip(near_offset, -1.0, 1.0)),
                float(np.clip(heading_error, -1.0, 1.0)),
                float(curvature)
            )
        except:
            return None, None, 0.0, 0.0, 0.0

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
    
    def display_visualization(self, seg_frame, original_frame, h, w):
        """
        显示可视化窗口：语义分割结果 + 目标检测框
        Args:
            seg_frame: 语义分割结果图像（BGR格式）
            original_frame: 原始帧（用于获取尺寸）
            h, w: 图像高度和宽度
        """
        try:
            import cv2
            
            # ⭐ 直接在分割结果上绘制（不复制，提高性能）
            display_frame = seg_frame
            
            # ⭐ 绘制 Band 调试信息（如果启用）
            if self.show_branch_debug and hasattr(self, 'current_segments'):
                if getattr(self, 'current_obstacle_zones', None):
                    for zone in self.current_obstacle_zones:
                        cv2.rectangle(
                            display_frame,
                            (int(zone['x0']), int(zone['y0'])),
                            (int(zone['x1']), int(zone['y1'])),
                            (0, 0, 255),
                            2
                        )
                        cv2.putText(
                            display_frame,
                            f"avoid:{zone.get('label', '')}",
                            (int(zone['x0']), max(0, int(zone['y0']) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 255),
                            1
                        )

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
                        cv2.circle(display_frame, (center_x, int((y0+y1)/2)), 3, (0, 255, 255), -1)
                
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
                            cv2.line(display_frame, pt1, pt2, (0, 0, 255), 2)
            
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
