#!/usr/bin/env python3
"""
巡线控制节点
订阅语义分割的center_offset和is_valid，发布/cmd_vel控制底盘
使用PD控制器实现赛道跟随
"""

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Bool, String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
import json
import math
import time


class LineFollowerController(Node):
    def __init__(self):
        super().__init__('line_follower_controller')
        
        # ==================== 参数声明 ====================
        # 标准PID参数
        self.declare_parameter('Kp', 0.78)          # 比例增益
        self.declare_parameter('Ki', 0.05)          # 积分增益
        self.declare_parameter('Kd', 0.6)           # 微分增益
        
        # 积分限幅
        self.declare_parameter('integral_max', 1.0)  # 积分项最大值
        
        # 车辆参数
        self.declare_parameter('linear_speed', 0.3)      # 前进线速度 (m/s)
        self.declare_parameter('wheel_radius', 0.035)    # 轮子半径 (m)，默认3.5cm
        self.declare_parameter('max_steering', 1.0)      # 最大转向比例
        self.declare_parameter('steering_slew_rate', 0.0)  # 最大转向变化率，0表示关闭
        self.declare_parameter('invalid_timeout', 0.5)   # is_valid=False超时时间(秒)
        self.declare_parameter('enable_perception_stop', True)
        self.declare_parameter('ignore_stop_requests', False)
        self.declare_parameter('perception_stop_timeout', 0.5)
        self.declare_parameter('autonomous_enabled_on_start', False)
        self.declare_parameter('publish_stop_when_disabled', True)
        self.declare_parameter('enable_manual_override', True)
        self.declare_parameter('use_lane_state', True)
        self.declare_parameter('lane_state_timeout', 0.3)
        self.declare_parameter('heading_gain', 0.0)
        self.declare_parameter('curvature_gain', 0.0)
        self.declare_parameter('enable_start_boost', True)
        self.declare_parameter('start_boost_task_state', 'START_BOOST')
        self.declare_parameter('start_boost_duration', 0.8)
        self.declare_parameter('start_boost_speed', 0.90)
        self.declare_parameter('start_boost_steering', 0.0)
        
        # 日志参数
        self.declare_parameter('controller_log_mode', 'normal')  # normal, pid_tuning, off
        self.declare_parameter('pid_tuning_log_hz', 10.0)        # PID调参日志频率
        self.declare_parameter('pid_tuning_bar_width', 41)       # 误差条宽度，建议使用奇数
        self.declare_parameter('controller_debug_hz', 1.0)
        
        self.load_parameters()
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # ==================== 状态变量 ====================
        self.current_offset = 0.0          # 当前偏移量
        self.current_lateral_offset = 0.0
        self.current_heading_error = 0.0
        self.current_curvature = 0.0
        self.lane_confidence = 0.0
        self.road_state = 'UNKNOWN'
        self.task_state = 'CLEAR'
        self.last_lane_state_time = None
        self.prev_offset = 0.0             # 上一帧偏移量（用于计算微分）
        self.integral = 0.0                # 积分项累积
        self.prev_time = time.time()       # 上一帧时间
        self.is_valid = True               # 是否有赛道
        self.invalid_start_time = None     # is_valid=False的开始时间
        self.last_pid_terms = (0.0, 0.0, 0.0)
        self.last_lane_terms = (0.0, 0.0)
        self.last_steering = 0.0
        self.last_tuning_log_time = 0.0
        self.track_lost_logged = False
        self.perception_stop_active = False
        self.last_perception_stop_time = None
        self.perception_stop_logged = False
        self.autonomous_enabled = bool(self.autonomous_enabled_on_start)
        self.emergency_stop_active = False
        self.manual_override_active = False
        self.disabled_stop_logged = False
        self.manual_override_logged = False
        self.start_boost_active = False
        self.start_boost_used = False
        self.start_boost_start_time = None
        self.start_boost_logged = False
        self.last_control_mode = 'init'
        self.debug_window_start = time.time()
        self.debug_loop_count = 0
        self.debug_cmd_count = 0
        self.debug_stop_count = 0
        self.debug_skip_count = 0
        self.debug_dt_sum = 0.0
        self.debug_dt_max = 0.0
        
        # ==================== 订阅者 ====================
        # ⭐ 使用与segmentation_node相同的QoS配置
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 订阅center_offset（赛道中心偏移）
        self.offset_subscription = self.create_subscription(
            Float32,
            '/segmentation/center_offset',
            self.offset_callback,
            sensor_qos
        )
        
        # 订阅is_valid（是否有赛道）
        self.valid_subscription = self.create_subscription(
            Bool,
            '/segmentation/is_valid',
            self.valid_callback,
            sensor_qos
        )
        self.lane_state_subscription = self.create_subscription(
            String,
            '/perception/lane_state',
            self.lane_state_callback,
            10
        )
        self.perception_stop_subscription = self.create_subscription(
            Bool,
            '/perception/stop_request',
            self.perception_stop_callback,
            10
        )
        self.emergency_stop_subscription = self.create_subscription(
            Bool,
            '/race/emergency_stop',
            self.emergency_stop_callback,
            10
        )
        self.manual_override_subscription = self.create_subscription(
            Bool,
            '/race/manual_override',
            self.manual_override_callback,
            10
        )
        self.enabled_service = self.create_service(
            SetBool,
            '/line_follower/set_enabled',
            self.set_enabled_callback
        )
        
        # ==================== 发布者 ====================
        # 发布/cmd_vel到chassis_controller
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.debug_publisher = self.create_publisher(
            String,
            '/line_follower/debug',
            10
        )
        
        # ==================== 定时器 ====================
        # 以50Hz频率发布控制指令
        self.control_timer = self.create_timer(0.02, self.control_loop)
        self.debug_timer = self.create_timer(
            max(0.1, 1.0 / max(0.1, self.controller_debug_hz)),
            self.publish_debug_status
        )
        
        # 打印配置信息
        self.get_logger().info('🚗 Line Follower Controller Initialized (Standard PID)')
        self.get_logger().info(f'   PID Gains: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}')
        self.get_logger().info(f'   Integral Max: {self.integral_max}')
        self.get_logger().info(f'   Linear Speed: {self.linear_speed_mps} m/s')
        self.get_logger().info(f'   Wheel Radius: {self.wheel_radius} m')
        self.get_logger().info(f'   Wheel Speed: {self.wheel_speed_rps:.2f} rps (revolutions per second)')
        self.get_logger().info(f'   Max Steering: {self.max_steering}')
        self.get_logger().info(f'   Invalid Timeout: {self.invalid_timeout} s')
        self.get_logger().info(
            f'   Perception Stop: {self.enable_perception_stop} '
            f'(ignore={self.ignore_stop_requests}, timeout={self.perception_stop_timeout:.2f}s)'
        )
        self.get_logger().info(
            f'   Autonomous Enabled: {self.autonomous_enabled}, '
            f'publish_stop_when_disabled={self.publish_stop_when_disabled}, '
            f'enable_manual_override={self.enable_manual_override}'
        )
        self.get_logger().info(
            f'   Lane State: use={self.use_lane_state}, timeout={self.lane_state_timeout:.2f}s, '
            f'heading_gain={self.heading_gain}, curvature_gain={self.curvature_gain}'
        )
        self.get_logger().info(
            f'   Start Boost: enable={self.enable_start_boost}, state={self.start_boost_task_state}, '
            f'duration={self.start_boost_duration:.2f}s, speed={self.start_boost_speed_mps:.2f}m/s, '
            f'steering={self.start_boost_steering:.2f}'
        )
        self.get_logger().info(
            f'   Log Mode: {self.controller_log_mode}, '
            f'PID Tuning Log Hz: {self.pid_tuning_log_hz}, '
            f'Bar Width: {self.pid_tuning_bar_width}'
        )

    def load_parameters(self):
        """从ROS参数服务器读取配置，支持launch YAML覆盖默认值。"""
        self.Kp = self.get_parameter('Kp').value
        self.Ki = self.get_parameter('Ki').value
        self.Kd = self.get_parameter('Kd').value
        self.integral_max = self.get_parameter('integral_max').value

        self.linear_speed_mps = self.get_parameter('linear_speed').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.max_steering = self.get_parameter('max_steering').value
        self.steering_slew_rate = self.get_parameter('steering_slew_rate').value
        self.invalid_timeout = self.get_parameter('invalid_timeout').value
        self.enable_perception_stop = self.get_parameter('enable_perception_stop').value
        self.ignore_stop_requests = self.get_parameter('ignore_stop_requests').value
        self.perception_stop_timeout = self.get_parameter('perception_stop_timeout').value
        self.autonomous_enabled_on_start = self.get_parameter('autonomous_enabled_on_start').value
        self.publish_stop_when_disabled = self.get_parameter('publish_stop_when_disabled').value
        self.enable_manual_override = self.get_parameter('enable_manual_override').value
        self.use_lane_state = self.get_parameter('use_lane_state').value
        self.lane_state_timeout = self.get_parameter('lane_state_timeout').value
        self.heading_gain = self.get_parameter('heading_gain').value
        self.curvature_gain = self.get_parameter('curvature_gain').value
        self.enable_start_boost = self.get_parameter('enable_start_boost').value
        self.start_boost_task_state = self.get_parameter('start_boost_task_state').value
        self.start_boost_duration = self.get_parameter('start_boost_duration').value
        self.start_boost_speed_mps = self.get_parameter('start_boost_speed').value
        self.start_boost_steering = self.get_parameter('start_boost_steering').value

        self.controller_log_mode = self.get_parameter('controller_log_mode').value
        self.pid_tuning_log_hz = self.get_parameter('pid_tuning_log_hz').value
        self.pid_tuning_bar_width = self.normalize_bar_width(
            self.get_parameter('pid_tuning_bar_width').value
        )
        self.controller_debug_hz = self.get_parameter('controller_debug_hz').value
        self.update_wheel_speed()

    def parameters_callback(self, parameters):
        """支持rqt_reconfigure/ros2 param运行时调参。"""
        pending = {
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'integral_max': self.integral_max,
            'linear_speed': self.linear_speed_mps,
            'wheel_radius': self.wheel_radius,
            'max_steering': self.max_steering,
            'steering_slew_rate': self.steering_slew_rate,
            'invalid_timeout': self.invalid_timeout,
            'enable_perception_stop': self.enable_perception_stop,
            'ignore_stop_requests': self.ignore_stop_requests,
            'perception_stop_timeout': self.perception_stop_timeout,
            'publish_stop_when_disabled': self.publish_stop_when_disabled,
            'enable_manual_override': self.enable_manual_override,
            'use_lane_state': self.use_lane_state,
            'lane_state_timeout': self.lane_state_timeout,
            'heading_gain': self.heading_gain,
            'curvature_gain': self.curvature_gain,
            'enable_start_boost': self.enable_start_boost,
            'start_boost_task_state': self.start_boost_task_state,
            'start_boost_duration': self.start_boost_duration,
            'start_boost_speed': self.start_boost_speed_mps,
            'start_boost_steering': self.start_boost_steering,
            'controller_log_mode': self.controller_log_mode,
            'pid_tuning_log_hz': self.pid_tuning_log_hz,
            'pid_tuning_bar_width': self.pid_tuning_bar_width,
            'controller_debug_hz': self.controller_debug_hz,
        }

        for parameter in parameters:
            if parameter.name not in pending:
                continue
            pending[parameter.name] = parameter.value

        try:
            pending['Kp'] = float(pending['Kp'])
            pending['Ki'] = float(pending['Ki'])
            pending['Kd'] = float(pending['Kd'])
            pending['integral_max'] = float(pending['integral_max'])
            pending['linear_speed'] = float(pending['linear_speed'])
            pending['wheel_radius'] = float(pending['wheel_radius'])
            pending['max_steering'] = float(pending['max_steering'])
            pending['steering_slew_rate'] = float(pending['steering_slew_rate'])
            pending['invalid_timeout'] = float(pending['invalid_timeout'])
            pending['enable_perception_stop'] = bool(pending['enable_perception_stop'])
            pending['ignore_stop_requests'] = bool(pending['ignore_stop_requests'])
            pending['perception_stop_timeout'] = float(pending['perception_stop_timeout'])
            pending['publish_stop_when_disabled'] = bool(pending['publish_stop_when_disabled'])
            pending['enable_manual_override'] = bool(pending['enable_manual_override'])
            pending['use_lane_state'] = bool(pending['use_lane_state'])
            pending['lane_state_timeout'] = float(pending['lane_state_timeout'])
            pending['heading_gain'] = float(pending['heading_gain'])
            pending['curvature_gain'] = float(pending['curvature_gain'])
            pending['enable_start_boost'] = bool(pending['enable_start_boost'])
            pending['start_boost_task_state'] = str(pending['start_boost_task_state'])
            pending['start_boost_duration'] = float(pending['start_boost_duration'])
            pending['start_boost_speed'] = float(pending['start_boost_speed'])
            pending['start_boost_steering'] = float(pending['start_boost_steering'])
            pending['pid_tuning_log_hz'] = float(pending['pid_tuning_log_hz'])
            pending['pid_tuning_bar_width'] = int(pending['pid_tuning_bar_width'])
            pending['controller_debug_hz'] = float(pending['controller_debug_hz'])
            pending['controller_log_mode'] = str(pending['controller_log_mode'])
        except (TypeError, ValueError) as exc:
            return SetParametersResult(
                successful=False,
                reason=f'Invalid controller parameter type/value: {exc}'
            )

        if pending['integral_max'] < 0.0:
            return SetParametersResult(successful=False, reason='integral_max must be >= 0')
        if pending['wheel_radius'] < 0.0:
            return SetParametersResult(successful=False, reason='wheel_radius must be >= 0')
        if pending['max_steering'] < 0.0:
            return SetParametersResult(successful=False, reason='max_steering must be >= 0')
        if pending['steering_slew_rate'] < 0.0:
            return SetParametersResult(successful=False, reason='steering_slew_rate must be >= 0')
        if pending['invalid_timeout'] < 0.0:
            return SetParametersResult(successful=False, reason='invalid_timeout must be >= 0')
        if pending['perception_stop_timeout'] < 0.0:
            return SetParametersResult(successful=False, reason='perception_stop_timeout must be >= 0')
        if pending['lane_state_timeout'] < 0.0:
            return SetParametersResult(successful=False, reason='lane_state_timeout must be >= 0')
        if pending['start_boost_duration'] < 0.0:
            return SetParametersResult(successful=False, reason='start_boost_duration must be >= 0')
        if pending['start_boost_speed'] < 0.0:
            return SetParametersResult(successful=False, reason='start_boost_speed must be >= 0')
        if abs(pending['start_boost_steering']) > 1.0:
            return SetParametersResult(successful=False, reason='start_boost_steering must be in [-1, 1]')
        if pending['pid_tuning_log_hz'] <= 0.0:
            return SetParametersResult(successful=False, reason='pid_tuning_log_hz must be > 0')
        if pending['controller_debug_hz'] <= 0.0:
            return SetParametersResult(successful=False, reason='controller_debug_hz must be > 0')
        if pending['controller_log_mode'] not in ('normal', 'pid_tuning', 'off'):
            return SetParametersResult(
                successful=False,
                reason='controller_log_mode must be one of: normal, pid_tuning, off'
            )

        self.Kp = pending['Kp']
        self.Ki = pending['Ki']
        self.Kd = pending['Kd']
        self.integral_max = pending['integral_max']
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        self.linear_speed_mps = pending['linear_speed']
        self.wheel_radius = pending['wheel_radius']
        self.max_steering = pending['max_steering']
        self.steering_slew_rate = pending['steering_slew_rate']
        self.invalid_timeout = pending['invalid_timeout']
        self.enable_perception_stop = pending['enable_perception_stop']
        self.ignore_stop_requests = pending['ignore_stop_requests']
        self.perception_stop_timeout = pending['perception_stop_timeout']
        self.publish_stop_when_disabled = pending['publish_stop_when_disabled']
        self.enable_manual_override = pending['enable_manual_override']
        self.use_lane_state = pending['use_lane_state']
        self.lane_state_timeout = pending['lane_state_timeout']
        self.heading_gain = pending['heading_gain']
        self.curvature_gain = pending['curvature_gain']
        self.enable_start_boost = pending['enable_start_boost']
        self.start_boost_task_state = pending['start_boost_task_state']
        self.start_boost_duration = pending['start_boost_duration']
        self.start_boost_speed_mps = pending['start_boost_speed']
        self.start_boost_steering = pending['start_boost_steering']
        self.controller_log_mode = pending['controller_log_mode']
        self.pid_tuning_log_hz = pending['pid_tuning_log_hz']
        self.pid_tuning_bar_width = self.normalize_bar_width(pending['pid_tuning_bar_width'])
        self.controller_debug_hz = pending['controller_debug_hz']
        self.update_wheel_speed()

        self.get_logger().info(
            'Updated controller parameters: '
            f'Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, '
            f'linear_speed={self.linear_speed_mps}, wheel_radius={self.wheel_radius}, '
            f'max_steering={self.max_steering}, steering_slew_rate={self.steering_slew_rate}, '
            f'heading_gain={self.heading_gain}, curvature_gain={self.curvature_gain}, '
            f'log_mode={self.controller_log_mode}'
        )
        return SetParametersResult(successful=True)

    def normalize_bar_width(self, width: int) -> int:
        width = max(11, min(81, int(width)))
        if width % 2 == 0:
            width += 1
        return width

    def update_wheel_speed(self):
        """将线速度(m/s)转换为底盘控制器期望的轮速(rps)。"""
        if self.wheel_radius > 0:
            self.wheel_speed_rps = self.linear_speed_mps / (2 * math.pi * self.wheel_radius)
        else:
            self.wheel_speed_rps = self.linear_speed_mps
            self.get_logger().warn('⚠️ wheel_radius=0, using linear_speed directly as wheel speed')
    
    def offset_callback(self, msg: Float32):
        """接收center_offset"""
        self.current_offset = msg.data
    
    def valid_callback(self, msg: Bool):
        """接收is_valid状态"""
        self.is_valid = msg.data
        
        if not self.is_valid:
            # 记录is_valid=False的开始时间
            if self.invalid_start_time is None:
                self.invalid_start_time = time.time()
                self.get_logger().warn('⚠️ No track detected!')
                self.track_lost_logged = False
        else:
            # 重置计时器
            self.invalid_start_time = None
            self.track_lost_logged = False

    def lane_state_callback(self, msg: String):
        """接收增强车道状态；保留 center_offset 话题作为回退。"""
        try:
            data = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Invalid /perception/lane_state JSON: {exc}')
            return

        self.current_offset = float(data.get('control_offset', self.current_offset))
        self.current_lateral_offset = float(data.get('lateral_offset', self.current_lateral_offset))
        self.current_heading_error = float(data.get('heading_error', 0.0))
        self.current_curvature = float(data.get('curvature', 0.0))
        self.lane_confidence = float(data.get('confidence', 0.0))
        self.road_state = str(data.get('road_state', 'UNKNOWN'))
        self.task_state = str(data.get('task_state', 'CLEAR'))
        self.last_lane_state_time = time.time()

    def has_fresh_lane_state(self, current_time: float) -> bool:
        if not self.use_lane_state or self.last_lane_state_time is None:
            return False
        if self.lane_state_timeout <= 0.0:
            return True
        return current_time - self.last_lane_state_time <= self.lane_state_timeout

    def perception_stop_callback(self, msg: Bool):
        """接收感知层停车请求。"""
        self.perception_stop_active = bool(msg.data)
        self.last_perception_stop_time = time.time()
        if not self.perception_stop_active:
            self.perception_stop_logged = False

    def emergency_stop_callback(self, msg: Bool):
        """接收UI/比赛控制层急停请求。"""
        self.emergency_stop_active = bool(msg.data)
        if self.emergency_stop_active:
            self.autonomous_enabled = False
            self.integral = 0.0
            self.get_logger().error('🛑 Emergency stop active; autonomous control disabled')
        else:
            self.get_logger().info('Emergency stop cleared; press Start to enable autonomous control')

    def manual_override_callback(self, msg: Bool):
        """键盘/人工接管时释放/cmd_vel，避免自动巡线和键盘同时写速度。"""
        if not self.enable_manual_override:
            return
        self.manual_override_active = bool(msg.data)
        if self.manual_override_active:
            self.autonomous_enabled = False
            self.integral = 0.0
            self.disabled_stop_logged = False
        else:
            self.manual_override_logged = False

    def set_enabled_callback(self, request, response):
        """发车/禁用巡线服务。"""
        self.autonomous_enabled = bool(request.data)
        if self.autonomous_enabled:
            self.emergency_stop_active = False
            self.manual_override_active = False
        self.integral = 0.0
        self.prev_offset = self.current_offset
        self.prev_time = time.time()
        self.start_boost_active = False
        self.start_boost_start_time = None
        self.start_boost_logged = False
        if self.autonomous_enabled:
            self.start_boost_used = False
        self.disabled_stop_logged = False
        self.manual_override_logged = False
        self.perception_stop_logged = False

        response.success = True
        if self.autonomous_enabled:
            response.message = 'Autonomous line following enabled'
            self.get_logger().info('▶️ Autonomous line following enabled')
        else:
            response.message = 'Autonomous line following disabled'
            self.get_logger().warn('⏸️ Autonomous line following disabled')
        return response

    def should_stop_for_perception(self, current_time: float) -> bool:
        if self.ignore_stop_requests:
            return False
        if not self.enable_perception_stop or not self.perception_stop_active:
            return False
        if self.last_perception_stop_time is None:
            return False
        if self.perception_stop_timeout > 0.0:
            elapsed = current_time - self.last_perception_stop_time
            if elapsed > self.perception_stop_timeout:
                self.perception_stop_active = False
                self.perception_stop_logged = False
                return False
        return True

    def handle_start_boost(self, current_time: float) -> bool:
        """Run one-shot open-loop launch while perception reports START_BOOST."""
        if not self.enable_start_boost:
            self.start_boost_active = False
            return False

        if self.start_boost_active:
            elapsed = current_time - self.start_boost_start_time
            if self.task_state != self.start_boost_task_state or elapsed >= self.start_boost_duration:
                self.start_boost_active = False
                self.start_boost_used = True
                self.start_boost_start_time = None
                self.start_boost_logged = False
                self.integral = 0.0
                self.last_control_mode = 'run'
                self.get_logger().info('Start boost finished; returning to PID line following')
                return False

            self.last_control_mode = 'start_boost'
            self.publish_start_boost()
            return True

        if self.start_boost_used or self.task_state != self.start_boost_task_state:
            return False

        self.start_boost_active = True
        self.start_boost_start_time = current_time
        self.integral = 0.0
        self.last_steering = self.start_boost_steering
        if not self.start_boost_logged:
            self.get_logger().warn(
                f'START_BOOST active: speed={self.start_boost_speed_mps:.2f}m/s, '
                f'steering={self.start_boost_steering:.2f}, max_duration={self.start_boost_duration:.2f}s'
            )
            self.start_boost_logged = True
        self.last_control_mode = 'start_boost'
        self.publish_start_boost()
        return True
    
    def control_loop(self):
        """控制循环（50Hz）"""
        current_time = time.time()
        dt = current_time - self.prev_time
        self.debug_loop_count += 1
        self.debug_dt_sum += dt
        self.debug_dt_max = max(self.debug_dt_max, dt)

        if self.emergency_stop_active:
            self.integral = 0.0
            self.last_control_mode = 'emergency_stop'
            self.publish_stop(log=False)
            self.prev_offset = self.current_offset
            self.prev_time = current_time
            return

        if self.manual_override_active:
            self.integral = 0.0
            self.last_control_mode = 'manual_override'
            self.debug_skip_count += 1
            if not self.manual_override_logged:
                self.get_logger().info('Manual override active; releasing /cmd_vel to keyboard controller')
                self.manual_override_logged = True
            self.prev_offset = self.current_offset
            self.prev_time = current_time
            return

        if not self.autonomous_enabled:
            self.integral = 0.0
            self.last_control_mode = 'disabled'
            if self.publish_stop_when_disabled:
                self.publish_stop(log=False)
            else:
                self.debug_skip_count += 1
            if not self.disabled_stop_logged:
                self.get_logger().info('Autonomous disabled; waiting for /line_follower/set_enabled')
                self.disabled_stop_logged = True
            self.prev_offset = self.current_offset
            self.prev_time = current_time
            return

        if self.handle_start_boost(current_time):
            self.prev_offset = self.current_offset
            self.prev_time = current_time
            return

        if self.should_stop_for_perception(current_time):
            self.integral = 0.0
            self.last_control_mode = 'perception_stop'
            self.publish_stop(log=False)
            if not self.perception_stop_logged:
                self.get_logger().warn('🛑 Perception stop request active; publishing STOP')
                self.perception_stop_logged = True
            self.prev_offset = self.current_offset
            self.prev_time = current_time
            return
        
        if not self.is_valid:
            invalid_duration = (
                current_time - self.invalid_start_time
                if self.invalid_start_time is not None
                else 0.0
            )
            if invalid_duration <= self.invalid_timeout:
                self.last_control_mode = 'track_hold'
                self.publish_cmd_vel(self.last_steering)
                self.prev_offset = self.current_offset
                self.prev_time = current_time
                return

            self.last_control_mode = 'invalid_track'
            self.debug_skip_count += 1
            if invalid_duration > self.invalid_timeout and not self.track_lost_logged:
                self.get_logger().error('🛑 Track lost for too long; not publishing cmd_vel.')
                self.track_lost_logged = True
            self.prev_offset = self.current_offset
            self.prev_time = current_time
            return

        # 有赛道，正常控制
        self.invalid_start_time = None

        # 固定速度巡线：只增强转向误差，不动态改速度。
        steering = self.compute_steering(dt, current_time)

        # 限幅
        steering = max(-self.max_steering, min(self.max_steering, steering))
        steering = self.apply_steering_slew_limit(steering, dt)

        # 发布控制指令
        self.last_control_mode = 'run'
        self.last_steering = steering
        self.publish_cmd_vel(steering)

        self.log_control_status(steering, dt, current_time)
        
        # 更新状态
        self.prev_offset = self.current_offset
        self.prev_time = current_time

    def apply_steering_slew_limit(self, steering: float, dt: float) -> float:
        """限制舵量变化率，降低高速下短周期左右猛打。"""
        if self.steering_slew_rate <= 0.0 or dt <= 0.0:
            return steering
        max_delta = self.steering_slew_rate * dt
        delta = steering - self.last_steering
        if delta > max_delta:
            return self.last_steering + max_delta
        if delta < -max_delta:
            return self.last_steering - max_delta
        return steering

    def compute_steering(self, dt: float, current_time: float) -> float:
        steering = self.pid_control(self.current_offset, dt)
        heading_term = 0.0
        curvature_term = 0.0

        if self.has_fresh_lane_state(current_time):
            heading_term = self.heading_gain * self.current_heading_error
            curvature_term = self.curvature_gain * self.current_curvature
            steering += heading_term + curvature_term

        self.last_lane_terms = (heading_term, curvature_term)
        return steering
    
    def pid_control(self, error: float, dt: float) -> float:
        """
        标准PID控制器
        
        Args:
            error: 当前误差 (center_offset)
            dt: 时间间隔 (秒)
            
        Returns:
            float: 控制输出 (steering)
        """
        # 1. 比例项
        P = self.Kp * error
        
        # 2. 积分项（累积误差）
        self.integral += error * dt
        # 积分限幅，防止积分饱和
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        I = self.Ki * self.integral
        
        # 3. 微分项（误差变化率）
        d_error = (error - self.prev_offset) / dt if dt > 0 else 0.0
        D = self.Kd * d_error
        self.last_pid_terms = (P, I, D)
        
        # 4. 总和
        output = P + I + D
        
        return output
    
    def log_control_status(self, steering: float, dt: float, current_time: float):
        """根据日志模式输出普通控制日志或PID调参日志"""
        if self.controller_log_mode == 'off':
            return
        
        if self.controller_log_mode == 'pid_tuning':
            log_hz = max(0.1, self.pid_tuning_log_hz)
            if current_time - self.last_tuning_log_time < 1.0 / log_hz:
                return
            self.last_tuning_log_time = current_time
            p_term, i_term, d_term = self.last_pid_terms
            heading_term, curvature_term = self.last_lane_terms
            self.get_logger().info(
                f'[PID_TUNE] err={self.current_offset:+.3f} '
                f'{self.format_offset_bar(self.current_offset)} '
                f'steer={steering:+.3f} '
                f'P={p_term:+.3f} I={i_term:+.3f} D={d_term:+.3f} '
                f'H={heading_term:+.3f} C={curvature_term:+.3f} '
                f'road={self.road_state} conf={self.lane_confidence:.2f} '
                f'dt={dt*1000:.1f}ms'
            )
            return
        
        if self.controller_log_mode != 'normal':
            self.get_logger().warn(
                f'Unknown controller_log_mode "{self.controller_log_mode}", using normal log behavior'
            )
            self.controller_log_mode = 'normal'
        
        # 普通日志：保持原来的每100帧打印一次
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
        
        if self._log_counter % 100 == 0:
            heading_term, curvature_term = self.last_lane_terms
            self.get_logger().info(
                f'📊 Offset: {self.current_offset:.3f}, '
                f'Steering: {steering:.3f}, '
                f'Integral: {self.integral:.3f}, '
                f'HeadingTerm: {heading_term:.3f}, '
                f'CurvTerm: {curvature_term:.3f}, '
                f'Road: {self.road_state}, '
                f'dt: {dt*1000:.1f}ms'
            )
    
    def format_offset_bar(self, offset: float) -> str:
        """将[-1, 1]误差画成固定宽度ASCII条，中心线表示0误差"""
        width = self.pid_tuning_bar_width
        center = width // 2
        clamped = max(-1.0, min(1.0, offset))
        marker = int(round((clamped + 1.0) * (width - 1) / 2.0))
        
        chars = ['-'] * width
        chars[center] = '|'
        chars[marker] = 'X' if marker == center else '*'
        return '[' + ''.join(chars) + ']'
    
    def publish_cmd_vel(self, steering: float):
        """发布速度控制指令"""
        twist_msg = Twist()
        # ⭐ 关键修复：发送转速 (rps) 而不是角速度 (rad/s)
        # chassis_controller 期望的是 rps (转/秒，1 rps = 360°/s)
        twist_msg.linear.x = self.wheel_speed_rps  # 轮子转速 (rps)
        twist_msg.angular.z = steering               # 转向比例 (-1.0 ~ 1.0)
        self.cmd_vel_publisher.publish(twist_msg)
        self.debug_cmd_count += 1

    def publish_start_boost(self):
        """发布弹射起步开环速度；linear.x 仍使用底盘期望的轮速 rps。"""
        twist_msg = Twist()
        if self.wheel_radius > 0:
            twist_msg.linear.x = self.start_boost_speed_mps / (2 * math.pi * self.wheel_radius)
        else:
            twist_msg.linear.x = self.start_boost_speed_mps
        twist_msg.angular.z = max(-1.0, min(1.0, self.start_boost_steering))
        self.cmd_vel_publisher.publish(twist_msg)
        self.debug_cmd_count += 1
    
    def publish_stop(self, log: bool = True):
        """发布停车指令"""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist_msg)
        self.debug_stop_count += 1
        if log:
            self.get_logger().info('🛑 Published STOP command (linear=0.0, angular=0.0)')

    def publish_debug_status(self):
        now = time.time()
        elapsed = max(1e-6, now - self.debug_window_start)
        loop_hz = self.debug_loop_count / elapsed
        cmd_hz = self.debug_cmd_count / elapsed
        stop_hz = self.debug_stop_count / elapsed
        skip_hz = self.debug_skip_count / elapsed
        avg_dt_ms = (self.debug_dt_sum / self.debug_loop_count * 1000.0) if self.debug_loop_count else 0.0

        msg = String()
        msg.data = (
            f'mode={self.last_control_mode} '
            f'enabled={self.autonomous_enabled} '
            f'valid={self.is_valid} '
            f'pstop={self.perception_stop_active} '
            f'ignore_stop={self.ignore_stop_requests} '
            f'estop={self.emergency_stop_active} '
            f'manual={self.manual_override_active} '
            f'offset={self.current_offset:+.3f} '
            f'heading={self.current_heading_error:+.3f} '
            f'curvature={self.current_curvature:+.3f} '
            f'conf={self.lane_confidence:.2f} '
            f'road={self.road_state} '
            f'task={self.task_state} '
            f'wheel_rps={self.wheel_speed_rps:.3f} '
            f'loop_hz={loop_hz:.1f} '
            f'cmd_hz={cmd_hz:.1f} '
            f'stop_hz={stop_hz:.1f} '
            f'skip_hz={skip_hz:.1f} '
            f'avg_dt_ms={avg_dt_ms:.1f} '
            f'max_dt_ms={self.debug_dt_max * 1000.0:.1f}'
        )
        self.debug_publisher.publish(msg)

        self.debug_window_start = now
        self.debug_loop_count = 0
        self.debug_cmd_count = 0
        self.debug_stop_count = 0
        self.debug_skip_count = 0
        self.debug_dt_sum = 0.0
        self.debug_dt_max = 0.0


def main(args=None):
    rclpy.init(args=args)
    controller = None
    
    try:
        controller = LineFollowerController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        if controller is not None:
            controller.get_logger().info('🛑 Controller stopped by user (KeyboardInterrupt)')
    except Exception as e:
        if controller is not None:
            controller.get_logger().error(f'❌ Unexpected error: {e}')
        else:
            raise
    finally:
        if controller is not None:
            # ⭐ Ctrl-C/退出时发布停车指令；无赛道运行中不发布cmd_vel。
            controller.get_logger().info('🛑 Shutting down - sending stop commands...')
            for _ in range(3):
                controller.publish_stop(log=False)
                rclpy.spin_once(controller, timeout_sec=0.05)

            controller.get_logger().info('✅ Stop commands sent, destroying node...')
            controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
