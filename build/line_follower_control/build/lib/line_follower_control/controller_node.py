#!/usr/bin/env python3
"""
巡线控制节点
订阅语义分割的center_offset和is_valid，发布/cmd_vel控制底盘
使用PD控制器实现赛道跟随
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
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
        self.declare_parameter('invalid_timeout', 0.5)   # is_valid=False超时时间(秒)
        
        # 日志参数
        self.declare_parameter('controller_log_mode', 'normal')  # normal, pid_tuning, off
        self.declare_parameter('pid_tuning_log_hz', 10.0)        # PID调参日志频率
        self.declare_parameter('pid_tuning_bar_width', 41)       # 误差条宽度，建议使用奇数
        
        # 获取PID参数
        self.Kp = self.get_parameter('Kp').get_parameter_value().double_value
        self.Ki = self.get_parameter('Ki').get_parameter_value().double_value
        self.Kd = self.get_parameter('Kd').get_parameter_value().double_value
        self.integral_max = self.get_parameter('integral_max').get_parameter_value().double_value
        
        # 获取车辆参数
        self.linear_speed_mps = self.get_parameter('linear_speed').get_parameter_value().double_value  # m/s
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value  # m
        self.max_steering = self.get_parameter('max_steering').get_parameter_value().double_value
        self.invalid_timeout = self.get_parameter('invalid_timeout').get_parameter_value().double_value
        
        # 获取日志参数
        self.controller_log_mode = self.get_parameter('controller_log_mode').get_parameter_value().string_value
        self.pid_tuning_log_hz = self.get_parameter('pid_tuning_log_hz').get_parameter_value().double_value
        self.pid_tuning_bar_width = self.get_parameter('pid_tuning_bar_width').get_parameter_value().integer_value
        self.pid_tuning_bar_width = max(11, min(81, self.pid_tuning_bar_width))
        if self.pid_tuning_bar_width % 2 == 0:
            self.pid_tuning_bar_width += 1
        
        # ⭐ 关键修复：将线速度 (m/s) 转换为转速 (rps, revolutions per second)
        # v = ω_rps × 2π × r  =>  ω_rps = v / (2π × r)
        import math
        if self.wheel_radius > 0:
            self.wheel_speed_rps = self.linear_speed_mps / (2 * math.pi * self.wheel_radius)  # rps (转/秒)
        else:
            self.wheel_speed_rps = self.linear_speed_mps  # 如果半径为0，直接使用原值
            self.get_logger().warn('⚠️ wheel_radius=0, using linear_speed directly as wheel speed')
        
        # ==================== 状态变量 ====================
        self.current_offset = 0.0          # 当前偏移量
        self.prev_offset = 0.0             # 上一帧偏移量（用于计算微分）
        self.integral = 0.0                # 积分项累积
        self.prev_time = time.time()       # 上一帧时间
        self.is_valid = True               # 是否有赛道
        self.invalid_start_time = None     # is_valid=False的开始时间
        self.last_pid_terms = (0.0, 0.0, 0.0)
        self.last_tuning_log_time = 0.0
        
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
        
        # ==================== 发布者 ====================
        # 发布/cmd_vel到chassis_controller
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # ==================== 定时器 ====================
        # 以50Hz频率发布控制指令
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
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
            f'   Log Mode: {self.controller_log_mode}, '
            f'PID Tuning Log Hz: {self.pid_tuning_log_hz}, '
            f'Bar Width: {self.pid_tuning_bar_width}'
        )
    
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
        else:
            # 重置计时器
            self.invalid_start_time = None
    
    def control_loop(self):
        """控制循环（50Hz）"""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # 检查is_valid超时
        if not self.is_valid:
            if self.invalid_start_time is not None:
                invalid_duration = current_time - self.invalid_start_time
                if invalid_duration > self.invalid_timeout:
                    # 超时，停车
                    self.get_logger().error('🛑 Track lost for too long! Stopping...')
                    self.publish_stop()
                    return
        else:
            # 有赛道，正常控制
            self.invalid_start_time = None
            
            # ⭐ 标准PID控制
            steering = self.pid_control(self.current_offset, dt)
            
            # 限幅
            steering = max(-self.max_steering, min(self.max_steering, steering))
            
            # 发布控制指令
            self.publish_cmd_vel(steering)
            
            self.log_control_status(steering, dt, current_time)
        
        # 更新状态
        self.prev_offset = self.current_offset
        self.prev_time = current_time
    
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
            self.get_logger().info(
                f'[PID_TUNE] err={self.current_offset:+.3f} '
                f'{self.format_offset_bar(self.current_offset)} '
                f'steer={steering:+.3f} '
                f'P={p_term:+.3f} I={i_term:+.3f} D={d_term:+.3f} '
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
            self.get_logger().info(
                f'📊 Offset: {self.current_offset:.3f}, '
                f'Steering: {steering:.3f}, '
                f'Integral: {self.integral:.3f}, '
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
    
    def publish_stop(self):
        """发布停车指令"""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist_msg)
        self.get_logger().info('🛑 Published STOP command (linear=0.0, angular=0.0)')
        # ⭐ 短暂等待确保消息发送
        time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    controller = LineFollowerController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('🛑 Controller stopped by user (KeyboardInterrupt)')
    except Exception as e:
        controller.get_logger().error(f'❌ Unexpected error: {e}')
    finally:
        # ⭐ 确保停车 - 多次发布提高可靠性
        controller.get_logger().info('🛑 Shutting down - sending stop commands...')
        for i in range(3):  # 发布3次确保收到
            controller.publish_stop()
            time.sleep(0.05)
        
        controller.get_logger().info('✅ Stop commands sent, destroying node...')
        controller.destroy_node()
        rclpy.shutdown()
        controller.get_logger().info('✅ Node shutdown complete')


if __name__ == '__main__':
    main()
