#!/usr/bin/env python3
"""
阿克曼底盘控制节点
- 通过串口发送控制指令到底盘（速度、转向）
- 从串口读取底盘反馈数据并发布到ROS2话题
通信格式: 
  - 发送: #Flag,DIR,speed_set,servo_pwm
  - 接收: enc:12,set:300
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, Int32, Int64
from geometry_msgs.msg import Twist
import serial
import re
import time


class ChassisController(Node):
    def __init__(self):
        super().__init__('chassis_controller')
        
        # 参数声明
        self.declare_parameter('serial_port', '/dev/ttyS0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('max_speed', 10.0)  # 最大转速 rps (转/秒)
        self.declare_parameter('servo_center', 3000)  # 舵机中位PWM值
        self.declare_parameter('servo_left_max', 2300)  # 舵机左打满PWM值
        self.declare_parameter('servo_right_max', 3700)  # 舵机右打满PWM值
        self.declare_parameter('cmd_vel_timeout_sec', 0.20)  # 控制指令断流后自动停车
        
        # 获取参数
        self.serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.servo_center = self.get_parameter('servo_center').get_parameter_value().integer_value
        self.servo_left_max = self.get_parameter('servo_left_max').get_parameter_value().integer_value
        self.servo_right_max = self.get_parameter('servo_right_max').get_parameter_value().integer_value
        self.cmd_vel_timeout_sec = (
            self.get_parameter('cmd_vel_timeout_sec').get_parameter_value().double_value
        )
        
        # 串口对象
        self.ser = None
        
        # 当前状态
        self.enabled = False  # Flag: 启1停0；必须收到显式 /chassis/enable 才启用
        self.direction = 1    # DIR: 前进1后退0
        self.speed = 0.0      # 速度 rad/s
        self.steering_ratio = 0.0  # 转向比例 -1.0(右满) 到 1.0(左满)
        self.last_cmd_vel_time = time.monotonic()
        self.has_cmd_vel = False
        self.cmd_vel_watchdog_active = False
        self.latest_encoder_delta = 0
        self.encoder_count = 0
        self.latest_speed_set_feedback = 0
        
        # 订阅话题 - 控制指令
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Int8, '/chassis/enable', self.enable_callback, 10)
        self.create_subscription(Int8, '/chassis/direction', self.direction_callback, 10)
        
        # 发布话题 - 底盘反馈
        self.encoder_delta_pub = self.create_publisher(Int32, '/chassis/encoder_delta', 10)
        self.encoder_count_pub = self.create_publisher(Int64, '/chassis/encoder_count', 10)
        self.speed_set_feedback_pub = self.create_publisher(Int32, '/chassis/speed_set_feedback', 10)
        
        # 初始化串口
        self.init_serial()
        
        # 定时器1：定期发送控制指令 (50Hz) - 修改为50Hz
        self.send_timer = self.create_timer(0.02, self.send_control_command)
        
        # 定时器2：定期读取串口数据 (200Hz) - 提高频率以减少延迟
        self.read_timer = self.create_timer(0.005, self.read_serial_data)
        
        self.get_logger().info('统一底盘控制节点已启动')
        self.get_logger().info(f'串口: {self.serial_port}, 波特率: {self.baudrate}')
        self.get_logger().info('功能: 发送控制指令 + 接收偏航角数据')
    
    def init_serial(self):
        """初始化串口连接"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01,  # 设置短超时，避免阻塞
                write_timeout=0.01
            )
            if self.ser.is_open:
                self.get_logger().info(f'✅ 串口 {self.serial_port} 打开成功')
            else:
                self.ser.open()
                self.get_logger().info(f'✅ 串口 {self.serial_port} 打开成功')
        except Exception as e:
            self.get_logger().error(f'❌ 串口打开失败: {str(e)}')
            self.ser = None
    
    def cmd_vel_callback(self, msg: Twist):
        """处理速度控制指令"""
        self.last_cmd_vel_time = time.monotonic()
        self.has_cmd_vel = True
        self.cmd_vel_watchdog_active = False

        # linear.x 控制轮子转速 (rps, revolutions per second)
        self.speed = msg.linear.x
        
        # angular.z 控制转向比例 (-1.0 到 1.0)
        # 正值左转，负值右转
        self.steering_ratio = msg.angular.z
        
        # 判断方向
        if self.speed >= 0:
            self.direction = 1  # 前进
        else:
            self.direction = 0  # 后退
            self.speed = abs(self.speed)
        
        # 限制转速范围 (rps)
        self.speed = max(0.0, min(self.speed, self.max_speed))
        
        # 限制转向比例范围
        self.steering_ratio = max(-1.0, min(self.steering_ratio, 1.0))
    
    def enable_callback(self, msg: Int8):
        """处理使能控制"""
        new_enabled = bool(msg.data)
        if new_enabled == self.enabled:
            return
        self.enabled = new_enabled
        if not self.enabled:
            # 禁用时同步清除缓存，避免后续只发 enable 就恢复旧速度。
            self.speed = 0.0
            self.steering_ratio = 0.0
        self.get_logger().info(f'底盘使能状态: {"启用" if self.enabled else "禁用"}')
    
    def direction_callback(self, msg: Int8):
        """处理方向控制（可选，通常由cmd_vel自动判断）"""
        self.direction = msg.data
        self.get_logger().info(f'底盘方向: {"前进" if self.direction == 1 else "后退"}')
    
    def pack_command(self) -> bytes:
        """
        打包控制指令
        格式: #Flag,DIR,speed_set,servo_pwm;
        speed_set = speed_rps * 247 (int16_t), 其中 speed_rps 单位为 rps (转/秒)
        servo_pwm = 舵机PWM值 (int16_t), 2300(左满) - 3000(中位) - 3700(右满)
        """
        flag = 1 if self.enabled else 0
        dir_val = 1 if self.direction == 1 else 0
        
        # 计算speed_set: 转速(rps) * 247
        # 1 rps = 360°/s (角度制转速)
        speed_set = int(self.speed * 247)
        # 限制在int16范围内
        speed_set = max(-32768, min(speed_set, 32767))
        
        # 计算servo_pwm: 根据转向比例映射到PWM范围
        # steering_ratio: -1.0(右满) 到 1.0(左满)
        # servo_pwm: 3700(右满) 到 2300(左满)
        if self.steering_ratio >= 0:
            # 左转: 3000 -> 2300
            servo_pwm = int(self.servo_center - (self.servo_center - self.servo_left_max) * self.steering_ratio)
        else:
            # 右转: 3000 -> 3700
            servo_pwm = int(self.servo_center + (self.servo_right_max - self.servo_center) * abs(self.steering_ratio))
        
        # 限制在合理范围内
        servo_pwm = max(self.servo_left_max, min(servo_pwm, self.servo_right_max))
        
        # 组装命令字符串（末尾添加分号）
        speed_set_str = f"{speed_set:04d}"
        command = f"#{flag},{dir_val},{speed_set_str},{servo_pwm};"
        
        return command.encode('utf-8')
    
    def send_control_command(self):
        """定时发送控制指令"""
        self.enforce_cmd_vel_watchdog()

        if self.ser is None or not self.ser.is_open:
            self.get_logger().warn('串口未打开，尝试重新连接...')
            self.init_serial()
            return
        
        try:
            # 打包命令
            command = self.pack_command()
            command_str = command.decode('utf-8')
            
            # 发送到串口
            self.ser.write(command)
            self.ser.flush()
            
            # 实时输出发送的命令（每10次输出一次，避免刷屏）
            if not hasattr(self, 'send_count'):
                self.send_count = 0
            self.send_count += 1
            
            # 每秒输出一次（50Hz更新率，每50次输出一次）
            if self.send_count % 50 == 1:
                # 解析命令各字段（去掉末尾的分号）
                command_clean = command_str.rstrip(';')
                parts = command_clean.split(',')
                if len(parts) == 4:
                    flag = parts[0][1:]  # 去掉#
                    dir_val = parts[1]
                    speed_set = parts[2]
                    servo_pwm = parts[3]
                    
                    # 计算实际值
                    speed_actual = int(speed_set) / 247.0  # rps (转/秒)
                    servo_pwm_val = int(servo_pwm)
                    
                    direction_str = "前进" if dir_val == "1" else "后退"
                    enable_str = "启用" if flag == "1" else "禁用"
                    
                    # 计算转向描述
                    if servo_pwm_val == self.servo_center:
                        steering_str = "直行"
                    elif servo_pwm_val < self.servo_center:
                        steering_ratio = (self.servo_center - servo_pwm_val) / (self.servo_center - self.servo_left_max)
                        steering_str = f"左转{steering_ratio*100:.0f}%"
                    else:
                        steering_ratio = (servo_pwm_val - self.servo_center) / (self.servo_right_max - self.servo_center)
                        steering_str = f"右转{steering_ratio*100:.0f}%"
                    
                    self.get_logger().info(
                        f'发送: {command_str} | '
                        f'状态:{enable_str} {direction_str} | '
                        f'转速:{speed_actual:.2f}rps({speed_set}) | '
                        f'舵机:{servo_pwm}({steering_str})'
                    )
                
        except Exception as e:
            self.get_logger().error(f'发送命令失败: {str(e)}')
            try:
                self.ser.close()
            except:
                pass
            self.ser = None

    def enforce_cmd_vel_watchdog(self):
        """控制器断流时清零速度、舵角并关闭底盘使能。"""
        if not self.enabled or self.cmd_vel_timeout_sec <= 0.0:
            return

        command_age = time.monotonic() - self.last_cmd_vel_time
        if self.has_cmd_vel and command_age <= self.cmd_vel_timeout_sec:
            return

        self.speed = 0.0
        self.steering_ratio = 0.0
        self.enabled = False
        if not self.cmd_vel_watchdog_active:
            self.cmd_vel_watchdog_active = True
            self.get_logger().error(
                f'/cmd_vel 超时 {command_age:.3f}s，底盘已自动清零并禁用'
            )
    
    def read_serial_data(self):
        """读取串口数据（底盘反馈）- 优化版本"""
        if not self.ser or not self.ser.is_open:
            return
        
        try:
            # 检查是否有数据可读
            if self.ser.in_waiting > 0:
                # 读取所有可用数据
                raw_data = self.ser.read(self.ser.in_waiting)
                
                # 添加到缓冲区
                if not hasattr(self, '_serial_buffer'):
                    self._serial_buffer = b''
                self._serial_buffer += raw_data
                
                # 按行分割处理
                while b'\n' in self._serial_buffer:
                    line_bytes, self._serial_buffer = self._serial_buffer.split(b'\n', 1)
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        match = re.match(r'enc:\s*([+-]?\d+)\s*,\s*set:\s*([+-]?\d+)', line)
                        if match:
                            try:
                                encoder_delta = int(match.group(1))
                                speed_set_feedback = int(match.group(2))

                                self.latest_encoder_delta = encoder_delta
                                self.encoder_count += encoder_delta
                                self.latest_speed_set_feedback = speed_set_feedback

                                encoder_msg = Int32()
                                encoder_msg.data = encoder_delta
                                self.encoder_delta_pub.publish(encoder_msg)

                                encoder_count_msg = Int64()
                                encoder_count_msg.data = self.encoder_count
                                self.encoder_count_pub.publish(encoder_count_msg)

                                speed_set_msg = Int32()
                                speed_set_msg.data = speed_set_feedback
                                self.speed_set_feedback_pub.publish(speed_set_msg)
                                
                                # 每秒打印一次日志
                                # if not hasattr(self, '_last_log') or time.time() - self._last_log >= 1.0:
                                #     yaw_rad = yaw_deg * 3.141592653589793 / 180.0
                                #     self.get_logger().info(f"📐 偏航角: {yaw_deg:.2f}° = {yaw_rad:.4f} rad")
                                #     self._last_log = time.time()
                                # else:
                                #     self.get_logger().debug(f"偏航角: {yaw_deg:.2f}°")
                            
                            except ValueError:
                                self.get_logger().warn(f"无效的偏航角数据: {line}")
        
        except Exception as e:
            self.get_logger().error(f"读取错误: {e}")
    
    def destroy_node(self):
        """清理资源"""
        if self.ser and self.ser.is_open:
            # ⭐ 发送停止命令：禁用底盘 + 速度0 + 舵机回正
            stop_command = f"#0,1,0000,{self.servo_center};".encode('utf-8')
            try:
                # 多次发送确保下位机收到
                for i in range(3):
                    self.ser.write(stop_command)
                    self.ser.flush()
                    time.sleep(0.05)
                self.get_logger().info(f'🛑 发送停车指令(3次): {stop_command.decode("utf-8")}')
            except Exception as e:
                self.get_logger().error(f'发送停车指令失败: {e}')
            self.ser.close()
            self.get_logger().info('串口已关闭')
        else:
            self.get_logger().warn('⚠️ 串口未打开或不存在')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ChassisController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Chassis controller stopped by user')
    except Exception as e:
        node.get_logger().error(f'❌ Unexpected error: {e}')
    finally:
        node.get_logger().info('🛑 Shutting down chassis controller...')
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info('✅ Chassis controller shutdown complete')


if __name__ == '__main__':
    main()
