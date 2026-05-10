#!/usr/bin/env python3
"""
终端键盘控制节点
使用tty直接读取键盘输入，无需图形界面
适合SSH远程使用
控制逻辑：
- W: 前进加速（按住加速，释放减速）
- S: 匀速后退
- A: 左转（按住逐渐左转到最大角度）
- D: 右转（按住逐渐右转到最大角度）
- G: 电机启停切换
- Q: 退出
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
import sys
import tty
import termios
import select
import time


class TerminalKeyboardController(Node):
    def __init__(self):
        super().__init__('terminal_keyboard_controller')
        
        # 参数声明
        self.declare_parameter('max_speed', 10.0)       # 最大前进速度 rad/s
        self.declare_parameter('reverse_speed', 2.0)    # 后退速度 rad/s
        self.declare_parameter('acceleration', 5.0)     # 加速度 rad/s²
        self.declare_parameter('deceleration', 5.0)     # 减速度 rad/s²
        self.declare_parameter('turn_rate', 1.0)        # 转向速率 (比例/s), -1.0到1.0
        self.declare_parameter('update_rate', 20.0)     # 更新频率 Hz
        
        # 获取参数
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.reverse_speed = self.get_parameter('reverse_speed').get_parameter_value().double_value
        self.acceleration = self.get_parameter('acceleration').get_parameter_value().double_value
        self.deceleration = self.get_parameter('deceleration').get_parameter_value().double_value
        self.turn_rate = self.get_parameter('turn_rate').get_parameter_value().double_value
        self.update_rate = self.get_parameter('update_rate').get_parameter_value().double_value
        
        # 当前状态
        self.current_speed = 0.0        # 当前速度
        self.current_steering = 0.0     # 当前转向比例 (-1.0右满 到 1.0左满)
        self.motor_enabled = False      # 电机使能状态
        
        # 按键状态
        self.key_w_pressed = False
        self.key_s_pressed = False
        self.key_a_pressed = False
        self.key_d_pressed = False
        
        # 发布者
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Int8, '/chassis/enable', 10)
        
        # 保存终端设置
        self.fd = sys.stdin.fileno()
        
        # 检查是否为终端设备
        if not sys.stdin.isatty():
            self.get_logger().warn("标准输入不是终端设备，尝试继续初始化...")
        
        try:
            self.old_settings = termios.tcgetattr(self.fd)
        except termios.error as e:
            self.get_logger().error(f"无法获取终端属性: {e}")
            self.get_logger().error("请确保使用 'ros2 run' 直接运行此节点，或使用 pty=True 的 launch 文件")
            raise RuntimeError("无法访问终端设备进行键盘控制") from e
        
        # 设置为原始模式
        tty.setraw(self.fd)
        
        # 定时器
        self.timer = self.create_timer(1.0 / self.update_rate, self.update_callback)
        
        # 上次UI更新时间
        self.last_ui_update = time.time()
        self.ui_update_interval = 0.1  # UI更新间隔（秒）
        
        self.get_logger().info('终端键盘控制节点已启动')
        self.print_help()
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print(" "*15 + "阿克曼底盘终端控制")
        print("="*60)
        print("\n控制说明:")
        print("  W - 前进加速（按住加速，释放减速）")
        print("  S - 匀速后退")
        print("  A - 左转（按住逐渐左转）")
        print("  D - 右转（按住逐渐右转）")
        print("  G - 电机启停切换")
        print("  Q - 退出程序")
        print("\n提示:")
        print("  - 直接在终端按键即可控制")
        print("  - 按G键启用电机后才能移动")
        print("  - 按Q键安全退出\n")
        print("="*60 + "\n")
    
    def get_key(self):
        """非阻塞读取键盘输入"""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ch.lower()
        return None
    
    def handle_input(self):
        """处理键盘输入"""
        key = self.get_key()
        
        if key is None:
            return True
        
        if key == 'q':
            return False
        elif key == 'w':
            self.key_w_pressed = not self.key_w_pressed
            status = "按下" if self.key_w_pressed else "释放"
            self.get_logger().debug(f'W键 {status}')
        elif key == 's':
            self.key_s_pressed = not self.key_s_pressed
            status = "按下" if self.key_s_pressed else "释放"
            self.get_logger().debug(f'S键 {status}')
        elif key == 'a':
            self.key_a_pressed = not self.key_a_pressed
            status = "按下" if self.key_a_pressed else "释放"
            self.get_logger().debug(f'A键 {status}')
        elif key == 'd':
            self.key_d_pressed = not self.key_d_pressed
            status = "按下" if self.key_d_pressed else "释放"
            self.get_logger().debug(f'D键 {status}')
        elif key == 'g':
            self.toggle_motor()
        
        return True
    
    def toggle_motor(self):
        """切换电机启停状态"""
        self.motor_enabled = not self.motor_enabled
        status = "启用" if self.motor_enabled else "禁用"
        self.get_logger().info(f'电机状态: {status}')
        
        # 发布使能消息
        msg = Int8()
        msg.data = 1 if self.motor_enabled else 0
        self.enable_pub.publish(msg)
        
        # 如果禁用电机，停止运动
        if not self.motor_enabled:
            self.current_speed = 0.0
            self.current_angle = 0.0
    
    def update_physics(self, dt):
        """更新物理状态"""
        # 处理前进逻辑
        if self.key_w_pressed:
            # 加速前进
            self.current_speed += self.acceleration * dt
            self.current_speed = min(self.current_speed, self.max_speed)
        elif self.key_s_pressed:
            # 匀速后退
            self.current_speed = -self.reverse_speed
        else:
            # 自然减速
            if self.current_speed > 0:
                self.current_speed -= self.deceleration * dt
                self.current_speed = max(self.current_speed, 0.0)
            elif self.current_speed < 0:
                self.current_speed += self.deceleration * dt
                self.current_speed = min(self.current_speed, 0.0)
        
        # 处理转向逻辑
        if self.key_a_pressed:
            # 左转（增加转向比例）
            self.current_steering += self.turn_rate * dt
            self.current_steering = min(self.current_steering, 1.0)
        elif self.key_d_pressed:
            # 右转（减小转向比例）
            self.current_steering -= self.turn_rate * dt
            self.current_steering = max(self.current_steering, -1.0)
        else:
            # 自动回正
            if self.current_steering > 0:
                self.current_steering -= self.turn_rate * dt
                self.current_steering = max(self.current_steering, 0.0)
            elif self.current_steering < 0:
                self.current_steering += self.turn_rate * dt
                self.current_steering = min(self.current_steering, 0.0)
    
    def publish_commands(self):
        """发布控制命令"""
        # 只有在电机启用时才发送速度命令
        if self.motor_enabled:
            twist_msg = Twist()
            twist_msg.linear.x = self.current_speed
            twist_msg.angular.z = self.current_steering  # 使用转向比例 (-1.0 到 1.0)
            self.cmd_vel_pub.publish(twist_msg)
    
    def print_ui(self):
        """打印终端UI"""
        current_time = time.time()
        
        # 控制UI更新频率
        if current_time - self.last_ui_update < self.ui_update_interval:
            return
        
        self.last_ui_update = current_time
        
        # 清屏并移动到顶部
        print('\033[2J\033[H', end='')
        
        # 标题
        print("="*60)
        print(" "*18 + "阿克曼底盘控制")
        print("="*60)
        
        # 电机状态
        status_str = "● 启用" if self.motor_enabled else "○ 禁用"
        status_color = "\033[92m" if self.motor_enabled else "\033[91m"  # 绿色或红色
        reset_color = "\033[0m"
        print(f"\n{status_color}电机状态: {status_str}{reset_color}")
        
        # 速度显示
        speed_bar_width = 30
        if self.current_speed >= 0:
            fill_ratio = self.current_speed / self.max_speed
            bar_char = "█"
            direction = "前进 ↑"
        else:
            fill_ratio = abs(self.current_speed) / self.reverse_speed
            bar_char = "▓"
            direction = "后退 ↓"
        
        fill_count = int(fill_ratio * speed_bar_width)
        empty_count = speed_bar_width - fill_count
        speed_bar = bar_char * fill_count + "░" * empty_count
        
        print(f"\n速度: {self.current_speed:+6.2f} rad/s  {direction}")
        print(f"[{speed_bar}]")
        
        # 转向显示
        steering_bar_width = 30
        center = steering_bar_width // 2
        # current_steering: -1.0(右满) 到 1.0(左满)
        steering_offset = int(self.current_steering * center)
        
        steering_bar = ["░"] * steering_bar_width
        steering_bar[center] = "┃"  # 中心标记
        
        if steering_offset != 0:
            pos = center + steering_offset
            if 0 <= pos < steering_bar_width:
                steering_bar[pos] = "●"
        
        # 计算舵机PWM值用于显示
        servo_center = 3000
        servo_left = 2300
        servo_right = 3700
        
        if self.current_steering >= 0:
            servo_pwm = int(servo_center - (servo_center - servo_left) * self.current_steering)
        else:
            servo_pwm = int(servo_center + (servo_right - servo_center) * abs(self.current_steering))
        
        steering_direction = "直行" if abs(self.current_steering) < 0.05 else \
                            f"左转{self.current_steering*100:.0f}%" if self.current_steering > 0 else \
                            f"右转{abs(self.current_steering)*100:.0f}%"
        
        print(f"\n转向: {steering_direction} (PWM:{servo_pwm})")
        print(f"[{''.join(steering_bar)}]")
        print(f" {'左':<{center}} {'|':^{1}} {'右':>{center}}")
        
        # 按键状态
        print("\n按键状态:")
        w_status = "\033[92m● W\033[0m" if self.key_w_pressed else "○ W"
        s_status = "\033[92m● S\033[0m" if self.key_s_pressed else "○ S"
        a_status = "\033[92m● A\033[0m" if self.key_a_pressed else "○ A"
        d_status = "\033[92m● D\033[0m" if self.key_d_pressed else "○ D"
        
        print(f"  {w_status}  {s_status}  {a_status}  {d_status}")
        
        # 操作提示
        print("\n" + "-"*60)
        print("操作: W-前进  S-后退  A-左转  D-右转  G-启停  Q-退出")
        print("="*60)
    
    def update_callback(self):
        """定时更新回调"""
        # 处理输入
        if not self.handle_input():
            self.shutdown_node()
            return
        
        # 计算时间步长
        dt = 1.0 / self.update_rate
        
        # 更新物理状态
        self.update_physics(dt)
        
        # 发布命令
        self.publish_commands()
        
        # 更新UI
        self.print_ui()
    
    def shutdown_node(self):
        """关闭节点"""
        print("\n\n正在关闭节点...")
        
        # 发送停止命令
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)
        
        # 禁用电机
        enable_msg = Int8()
        enable_msg.data = 0
        self.enable_pub.publish(enable_msg)
        
        # 恢复终端设置
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        
        print("节点已关闭\n")
        
        # 关闭ROS2
        rclpy.shutdown()
        sys.exit(0)
    
    def destroy_node(self):
        """清理资源"""
        try:
            self.shutdown_node()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TerminalKeyboardController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
