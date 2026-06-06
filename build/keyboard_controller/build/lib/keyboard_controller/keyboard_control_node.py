#!/usr/bin/env python3

"""
Keyboard Control Node
Uses pygame to capture keyboard input, publishes /cmd_vel and /chassis/enable topics
Control logic:
- W: Forward acceleration (hold to accelerate, release to decelerate)
- S: Constant reverse speed
- A: Turn right (immediate full turn when pressed)
- D: Turn left (immediate full turn when pressed)
- G: Toggle keyboard control handoff on/off
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int8
import pygame
import sys


class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')
        
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
        self.control_enabled = False    # 键盘控制接管状态
        self.emergency_stop_active = False
        
        # 按键状态
        self.key_w_pressed = False
        self.key_s_pressed = False
        self.key_a_pressed = False
        self.key_d_pressed = False
        
        # 发布者
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Int8, '/chassis/enable', 10)
        self.manual_override_pub = self.create_publisher(Bool, '/race/manual_override', 10)
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/race/emergency_stop',
            self.emergency_stop_callback,
            10
        )
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_mode((400, 300))
        pygame.display.set_caption('Ackermann Chassis Keyboard Control')
        pygame.key.set_repeat(100, 50)  # Key repeat
        
        # Timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.update_callback)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        self.get_logger().info('Keyboard control node started')
        self.get_logger().info('Control instructions:')
        self.get_logger().info('  W - Forward acceleration')
        self.get_logger().info('  S - Constant reverse')
        self.get_logger().info('  A - Turn right')
        self.get_logger().info('  D - Turn left')
        self.get_logger().info('  G - Toggle keyboard control')
        self.get_logger().info('  ESC - Exit')
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_w:
                    self.key_w_pressed = True
                elif event.key == pygame.K_s:
                    self.key_s_pressed = True
                elif event.key == pygame.K_a:
                    self.key_a_pressed = True
                elif event.key == pygame.K_d:
                    self.key_d_pressed = True
                elif event.key == pygame.K_g:
                    self.toggle_motor()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.key_w_pressed = False
                elif event.key == pygame.K_s:
                    self.key_s_pressed = False
                elif event.key == pygame.K_a:
                    self.key_a_pressed = False
                elif event.key == pygame.K_d:
                    self.key_d_pressed = False
        
        return True
    
    def toggle_motor(self):
        """Toggle keyboard control handoff state."""
        if self.emergency_stop_active:
            self.get_logger().warn('Emergency stop active; keyboard control remains disabled')
            return

        self.control_enabled = not self.control_enabled
        status = "Active" if self.control_enabled else "Standby"
        self.get_logger().info(f'Keyboard control: {status}')
        self.publish_manual_override(self.control_enabled)

        if self.control_enabled:
            # Keep chassis enabled when taking manual control. Releasing keyboard
            # control must not disable chassis, so autonomous controllers can keep running.
            msg = Int8()
            msg.data = 1
            self.enable_pub.publish(msg)
        else:
            # Release /cmd_vel ownership without publishing a zero command.
            self.current_speed = 0.0
            self.current_steering = 0.0
            self.key_w_pressed = False
            self.key_s_pressed = False
            self.key_a_pressed = False
            self.key_d_pressed = False

    def emergency_stop_callback(self, msg: Bool):
        """UI急停时立刻释放键盘控制，防止和停车命令抢/cmd_vel。"""
        self.emergency_stop_active = bool(msg.data)
        if not self.emergency_stop_active:
            return

        if self.control_enabled:
            self.get_logger().warn('Emergency stop received; keyboard control disabled')
        self.control_enabled = False
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.key_w_pressed = False
        self.key_s_pressed = False
        self.key_a_pressed = False
        self.key_d_pressed = False
        self.publish_manual_override(False)

    def publish_manual_override(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.manual_override_pub.publish(msg)
    
    def update_physics(self, dt):
        """Update physics state"""
        if not self.control_enabled:
            return

        # Handle forward logic
        if self.key_w_pressed:
            # Accelerate forward
            self.current_speed += self.acceleration * dt
            self.current_speed = min(self.current_speed, self.max_speed)
        elif self.key_s_pressed:
            # Constant reverse
            self.current_speed = -self.reverse_speed
        else:
            # Natural deceleration
            if self.current_speed > 0:
                self.current_speed -= self.deceleration * dt
                self.current_speed = max(self.current_speed, 0.0)
            elif self.current_speed < 0:
                self.current_speed += self.deceleration * dt
                self.current_speed = min(self.current_speed, 0.0)
        
        # Handle steering logic - immediate full turn
        if self.key_a_pressed:
            # Turn right (full right)
            self.current_steering = -1.0
        elif self.key_d_pressed:
            # Turn left (full left)
            self.current_steering = 1.0
        else:
            # Auto center immediately
            self.current_steering = 0.0
    
    def publish_commands(self):
        """Publish control commands"""
        # Only publish while keyboard control is actively taking over /cmd_vel.
        if self.control_enabled:
            twist_msg = Twist()
            twist_msg.linear.x = self.current_speed
            twist_msg.angular.z = self.current_steering  # Use steering ratio (-1.0 to 1.0)
            self.cmd_vel_pub.publish(twist_msg)
    
    def draw_ui(self):
        """Draw user interface"""
        screen = pygame.display.get_surface()
        screen.fill((30, 30, 30))  # Dark gray background
        
        # Title
        title = self.font_large.render('Ackermann Chassis Control', True, (255, 255, 255))
        screen.blit(title, (70, 20))
        
        # Motor status
        status_color = (0, 255, 0) if self.control_enabled else (255, 180, 0)
        status_text = "Keyboard: Active" if self.control_enabled else "Keyboard: Standby"
        status = self.font_medium.render(status_text, True, status_color)
        screen.blit(status, (20, 70))
        
        # Speed display
        speed_text = f"Speed: {self.current_speed:.2f} rad/s"
        speed = self.font_medium.render(speed_text, True, (255, 255, 255))
        screen.blit(speed, (20, 100))
        
        # Steering display
        steering_text = f"Steering: {self.current_steering:.2f}"
        steering = self.font_medium.render(steering_text, True, (255, 255, 255))
        screen.blit(steering, (20, 130))
        
        # Speed bar
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 160
        
        # Background bar
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        
        # Speed bar
        if self.current_speed >= 0:
            fill_width = int((self.current_speed / self.max_speed) * bar_width)
            color = (0, 200, 0)
            pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))
        else:
            fill_width = int((abs(self.current_speed) / self.reverse_speed) * bar_width)
            color = (200, 0, 0)
            pygame.draw.rect(screen, color, (bar_x + bar_width - fill_width, bar_y, fill_width, bar_height))
        
        # Steering indicator
        indicator_y = 200
        center_x = 200
        indicator_width = 160
        
        # Background
        pygame.draw.line(screen, (100, 100, 100), 
                        (center_x - indicator_width//2, indicator_y),
                        (center_x + indicator_width//2, indicator_y), 3)
        
        # Center mark
        pygame.draw.line(screen, (255, 255, 255),
                        (center_x, indicator_y - 10),
                        (center_x, indicator_y + 10), 2)
        
        # Current steering indicator
        steering_offset = int(self.current_steering * (indicator_width // 2))
        pygame.draw.circle(screen, (0, 150, 255), 
                          (center_x + steering_offset, indicator_y), 8)
        
        # Key status
        key_y = 240
        keys_text = [
            f"W: {'●' if self.key_w_pressed else '○'}",
            f"S: {'●' if self.key_s_pressed else '○'}",
            f"A: {'●' if self.key_a_pressed else '○'}",
            f"D: {'●' if self.key_d_pressed else '○'}",
        ]
        
        x_offset = 20
        for key_text in keys_text:
            text = self.font_small.render(key_text, True, (200, 200, 200))
            screen.blit(text, (x_offset, key_y))
            x_offset += 70
        
        # Help info
        help_texts = [
            "W-Fwd  S-Rev  A-Right  D-Left  G-Handoff  ESC-Exit",
        ]
        
        help_y = 270
        for help_text in help_texts:
            text = self.font_small.render(help_text, True, (150, 150, 150))
            text_rect = text.get_rect(center=(200, help_y))
            screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def update_callback(self):
        """Timer update callback"""
        # Handle events
        if not self.handle_events():
            self.shutdown_node()
            return
        
        # Calculate time step
        dt = 1.0 / self.update_rate
        
        # Update physics state
        self.update_physics(dt)
        
        # Publish commands
        self.publish_commands()
        
        # Draw UI
        self.draw_ui()
    
    def shutdown_node(self):
        """Shutdown node"""
        self.get_logger().info('Shutting down node...')
        
        if self.control_enabled:
            # If this node currently owns /cmd_vel, leave the chassis stopped.
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg)
        self.publish_manual_override(False)
        
        self.get_logger().info('Node shutdown')
        
        # Quit pygame
        pygame.quit()
        
        # Shutdown ROS2
        rclpy.shutdown()
        sys.exit(0)
    
    def destroy_node(self):
        """Clean up resources"""
        self.shutdown_node()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
