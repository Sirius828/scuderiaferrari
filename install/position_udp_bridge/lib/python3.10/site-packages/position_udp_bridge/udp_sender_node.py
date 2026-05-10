#!/usr/bin/env python3
"""
位置数据UDP发送节点
订阅融合后的UWB位置和偏航角数据，通过UDP发送到指定目标
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32
import socket
import json
import time


class PositionUDPSender(Node):
    """位置数据UDP发送节点"""
    
    def __init__(self):
        super().__init__('position_udp_sender')
        
        # 声明参数
        self.declare_parameter('udp_host', '172.20.10.2')
        self.declare_parameter('udp_port', 9005)
        self.declare_parameter('pose_topic', '/uwb/pose')
        self.declare_parameter('yaw_topic', '/yaw_angle')
        self.declare_parameter('swap_yz', True)  # 是否交换Y和Z坐标
        
        # 获取参数
        self.udp_host = self.get_parameter('udp_host').get_parameter_value().string_value
        self.udp_port = self.get_parameter('udp_port').get_parameter_value().integer_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.yaw_topic = self.get_parameter('yaw_topic').get_parameter_value().string_value
        self.swap_yz = self.get_parameter('swap_yz').get_parameter_value().bool_value
        
        # 创建UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 状态变量
        self.current_pose = None
        self.current_yaw = 0.0  # 度
        
        # 订阅UWB位置
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.pose_topic,
            self._pose_callback,
            10
        )
        
        # 订阅偏航角
        self.yaw_sub = self.create_subscription(
            Float32,
            self.yaw_topic,
            self._yaw_callback,
            10
        )
        
        # 创建定时器定期发送数据
        self.timer = self.create_timer(0.05, self._publish_loop)  # 20Hz
        
        # 统计信息
        self.send_count = 0
        self.last_log_time = 0.0
        
        self.get_logger().info(f"UDP发送节点已启动")
        self.get_logger().info(f"位置话题: {self.pose_topic}")
        self.get_logger().info(f"偏航角话题: {self.yaw_topic}")
        self.get_logger().info(f"目标地址: {self.udp_host}:{self.udp_port}")
        self.get_logger().info(f"坐标系转换: {'Y/Z交换' if self.swap_yz else '无转换'}")
    
    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """位置数据回调"""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
        }
    
    def _yaw_callback(self, msg: Float32):
        """偏航角回调"""
        self.current_yaw = -msg.data  # 直接使用度，不需要转换
        
        # 每秒打印一次日志
        now = time.time()
        if now - self.last_log_time >= 1.0:
            self.get_logger().info(f"📐 偏航角: {msg.data:.2f}°")
            self.last_log_time = now
    
    def _publish_loop(self):
        """定时发布循环"""
        if self.current_pose is None:
            self.get_logger().warn_throttle(5, "等待UWB定位数据...")
            return
        
        # 位置数据处理：Y固定为0
        pos_x = self.current_pose['x']
        pos_y = 0.16  # Y固定为0
        pos_z = self.current_pose['y']
        
        # 角度处理：euler[1] (pitch位置) 是yaw角
        euler_roll = 0.0
        euler_pitch = self.current_yaw  # yaw角放在第二个位置
        # euler_pitch = 0.0
        euler_yaw = 0.0
        
        # 构建JSON数据（扁平数组格式）
        data = {
            "type": "robot_position",
            "pos": [
                round(pos_x, 2),
                round(pos_y, 2),
                round(pos_z, 2)
            ],
            "euler": [
                round(euler_roll, 2),   # roll (固定0)
                round(euler_pitch, 2),  # pitch位置放yaw角
                round(euler_yaw, 2)     # yaw (固定0)
            ]
        }
        
        # 转换为JSON字符串
        json_str = json.dumps(data, ensure_ascii=False)
        
        # 发送UDP数据
        try:
            self.sock.sendto(json_str.encode('utf-8'), (self.udp_host, self.udp_port))
            self.send_count += 1
            
            # 打印每次发送的内容
            self.get_logger().info(
                f"📤 [{self.send_count}] {json_str}"
            )
        
        except Exception as e:
            self.get_logger().error(f"UDP发送失败: {e}")
    
    def destroy_node(self):
        """销毁节点"""
        self.sock.close()
        self.get_logger().info(f"总共发送 {self.send_count} 个UDP数据包")
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = PositionUDPSender()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"致命错误: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
