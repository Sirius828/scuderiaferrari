"""
UWB定位标签ROS2节点
从串口读取LinkTrack UWB数据，融合偏航角，并发布为ROS2消息
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from std_msgs.msg import String, Float32
import math
import json

from .serial_reader import UWBSerialReader
from .frame_parser import UWBFrameParser, NodeFrame2


class UWBTagNode(Node):
    """UWB定位标签节点"""
    
    def __init__(self):
        super().__init__('uwb_tag_node')
        
        # 声明参数
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('frame_id', 'uwb_link')
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('yaw_topic', '/yaw_angle')
        
        # 获取参数
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.publish_debug = self.get_parameter('publish_debug').get_parameter_value().bool_value
        self.yaw_topic = self.get_parameter('yaw_topic').get_parameter_value().string_value
        
        # 偏航角数据（度）
        self.current_yaw_deg = 0.0
        self.yaw_received = False
        
        # 创建发布者
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            'uwb/pose', 
            10
        )
        self.twist_pub = self.create_publisher(
            TwistStamped, 
            'uwb/twist', 
            10
        )
        # 发布融合后的位置和偏航角数据（JSON格式）
        self.fused_pose_pub = self.create_publisher(
            String,
            'uwb/pose_with_yaw',
            10
        )
        self.debug_pub = self.create_publisher(
            String, 
            'uwb/debug', 
            10
        )
        
        # 订阅偏航角话题
        self.yaw_sub = self.create_subscription(
            Float32,
            self.yaw_topic,
            self._yaw_callback,
            10
        )
        
        # 初始化串口和解析器
        self.serial_reader = UWBSerialReader(self.port, self.baudrate)
        self.frame_parser = UWBFrameParser()
        
        # 设置数据回调
        self.serial_reader.set_data_callback(self._on_serial_data)
        
        # 打开串口
        if not self.serial_reader.open():
            self.get_logger().error(f"无法打开串口: {self.port}")
            raise RuntimeError(f"Failed to open serial port: {self.port}")
        
        # 启动读取线程
        self.serial_reader.start()
        
        self.get_logger().info(f"UWB TAG节点已启动 - 端口: {self.port}, 波特率: {self.baudrate}")
        self.get_logger().info(f"订阅偏航角话题: {self.yaw_topic}")
    
    def _on_serial_data(self, data: bytes):
        """串口数据接收回调"""
        # 添加数据到解析器缓冲区
        self.frame_parser.add_data(data)
        
        # 尝试解析帧
        frames = self.frame_parser.parse_frames()
        
        # 发布解析出的帧
        for frame in frames:
            self._publish_frame(frame)
    
    def _yaw_callback(self, msg: Float32):
        """偏航角数据回调"""
        self.current_yaw_deg = msg.data
        self.yaw_received = True
        self.get_logger().debug(f"收到偏航角: {self.current_yaw_deg:.2f}°")
    
    def _publish_frame(self, frame: NodeFrame2):
        """发布UWB数据帧为ROS2消息"""
        now = self.get_clock().now().to_msg()
        
        # 发布位置消息
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.pose.position.x = frame.pos_x
        pose_msg.pose.pose.position.y = frame.pos_y
        pose_msg.pose.pose.position.z = frame.pos_z
        
        # 如果有偏航角，设置朝向
        if self.yaw_received:
            yaw_rad = math.radians(self.current_yaw_deg)
            pose_msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
            pose_msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
        else:
            pose_msg.pose.pose.orientation.w = 1.0  # 默认朝向
        
        # 根据EOP设置协方差（用于EKF融合）
        var_x = max(0.01, frame.eop_x * frame.eop_x)
        var_y = max(0.01, frame.eop_y * frame.eop_y)
        var_z = max(0.04, frame.eop_z * frame.eop_z)
        
        # 协方差矩阵对角线元素
        pose_msg.pose.covariance[0] = var_x   # x方差
        pose_msg.pose.covariance[7] = var_y   # y方差
        pose_msg.pose.covariance[14] = var_z  # z方差
        pose_msg.pose.covariance[21] = 9999.0  # roll方差（未测量）
        pose_msg.pose.covariance[28] = 9999.0  # pitch方差（未测量）
        pose_msg.pose.covariance[35] = 9999.0  # yaw方差（未测量）
        
        self.pose_pub.publish(pose_msg)
        
        # 发布速度消息
        twist_msg = TwistStamped()
        twist_msg.header.stamp = now
        twist_msg.header.frame_id = self.frame_id
        twist_msg.twist.linear.x = frame.vel_x
        twist_msg.twist.linear.y = frame.vel_y
        twist_msg.twist.linear.z = frame.vel_z
        
        self.twist_pub.publish(twist_msg)
        
        # 发布融合后的位置和偏航角数据（JSON格式）
        fused_data = {
            "robot_position": {
                "pos": {
                    "x": round(frame.pos_x, 4),
                    "y": round(frame.pos_y, 4),
                    "z": round(frame.pos_z, 4)
                },
                "euler": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": round(self.current_yaw_deg, 4) if self.yaw_received else 0.0
                }
            },
            "velocity": {
                "vx": round(frame.vel_x, 4),
                "vy": round(frame.vel_y, 4),
                "vz": round(frame.vel_z, 4)
            },
            "quality": {
                "eop_x": round(frame.eop_x, 4),
                "eop_y": round(frame.eop_y, 4),
                "eop_z": round(frame.eop_z, 4),
                "anchors": frame.valid_node_quantity
            },
            "timestamp": now.sec + now.nanosec / 1e9
        }
        
        fused_msg = String()
        fused_msg.data = json.dumps(fused_data, ensure_ascii=False)
        self.fused_pose_pub.publish(fused_msg)
        
        # 发布调试信息
        if self.publish_debug:
            debug_msg = String()
            debug_dict = self.frame_parser.frame_to_dict(frame)
            debug_msg.data = json.dumps(debug_dict, ensure_ascii=False)
            self.debug_pub.publish(debug_msg)
            
            # 打印简要信息到日志
            yaw_str = f"{self.current_yaw_deg:.2f}°" if self.yaw_received else "N/A"
            self.get_logger().debug(
                f"Pos: [{frame.pos_x:.3f}, {frame.pos_y:.3f}, {frame.pos_z:.3f}] | "
                f"Yaw: {yaw_str} | "
                f"Vel: [{frame.vel_x:.3f}, {frame.vel_y:.3f}, {frame.vel_z:.3f}] | "
                f"Anchors: {frame.valid_node_quantity}"
            )
    
    def destroy_node(self):
        """销毁节点"""
        self.serial_reader.close()
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = UWBTagNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"致命错误: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
