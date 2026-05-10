"""
UWB定位集成示例
演示如何在底盘控制节点中使用UWB定位数据
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
import math


class UWBIntegrationExample(Node):
    """
    UWB定位集成示例节点
    
    功能：
    1. 订阅UWB位置数据
    2. 结合里程计数据进行简单融合
    3. 发布融合后的位姿估计
    """
    
    def __init__(self):
        super().__init__('uwb_integration_example')
        
        # 声明参数
        self.declare_parameter('uwb_weight', 0.3)  # UWB权重
        self.declare_parameter('odom_weight', 0.7)  # 里程计权重
        
        self.uwb_weight = self.get_parameter('uwb_weight').value
        self.odom_weight = self.get_parameter('odom_weight').value
        
        # 状态变量
        self.last_uwb_pos = None
        self.last_odom_pos = None
        self.fused_pos = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        
        # 订阅UWB位置
        self.uwb_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/uwb/pose',
            self.uwb_callback,
            10
        )
        
        # 订阅里程计
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',  # 假设底盘发布的话题
            self.odom_callback,
            10
        )
        
        # 发布融合后的位姿
        self.fused_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/fused_pose',
            10
        )
        
        # 发布控制指令（示例）
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # 创建定时器进行定期融合
        self.fusion_timer = self.create_timer(
            0.1,  # 10Hz
            self.fusion_loop
        )
        
        self.get_logger().info("UWB集成示例节点已启动")
        self.get_logger().info(f"融合权重 - UWB: {self.uwb_weight}, 里程计: {self.odom_weight}")
    
    def uwb_callback(self, msg):
        """UWB位置回调"""
        self.last_uwb_pos = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'cov_x': msg.pose.covariance[0],
            'cov_y': msg.pose.covariance[7],
            'timestamp': msg.header.stamp,
        }
        
        self.get_logger().debug(
            f"收到UWB数据: x={self.last_uwb_pos['x']:.3f}, "
            f"y={self.last_uwb_pos['y']:.3f}"
        )
    
    def odom_callback(self, msg):
        """里程计回调"""
        self.last_odom_pos = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_euler(msg.pose.pose.orientation),
            'timestamp': msg.header.stamp,
        }
    
    def quaternion_to_euler(self, q):
        """四元数转欧拉角（只计算yaw）"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def fusion_loop(self):
        """融合循环"""
        if self.last_uwb_pos is None or self.last_odom_pos is None:
            return
        
        # 简单的加权融合
        fused_x = (
            self.uwb_weight * self.last_uwb_pos['x'] +
            self.odom_weight * self.last_odom_pos['x']
        )
        fused_y = (
            self.uwb_weight * self.last_uwb_pos['y'] +
            self.odom_weight * self.last_odom_pos['y']
        )
        
        # 角度主要来自里程计
        fused_theta = self.last_odom_pos['theta']
        
        self.fused_pos = {
            'x': fused_x,
            'y': fused_y,
            'theta': fused_theta
        }
        
        # 发布融合结果
        self.publish_fused_pose()
        
        # 示例：基于位置的简单控制
        # self.navigate_to_target(1.0, 1.0)
    
    def publish_fused_pose(self):
        """发布融合后的位姿"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.pose.position.x = self.fused_pos['x']
        pose_msg.pose.pose.position.y = self.fused_pos['y']
        pose_msg.pose.pose.position.z = 0.0
        
        # 简单位姿（只有yaw旋转）
        cos_theta = math.cos(self.fused_pos['theta'] / 2)
        sin_theta = math.sin(self.fused_pos['theta'] / 2)
        pose_msg.pose.pose.orientation.x = 0.0
        pose_msg.pose.pose.orientation.y = 0.0
        pose_msg.pose.pose.orientation.z = sin_theta
        pose_msg.pose.pose.orientation.w = cos_theta
        
        # 设置协方差（简化）
        pose_msg.pose.covariance[0] = 0.01   # x方差
        pose_msg.pose.covariance[7] = 0.01   # y方差
        pose_msg.pose.covariance[35] = 0.01  # theta方差
        
        self.fused_pub.publish(pose_msg)
        
        self.get_logger().debug(
            f"发布融合位姿: x={fused_x:.3f}, y={fused_y:.3f}, "
            f"theta={math.degrees(fused_theta):.1f}°"
        )
    
    def navigate_to_target(self, target_x, target_y):
        """
        简单的导航到目标点（比例控制）
        
        Args:
            target_x: 目标X坐标
            target_y: 目标Y坐标
        """
        if self.fused_pos is None:
            return
        
        # 计算距离和角度
        dx = target_x - self.fused_pos['x']
        dy = target_y - self.fused_pos['y']
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        
        # 角度误差
        angle_error = target_angle - self.fused_pos['theta']
        # 规范化到[-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # 比例控制
        k_linear = 0.5   # 线速度增益
        k_angular = 1.0  # 角速度增益
        
        linear_vel = k_linear * distance
        angular_vel = k_angular * angle_error
        
        # 限制最大速度
        max_linear = 0.5
        max_angular = 1.0
        linear_vel = max(-max_linear, min(max_linear, linear_vel))
        angular_vel = max(-max_angular, min(max_angular, angular_vel))
        
        # 到达判断
        if distance < 0.1 and abs(angle_error) < 0.1:
            linear_vel = 0.0
            angular_vel = 0.0
            self.get_logger().info("已到达目标点！")
        
        # 发布速度指令
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)
        
        self.get_logger().debug(
            f"导航: 距离={distance:.2f}m, "
            f"线速度={linear_vel:.2f}m/s, 角速度={angular_vel:.2f}rad/s"
        )


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = UWBIntegrationExample()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
