#!/usr/bin/env python3
"""
UDP发送节点启动文件
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # 声明启动参数
    udp_host_arg = DeclareLaunchArgument(
        'udp_host',
        default_value='172.20.10.2',
        description='UDP目标服务器地址'
    )
    
    udp_port_arg = DeclareLaunchArgument(
        'udp_port',
        default_value='9005',
        description='UDP目标端口'
    )
    
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/uwb/pose',
        description='UWB位置话题'
    )
    
    yaw_topic_arg = DeclareLaunchArgument(
        'yaw_topic',
        default_value='/yaw_angle',
        description='偏航角话题'
    )
    
    swap_yz_arg = DeclareLaunchArgument(
        'swap_yz',
        default_value='true',
        description='是否交换Y和Z坐标'
    )
    
    return LaunchDescription([
        udp_host_arg,
        udp_port_arg,
        pose_topic_arg,
        yaw_topic_arg,
        swap_yz_arg,
        Node(
            package='position_udp_bridge',
            executable='udp_sender_node',
            name='position_udp_sender',
            output='screen',
            parameters=[{
                'udp_host': LaunchConfiguration('udp_host'),
                'udp_port': LaunchConfiguration('udp_port'),
                'pose_topic': LaunchConfiguration('pose_topic'),
                'yaw_topic': LaunchConfiguration('yaw_topic'),
                'swap_yz': LaunchConfiguration('swap_yz'),
            }]
        )
    ])
