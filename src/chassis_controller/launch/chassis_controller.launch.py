#!/usr/bin/env python3
"""
底盘控制节点启动文件
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='chassis_controller',
            executable='chassis_controller_node',
            name='chassis_controller',
            output='screen',
            parameters=[
                {'serial_port': '/dev/ttyS0'},
                {'baudrate': 115200},
                {'max_speed': 10.0},
                {'servo_center': 3000},
                {'servo_left_max': 2300},
                {'servo_right_max': 3700},
                {'cmd_vel_timeout_sec': 0.20}
            ]
        )
    ])
