#!/usr/bin/env python3
"""
键盘控制节点启动文件
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='keyboard_controller',
            executable='keyboard_control_node',
            name='keyboard_controller',
            output='screen',
            parameters=[
                {'max_speed': 10.0},
                {'reverse_speed': 2.0},
                {'acceleration': 5.0},
                {'deceleration': 5.0},
                {'turn_rate': 1.0},
                {'update_rate': 20.0}
            ]
        )
    ])
