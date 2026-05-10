#!/usr/bin/env python3
"""
终端键盘控制节点启动文件（无需图形界面，适合SSH）
注意：由于launch系统的限制，建议使用以下命令直接运行：
  ros2 run keyboard_controller terminal_keyboard_node
或使用提供的脚本：
  ./src/keyboard_controller/scripts/run_terminal_control.sh
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='keyboard_controller',
            executable='terminal_keyboard_node',
            name='terminal_keyboard_controller',
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
