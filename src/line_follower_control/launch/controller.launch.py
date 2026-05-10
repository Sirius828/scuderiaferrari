from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Launch file for line follower controller"""
    
    # 获取包的路径
    pkg_share = get_package_share_directory('line_follower_control')
    
    # 声明启动参数 - 默认使用 src 目录的配置文件（方便不编译修改参数）
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='/home/orangepi/scuderiaferrari/src/line_follower_control/config/controller_params.yaml',
        description='Path to YAML configuration file (default: src directory for easy modification without recompilation)'
    )
    
    # 创建节点
    controller_node = Node(
        package='line_follower_control',
        executable='controller_node',
        name='line_follower_controller',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )
    
    return LaunchDescription([
        config_file_arg,
        controller_node,
    ])
