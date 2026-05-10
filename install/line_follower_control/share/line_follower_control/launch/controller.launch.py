from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def _default_config_file():
    """Prefer the workspace src config so parameter edits do not require rebuild."""
    pkg_share = Path(get_package_share_directory('line_follower_control')).resolve()

    for parent in pkg_share.parents:
        src_config = parent / 'src' / 'line_follower_control' / 'config' / 'controller_params.yaml'
        if src_config.exists():
            return str(src_config)

    return str(pkg_share / 'config' / 'controller_params.yaml')


def generate_launch_description():
    """Launch file for line follower controller"""
    
    # 声明启动参数 - 默认使用 src 目录的配置文件（方便不编译修改参数）
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=_default_config_file(),
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
