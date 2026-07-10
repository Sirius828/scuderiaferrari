from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def _default_config_file():
    """Prefer the workspace src config so parameter edits do not require rebuild."""
    pkg_share = Path(get_package_share_directory('line_follower_control_cpp')).resolve()

    for parent in pkg_share.parents:
        src_config = parent / 'src' / 'line_follower_control_cpp' / 'config' / 'controller_params.yaml'
        if src_config.exists():
            return str(src_config)

    return str(pkg_share / 'config' / 'controller_params.yaml')


def _launch_controller(context):
    config_file = LaunchConfiguration('controller_config_file').perform(context)
    return [
        Node(
            package='line_follower_control_cpp',
            executable='controller_node',
            name='line_follower_controller_cpp',
            output='screen',
            parameters=[config_file],
        )
    ]


def generate_launch_description():
    config_file_arg = DeclareLaunchArgument(
        'controller_config_file',
        default_value=_default_config_file(),
        description='Path to YAML configuration file (default: src directory for no-rebuild tuning)',
    )

    return LaunchDescription([
        config_file_arg,
        LogInfo(msg=['C++ line follower config file: ', LaunchConfiguration('controller_config_file')]),
        OpaqueFunction(function=_launch_controller),
    ])
