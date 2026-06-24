from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch_ros.actions import Node
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


def _optional_bool_override(value, parameter_name, launch_arg_name):
    normalized = str(value).strip().lower()
    if normalized == 'yaml':
        return None
    if normalized in ('true', '1', 'yes', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(
        f'{launch_arg_name} must be true, false, or yaml; got "{value}" '
        f'for parameter {parameter_name}'
    )


def _launch_controller(context):
    """Create the node after resolving optional launch overrides."""
    config_file = LaunchConfiguration('controller_config_file').perform(context)
    overrides = {}

    auto_start = _optional_bool_override(
        LaunchConfiguration('auto_start').perform(context),
        'autonomous_enabled_on_start',
        'auto_start'
    )
    if auto_start is not None:
        overrides['autonomous_enabled_on_start'] = auto_start

    ignore_stop_requests = _optional_bool_override(
        LaunchConfiguration('ignore_stop_requests').perform(context),
        'ignore_stop_requests',
        'ignore_stop_requests'
    )
    if ignore_stop_requests is not None:
        overrides['ignore_stop_requests'] = ignore_stop_requests

    parameters = [config_file]
    if overrides:
        parameters.append(overrides)

    return [
        Node(
            package='line_follower_control',
            executable='controller_node',
            name='line_follower_controller',
            output='screen',
            parameters=parameters,
        )
    ]


def generate_launch_description():
    """Launch file for line follower controller"""
    
    # 声明启动参数 - 默认使用 src 目录的配置文件（方便不编译修改参数）
    config_file_arg = DeclareLaunchArgument(
        'controller_config_file',
        default_value=_default_config_file(),
        description='Path to YAML configuration file (default: src directory for easy modification without recompilation)'
    )
    auto_start_arg = DeclareLaunchArgument(
        'auto_start',
        default_value='yaml',
        description='Optional override for autonomous_enabled_on_start: true, false, or yaml'
    )
    ignore_stop_requests_arg = DeclareLaunchArgument(
        'ignore_stop_requests',
        default_value='yaml',
        description='Optional override for ignore_stop_requests: true, false, or yaml'
    )
    
    return LaunchDescription([
        config_file_arg,
        auto_start_arg,
        ignore_stop_requests_arg,
        LogInfo(msg=['Line follower config file: ', LaunchConfiguration('controller_config_file')]),
        OpaqueFunction(function=_launch_controller),
    ])
