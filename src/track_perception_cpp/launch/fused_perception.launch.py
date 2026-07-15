from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def _launch_fused_perception(context):
    parameter_overrides = {
        'enable_debug_screenshots': ParameterValue(
            LaunchConfiguration('enable_debug_screenshots'), value_type=bool
        ),
        'debug_screenshot_interval_sec': ParameterValue(
            LaunchConfiguration('debug_screenshot_interval_sec'), value_type=float
        ),
        'debug_screenshot_branch_only': ParameterValue(
            LaunchConfiguration('debug_screenshot_branch_only'), value_type=bool
        ),
        'debug_screenshot_dir': LaunchConfiguration('debug_screenshot_dir'),
    }

    # An empty launch argument means "use the value from the YAML file".  Only
    # an explicit show_window:=true/false should override that file.
    show_window_override = LaunchConfiguration('show_window').perform(context).strip().lower()
    if show_window_override:
        if show_window_override not in ('true', 'false'):
            raise ValueError('show_window must be true, false, or omitted to use YAML')
        # The value has already been resolved by OpaqueFunction.  Pass a real
        # bool here; wrapping the resolved string in ParameterValue leaves a
        # string substitution and makes show_window:=false fail type checking.
        parameter_overrides['show_window'] = show_window_override == 'true'

    return [
        Node(
            package='track_perception_cpp',
            executable='fused_perception_node',
            name='fused_perception_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                parameter_overrides,
            ],
        )
    ]


def generate_launch_description():
    share_config_file = os.path.join(
        get_package_share_directory('track_perception_cpp'),
        'config',
        'fused_perception.yaml',
    )
    source_config_file = '/home/orangepi/scuderiaferrari/src/track_perception_cpp/config/fused_perception.yaml'
    default_config_file = source_config_file if os.path.exists(source_config_file) else share_config_file

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to the perception YAML config file',
    )
    show_window_arg = DeclareLaunchArgument(
        'show_window',
        default_value='',
        description='Optional true/false override; empty uses show_window from the YAML file',
    )
    enable_debug_screenshots_arg = DeclareLaunchArgument(
        'enable_debug_screenshots',
        default_value='false',
        description='Save rendered debug screenshots',
    )
    debug_screenshot_interval_sec_arg = DeclareLaunchArgument(
        'debug_screenshot_interval_sec',
        default_value='0.0',
        description='Screenshot interval in seconds; 0 disables automatic screenshots',
    )
    debug_screenshot_branch_only_arg = DeclareLaunchArgument(
        'debug_screenshot_branch_only',
        default_value='false',
        description='Only save automatic screenshots while branch is locked',
    )
    debug_screenshot_dir_arg = DeclareLaunchArgument(
        'debug_screenshot_dir',
        default_value='/tmp/fused_perception_screenshots',
        description='Directory for debug screenshots',
    )

    return LaunchDescription([
        config_file_arg,
        show_window_arg,
        enable_debug_screenshots_arg,
        debug_screenshot_interval_sec_arg,
        debug_screenshot_branch_only_arg,
        debug_screenshot_dir_arg,
        OpaqueFunction(function=_launch_fused_perception),
    ])
