from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


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
        default_value='true',
        description='Show OpenCV fused perception preview window',
    )
    enable_status_log_arg = DeclareLaunchArgument(
        'enable_status_log',
        default_value='false',
        description='Print periodic lane decision status logs',
    )
    enable_branch_event_log_arg = DeclareLaunchArgument(
        'enable_branch_event_log',
        default_value='false',
        description='Print branch/GuideBoard decision event logs',
    )

    fused_node = Node(
        package='track_perception_cpp',
        executable='fused_perception_node',
        name='fused_perception_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'show_window': ParameterValue(LaunchConfiguration('show_window'), value_type=bool),
                'enable_status_log': ParameterValue(LaunchConfiguration('enable_status_log'), value_type=bool),
                'enable_branch_event_log': ParameterValue(
                    LaunchConfiguration('enable_branch_event_log'), value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription([
        config_file_arg,
        show_window_arg,
        enable_status_log_arg,
        enable_branch_event_log_arg,
        fused_node,
    ])
