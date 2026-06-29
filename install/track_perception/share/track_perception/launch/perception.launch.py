from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    """Launch object detection and perception decision nodes."""
    workspace_root = '/home/orangepi/scuderiaferrari'
    default_config_file = os.path.join(
        workspace_root,
        'src',
        'track_perception',
        'config',
        'intersection_params.yaml',
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to the track perception YAML config file',
    )

    object_detection_node = Node(
        package='track_perception',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    perception_decision_node = Node(
        package='track_perception',
        executable='perception_decision_node',
        name='perception_decision_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([
        config_file_arg,
        object_detection_node,
        perception_decision_node,
    ])
