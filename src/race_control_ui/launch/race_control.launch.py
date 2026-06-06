from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def _launch_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def _src_file(package_name, relative_path):
    """Prefer workspace src files so config edits do not require rebuild."""
    pkg_share = Path(get_package_share_directory(package_name)).resolve()
    for parent in pkg_share.parents:
        candidate = parent / 'src' / package_name / relative_path
        if candidate.exists():
            return str(candidate)
    return str(pkg_share / relative_path)


def generate_launch_description():
    chassis_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_launch_file('chassis_controller', 'launch/chassis_controller.launch.py'))
    )
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_launch_file('track_perception', 'launch/perception.launch.py')),
        launch_arguments={
            'config_file': _src_file('track_perception', 'config/intersection_params.yaml'),
        }.items()
    )
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_launch_file('line_follower_control', 'launch/controller.launch.py')),
        launch_arguments={
            'controller_config_file': _src_file('line_follower_control', 'config/controller_params.yaml'),
        }.items()
    )
    ui_node = Node(
        package='race_control_ui',
        executable='race_control_ui',
        name='race_control_ui',
        output='screen',
    )

    return LaunchDescription([
        chassis_launch,
        perception_launch,
        controller_launch,
        ui_node,
    ])
