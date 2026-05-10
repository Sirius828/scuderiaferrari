"""
UWB定位节点启动文件
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    """生成启动描述"""
    
    # 声明启动参数
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB1',
        description='UWB串口设备路径'
    )
    
    baudrate_arg = DeclareLaunchArgument(
        'baudrate',
        default_value='115200',  # LinkTrack默认波特率
        description='串口波特率'
    )
    
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='uwb_link',
        description='ROS坐标帧ID'
    )
    
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug',
        default_value='true',
        description='是否发布调试信息'
    )
    
    # 创建UWB节点
    uwb_node = Node(
        package='uwb_locator',
        executable='uwb_tag_node',
        name='uwb_tag_node',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'baudrate': LaunchConfiguration('baudrate'),
            'frame_id': LaunchConfiguration('frame_id'),
            'publish_debug': LaunchConfiguration('publish_debug'),
        }],
        remappings=[
            ('uwb/pose', '/uwb/pose'),
            ('uwb/twist', '/uwb/twist'),
            ('uwb/debug', '/uwb/debug'),
        ]
    )
    
    return LaunchDescription([
        port_arg,
        baudrate_arg,
        frame_id_arg,
        publish_debug_arg,
        uwb_node,
    ])
