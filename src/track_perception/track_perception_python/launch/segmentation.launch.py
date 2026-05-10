from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for semantic segmentation node"""
    
    # 声明启动参数
    shm_name_arg = DeclareLaunchArgument(
        'shm_name',
        default_value='shm_ar_video',
        description='Shared memory name for video stream'
    )
    
    model_dir_arg = DeclareLaunchArgument(
        'model_dir',
        default_value='model',
        description='Directory containing the RKNN model'
    )
    
    tpes_arg = DeclareLaunchArgument(
        'tpes',
        default_value='3',
        description='Number of thread pool executors'
    )
    
    blend_alpha_arg = DeclareLaunchArgument(
        'blend_alpha',
        default_value='-1.0',  # ⭐ 默认不混合，只显示分割结果
        description='Blend alpha (0-1) for overlay, -1 for segmentation only'
    )
    
    show_window_arg = DeclareLaunchArgument(
        'show_window',
        default_value='true',  # ⭐ 默认不显示窗口（巡线模式）
        description='Show visualization window'
    )
    
    show_debug_window_arg = DeclareLaunchArgument(
        'show_debug_window',
        default_value='false',
        description='Show debug window with performance stats'
    )
    
    enable_perf_stats_arg = DeclareLaunchArgument(
        'enable_perf_stats',
        default_value='false',
        description='Enable detailed performance statistics'
    )
    
    # ⭐ 第二轮优化：新增参数
    enable_flip_arg = DeclareLaunchArgument(
        'enable_flip',
        default_value='true',  # ⭐ 默认禁用flip（如果共享内存已正确方向）
        description='Enable image flip'
    )
    
    flip_code_arg = DeclareLaunchArgument(
        'flip_code',
        default_value='0',
        description='Flip code: 0=vertical, 1=horizontal, -1=both'
    )
    
    input_format_arg = DeclareLaunchArgument(
        'input_format',
        default_value='RGB',
        description='Input image format: RGB or BGR'
    )
    
    model_input_format_arg = DeclareLaunchArgument(
        'model_input_format',
        default_value='RGB',
        description='Model input format: RGB or BGR'
    )
    
    # ⭐ 第三轮优化：新增参数
    publish_mask_arg = DeclareLaunchArgument(
        'publish_mask',
        default_value='false',  # ⭐ 默认不发布mask图像，发布Float32 offset
        description='Enable mask publishing (false=发布Float32 offset)'
    )
    
    publish_mask_every_n_arg = DeclareLaunchArgument(
        'publish_mask_every_n',
        default_value='1',
        description='Publish mask every N frames'
    )
    
    # ⭐ 岔路口检测与外圈锁定参数
    enable_intersection_logic_arg = DeclareLaunchArgument(
        'enable_intersection_logic',
        default_value='true',
        description='Enable intersection detection and outer branch locking'
    )
    
    outer_side_arg = DeclareLaunchArgument(
        'outer_side',
        default_value='left',
        description='Outer side at intersection: left or right'
    )
    
    far_roi_y0_ratio_arg = DeclareLaunchArgument(
        'far_roi_y0_ratio',
        default_value='0.35',  # ⭐ 与YAML配置保持一致
        description='Far ROI start y ratio (0.0-1.0)'
    )
    
    far_roi_y1_ratio_arg = DeclareLaunchArgument(
        'far_roi_y1_ratio',
        default_value='0.75',  # ⭐ 与YAML配置保持一致
        description='Far ROI end y ratio (0.0-1.0)'
    )
    
    near_roi_y0_ratio_arg = DeclareLaunchArgument(
        'near_roi_y0_ratio',
        default_value='0.75',  # ⭐ 与YAML配置保持一致
        description='Near ROI start y ratio (0.0-1.0)'
    )
    
    near_roi_y1_ratio_arg = DeclareLaunchArgument(
        'near_roi_y1_ratio',
        default_value='0.95',
        description='Near ROI end y ratio (0.0-1.0)'
    )
    
    far_width_threshold_arg = DeclareLaunchArgument(
        'far_width_threshold',
        default_value='0.65',
        description='Far intersection width threshold (ratio of image width)'
    )
    
    near_width_threshold_arg = DeclareLaunchArgument(
        'near_width_threshold',
        default_value='0.60',
        description='Near intersection width threshold (ratio of image width)'
    )
    
    exit_width_threshold_arg = DeclareLaunchArgument(
        'exit_width_threshold',
        default_value='0.45',
        description='Exit lock width threshold (ratio of image width)'
    )
    
    exit_confirm_frames_arg = DeclareLaunchArgument(
        'exit_confirm_frames',
        default_value='5',
        description='Number of frames to confirm exit from lock state'
    )
    
    max_lock_time_arg = DeclareLaunchArgument(
        'max_lock_time',
        default_value='2.0',
        description='Maximum lock duration in seconds'
    )
    
    min_road_pixels_arg = DeclareLaunchArgument(
        'min_road_pixels',
        default_value='300',
        description='Minimum road pixels to consider valid detection'
    )
    
    # YAML配置文件路径参数
    use_config_arg = DeclareLaunchArgument(
        'use_config',
        default_value='true',  # ⭐ 默认启用YAML配置文件
        description='Use YAML config file for parameters'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            get_package_share_directory('semantic_segmentation'),
            'config',
            'intersection_params.yaml'
        ),
        description='Path to YAML config file'
    )
    
    # 创建节点 - ⭐ 默认加载YAML配置文件
    segmentation_node = Node(
        package='semantic_segmentation',
        executable='segmentation_node',
        name='semantic_segmentation_node',
        output='screen',
        parameters=[
            # ⭐ YAML配置文件（主要配置来源）
            LaunchConfiguration('config_file'),
            # ⭐ 只传递基础参数（不会覆盖YAML中的岔路口检测参数）
            {
                'shm_name': LaunchConfiguration('shm_name'),
                'model_dir': LaunchConfiguration('model_dir'),
                'tpes': LaunchConfiguration('tpes'),
                'blend_alpha': LaunchConfiguration('blend_alpha'),
                'show_window': LaunchConfiguration('show_window'),
                'show_debug_window': LaunchConfiguration('show_debug_window'),
                'enable_perf_stats': LaunchConfiguration('enable_perf_stats'),
                'enable_flip': LaunchConfiguration('enable_flip'),
                'flip_code': LaunchConfiguration('flip_code'),
                'input_format': LaunchConfiguration('input_format'),
                'model_input_format': LaunchConfiguration('model_input_format'),
                'publish_mask': LaunchConfiguration('publish_mask'),
                'publish_mask_every_n': LaunchConfiguration('publish_mask_every_n'),
                # ⚠️ 注意：岔路口检测参数完全由YAML控制，不在这里传递
                # 如果需要覆盖，请在命令行显式指定，例如：
                # ros2 launch ... outer_side:=right
            }
        ],
    )
    
    return LaunchDescription([
        shm_name_arg,
        model_dir_arg,
        tpes_arg,
        blend_alpha_arg,
        show_window_arg,
        show_debug_window_arg,  # 新增
        enable_perf_stats_arg,  # 性能统计
        enable_flip_arg,  # ⭐ 第二轮优化
        flip_code_arg,  # ⭐ 第二轮优化
        input_format_arg,  # ⭐ 第二轮优化
        model_input_format_arg,  # ⭐ 第二轮优化
        publish_mask_arg,  # ⭐ 第三轮优化
        publish_mask_every_n_arg,  # ⭐ 第三轮优化
        # ⭐ 岔路口检测参数
        enable_intersection_logic_arg,
        outer_side_arg,
        far_roi_y0_ratio_arg,
        far_roi_y1_ratio_arg,
        near_roi_y0_ratio_arg,
        near_roi_y1_ratio_arg,
        far_width_threshold_arg,
        near_width_threshold_arg,
        exit_width_threshold_arg,
        exit_confirm_frames_arg,
        max_lock_time_arg,
        min_road_pixels_arg,
        # ⭐ YAML配置文件参数
        use_config_arg,
        config_file_arg,
        segmentation_node,
    ])
