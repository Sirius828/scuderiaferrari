from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for track perception system (object detection + semantic segmentation decision)"""
    
    # ==================== 共享内存参数 ====================
    shm_name_arg = DeclareLaunchArgument(
        'shm_name',
        default_value='shm_ar_video',
        description='Shared memory name for video stream'
    )
    
    # ==================== 目标检测节点参数 ====================
    det_model_path_arg = DeclareLaunchArgument(
        'det_model_path',
        default_value='model/rknn_lt.rknn',
        description='Path to object detection RKNN model'
    )
    
    det_label_list_arg = DeclareLaunchArgument(
        'det_label_list_path',
        default_value='model/label_list.txt',
        description='Path to label list file'
    )
    
    det_tpes_arg = DeclareLaunchArgument(
        'det_tpes',
        default_value='3',
        description='Number of TPEs for object detection'
    )
    
    det_enable_flip_arg = DeclareLaunchArgument(
        'det_enable_flip',
        default_value='true',
        description='Enable image flip for object detection'
    )
    
    det_flip_code_arg = DeclareLaunchArgument(
        'det_flip_code',
        default_value='0',
        description='Flip code for object detection: 0=vertical, 1=horizontal, -1=both'
    )
    
    det_input_format_arg = DeclareLaunchArgument(
        'det_input_format',
        default_value='RGB',
        description='Input image format for object detection: RGB or BGR'
    )
    
    det_publish_rate_arg = DeclareLaunchArgument(
        'det_publish_rate',
        default_value='30',  # ⭐ 提高到 30Hz，匹配异步推理的高吞吐量
        description='Object detection publish rate (Hz)'
    )
    
    # ==================== 语义分割决策节点参数 ====================
    seg_model_dir_arg = DeclareLaunchArgument(
        'seg_model_dir',
        default_value='model',
        description='Directory containing semantic segmentation model'
    )
    
    seg_model_filename_arg = DeclareLaunchArgument(
        'seg_model_filename',
        default_value='pp_liteseg.rknn',  # ⭐ 默认使用旧版本，YAML可覆盖
        description='Semantic segmentation model filename'
    )
    
    seg_tpes_arg = DeclareLaunchArgument(
        'seg_tpes',
        default_value='3',
        description='Number of TPEs for semantic segmentation'
    )
    
    blend_alpha_arg = DeclareLaunchArgument(
        'blend_alpha',
        default_value='-1.0',
        description='Blend alpha for segmentation visualization (-1.0 = no blend)'
    )
    
    seg_enable_flip_arg = DeclareLaunchArgument(
        'seg_enable_flip',
        default_value='true',
        description='Enable image flip for segmentation'
    )
    
    seg_flip_code_arg = DeclareLaunchArgument(
        'seg_flip_code',
        default_value='0',
        description='Flip code for segmentation: 0=vertical, 1=horizontal, -1=both'
    )
    
    seg_input_format_arg = DeclareLaunchArgument(
        'seg_input_format',
        default_value='RGB',
        description='Input image format for segmentation: RGB or BGR'
    )
    
    seg_model_input_format_arg = DeclareLaunchArgument(
        'seg_model_input_format',
        default_value='RGB',
        description='Model input format for segmentation: RGB or BGR'
    )
    
    show_window_arg = DeclareLaunchArgument(
        'show_window',
        default_value='false',  # ⭐ 默认关闭可视化，需要时通过命令行参数开启
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
    
    # ==================== 岔路口检测参数 ====================
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
        default_value='0.35',
        description='Far ROI start y ratio (0.0-1.0)'
    )
    
    far_roi_y1_ratio_arg = DeclareLaunchArgument(
        'far_roi_y1_ratio',
        default_value='0.75',
        description='Far ROI end y ratio (0.0-1.0)'
    )
    
    near_roi_y0_ratio_arg = DeclareLaunchArgument(
        'near_roi_y0_ratio',
        default_value='0.75',
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
    
    # ⭐ GuideBoard 岔路选择参数
    enable_guideboard_branch_selection_arg = DeclareLaunchArgument(
        'enable_guideboard_branch_selection',
        default_value='true',
        description='Enable GuideBoard-based branch selection at intersection'
    )
    
    guideboard_branch_arg = DeclareLaunchArgument(
        'guideboard_branch',
        default_value='right',
        description='Branch direction when GuideBoard is detected: left or right'
    )
    
    # ⭐ 高级岔路检测与中心线拟合参数（新逻辑）
    enable_segment_branch_logic_arg = DeclareLaunchArgument(
        'enable_segment_branch_logic',
        default_value='true',
        description='Enable multi-band scanning and centerline fitting logic'
    )
    
    band_count_arg = DeclareLaunchArgument(
        'band_count',
        default_value='8',
        description='Number of scanning bands'
    )
    
    band_y_min_ratio_arg = DeclareLaunchArgument(
        'band_y_min_ratio',
        default_value='0.25',
        description='Band scanning start y ratio (0.0=top, 1.0=bottom)'
    )
    
    band_y_max_ratio_arg = DeclareLaunchArgument(
        'band_y_max_ratio',
        default_value='0.95',
        description='Band scanning end y ratio'
    )
    
    band_height_ratio_arg = DeclareLaunchArgument(
        'band_height_ratio',
        default_value='0.04',
        description='Height ratio of each band'
    )
    
    min_segment_width_px_arg = DeclareLaunchArgument(
        'min_segment_width_px',
        default_value='25',
        description='Minimum segment width in pixels'
    )
    
    min_segment_gap_px_arg = DeclareLaunchArgument(
        'min_segment_gap_px',
        default_value='40',
        description='Minimum gap between segments in pixels'
    )
    
    min_pixels_per_band_arg = DeclareLaunchArgument(
        'min_pixels_per_band',
        default_value='80',
        description='Minimum pixels per band for valid segment'
    )
    
    branch_detect_min_bands_arg = DeclareLaunchArgument(
        'branch_detect_min_bands',
        default_value='3',
        description='Minimum bands with 2+ segments to detect branch'
    )
    
    branch_detect_far_band_ratio_arg = DeclareLaunchArgument(
        'branch_detect_far_band_ratio',
        default_value='0.6',
        description='Ratio of far bands used for branch detection'
    )
    
    branch_lock_time_arg = DeclareLaunchArgument(
        'branch_lock_time',
        default_value='2.0',
        description='Maximum branch lock duration in seconds'
    )
    
    exit_single_path_confirm_frames_arg = DeclareLaunchArgument(
        'exit_single_path_confirm_frames',
        default_value='5',
        description='Frames to confirm single path before exiting lock'
    )
    
    fit_min_points_arg = DeclareLaunchArgument(
        'fit_min_points',
        default_value='4',
        description='Minimum points for centerline fitting'
    )
    
    fit_order_arg = DeclareLaunchArgument(
        'fit_order',
        default_value='1',
        description='Fitting order: 1=linear, 2=quadratic'
    )
    
    use_heading_term_arg = DeclareLaunchArgument(
        'use_heading_term',
        default_value='true',
        description='Use heading error term in offset calculation'
    )
    
    heading_weight_arg = DeclareLaunchArgument(
        'heading_weight',
        default_value='0.35',
        description='Weight of heading error term'
    )
    
    near_offset_weight_arg = DeclareLaunchArgument(
        'near_offset_weight',
        default_value='0.65',
        description='Weight of near offset term'
    )
    
    max_offset_jump_arg = DeclareLaunchArgument(
        'max_offset_jump',
        default_value='0.6',
        description='Maximum offset jump limit'
    )
    
    offset_smoothing_alpha_arg = DeclareLaunchArgument(
        'offset_smoothing_alpha',
        default_value='0.4',
        description='Smoothing alpha for low-pass filter'
    )
    
    publish_debug_info_arg = DeclareLaunchArgument(
        'publish_debug_info',
        default_value='true',
        description='Publish debug information'
    )
    
    show_branch_debug_arg = DeclareLaunchArgument(
        'show_branch_debug',
        default_value='false',
        description='Show band and segment debugging visualization'
    )
    
    # YAML配置文件路径参数
    use_config_arg = DeclareLaunchArgument(
        'use_config',
        default_value='true',
        description='Use YAML config file for parameters'
    )
    
    # ⭐ 默认使用 src 目录的 YAML，修改后无需编译即可生效
    # 通过查找工作区根目录来定位 src 中的配置文件
    import subprocess
    try:
        # 获取当前工作区根目录
        result = subprocess.run(['colcon', 'list', '--paths-only'], 
                              capture_output=True, text=True, cwd='/home/orangepi/scuderiaferrari')
        if result.returncode == 0 and result.stdout.strip():
            workspace_root = '/home/orangepi/scuderiaferrari'
        else:
            workspace_root = '/home/orangepi/scuderiaferrari'
    except:
        workspace_root = '/home/orangepi/scuderiaferrari'
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            workspace_root,
            'src',
            'track_perception',
            'track_perception_python',
            'config',
            'intersection_params.yaml'
        ),
        description='Path to YAML config file (defaults to src directory)'
    )
    
    # ==================== 创建节点 ====================
    
    # 1. 目标检测节点
    object_detection_node = Node(
        package='track_perception',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        parameters=[
            # ⭐ 其他参数（先加载，优先级低）
            {
                'shm_name': LaunchConfiguration('shm_name'),
                'model_path': LaunchConfiguration('det_model_path'),
                'label_list_path': LaunchConfiguration('det_label_list_path'),
                'tpes': LaunchConfiguration('det_tpes'),
                'enable_flip': LaunchConfiguration('det_enable_flip'),
                'flip_code': LaunchConfiguration('det_flip_code'),
                'input_format': LaunchConfiguration('det_input_format'),
                'publish_detections': True,
                'publish_rate': LaunchConfiguration('det_publish_rate'),
            },
            # ⭐ YAML配置文件（后加载，优先级高，可覆盖上面的参数）
            LaunchConfiguration('config_file'),
        ],
    )
    
    # 2. 感知决策节点（语义分割 + 决策）
    perception_decision_node = Node(
        package='track_perception',
        executable='perception_decision_node',
        name='perception_decision_node',
        output='screen',
        parameters=[
            # ⭐ 其他参数（先加载，优先级低）
            {
                'shm_name': LaunchConfiguration('shm_name'),
                'seg_model_dir': LaunchConfiguration('seg_model_dir'),
                'seg_model_filename': LaunchConfiguration('seg_model_filename'),  # ⭐ 添加模型文件名参数
                'seg_tpes': LaunchConfiguration('seg_tpes'),
                'blend_alpha': LaunchConfiguration('blend_alpha'),
                'enable_flip': LaunchConfiguration('seg_enable_flip'),
                'flip_code': LaunchConfiguration('seg_flip_code'),
                'input_format': LaunchConfiguration('seg_input_format'),
                'model_input_format': LaunchConfiguration('seg_model_input_format'),
                'show_window': LaunchConfiguration('show_window'),
                'show_debug_window': LaunchConfiguration('show_debug_window'),
                'enable_perf_stats': LaunchConfiguration('enable_perf_stats'),
            },
            # ⭐ YAML配置文件（后加载，优先级高，可覆盖上面的参数）
            LaunchConfiguration('config_file'),
        ],
    )
    
    return LaunchDescription([
        # 共享内存参数
        shm_name_arg,
        
        # 目标检测节点参数
        det_model_path_arg,
        det_label_list_arg,
        det_tpes_arg,
        det_enable_flip_arg,
        det_flip_code_arg,
        det_input_format_arg,
        det_publish_rate_arg,
        
        # 语义分割决策节点参数
        seg_model_dir_arg,
        seg_model_filename_arg,  # ⭐ 添加模型文件名参数
        seg_tpes_arg,
        blend_alpha_arg,
        seg_enable_flip_arg,
        seg_flip_code_arg,
        seg_input_format_arg,
        seg_model_input_format_arg,
        show_window_arg,
        show_debug_window_arg,
        enable_perf_stats_arg,
        
        # 岔路口检测参数
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
        
        # ⭐ GuideBoard 岔路选择参数
        enable_guideboard_branch_selection_arg,
        guideboard_branch_arg,
        
        # ⭐ 高级岔路检测与中心线拟合参数（新逻辑）
        enable_segment_branch_logic_arg,
        band_count_arg,
        band_y_min_ratio_arg,
        band_y_max_ratio_arg,
        band_height_ratio_arg,
        min_segment_width_px_arg,
        min_segment_gap_px_arg,
        min_pixels_per_band_arg,
        branch_detect_min_bands_arg,
        branch_detect_far_band_ratio_arg,
        branch_lock_time_arg,
        exit_single_path_confirm_frames_arg,
        fit_min_points_arg,
        fit_order_arg,
        use_heading_term_arg,
        heading_weight_arg,
        near_offset_weight_arg,
        max_offset_jump_arg,
        offset_smoothing_alpha_arg,
        publish_debug_info_arg,
        show_branch_debug_arg,
        
        # YAML配置文件参数
        use_config_arg,
        config_file_arg,
        
        # 节点
        object_detection_node,
        perception_decision_node,
    ])
