# Track Perception System

智能巡线小车感知系统，整合语义分割和目标检测功能。

## 📦 包结构

```
track_perception/
├── track_perception/
│   ├── object_detection_node.py      # 目标检测节点（RKNN推理）
│   ├── object_detection_infer.py     # RKNN目标检测推理封装
│   ├── perception_decision_node.py   # 感知决策节点（语义分割+融合决策）
│   ├── ppseg_infer.py                # PP-LiteSeg推理封装
│   ├── model/                        # 模型文件
│   │   ├── rknn_lt.rknn             # 目标检测模型
│   │   ├── label_list.txt           # 标签列表
│   │   └── [segmentation models]    # 语义分割模型
│   ├── config/                       # 配置文件
│   │   └── intersection_params.yaml # 岔路口检测参数
│   └── launch/                       # 启动文件
│       └── perception.launch.py     # 主启动文件
├── package.xml
└── setup.py
```

## 🚀 功能特性

### 1. 目标检测节点 (`object_detection_node`)
- **输入**: 共享内存视频流 (`shm_ar_video`)
- **推理引擎**: RKNN (Rockchip NPU)
- **检测类别**: Car, Human, Gold, Go, Gate, GuideBoard, Speed_Limit, red_light, yellow_light, green_light, Stop
- **输出话题**:
  - `/detection/results` (Float32MultiArray): 检测结果数组
  - `/detection/labels` (String): 检测到的标签列表

### 2. 感知决策节点 (`perception_decision_node`)
- **输入**:
  - 共享内存视频流 (用于语义分割)
  - `/detection/results` (目标检测结果)
- **功能**:
  - 语义分割推理（PP-LiteSeg）
  - 岔路口检测与外圈锁定
  - 目标检测融合决策（障碍物避障）
- **输出话题**:
  - `/segmentation/center_offset` (Float32): 赛道中心偏移量 [-1.0, 1.0]
  - `/segmentation/is_valid` (Bool): 是否有有效赛道

## 🔧 安装

```bash
cd ~/scuderiaferrari
colcon build --packages-select track_perception
source install/setup.bash
```

## 🎯 使用方法

### 启动完整感知系统

```bash
ros2 launch track_perception perception.launch.py
```

### 自定义参数

```bash
# 调整目标检测发布频率
ros2 launch track_perception perception.launch.py det_publish_rate:=20

# 禁用目标检测融合
ros2 launch track_perception perception.launch.py enable_detection_fusion:=false

# 调整岔路口检测参数
ros2 launch track_perception perception.launch.py outer_side:=right

# 启用可视化窗口
ros2 launch track_perception perception.launch.py show_window:=true
```

### 使用 YAML 配置文件

默认会加载 `config/intersection_params.yaml`，可以通过命令行覆盖：

```bash
ros2 launch track_perception perception.launch.py \
    use_config:=true \
    config_file:=/path/to/your/config.yaml
```

## 📊 数据格式

### 目标检测结果 (`/detection/results`)

Float32MultiArray 格式：`[class_id, confidence, x1, y1, x2, y2, cx, cy] * N`

- `class_id`: 类别ID (0-10)
- `confidence`: 置信度 (0.0-1.0)
- `x1, y1, x2, y2`: 边界框坐标（像素）
- `cx, cy`: 中心点坐标（像素）

### 赛道偏移量 (`/segmentation/center_offset`)

Float32 格式：范围 [-1.0, 1.0]
- `0.0`: 居中
- `-1.0`: 最左侧
- `1.0`: 最右侧

### 赛道有效性 (`/segmentation/is_valid`)

Bool 格式：
- `True`: 检测到有效赛道
- `False`: 未检测到赛道或有障碍物

## ⚙️ 配置参数

### 目标检测节点参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `shm_name` | `shm_ar_video` | 共享内存名称 |
| `det_model_path` | `model/rknn_lt.rknn` | RKNN模型路径 |
| `det_label_list_path` | `model/label_list.txt` | 标签列表路径 |
| `det_tpes` | `3` | 线程池执行器数量 |
| `det_publish_rate` | `10` | 发布频率 (Hz) |
| `det_enable_flip` | `true` | 是否启用图像翻转 |
| `det_flip_code` | `0` | 翻转代码 (0=垂直, 1=水平, -1=两者) |
| `det_input_format` | `RGB` | 输入图像格式 |

### 感知决策节点参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_intersection_logic` | `true` | 启用岔路口检测 |
| `outer_side` | `left` | 外圈方向 (left/right) |
| `enable_detection_fusion` | `true` | 启用目标检测融合 |
| `detection_obstacle_classes` | `['Human', 'Car', 'Stop']` | 障碍物类别 |
| `detection_stop_distance` | `0.3` | 停止距离阈值 |

更多岔路口参数请参考 `config/intersection_params.yaml`。

## 🔍 故障排查

### 问题1: 找不到共享内存
```
Waiting for server...
```
**解决**: 确保视频流服务已启动并创建了 `shm_ar_video` 共享内存。

### 问题2: RKNN 模型加载失败
```
Failed to load RKNN model
```
**解决**: 
1. 检查模型文件是否存在于 `model/` 目录
2. 确认已安装 `rknnlite` 库
3. 检查模型文件权限

### 问题3: 目标检测无输出
**解决**:
1. 检查 `/detection/results` 话题是否有数据: `ros2 topic echo /detection/results`
2. 调整 `det_publish_rate` 参数
3. 查看节点日志: `ros2 logs --all | grep object_detection`

## 📝 注意事项

1. **资源占用**: 同时运行两个推理模型（目标检测 + 语义分割）会占用较多 NPU 资源，建议 TPEs 总数不超过 6。

2. **同步问题**: 目标检测和语义分割是独立运行的，可能存在帧不同步。如需严格同步，可考虑使用 ROS2 TimeSynchronizer。

3. **障碍物避障**: 当检测到障碍物时，系统会将 `is_valid` 设为 `False`，触发下游控制器停车。

4. **YAML 配置优先级**: YAML 配置文件中的参数优先级高于 launch 文件中的默认值，但低于命令行参数。

## 🛠️ 开发指南

### 添加新的检测类别

1. 编辑 `model/label_list.txt`，添加新类别
2. 更新 `detection_obstacle_classes` 参数
3. 重新编译包

### 修改决策逻辑

编辑 `perception_decision_node.py` 中的 `make_decision()` 方法，添加自定义融合策略。

## 📄 许可证

MIT License
