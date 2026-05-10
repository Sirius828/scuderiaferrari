# Perception 节点简化说明

## 📋 修改概述

将 `perception_decision_node` 简化为与 `segmentation_node` 功能一致，仅订阅目标检测结果但不用于决策控制。

## ✅ 已移除的功能

### 1. **避障逻辑完全移除**
- ❌ 删除障碍物检测参数声明
- ❌ 删除障碍物类别配置
- ❌ 删除停车距离阈值
- ❌ 删除融合决策代码（步骤2）
- ❌ 删除相关日志输出

### 2. **Launch 文件清理**
- ❌ 移除 `enable_detection_fusion` 参数
- ❌ 移除 `detection_obstacle_classes` 参数
- ❌ 移除 `detection_stop_distance` 参数

## 🎯 当前功能

### perception_decision_node 现在只做：
1. ✅ 从共享内存读取视频流
2. ✅ 执行语义分割推理
3. ✅ 岔路口检测与外圈锁定
4. ✅ 计算赛道中心偏移量
5. ✅ 发布 `/segmentation/center_offset` (Float32)
6. ✅ 发布 `/segmentation/is_valid` (Bool)
7. ✅ **订阅** `/detection/results` (仅记录，不影响决策)

### 与 segmentation_node 的对比

| 功能 | segmentation_node | perception_decision_node |
|------|------------------|-------------------------|
| 语义分割 | ✅ | ✅ |
| 岔路口检测 | ✅ | ✅ |
| 偏移量计算 | ✅ | ✅ |
| 发布 offset | ✅ | ✅ |
| 发布 is_valid | ✅ | ✅ |
| 订阅检测结果 | ❌ | ✅ (仅记录) |
| 避障决策 | ❌ | ❌ (已移除) |

## 📊 终端输出对比

### segmentation.launch.py
```
[INFO] [semantic_segmentation_node]: ✅ Semantic Segmentation model initialized
[INFO] [semantic_segmentation_node]: 📡 Semantic Segmentation Node Ready
[INFO] [semantic_segmentation_node]:    SHM Name: shm_ar_video
[INFO] [semantic_segmentation_node]: ✅ Connected to shared memory!
```

### perception.launch.py (修改后)
```
[INFO] [perception_decision_node]: ✅ Semantic Segmentation model initialized
[INFO] [perception_decision_node]: 📡 Perception Decision Node Ready
[INFO] [perception_decision_node]:    SHM Name: shm_ar_video
[INFO] [perception_decision_node]: ✅ Connected to shared memory!
```

**✅ 输出格式完全一致！**

## 🔧 技术细节

### 移除的代码位置

**perception_decision_node.py**:
- 第 60-63 行：删除参数声明
- 第 92-95 行：删除参数获取
- 第 169-173 行：删除日志输出
- 第 364-384 行：删除避障决策逻辑

**perception.launch.py**:
- 第 197-214 行：删除 launch 参数声明
- 第 277-281 行：删除节点参数传递
- 第 325-328 行：删除 LaunchDescription 中的参数

### 保留的功能

```python
# 仍然订阅检测结果（用于未来扩展或调试）
self.detections_subscription = self.create_subscription(
    Float32MultiArray,
    '/detection/results',
    self.detection_callback,
    detection_qos
)

# 缓存最新检测结果
self.latest_detections = []
self.detections_timestamp = 0
```

## 🚀 使用方式

### 启动感知系统
```bash
ros2 launch track_perception perception.launch.py
```

### 自定义岔路口参数
```bash
ros2 launch track_perception perception.launch.py \
    outer_side:=right \
    far_width_threshold:=0.70
```

### 禁用岔路口检测
```bash
ros2 launch track_perception perception.launch.py \
    enable_intersection_logic:=false
```

## 📝 注意事项

1. **目标检测节点仍在运行**：`object_detection_node` 继续发布检测结果，但不会被用于决策
2. **可以后续添加智能避障**：如果需要，可以重新启用融合决策逻辑
3. **与 line_follower_control 兼容**：发布的 offset 和 is_valid 话题格式与 segmentation_node 完全一致
4. **YAML 配置文件**：仍然支持通过 YAML 配置岔路口检测参数

## ✨ 优势

- ✅ **简洁清晰**：移除了复杂的避障逻辑
- ✅ **稳定可靠**：与经过测试的 segmentation_node 行为一致
- ✅ **易于维护**：代码结构更简单
- ✅ **可扩展**：保留了检测结果订阅，方便后续添加高级功能

## 🔄 迁移指南

如果之前使用 `segmentation.launch.py`，现在可以无缝切换到 `perception.launch.py`：

```bash
# 旧方式
ros2 launch semantic_segmentation segmentation.launch.py

# 新方式（功能相同，额外提供目标检测数据）
ros2 launch track_perception perception.launch.py
```

两者发布的控制话题完全一致，无需修改下游节点（如 `line_follower_control`）。
