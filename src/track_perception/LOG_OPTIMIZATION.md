# 日志优化说明

## 📋 修改概述

本次修改优化了两个节点的日志输出：
1. **object_detection_node**：禁用性能统计日志，减少日志噪音
2. **perception_decision_node**：添加心跳日志，实时显示道路情况和偏移量

## ✅ 修改内容

### 1. object_detection_node 日志优化

**文件**: `src/track_perception/track_perception_python/object_detection_node.py`

#### 修改前
每 5 秒输出一次性能统计：
```
[INFO] [object_detection_node]: ⏱️ Performance Stats:
   Preprocess:      5.23 ms
   Inference:      12.45 ms
   Publish:         1.02 ms
   Total:          18.70 ms (53.5 FPS)
```

#### 修改后
- ❌ 禁用性能统计日志（已注释）
- ✅ 保留启动日志和错误日志
- 📝 如需重新启用，取消代码中的注释即可

#### 保留的日志
```python
# 启动日志（保留）
self.get_logger().info('✅ Object Detection model initialized successfully')
self.get_logger().info(f'📡 Object Detection Node Ready')
self.get_logger().info(f'   SHM Name: {self.shm_name}')
...

# 错误日志（保留）
self.get_logger().error(f'Error processing frame: {e}')
self.get_logger().warn('❌ Connection lost...')
```

### 2. perception_decision_node 心跳日志

**文件**: `src/track_perception/track_perception_python/perception_decision_node.py`

#### 新增功能
每秒输出一次心跳日志，包含：
- **Road**: 当前道路情况（简短描述）
- **Offset**: 赛道中心偏移量（-1.0 ~ 1.0）

#### 日志格式
```
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 正常行驶 | Offset: 0.123
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路 | Offset: -0.045
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路(有GuideBoard) | Offset: 0.234
```

#### 道路情况说明

| 道路情况 | 触发条件 | 说明 |
|---------|---------|------|
| `正常行驶` | 未检测到岔路口 | 普通赛道行驶状态 |
| `遇到岔路` | 检测到远端岔路口，但无 GuideBoard | 即将进入岔路口 |
| `遇到岔路(有GuideBoard)` | 检测到远端岔路口 + far_roi 区域有 GuideBoard | 需要根据 GuideBoard 选择方向 |

#### 实现细节

**1. 参数初始化**（`__init__` 方法）
```python
# ⭐ 心跳日志参数
self.heartbeat_interval = 1.0  # 心跳间隔（秒）
self.last_heartbeat_time = time.time()

# ⭐ 道路情况描述
self.road_condition = "正常行驶"  # 默认道路情况
```

**2. 道路情况更新**（`make_decision` 方法）
```python
# ⭐ 更新道路情况描述
if guideboard_detected_in_far and far_intersection:
    self.road_condition = f"遇到岔路(有GuideBoard)"
elif far_intersection:
    self.road_condition = "遇到岔路"
else:
    self.road_condition = "正常行驶"
```

**3. 心跳输出**（`process_frame` 方法）
```python
# ⭐ 7. 心跳日志（每秒输出一次）
current_time = time.time()
if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
    self.get_logger().info(
        f'💓 Heartbeat | Road: {self.road_condition} | Offset: {center_offset:.3f}'
    )
    self.last_heartbeat_time = current_time
```

## 🎯 使用效果

### 修改前的终端输出
```
[INFO] [object_detection_node]: ⏱️ Performance Stats:
   Preprocess:      5.23 ms
   Inference:      12.45 ms
   ...
[INFO] [object_detection_node]: ⏱️ Performance Stats:
   Preprocess:      4.98 ms
   Inference:      11.87 ms
   ...
[INFO] [perception_decision_node]: 🚩 GuideBoard detected in far ROI, choosing right branch
```

### 修改后的终端输出
```
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 正常行驶 | Offset: 0.123
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 正常行驶 | Offset: 0.089
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路 | Offset: -0.045
[INFO] [perception_decision_node]: 🚩 GuideBoard detected in far ROI, choosing right branch
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路(有GuideBoard) | Offset: 0.234
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路(有GuideBoard) | Offset: 0.198
```

## 🔧 自定义配置

### 调整心跳间隔

编辑 `perception_decision_node.py`：
```python
self.heartbeat_interval = 0.5  # 改为 0.5 秒（更频繁）
# 或
self.heartbeat_interval = 2.0  # 改为 2.0 秒（更稀疏）
```

### 重新启用性能统计

编辑 `object_detection_node.py`，取消以下代码的注释：
```python
# 将这段代码从注释状态恢复
if need_perf_stats:
    t_end = time.perf_counter()
    t_preprocess = t_inference_start - t_preprocess_start if t_inference_start and t_preprocess_start else 0
    ...
    self.get_logger().info(
        f'⏱️ Performance Stats:\n'
        f'   Preprocess:    {t_preprocess*1000:6.2f} ms\n'
        ...
    )
    self.last_perf_time = current_time
```

### 扩展道路情况描述

在 `make_decision` 方法中添加新的条件判断：
```python
# 示例：添加障碍物检测
if obstacle_detected:
    self.road_condition = "前方有障碍物"
elif guideboard_detected_in_far and far_intersection:
    self.road_condition = f"遇到岔路(有GuideBoard)"
elif far_intersection:
    self.road_condition = "遇到岔路"
else:
    self.road_condition = "正常行驶"
```

## 📊 日志级别说明

### object_detection_node
- **INFO**: 启动信息、连接状态
- **WARN**: 连接丢失警告
- **ERROR**: 处理错误
- ~~INFO: 性能统计~~（已禁用）

### perception_decision_node
- **INFO**: 启动信息、心跳日志、GuideBoard 检测
- **DEBUG**: GuideBoard 详细信息（坐标、置信度）
- **WARN**: 连接丢失警告
- **ERROR**: 处理错误

## 💡 优势特点

### 1. 减少日志噪音
- object_detection_node 不再每 5 秒输出性能统计
- 终端更清爽，便于观察关键信息

### 2. 实时监控
- 每秒输出一次心跳，实时了解系统状态
- 道路情况一目了然
- offset 值便于调试控制算法

### 3. 易于扩展
- 道路情况描述采用字符串，方便添加新状态
- 心跳间隔可配置
- 可随时启用/禁用性能统计

### 4. 调试友好
- 清晰的日志格式
- 关键事件（如 GuideBoard 检测）仍然输出
- 错误日志完整保留

## 🔍 监控建议

### 正常运行时
观察心跳日志的频率和内容：
```bash
ros2 log echo /perception_decision_node | grep "Heartbeat"
```

### 调试岔路口逻辑
关注道路情况变化：
```bash
ros2 log echo /perception_decision_node | grep -E "(Heartbeat|GuideBoard)"
```

### 检查 offset 稳定性
```bash
ros2 topic echo /segmentation/center_offset
```

## ⚠️ 注意事项

1. **性能影响**：心跳日志每秒输出一次，对性能影响极小
2. **日志缓冲**：ROS2 日志可能有短暂延迟，属于正常现象
3. **远程运行**：通过 SSH 运行时，确保日志级别设置正确
4. **日志文件**：大量运行时注意清理日志文件

## 📝 相关文件

- **object_detection_node.py**: `src/track_perception/track_perception_python/object_detection_node.py`
- **perception_decision_node.py**: `src/track_perception/track_perception_python/perception_decision_node.py`
- **intersection_params.yaml**: `src/track_perception/track_perception_python/config/intersection_params.yaml`

## 🚀 快速测试

1. **重新启动系统**：
   ```bash
   ros2 launch track_perception perception.launch.py
   ```

2. **观察心跳日志**：
   应该每秒看到一行：
   ```
   [INFO] [perception_decision_node]: 💓 Heartbeat | Road: 正常行驶 | Offset: 0.xxx
   ```

3. **验证性能统计已禁用**：
   object_detection_node 不应再输出 `⏱️ Performance Stats`

---

**修改日期**: 2026-05-09  
**版本**: 1.0
