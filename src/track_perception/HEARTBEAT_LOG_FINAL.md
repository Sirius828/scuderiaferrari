# 心跳日志格式最终版

## 📋 修改概述

将 `perception_decision_node` 的心跳日志格式调整为三段式输出：
**行驶方向 | 道路状况 | 偏移量**

## ✅ 最终日志格式

### 格式
```
{driving_direction} | {road_status} | Offset: {value}
```

### 字段说明

#### 1. driving_direction（行驶方向）
- **直行**: 默认方向，对应 left/outer_side
- **右转**: GuideBoard 触发时的方向，对应 right/guideboard_branch

#### 2. road_status（道路状况）
- **正常**: 未检测到岔路口（NORMAL 状态）
- **道路状况（岔路）**: 正在处理岔路口（APPROACH_INTERSECTION 或 LOCK_OUTER_BRANCH 状态）

#### 3. Offset（偏移量）
- 赛道中心偏移量，范围 -1.0 ~ 1.0

## 🎯 日志输出示例

### 场景 1：正常直行
```
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.089
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.102
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.095
```

### 场景 2：检测到岔路（无 GuideBoard）
```
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.078
[INFO] [perception_decision_node]: 直行 | 道路状况（岔路） | Offset: -0.045
[INFO] [perception_decision_node]: 直行 | 道路状况（岔路） | Offset: -0.032
[INFO] [perception_decision_node]: 直行 | 道路状况（岔路） | Offset: 0.156
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.112
```

### 场景 3：检测到岔路 + GuideBoard（右转）
```
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.067
[INFO] [perception_decision_node]: 🚩 GuideBoard detected in far ROI, choosing right branch
[INFO] [perception_decision_node]: 右转 | 道路状况（岔路） | Offset: 0.234
[INFO] [perception_decision_node]: 右转 | 道路状况（岔路） | Offset: 0.198
[INFO] [perception_decision_node]: 右转 | 道路状况（岔路） | Offset: 0.176
[INFO] [perception_decision_node]: 直行 | 正常 | Offset: 0.089
```

## 🔧 实现逻辑

### 代码位置
**文件**: `perception_decision_node.py`  
**行数**: 第 322-334 行

### 核心逻辑
```python
# ⭐ 7. 心跳日志（每秒输出一次）
current_time = time.time()
if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
    # 判断是否处于岔路口状态
    if self.intersection_state in ['APPROACH_INTERSECTION', 'LOCK_OUTER_BRANCH']:
        road_status = "道路状况（岔路）"
    else:
        road_status = "正常"
    
    self.get_logger().info(
        f'{self.driving_direction} | {road_status} | Offset: {center_offset:.3f}'
    )
    self.last_heartbeat_time = current_time
```

### driving_direction 设置时机

#### 1. 初始化（第 199 行）
```python
self.driving_direction = "直行"  # 默认直行
```

#### 2. 检测到岔路时设置（第 374-379 行）
```python
if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
    self.decision = self.guideboard_branch
    self.driving_direction = "右转"  # GuideBoard 触发右转
    self.get_logger().info(f'🚩 GuideBoard detected in far ROI, choosing {self.guideboard_branch} branch')
else:
    self.decision = 'outer'
    self.driving_direction = "直行"  # 默认直行
```

#### 3. 退出岔路时重置（第 403、412 行）
```python
self.driving_direction = "直行"  # 重置为直行
```

## 📊 状态流转图

```
NORMAL 状态
driving_direction = "直行"
road_status = "正常"
    ↓
检测到 far_intersection
    ↓
┌───────────┴───────────┐
│                       │
有 GuideBoard          无 GuideBoard
│                       │
↓                       ↓
driving_direction      driving_direction
= "右转"               = "直行"
    ↓                       ↓
APPROACH_INTERSECTION 状态
road_status = "道路状况（岔路）"
    ↓
检测到 near_intersection
    ↓
LOCK_OUTER_BRANCH 状态
road_status = "道路状况（岔路）"
    ↓
退出条件满足
    ↓
回到 NORMAL 状态
driving_direction = "直行" (重置)
road_status = "正常"
```

## 💡 优势特点

### 1. 信息完整
- **行驶方向**: 明确当前决策方向（直行/右转）
- **道路状况**: 清晰标识是否在岔路口
- **偏移量**: 实时控制参数

### 2. 易于解析
三段式格式，便于后续日志分析：
```bash
# 提取行驶方向
ros2 log echo /perception_decision_node | grep -oP '^\S+(?= \|)'

# 提取道路状况
ros2 log echo /perception_decision_node | grep -oP '(?<=\| )\S+(?= \|)'

# 提取 offset 值
ros2 log echo /perception_decision_node | grep -oP 'Offset: \S+'
```

### 3. 便于监控
```bash
# 只看岔路情况
ros2 log echo /perception_decision_node | grep "岔路"

# 只看右转决策
ros2 log echo /perception_decision_node | grep "^右转"

# 实时监控 offset
ros2 log echo /perception_decision_node | grep "Offset"
```

### 4. 语义清晰
- "直行" 和 "右转" 直观易懂
- "正常" 和 "道路状况（岔路）" 明确区分状态
- 符合人类阅读习惯

## 🔄 与之前版本的对比

### 版本 1（最初）
```
💓 Heartbeat | Road: 正常行驶 | Offset: 0.123
💓 Heartbeat | Road: 遇到岔路(有GuideBoard) | Offset: 0.234
```
❌ 前缀冗长，信息重复

### 版本 2（第一次优化）
```
直行 | Offset: 0.123
道路情况（岔路） | Offset: 0.234
```
❌ 缺少行驶方向信息

### 版本 3（最终版）✅
```
直行 | 正常 | Offset: 0.123
右转 | 道路状况（岔路） | Offset: 0.234
```
✅ 三段式，信息完整，简洁清晰

## ⚠️ 注意事项

1. **driving_direction 的持久性**
   - 一旦设置为"右转"，会持续到退出岔路状态
   - 即使在 APPROACH_INTERSECTION 期间 GuideBoard 消失，也不会改回"直行"

2. **road_status 的动态性**
   - 根据 intersection_state 实时判断
   - 每次心跳都会重新评估

3. **重置时机**
   - 只有在退出锁定状态回到 NORMAL 时
   - driving_direction 才会重置为"直行"

## 🚀 测试验证

1. **重新启动系统**
   ```bash
   colcon build --packages-select track_perception
   source install/setup.bash
   ros2 launch track_perception perception.launch.py
   ```

2. **观察正常行驶日志**
   ```
   直行 | 正常 | Offset: x.xxx
   ```

3. **接近岔路口（无 GuideBoard）**
   ```
   直行 | 道路状况（岔路） | Offset: x.xxx
   ```

4. **接近岔路口（有 GuideBoard）**
   ```
   🚩 GuideBoard detected in far ROI, choosing right branch
   右转 | 道路状况（岔路） | Offset: x.xxx
   ```

5. **通过岔路口后**
   ```
   直行 | 正常 | Offset: x.xxx
   ```

## 📝 相关文件

- **节点代码**: `src/track_perception/track_perception_python/perception_decision_node.py`
- **配置文件**: `src/track_perception/track_perception_python/config/intersection_params.yaml`
- **启动文件**: `src/track_perception/track_perception_python/launch/perception.launch.py`

---

**修改日期**: 2026-05-09  
**版本**: 3.0 (最终版)
