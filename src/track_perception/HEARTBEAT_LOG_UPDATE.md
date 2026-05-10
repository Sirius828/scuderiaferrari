# 心跳日志格式更新说明

## 📋 修改概述

优化了 `perception_decision_node` 的心跳日志输出格式，使其更简洁清晰。

## ✅ 修改内容

### 1. 变量重命名
- **修改前**: `self.road_condition`（道路情况描述）
- **修改后**: `self.driving_direction`（行驶方向：直行/右转）

### 2. 日志格式简化

#### 修改前
```
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 正常行驶 | Offset: 0.123
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路 | Offset: -0.045
[INFO] [perception_decision_node]: 💓 Heartbeat | Road: 遇到岔路(有GuideBoard) | Offset: 0.234
```

#### 修改后
```
[INFO] [perception_decision_node]: 直行 | Offset: 0.123
[INFO] [perception_decision_node]: 道路情况（岔路） | Offset: -0.045
[INFO] [perception_decision_node]: 道路情况（岔路） | Offset: 0.234
```

### 3. 道路状态分类

现在只有两种状态：

| 状态 | 显示文本 | 触发条件 |
|------|---------|---------|
| **直行** | `直行` | 正常行驶，未检测到岔路口 |
| **右转** | `直行` | 检测到岔路口但无 GuideBoard（默认走 outer_side，即左侧=直行） |
| **岔路** | `道路情况（岔路）` | 处于 APPROACH_INTERSECTION 或 LOCK_OUTER_BRANCH 状态 |

**注意**：
- "直行" 对应语义上的 left 方向（默认外圈）
- "右转" 对应语义上的 right 方向（GuideBoard 触发）
- 但在日志中统一显示为"直行"或"道路情况（岔路）"，简化输出

### 4. 实现逻辑

#### 状态判断
```python
# 判断是否处于岔路口状态
if self.intersection_state in ['APPROACH_INTERSECTION', 'LOCK_OUTER_BRANCH']:
    road_status = "道路情况（岔路）"
else:
    road_status = self.driving_direction  # "直行"
```

#### 方向设置
```python
# 在 NORMAL → APPROACH_INTERSECTION 转换时设置
if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
    self.decision = self.guideboard_branch  # "right"
    self.driving_direction = "右转"  # GuideBoard 触发右转
else:
    self.decision = 'outer'  # "left"
    self.driving_direction = "直行"  # 默认直行
```

#### 状态重置
```python
# 退出锁定状态回到 NORMAL 时重置
self.intersection_state = 'NORMAL'
self.decision = 'none'
self.driving_direction = "直行"  # 重置为直行
```

## 🎯 使用效果

### 正常运行时
```
[INFO] [perception_decision_node]: 直行 | Offset: 0.089
[INFO] [perception_decision_node]: 直行 | Offset: 0.102
[INFO] [perception_decision_node]: 直行 | Offset: 0.095
```

### 接近岔路口时
```
[INFO] [perception_decision_node]: 直行 | Offset: 0.078
[INFO] [perception_decision_node]: 道路情况（岔路） | Offset: -0.045
[INFO] [perception_decision_node]: 道路情况（岔路） | Offset: -0.032
[INFO] [perception_decision_node]: 道路情况（岔路） | Offset: 0.156
```

### 通过岔路口后
```
[INFO] [perception_decision_node]: 道路情况（岔路） | Offset: 0.234
[INFO] [perception_decision_node]: 直行 | Offset: 0.112
[INFO] [perception_decision_node]: 直行 | Offset: 0.098
```

## 🔧 代码修改位置

### 文件: `perception_decision_node.py`

#### 1. 参数初始化（第 198-199 行）
```python
# ⭐ 道路情况描述（直行/右转）
self.driving_direction = "直行"  # 默认直行
```

#### 2. 方向设置（第 374-379 行）
```python
if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
    self.decision = self.guideboard_branch
    self.driving_direction = "右转"  # GuideBoard 触发右转
    self.get_logger().info(f'🚩 GuideBoard detected in far ROI, choosing {self.guideboard_branch} branch')
else:
    self.decision = 'outer'
    self.driving_direction = "直行"  # 默认直行
```

#### 3. 心跳日志输出（第 322-333 行）
```python
# ⭐ 7. 心跳日志（每秒输出一次）
current_time = time.time()
if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
    # 判断是否处于岔路口状态
    if self.intersection_state in ['APPROACH_INTERSECTION', 'LOCK_OUTER_BRANCH']:
        road_status = "道路情况（岔路）"
    else:
        road_status = self.driving_direction
    
    self.get_logger().info(
        f'{road_status} | Offset: {center_offset:.3f}'
    )
    self.last_heartbeat_time = current_time
```

#### 4. 状态重置（第 401、410 行）
```python
self.driving_direction = "直行"  # 重置为直行
```

## 💡 优势特点

### 1. 更简洁的输出
- 删除了 `💓 Heartbeat |` 前缀
- 删除了 `Road:` 标签
- 日志更短，更易读

### 2. 清晰的状态区分
- **直行**: 正常行驶状态
- **道路情况（岔路）**: 明确标识正在处理岔路口

### 3. 易于监控
```bash
# 只看岔路情况
ros2 log echo /perception_decision_node | grep "岔路"

# 只看 offset 值
ros2 log echo /perception_decision_node | grep "Offset"
```

### 4. 便于后续扩展
如果需要添加更多方向（如左转、掉头），只需：
1. 添加新的 driving_direction 值
2. 在决策逻辑中设置相应的方向

## 📊 状态流转图

```
NORMAL 状态
driving_direction = "直行"
    ↓
检测到 far_intersection
    ↓
┌───────────┴───────────┐
│                       │
有 GuideBoard          无 GuideBoard
│                       │
↓                       ↓
decision = "right"     decision = "left"
driving_direction      driving_direction
保持"直行"             保持"直行"
    ↓                       ↓
APPROACH_INTERSECTION 状态
日志显示: "道路情况（岔路）"
    ↓
检测到 near_intersection
    ↓
LOCK_OUTER_BRANCH 状态
日志显示: "道路情况（岔路）"
    ↓
退出条件满足
    ↓
回到 NORMAL 状态
driving_direction = "直行" (重置)
```

## ⚠️ 注意事项

1. **方向语义**：
   - 代码中的 `decision = "right"` 表示右转
   - 但日志中仍然显示"直行"或"道路情况（岔路）"
   - 如果需要显示具体方向，可以修改日志格式

2. **状态判断**：
   - 只要处于 `APPROACH_INTERSECTION` 或 `LOCK_OUTER_BRANCH` 状态
   - 日志就显示"道路情况（岔路）"
   - 不区分是否有 GuideBoard

3. **重置时机**：
   - 退出锁定状态回到 NORMAL 时
   - driving_direction 重置为"直行"

## 🚀 测试验证

1. **重新启动系统**：
   ```bash
   ros2 launch track_perception perception.launch.py
   ```

2. **观察日志输出**：
   - 正常行驶时应显示：`直行 | Offset: x.xxx`
   - 接近岔路时应显示：`道路情况（岔路） | Offset: x.xxx`

3. **验证 GuideBoard 检测**：
   - 看到 `🚩 GuideBoard detected...` 日志时
   - 下一帧心跳应显示"道路情况（岔路）"

---

**修改日期**: 2026-05-09  
**版本**: 1.0
