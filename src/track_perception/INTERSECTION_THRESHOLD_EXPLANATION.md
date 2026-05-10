# 岔路口检测阈值计算方法

## 📋 日志输出格式

修改后的日志格式：
```
直行 | NORMAL | Far:F Near:F | Offset: 0.089
直行 | APPROACH | Far:T Near:F | Offset: -0.045
右转 | LOCK | Far:T Near:T | Offset: 0.234
```

### 字段说明

| 字段 | 含义 | 取值 |
|------|------|------|
| `Far` | far_roi 是否超出宽度阈值 | `T` (True) / `F` (False) |
| `Near` | near_roi 是否超出宽度阈值 | `T` (True) / `F` (False) |

---

## 🔍 阈值计算方法详解

### 1. ROI 区域定义

#### Far ROI（远处检测区域）
```yaml
far_roi_y0_ratio: 0.40   # 从图像高度 40% 开始
far_roi_y1_ratio: 0.60   # 到图像高度 60% 结束
far_width_threshold: 0.55 # 宽度阈值 55%
```

**作用区域**: 图像中部偏上，用于提前检测远处的岔路口

#### Near ROI（近处检测区域）
```yaml
near_roi_y0_ratio: 0.70  # 从图像高度 70% 开始
near_roi_y1_ratio: 0.95  # 到图像高度 95% 结束
near_width_threshold: 0.80 # 宽度阈值 80%
```

**作用区域**: 图像底部，用于确认车辆已进入岔路口控制区域

---

### 2. 宽度计算流程

以 `detect_intersection_near` 为例：

```python
def detect_intersection_near(self, road_mask):
    h, w = road_mask.shape
    
    # 步骤 1: 提取 ROI 区域
    y0 = int(h * self.near_roi_y0_ratio)  # 例如: 480 * 0.70 = 336
    y1 = int(h * self.near_roi_y1_ratio)  # 例如: 480 * 0.95 = 456
    roi = road_mask[y0:y1, :]              # 提取 y=336~456 的行
    
    # 步骤 2: 检查是否有足够的道路像素
    road_pixels = np.sum(roi == 1)
    if road_pixels < self.min_road_pixels:
        return False, 0.0  # 道路像素太少，不认为是有效道路
    
    # 步骤 3: 找到所有包含道路像素的列
    cols = np.where(np.any(roi == 1, axis=0))[0]
    # np.any(roi == 1, axis=0): 对每一列检查是否有道路像素
    # 返回一个布尔数组，长度为 w
    # np.where: 找出所有为 True 的列索引
    
    if len(cols) == 0:
        return False, 0.0  # 没有检测到道路
    
    # 步骤 4: 计算道路宽度比例
    road_width = (cols.max() - cols.min() + 1) / w
    # cols.max(): 最右侧有道路像素的列索引
    # cols.min(): 最左侧有道路像素的列索引
    # +1: 包含边界
    # /w: 归一化到 [0, 1] 范围
    
    # 步骤 5: 判断是否超出阈值
    near_intersection = road_width > self.near_width_threshold
    # 如果 road_width > 0.80，则判定为岔路
    
    return near_intersection, road_width
```

---

### 3. 可视化示例

假设图像尺寸: `640 x 480` (宽 x 高)

#### 场景 1: 正常单路

```
y=0   ┌─────────────┐
      │             │
y=192 ├─────────────┤ ← far_roi_y0 (40%)
      │   ██████    │ ← far_roi 区域 (y=192~288)
y=288 ├─────────────┤ ← far_roi_y1 (60%)
      │             │
y=336 ├─────────────┤ ← near_roi_y0 (70%)
      │   ██████    │ ← near_roi 区域 (y=336~456)
y=456 ├─────────────┤ ← near_roi_y1 (95%)
      │             │
y=480 └─────────────┘

far_roi 道路宽度: 100/640 = 0.156 (15.6%)
near_roi 道路宽度: 100/640 = 0.156 (15.6%)

结果:
- Far: F (0.156 < 0.55)
- Near: F (0.156 < 0.80)
状态: NORMAL
```

#### 场景 2: 接近岔路口 (Y型分叉)

```
y=0   ┌─────────────┐
      │             │
y=192 ├─────────────┤ ← far_roi_y0 (40%)
      │ ███████████ │ ← far_roi: 道路变宽 (两个分支都可见)
y=288 ├─────────────┤ ← far_roi_y1 (60%)
      │             │
y=336 ├─────────────┤ ← near_roi_y0 (70%)
      │   ██████    │ ← near_roi: 还未分开
y=456 ├─────────────┤ ← near_roi_y1 (95%)
      │             │
y=480 └─────────────┘

far_roi 道路宽度: 400/640 = 0.625 (62.5%)
near_roi 道路宽度: 100/640 = 0.156 (15.6%)

结果:
- Far: T (0.625 > 0.55) ✅ 触发
- Near: F (0.156 < 0.80)
状态: APPROACH_INTERSECTION
```

#### 场景 3: 进入岔路口

```
y=0   ┌─────────────┐
      │             │
y=192 ├─────────────┤ ← far_roi_y0 (40%)
      │ ███████████ │ ← far_roi: 道路很宽
y=288 ├─────────────┤ ← far_roi_y1 (60%)
      │             │
y=336 ├─────────────┤ ← near_roi_y0 (70%)
      │ ███████████ │ ← near_roi: 两个分支都已进入
y=456 ├─────────────┤ ← near_roi_y1 (95%)
      │             │
y=480 └─────────────┘

far_roi 道路宽度: 400/640 = 0.625 (62.5%)
near_roi 道路宽度: 550/640 = 0.859 (85.9%)

结果:
- Far: T (0.625 > 0.55) ✅
- Near: T (0.859 > 0.80) ✅ 触发
状态: LOCK_OUTER_BRANCH
```

---

### 4. 关键参数说明

#### `min_road_pixels`
```yaml
min_road_pixels: 100  # 最少道路像素数
```

**作用**: 过滤噪声和误检。如果 ROI 区域内的道路像素少于这个值，直接返回 `False`。

**计算公式**:
```python
road_pixels = np.sum(roi == 1)
if road_pixels < min_road_pixels:
    return False, 0.0
```

#### `far_width_threshold` / `near_width_threshold`
```yaml
far_width_threshold: 0.55   # 55%
near_width_threshold: 0.80  # 80%
```

**含义**: 道路宽度占图像宽度的比例阈值

**为什么 far 阈值更低?**
- far_roi 在图像上半部分，透视效果导致道路看起来更窄
- 需要更早检测到远处的岔路，所以阈值设置较低
- 55% 意味着只要道路宽度超过图像一半就认为是岔路

**为什么 near 阈值更高?**
- near_roi 在图像下半部分，更接近真实尺寸
- 需要确认车辆真正进入岔路控制区域
- 80% 意味着道路必须非常宽才触发锁定，避免弯道误判

---

### 5. 调优建议

#### 问题 1: 弯道误判为岔路

**现象**: 
```
直行 | APPROACH | Far:T Near:F | Offset: xxx
```
在弯道时频繁出现 `Far:T`

**解决方案**:
1. **提高 `far_width_threshold`**: `0.55 → 0.60` 或 `0.65`
2. **降低 `far_roi_y1_ratio`**: `0.60 → 0.55` (让 far_roi 更靠上，避开弯道外侧)
3. **增加 `min_road_pixels`**: `100 → 200` (过滤小面积误检)

#### 问题 2: 岔路检测太晚

**现象**: 
```
直行 | NORMAL | Far:F Near:F | Offset: xxx
直行 | LOCK | Far:T Near:T | Offset: xxx  # 突然跳到 LOCK
```
没有经过 `APPROACH` 状态

**解决方案**:
1. **降低 `far_width_threshold`**: `0.55 → 0.50`
2. **提高 `far_roi_y1_ratio`**: `0.60 → 0.65` (让 far_roi 更靠下，更容易检测到)
3. **降低 `near_width_threshold`**: `0.80 → 0.75`

#### 问题 3: LOCK 状态退出太早

**现象**: 
```
右转 | LOCK | Far:T Near:T | Offset: 0.234
右转 | NORMAL | Far:F Near:F | Offset: 0.089  # 过早退出
```

**解决方案**:
1. **降低 `exit_width_threshold`**: `0.45 → 0.40`
2. **增加 `exit_confirm_frames`**: `5 → 10` (需要更多帧确认)
3. **增加 `max_lock_time`**: `3.0 → 5.0` (延长最大锁定时间)

---

### 6. 调试技巧

#### 启用 DEBUG 日志查看详细数据

```bash
ros2 run track_perception perception_decision_node --ros-args --log-level debug
```

会看到类似输出：
```
[DEBUG] [perception_decision_node]: Far ROI: width=0.625, threshold=0.55, result=True
[DEBUG] [perception_decision_node]: Near ROI: width=0.859, threshold=0.80, result=True
```

#### 实时监控日志

```bash
ros2 topic echo /segmentation/center_offset
```

观察 offset 值在不同状态下的变化：
- `NORMAL`: offset 应该在小范围内波动 (-0.1 ~ +0.1)
- `APPROACH`: offset 可能开始偏移
- `LOCK`: offset 应该明显偏向选定方向 (如选择 left，offset 应为负值)

#### 调整参数并重新编译

```bash
# 1. 修改 YAML 配置
nano src/track_perception/track_perception_python/config/intersection_params.yaml

# 2. 重新编译
cd /home/orangepi/scuderiaferrari
colcon build --packages-select track_perception
source install/setup.bash

# 3. 重新启动
ros2 launch track_perception perception.launch.py
```

---

## 📊 总结

### 核心公式

```python
# 道路宽度比例计算
road_width = (最右列 - 最左列 + 1) / 图像宽度

# 岔路判断
is_intersection = road_width > width_threshold
```

### 状态转换条件

```
NORMAL → APPROACH:
  条件: far_intersection == True (road_width_far > far_width_threshold)

APPROACH → LOCK:
  条件: near_intersection == True (road_width_near > near_width_threshold)

LOCK → NORMAL:
  条件 1: road_width_near < exit_width_threshold (连续 N 帧)
  条件 2: lock_duration > max_lock_time (超时保护)
```

### 日志解读

```
直行 | NORMAL | Far:F Near:F | Offset: 0.089
       ↑        ↑     ↑           ↑
     方向     状态  Far/Near   偏移量
                检测结果

T = True (超出阈值，认为是岔路)
F = False (未超阈值，正常道路)
```

通过观察 `Far` 和 `Near` 的组合，可以判断当前处于哪个阶段：
- `F F`: 正常行驶
- `T F`: 接近岔路 (APPROACH)
- `T T`: 进入岔路 (LOCK)
- `F T`: 异常情况 (可能是弯道误判)
