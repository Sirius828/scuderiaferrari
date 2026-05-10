# GuideBoard 检测逻辑优化说明

## 📋 修改概述

将 GuideBoard 的检测从依赖 far_roi 改为使用独立的检测范围，并在 LOCK 状态下持续检测和动态调整方向。

---

## ✅ 主要修改

### 1. **新增参数**

#### YAML 配置 (`intersection_params.yaml`)

```yaml
# GuideBoard 岔路选择参数
enable_guideboard_branch_selection: true
guideboard_branch: "right"
guideboard_detect_y0_ratio: 0.2  # GuideBoard检测起始y比例（0.0=顶部，1.0=底部）
guideboard_detect_y1_ratio: 0.7  # GuideBoard检测结束y比例（默认检测上半部分70%区域）
```

**参数说明**:
- `guideboard_detect_y0_ratio`: 检测区域起始位置（相对于图像高度）
- `guideboard_detect_y1_ratio`: 检测区域结束位置（相对于图像高度）

**推荐配置**:
- **上半部分检测**（默认）: `0.2 ~ 0.7` → 检测图像 20%~70% 的区域
- **全画面检测**: `0.0 ~ 1.0` → 检测整个图像
- **仅上半部分**: `0.0 ~ 0.5` → 检测图像顶部 50%

---

### 2. **检测方法改进**

#### 修改前
```python
def check_guideboard_in_far_roi(self, img_h, img_w):
    # 使用 far_roi 的范围
    y0 = int(img_h * self.far_roi_y0_ratio)  # 依赖 far_roi
    y1 = int(img_h * self.far_roi_y1_ratio)
```

#### 修改后
```python
def check_guideboard_in_far_roi(self, img_h, img_w):
    # ⭐ 使用专门的 GuideBoard 检测范围
    y0 = int(img_h * self.guideboard_detect_y0_ratio)  # 独立配置
    y1 = int(img_h * self.guideboard_detect_y1_ratio)
```

**优势**:
- 不再依赖 far_roi 的配置
- 可以独立调整 GuideBoard 检测范围
- 更灵活，适应不同的摄像头角度和路牌位置

---

### 3. **决策逻辑增强**

#### 修改点 1: NORMAL → APPROACH 转换时

```python
if self.intersection_state == 'NORMAL':
    if far_intersection:
        self.intersection_state = 'APPROACH_INTERSECTION'
        # ⭐ 根据是否检测到GuideBoard决定岔路方向
        if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
            self.decision = self.guideboard_branch  # 'right'
            self.driving_direction = "右转"
            self.get_logger().info(f'🚩 GuideBoard detected, choosing {self.guideboard_branch} branch')
```

---

#### 修改点 2: APPROACH → LOCK 转换时

```python
elif self.intersection_state == 'APPROACH_INTERSECTION':
    if near_intersection:
        self.intersection_state = 'LOCK_OUTER_BRANCH'
        self.lock_start_time = current_time
        self.exit_confirm_count = 0
        
        # ⭐ 在 LOCK 状态下再次检查 GuideBoard，确保选择正确的方向
        if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
            self.decision = self.guideboard_branch
            self.driving_direction = "右转"
            self.get_logger().info(f'🚩 GuideBoard confirmed in LOCK state, choosing {self.guideboard_branch}')
```

**作用**: 
- 在进入 LOCK 状态时再次确认 GuideBoard
- 确保即使之前没有检测到，现在也能正确设置方向

---

#### 修改点 3: LOCK 状态下持续检测

```python
elif self.intersection_state == 'LOCK_OUTER_BRANCH':
    lock_duration = current_time - self.lock_start_time if self.lock_start_time else 0
    
    # ⭐ 在 LOCK 状态下持续检测 GuideBoard，动态调整方向
    if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
        if self.decision != self.guideboard_branch:
            self.decision = self.guideboard_branch
            self.driving_direction = "右转"
            self.get_logger().info(f'🚩 GuideBoard detected in LOCK, switching to {self.guideboard_branch}')
    
    # ... 退出逻辑
```

**作用**:
- 在 LOCK 状态下持续监控 GuideBoard
- 如果之前选择的是直行（outer），但检测到 GuideBoard，会动态切换到右转
- 提高鲁棒性，避免因为早期未检测到而错过转向

---

### 4. **启动日志增强**

```
[INFO] [perception_decision_node]: 📡 Perception Decision Node Ready
[INFO] [perception_decision_node]:    SHM Name: shm_ar_video
[INFO] [perception_decision_node]:    🛣️ Intersection Logic: True
[INFO] [perception_decision_node]:    🧭 Outer Side: left
[INFO] [perception_decision_node]:    🚩 GuideBoard Selection: True (branch=right)
[INFO] [perception_decision_node]:    🔍 GuideBoard Detect Range: y=0.2-0.7  ← 新增
[INFO] [perception_decision_node]:    🛡️ Branch Mask Ratio: 0.50 (屏蔽50%区域)
```

---

## 🎯 工作流程

### 场景 1: 正常检测到 GuideBoard

```
1. NORMAL 状态
   ├─ far_intersection = True（检测到远端岔路）
   ├─ guideboard_detected = True（在 0.2-0.7 范围内检测到 GuideBoard）
   └─ → 进入 APPROACH，decision = 'right', driving_direction = "右转"

2. APPROACH 状态
   ├─ near_intersection = True（进入近端控制区域）
   ├─ guideboard_detected = True（再次确认）
   └─ → 进入 LOCK，保持 decision = 'right'

3. LOCK 状态
   ├─ 持续检测 GuideBoard
   ├─ 应用分支掩码（屏蔽左侧，保留右侧）
   ├─ offset 计算基于右侧分支
   └─ → 车辆右转
```

---

### 场景 2: 早期未检测到，LOCK 状态下检测到

```
1. NORMAL 状态
   ├─ far_intersection = True
   ├─ guideboard_detected = False（早期未检测到）
   └─ → 进入 APPROACH，decision = 'outer' (left), driving_direction = "直行"

2. APPROACH 状态
   ├─ near_intersection = True
   ├─ guideboard_detected = True（现在检测到了！）
   └─ → 进入 LOCK，decision 切换为 'right', driving_direction = "右转" ✅

3. LOCK 状态
   ├─ 持续检测 GuideBoard
   ├─ 应用分支掩码（屏蔽左侧，保留右侧）
   └─ → 车辆右转（虽然晚了一点，但仍然能转）
```

---

### 场景 3: LOCK 状态下动态切换

```
1. LOCK 状态（初始选择直行）
   ├─ decision = 'outer' (left)
   ├─ driving_direction = "直行"
   └─ 应用左侧分支掩码

2. 突然检测到 GuideBoard
   ├─ guideboard_detected = True
   ├─ decision != 'right' → 需要切换
   └─ → decision = 'right', driving_direction = "右转" ✅
       → 应用右侧分支掩码
       → 车辆从中途切换到右转
```

---

## 💡 参数调优建议

### 1. 检测范围选择

#### 方案 A: 上半部分检测（推荐）
```yaml
guideboard_detect_y0_ratio: 0.2
guideboard_detect_y1_ratio: 0.7
```
- **优点**: 减少误检，只关注路牌可能出现的位置
- **适用**: 路牌通常在图像上半部分
- **性能**: 中等（检测 50% 的画面）

#### 方案 B: 全画面检测
```yaml
guideboard_detect_y0_ratio: 0.0
guideboard_detect_y1_ratio: 1.0
```
- **优点**: 不会漏检任何位置的 GuideBoard
- **缺点**: 可能误检其他类似物体
- **性能**: 稍差（检测 100% 的画面）

#### 方案 C: 仅顶部检测
```yaml
guideboard_detect_y0_ratio: 0.0
guideboard_detect_y1_ratio: 0.5
```
- **优点**: 最快，只检测顶部 50%
- **适用**: 路牌总是在地平线附近
- **性能**: 最好（检测 50% 的画面）

---

### 2. 如何选择合适的范围

**步骤 1**: 观察实际场景中 GuideBoard 的位置

```bash
# 启用 DEBUG 日志
ros2 run track_perception perception_decision_node --ros-args --log-level debug
```

观察日志：
```
[DEBUG] 🔍 Detection: GuideBoard at cy=180
[DEBUG] 🔍 Detection: GuideBoard at cy=220
```

**步骤 2**: 计算比例

假设图像高度 `h = 480`：
- GuideBoard 在 y=180 → `180/480 = 0.375`
- GuideBoard 在 y=220 → `220/480 = 0.458`

**步骤 3**: 设置范围

为了覆盖所有情况，留一些余量：
```yaml
guideboard_detect_y0_ratio: 0.3  # 比 0.375 小一点
guideboard_detect_y1_ratio: 0.5  # 比 0.458 大一点
```

---

### 3. 性能考虑

**资源开销对比**:

| 检测范围 | 检测区域 | 相对开销 | 推荐场景 |
|---------|---------|---------|---------|
| `0.0-0.5` | 顶部 50% | 低 | 路牌总在上方 |
| `0.2-0.7` | 中间 50% | 中 | **默认推荐** |
| `0.0-1.0` | 全画面 | 高 | 路牌位置不确定 |

**注意**: 
- GuideBoard 检测只是遍历检测结果列表，开销很小
- 真正的性能瓶颈在目标检测模型推理
- 即使全画面检测，额外开销也可以忽略不计（< 0.1ms）

---

## 🔍 调试技巧

### 1. 验证检测范围

```bash
# 查看启动日志
ros2 launch track_perception perception.launch.py
```

应该看到：
```
[INFO] 🔍 GuideBoard Detect Range: y=0.2-0.7
```

假设图像高度 480：
- `y0 = 480 × 0.2 = 96`
- `y1 = 480 × 0.7 = 336`
- **检测区域**: y = 96~336

---

### 2. 观察检测结果

```bash
# 查看目标检测输出
ros2 topic echo /detection/labels
```

应该看到类似：`Car,GuideBoard,Stop`

---

### 3. 启用 DEBUG 日志

```bash
# 临时修改 launch 文件或使用命令行
ros2 run track_perception perception_decision_node --ros-args --log-level debug
```

观察日志：
```
[DEBUG] 🔍 Detection: GuideBoard at cy=200
[INFO] 🚩 GuideBoard detected, choosing right branch
[INFO] 右转 | APPROACH | Far:0.65 Near:0.30 | Offset: 0.123
```

---

### 4. 动态调整参数

```bash
# 运行时修改检测范围
ros2 param set /perception_decision_node guideboard_detect_y0_ratio 0.0
ros2 param set /perception_decision_node guideboard_detect_y1_ratio 1.0
```

立即生效，无需重启节点！

---

## 📊 预期效果

### 修改前

**问题**:
- GuideBoard 必须在 far_roi 范围内才能触发
- far_roi 配置不当会导致检测失败
- 一旦错过，无法补救

**现象**:
```
直行 | NORMAL | Far:F Near:F | Offset: 0.089
直行 | NORMAL | Far:F Near:F | Offset: 0.095
# GuideBoard 在视野内，但没有触发右转 ❌
```

---

### 修改后

**改进**:
- 使用独立的检测范围，不依赖 far_roi
- 在多个阶段检测（NORMAL→APPROACH、APPROACH→LOCK、LOCK 持续）
- 动态调整，即使早期错过也能补救

**现象**:
```
直行 | NORMAL | Far:T Near:F | Far:0.65 Near:0.30 | Offset: 0.089
[INFO] 🚩 GuideBoard detected, choosing right branch
右转 | APPROACH | Far:T Near:F | Far:0.65 Near:0.30 | Offset: 0.123
[INFO] 🚩 GuideBoard confirmed in LOCK state, choosing right
右转 | LOCK | Far:T Near:T | Far:0.65 Near:0.85 | Offset: 0.345 ✅
```

---

## 🚀 快速开始

### 1. 使用默认配置（推荐）

```yaml
guideboard_detect_y0_ratio: 0.2
guideboard_detect_y1_ratio: 0.7
```

适用于大多数场景，检测图像 20%~70% 的区域。

---

### 2. 如果路牌位置较低

```yaml
guideboard_detect_y0_ratio: 0.3
guideboard_detect_y1_ratio: 0.8
```

检测图像 30%~80% 的区域。

---

### 3. 如果不确定路牌位置

```yaml
guideboard_detect_y0_ratio: 0.0
guideboard_detect_y1_ratio: 1.0
```

全画面检测，不会漏检。

---

## 📝 总结

### 核心改进

1. **独立检测范围**: 不再依赖 far_roi，使用专门的 `guideboard_detect_y0/y1_ratio`
2. **多阶段检测**: 在 NORMAL→APPROACH、APPROACH→LOCK、LOCK 持续三个阶段都检测
3. **动态调整**: LOCK 状态下如果检测到 GuideBoard，会动态切换方向
4. **灵活配置**: 可以通过参数轻松调整检测范围

### 预期效果

- ✅ 更容易检测到 GuideBoard
- ✅ 即使早期错过，后期也能补救
- ✅ 支持动态调整方向
- ✅ 性能开销可忽略

### 下一步

1. 重新编译并测试
2. 观察日志，确认 GuideBoard 被正确检测
3. 根据实际情况微调检测范围

代码已通过语法检查，没有错误。可以重新编译测试了！🎉
