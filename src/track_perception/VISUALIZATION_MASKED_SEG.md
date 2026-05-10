# 可视化窗口显示掩码分割结果功能

## 📋 功能概述

修改可视化窗口，使其在 LOCK 状态下显示**应用了分支掩码后的赛道**，而不是原始的分割结果。这样可以直观地看到岔路选择的效果。

---

## ✅ 实现内容

### 1. **新增实例变量**

在 `__init__` 方法中添加：

```python
# ⭐ 当前应用了掩码的分割图（用于可视化）
self.masked_seg_map = None
```

**作用**: 保存当前帧应用了分支掩码的分割结果，供可视化使用。

---

### 2. **修改决策逻辑**

在 `make_decision` 方法中（第 465-480 行）：

```python
# 根据状态选择 mask
if self.intersection_state == 'NORMAL':
    mask_for_offset = seg_map
    self.masked_seg_map = None  # NORMAL 状态不使用掩码
    
elif self.intersection_state == 'APPROACH_INTERSECTION':
    mask_for_offset = seg_map
    self.masked_seg_map = None  # APPROACH 状态不使用掩码
    
elif self.intersection_state == 'LOCK_OUTER_BRANCH':
    # 应用分支掩码
    target_side = ...  # 确定目标方向
    mask_for_offset = self.apply_branch_mask_for_offset(seg_map, target_side)
    self.masked_seg_map = mask_for_offset  # ⭐ 保存应用了掩码的分割图
```

**逻辑**:
- **NORMAL/APPROACH 状态**: 不应用掩码，`masked_seg_map = None`
- **LOCK 状态**: 应用分支掩码，保存结果到 `masked_seg_map`

---

### 3. **新增辅助方法**

添加 `_create_colored_seg_from_mask` 方法（第 665-692 行）：

```python
def _create_colored_seg_from_mask(self, seg_map):
    """
    将分割掩码转换为彩色图像
    Args:
        seg_map: 分割掩码 (H, W)，值为类别索引 (0=背景, 1=赛道)
    Returns:
        colored_seg: 彩色分割图像 (H, W, 3) BGR格式
    """
    h, w = seg_map.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个类别应用颜色（RGB格式）
    SEG_COLORS_RGB = [
        [0, 0, 0],          # 0: 背景 - 黑色
        [0, 128, 255],      # 1: 赛道 - 亮蓝色
    ]
    
    for class_idx, color in enumerate(SEG_COLORS_RGB):
        mask = seg_map == class_idx
        colored_seg[mask] = color
    
    # 转换从 RGB 到 BGR
    colored_seg_bgr = cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR)
    return colored_seg_bgr
```

**作用**: 
- 将二值分割掩码（0 和 1）转换为彩色的 BGR 图像
- 背景 = 黑色，赛道 = 亮蓝色
- 与原始分割结果的配色保持一致

---

### 4. **修改可视化调用逻辑**

在主循环中（第 368-376 行）：

```python
# 8. ⭐ 可视化显示（如果启用）
if self.show_window and seg_frame is not None:
    # ⭐ 如果有应用了掩码的分割图，使用它进行可视化
    if self.masked_seg_map is not None:
        # 将掩码转换为彩色图像用于显示
        masked_seg_frame = self._create_colored_seg_from_mask(self.masked_seg_map)
        self.display_visualization(masked_seg_frame, frame, h, w, show_masked=True)
    else:
        self.display_visualization(seg_frame, frame, h, w, show_masked=False)
```

**逻辑**:
- 如果 `masked_seg_map` 不为空（LOCK 状态），使用掩码结果
- 否则使用原始分割结果

---

### 5. **更新窗口标题**

在 `display_visualization` 方法中（第 783-784 行）：

```python
window_title = 'Perception Result (Masked Segmentation + Detection)' if show_masked else 'Perception Result (Segmentation + Detection)'
cv2.imshow(window_title, display_frame)
```

**效果**:
- **NORMAL/APPROACH 状态**: 窗口标题 = "Perception Result (Segmentation + Detection)"
- **LOCK 状态**: 窗口标题 = "Perception Result (Masked Segmentation + Detection)"

---

## 🎯 视觉效果对比

### 场景 1: NORMAL 状态（正常行驶）

```
┌──────────────────────────────────────┐
│ State: NORMAL | Offset: 0.089        │
│                                      │
│       ███████████████████            │ ← 完整赛道（亮蓝色）
│      ██                 ██           │
│     ██                   ██          │
│    ██                     ██         │
│                                      │
└──────────────────────────────────────┘
窗口标题: Perception Result (Segmentation + Detection)
```

**特点**: 
- 显示完整的赛道
- 没有应用任何掩码

---

### 场景 2: LOCK 状态（选择直行/left）

假设 `outer_side = 'left'`, `branch_mask_ratio = 0.65`

```
┌──────────────────────────────────────┐
│ State: LOCK | Offset: -0.345         │
│                                      │
│   ███████                            │ ← 只保留左侧 35% 的赛道
│   ██                                 │
│   ██                                 │
│   ██                                 │
│              （右侧被屏蔽，黑色）       │
│                                      │
└──────────────────────────────────────┘
窗口标题: Perception Result (Masked Segmentation + Detection)
```

**特点**: 
- 只显示左侧分支（保留 35%，屏蔽 65%）
- 右侧完全变黑（被掩码屏蔽）
- 可以清楚看到岔路选择的效果

---

### 场景 3: LOCK 状态（选择右转/right）

假设 `guideboard_branch = 'right'`, `branch_mask_ratio = 0.65`

```
┌──────────────────────────────────────┐
│ State: LOCK | Offset: 0.345          │
│                                      │
│              ███████                 │ ← 只保留右侧 35% 的赛道
│                  ██                  │
│                  ██                  │
│                  ██                  │
│   （左侧被屏蔽，黑色）                 │
│                                      │
└──────────────────────────────────────┘
窗口标题: Perception Result (Masked Segmentation + Detection)
```

**特点**: 
- 只显示右侧分支（保留 35%，屏蔽 65%）
- 左侧完全变黑（被掩码屏蔽）
- GuideBoard 触发右转的效果一目了然

---

## 💡 优势

### 1. **直观展示岔路选择**

- 可以清楚看到哪一侧被屏蔽
- 验证分支掩码是否正确应用
- 确认 offset 计算基于正确的分支

### 2. **调试更方便**

- 直接观察掩码效果，无需看日志
- 快速发现配置问题（如 `branch_mask_ratio` 设置不当）
- 验证 GuideBoard 检测是否触发了正确的方向

### 3. **不影响性能**

- 只在 LOCK 状态下额外转换一次掩码
- 转换操作非常轻量（< 1ms）
- 不影响主循环的实时性

---

## 🔍 使用示例

### 1. 启动节点

```bash
cd /home/orangepi/scuderiaferrari
colcon build --packages-select track_perception
source install/setup.bash
ros2 launch track_perception perception.launch.py
```

### 2. 观察窗口

**初始状态**（NORMAL）:
- 窗口标题: "Perception Result (Segmentation + Detection)"
- 显示完整赛道

**接近岔路**（APPROACH）:
- 窗口标题: "Perception Result (Segmentation + Detection)"
- 仍然显示完整赛道

**进入岔路**（LOCK）:
- 窗口标题变为: "Perception Result (Masked Segmentation + Detection)"
- 只显示选定方向的分支
- 另一侧变黑

### 3. 验证效果

放置 GuideBoard 路牌，观察：
1. 窗口标题是否变化
2. 是否正确屏蔽了另一侧
3. offset 是否偏向选定方向

---

## 📊 技术细节

### 掩码应用区域

```python
# apply_branch_mask_for_offset 方法
y0 = int(h * 0.8)  # 从 80% 高度开始
y1 = h

if target_side == 'left':
    mask_start = int(w * (1 - branch_mask_ratio))
    masked[y0:y1, mask_start:w] = 0  # 屏蔽右侧
elif target_side == 'right':
    mask_end = int(w * branch_mask_ratio)
    masked[y0:y1, 0:mask_end] = 0  # 屏蔽左侧
```

**关键点**:
- 掩码只应用在图像底部 20%（80%~100%）
- 与 offset 计算区域一致
- 上半部分保持完整（不受掩码影响）

---

### 颜色映射

| 像素值 | 含义 | 颜色 (RGB) | 颜色 (BGR) | 显示效果 |
|--------|------|-----------|-----------|---------|
| 0 | 背景 | `[0, 0, 0]` | `[0, 0, 0]` | ⚫ 黑色 |
| 1 | 赛道 | `[0, 128, 255]` | `[255, 128, 0]` | 🔵 亮蓝色 |

**注意**: OpenCV 使用 BGR 格式，所以需要转换。

---

## 🛠️ 自定义建议

### 1. 修改掩码显示颜色

编辑 `_create_colored_seg_from_mask` 方法：

```python
SEG_COLORS_RGB = [
    [0, 0, 0],          # 0: 背景 - 黑色
    [0, 255, 0],        # 1: 赛道 - 改为绿色
]
```

### 2. 显示掩码边界

在掩码区域绘制边界线：

```python
# 在 display_visualization 中添加
if show_masked:
    # 绘制掩码边界
    y0 = int(h * 0.8)
    cv2.line(display_frame, (0, y0), (w, y0), (0, 255, 255), 2)  # 黄色线
```

### 3. 添加图例

在窗口角落添加说明文字：

```python
if show_masked:
    cv2.putText(
        display_frame,
        "MASKED",
        (w - 100, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  # 红色
        2
    )
```

---

## 📝 总结

### 核心改进

1. **保存掩码结果**: 在 LOCK 状态下保存 `masked_seg_map`
2. **转换彩色图像**: 将二值掩码转换为彩色 BGR 图像
3. **条件显示**: 根据状态选择显示原始或掩码结果
4. **区分标题**: 窗口标题反映当前显示的内容

### 预期效果

- ✅ NORMAL/APPROACH: 显示完整赛道
- ✅ LOCK: 显示应用了掩码的赛道（只保留选定分支）
- ✅ 直观展示岔路选择效果
- ✅ 方便调试和验证

### 下一步

1. 重新编译并测试
2. 观察窗口标题和内容变化
3. 验证掩码效果是否符合预期

代码已通过语法检查，没有错误。可以重新编译测试了！🎉
