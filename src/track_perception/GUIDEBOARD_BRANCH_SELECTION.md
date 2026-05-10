# GuideBoard 岔路选择功能说明

## 📋 功能概述

在 `perception_decision_node` 中实现了基于目标检测的岔路选择功能：
- **默认行为**：检测到岔路口时，走 `outer_side` 指定的方向（默认左侧）
- **GuideBoard 触发**：当在 `far_roi` 区域检测到 `GuideBoard` 目标时，切换到 `guideboard_branch` 指定的方向（默认右侧）

## 🎯 工作原理

### 1. 检测流程

```
正常行驶 → 检测到岔路口(far_intersection) 
         ↓
    检查 far_roi 区域是否有 GuideBoard
         ↓
    ┌────┴────┐
    │         │
  有GuideBoard  无GuideBoard
    │         │
    ↓         ↓
走right分支  走outer_side分支(默认left)
```

### 2. 关键参数

#### YAML 配置 (`intersection_params.yaml`)

```yaml
perception_decision_node:
  ros__parameters:
    # 基础岔路口参数
    outer_side: "left"  # 默认外圈方向
    
    # GuideBoard 岔路选择参数
    enable_guideboard_branch_selection: true  # 是否启用此功能
    guideboard_branch: "right"  # 检测到GuideBoard时的选择方向
```

#### Launch 文件参数覆盖

```bash
# 启用 GuideBoard 岔路选择，检测到 GuideBoard 时走右侧
ros2 launch track_perception perception.launch.py \
    enable_guideboard_branch_selection:=true \
    guideboard_branch:=right

# 禁用此功能，始终走 outer_side 指定的方向
ros2 launch track_perception perception.launch.py \
    enable_guideboard_branch_selection:=false
```

### 3. 检测区域 (far_roi)

GuideBoard 必须在 `far_roi` 区域内才会触发岔路选择：

```yaml
# far_roi 定义（相对于图像高度的比例）
far_roi_y0_ratio: 0.35  # ROI 起始 y 坐标（图像顶部 35%）
far_roi_y1_ratio: 0.75  # ROI 结束 y 坐标（图像底部 75%）
```

**注意**：
- 检测的是目标的中心点 `cy` 是否在 ROI 范围内
- ROI 范围应该覆盖远处的赛道区域，以便提前做出决策

## 🔧 使用示例

### 场景 1：默认走左侧，看到 GuideBoard 走右侧

```yaml
perception_decision_node:
  ros__parameters:
    outer_side: "left"                    # 默认走左侧
    enable_guideboard_branch_selection: true
    guideboard_branch: "right"            # 看到 GuideBoard 走右侧
```

### 场景 2：默认走右侧，看到 GuideBoard 走左侧

```yaml
perception_decision_node:
  ros__parameters:
    outer_side: "right"                   # 默认走右侧
    enable_guideboard_branch_selection: true
    guideboard_branch: "left"             # 看到 GuideBoard 走左侧
```

### 场景 3：不使用 GuideBoard，始终走固定方向

```yaml
perception_decision_node:
  ros__parameters:
    outer_side: "left"                    # 始终走左侧
    enable_guideboard_branch_selection: false  # 禁用 GuideBoard 检测
```

## 📊 日志输出

当检测到 GuideBoard 并触发岔路选择时，会输出以下日志：

```
[INFO] [perception_decision_node]: 🚩 GuideBoard detected in far ROI, choosing right branch
```

调试模式下还会输出详细信息：

```
[DEBUG] [perception_decision_node]: 🚩 GuideBoard detected at (cx, cy), confidence=0.85
```

## ⚙️ 调优建议

### 1. 如果 GuideBoard 检测不到

**可能原因**：
- GuideBoard 不在 `far_roi` 区域内
- 目标检测模型置信度太低

**解决方法**：
- 调整 `far_roi_y0_ratio` 和 `far_roi_y1_ratio` 扩大检测区域
- 检查目标检测模型的识别效果

### 2. 如果误触发（没有 GuideBoard 也切换了）

**可能原因**：
- 其他物体被误识别为 GuideBoard

**解决方法**：
- 提高目标检测模型的准确性
- 可以在代码中添加置信度阈值过滤

### 3. 如果切换时机不对

**可能原因**：
- `far_roi` 位置不合适

**解决方法**：
- 调整 `far_roi_y0_ratio` 和 `far_roi_y1_ratio`
- 确保 ROI 覆盖到需要提前决策的区域

## 🔍 技术细节

### 检测逻辑

```python
def check_guideboard_in_far_roi(self, img_h, img_w):
    """检查在 far_roi 区域是否检测到 GuideBoard 目标"""
    if not self.latest_detections:
        return False
    
    # 计算 far_roi 的像素坐标范围
    y0 = int(img_h * self.far_roi_y0_ratio)
    y1 = int(img_h * self.far_roi_y1_ratio)
    
    # 检查每个检测结果
    for det in self.latest_detections:
        class_name = det.get('class_name', '')
        cy = det.get('cy', 0)  # 目标中心 y 坐标
        
        # 如果是 GuideBoard 且在 far_roi 区域内
        if class_name == 'GuideBoard' and y0 <= cy <= y1:
            return True
    
    return False
```

### 决策逻辑

```python
# 状态机转换
if self.intersection_state == 'NORMAL':
    if far_intersection:
        self.intersection_state = 'APPROACH_INTERSECTION'
        # 根据是否检测到GuideBoard决定岔路方向
        if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
            self.decision = self.guideboard_branch
        else:
            self.decision = 'outer'
```

## 📝 相关文件

- **配置文件**: `src/track_perception/track_perception_python/config/intersection_params.yaml`
- **节点代码**: `src/track_perception/track_perception_python/perception_decision_node.py`
- **Launch 文件**: `src/track_perception/track_perception_python/launch/perception.launch.py`

## 🚀 快速开始

1. **编辑配置文件**：
   ```bash
   nano src/track_perception/track_perception_python/config/intersection_params.yaml
   ```

2. **设置参数**：
   ```yaml
   enable_guideboard_branch_selection: true
   guideboard_branch: "right"
   ```

3. **重新编译**（如果需要）：
   ```bash
   cd /home/orangepi/scuderiaferrari
   colcon build --packages-select track_perception
   source install/setup.bash
   ```

4. **启动系统**：
   ```bash
   ros2 launch track_perception perception.launch.py
   ```

5. **观察日志**：
   ```bash
   ros2 log echo /perception_decision_node
   ```

## ❓ 常见问题

**Q: GuideBoard 检测的置信度阈值是多少？**  
A: 目前没有限制，只要目标检测模型识别为 GuideBoard 就会触发。如需添加阈值，可以修改 `check_guideboard_in_far_roi` 方法。

**Q: 可以同时使用多个标志物吗？**  
A: 当前只支持 GuideBoard。如需支持其他标志物，可以扩展 `check_guideboard_in_far_roi` 方法。

**Q: 如何验证功能是否正常工作？**  
A: 启用可视化窗口 (`show_window: true`)，观察终端输出的日志信息，确认 GuideBoard 检测状态和岔路选择决策。
