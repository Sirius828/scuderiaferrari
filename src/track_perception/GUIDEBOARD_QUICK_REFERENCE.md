# GuideBoard 岔路选择 - 快速参考

## 🎯 一句话总结
**检测到岔路口时，如果 far_roi 区域有 GuideBoard，就走右侧；否则走左侧（outer_side）**

## ⚙️ 核心参数

### YAML 配置位置
`src/track_perception/track_perception_python/config/intersection_params.yaml`

### 关键参数
```yaml
perception_decision_node:
  ros__parameters:
    # 基础设置
    outer_side: "left"                    # 默认方向
    
    # GuideBoard 功能
    enable_guideboard_branch_selection: true  # 启用/禁用
    guideboard_branch: "right"                # GuideBoard 触发时的方向
```

## 🔧 常用配置场景

### 场景 1：默认左，看到标志走右（当前配置）
```yaml
outer_side: "left"
enable_guideboard_branch_selection: true
guideboard_branch: "right"
```

### 场景 2：默认右，看到标志走左
```yaml
outer_side: "right"
enable_guideboard_branch_selection: true
guideboard_branch: "left"
```

### 场景 3：始终走固定方向（不使用标志）
```yaml
outer_side: "left"
enable_guideboard_branch_selection: false
```

## 📍 检测区域 (far_roi)

GuideBoard 必须在这个区域内才会触发：

```yaml
far_roi_y0_ratio: 0.35  # ROI 起始 y = 图像高度 × 35%
far_roi_y1_ratio: 0.75  # ROI 结束 y = 图像高度 × 75%
```

**示例**：如果图像高度是 480 像素
- y0 = 480 × 0.35 = 168 像素
- y1 = 480 × 0.75 = 360 像素
- GuideBoard 中心点必须在 y=168~360 之间

## 🚀 启动命令

### 使用默认配置
```bash
ros2 launch track_perception perception.launch.py
```

### 命令行覆盖参数
```bash
ros2 launch track_perception perception.launch.py \
    enable_guideboard_branch_selection:=true \
    guideboard_branch:=right \
    outer_side:=left
```

### 禁用功能
```bash
ros2 launch track_perception perception.launch.py \
    enable_guideboard_branch_selection:=false
```

## 📊 日志关键词

### 成功检测到 GuideBoard
```
🚩 GuideBoard detected in far ROI, choosing right branch
```

### 调试信息（需要开启 debug 级别）
```
🚩 GuideBoard detected at (cx, cy), confidence=0.85
```

## 🔍 调试技巧

### 1. 查看实时日志
```bash
ros2 log echo /perception_decision_node | grep "GuideBoard"
```

### 2. 启用可视化窗口
修改 `intersection_params.yaml`：
```yaml
show_window: true
```

### 3. 检查目标检测是否正常
```bash
ros2 topic echo /detection/results
ros2 topic echo /detection/labels
```

### 4. 验证参数是否生效
```bash
ros2 param get /perception_decision_node enable_guideboard_branch_selection
ros2 param get /perception_decision_node guideboard_branch
```

## ⚠️ 常见问题速查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| GuideBoard 不触发 | 不在 far_roi 区域 | 调整 far_roi_y0/y1_ratio |
| 误触发 | 其他物体被识别为 GuideBoard | 检查目标检测模型 |
| 切换太晚 | far_roi 位置太低 | 减小 far_roi_y0_ratio |
| 切换太早 | far_roi 位置太高 | 增大 far_roi_y0_ratio |

## 📁 相关文件

- **配置**: `src/track_perception/track_perception_python/config/intersection_params.yaml`
- **代码**: `src/track_perception/track_perception_python/perception_decision_node.py`
- **启动**: `src/track_perception/track_perception_python/launch/perception.launch.py`
- **文档**: `src/track_perception/GUIDEBOARD_BRANCH_SELECTION.md`
- **测试**: `test_guideboard_branch.sh`

## 💡 快速测试流程

```bash
# 1. 运行测试脚本
bash test_guideboard_branch.sh

# 2. 重新编译（如果需要）
colcon build --packages-select track_perception
source install/setup.bash

# 3. 启动系统
ros2 launch track_perception perception.launch.py

# 4. 观察日志
# 看到 "🚩 GuideBoard detected..." 表示功能正常
```

## 🎨 工作流程图

```
正常行驶
   ↓
发现岔路口 (far_intersection=true)
   ↓
检查 far_roi 有无 GuideBoard
   ↓
   ├─ 有 → decision = guideboard_branch (right)
   └─ 无 → decision = outer_side (left)
   ↓
进入岔路口区域
   ↓
应用分支掩码（屏蔽另一侧）
   ↓
计算偏移量并控制舵机
```

## 🔑 关键代码位置

### 检测方法
`perception_decision_node.py` 第 480-507 行
```python
def check_guideboard_in_far_roi(self, img_h, img_w):
```

### 决策逻辑
`perception_decision_node.py` 第 354-362 行
```python
if self.intersection_state == 'NORMAL':
    if far_intersection:
        self.intersection_state = 'APPROACH_INTERSECTION'
        if self.enable_guideboard_branch_selection and guideboard_detected_in_far:
            self.decision = self.guideboard_branch
        else:
            self.decision = 'outer'
```

---

**最后更新**: 2026-05-09
**版本**: 1.0
