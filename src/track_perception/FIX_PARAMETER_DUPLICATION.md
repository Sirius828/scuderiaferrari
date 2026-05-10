# 修复参数重复声明问题

## ❌ 问题描述

启动节点时出现错误：
```
rclpy.exceptions.ParameterAlreadyDeclaredException: ('Parameter(s) already declared', ['outer_side'])
```

## 🔍 原因分析

`outer_side` 参数在代码中被**重复声明**了两次：

1. **第 49 行**（旧逻辑部分）：
   ```python
   self.declare_parameter('outer_side', 'left')
   ```

2. **第 85 行**（新逻辑部分）：
   ```python
   self.declare_parameter('outer_side', 'left')
   ```

ROS2 不允许同一个参数被声明多次，因此抛出异常。

## ✅ 解决方案

删除旧逻辑部分的 `outer_side` 声明和获取，只保留新逻辑部分的：

### 修改 1: 删除重复声明（第 49 行）

**修改前**：
```python
# 岔路口检测参数
self.declare_parameter('enable_intersection_logic', True)
self.declare_parameter('outer_side', 'left')  # ← 删除这行
self.declare_parameter('far_roi_y0_ratio', 0.35)
```

**修改后**：
```python
# 岔路口检测参数（旧逻辑，已弃用）
self.declare_parameter('enable_intersection_logic', True)
# ⭐ outer_side 已在高级逻辑中声明，此处不再重复
self.declare_parameter('far_roi_y0_ratio', 0.35)
```

### 修改 2: 删除重复获取（第 119 行）

**修改前**：
```python
# 岔路口参数
self.enable_intersection_logic = self.get_parameter('enable_intersection_logic').get_parameter_value().bool_value
self.outer_side = self.get_parameter('outer_side').get_parameter_value().string_value  # ← 删除这行
self.far_roi_y0_ratio = self.get_parameter('far_roi_y0_ratio').get_parameter_value().double_value
```

**修改后**：
```python
# 岔路口参数（旧逻辑，已弃用）
self.enable_intersection_logic = self.get_parameter('enable_intersection_logic').get_parameter_value().bool_value
# ⭐ outer_side 已在高级逻辑中获取，此处不再重复
self.far_roi_y0_ratio = self.get_parameter('far_roi_y0_ratio').get_parameter_value().double_value
```

## 📝 说明

- `outer_side` 参数现在只在**新逻辑部分**（第 85 行和第 151 行）声明和获取
- 旧逻辑部分仍然保留了其他参数（如 `far_roi_y0_ratio` 等），以备兼容
- 添加了注释说明为什么删除这些行

## 🚀 验证

重新编译并运行：
```bash
cd /home/orangepi/scuderiaferrari
colcon build --packages-select track_perception
source install/setup.bash
ros2 launch track_perception perception.launch.py
```

应该能正常启动，不再出现参数重复声明的错误。

## ⚠️ 注意事项

如果将来需要同时支持新旧两套逻辑，可以考虑：
1. 使用不同的参数名称（如 `outer_side_old` 和 `outer_side_new`）
2. 或者完全移除旧逻辑相关代码

目前的做法是：**保留旧逻辑的参数声明但不使用**，以便将来如果需要回退时可以快速恢复。
