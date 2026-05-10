# YAML 配置文件更新完成

## ✅ 已完成的更新

### 1. 更新了 `intersection_params.yaml`

在 `/home/orangepi/scuderiaferrari/src/track_perception/track_perception_python/config/intersection_params.yaml` 中添加了所有新逻辑参数：

#### 新增参数列表（共 20 个）

**功能开关**（1个）:
- `enable_segment_branch_logic: true`

**Band 扫描参数**（4个）:
- `band_count: 8`
- `band_y_min_ratio: 0.25`
- `band_y_max_ratio: 0.95`
- `band_height_ratio: 0.04`

**Segment 提取参数**（3个）:
- `min_segment_width_px: 25`
- `min_segment_gap_px: 40`
- `min_pixels_per_band: 80`

**岔路确认参数**（2个）:
- `branch_detect_min_bands: 3`
- `branch_detect_far_band_ratio: 0.6`

**分支选择参数**（2个）:
- `branch_lock_time: 2.0`
- `exit_single_path_confirm_frames: 5`

**中心线拟合参数**（5个）:
- `fit_min_points: 4`
- `fit_order: 1`
- `use_heading_term: true`
- `heading_weight: 0.35`
- `near_offset_weight: 0.65`

**安全参数**（2个）:
- `max_offset_jump: 0.6`
- `offset_smoothing_alpha: 0.4`

**调试参数**（2个）:
- `publish_debug_info: true`
- `show_branch_debug: false`

### 2. 更新了 `perception.launch.py`

在 launch 文件中添加了所有新参数的声明，确保可以通过命令行覆盖 YAML 中的值。

### 3. 配置了从 src 目录读取 YAML

Launch 文件已经配置为从 **src 目录**直接读取 YAML 配置文件：

```python
config_file_arg = DeclareLaunchArgument(
    'config_file',
    default_value=os.path.join(
        workspace_root,
        'src',
        'track_perception',
        'track_perception_python',
        'config',
        'intersection_params.yaml'
    ),
    description='Path to YAML config file (defaults to src directory)'
)
```

**重要优势**：
- ✅ 修改 YAML 后**无需重新编译**即可生效
- ✅ 直接读取 `src/track_perception/.../config/intersection_params.yaml`
- ✅ 实时修改，立即生效

---

## 🚀 如何使用

### 方式 1: 使用默认配置（从 src 目录读取）

```bash
ros2 launch track_perception perception.launch.py
```

节点会自动从以下路径加载配置：
```
/home/orangepi/scuderiaferrari/src/track_perception/track_perception_python/config/intersection_params.yaml
```

### 方式 2: 通过命令行覆盖特定参数

```bash
# 示例：关闭新逻辑，使用旧逻辑
ros2 launch track_perception perception.launch.py \
    enable_segment_branch_logic:=false

# 示例：开启可视化调试
ros2 launch track_perception perception.launch.py \
    show_branch_debug:=true

# 示例：调整 Band 数量
ros2 launch track_perception perception.launch.py \
    band_count:=6

# 示例：多个参数同时调整
ros2 launch track_perception perception.launch.py \
    enable_segment_branch_logic:=true \
    branch_detect_min_bands:=4 \
    min_segment_gap_px:=50 \
    show_branch_debug:=true
```

### 方式 3: 修改 YAML 文件（推荐用于持久化配置）

1. 编辑 `src/track_perception/track_perception_python/config/intersection_params.yaml`
2. 修改需要的参数值
3. **无需编译**，直接重新启动节点即可生效

```bash
# 编辑 YAML 文件
nano /home/orangepi/scuderiaferrari/src/track_perception/track_perception_python/config/intersection_params.yaml

# 重新启动节点（配置立即生效）
ros2 launch track_perception perception.launch.py
```

---

## 📊 参数优先级

ROS2 参数加载顺序（从高到低）：

1. **命令行参数**（最高优先级）
   ```bash
   ros2 launch ... param_name:=value
   ```

2. **YAML 配置文件**
   ```yaml
   perception_decision_node:
     ros__parameters:
       param_name: value
   ```

3. **代码中的默认值**（最低优先级）
   ```python
   self.declare_parameter('param_name', default_value)
   ```

**示例**：
- YAML 中设置 `band_count: 8`
- 命令行传入 `band_count:=6`
- 最终使用值：`6`（命令行优先）

---

## 🔍 验证配置是否生效

### 方法 1: 查看启动日志

启动节点时，会显示所有加载的参数：

```bash
ros2 launch track_perception perception.launch.py
```

应该看到：
```
[INFO] [perception_decision_node]: 🎯 Segment Branch Logic: True
```

### 方法 2: 使用 ros2 param 命令

```bash
# 列出所有参数
ros2 param list /perception_decision_node

# 查看特定参数
ros2 param get /perception_decision_node enable_segment_branch_logic
ros2 param get /perception_decision_node band_count
ros2 param get /perception_decision_node outer_side
```

### 方法 3: 运行时动态修改参数

```bash
# 运行时修改参数（无需重启节点）
ros2 param set /perception_decision_node show_branch_debug true
ros2 param set /perception_decision_node band_count 6
```

---

## 💡 常用配置场景

### 场景 1: 测试新逻辑

```yaml
# intersection_params.yaml
enable_segment_branch_logic: true
show_branch_debug: true  # 开启可视化调试
```

### 场景 2: 回退到旧逻辑

```yaml
# intersection_params.yaml
enable_segment_branch_logic: false
```

或通过命令行：
```bash
ros2 launch track_perception perception.launch.py enable_segment_branch_logic:=false
```

### 场景 3: 调优岔路检测灵敏度

```yaml
# 如果误判岔路，增大以下值
min_segment_gap_px: 60          # 从 40 增加到 60
branch_detect_min_bands: 4      # 从 3 增加到 4

# 如果检测不到岔路，减小以下值
min_segment_gap_px: 30          # 从 40 减小到 30
branch_detect_min_bands: 2      # 从 3 减小到 2
```

### 场景 4: 调整转向平滑度

```yaml
# 如果转向太抖，增强平滑
offset_smoothing_alpha: 0.25    # 从 0.4 减小到 0.25
max_offset_jump: 0.4            # 从 0.6 减小到 0.4

# 如果响应太慢，减弱平滑
offset_smoothing_alpha: 0.5     # 从 0.4 增大到 0.5
max_offset_jump: 0.8            # 从 0.6 增大到 0.8
```

---

## ⚠️ 注意事项

1. **YAML 文件格式**
   - 注意缩进（使用空格，不是 Tab）
   - 布尔值使用 `true`/`false`（小写）
   - 字符串需要引号（如 `"left"`）

2. **参数名称必须完全匹配**
   - `enable_segment_branch_logic` ≠ `enable_segment_logic`
   - 大小写敏感

3. **修改后立即生效**
   - 因为配置为从 src 目录读取
   - 无需重新编译
   - 但需要重启节点

4. **命令行参数优先级最高**
   - 如果命令行传入了参数，会覆盖 YAML 中的值
   - 调试时非常有用

5. **运行时修改参数**
   - 使用 `ros2 param set` 可以动态修改
   - 修改后立即生效，无需重启
   - 但重启后会恢复为 YAML 或默认值

---

## 📝 完整的参数列表

所有可用参数请参考：
- [BAND_SEGMENT_UPGRADE.md](./BAND_SEGMENT_UPGRADE.md) - 详细参数说明和调参建议
- [BAND_SEGMENT_QUICKSTART.md](./BAND_SEGMENT_QUICKSTART.md) - 快速开始指南

---

## ✅ 验证清单

- [x] YAML 文件已更新（包含所有新参数）
- [x] Launch 文件已更新（包含所有参数声明）
- [x] 配置为从 src 目录读取 YAML
- [x] 已重新编译
- [x] 已运行 fix_install.sh
- [x] install 目录中的 YAML 已同步

**现在可以直接使用新配置了！** 🎉
