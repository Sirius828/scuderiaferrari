# 键盘控制包 (Keyboard Controller)

## 功能说明

该ROS2包提供两种键盘控制方式：

### 1. 图形界面版本 (keyboard_control_node)
使用pygame实现，适合本地有显示器的情况。
- 实时图形界面
- 按住按键持续作用
- 释放按键停止作用

### 2. 终端版本 (terminal_keyboard_node) ⭐ 推荐SSH使用
纯文本终端实现，适合SSH远程连接。
- 无需图形界面
- 按一次切换状态（开/关）
- 彩色终端UI

**SSH用户请使用终端版本！**

## 控制方式

### 图形界面版本按键说明

| 按键 | 功能 | 说明 |
|------|------|------|
| **W** | 前进加速 | 按住加速，释放后自然减速 |
| **S** | 匀速后退 | 以固定速度后退 |
| **A** | 左转 | 按住逐渐左转到最大角度 |
| **D** | 右转 | 按住逐渐右转到最大角度 |
| **G** | 键盘接管 | 切换键盘控制是否发布 `/cmd_vel` |
| **ESC** | 退出 | 关闭控制节点 |

### 终端版本按键说明

| 按键 | 功能 | 说明 |
|------|------|------|
| **W** | 前进 | 按一次开始加速，再按一次停止 |
| **S** | 后退 | 按一次开始后退，再按一次停止 |
| **A** | 左转 | 按一次开始左转，再按一次停止 |
| **D** | 右转 | 按一次开始右转，再按一次停止 |
| **G** | 电机启停 | 切换电机使能状态 |
| **Q** | 退出 | 关闭控制节点 |

**注意**: 终端版本使用切换模式（toggle），而非按住模式。

### 控制逻辑

1. **前进控制（W键）**
   - 按住W键：速度从0开始加速，直到达到最大速度
   - 释放W键：速度自然减速到0
   - 加速度和减速度可配置

2. **后退控制（S键）**
   - 按住S键：以固定的低速后退
   - 释放S键：停止后退

3. **转向控制（A/D键）**
   - 按住A键：转向角度逐渐增加到最大左转角
   - 按住D键：转向角度逐渐增加到最大右转角
   - 释放按键：转向角度自动回正到0
   - 转向速率可配置

4. **键盘接管（G键，图形界面版本）**
   - 按G键切换键盘控制接管/释放状态
   - 释放键盘控制时，速度立即归零，但不会发布停车 `/cmd_vel`
   - 释放键盘控制时，不会向 `/chassis/enable` 发布禁用指令，方便和自动巡线节点共存

## 用户界面

程序运行时会打开一个窗口，显示：
- 当前电机状态（启用/禁用）
- 当前速度值和进度条
- 当前转向角度和指示器
- 按键状态指示
- 操作提示

## 参数配置

可通过launch文件或命令行设置以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_speed | double | 10.0 | 最大前进速度 (rad/s) |
| reverse_speed | double | 2.0 | 后退速度 (rad/s) |
| max_angle | double | 45.0 | 最大转向角度 (度) |
| acceleration | double | 5.0 | 加速度 (rad/s²) |
| deceleration | double | 5.0 | 减速度 (rad/s²) |
| turn_rate | double | 90.0 | 转向速率 (度/s) |
| update_rate | double | 20.0 | 更新频率 (Hz) |

## 依赖

- rclpy
- std_msgs
- geometry_msgs
- pygame

## 安装pygame

如果系统未安装pygame，需要先安装：

```bash
pip3 install pygame
```

或者使用系统包管理器：

```bash
sudo apt install python3-pygame
```

## 使用方法

### 1. 编译包

```bash
cd ~/scuderiaferrari
colcon build --packages-select keyboard_controller chassis_controller
source install/setup.bash
```

### 2. 启动底盘控制节点

在第一个终端中：
```bash
ros2 launch chassis_controller chassis_control.launch.py
```

### 3. 启动键盘控制节点

在第二个终端中：

#### 图形界面版本（本地使用）

**方式一：使用launch文件**
```bash
ros2 launch keyboard_controller keyboard_control.launch.py
```

**方式二：直接运行节点**
```bash
ros2 run keyboard_controller keyboard_control_node
```

#### 终端版本（SSH远程使用）⭐

**方式一：使用launch文件**
```bash
ros2 launch keyboard_controller terminal_keyboard_control.launch.py
```

**方式二：直接运行节点**
```bash
ros2 run keyboard_controller terminal_keyboard_node
```

**详细文档**: 请查看 [TERMINAL_CONTROL_GUIDE.md](TERMINAL_CONTROL_GUIDE.md)

**方式三：自定义参数**
```bash
ros2 run keyboard_controller keyboard_control_node \
  --ros-args \
  -p max_speed:=8.0 \
  -p reverse_speed:=1.5 \
  -p max_angle:=30.0 \
  -p acceleration:=3.0
```

### 4. 开始控制

1. 确保底盘控制节点正在运行
2. 键盘控制窗口会自动弹出
3. 按 **G** 键接管键盘控制
4. 使用 **WASD** 控制底盘移动
5. 按 **ESC** 退出

## 话题接口

### 发布话题

- `/cmd_vel` (geometry_msgs/Twist)
  - `linear.x`: 前后速度 (rad/s)
  - `angular.z`: 转向角度 (度)

- `/chassis/enable` (std_msgs/Int8)
  - 1: 启用电机
  - 0: 禁用电机

## 调整参数示例

### 更温和的控制
```bash
ros2 run keyboard_controller keyboard_control_node \
  --ros-args \
  -p acceleration:=3.0 \
  -p deceleration:=3.0 \
  -p turn_rate:=60.0
```

### 更灵敏的控制
```bash
ros2 run keyboard_controller keyboard_control_node \
  --ros-args \
  -p acceleration:=8.0 \
  -p deceleration:=8.0 \
  -p turn_rate:=120.0
```

### 更高的最大速度
```bash
ros2 run keyboard_controller keyboard_control_node \
  --ros-args \
  -p max_speed:=15.0
```

## 注意事项

1. **显示环境**: 需要有图形显示环境（X11/Wayland）才能运行pygame窗口
2. **焦点问题**: 确保键盘控制窗口处于焦点状态才能接收按键
3. **安全提示**: 
   - 首次使用前先在开阔场地测试
   - 随时准备按G键禁用电机
   - ESC退出时会自动发送停止命令
4. **远程访问**: 如果通过SSH远程连接，需要配置X11转发：
   ```bash
   ssh -X orangepi@your_orange_pi
   ```

## 故障排除

### 问题1：pygame窗口无法打开
**解决方法：**
```bash
# 检查是否安装了pygame
python3 -c "import pygame; print(pygame.ver)"

# 如果没有安装
pip3 install pygame
```

### 问题2：按键无响应
**解决方法：**
- 确保键盘控制窗口处于焦点状态
- 点击一下窗口标题栏
- 检查是否有其他程序占用了键盘

### 问题3：底盘不响应
**解决方法：**
1. 检查底盘控制节点是否运行
2. 确认已按G键启用电机
3. 查看终端日志是否有错误
4. 检查话题是否正常发布：
   ```bash
   ros2 topic echo /cmd_vel
   ros2 topic echo /chassis/enable
   ```

### 问题4：远程控制时窗口不显示
**解决方法：**
```bash
# 使用X11转发
ssh -X orangepi@your_orange_pi

# 或者设置DISPLAY环境变量
export DISPLAY=:0
```

## 扩展建议

1. 添加更多按键功能（如急停、模式切换等）
2. 支持游戏手柄控制
3. 添加速度曲线配置
4. 实现预设动作序列
5. 添加遥测数据显示

## 作者

orangepi

## 许可证

MIT
