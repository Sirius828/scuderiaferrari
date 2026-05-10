# 阿克曼底盘串口控制包 (Chassis Controller)

## 📋 功能说明

该ROS2包用于通过串口控制阿克曼运动底盘。上位机（香橙派5）通过ttyS0串口与底盘通信，实现双向数据传输：
- **发送**：向底盘发送控制指令（速度、转向）
- **接收**：从底盘读取偏航角数据

## 通信协议

### 发送格式（控制指令）
**格式**: `#Flag,DIR,speed_set,servo_pwm`

**参数说明**:
- `#`: 起始标志
- `Flag`: 启停标志 (int8_t) - 1:启动, 0:停止
- `DIR`: 方向标志 (int8_t) - 1:前进, 0:后退
- `speed_set`: 速度设定值 (int16_t) - 单位: rad/s × 247
- `servo_pwm`: 舵机PWM值 (int16_t) - 2300(左满) ~ 3000(中位) ~ 3700(右满)

**示例**: `#1,1,1210,3000` 表示启动、前进、速度约4.9 rad/s、直行

### 接收格式（偏航角数据）
**格式**: `angle: 56.16`
- 单位为**度（°）**

## 话题接口

**订阅话题**:
- `/cmd_vel` (geometry_msgs/Twist)
  - `linear.x`: 前后速度 (rad/s)，正值前进，负值后退
  - `angular.z`: 转向比例 (-1.0到1.0)，正值左转，负值右转

- `/chassis/enable` (std_msgs/Int8)
  - 底盘使能控制: 1-启用, 0-禁用

- `/chassis/direction` (std_msgs/Int8) [可选]
  - 手动方向控制: 1-前进, 0-后退 (通常由cmd_vel自动判断)

**发布话题**:
- `/yaw_angle` (std_msgs/Float32)
  - 偏航角数据，单位：**度（°）**

## 参数配置

可通过launch文件或命令行设置以下参数：

- `serial_port`: 串口号 (默认: /dev/ttyS0)
- `baudrate`: 波特率 (默认: 115200)
- `max_speed`: 最大速度限制 (默认: 10.0 rad/s)
- `servo_center`: 舵机中位PWM值 (默认: 3000)
- `servo_left_max`: 舵机左打满PWM值 (默认: 2300)
- `servo_right_max`: 舵机右打满PWM值 (默认: 3700)

## 使用方法

### 1. 编译包

```bash
cd ~/scuderiaferrari
colcon build --packages-select chassis_controller
source install/setup.bash
```

### 2. 运行节点

**方式一：使用launch文件（推荐）**
```bash
ros2 launch chassis_controller chassis_controller.launch.py
```

**方式二：直接运行节点**
```bash
ros2 run chassis_controller chassis_controller_node
```

**方式三：自定义参数**
```bash
ros2 run chassis_controller chassis_controller_node \
  --ros-args \
  -p serial_port:=/dev/ttyS0 \
  -p baudrate:=115200 \
  -p max_speed:=10.0 \
  -p servo_center:=3000 \
  -p servo_left_max:=2300 \
  -p servo_right_max:=3700
```

### 3. 测试控制

**启用底盘并发布速度指令：**
```bash
# 终端1：启动节点
ros2 launch chassis_controller chassis_controller.launch.py

# 终端2：启用底盘
ros2 topic pub /chassis/enable std_msgs/msg/Int8 "data: 1"

# 终端2：发布速度指令（前进，速度 2 rad/s，直行）
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0}, angular: {z: 0.0}}"

# 终端2：查看偏航角数据
ros2 topic echo /yaw_angle
```

## 注意事项

1. **串口权限**: 确保当前用户有访问串口的权限
   ```bash
   sudo usermod -a -G dialout $USER
   # 重新登录生效
   ```

2. **串口设备**: 确认ttyS0存在且未被其他程序占用
   ```bash
   ls -l /dev/ttyS0
   ```

3. **安全机制**: 
   - 节点退出时会自动发送停止命令 `#0,1,0,0`
   - 建议在使用前先启用底盘 (`/chassis/enable`)

4. **调试**: 查看节点日志
   ```bash
   ros2 run chassis_controller chassis_controller_node --ros-args --log-level debug
   ```

## 依赖

- rclpy
- std_msgs
- geometry_msgs
- pyserial

## 作者

orangepi

## 许可证

MIT
