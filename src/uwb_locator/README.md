# UWB定位数据解析包

LinkTrack UWB标签节点的Python实现，用于从串口读取UWB定位数据并发布为ROS2消息。

## 功能特性

- ✅ 从串口读取LinkTrack UWB标签数据
- ✅ 解析NodeFrame2格式数据帧
- ✅ 发布位置信息（PoseWithCovarianceStamped）
- ✅ 发布速度信息（TwistStamped）
- ✅ 发布调试信息（JSON格式）
- ✅ 支持自定义串口配置
- ✅ 完整的校验和验证

## 数据结构

### NodeFrame2数据帧

| 字段 | 类型 | 说明 | 单位 |
|------|------|------|------|
| role | uint8 | 角色 | - |
| id | uint8 | 标签ID | - |
| system_time_ms | uint32 | 系统时间 | ms |
| eop_x/y/z | float | 误差范围 | m |
| pos_x/y/z | float | 位置坐标 | m |
| vel_x/y/z | float | 速度 | m/s |
| local_time_ms | uint32 | 本地时间 | ms |
| voltage_v | float | 电压 | V |
| valid_node_quantity | uint8 | 有效锚点数量 | - |
| anchors | AnchorObs[] | 锚点观测列表 | - |

### AnchorObservation锚点观测

| 字段 | 类型 | 说明 | 单位 |
|------|------|------|------|
| role | uint8 | 角色 | - |
| id | uint8 | 锚点ID | - |
| distance_m | float | 距离 | m |
| fp_rssi_db | float | 首径RSSI | dB |
| rx_rssi_db | float | 接收RSSI | dB |

## 安装依赖

```bash
# 安装pyserial
pip3 install pyserial

# 或者使用apt
sudo apt install python3-serial
```

## 编译

```bash
cd ~/scuderiaferrari
colcon build --packages-select uwb_locator
source install/setup.bash
```

## 使用方法

### 方法1: 使用launch文件启动

```bash
# 使用默认配置
ros2 launch uwb_locator uwb_tag.launch.py

# 自定义参数
ros2 launch uwb_locator uwb_tag.launch.py port:=/dev/ttyUSB1 baudrate:=921600
```

### 方法2: 直接运行节点

```bash
ros2 run uwb_locator uwb_tag_node
```

### 方法3: 使用配置文件

```bash
ros2 run uwb_locator uwb_tag_node --ros-args --params-file src/uwb_locator/config/uwb_config.yaml
```

## 发布的Topic

| Topic名称 | 消息类型 | 说明 |
|-----------|----------|------|
| `/uwb/pose` | geometry_msgs/PoseWithCovarianceStamped | 位置信息（含协方差） |
| `/uwb/twist` | geometry_msgs/TwistStamped | 速度信息 |
| `/uwb/debug` | std_msgs/String | 调试信息（JSON格式） |

## 订阅示例

### Python订阅位置数据

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class UWBSubscriber(Node):
    def __init__(self):
        super().__init__('uwb_subscriber')
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/uwb/pose',
            self.pose_callback,
            10
        )
    
    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.get_logger().info(f'Position: x={x:.3f}, y={y:.3f}, z={z:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = UWBSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 命令行查看数据

```bash
# 查看位置数据
ros2 topic echo /uwb/pose

# 查看速度数据
ros2 topic echo /uwb/twist

# 查看调试信息
ros2 topic echo /uwb/debug

# 查看话题频率
ros2 topic hz /uwb/pose
```

## 参数配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `port` | string | /dev/ttyUSB0 | 串口设备路径 |
| `baudrate` | int | 921600 | 波特率 |
| `frame_id` | string | uwb_link | ROS坐标帧ID |
| `publish_debug` | bool | true | 是否发布调试信息 |

## 数据帧格式

LinkTrack NodeFrame2数据帧结构：

```
帧头 (2字节): 0x55 0x04
帧长度 (2字节): 小端序
数据内容 (可变): 
  - 基本信息 (role, id, system_time)
  - EOP误差 (eop_x, eop_y, eop_z)
  - 位置 (pos_x, pos_y, pos_z) - 24位有符号整数 / 1000
  - 速度 (vel_x, vel_y, vel_z) - 24位有符号整数 / 10000
  - 其他信息 (local_time, voltage, valid_node_quantity)
  - 锚点观测数据 (每个锚点13字节)
校验和 (1字节): 所有字节之和的低8位
```

## 注意事项

1. **串口权限**: 确保用户有访问串口的权限
   ```bash
   sudo usermod -a -G dialout $USER
   # 重新登录后生效
   ```

2. **波特率匹配**: 确保设置的波特率与UWB模块一致

3. **坐标系**: UWB输出的坐标系需要根据实际安装情况设置TF变换

4. **EKF融合**: 协方差矩阵已根据EOP设置，可直接用于机器人定位融合

## 故障排查

### 无法打开串口

```bash
# 检查串口是否存在
ls -l /dev/ttyUSB*

# 检查权限
ls -l /dev/ttyUSB0

# 添加用户到dialout组
sudo usermod -a -G dialout $USER
```

### 没有数据输出

```bash
# 检查串口是否有数据
cat /dev/ttyUSB0

# 查看节点日志
ros2 run uwb_locator uwb_tag_node --ros-args --log-level debug
```

### 校验和失败

- 检查波特率设置是否正确
- 检查串口连接是否稳定
- 尝试降低波特率

## 与C++版本对比

| 特性 | C++版本 | Python版本 |
|------|---------|------------|
| 性能 | 高 | 中 |
| 易用性 | 中 | 高 |
| 依赖 | 少 | 需要pyserial |
| 调试 | 较难 | 容易 |
| 跨平台 | 好 | 好 |

## 许可证

MIT License

## 作者

OrangePi

## 更新日志

### v1.0.0 (2026-04-16)
- 初始版本
- 实现基本的UWB数据解析
- 支持ROS2 Humble
