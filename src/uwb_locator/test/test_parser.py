"""
UWB数据帧解析器测试脚本
用于验证解析逻辑是否正确
"""

import sys
import struct

# 模拟一个NodeFrame2数据帧
def create_test_frame():
    """创建一个测试用的数据帧"""
    
    # 帧头和长度
    frame = bytearray()
    frame.extend([0x55, 0x04])  # 帧头
    
    # 计算帧长度（假设4个锚点）
    num_anchors = 4
    frame_length = 119 + 13 * num_anchors + 1  # 基础数据 + 锚点数据 + 校验和
    frame.extend(struct.pack('<H', frame_length))
    
    # 基本信息
    frame.append(1)  # role
    frame.append(10)  # id
    frame.extend(struct.pack('<I', 12345678))  # system_time_ms
    
    # EOP (误差范围)
    frame.append(5)  # eop_x = 0.05m
    frame.append(5)  # eop_y = 0.05m
    frame.append(10)  # eop_z = 0.10m
    
    # 位置 (24位有符号整数，单位mm)
    pos_x = int(1.234 * 1000)  # 1.234m
    pos_y = int(2.345 * 1000)  # 2.345m
    pos_z = int(0.100 * 1000)  # 0.100m
    
    # 写入24位有符号整数（小端序）
    def write_i24_le(value):
        # 转换为24位补码
        if value < 0:
            value = (1 << 24) + value
        return bytes([value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF])
    
    frame.extend(write_i24_le(pos_x))
    frame.extend(write_i24_le(pos_y))
    frame.extend(write_i24_le(pos_z))
    
    # 速度 (24位有符号整数，单位0.0001m/s)
    vel_x = int(0.1 * 10000)  # 0.1 m/s
    vel_y = int(-0.05 * 10000)  # -0.05 m/s
    vel_z = int(0.0 * 10000)  # 0.0 m/s
    
    frame.extend(write_i24_le(vel_x))
    frame.extend(write_i24_le(vel_y))
    frame.extend(write_i24_le(vel_z))
    
    # 填充到102字节（local_time位置）
    while len(frame) < 102:
        frame.append(0)
    
    # local_time_ms
    frame.extend(struct.pack('<I', 87654321))
    
    # 填充到116字节（voltage位置）
    while len(frame) < 116:
        frame.append(0)
    
    # voltage (16位，单位mV)
    voltage_mv = int(3.7 * 1000)  # 3.7V
    frame.extend(struct.pack('<H', voltage_mv))
    
    # valid_node_quantity
    frame.append(num_anchors)
    
    # 锚点数据
    for i in range(num_anchors):
        frame.append(0)  # role
        frame.append(i + 1)  # anchor id
        
        # 距离 (24位，单位mm)
        distance = int((1.0 + i * 0.5) * 1000)
        frame.extend(write_i24_le(distance))
        
        # RSSI
        frame.append(50)  # fp_rssi (实际值 = -2 * 50 = -100 dB)
        frame.append(45)  # rx_rssi (实际值 = -2 * 45 = -90 dB)
        
        # 填充剩余字节
        frame.extend([0] * 6)
    
    # 计算校验和
    checksum = sum(frame) & 0xFF
    frame.append(checksum)
    
    return bytes(frame)


def test_parser():
    """测试解析器"""
    print("=" * 60)
    print("UWB数据帧解析器测试")
    print("=" * 60)
    
    # 创建测试数据
    test_data = create_test_frame()
    print(f"\n生成测试数据帧，长度: {len(test_data)} 字节")
    print(f"帧头: 0x{test_data[0]:02X} 0x{test_data[1]:02X}")
    print(f"帧长度: {struct.unpack_from('<H', test_data, 2)[0]}")
    
    # 导入解析器
    sys.path.insert(0, '/home/orangepi/scuderiaferrari/src/uwb_locator')
    from uwb_locator.frame_parser import UWBFrameParser
    
    # 创建解析器
    parser = UWBFrameParser()
    
    # 添加数据
    parser.add_data(test_data)
    
    # 解析帧
    frames = parser.parse_frames()
    
    if not frames:
        print("\n❌ 解析失败！")
        return False
    
    print(f"\n✅ 成功解析 {len(frames)} 个数据帧\n")
    
    # 打印解析结果
    frame = frames[0]
    print("解析结果:")
    print(f"  标签ID: {frame.id}")
    print(f"  角色: {frame.role}")
    print(f"  系统时间: {frame.system_time_ms} ms")
    print(f"  本地时间: {frame.local_time_ms} ms")
    print(f"  电压: {frame.voltage_v:.2f} V")
    print()
    print(f"  位置 (m):")
    print(f"    X: {frame.pos_x:.3f}")
    print(f"    Y: {frame.pos_y:.3f}")
    print(f"    Z: {frame.pos_z:.3f}")
    print()
    print(f"  速度 (m/s):")
    print(f"    VX: {frame.vel_x:.3f}")
    print(f"    VY: {frame.vel_y:.3f}")
    print(f"    VZ: {frame.vel_z:.3f}")
    print()
    print(f"  误差范围 (m):")
    print(f"    EOP_X: {frame.eop_x:.2f}")
    print(f"    EOP_Y: {frame.eop_y:.2f}")
    print(f"    EOP_Z: {frame.eop_z:.2f}")
    print()
    print(f"  有效锚点数: {frame.valid_node_quantity}")
    print(f"  锚点观测数据:")
    
    for i, anchor in enumerate(frame.anchors):
        print(f"    锚点 {i+1} (ID={anchor.id}):")
        print(f"      距离: {anchor.distance_m:.3f} m")
        print(f"      FP-RSSI: {anchor.fp_rssi_db:.1f} dB")
        print(f"      RX-RSSI: {anchor.rx_rssi_db:.1f} dB")
    
    # 验证数据
    print("\n" + "=" * 60)
    print("数据验证:")
    
    checks = [
        ("标签ID", frame.id == 10),
        ("位置X", abs(frame.pos_x - 1.234) < 0.001),
        ("位置Y", abs(frame.pos_y - 2.345) < 0.001),
        ("位置Z", abs(frame.pos_z - 0.100) < 0.001),
        ("速度X", abs(frame.vel_x - 0.1) < 0.001),
        ("速度Y", abs(frame.vel_y - (-0.05)) < 0.001),
        ("锚点数", frame.valid_node_quantity == 4),
        ("电压", abs(frame.voltage_v - 3.7) < 0.01),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {'通过' if passed else '失败'}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️  部分测试失败")
        return False


if __name__ == '__main__':
    success = test_parser()
    sys.exit(0 if success else 1)
