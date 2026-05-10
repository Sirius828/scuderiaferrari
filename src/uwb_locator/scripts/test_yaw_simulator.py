#!/usr/bin/env python3
"""
偏航角串口模拟器
用于测试uwb_to_udp_enhanced_node的偏航角读取功能
"""

import serial
import time
import math


def simulate_yaw_data(port='/dev/ttyS0', baudrate=115200, duration=60):
    """
    模拟发送偏航角数据
    
    Args:
        port: 串口设备
        baudrate: 波特率
        duration: 运行时长（秒）
    """
    
    print("=" * 60)
    print("偏航角串口模拟器")
    print("=" * 60)
    print(f"串口: {port}")
    print(f"波特率: {baudrate}")
    print(f"格式: angle: X.XX (弧度)")
    print("=" * 60)
    
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"\n✅ 串口已打开: {port}\n")
        
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration:
            # 生成变化的偏航角（0到2π之间循环）
            elapsed = time.time() - start_time
            yaw_rad = (elapsed * 0.5) % (2 * math.pi)  # 每秒0.5弧度
            yaw_deg = yaw_rad * 180.0 / math.pi
            
            # 构建消息
            message = f"angle: {yaw_rad:.2f}\n"
            
            # 发送数据
            ser.write(message.encode('utf-8'))
            
            # 打印信息
            count += 1
            if count % 10 == 0:  # 每10次打印一次
                print(f"[{count}] 发送: {message.strip()} = {yaw_deg:.2f}°")
            
            # 控制发送频率（每秒10次）
            time.sleep(0.1)
        
        print(f"\n✅ 模拟完成，共发送 {count} 条数据")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示:")
        print("1. 检查串口是否存在: ls -l /dev/ttyS0")
        print("2. 检查权限: sudo chmod 666 /dev/ttyS0")
        print("3. 或使用其他串口: python3 test_yaw_simulator.py /dev/ttyUSB0")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == '__main__':
    import sys
    
    port = '/dev/ttyS0'
    baudrate = 115200
    duration = 60
    
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        baudrate = int(sys.argv[2])
    if len(sys.argv) > 3:
        duration = int(sys.argv[3])
    
    simulate_yaw_data(port, baudrate, duration)
