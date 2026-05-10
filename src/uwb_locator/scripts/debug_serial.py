"""
UWB串口数据调试工具
用于查看原始串口数据，帮助诊断问题
"""

import serial
import sys
import time


def hex_dump(data, offset=0):
    """以十六进制格式打印数据"""
    lines = []
    for i in range(0, len(data), 16):
        # 地址
        addr = f"{offset + i:08X}"
        
        # 十六进制值
        hex_part = ' '.join(f"{b:02X}" for b in data[i:i+16])
        hex_part = hex_part.ljust(48)
        
        # ASCII字符
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        
        lines.append(f"{addr}  {hex_part}  |{ascii_part}|")
    
    return '\n'.join(lines)


def find_frame_headers(data):
    """查找所有帧头 0x55 0x04 的位置"""
    positions = []
    for i in range(len(data) - 1):
        if data[i] == 0x55 and data[i+1] == 0x04:
            positions.append(i)
    return positions


def main():
    port = '/dev/ttyUSB0'
    baudrate = 921600
    
    print("=" * 80)
    print("UWB串口数据调试工具")
    print("=" * 80)
    print(f"端口: {port}")
    print(f"波特率: {baudrate}")
    print("=" * 80)
    
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0
        )
        print(f"\n✓ 串口已打开\n")
        
        frame_count = 0
        total_bytes = 0
        
        while True:
            # 读取数据
            data = ser.read(1024)
            
            if data:
                total_bytes += len(data)
                
                # 查找帧头
                headers = find_frame_headers(data)
                
                if headers:
                    frame_count += len(headers)
                    
                    print(f"\n{'='*80}")
                    print(f"收到 {len(data)} 字节，发现 {len(headers)} 个帧头")
                    print(f"累计: {frame_count} 帧, {total_bytes} 字节")
                    print(f"{'='*80}")
                    
                    # 显示每个帧头位置附近的数据
                    for idx, pos in enumerate(headers):
                        start = max(0, pos - 4)
                        end = min(len(data), pos + 132)  # 显示帧头后128字节
                        chunk = data[start:end]
                        
                        print(f"\n--- 帧头 #{idx+1} (位置 {pos}) ---")
                        print(hex_dump(chunk, offset=start))
                        
                        # 如果可能，解析帧长度
                        if pos + 4 <= len(data):
                            frame_len = data[pos+2] | (data[pos+3] << 8)
                            print(f"帧长度字段: {frame_len} 字节")
                            
                            # 检查是否有足够数据
                            if pos + frame_len <= len(data):
                                print(f"✓ 完整帧可用")
                                # 计算校验和
                                frame_data = data[pos:pos+frame_len]
                                checksum_calc = sum(frame_data[:-1]) & 0xFF
                                checksum_recv = frame_data[-1]
                                print(f"校验和: 计算={checksum_calc:02X}, 接收={checksum_recv:02X}, {'✓' if checksum_calc == checksum_recv else '✗ 失败'}")
                            else:
                                print(f"✗ 数据不完整，需要 {frame_len - (len(data) - pos)} 更多字节")
                
                # 如果没有找到帧头，显示前64字节
                elif len(data) > 0 and frame_count == 0:
                    print(f"\n收到 {len(data)} 字节，但未找到帧头 (0x55 0x04)")
                    print("前64字节:")
                    print(hex_dump(data[:64]))
            
            # Ctrl+C退出
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print(f"统计: 共 {frame_count} 帧, {total_bytes} 字节")
        print(f"{'='*80}")
        ser.close()
        print("串口已关闭")
        
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
