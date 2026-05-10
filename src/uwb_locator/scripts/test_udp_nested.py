#!/usr/bin/env python3
"""
UDP嵌套格式数据发送测试脚本
"""

import socket
import json
import time


def test_udp_nested_format(host='127.0.0.1', port=9000):
    """测试UDP嵌套格式连接"""
    
    print("=" * 60)
    print("UDP嵌套格式连接测试")
    print("=" * 60)
    print(f"目标地址: {host}:{port}")
    print("=" * 60)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    
    try:
        # 发送测试数据（嵌套格式）
        test_data = {
            "pos": {
                "x": 0.0,
                "y": 0.16,
                "z": 0.0
            },
            "euler": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            }
        }
        
        json_str = json.dumps(test_data, ensure_ascii=False)
        print(f"\n发送测试数据:")
        print(json.dumps(test_data, indent=2, ensure_ascii=False))
        
        sock.sendto(json_str.encode('utf-8'), (host, port))
        print(f"\n✓ 数据已发送到 {host}:{port}")
        print(f"  数据大小: {len(json_str)} 字节")
        
        try:
            data, addr = sock.recvfrom(1024)
            print(f"\n✓ 收到响应 from {addr}:")
            print(f"  {data.decode('utf-8')}")
        except socket.timeout:
            print(f"\n⚠ 未收到响应（UDP是无连接的，这可能是正常的）")
        
        print("\n" + "=" * 60)
        print("✅ UDP嵌套格式测试完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因:")
        print("1. 目标主机不可达")
        print("2. 防火墙阻止了UDP端口")
        print("3. 目标服务未启动")
        return False
        
    finally:
        sock.close()


def send_continuous_nested_data(host='127.0.0.1', port=9000, count=10, interval=0.5):
    """连续发送嵌套格式测试数据"""
    
    print("\n" + "=" * 60)
    print(f"连续发送测试 ({count} 次, 间隔 {interval}s)")
    print("=" * 60)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        for i in range(count):
            data = {
                "pos": {
                    "x": round(i * 0.1, 2),
                    "y": round(0.16 + i * 0.05, 2),
                    "z": 0.0
                },
                "euler": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": round(i * 5.0, 2)
                }
            }
            
            json_str = json.dumps(data, ensure_ascii=False)
            sock.sendto(json_str.encode('utf-8'), (host, port))
            
            print(f"[{i+1}/{count}] 发送: pos=({data['pos']['x']}, {data['pos']['y']}, {data['pos']['z']}), "
                  f"euler=({data['euler']['x']}, {data['euler']['y']}, {data['euler']['z']})")
            time.sleep(interval)
        
        print("\n✅ 连续发送测试完成")
        
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        sock.close()


if __name__ == '__main__':
    import sys
    
    host = '127.0.0.1'
    port = 9000
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    success = test_udp_nested_format(host, port)
    
    if success:
        try:
            response = input("\n是否进行连续发送测试? (y/n): ").strip().lower()
            if response == 'y':
                send_continuous_nested_data(host, port)
        except:
            pass
