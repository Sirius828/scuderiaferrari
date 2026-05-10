#!/usr/bin/env python3
"""
UDP数据发送测试脚本
用于测试UDP连接和数据格式
"""

import socket
import json
import time


def test_udp_connection(host='172.20.10.2', port=9005):
    """测试UDP连接"""
    
    print("=" * 60)
    print("UDP连接测试")
    print("=" * 60)
    print(f"目标地址: {host}:{port}")
    print("=" * 60)
    
    # 创建UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)  # 2秒超时
    
    try:
        # 发送测试数据（新格式）
        test_data = {
            "type": "robot_position",
            "pos": [0.0, 0.0, 0.0],
            "euler": [0.0, 0.0, 90.0]
        }
        
        json_str = json.dumps(test_data, ensure_ascii=False)
        print(f"\n发送测试数据:")
        print(json.dumps(test_data, indent=2, ensure_ascii=False))
        
        # 发送数据
        sock.sendto(json_str.encode('utf-8'), (host, port))
        print(f"\n✓ 数据已发送到 {host}:{port}")
        print(f"  数据大小: {len(json_str)} 字节")
        
        # 尝试接收响应（可选）
        try:
            data, addr = sock.recvfrom(1024)
            print(f"\n✓ 收到响应 from {addr}:")
            print(f"  {data.decode('utf-8')}")
        except socket.timeout:
            print(f"\n⚠ 未收到响应（UDP是无连接的，这可能是正常的）")
        
        print("\n" + "=" * 60)
        print("✅ UDP连接测试完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因:")
        print("1. 目标主机不可达")
        print("2. 防火墙阻止了UDP端口")
        print("3. 目标服务未启动")
        print("4. 网络配置问题")
        return False
        
    finally:
        sock.close()


def send_continuous_data(host='172.20.10.2', port=9005, count=10, interval=0.5):
    """连续发送测试数据"""
    
    print("\n" + "=" * 60)
    print(f"连续发送测试 ({count} 次, 间隔 {interval}s)")
    print("=" * 60)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        for i in range(count):
            data = {
                "type": "robot_position",
                "pos": [round(i * 0.1, 2), round(i * 0.05, 2), 0.0],
                "euler": [0.0, 0.0, 90.0]
            }
            
            json_str = json.dumps(data, ensure_ascii=False)
            sock.sendto(json_str.encode('utf-8'), (host, port))
            
            print(f"[{i+1}/{count}] 发送: pos={data['pos']}, euler={data['euler']}")
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
    
    # 解析命令行参数
    host = '172.20.10.2'
    port = 9005
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    # 运行测试
    success = test_udp_connection(host, port)
    
    if success:
        # 询问是否进行连续测试
        try:
            response = input("\n是否进行连续发送测试? (y/n): ").strip().lower()
            if response == 'y':
                send_continuous_data(host, port)
        except:
            pass
