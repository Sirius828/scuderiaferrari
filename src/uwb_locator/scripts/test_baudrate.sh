#!/bin/bash
# UWB波特率测试脚本

PORT="/dev/ttyUSB0"
BAUDRATES=(115200 230400 460800 921600)

echo "=========================================="
echo "UWB波特率测试"
echo "=========================================="
echo "端口: $PORT"
echo ""

for baud in "${BAUDRATES[@]}"; do
    echo "----------------------------------------"
    echo "测试波特率: $baud"
    echo "----------------------------------------"
    
    # 使用stty设置波特率并读取少量数据
    stty -F "$PORT" "$baud" raw -echo 2>/dev/null
    
    # 读取1秒的数据
    timeout 1 cat "$PORT" > /tmp/uwb_test_${baud}.bin 2>/dev/null
    
    # 检查文件大小
    SIZE=$(wc -c < /tmp/uwb_test_${baud}.bin)
    
    if [ "$SIZE" -eq 0 ]; then
        echo "✗ 无数据"
    else
        echo "✓ 收到 $SIZE 字节"
        
        # 查找帧头
        HEADER_COUNT=$(grep -c -P '\x55\x04' /tmp/uwb_test_${baud}.bin 2>/dev/null || echo 0)
        
        if [ "$HEADER_COUNT" -gt 0 ]; then
            echo "🎉 找到 $HEADER_COUNT 个帧头 (0x55 0x04)!"
            echo "✅ 推荐波特率: $baud"
            
            # 显示前32字节的十六进制
            echo "前32字节:"
            hexdump -C /tmp/uwb_test_${baud}.bin | head -2
            
            # 清理并退出
            rm -f /tmp/uwb_test_*.bin
            exit 0
        else
            echo "✗ 未找到帧头"
            # 显示前16字节
            echo "前16字节: $(hexdump -C /tmp/uwb_test_${baud}.bin | head -1)"
        fi
    fi
    
    echo ""
done

echo "----------------------------------------"
echo "测试完成"
echo "----------------------------------------"
echo "建议:"
echo "1. 检查UWB模块的实际波特率设置"
echo "2. 查看UWB模块的文档或配置工具"
echo "3. 尝试其他波特率: 57600, 1000000, 1500000"
echo ""

# 清理
rm -f /tmp/uwb_test_*.bin
