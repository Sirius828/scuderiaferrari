#!/bin/bash
# 终端键盘控制包装脚本
# 这个脚本确保节点在正确的终端环境中运行

echo "========================================"
echo "  阿克曼底盘终端键盘控制"
echo "========================================"
echo ""
echo "正在启动键盘控制节点..."
echo "按 Q 键退出程序"
echo ""

# 直接运行节点，确保stdin连接到终端
ros2 run keyboard_controller terminal_keyboard_node
