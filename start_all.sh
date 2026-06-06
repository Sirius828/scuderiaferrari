#!/bin/bash

# 启动所有服务的脚本
# 在两个终端中分别启动ROS2和SetupUI

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 第一个终端：启动ROS2 launch文件
echo '#!/bin/bash' > /tmp/terminal1.sh
echo 'source ~/scuderiaferrari/install/setup.bash' >> /tmp/terminal1.sh
echo 'ros2 launch race_control_ui race_control.launch.py' >> /tmp/terminal1.sh
chmod +x /tmp/terminal1.sh

# 第二个终端：启动setupui程序
echo '#!/bin/bash' > /tmp/terminal2.sh
echo 'cd ~/Desktop/setupui_new/setupUI/dist/' >> /tmp/terminal2.sh
echo './setup_webui' >> /tmp/terminal2.sh
chmod +x /tmp/terminal2.sh

# 使用终端模拟器打开两个终端窗口
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --tab --title="ROS2 Launch" -- bash -c "/tmp/terminal1.sh; exec bash"
    gnome-terminal --tab --title="SetupUI" -- bash -c "/tmp/terminal2.sh; exec bash"
elif command -v xfce4-terminal &> /dev/null; then
    xfce4-terminal --title="ROS2 Launch" -e "bash -c '/tmp/terminal1.sh; exec bash'" &
    xfce4-terminal --title="SetupUI" -e "bash -c '/tmp/terminal2.sh; exec bash'" &
elif command -v xterm &> /dev/null; then
    xterm -title "ROS2 Launch" -e "/tmp/terminal1.sh; exec bash" &
    xterm -title "SetupUI" -e "/tmp/terminal2.sh; exec bash" &
else
    echo "未找到支持的终端模拟器"
    echo "请安装以下任一终端模拟器："
    echo "  - gnome-terminal"
    echo "  - xfce4-terminal"
    echo "  - xterm"
    exit 1
fi

# 清理临时文件
sleep 2
rm -f /tmp/terminal1.sh /tmp/terminal2.sh
