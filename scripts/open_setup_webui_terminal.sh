#!/bin/bash

if command -v xfce4-terminal &> /dev/null; then
    xfce4-terminal --title="SetupUI" --command="bash -c '/home/orangepi/scuderiaferrari/scripts/start_setup_webui.sh; exec bash'"
elif command -v gnome-terminal &> /dev/null; then
    gnome-terminal --title="SetupUI" -- bash -c "/home/orangepi/scuderiaferrari/scripts/start_setup_webui.sh; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -title "SetupUI" -e "bash -c '/home/orangepi/scuderiaferrari/scripts/start_setup_webui.sh; exec bash'"
else
    echo "未找到支持的终端模拟器"
    echo "请安装以下任一终端模拟器："
    echo "  - xfce4-terminal"
    echo "  - gnome-terminal"
    echo "  - xterm"
    exit 1
fi
