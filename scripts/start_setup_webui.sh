#!/bin/bash

set -e

APP_DIR=/home/orangepi/Desktop/setupui_new/setupUI/dist
cd "$APP_DIR"

# 更新/解压 SetupUI 后，二进制可能丢失执行权限。
for binary in ./setup_webui ./app; do
    if [[ -f "$binary" && ! -x "$binary" ]]; then
        chmod +x "$binary"
    fi
done

exec ./setup_webui
