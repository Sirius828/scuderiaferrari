#!/bin/bash

set -e

APP_DIR=/home/orangepi/Desktop/setupui_new/setupUIv2/dist
cd "$APP_DIR"

# 更新/解压 SetupUI v2 后，二进制可能丢失执行权限。
for binary in ./setup_webui ./app ./xverse_ar_engine; do
    if [[ -f "$binary" && ! -x "$binary" ]]; then
        chmod +x "$binary"
    fi
done

exec ./setup_webui
