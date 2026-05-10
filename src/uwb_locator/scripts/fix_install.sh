#!/bin/bash
# UWB Locator包安装后修复脚本
# 此脚本在colcon build后运行，创建必要的符号链接

set -e

INSTALL_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PACKAGE_NAME="uwb_locator"

echo "修复 $PACKAGE_NAME 包的安装结构..."

LIB_DIR="$INSTALL_DIR/install/$PACKAGE_NAME/lib/$PACKAGE_NAME"
BIN_FILE="$INSTALL_DIR/install/$PACKAGE_NAME/bin"

# 检查bin目录是否存在
if [ ! -d "$BIN_FILE" ]; then
    echo "错误: 找不到bin目录 $BIN_FILE"
    exit 1
fi

# 创建lib目录
mkdir -p "$LIB_DIR"

# 为所有可执行文件创建符号链接
for exe in "$BIN_FILE"/*; do
    if [ -f "$exe" ]; then
        exe_name=$(basename "$exe")
        if [ -L "$LIB_DIR/$exe_name" ] || [ -f "$LIB_DIR/$exe_name" ]; then
            rm -f "$LIB_DIR/$exe_name"
        fi
        ln -sf ../../bin/$exe_name "$LIB_DIR/$exe_name"
        echo "✓ 已创建符号链接: $LIB_DIR/$exe_name -> ../../bin/$exe_name"
    fi
done
echo "✓ 修复完成！"
