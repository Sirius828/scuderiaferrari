#!/usr/bin/env bash

set -Eeuo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "用法: $0 <模式名> <控制器YAML> [--dry-run]" >&2
    exit 2
fi

MODE_NAME="$1"
CONFIG_FILE="$2"
DRY_RUN="${3:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$WORKSPACE_ROOT/src/line_follower_control_cpp/config/$CONFIG_FILE"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "错误：找不到控制器配置文件：$CONFIG_FILE" >&2
    exit 1
fi

set +u
if [[ -z "${ROS_DISTRO:-}" ]]; then
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        # shellcheck disable=SC1091
        source /opt/ros/humble/setup.bash
    else
        echo "错误：未找到 /opt/ros/humble/setup.bash" >&2
        exit 1
    fi
fi

if [[ -f "$WORKSPACE_ROOT/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE_ROOT/install/setup.bash"
else
    echo "错误：未找到工作空间环境：$WORKSPACE_ROOT/install/setup.bash" >&2
    exit 1
fi
set -u

if ! command -v ros2 >/dev/null 2>&1; then
    echo "错误：ros2 命令不可用，请先加载 ROS 2 环境" >&2
    exit 1
fi

run_launch() {
    local package_name="$1"
    local launch_file="$2"
    shift 2

    echo "[manettino][$MODE_NAME] ros2 launch $package_name $launch_file $*"
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        ros2 launch -p "$package_name" "$launch_file" "$@"
    else
        ros2 launch "$package_name" "$launch_file" "$@" &
        CHILD_PIDS+=("$!")
    fi
}

if [[ -n "$DRY_RUN" && "$DRY_RUN" != "--dry-run" ]]; then
    echo "错误：只支持可选参数 --dry-run" >&2
    exit 2
fi

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    CHILD_PIDS=()
    run_launch chassis_controller chassis_controller.launch.py
    run_launch track_perception_cpp fused_perception.launch.py
    run_launch line_follower_control_cpp controller.launch.py \
        "controller_config_file:=$CONFIG_FILE"
    echo "[manettino][$MODE_NAME] launch 描述检查完成，配置文件：$CONFIG_FILE"
    exit 0
fi

CHILD_PIDS=()

cleanup() {
    local exit_status=$?
    trap - EXIT INT TERM

    for pid in "${CHILD_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${CHILD_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    exit "$exit_status"
}

trap cleanup EXIT INT TERM

run_launch chassis_controller chassis_controller.launch.py
run_launch track_perception_cpp fused_perception.launch.py
run_launch line_follower_control_cpp controller.launch.py \
    "controller_config_file:=$CONFIG_FILE"

echo "[manettino][$MODE_NAME] 已启动底盘、融合感知和 C++ 巡线控制器。"
echo "[manettino][$MODE_NAME] 按 Ctrl-C 可同时停止这三个 launch。"

set +e
wait -n "${CHILD_PIDS[@]}"
exit_status=$?
set -e

if (( exit_status != 0 )); then
    echo "[manettino][$MODE_NAME] 某个 launch 已退出，正在停止其余节点。" >&2
fi
exit "$exit_status"
