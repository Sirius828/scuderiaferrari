#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/_launch_mode.sh" "tc_off" "controller_params.yaml" "$@"
