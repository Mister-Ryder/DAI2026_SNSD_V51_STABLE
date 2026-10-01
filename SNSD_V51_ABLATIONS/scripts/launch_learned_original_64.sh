#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/launch_matrix_64.sh"   --mode learned --seeds "11,22,33,44,55,66,77,88" "$@"
