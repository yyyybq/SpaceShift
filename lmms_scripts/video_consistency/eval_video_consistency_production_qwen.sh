#!/bin/bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../video_consistency_production" && pwd)/qwen3vl_8b.sh" "$@"
