#!/bin/bash
# SoulX-Singer online serving (SVS / SVC via OpenAI chat completions + extra_args).
#
# Usage:
#   ./run_server.sh svs                    # SVS (SoulXSingerPipeline), default
#   ./run_server.sh svc                    # SVC (SoulXSingerSVCPipeline)
#
# Environment:
#   MODEL   — weight directory (SVS: model.pt + config; SVC: model-svc.pt + config)
#   PORT    — HTTP port (default 8091)
#   DIFFUSION_ATTENTION_BACKEND — optional, default FLASH_ATTN
#
# Example:
#   MODEL=/path/to/SoulX-Singer-svs ./run_server.sh svs
#   MODEL=/path/to/SoulX-Singer-svc ./run_server.sh svc

set -euo pipefail

MODE="${1:-svs}"
MODEL="${MODEL:-Soul-AILab/SoulX-Singer}"
PORT="${PORT:-8091}"
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-FLASH_ATTN}"

case "$MODE" in
    svs)
        PIPELINE_CLASS="SoulXSingerPipeline"
        ;;
    svc)
        PIPELINE_CLASS="SoulXSingerSVCPipeline"
        ;;
    *)
        echo "Unknown mode: $MODE (expected svs or svc)"
        exit 1
        ;;
esac

echo "Starting SoulX-Singer server..."
echo "  mode:     $MODE ($PIPELINE_CLASS)"
echo "  model:    $MODEL"
echo "  port:     $PORT"
echo "  backend:  $DIFFUSION_ATTENTION_BACKEND"

DIFFUSION_ATTENTION_BACKEND="$DIFFUSION_ATTENTION_BACKEND" \
    vllm serve "$MODEL" --omni \
        --model-class-name "$PIPELINE_CLASS" \
        --port "$PORT" \
        --host 0.0.0.0
