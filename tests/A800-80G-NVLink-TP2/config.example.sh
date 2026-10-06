#!/usr/bin/env bash

# cp config.example.sh config.sh, then edit MODEL_PATH if desired.
MODEL_PATH="/root/autodl-tmp/models/Qwen3.5-9B"
PYTHON_BIN="/root/autodl-tmp/nano-vllm/.venv/bin/python"
CUDA_VISIBLE_DEVICES="0,1"

# quick: connectivity + reduced headline matrix
# full: complete headline + 64K/128K capacity sweep + B64 soak
# extreme: full plus deliberately aggressive capacity points + B128 soak
SUITE_MODE="full"
GPU_MEMORY_UTILIZATION="0.82"

RUN_TP1_BASELINE="1"
RUN_FLA_BASELINE="1"
RUN_GRAPH="1"
RUN_LONG_CONTEXT="1"
RUN_PREFIX="1"
RUN_CAPACITY="1"
RUN_SOAK="0"
RUN_VISION="1"
RUN_DYNAMIC="1"

# Profilers remain optional because rental hosts may block GPU counters.
RUN_NSYS="0"
RUN_NCU="0"

