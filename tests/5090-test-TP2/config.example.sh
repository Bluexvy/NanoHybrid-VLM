#!/usr/bin/env bash

# Copy this file to config.sh and edit the values for the rental server.
# config.sh is intentionally not required; every value can also be exported.

MODEL_PATH="/workspace/models/Qwen3.5-9B"
PYTHON_BIN="python"
CUDA_VISIBLE_DEVICES="0,1"

# quick: one repeat, 32 output tokens, reduced case matrix.
# full: three repeats, 128 output tokens, B=1/2/4/8/16 plus long-prefill cases.
SUITE_MODE="full"

GPU_MEMORY_UTILIZATION="0.82"
RUN_TP1_BASELINE="1"
RUN_GRAPH="1"
RUN_PREFIX="1"
RUN_DYNAMIC="1"
RUN_VISION="1"

# Set to 1 only when Nsight Systems is installed. This is separate from the
# main suite because tracing model initialization creates a large report.
RUN_NSYS="0"
