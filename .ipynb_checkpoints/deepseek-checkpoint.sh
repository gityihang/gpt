#!/bin/bash

# 设置变量，方便以后修改
MODEL_PATH="model/DeepSeek-R1-Distill-Llama-8B"
HOST="0.0.0.0"
PORT="8000"
SERVED_MODEL_NAME="deepseek"  # 建议使用一个简单的名字，比如 deepseek
MAX_NUM_SEQS="64"

# --- 核心启动命令 ---
# 注意每一行末尾的反斜杠 \ ，除了最后一行
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --max-num-seqs "${MAX_NUM_SEQS}"
