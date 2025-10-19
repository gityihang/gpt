#!/bin/bash

# 检查目录
current_dir=$(pwd)
if [[ $current_dir == */app/train ]]; then
    echo "检测到在 app/train/ 目录下，切换到根目录"
    cd ../..
    echo "当前目录: $(pwd)"
else
    echo "不在 app/train/ 目录下，当前目录: $(pwd)"
fi

# 导出模型（如果需要）
llamafactory-cli export app/infer/llama3_lora_sft.yaml

# 设置变量 - 确保没有空值
MODEL_PATH="model/merge"
HOST="0.0.0.0"
PORT="8000"
SERVED_MODEL_NAME="deepseek"
MAX_NUM_SEQS="64"
MAX_MODEL_LEN="80000"  # 确保这个值不为空
GPU_MEMORY_UTILIZATION="0.95"

# 调试：打印变量值
echo "=== 调试信息 ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "================"

echo "启动 vLLM API 服务器..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"