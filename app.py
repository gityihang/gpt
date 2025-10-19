'''
Author: byh 2291816033@qq.com
Date: 2025-10-19 13:33:21
LastEditors: byh 2291816033@qq.com
LastEditTime: 2025-10-19 14:02:58
FilePath: /byh/gpt/gradio.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gradio as gr
import requests
import json

# --- 配置你的 vLLM 服务器信息 ---
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek"  # 必须与你启动 vLLM 时的 --served-model-name 参数一致

def stream_chat(message: str, history: list, temperature: float, max_tokens: int):
    """
    与 vLLM API 交互并流式返回结果的函数
    """
    # 构造请求的 payload
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True  # 关键：开启流式传输
    }

    # 使用 requests.post 发送请求，并设置 stream=True
    with requests.post(API_URL, json=payload, stream=True) as response:
        response.raise_for_status()  # 如果请求失败，会抛出异常
        
        full_response_text = ""
        # 遍历响应的每一行
        for line in response.iter_lines():
            if line:
                # 解码并移除 "data: " 前缀
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    json_str = line[6:]  # 移除 "data: "
                    # 跳过 "[DONE]" 信号
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        # 解析 JSON
                        data = json.loads(json_str)
                        # 提取内容
                        delta_content = data['choices'][0]['delta'].get('content', '')
                        if delta_content:
                            full_response_text += delta_content
                            # 使用 yield 实现流式输出
                            yield full_response_text
                    except json.JSONDecodeError:
                        # 忽略无法解析的行
                        continue

# 创建 Gradio 界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 与 {MODEL_NAME} 聊天")
    gr.ChatInterface(
        fn=stream_chat,
        additional_inputs=[
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="控制生成的随机性，值越高越有创意"
            ),
            gr.Slider(
                minimum=1,
                maximum=4096,
                value=512,
                step=1,
                label="Max Tokens",
                info="生成内容的最大长度"
            ),
        ],
        title=f"vLLM API Chat - {MODEL_NAME}",
    )

# 启动 Gradio 应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

