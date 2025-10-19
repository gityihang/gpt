import gradio as gr
import requests
import json

# --- 1. 后端函数定义 ---

# ==============================================================================
# 修改点 1: 引入你的流式聊天函数，并确保 MODEL_NAME 正确
# ==============================================================================
# --- 配置你的 vLLM 服务器信息 ---
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek"  # 【关键】必须与你启动 vLLM 时的 --served-model-name 参数一致
# 【新增】定义系统提示
SYSTEM_PROMPT = "you are an expert in economics"
def stream_chat(message: str, history: list, temperature: float, max_tokens: int):
    """
    与 vLLM API 交互并流式返回结果的函数。
    这个函数将直接被 gr.ChatInterface 使用。
    """
    # 1. 构造 messages 列表，从系统提示词开始
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 2. 遍历 history，将历史对话添加到 messages 列表中
    # 【关键修改点】安全地解包 history 元组
    for turn in history:
        if isinstance(turn, tuple) and len(turn) >= 3:
            # 新版 Gradio 格式: (user_message, user_files, assistant_message)
            user_msg, _, assistant_msg = turn
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        elif isinstance(turn, tuple) and len(turn) == 2:
            # 旧版 Gradio 格式: (user_message, assistant_message)
            user_msg, assistant_msg = turn
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    # 3. 将当前用户输入的消息添加到列表
    messages.append({"role": "user", "content": message})

    # 【调试】打印最终发送给 API 的完整消息列表
    print("发送给API的完整消息列表:")
    print(json.dumps(messages, indent=2, ensure_ascii=False))

    # 4. 构造请求的 payload
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    # 5. 发送流式请求并处理响应
    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, json=payload, stream=True)
        response.raise_for_status()  # 如果状态码不是 2xx，会抛出 HTTPError

        full_response_text = ""
        for line in response.iter_lines():
            if line:
                # vLLM 的流式响应格式是 "data: {json_string}"
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    json_str = line[len("data: "):]
                    try:
                        data = json.loads(json_str)
                        delta_content = data['choices'][0]['delta'].get('content', '')
                        if delta_content:
                            full_response_text += delta_content
                            yield full_response_text
                    except json.JSONDecodeError:
                        # 忽略无法解析的行
                        continue
    except requests.exceptions.RequestException as e:
        yield f"API请求出错: {e}"
    except Exception as e:
        yield f"发生未知错误: {e}"


# ==============================================================================
# 原有的其他功能函数保持不变
# ==============================================================================

def process_pdf_and_summarize(file_obj):
    """处理上传的PDF文件并生成摘要"""
    if file_obj is None:
        return "请先上传一个PDF文件。", gr.Button(interactive=False)
    
    print(f"接收到文件: {file_obj}")
    summary = f"这是对文件 {file_obj.name} 的自动生成摘要。\n\n核心内容：\n1. 关键点A\n2. 关键点B\n3. 关键点C"
    
    return summary, gr.Button(interactive=True)

def train_model(summary_data):
    """模拟模型训练过程"""
    if not summary_data:
        return "没有可用的摘要数据，无法开始训练。"
    
    print(f"开始基于以下摘要进行训练: {summary_data}")
    log = "训练日志:\n"
    for i in range(1, 6):
        log += f"Epoch {i}/5: loss={0.5/i:.4f}\n"
        yield log
    log += "\n训练完成！"
    
    return log

# --- 2. 所有UI组件定义 ---

with gr.Blocks(theme=gr.themes.Soft(), title="智能文档处理与问答助手") as demo:
    gr.Markdown("# 智能文档处理与问答助手")
    
    pdf_summary_state = gr.State(value="")

    with gr.Tabs():
        with gr.Tab("文档摘要"):
            with gr.Row():
                with gr.Column(scale=3):
                    pdf_input = gr.File(label="上传PDF文档", file_types=[".pdf"], type="filepath")
                    processbtn = gr.Button("生成摘要", variant="primary")
                with gr.Column(scale=7):
                    summary_display = gr.Markdown(value="*摘要内容将在这里显示...*", label="文档摘要")
        
        # ==============================================================================
        # 修改点 2: 更新 ChatInterface，使用新的流式函数并添加额外参数
        # ==============================================================================
        with gr.Tab("智能问答"):
            # 我们用一个额外的 gr.Blocks 来包裹 ChatInterface，以便添加更多控件
            with gr.Blocks():
                chatbot = gr.ChatInterface(
                    fn=stream_chat,  # 【关键】使用我们新的流式函数
                    title="基于文档内容的智能问答",
                    additional_inputs=[
                        # 【修改】为每个 Slider 添加一个明确的 value 属性
                        gr.Slider(minimum=0, maximum=1, value=0.7, step=0.1, label="Temperature", info="控制生成文本的随机性"),
                        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max Tokens", info="控制生成回复的最大长度"),
                    ],
                    # 【修改】简化 examples，只提供用户输入
                    # Gradio 会自动使用 additional_inputs 中定义的默认值
                    examples=[
                        ["how are you?"],
                        ["hello, how are you, can you tell me?"]
                    ],
                    type="messages"
                )

        with gr.Tab("模型训练"):
            gr.Markdown("### 使用生成的摘要来微调模型")
            current_summary_for_training = gr.Markdown(value="*当前没有可用的摘要*", label="当前摘要")
            trainbtn = gr.Button("开始微调模型", variant="primary", interactive=False)
            trainlog = gr.Textbox(label="训练日志", lines=10, interactive=False)

    # --- 3. 所有交互事件绑定 ---
    # (这部分保持不变)
    processbtn.click(
        fn=process_pdf_and_summarize,
        inputs=pdf_input,
        outputs=[summary_display, trainbtn]
    ).then(
        fn=lambda summary: (summary, summary),
        inputs=summary_display,
        outputs=[pdf_summary_state, current_summary_for_training]
    )

    trainbtn.click(
        fn=train_model,
        inputs=pdf_summary_state,
        outputs=trainlog
    )

# --- 4. 启动应用 ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
