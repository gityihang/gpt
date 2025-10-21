import gradio as gr
from pdf2txt import converter
import os
from pathlib import Path
import time
import uuid
import shutil
import json
import subprocess
import requests
from txt2qa import (
    main as qa_main, 
    read_multiple_txt_files, 
    analyze_research_domains, 
    adaptive_question_generation, 
    save_qa_dataset,
    adaptive_question_generation_parallel,
    get_qa_progress,
    reset_qa_progress
)

# 定义统一的路径常量
BASE_DIR = Path("app")  # app目录
APPDATA_PATH = BASE_DIR
APPDATA_PATH_JSON = BASE_DIR / "train" / "data"
PDF_FOLDER = BASE_DIR / "pdf"
TXT_FOLDER = BASE_DIR / "txt"
JSONL_FOLDER = APPDATA_PATH_JSON / "jsonl"
JSONL_TRAIN_PATH = JSONL_FOLDER / "train.jsonl"
TRAIN_SCRIPT_PATH = BASE_DIR / "train" / "run.sh"

# 推理相关常量
SYSTEM_PROMPT = "你是一个专业的AI助手，能够准确回答用户的问题。"
MODEL_NAME = "deepseek-chat"
API_URL = "http://localhost:11434/v1/chat/completions"

def ensure_directories():
    """确保所有所需目录存在"""
    try:
        PDF_FOLDER.mkdir(parents=True, exist_ok=True)
        TXT_FOLDER.mkdir(parents=True, exist_ok=True)
        JSONL_FOLDER.mkdir(parents=True, exist_ok=True)
        TRAIN_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"所有目录已创建在: {APPDATA_PATH}")
        return True
    except Exception as e:
        print(f"创建目录失败: {e}")
        return False

def cleanup_folders():
    """清理所有文件夹"""
    try:
        # 清理pdf文件夹
        if PDF_FOLDER.exists():
            for file in PDF_FOLDER.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                except Exception as e:
                    print(f"删除文件 {file} 失败: {e}")
        
        # 清理txt文件夹
        if TXT_FOLDER.exists():
            for file in TXT_FOLDER.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                except Exception as e:
                    print(f"删除文件 {file} 失败: {e}")
        
        print("文件夹清理完成")
    except Exception as e:
        print(f"清理文件夹时出错: {e}")

def cleanup_jsonl_folder():
    """清理jsonl文件夹"""
    try:
        if JSONL_FOLDER.exists():
            for file in JSONL_FOLDER.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                except Exception as e:
                    print(f"删除文件 {file} 失败: {e}")
        print("JSONL文件夹清理完成")
        return True
    except Exception as e:
        print(f"清理JSONL文件夹时出错: {e}")
        return False

def create_pdf_converter_tab():
    """创建PDF转换页面"""
    
    with gr.Blocks() as tab:
        gr.Markdown("""
        # 📄 GPT低秩训练&推理
        
        使用AI技术智能处理PDF文档，提取论文标题、清理格式并转换数学公式。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 上传PDF")
                pdf_input = gr.File(
                    label="选择PDF文件（支持多选）",
                    file_types=[".pdf"],
                    file_count="multiple",
                    height=100
                )
                
                gr.Markdown("### 2. 处理选项")
                use_ai_processing = gr.Checkbox(
                    label="使用AI增强处理",
                    value=True,
                    info="启用后会自动清理格式、提取标题、转换数学公式"
                )
                
                max_workers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="最大并行处理数",
                    info="同时处理的PDF文件数量"
                )
                
                convert_btn = gr.Button(
                    "🚀 开始批量转换",
                    variant="primary",
                    size="lg"
                )
                
                cleanup_btn = gr.Button(
                    "🧹 清理文件夹",
                    variant="secondary"
                )
                cleanup_output = gr.Textbox(
                    label="清理结果",
                    visible=False
                )
                
                gr.Markdown("""
                ### 💡 使用说明
                - **AI增强处理**: 自动提取论文标题、清理格式、转换数学公式为LaTeX
                - **基础处理**: 仅提取原始文本内容
                - **并行处理**: 可同时处理多个PDF文件，提高效率
                - 文件自动保存到: `%APPDATA%/app/pdf` 和 `%APPDATA%/app/txt` 文件夹
                - 支持中英文文档和数学公式处理
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 3. 批量处理结果")
                
                task_status = gr.Markdown(
                    value="**等待处理任务...**",
                    label="📊 任务状态"
                )
                
                progress_display = gr.Textbox(
                    label="🔄 处理进度",
                    lines=6,
                    max_lines=10,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="处理进度将在这里显示..."
                )
                
                download_section = gr.Group(visible=False)
                with download_section:
                    gr.Markdown("### 📥 下载转换结果")
                    txt_download = gr.File(
                        label="下载所有TXT文件",
                        file_count="multiple",
                        interactive=False
                    )
        
        with gr.Accordion("🔧 实时处理日志", open=True):
            terminal_output = gr.Textbox(
                label="实时日志",
                lines=10,
                max_lines=15,
                show_copy_button=True,
                interactive=False,
                placeholder="实时处理日志将在这里显示...",
                elem_id="terminal-output",
                autoscroll=True
            )
        
        return {
            "component": tab,
            "inputs": {
                "pdf_input": pdf_input,
                "use_ai_processing": use_ai_processing,
                "max_workers": max_workers,
                "convert_btn": convert_btn,
                "cleanup_btn": cleanup_btn
            },
            "outputs": {
                "task_status": task_status,
                "progress_display": progress_display,
                "download_section": download_section,
                "txt_download": txt_download,
                "terminal_output": terminal_output,
                "cleanup_output": cleanup_output
            }
        }

def create_qa_generator_tab():
    """创建智能问答生成页面"""
    
    with gr.Blocks() as tab:
        gr.Markdown("""
        # 🤖 智能问答生成器
        
        基于转换后的TXT文件，使用AI技术生成深度学术问答对，适用于模型训练和研究分析。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 输入配置")
                
                txt_folder_input = gr.Textbox(
                    label="TXT文件文件夹路径",
                    value=str(TXT_FOLDER),
                    placeholder="请输入包含TXT文件的文件夹路径",
                    info=f"默认读取PDF转换页面生成的TXT文件"
                )
                
                instruction_input = gr.Textbox(
                    label="专家角色指令 (Instruction)",
                    value="你是一位专业研究人员",
                    placeholder="例如：你是一位经济学专家",
                    info="定义AI的专家角色身份"
                )
                
                language_radio = gr.Radio(
                    choices=[
                        ("英文", "english"),
                        ("中文", "chinese"), 
                        ("中英双语", "both")
                    ],
                    label="输出语言",
                    value="both",
                    info="选择问答对的语言版本"
                )
                
                num_questions_slider = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=20,
                    step=5,
                    label="生成问题数量",
                    info="建议20-50个问题以获得最佳效果"
                )
                
                qa_max_workers_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="并行处理数",
                    info="同时处理的文件数量，建议1-5个"
                )
                
                gr.Markdown("### 2. 输出选项")
                clear_jsonl_checkbox = gr.Checkbox(
                    label="生成前清空JSONL文件夹",
                    value=True,
                    info="启用后会在生成新数据前清空jsonl文件夹"
                )
                
                generate_btn = gr.Button(
                    "🚀 开始生成问答对",
                    variant="primary",
                    size="lg"
                )
                
                check_folder_btn = gr.Button(
                    "📁 检查TXT文件夹",
                    variant="secondary"
                )
                
                gr.Markdown("""
                ### 💡 使用说明
                
                **功能特点：**
                - 🎯 深度学术问题生成
                - 🔍 智能领域分析  
                - 🌐 多语言支持
                - 📚 专业学术语言
                
                **处理流程：**
                1. 读取TXT文件夹中的所有文件
                2. 分析研究领域和主题
                3. 生成深度学术问答对
                4. 保存为JSONL格式到 `%APPDATA%/app/jsonl` 文件夹
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 3. 处理结果")
                
                folder_status = gr.Markdown(
                    value="**请先检查TXT文件夹状态**",
                    label="📁 文件夹状态"
                )
                
                processing_status = gr.Markdown(
                    value="**等待开始处理...**",
                    label="📈 处理状态"
                )
                
                progress_display = gr.Textbox(
                    label="🔄 处理进度",
                    lines=8,
                    max_lines=12,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="处理进度将在这里显示..."
                )
                
                download_section = gr.Group(visible=False)
                with download_section:
                    gr.Markdown("### 📥 下载结果")
                    output_download = gr.File(
                        label="下载问答数据集",
                        file_count="single",
                        interactive=False
                    )
        
        with gr.Accordion("🔧 实时处理日志", open=True):
            terminal_output = gr.Textbox(
                label="实时日志",
                lines=10,
                max_lines=15,
                interactive=False,
                placeholder="实时处理日志将在这里显示...",
                autoscroll=True
            )
        
        return {
            "component": tab,
            "inputs": {
                "txt_folder": txt_folder_input,
                "instruction": instruction_input,
                "language": language_radio,
                "num_questions": num_questions_slider,
                "qa_max_workers": qa_max_workers_slider,
                "clear_jsonl_checkbox": clear_jsonl_checkbox,
                "generate_btn": generate_btn,
                "check_folder_btn": check_folder_btn
            },
            "outputs": {
                "folder_status": folder_status,
                "processing_status": processing_status,
                "progress_display": progress_display,
                "download_section": download_section,
                "output_download": output_download,
                "terminal_output": terminal_output
            }
        }

def create_deepseek_training_tab():
    """创建DeepSeek训练页面"""
    with gr.Blocks() as tab:
        gr.Markdown("""
        # 🚀 DeepSeek模型训练
        
        基于问答数据，启动DeepSeek模型训练任务。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 训练数据")
                
                data_source_radio = gr.Radio(
                    choices=[
                        ("使用生成的JSONL数据", "generated"),
                        ("上传JSON/JSONL文件", "upload")
                    ],
                    label="数据来源",
                    value="generated",
                    info="选择训练数据的来源"
                )
                
                json_upload_input = gr.File(
                    label="上传JSON/JSONL文件",
                    file_types=[".json", ".jsonl"],
                    file_count="single",
                    interactive=True,
                    visible=False
                )
                
                gr.Markdown("### 2. 启动训练")
                
                start_training_btn = gr.Button(
                    "🎯 开始训练",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 使用说明
                
                **数据来源选项：**
                - **生成的JSONL数据**: 使用智能问答生成页面生成的数据
                - **文件上传**: 上传自定义的JSON/JSONL文件进行训练
                
                **处理流程：**
                1. 选择数据来源
                2. 点击开始训练
                3. 系统会自动执行训练脚本
                
                **文件位置：**
                - 训练数据: `%APPDATA%/app/jsonl/train.jsonl`
                - 训练脚本: `%APPDATA%/app/train/run.sh`
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 3. 训练状态")
                
                data_status = gr.Markdown(
                    value="**请选择数据来源**",
                    label="📊 数据状态"
                )
                
                training_status = gr.Markdown(
                    value="**等待开始训练...**",
                    label="📈 训练状态"
                )
                
                training_progress = gr.Textbox(
                    label="🔄 训练进度",
                    lines=12,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="训练进度将在这里显示..."
                )
                
                model_download_section = gr.Group(visible=False)
                with model_download_section:
                    gr.Markdown("### 📥 下载训练结果")
                    model_download = gr.File(
                        label="下载训练好的模型",
                        file_count="single",
                        interactive=False
                    )
        
        def on_data_source_change(source):
            if source == "generated":
                return [
                    gr.File(visible=False),
                    "**已选择生成的JSONL数据**\n\n将使用智能问答生成页面生成的数据进行训练。"
                ]
            else:
                return [
                    gr.File(visible=True),
                    "**已选择文件上传**\n\n请上传JSON或JSONL文件进行训练。"
                ]
        
        data_source_radio.change(
            fn=on_data_source_change,
            inputs=data_source_radio,
            outputs=[json_upload_input, data_status]
        )
        
        return {
            "component": tab,
            "inputs": {
                "data_source_radio": data_source_radio,
                "json_upload_input": json_upload_input,
                "start_training_btn": start_training_btn
            },
            "outputs": {
                "data_status": data_status,
                "training_status": training_status,
                "training_progress": training_progress,
                "model_download_section": model_download_section,
                "model_download": model_download
            }
        }

def create_inference_tab():
    """创建模型推理页面"""
    with gr.Blocks() as tab:
        gr.Markdown("""
        # 🔮 模型推理测试
        
        使用训练好的模型进行推理测试。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 模型配置")
                
                model_path = gr.Textbox(
                    label="模型路径",
                    value="model/merge",
                    placeholder="模型目录路径",
                    info="请输入训练好的模型路径"
                )
                
                load_model_btn = gr.Button(
                    "📥 加载模型",
                    variant="primary",
                    size="lg"
                )
                
                model_status = gr.Markdown(
                    value="**模型未加载**",
                    label="📊 模型状态"
                )
                
                gr.Markdown("### 2. 推理设置")
                
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=3,
                    placeholder="请输入问题...",
                    info="输入您想要询问的问题"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="控制生成文本的随机性"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=100,
                    maximum=4000,
                    value=2048,
                    step=100,
                    label="最大生成长度",
                    info="控制生成文本的最大长度"
                )
                
                run_inference_btn = gr.Button(
                    "🔍 开始推理",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 使用说明
                
                **操作流程：**
                1. 输入模型路径
                2. 点击"加载模型"按钮
                3. 输入问题并设置参数
                4. 点击"开始推理"获取结果
                
                **参数说明：**
                - **Temperature**: 值越高结果越随机，值越低结果越确定
                - **最大生成长度**: 限制生成文本的长度
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("### 3. 推理结果")
                
                inference_progress = gr.Markdown(
                    value="**等待推理...**",
                    label="🔄 推理状态"
                )
                
                inference_result = gr.Textbox(
                    label="推理结果",
                    lines=12,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="推理结果将在这里显示..."
                )
                
                gr.Markdown("### 4. 推理信息")
                
                inference_info = gr.Textbox(
                    label="推理详情",
                    lines=4,
                    show_copy_button=True,
                    interactive=False,
                    placeholder="推理的详细信息将在这里显示..."
                )
        
        return {
            "component": tab,
            "inputs": {
                "model_path": model_path,
                "question_input": question_input,
                "temperature_slider": temperature_slider,
                "max_tokens_slider": max_tokens_slider,
                "load_model_btn": load_model_btn,
                "run_inference_btn": run_inference_btn
            },
            "outputs": {
                "model_status": model_status,
                "inference_progress": inference_progress,
                "inference_result": inference_result,
                "inference_info": inference_info
            }
        }

def start_training_ui(data_source, uploaded_file):
    """启动训练的UI函数 - 实时输出版本"""
    try:
        if not ensure_directories():
            return (
                "<div class='error-msg'>❌ 创建目录失败，请检查权限</div>",
                "错误：无法创建所需目录",
                gr.Group(visible=False),
                gr.File(value=None)
            )
        
        training_logs = []
        
        if data_source == "upload":
            if not uploaded_file:
                return (
                    "<div class='error-msg'>❌ 请先上传JSON/JSONL文件</div>",
                    "错误：未选择文件",
                    gr.Group(visible=False),
                    gr.File(value=None)
                )
            
            try:
                uploaded_path = uploaded_file.name
                # 直接复制上传的文件到训练路径
                shutil.copy2(uploaded_path, JSONL_TRAIN_PATH)
                training_logs.append(f"✅ 已保存训练数据到: {JSONL_TRAIN_PATH}")
            except Exception as e:
                return (
                    f"<div class='error-msg'>❌ 保存训练数据失败: {str(e)}</div>",
                    f"错误：保存文件失败 - {str(e)}",
                    gr.Group(visible=False),
                    gr.File(value=None)
                )
        
        else:  # 使用生成的数据
            jsonl_files = list(JSONL_FOLDER.glob("*.jsonl"))
            if not jsonl_files:
                return (
                    "<div class='error-msg'>❌ 未找到生成的JSONL数据，请先在智能问答生成页面生成数据</div>",
                    "错误：未找到训练数据",
                    gr.Group(visible=False),
                    gr.File(value=None)
                )
            
            latest_jsonl = max(jsonl_files, key=lambda x: x.stat().st_mtime)
            
            # 检查源文件和目标文件是否相同
            if latest_jsonl.resolve() == JSONL_TRAIN_PATH.resolve():
                training_logs.append(f"✅ 训练数据已在正确位置: {JSONL_TRAIN_PATH}")
            else:
                try:
                    shutil.copy2(latest_jsonl, JSONL_TRAIN_PATH)
                    training_logs.append(f"✅ 已复制训练数据: {latest_jsonl.name} -> {JSONL_TRAIN_PATH}")
                except Exception as e:
                    return (
                        f"<div class='error-msg'>❌ 复制训练数据失败: {str(e)}</div>",
                        f"错误：复制文件失败 - {str(e)}",
                        gr.Group(visible=False),
                        gr.File(value=None)
                    )
        
        # 检查训练脚本是否存在
        if not TRAIN_SCRIPT_PATH.exists():
            training_logs.append(f"⚠️ 训练脚本不存在: {TRAIN_SCRIPT_PATH}")
            return (
                "<div class='error-msg'>❌ 训练脚本不存在</div>",
                "\n".join(training_logs),
                gr.Group(visible=False),
                gr.File(value=None)
            )
        
        training_logs.append("🚀 开始执行训练脚本...")
        training_logs.append(f"脚本路径: {TRAIN_SCRIPT_PATH}")
        training_logs.append(f"训练数据: {JSONL_TRAIN_PATH}")
        training_logs.append("-" * 50)
        
        try:
            # 切换到train目录
            train_dir = TRAIN_SCRIPT_PATH.parent
            current_dir = os.getcwd()
            os.chdir(train_dir)
            
            # 执行训练命令
            cmd = ["llamafactory-cli", "train", "llama3_lora_sft.yaml"]
            
            training_logs.append(f"执行命令: {' '.join(cmd)}")
            training_logs.append(f"工作目录: {train_dir}")
            training_logs.append("=" * 50)
            
            # 启动训练进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并标准输出和错误输出
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            training_logs.append("✅ 训练进程已启动")
            training_logs.append("📊 实时输出:")
            
            # 实时读取输出
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    cleaned_line = line.strip()
                    output_lines.append(cleaned_line)
                    training_logs.append(cleaned_line)
            
            # 等待进程完成
            return_code = process.wait()
            
            # 切换回原目录
            os.chdir(current_dir)
            
            training_logs.append("=" * 50)
            
            # 检查训练结果
            if return_code == 0:
                training_logs.append("🎉 训练成功完成!")
                training_logs.append(f"退出码: {return_code}")
                
                # 检查是否有模型输出
                output_dir = train_dir / "output"
                if output_dir.exists():
                    model_files = list(output_dir.glob("**/*.bin")) + list(output_dir.glob("**/*.safetensors"))
                    if model_files:
                        training_logs.append(f"✅ 找到训练好的模型文件: {len(model_files)} 个")
                        for model_file in model_files[:3]:  # 显示前3个文件
                            training_logs.append(f"  - {model_file.name}")
                    else:
                        training_logs.append("⚠️ 未找到模型权重文件，可能训练配置有问题")
                
                success_msg = f"""
                <div class='success-msg'>
                ✅ 训练任务成功完成!
                - 退出码: {return_code}
                - 训练日志已保存
                </div>
                """
                
                return (
                    success_msg,
                    "\n".join(training_logs),
                    gr.Group(visible=True),
                    gr.File(value=None)
                )
            else:
                training_logs.append(f"❌ 训练失败!")
                training_logs.append(f"退出码: {return_code}")
                training_logs.append("请检查训练配置和数据格式")
                
                # 分析可能的错误原因
                error_output = "\n".join(output_lines[-10:])  # 显示最后10行输出
                training_logs.append("最近输出:")
                training_logs.append(error_output)
                
                error_msg = f"""
                <div class='error-msg'>
                ❌ 训练任务失败!
                - 退出码: {return_code}
                - 请检查训练配置、数据格式和依赖包
                </div>
                """
                
                return (
                    error_msg,
                    "\n".join(training_logs),
                    gr.Group(visible=False),
                    gr.File(value=None)
                )
            
        except Exception as e:
            # 确保切换回原目录
            try:
                os.chdir(current_dir)
            except:
                pass
            
            training_logs.append(f"❌ 执行训练脚本失败: {str(e)}")
            training_logs.append(f"错误类型: {type(e).__name__}")
            return (
                f"<div class='error-msg'>❌ 训练启动失败: {str(e)}</div>",
                "\n".join(training_logs),
                gr.Group(visible=False),
                gr.File(value=None)
            )
            
    except Exception as e:
        return (
            f"<div class='error-msg'>❌ 训练过程出错: {str(e)}</div>",
            f"错误: {str(e)}",
            gr.Group(visible=False),
            gr.File(value=None)
        )

def load_instructions_from_jsonl(jsonl_path, max_instructions=5):
    """
    从JSONL文件中读取instruction作为历史对话
    """
    instructions = []
    try:
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_instructions:
                        break
                    try:
                        data = json.loads(line.strip())
                        if 'instruction' in data and data['instruction']:
                            instructions.append(data['instruction'])
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"警告: JSONL文件不存在: {jsonl_path}")
    except Exception as e:
        print(f"读取JSONL文件出错: {e}")
    
    return instructions

def load_model_handler(model_path):
    """处理模型加载"""
    try:
        # 检查模型路径是否存在
        model_dir = Path(model_path)
        if not model_dir.exists():
            return (
                "<div class='error-msg'>❌ 模型路径不存在</div>",
                f"错误：找不到模型路径 {model_path}"
            )
        
        # 检查模型文件
        model_files = list(model_dir.glob("**/*.bin")) + list(model_dir.glob("**/*.safetensors"))
        if not model_files:
            return (
                "<div class='error-msg'>❌ 未找到模型文件</div>",
                f"在 {model_path} 中未找到 .bin 或 .safetensors 文件"
            )
        
        # 这里可以添加实际的模型加载逻辑
        # 例如调用相应的模型加载函数
        
        return (
            f"<div class='success-msg'>✅ 模型加载成功</div>",
            f"模型信息:\n"
            f"- 路径: {model_path}\n"
            f"- 找到 {len(model_files)} 个模型文件\n"
            f"- 示例文件: {model_files[0].name if model_files else '无'}\n"
            f"- 加载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
    except Exception as e:
        return (
            f"<div class='error-msg'>❌ 模型加载失败: {str(e)}</div>",
            f"错误详情: {str(e)}"
        )

def run_inference_handler(model_path, question, temperature, max_tokens):
    """处理推理请求"""
    try:
        # 检查模型是否已加载
        model_dir = Path(model_path)
        if not model_dir.exists():
            return (
                "<div class='error-msg'>❌ 请先加载模型</div>",
                "错误：模型路径不存在，请先点击'加载模型'",
                f"模型路径: {model_path}\n问题: {question}\n状态: 模型未加载"
            )
        
        if not question.strip():
            return (
                "<div class='error-msg'>❌ 请输入问题</div>",
                "错误：问题不能为空",
                "请先输入您想要询问的问题"
            )
        
        # 显示推理开始信息
        inference_info = (
            f"推理参数:\n"
            f"- 模型路径: {model_path}\n"
            f"- 问题: {question}\n"
            f"- Temperature: {temperature}\n"
            f"- 最大长度: {max_tokens}\n"
            f"- 开始时间: {time.strftime('%H:%M:%S')}"
        )
        
        # 调用推理函数
        result = run_inference_ui_with_params(model_path, question, temperature, max_tokens)
        
        return (
            "<div class='success-msg'>✅ 推理完成</div>",
            result,
            inference_info + f"\n- 完成时间: {time.strftime('%H:%M:%S')}"
        )
        
    except Exception as e:
        return (
            f"<div class='error-msg'>❌ 推理失败: {str(e)}</div>",
            f"错误：{str(e)}",
            f"模型路径: {model_path}\n问题: {question}\n错误: {str(e)}"
        )

def run_inference_ui_with_params(model_path, question, temperature=0.7, max_tokens=2048):
    """
    带参数的推理函数
    """
    try:
        # 0. 先检查服务是否已启动，如果没有则启动
        if not is_service_running():
            start_result = start_inference_service()
            if not start_result:
                return "❌ 启动推理服务失败，请检查 infer.sh 文件"
            time.sleep(5)  # 等待服务完全启动
        
        # 1. 从JSONL文件加载instructions作为历史
        instructions = load_instructions_from_jsonl(JSONL_TRAIN_PATH)
        
        # 2. 构造 messages 列表，从系统提示词开始
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 3. 将instructions作为历史对话添加到messages中
        for instruction in instructions:
            messages.append({"role": "user", "content": instruction})

        # 4. 将当前用户输入的消息添加到列表
        messages.append({"role": "user", "content": question})

        # 5. 构造请求的 payload
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        # 6. 发送请求
        response = requests.post(
            API_URL, 
            headers={"Content-Type": "application/json"}, 
            json=payload, 
            timeout=60
        )
        response.raise_for_status()

        # 7. 解析响应
        data = response.json()
        full_response = data['choices'][0]['message']['content']
        
        return f"模型路径: {model_path}\n问题: {question}\n回答: {full_response}"

    except requests.exceptions.RequestException as e:
        return f"模型路径: {model_path}\n问题: {question}\n错误: API请求出错 - {e}"
    except Exception as e:
        return f"模型路径: {model_path}\n问题: {question}\n错误: 发生未知错误 - {e}"

def is_service_running():
    """检查推理服务是否已经在运行"""
    try:
        response = requests.get(API_URL.replace("/v1/chat/completions", "/api/tags"), timeout=5)
        return response.status_code == 200
    except:
        return False

def start_inference_service():
    """启动推理服务"""
    try:
        # 运行 infer.sh 脚本
        process = subprocess.Popen(
            ["bash", "app/infer/infer.sh"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待一段时间看是否启动成功
        time.sleep(3)
        return process.poll() is None  # 如果进程还在运行，说明启动成功
    except Exception as e:
        print(f"启动服务失败: {e}")
        return False

# 全局变量用于存储实时日志
real_time_logs = []
qa_processing_logs = []

def add_log_message(message):
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    real_time_logs.append(log_entry)
    if len(real_time_logs) > 100:
        real_time_logs.pop(0)

def add_qa_log(message):
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    qa_processing_logs.append(log_entry)
    if len(qa_processing_logs) > 100:
        qa_processing_logs.pop(0)

def get_real_time_logs():
    return "\n".join(real_time_logs)

def get_qa_logs():
    return "\n".join(qa_processing_logs)

def process_multiple_pdfs(pdf_files, use_ai_processing, max_workers):
    real_time_logs.clear()
    
    if not pdf_files:
        add_log_message("❌ 请先上传PDF文件")
        return (
            "<div class='error-msg'>❌ 请先上传PDF文件</div>",
            "等待处理任务...",
            gr.Group(visible=False),
            None,
            get_real_time_logs()
        )
    
    converter.max_workers = max_workers
    
    try:
        add_log_message(f"🚀 开始并行处理 {len(pdf_files)} 个PDF文件")
        add_log_message(f"📊 最大并行数: {max_workers}")
        add_log_message(f"🔧 AI增强处理: {'启用' if use_ai_processing else '禁用'}")
        
        task_ids = converter.convert_multiple_pdfs(pdf_files, use_ai_processing)
        
        status_msg = f"""
        <div class='info-box'>
        🔄 开始并行处理 {len(pdf_files)} 个PDF文件
        - 最大并行数: {max_workers}
        - 任务ID: {', '.join(task_ids)}
        - 状态: 处理中...
        </div>
        """
        
        progress_text = f"开始处理 {len(pdf_files)} 个文件...\n"
        progress_text += f"并行处理数: {max_workers}\n"
        progress_text += "=" * 50 + "\n"
        
        add_log_message("📋 任务已提交，开始处理...")
        
        return (
            status_msg,
            progress_text,
            gr.Group(visible=False),
            None,
            get_real_time_logs()
        )
        
    except Exception as e:
        error_msg = f"启动并行处理时出错: {str(e)}"
        add_log_message(f"❌ {error_msg}")
        return (
            f"<div class='error-msg'>❌ {error_msg}</div>",
            f"错误: {error_msg}",
            gr.Group(visible=False),
            None,
            get_real_time_logs()
        )

def update_progress():
    progress_updates = converter.get_progress()
    
    download_files = []
    all_completed = True
    completed_count = 0
    total_count = len(converter.current_tasks)
    
    new_logs = []
    for update in progress_updates:
        if update['type'] == 'progress':
            new_logs.append(f"[{update['task_id']}] {update['message']}")
        elif update['type'] == 'result':
            if update['result']['success']:
                download_files.append(update['result']['txt_path'])
                completed_count += 1
                new_logs.append(f"✅ {update['task_id']} 处理完成: {update['result']['filename']}")
            else:
                new_logs.append(f"❌ {update['task_id']} 处理失败: {update['result'].get('error', '未知错误')}")
            
            task_status = converter.get_task_status(update['task_id'])
            if task_status and task_status['result'] is None:
                all_completed = False
    
    for log in new_logs:
        add_log_message(log)
    
    progress_text = f"处理进度: {completed_count}/{total_count} 个任务完成\n"
    progress_text += "=" * 50 + "\n"
    
    recent_tasks = list(converter.current_tasks.keys())[-5:]
    for task_id in recent_tasks:
        task = converter.current_tasks.get(task_id)
        if task:
            status = "✅ 完成" if task['result'] else "🔄 处理中"
            progress_text += f"- {task_id}: {status}\n"
            if task['result'] and task['result']['success']:
                progress_text += f"  输出: {task['result']['filename']}\n"
    
    if all_completed and total_count > 0:
        status_msg = f"<div class='success-msg'>✅ 所有任务处理完成！共完成 {completed_count} 个文件</div>"
        progress_text += f"\n✅ 所有 {completed_count} 个文件处理完成！"
        add_log_message(f"🎉 所有任务处理完成！共完成 {completed_count} 个文件")
        return (
            status_msg,
            progress_text,
            gr.Group(visible=True),
            download_files,
            get_real_time_logs()
        )
    else:
        status_msg = f"<div class='info-box'>🔄 处理中... ({completed_count}/{total_count} 完成)</div>"
        return (
            status_msg,
            progress_text,
            gr.Group(visible=False),
            None,
            get_real_time_logs()
        )

def clear_tasks():
    converter.current_tasks.clear()
    while not converter.progress_queue.empty():
        converter.progress_queue.get()
    real_time_logs.clear()
    return "任务历史已清空"

def cleanup_folders_ui():
    try:
        cleanup_folders()
        return "✅ 文件夹清理完成"
    except Exception as e:
        return f"❌ 清理失败: {str(e)}"

def get_monitoring_info():
    total_tasks = len(converter.current_tasks)
    completed = sum(1 for task in converter.current_tasks.values() if task['result'])
    
    log_text = f"系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_text += f"总任务数: {total_tasks}\n"
    log_text += f"已完成: {completed}\n"
    log_text += f"处理中: {total_tasks - completed}\n"
    log_text += "=" * 30 + "\n"
    
    for task_id, task in list(converter.current_tasks.items())[-10:]:
        status = "✅ 完成" if task['result'] else "🔄 处理中"
        log_text += f"{task_id}: {status}\n"
    
    return total_tasks, completed, log_text

def check_txt_folder(txt_folder):
    if not os.path.exists(txt_folder):
        add_qa_log(f"❌ 文件夹不存在: {txt_folder}")
        return (
            "<div class='error-msg'>❌ 文件夹不存在</div>",
            f"❌ 文件夹不存在: {txt_folder}",
            get_qa_logs()
        )
    
    txt_files = list(Path(txt_folder).glob("*.txt"))
    if not txt_files:
        add_qa_log(f"❌ 文件夹中未找到TXT文件: {txt_folder}")
        return (
            "<div class='error-msg'>❌ 未找到TXT文件</div>",
            f"❌ 文件夹中未找到TXT文件: {txt_folder}",
            get_qa_logs()
        )
    
    file_info = "\n".join([f"- {f.name} ({os.path.getsize(f)} bytes)" for f in txt_files[:10]])
    if len(txt_files) > 10:
        file_info += f"\n- ... 还有 {len(txt_files) - 10} 个文件"
    
    add_qa_log(f"✅ 找到 {len(txt_files)} 个TXT文件")
    return (
        f"<div class='success-msg'>✅ 找到 {len(txt_files)} 个TXT文件</div>",
        f"✅ 找到 {len(txt_files)} 个TXT文件:\n{file_info}",
        get_qa_logs()
    )

def run_qa_generation(txt_folder, instruction, language, num_questions, clear_jsonl, qa_max_workers):
    """运行智能问答生成"""
    # 清空日志和进度
    qa_processing_logs.clear()
    reset_qa_progress()
    
    add_qa_log("🚀 开始智能问答生成流程")
    add_qa_log(f"📁 TXT文件夹: {txt_folder}")
    add_qa_log(f"🎯 专家指令: {instruction}")
    add_qa_log(f"🌐 输出语言: {language}")
    add_qa_log(f"📊 问题数量: {num_questions}")
    add_qa_log(f"🧹 清空JSONL: {'是' if clear_jsonl else '否'}")
    add_qa_log(f"⚡ 并行处理数: {qa_max_workers}")
    
    try:
        # 0. 如果需要，清空JSONL文件夹
        if clear_jsonl:
            add_qa_log("🧹 正在清空JSONL文件夹...")
            if cleanup_jsonl_folder():
                add_qa_log("✅ JSONL文件夹清空完成")
            else:
                add_qa_log("⚠️ JSONL文件夹清空失败，但继续处理")
        
        # 1. 读取TXT文件
        add_qa_log("📖 正在读取TXT文件...")
        contents = read_multiple_txt_files(txt_folder)
        
        if not contents:
            error_msg = f"❌ 在文件夹 {txt_folder} 中未找到TXT文件"
            add_qa_log(error_msg)
            return (
                f"<div class='error-msg'>{error_msg}</div>",
                error_msg,
                gr.Group(visible=False),
                gr.File(value=None),
                get_qa_logs()
            )
        
        # 检查并行数是否合理
        if qa_max_workers > len(contents):
            error_msg = f"❌ 并行处理数 {qa_max_workers} 大于文件数 {len(contents)}，请减少并行数"
            add_qa_log(error_msg)
            return (
                f"<div class='error-msg'>{error_msg}</div>",
                error_msg,
                gr.Group(visible=False),
                gr.File(value=None),
                get_qa_logs()
            )
        
        add_qa_log(f"✅ 成功读取 {len(contents)} 个TXT文件")
        
        # 2. 分析研究领域
        add_qa_log("🔍 正在分析研究领域和主题...")
        domain_analysis = analyze_research_domains(contents)
        
        progress_text = f"处理进度:\n"
        progress_text += f"- 已读取 {len(contents)} 个TXT文件\n"
        progress_text += f"- 主要领域: {', '.join(domain_analysis['primary_domains'])}\n"
        progress_text += f"- 核心主题: {', '.join(domain_analysis['primary_themes'])}\n"
        progress_text += f"- 关键词: {', '.join(domain_analysis['top_keywords'][:5])}\n"
        progress_text += f"- 并行处理数: {qa_max_workers}\n"
        progress_text += "-" * 50 + "\n"
        progress_text += "正在并行生成深度学术问题...\n"
        
        # 3. 生成问答对（使用并行版本）
        add_qa_log("🤖 开始并行生成深度学术问答对...")
        qa_pairs = adaptive_question_generation_parallel(contents, num_questions, language, domain_analysis, qa_max_workers)
        
        if not qa_pairs:
            error_msg = "❌ 问答对生成失败，请检查API配置或重试"
            add_qa_log(error_msg)
            return (
                f"<div class='error-msg'>{error_msg}</div>",
                progress_text + f"\n{error_msg}",
                gr.Group(visible=False),
                gr.File(value=None),
                get_qa_logs()
            )
        
        add_qa_log(f"✅ 成功生成 {len(qa_pairs)} 个深度学术问答对")
        
        # 4. 保存结果到本地jsonl文件夹和训练路径
        add_qa_log("💾 正在保存结果...")
        
        # 确保目录存在
        ensure_directories()
        
        # 确定输出文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        local_output_file = JSONL_FOLDER / f"qa_academic_{timestamp}.jsonl"
        
        # 保存到本地
        total_records = save_qa_dataset(qa_pairs, str(local_output_file), instruction, language)
        
        # 检查是否需要复制到训练路径（如果文件不同）
        if local_output_file.resolve() != JSONL_TRAIN_PATH.resolve():
            try:
                shutil.copy2(local_output_file, JSONL_TRAIN_PATH)
                add_qa_log(f"✅ 训练数据已保存到: {JSONL_TRAIN_PATH}")
            except Exception as e:
                add_qa_log(f"⚠️ 保存到训练路径失败: {e}")
        else:
            add_qa_log(f"✅ 训练数据已在正确位置: {JSONL_TRAIN_PATH}")
        
        progress_text = f"处理进度:\n"
        progress_text += f"- 已读取 {len(contents)} 个TXT文件\n"
        progress_text += f"- 成功生成: {len(qa_pairs)} 个问答对\n"
        progress_text += f"- 本地保存: {local_output_file}\n"
        progress_text += f"- 训练路径: {JSONL_TRAIN_PATH}\n"
        progress_text += f"- 总记录: {total_records} 条"
        
        success_msg = f"""
        <div class='success-msg'>
        ✅ 智能问答生成完成!
        - 生成问答对: {len(qa_pairs)} 个
        - 本地文件: {local_output_file.name}
        - 训练文件: {JSONL_TRAIN_PATH}
        - 总记录数: {total_records} 条
        </div>
        """
        
        add_qa_log("🎉 所有处理完成!")
        
        return (
            success_msg,
            progress_text,
            gr.Group(visible=True),
            gr.File(value=str(local_output_file)),
            get_qa_logs()
        )
        
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        add_qa_log(f"❌ {error_msg}")
        return (
            f"<div class='error-msg'>{error_msg}</div>",
            f"错误: {error_msg}",
            gr.Group(visible=False),
            gr.File(value=None),
            get_qa_logs()
        )

def main():
    ensure_directories()
    
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .success-msg {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    .error-msg {
        background: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    #terminal-output {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        background-color: #1e1e1e;
        color: #00ff00;
        border: 1px solid #444;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .progress-bar {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .cleanup-btn {
        background: #ffc107;
        border: 1px solid #ffc107;
    }
    .cleanup-btn:hover {
        background: #e0a800;
        border: 1px solid #e0a800;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple"
        ),
        css=custom_css,
        title="高级PDF处理工具 - 全流程AI解决方案"
    ) as demo:
        
        gr.Markdown("# 📄 高级PDF处理工具")
        gr.Markdown("一站式PDF转TXT、问答生成、模型训练与推理解决方案")
        
        with gr.Tabs() as tabs:
            with gr.Tab("🔄 批量PDF转TXT", id=0):
                pdf_tab = create_pdf_converter_tab()
            
            with gr.Tab("🤖 智能问答生成", id=1):
                qa_tab = create_qa_generator_tab()
            
            with gr.Tab("🚀 DeepSeek训练", id=2):
                training_tab = create_deepseek_training_tab()
            
            with gr.Tab("🔮 模型推理", id=3):
                inference_tab = create_inference_tab()
            
            with gr.Tab("📊 任务监控", id=4):
                with gr.Blocks() as monitoring_tab:
                    gr.Markdown("# 📊 任务监控")
                    gr.Markdown("实时监控所有处理任务的状态和进度")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 系统状态")
                            active_tasks = gr.Number(
                                label="活跃任务数",
                                value=0,
                                interactive=False
                            )
                            completed_tasks = gr.Number(
                                label="已完成任务",
                                value=0,
                                interactive=False
                            )
                            
                            refresh_monitoring_btn = gr.Button(
                                "🔄 刷新状态",
                                variant="secondary"
                            )
                            clear_btn = gr.Button(
                                "🧹 清空任务历史",
                                variant="secondary"
                            )
                            clear_output = gr.Textbox(
                                label="操作结果",
                                interactive=False,
                                visible=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### 实时日志")
                            monitoring_log = gr.Textbox(
                                label="系统日志",
                                lines=10,
                                interactive=False,
                                show_copy_button=True
                            )
                    
                    refresh_monitoring_btn.click(
                        fn=get_monitoring_info,
                        outputs=[active_tasks, completed_tasks, monitoring_log]
                    )
                    
                    clear_btn.click(
                        fn=clear_tasks,
                        outputs=clear_output
                    ).then(
                        fn=lambda: (0, 0, "任务历史已清空"),
                        outputs=[active_tasks, completed_tasks, monitoring_log]
                    )
        
        # PDF转换事件绑定
        pdf_tab["inputs"]["convert_btn"].click(
            fn=process_multiple_pdfs,
            inputs=[
                pdf_tab["inputs"]["pdf_input"],
                pdf_tab["inputs"]["use_ai_processing"],
                pdf_tab["inputs"]["max_workers"]
            ],
            outputs=[
                pdf_tab["outputs"]["task_status"],
                pdf_tab["outputs"]["progress_display"],
                pdf_tab["outputs"]["download_section"],
                pdf_tab["outputs"]["txt_download"],
                pdf_tab["outputs"]["terminal_output"]
            ]
        )
        
        pdf_tab["inputs"]["cleanup_btn"].click(
            fn=cleanup_folders_ui,
            outputs=pdf_tab["outputs"]["cleanup_output"]
        )
        
        # 问答生成事件绑定
        qa_tab["inputs"]["generate_btn"].click(
            fn=run_qa_generation,
            inputs=[
                qa_tab["inputs"]["txt_folder"],
                qa_tab["inputs"]["instruction"], 
                qa_tab["inputs"]["language"],
                qa_tab["inputs"]["num_questions"],
                qa_tab["inputs"]["clear_jsonl_checkbox"],
                qa_tab["inputs"]["qa_max_workers"]
            ],
            outputs=[
                qa_tab["outputs"]["processing_status"],
                qa_tab["outputs"]["progress_display"],
                qa_tab["outputs"]["download_section"],
                qa_tab["outputs"]["output_download"],
                qa_tab["outputs"]["terminal_output"]
            ]
        )
        
        qa_tab["inputs"]["check_folder_btn"].click(
            fn=check_txt_folder,
            inputs=qa_tab["inputs"]["txt_folder"],
            outputs=[
                qa_tab["outputs"]["folder_status"],
                qa_tab["outputs"]["progress_display"],
                qa_tab["outputs"]["terminal_output"]
            ]
        )
        
        # 训练事件绑定
        training_tab["inputs"]["start_training_btn"].click(
            fn=start_training_ui,
            inputs=[
                training_tab["inputs"]["data_source_radio"],
                training_tab["inputs"]["json_upload_input"]
            ],
            outputs=[
                training_tab["outputs"]["training_status"],
                training_tab["outputs"]["training_progress"],
                training_tab["outputs"]["model_download_section"],
                training_tab["outputs"]["model_download"]
            ]
        )
        
        # 推理事件绑定
        inference_tab["inputs"]["load_model_btn"].click(
            fn=load_model_handler,
            inputs=inference_tab["inputs"]["model_path"],
            outputs=[
                inference_tab["outputs"]["model_status"],
                inference_tab["outputs"]["inference_info"]
            ]
        )
        
        inference_tab["inputs"]["run_inference_btn"].click(
            fn=run_inference_handler,
            inputs=[
                inference_tab["inputs"]["model_path"],
                inference_tab["inputs"]["question_input"],
                inference_tab["inputs"]["temperature_slider"],
                inference_tab["inputs"]["max_tokens_slider"]
            ],
            outputs=[
                inference_tab["outputs"]["inference_progress"],
                inference_tab["outputs"]["inference_result"],
                inference_tab["outputs"]["inference_info"]
            ]
        )
        
        # 进度控制
        with gr.Row():
            gr.Markdown("### 进度控制")
            manual_refresh_btn = gr.Button("🔄 手动刷新进度", variant="secondary")
            manual_refresh_btn.click(
                fn=update_progress,
                outputs=[
                    pdf_tab["outputs"]["task_status"],
                    pdf_tab["outputs"]["progress_display"],
                    pdf_tab["outputs"]["download_section"],
                    pdf_tab["outputs"]["txt_download"],
                    pdf_tab["outputs"]["terminal_output"]
                ]
            )
            
    return demo

if __name__ == "__main__":
    app = main()
    app.launch(
        server_name="0.0.0.0",
        server_port=7888,
        share=True,
        show_error=True,
        inbrowser=True
    )
