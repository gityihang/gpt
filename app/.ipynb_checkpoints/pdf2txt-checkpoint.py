import json
from openai import OpenAI
import re
import PyPDF2
import os
from pathlib import Path
import shutil
import concurrent.futures
import threading
from queue import Queue
import time

os.environ["DEEPSEEK_API_KEY"] = "sk-b6118335f5c34520abffbe6fa324257a"

# --- 设置 DeepSeek API Key ---
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Please set the environment variable DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

class AdvancedPDFToTXTConverter:
    def __init__(self, pdf_folder="app/pdf", txt_folder="app/txt", max_workers=3):
        self.pdf_folder = Path(pdf_folder)
        self.txt_folder = Path(txt_folder)
        self.pdf_folder.mkdir(exist_ok=True)
        self.txt_folder.mkdir(exist_ok=True)
        self.max_workers = max_workers  # 最大并行处理数
        self.progress_queue = Queue()  # 进度队列
        self.current_tasks = {}  # 当前任务状态
        self.lock = threading.Lock()  # 线程锁
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF提取文本内容
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text_content.append(f"--- Page {page_num + 1} ---")
                    text_content.append(page_text)
                
                return '\n'.join(text_content)
        except Exception as e:
            print(f"读取PDF文件时出错: {e}")
            return ""

    def extract_paper_title(self, pdf_path: str) -> str:
        """
        从PDF中提取论文标题
        """
        try:
            # 首先尝试从文件名获取（去掉.pdf扩展名）
            filename = Path(pdf_path).stem
            filename_title = re.sub(r'[^\w\s-]', '', filename).strip()
            
            # 然后从PDF内容中提取标题
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                return filename_title
            
            # 提取前几页内容寻找标题
            first_page_content = ""
            pages = re.split(r'--- Page \d+ ---', raw_text)
            if pages:
                first_page_content = pages[0] if len(pages) > 0 else ""
                if len(pages) > 1:
                    first_page_content += "\n" + pages[1]  # 前两页内容
            
            # 使用DeepSeek提取标题
            prompt = f"""
请从以下学术论文内容中提取论文的正式标题。只输出标题本身，不要添加任何解释或额外文本。

论文内容片段：
{first_page_content[:3000]}  # 限制内容长度

请识别并输出论文的完整标题。
            """
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的学术论文分析助手，擅长从论文内容中准确提取标题。"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=0.1,
                    max_tokens=200
                )
                
                title = response.choices[0].message.content.strip()
                
                # 清理标题
                title = re.sub(r'^["\'](.*)["\']$', r'\1', title)  # 移除引号
                title = re.sub(r'^标题[:：]\s*', '', title)  # 移除"标题："前缀
                title = re.sub(r'^Title[:：]\s*', '', title, flags=re.IGNORECASE)
                title = title.strip()
                
                # 验证标题质量
                if (len(title) > 10 and len(title) < 200 and 
                    not title.startswith("抱歉") and 
                    not title.startswith("I cannot")):
                    return title
                else:
                    return filename_title
                    
            except Exception as e:
                print(f"使用API提取标题失败: {e}")
                return filename_title
                
        except Exception as e:
            print(f"提取标题时出错: {e}")
            return Path(pdf_path).stem

    def clean_filename(self, title: str) -> str:
        """
        清理文件名，移除非法字符
        """
        # 移除或替换文件名中的非法字符
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', title)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 合并多个空格
        cleaned = cleaned.strip()
        
        # 限制文件名长度
        if len(cleaned) > 100:
            cleaned = cleaned[:100]
        
        return cleaned

    def clean_gpt_output(self, gpt_output: str) -> str:
        """
        清理GPT输出，保留纯文本内容
        """
        gpt_output = gpt_output.strip()
        
        # 移除代码块标记
        gpt_output = re.sub(r"```.*?\n?", "", gpt_output, flags=re.DOTALL)
        gpt_output = re.sub(r"```", "", gpt_output)
        
        # 移除可能的JSON标记
        gpt_output = re.sub(r"^json\s*", "", gpt_output, flags=re.IGNORECASE)
        
        return gpt_output

    def save_uploaded_pdf(self, pdf_file_path: str, task_id: str) -> str:
        """
        保存上传的PDF文件到本地，返回保存路径
        """
        try:
            # 获取原始文件名
            original_filename = Path(pdf_file_path).name
            local_pdf_path = self.pdf_folder / original_filename
            
            # 如果文件已存在，添加时间戳避免冲突
            counter = 1
            original_stem = local_pdf_path.stem
            original_suffix = local_pdf_path.suffix
            
            while local_pdf_path.exists():
                timestamp = str(counter)
                local_pdf_path = self.pdf_folder / f"{original_stem}_{timestamp}{original_suffix}"
                counter += 1
            
            # 复制文件（Gradio已经保存了临时文件）
            shutil.copy2(pdf_file_path, local_pdf_path)
            
            self._update_progress(task_id, f"PDF文件已保存到: {local_pdf_path}")
            return str(local_pdf_path)
            
        except Exception as e:
            self._update_progress(task_id, f"保存PDF文件时出错: {e}")
            return ""

    def _update_progress(self, task_id: str, message: str):
        """更新进度信息"""
        with self.lock:
            if task_id in self.current_tasks:
                self.current_tasks[task_id]['progress'].append(message)
                self.progress_queue.put({
                    'task_id': task_id,
                    'message': message,
                    'type': 'progress'
                })

    def _update_result(self, task_id: str, result: dict):
        """更新任务结果"""
        with self.lock:
            self.current_tasks[task_id]['result'] = result
            self.progress_queue.put({
                'task_id': task_id,
                'result': result,
                'type': 'result'
            })

    def process_pdf_with_deepseek(self, pdf_path: str, output_txt_path: str, task_id: str):
        """
        使用DeepSeek处理PDF内容并保存为格式化的txt文件，特别处理数学公式
        """
        # 提取PDF原始文本
        self._update_progress(task_id, "正在提取PDF文本内容...")
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text:
            self._update_progress(task_id, "无法提取PDF文本内容")
            return False
        
        self._update_progress(task_id, f"原始文本长度: {len(raw_text)} 字符")
        
        # 如果文本太长，进行分段处理
        max_chunk_size = 8000
        text_chunks = []
        
        if len(raw_text) > max_chunk_size:
            pages = re.split(r'--- Page \d+ ---', raw_text)
            current_chunk = ""
            
            for page in pages:
                page = page.strip()
                if not page:
                    continue
                    
                if len(current_chunk) + len(page) <= max_chunk_size:
                    current_chunk += "\n\n" + page if current_chunk else page
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk)
                    current_chunk = page
            
            if current_chunk:
                text_chunks.append(current_chunk)
        else:
            text_chunks = [raw_text]
        
        self._update_progress(task_id, f"将文本分成 {len(text_chunks)} 个部分进行处理")
        
        all_processed_content = []
        
        for i, chunk in enumerate(text_chunks):
            self._update_progress(task_id, f"正在处理第 {i+1}/{len(text_chunks)} 部分...")
            
            prompt = f"""
请处理以下学术论文内容，将其整理成清晰、格式良好的纯文本，并特别注意数学公式的处理。

要求：
1. 移除所有页眉、页脚、页码、图表标题、表格、参考文献等非正文内容
2. 保留论文的核心正文内容：引言、方法、实验、结果、讨论、结论等
3. 确保每个段落都有适当的换行分隔
4. 对于数学公式，请按照以下规则处理：
   - 将识别到的数学公式转换为正确的LaTeX格式
   - 行内公式用 $...$ 包围
   - 独立公式用 $$...$$ 包围
   - 如果原始公式识别有问题，请根据您的知识修正为正确的LaTeX格式
   - 常见的数学符号：α, β, γ, ∑, ∫, ∂, ∞, →, ≤, ≥, ≠ 等
5. 保持原文的技术术语和学术表达
6. 不要添加任何额外的解释或评论，只输出清理后的文本内容

示例：
- 输入："f(x) = x^2 + 2x + 1"
- 输出："函数定义为 $f(x) = x^2 + 2x + 1$"

- 输入："the integral from a to b of f(x) dx"
- 输出："积分表达式为 $\\int_a^b f(x) dx$"

请直接输出处理后的文本，不要用任何标记包围。

需要处理的文本内容：
{chunk}
            """
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的学术文本处理助手，擅长清理和格式化论文内容，特别精通数学公式的LaTeX转换。"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=0.1,
                    max_tokens=4000
                )
                
                processed_chunk = response.choices[0].message.content
                processed_chunk = self.clean_gpt_output(processed_chunk)
                
                if processed_chunk and len(processed_chunk.strip()) > 50:
                    all_processed_content.append(processed_chunk)
                    self._update_progress(task_id, f"第 {i+1} 部分处理完成，长度: {len(processed_chunk)} 字符")
                    
                    latex_formulas = re.findall(r'\$.*?\$', processed_chunk)
                    if latex_formulas:
                        self._update_progress(task_id, f"  发现 {len(latex_formulas)} 个LaTeX公式")
                else:
                    self._update_progress(task_id, f"第 {i+1} 部分处理结果为空或过短")
                    backup_cleaned = self.simple_text_clean_with_formulas(chunk)
                    all_processed_content.append(backup_cleaned)
                    
            except Exception as e:
                self._update_progress(task_id, f"处理第 {i+1} 部分时出错: {e}")
                backup_cleaned = self.simple_text_clean_with_formulas(chunk)
                all_processed_content.append(backup_cleaned)
        
        if not all_processed_content:
            self._update_progress(task_id, "所有部分处理都失败了")
            return False
        
        # 合并所有处理后的内容
        final_content = '\n\n'.join(all_processed_content)
        
        # 最终格式清理
        final_content = self.final_format_clean(final_content)
        
        # 保存到文件
        try:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            self._update_progress(task_id, f"处理完成！结果已保存到: {output_txt_path}")
            self._update_progress(task_id, f"最终文本长度: {len(final_content)} 字符")
            
            total_formulas = len(re.findall(r'\$.*?\$', final_content))
            self._update_progress(task_id, f"总共识别并转换了 {total_formulas} 个数学公式")
            
            return True
            
        except Exception as e:
            self._update_progress(task_id, f"保存文件时出错: {e}")
            return False

    def simple_text_clean_with_formulas(self, text: str) -> str:
        """简单的文本清理备用方案"""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if (re.search(r'^\d+$', line) or
                re.search(r'Page\s*\d+', line, re.IGNORECASE) or
                re.search(r'第\s*\d+\s*页', line) or
                (len(line) < 5 and not re.search(r'[α-ωΑ-Ω∑∫∂∞]', line))):
                continue
                
            line = self.enhance_mathematical_expressions(line)
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'([.!?。！？])\s*', r'\1\n\n', cleaned_text)
        
        return cleaned_text

    def enhance_mathematical_expressions(self, text: str) -> str:
        """增强数学表达式识别"""
        patterns = [
            (r'(\w+)\(([^)]+)\)\s*=\s*([^,.;!?\s]+)', r'$\1(\2) = \3$'),
            (r'sum_\{([^}]+)\}', r'$\sum_{\1}$'),
            (r'int_\{([^}]+)\}', r'$\int_{\1}$'),
            (r'(\d+)/(\d+)', r'$\frac{\1}{\2}$'),
            (r'(\w+)\^(\d+)', r'$\1^{\2}$'),
            (r'\\alpha', r'$\\alpha$'),
            (r'\\beta', r'$\\beta$'),
            (r'\\gamma', r'$\\gamma$'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text

    def final_format_clean(self, text: str) -> str:
        """最终格式清理"""
        if not text:
            return ""
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = self.fix_broken_latex(text)
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            else:
                cleaned_lines.append('')
        
        formatted_text = []
        prev_empty = False
        
        for line in cleaned_lines:
            if line == '':
                if not prev_empty:
                    formatted_text.append(line)
                    prev_empty = True
            else:
                formatted_text.append(line)
                prev_empty = False
        
        return '\n'.join(formatted_text)

    def fix_broken_latex(self, text: str) -> str:
        """修复可能损坏的LaTeX公式"""
        dollar_count = text.count('$')
        if dollar_count % 2 != 0:
            text += '$'
        
        text = re.sub(r'\$\$+', '$$', text)
        return text

    def convert_single_pdf_worker(self, pdf_file_path: str, use_ai_processing: bool, task_id: str):
        """
        单个PDF转换的工作线程
        """
        try:
            self._update_progress(task_id, f"开始处理PDF: {pdf_file_path}")
            
            # 1. 保存PDF到本地
            local_pdf_path = self.save_uploaded_pdf(pdf_file_path, task_id)
            if not local_pdf_path:
                self._update_result(task_id, {
                    "success": False,
                    "error": "保存PDF文件失败"
                })
                return
            
            # 2. 提取论文标题
            self._update_progress(task_id, "正在提取论文标题...")
            paper_title = self.extract_paper_title(local_pdf_path)
            clean_title = self.clean_filename(paper_title)
            
            # 3. 生成输出文件路径
            output_filename = f"{clean_title}.txt"
            output_path = self.txt_folder / output_filename
            
            # 如果文件已存在，添加时间戳
            counter = 1
            original_output_path = output_path
            while output_path.exists():
                output_path = self.txt_folder / f"{clean_title}_{counter}.txt"
                counter += 1
            
            self._update_progress(task_id, f"论文标题: {paper_title}")
            self._update_progress(task_id, f"输出文件: {output_path}")
            
            # 4. 处理PDF
            if use_ai_processing:
                self._update_progress(task_id, "使用AI增强处理...")
                success = self.process_pdf_with_deepseek(local_pdf_path, str(output_path), task_id)
            else:
                self._update_progress(task_id, "使用基础处理...")
                raw_text = self.extract_text_from_pdf(local_pdf_path)
                if raw_text:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(raw_text)
                    success = True
                else:
                    success = False
            
            if success:
                # 读取生成的文件内容用于预览
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 生成预览
                preview = self.generate_preview(content)
                
                # 统计信息
                stats = self.get_file_stats(content)
                
                self._update_result(task_id, {
                    "success": True,
                    "txt_path": str(output_path),
                    "preview": preview,
                    "stats": stats,
                    "filename": output_path.name,
                    "original_name": Path(local_pdf_path).name,
                    "paper_title": paper_title,
                    "formulas_count": len(re.findall(r'\$.*?\$', content)) if use_ai_processing else 0,
                    "task_id": task_id
                })
            else:
                self._update_result(task_id, {
                    "success": False,
                    "error": "PDF处理失败",
                    "task_id": task_id
                })
                
        except Exception as e:
            self._update_progress(task_id, f"处理过程中出错: {e}")
            self._update_result(task_id, {
                "success": False,
                "error": f"处理过程中出错: {str(e)}",
                "task_id": task_id
            })

    def convert_multiple_pdfs(self, pdf_files: list, use_ai_processing: bool = True):
        """
        并行处理多个PDF文件
        """
        task_ids = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = []
            for pdf_file in pdf_files:
                task_id = f"task_{len(self.current_tasks)}_{int(time.time()*1000)}"
                task_ids.append(task_id)
                
                # 初始化任务状态
                self.current_tasks[task_id] = {
                    'progress': [],
                    'result': None,
                    'status': 'processing'
                }
                
                # 提交任务到线程池
                future = executor.submit(
                    self.convert_single_pdf_worker,
                    pdf_file.name,
                    use_ai_processing,
                    task_id
                )
                futures.append((task_id, future))
            
            # 等待所有任务完成
            for task_id, future in futures:
                try:
                    future.result()  # 等待任务完成
                except Exception as e:
                    self._update_progress(task_id, f"任务执行异常: {e}")
        
        return task_ids

    def get_progress(self):
        """获取进度信息"""
        progress_updates = []
        while not self.progress_queue.empty():
            progress_updates.append(self.progress_queue.get())
        return progress_updates

    def get_task_status(self, task_id: str):
        """获取特定任务的状态"""
        with self.lock:
            if task_id in self.current_tasks:
                return self.current_tasks[task_id]
            return None

    def generate_preview(self, text: str, max_lines: int = 15) -> str:
        """生成文本预览"""
        lines = text.split('\n')
        preview_lines = []
        
        for line in lines[:max_lines]:
            if line.strip():
                if len(line) > 100:
                    line = line[:100] + "..."
                preview_lines.append(line)
        
        preview = '\n'.join(preview_lines)
        if len(lines) > max_lines:
            preview += "\n\n... (内容继续)"
        
        return preview

    def get_file_stats(self, text: str) -> dict:
        """获取文件统计信息"""
        if not text:
            return {"pages": 0, "chars": 0, "words": 0, "lines": 0}
        
        pages = len(re.findall(r'--- Page \d+ ---', text))
        clean_text = re.sub(r'--- Page \d+ ---', '', text)
        chars = len(clean_text)
        words = len(re.findall(r'\b\w+\b', clean_text))
        lines = len(clean_text.split('\n'))
        
        return {
            "pages": pages,
            "chars": chars,
            "words": words,
            "lines": lines
        }

# 创建全局转换器实例
converter = AdvancedPDFToTXTConverter(max_workers=3)  # 最多同时处理3个PDF