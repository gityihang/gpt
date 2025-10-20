import json
from openai import OpenAI
import re
import PyPDF2
import os
from pathlib import Path

os.environ["DEEPSEEK_API_KEY"] = "sk-b6118335f5c34520abffbe6fa324257a"  # my key

# --- 设置 DeepSeek API Key ---
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Please set the environment variable DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def extract_text_from_pdf(pdf_path: str) -> str:
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

def extract_paper_title(pdf_path: str) -> str:
    """
    从PDF中提取论文标题
    """
    try:
        # 首先尝试从文件名获取（去掉.pdf扩展名）
        filename = Path(pdf_path).stem
        filename_title = re.sub(r'[^\w\s-]', '', filename).strip()
        
        # 然后从PDF内容中提取标题
        raw_text = extract_text_from_pdf(pdf_path)
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

def clean_filename(title: str) -> str:
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

def clean_gpt_output(gpt_output: str) -> str:
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

def process_pdf_with_deepseek(pdf_path: str, output_txt_path: str):
    """
    使用DeepSeek处理PDF内容并保存为格式化的txt文件，特别处理数学公式
    """
    # 提取PDF原始文本
    print("正在提取PDF文本内容...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        print("无法提取PDF文本内容")
        return False
    
    print(f"原始文本长度: {len(raw_text)} 字符")
    
    # 如果文本太长，进行分段处理
    max_chunk_size = 8000  # 减小chunk大小以处理公式
    text_chunks = []
    
    if len(raw_text) > max_chunk_size:
        # 按页面分割文本
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
    
    print(f"将文本分成 {len(text_chunks)} 个部分进行处理")
    
    all_processed_content = []
    
    for i, chunk in enumerate(text_chunks):
        print(f"正在处理第 {i+1}/{len(text_chunks)} 部分...")
        
        # --- DeepSeek Prompt - 特别添加公式处理要求 ---
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
            # --- 调用 DeepSeek API ---
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的学术文本处理助手，擅长清理和格式化论文内容，特别精通数学公式的LaTeX转换。"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,  # 低温度以确保稳定性
                max_tokens=4000   # 增加token限制以处理公式
            )
            
            processed_chunk = response.choices[0].message.content
            processed_chunk = clean_gpt_output(processed_chunk)
            
            if processed_chunk and len(processed_chunk.strip()) > 50:  # 确保有实际内容
                all_processed_content.append(processed_chunk)
                print(f"第 {i+1} 部分处理完成，长度: {len(processed_chunk)} 字符")
                
                # 检查是否包含LaTeX公式
                latex_formulas = re.findall(r'\$.*?\$', processed_chunk)
                if latex_formulas:
                    print(f"  发现 {len(latex_formulas)} 个LaTeX公式")
            else:
                print(f"第 {i+1} 部分处理结果为空或过短")
                # 使用基础清理但保留可能的公式
                backup_cleaned = simple_text_clean_with_formulas(chunk)
                all_processed_content.append(backup_cleaned)
                
        except Exception as e:
            print(f"处理第 {i+1} 部分时出错: {e}")
            # 如果API调用失败，使用备用方案：简单的文本清理但尝试保留公式
            backup_cleaned = simple_text_clean_with_formulas(chunk)
            all_processed_content.append(backup_cleaned)
    
    if not all_processed_content:
        print("所有部分处理都失败了")
        return False
    
    # 合并所有处理后的内容
    final_content = '\n\n'.join(all_processed_content)
    
    # 最终格式清理
    final_content = final_format_clean(final_content)
    
    # 保存到文件
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"处理完成！结果已保存到: {output_txt_path}")
        print(f"最终文本长度: {len(final_content)} 字符")
        
        # 统计公式数量
        total_formulas = len(re.findall(r'\$.*?\$', final_content))
        print(f"总共识别并转换了 {total_formulas} 个数学公式")
        
        return True
        
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

def simple_text_clean_with_formulas(text: str) -> str:
    """
    简单的文本清理备用方案，但尝试识别和保留数学表达式
    """
    if not text:
        return ""
    
    # 移除明显的页眉页脚
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 跳过明显的页码和页眉
        if (re.search(r'^\d+$', line) or  # 纯数字
            re.search(r'Page\s*\d+', line, re.IGNORECASE) or  # Page X
            re.search(r'第\s*\d+\s*页', line) or  # 第X页
            (len(line) < 5 and not re.search(r'[α-ωΑ-Ω∑∫∂∞]', line))):  # 过短的行，除非包含数学符号
            continue
            
        # 尝试识别简单的数学表达式并添加LaTeX标记
        line = enhance_mathematical_expressions(line)
        cleaned_lines.append(line)
    
    # 合并为段落
    cleaned_text = '\n'.join(cleaned_lines)
    
    # 基本的段落格式化
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # 多个换行合并为两个
    cleaned_text = re.sub(r'([.!?。！？])\s*', r'\1\n\n', cleaned_text)  # 句子后换行
    
    return cleaned_text

def enhance_mathematical_expressions(text: str) -> str:
    """
    增强数学表达式识别，添加基本的LaTeX标记
    """
    # 常见的数学模式
    patterns = [
        # 简单等式: f(x) = x^2 + 1
        (r'(\w+)\(([^)]+)\)\s*=\s*([^,.;!?\s]+)', r'$\1(\2) = \3$'),
        # 求和符号: sum from i=1 to n
        (r'sum_\{([^}]+)\}', r'$\sum_{\1}$'),
        # 积分: integral from a to b
        (r'int_\{([^}]+)\}', r'$\int_{\1}$'),
        # 分数: a/b
        (r'(\d+)/(\d+)', r'$\frac{\1}{\2}$'),
        # 上标: x^2
        (r'(\w+)\^(\d+)', r'$\1^{\2}$'),
        # 希腊字母
        (r'\\alpha', r'$\\alpha$'),
        (r'\\beta', r'$\\beta$'),
        (r'\\gamma', r'$\\gamma$'),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text

def final_format_clean(text: str) -> str:
    """
    最终格式清理，特别注意LaTeX公式的完整性
    """
    if not text:
        return ""
    
    # 移除多余的空行，但要确保公式完整性
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 修复可能损坏的LaTeX公式
    text = fix_broken_latex(text)
    
    # 确保段落开头没有空格
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
        else:
            cleaned_lines.append('')  # 保留空行
    
    # 重新组合，确保段落间有适当的空行
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

def fix_broken_latex(text: str) -> str:
    """
    修复可能损坏的LaTeX公式
    """
    # 修复未闭合的$符号
    dollar_count = text.count('$')
    if dollar_count % 2 != 0:
        # 如果$符号数量为奇数，在末尾添加一个$
        text += '$'
    
    # 修复连续的$符号
    text = re.sub(r'\$\$+', '$$', text)
    
    return text

def batch_process_pdfs(pdf_folder: str = "pdf", txt_folder: str = "txt"):
    """
    批量处理PDF文件夹中的所有PDF文件
    """
    # 创建输出文件夹
    Path(txt_folder).mkdir(exist_ok=True)
    
    # 获取所有PDF文件
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"在 '{pdf_folder}' 文件夹中没有找到PDF文件")
        return
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    processed_count = 0
    failed_count = 0
    
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"开始处理: {pdf_path.name}")
        print(f"{'='*60}")
        
        # 提取论文标题作为文件名
        print("正在提取论文标题...")
        paper_title = extract_paper_title(str(pdf_path))
        clean_title = clean_filename(paper_title)
        
        # 生成输出文件路径
        output_filename = f"{clean_title}.txt"
        output_path = Path(txt_folder) / output_filename
        
        # 如果文件已存在，添加序号
        counter = 1
        original_output_path = output_path
        while output_path.exists():
            output_path = original_output_path.parent / f"{clean_title}_{counter}.txt"
            counter += 1
        
        print(f"论文标题: {paper_title}")
        print(f"输出文件: {output_path}")
        
        # 处理PDF
        success = process_pdf_with_deepseek(str(pdf_path), str(output_path))
        
        if success:
            processed_count += 1
            print(f"✓ 成功处理: {pdf_path.name}")
        else:
            failed_count += 1
            print(f"✗ 处理失败: {pdf_path.name}")
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！")
    print(f"成功: {processed_count} 个文件")
    print(f"失败: {failed_count} 个文件")
    print(f"输出文件夹: {txt_folder}")
    print(f"{'='*60}")

# --- 主程序 ---
if __name__ == "__main__":
    # 设置文件夹路径
    pdf_folder = "app/pdf"  # PDF文件所在文件夹
    txt_folder = "app/txt"  # 输出文本文件文件夹
    
    if not os.path.exists(pdf_folder):
        print(f"PDF文件夹 '{pdf_folder}' 不存在，正在创建...")
        os.makedirs(pdf_folder)
        print(f"请将PDF文件放入 '{pdf_folder}' 文件夹中，然后重新运行程序")
    else:
        print(f"开始批量处理PDF文件夹: {pdf_folder}")
        print("特别注意：将自动提取论文标题作为文件名，并识别转换数学公式为LaTeX格式")
        batch_process_pdfs(pdf_folder, txt_folder)