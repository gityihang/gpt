import re
import PyPDF2
from typing import List, Tuple, Dict

class PDFTextExtractor:
    def __init__(self):
        self.single_column_threshold = 0.6
        
        # 定义过滤模式
        self.filter_patterns = {
            'table': [
                r'\b(表|表格|Table)\s*\d+[.:]?',
                r'\+[-]+\+',
                r'\|.*\|',
                r'\b\d+\s*\|\s*\d+\b',
            ],
            'figure': [
                r'\b(图|图表|图片|Figure|Fig\.?)\s*\d+[.:]?',
                r'\(.*[图表].*\)',
            ],
            'header_footer': [
                r'^\d+$',  # 单独的数字
                r'第\s*\d+\s*页',
                r'Page\s*\d+',
                r'^\s*\d{1,2}[-/]\d{1,2}[-/]\d{4}\s*$',  # 日期
                r'©|©|版权|Copyright',  # 版权信息
                r'Confidential|机密',  # 机密信息
            ],
            'reference': [
                r'参考文献|参考书目|References|Bibliography',
                r'\[\d+\]',
                r'\(.*\d{4}.*\)',
            ],
            'code': [
                r'[{};=<>]+',
                r'\b(def|class|import|function)\b',
            ]
        }
        
        # 页眉页脚模式（更严格的匹配）
        self.header_footer_patterns = [
            # 页眉模式
            r'^.*\d{4}.*$',  # 包含年份的行
            r'^.*[第卷期章节].*$',  # 包含章节信息的行
            r'^[A-Z][A-Z\s]+$',  # 全大写的行
            r'^.*报告.*$',  # 包含报告标题的行
            r'^.*论文.*$',  # 包含论文标题的行
            # 页脚模式
            r'^\d+/\d+$',  # 页码格式 1/10
            r'^- \d+ -$',  # 页码格式 - 1 -
            r'^• \d+ •$',  # 页码格式 • 1 •
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """
        从PDF提取文本并分析版面结构，过滤非正文内容
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                result = {
                    'total_pages': len(pdf_reader.pages),
                    'pages': [],
                    'layout_type': 'unknown'
                }
                
                all_texts = []
                layout_counts = {'single': 0, 'double': 0}
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # 首先移除页眉页脚
                    cleaned_text = self.remove_header_footer(page_text, page_num + 1)
                    
                    # 然后过滤其他非正文内容
                    filtered_text = self.filter_non_content(cleaned_text)
                    
                    # 格式化文本（大标题后换行）
                    formatted_text = self.format_text_structure(filtered_text)
                    
                    # 分析页面布局
                    layout_info = self.analyze_page_layout(formatted_text, page_num + 1)
                    
                    page_result = {
                        'page_number': page_num + 1,
                        'original_text': page_text,
                        'cleaned_text': cleaned_text,
                        'filtered_text': filtered_text,
                        'formatted_text': formatted_text,
                        'layout_type': layout_info['layout_type'],
                        'line_count': layout_info['line_count'],
                        'max_line_length': layout_info['max_line_length']
                    }
                    
                    result['pages'].append(page_result)
                    all_texts.append(formatted_text)
                    
                    # 统计布局类型
                    if layout_info['layout_type'] == 'single_column':
                        layout_counts['single'] += 1
                    else:
                        layout_counts['double'] += 1
                
                # 确定整体布局类型
                if layout_counts['single'] > layout_counts['double']:
                    result['layout_type'] = 'mainly_single_column'
                elif layout_counts['double'] > layout_counts['single']:
                    result['layout_type'] = 'mainly_double_column'
                else:
                    result['layout_type'] = 'mixed'
                
                result['full_text'] = '\n'.join(all_texts)
                return result
                
        except Exception as e:
            print(f"读取PDF文件时出错: {e}")
            return None
    
    def remove_header_footer(self, text: str, page_num: int) -> str:
        """
        专门移除页眉和页脚
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        if len(lines) <= 2:
            return text
        
        # 识别并移除页眉（通常在前几行）
        header_lines = 0
        for i in range(min(3, len(lines))):  # 检查前3行
            line = lines[i].strip()
            if self.is_header_footer_line(line, page_num, 'header'):
                header_lines = i + 1
        
        # 识别并移除页脚（通常在后几行）
        footer_lines = 0
        for i in range(min(3, len(lines))):  # 检查后3行
            line = lines[-(i+1)].strip()
            if self.is_header_footer_line(line, page_num, 'footer'):
                footer_lines = i + 1
        
        # 移除页眉页脚
        if header_lines > 0 or footer_lines > 0:
            start_idx = header_lines
            end_idx = len(lines) - footer_lines if footer_lines > 0 else len(lines)
            cleaned_lines = lines[start_idx:end_idx]
            return '\n'.join(cleaned_lines)
        
        return text
    
    def is_header_footer_line(self, line: str, page_num: int, position: str) -> bool:
        """
        判断一行是否是页眉或页脚
        """
        if not line or len(line.strip()) < 2:
            return False
        
        line_lower = line.lower()
        
        # 通用页眉页脚特征
        if position == 'header':
            # 页眉特征
            if (re.search(r'\d{4}', line) and len(line) < 50):  # 包含年份且较短
                return True
            if re.search(r'^(第[一二三四五六七八九十\d]+[章节条])', line):  # 章节标题
                return True
            if line.isupper() and len(line) < 100:  # 全大写且不长
                return True
            if re.search(r'(报告|论文|摘要|目录|致谢)', line):  # 常见页眉内容
                return True
        
        # 页脚特征
        if position == 'footer':
            # 页码相关模式
            if re.search(r'^\d+$', line):  # 纯数字
                return True
            if re.search(r'^[-\s•]*\d+[-\s•]*$', line):  # 带装饰的页码
                return True
            if re.search(r'\d+\s*/\s*\d+', line):  # 1/10 格式
                return True
            if re.search(r'(page|页码|第.*页)', line_lower):  # 包含页码文字
                return True
        
        # 检查预定义模式
        for pattern in self.header_footer_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def filter_non_content(self, text: str) -> str:
        """
        使用正则表达式过滤掉表格、图表等非正文内容
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 跳过过短的行
            if len(line) < 5:
                continue
                
            # 检查是否匹配过滤模式
            should_skip = False
            for category, patterns in self.filter_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        should_skip = True
                        break
                if should_skip:
                    break
            
            # 跳过包含大量特殊字符的行
            if not should_skip:
                special_chars = re.findall(r'[|+=\-*/><{}]', line)
                if len(special_chars) / len(line) > 0.3:
                    should_skip = True
            
            # 跳过URL或文件路径
            if not should_skip and (re.search(r'http[s]?://', line) or re.search(r'[a-zA-Z]:\\', line)):
                should_skip = True
            
            if not should_skip:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def format_text_structure(self, text: str) -> str:
        """
        格式化文本结构，大标题后换行
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检测大标题（通常较短、有特定格式）
            if self.is_large_title(line):
                formatted_lines.append("")  # 添加空行
                formatted_lines.append(line)
                formatted_lines.append("")  # 标题后换行
            else:
                formatted_lines.append(line)
        
        # 合并连续空行
        result = []
        prev_empty = False
        for line in formatted_lines:
            if line == "":
                if not prev_empty:
                    result.append(line)
                    prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        
        return '\n'.join(result)
    
    def is_large_title(self, line: str) -> bool:
        """
        判断一行是否是大标题
        """
        if not line or len(line) > 100:  # 太长的不是标题
            return False
        
        # 标题特征
        title_indicators = [
            r'^第[一二三四五六七八九十\d]+[章节条]',  # 第1章, 第一节
            r'^[一二三四五六七八九十]、',  # 一、标题
            r'^\d+\.\s',  # 1. 标题
            r'^[A-Z][A-Z\s]{5,}',  # 全大写的标题
            r'^(摘要|目录|引言|前言|结论|参考文献|致谢)',  # 常见章节标题
            r'^[A-Z][a-zA-Z\s]{10,50}$',  # 英文标题格式
        ]
        
        for pattern in title_indicators:
            if re.search(pattern, line):
                return True
        
        # 检查字体大小特征（通过字符密度粗略判断）
        if len(line) < 50 and not re.search(r'[.,;!?]', line):
            # 没有标点符号的短行可能是标题
            return True
        
        return False
    
    def analyze_page_layout(self, text: str, page_num: int) -> Dict:
        """
        分析单页的布局类型（单栏/双栏）
        """
        lines = text.split('\n')
        if not lines:
            return {'layout_type': 'unknown', 'line_count': 0, 'max_line_length': 0}
        
        valid_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        if not valid_lines:
            return {'layout_type': 'unknown', 'line_count': 0, 'max_line_length': 0}
        
        line_lengths = [len(line) for line in valid_lines]
        avg_length = sum(line_lengths) / len(line_lengths)
        max_length = max(line_lengths)
        
        long_lines = [length for length in line_lengths if length > 50]
        long_line_ratio = len(long_lines) / len(valid_lines)
        
        space_pattern = r'\s{10,}'
        large_spaces = re.findall(space_pattern, text)
        
        length_variance = sum((length - avg_length) ** 2 for length in line_lengths) / len(line_lengths)
        
        layout_type = self.determine_layout_type(
            long_line_ratio, 
            len(large_spaces), 
            avg_length, 
            max_length,
            length_variance
        )
        
        return {
            'layout_type': layout_type,
            'line_count': len(valid_lines),
            'max_line_length': max_length,
            'avg_line_length': avg_length,
            'long_line_ratio': long_line_ratio,
            'large_space_count': len(large_spaces)
        }
    
    def determine_layout_type(self, long_line_ratio: float, large_space_count: int, 
                            avg_length: float, max_length: float, variance: float) -> str:
        """
        根据特征判断布局类型
        """
        single_column_score = 0
        double_column_score = 0
        
        if long_line_ratio > 0.7:
            single_column_score += 2
        elif long_line_ratio < 0.3:
            double_column_score += 2
        
        if avg_length > 60:
            single_column_score += 1
        elif avg_length < 40:
            double_column_score += 1
        
        if large_space_count > 3:
            double_column_score += 1
        else:
            single_column_score += 1
        
        if variance > 400:
            double_column_score += 1
        else:
            single_column_score += 1
        
        if single_column_score > double_column_score:
            return 'single_column'
        elif double_column_score > single_column_score:
            return 'double_column'
        else:
            return 'unknown'
    
    def extract_text_by_layout(self, pdf_path: str) -> Dict:
        """
        按布局类型分别提取文本
        """
        extraction_result = self.extract_text_from_pdf(pdf_path)
        if not extraction_result:
            return None
        
        single_column_texts = []
        double_column_texts = []
        
        for page in extraction_result['pages']:
            if page['layout_type'] == 'single_column':
                single_column_texts.append({
                    'page': page['page_number'],
                    'text': page['formatted_text']  # 使用格式化后的文本
                })
            elif page['layout_type'] == 'double_column':
                double_column_texts.append({
                    'page': page['page_number'],
                    'text': page['formatted_text']  # 使用格式化后的文本
                })
        
        cleaned_single = self.final_clean_text([item['text'] for item in single_column_texts])
        cleaned_double = self.final_clean_text([item['text'] for item in double_column_texts])
        
        return {
            'overall_layout': extraction_result['layout_type'],
            'single_column_pages': {
                'count': len(single_column_texts),
                'pages': [item['page'] for item in single_column_texts],
                'text': cleaned_single
            },
            'double_column_pages': {
                'count': len(double_column_texts),
                'pages': [item['page'] for item in double_column_texts],
                'text': cleaned_double
            },
            'total_pages': extraction_result['total_pages']
        }
    
    def final_clean_text(self, text_list: List[str]) -> str:
        """
        最终文本清理
        """
        full_text = '\n'.join(text_list)
        
        # 移除过多的空白字符
        full_text = re.sub(r'\s+', ' ', full_text)
        
        # 修复断词
        full_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', full_text)
        
        # 规范化段落
        full_text = re.sub(r'\.\s+([A-Z])', r'.\n\n\1', full_text)
        
        # 移除残留的特殊格式
        full_text = re.sub(r'\d+\.\d+\.\d+', '', full_text)
        full_text = re.sub(r'@\w+', '', full_text)
        
        return full_text.strip()
    
    def print_analysis_report(self, pdf_path: str):
        """
        打印分析报告
        """
        result = self.extract_text_by_layout(pdf_path)
        if not result:
            print("分析失败")
            return
        
        print("=" * 60)
        print("PDF文本提取报告（已移除页眉页脚，格式化标题）")
        print("=" * 60)
        print(f"总页数: {result['total_pages']}")
        print(f"整体布局类型: {result['overall_layout']}")
        print(f"单栏页数: {result['single_column_pages']['count']}")
        print(f"双栏页数: {result['double_column_pages']['count']}")
        print(f"单栏页码: {result['single_column_pages']['pages']}")
        print(f"双栏页码: {result['double_column_pages']['pages']}")
        
        single_text = result['single_column_pages']['text']
        double_text = result['double_column_pages']['text']
        
        print("\n单栏正文文本预览:")
        print("-" * 40)
        if single_text:
            print(single_text[:500] + "..." if len(single_text) > 500 else single_text)
        else:
            print("无单栏正文内容")
            
        print("\n双栏正文文本预览:")
        print("-" * 40)
        if double_text:
            print(double_text[:500] + "..." if len(double_text) > 500 else double_text)
        else:
            print("无双栏正文内容")

# 使用示例
if __name__ == "__main__":
    extractor = PDFTextExtractor()
    
    pdf_file = "test.pdf"  # 替换为您的PDF文件路径
    
    extractor.print_analysis_report(pdf_file)
    
    detailed_result = extractor.extract_text_by_layout(pdf_file)
    
    if detailed_result:
        with open("formatted_single_column.txt", "w", encoding="utf-8") as f:
            f.write(detailed_result['single_column_pages']['text'])
        
        with open("formatted_double_column.txt", "w", encoding="utf-8") as f:
            f.write(detailed_result['double_column_pages']['text'])
        
        print("\n格式化后的文本已保存到文件:")
        print("- formatted_single_column.txt")
        print("- formatted_double_column.txt")