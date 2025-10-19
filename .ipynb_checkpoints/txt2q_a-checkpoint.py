import json
from openai import OpenAI
import re
import os
import glob
from typing import List, Dict
import time
from collections import Counter

os.environ["DEEPSEEK_API_KEY"] = "sk-b6118335f5c34520abffbe6fa324257a"

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Please set the environment variable DEEPSEEK_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def read_multiple_txt_files(txt_folder: str = "txt_files") -> Dict[str, str]:
    """
    读取多个txt文件，返回文件名和内容的字典
    """
    txt_files = glob.glob(os.path.join(txt_folder, "*.txt"))
    contents = {}
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                file_name = os.path.basename(txt_file)
                contents[file_name] = content
                print(f"✅ 已读取: {file_name} (长度: {len(content)} 字符)")
        except Exception as e:
            print(f"❌ 读取文件 {txt_file} 时出错: {e}")
    
    return contents

def debug_gpt_output(gpt_output: str) -> str:
    """
    调试GPT输出，详细分析问题
    """
    print(f"=== DEBUG INFO ===")
    print(f"原始输出长度: {len(gpt_output)}")
    print(f"前200字符: {gpt_output[:200]}")
    
    # 检查常见问题
    if "error" in gpt_output.lower():
        print("⚠️ 包含错误信息")
    if "rate limit" in gpt_output.lower():
        print("⚠️ 可能达到频率限制")
    if "```" in gpt_output:
        print("⚠️ 包含代码块标记")
    
    return gpt_output

def robust_clean_gpt_output(gpt_output: str) -> str:
    """
    更健壮的GPT输出清理
    """
    if not gpt_output:
        print("❌ 输出为空")
        return ""
    
    original_output = gpt_output
    gpt_output = gpt_output.strip()
    
    print(f"清理前长度: {len(gpt_output)}")
    
    # 多层清理
    cleaning_steps = [
        # 移除代码块
        (r'```json\s*', ''),
        (r'```\s*', ''),
        (r'\s*```', ''),
        # 移除额外的空白
        (r'\n+', ' '),
        (r'\s+', ' '),
    ]
    
    for pattern, replacement in cleaning_steps:
        gpt_output = re.sub(pattern, replacement, gpt_output, flags=re.DOTALL | re.IGNORECASE)
    
    gpt_output = gpt_output.strip()
    print(f"清理后长度: {len(gpt_output)}")
    
    return gpt_output

def extract_qa_pairs_from_response(parsed_data) -> list:
    """
    从API响应中提取问答对，处理多种可能的格式
    """
    qa_pairs = []
    
    if isinstance(parsed_data, list):
        # 格式1: [{"question": "...", "answer": "..."}]
        for item in parsed_data:
            if isinstance(item, dict):
                if "question" in item and "answer" in item:
                    qa_pairs.append({
                        "question": item["question"],
                        "answer": item["answer"]
                    })
                elif "question_en" in item and "answer_en" in item:
                    qa_pairs.append({
                        "question_en": item["question_en"],
                        "answer_en": item["answer_en"],
                        "question_zh": item.get("question_zh", ""),
                        "answer_zh": item.get("answer_zh", "")
                    })
    
    elif isinstance(parsed_data, dict):
        # 格式2: {"questions": [...], "answers": [...]}
        if "questions" in parsed_data and "answers" in parsed_data:
            questions = parsed_data["questions"]
            answers = parsed_data["answers"]
            if isinstance(questions, list) and isinstance(answers, list):
                for i, (q, a) in enumerate(zip(questions, answers)):
                    if isinstance(q, dict) and "question" in q:
                        q_text = q["question"]
                        a_text = a["answer"] if isinstance(a, dict) and "answer" in a else a
                    else:
                        q_text = str(q)
                        a_text = str(a)
                    qa_pairs.append({"question": q_text, "answer": a_text})
        
        # 格式3: {"qa_pairs": [{"q": "...", "a": "..."}]}
        elif "qa_pairs" in parsed_data:
            for item in parsed_data["qa_pairs"]:
                if isinstance(item, dict):
                    question = item.get("q") or item.get("question")
                    answer = item.get("a") or item.get("answer")
                    if question and answer:
                        qa_pairs.append({"question": question, "answer": answer})
        
        # 格式4: 直接包含question和answer的字典
        elif "question" in parsed_data and "answer" in parsed_data:
            qa_pairs.append({
                "question": parsed_data["question"],
                "answer": parsed_data["answer"]
            })
    
    return qa_pairs

def _get_system_message(language: str) -> str:
    """获取系统消息 - 更严格的指令"""
    if language == "chinese":
        return """你是一位专业的研究人员。请严格按以下JSON格式输出问答对，不要包含任何额外文本：
[{"question": "问题1", "answer": "答案1"}, {"question": "问题2", "answer": "答案2"}]
必须返回JSON数组，每个元素包含question和answer字段。"""
    elif language == "english":
        return """You are a professional researcher. Please output Q&A pairs in strict JSON format only:
[{"question": "question1", "answer": "answer1"}, {"question": "question2", "answer": "answer2"}]
Return ONLY a JSON array with question and answer fields."""
    else:
        return """You are a bilingual research expert. Output in strict JSON format only:
[{"question_en": "q1", "answer_en": "a1", "question_zh": "q1中文", "answer_zh": "a1中文"}]
Return ONLY a JSON array."""

def _call_deepseek_api_with_retry(prompt: str, language: str, max_retries: int = 3) -> list:
    """
    带重试机制的API调用 - 修复版
    """
    for attempt in range(max_retries):
        try:
            print(f"🔄 API调用尝试 {attempt + 1}/{max_retries}")
            
            system_message = _get_system_message(language)
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"}  # 强制JSON格式
            )

            gpt_output = response.choices[0].message.content
            
            # 调试输出
            debug_gpt_output(gpt_output)
            
            # 清理输出
            cleaned_output = robust_clean_gpt_output(gpt_output)
            
            try:
                parsed_data = json.loads(cleaned_output)
                print(f"✅ JSON解析成功，类型: {type(parsed_data)}")
                
                # 提取问答对
                qa_pairs = extract_qa_pairs_from_response(parsed_data)
                
                if qa_pairs:
                    print(f"✅ 成功提取 {len(qa_pairs)} 个问答对")
                    return qa_pairs
                else:
                    print(f"❌ 未提取到有效问答对")
                    print(f"解析的数据结构: {parsed_data}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return []
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"清理后的内容: {cleaned_output[:500]}...")
                
                if attempt < max_retries - 1:
                    print("等待后重试...")
                    time.sleep(2)
                    continue
                else:
                    return []
                    
        except Exception as e:
            print(f"❌ API调用异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            else:
                return []
    
    return []

def extract_paper_keywords(content: str, file_name: str, max_keywords: int = 8) -> List[str]:
    """
    从论文内容和标题中提取有意义的学术关键词
    """
    # 从文件名提取有意义的部分（去除扩展名和常见无意义词）
    file_base = os.path.splitext(file_name)[0]  # 去除.txt扩展名
    file_words = re.findall(r'[a-zA-Z\u4e00-\u9fff]{2,}', file_base)
    
    # 从内容提取高频词，但进行更严格的过滤
    content_words = re.findall(r'[a-zA-Z\u4e00-\u9fff]{3,}', content.lower()[:5000])  # 只分析前5000字符
    
    # 扩展停用词列表
    stop_words = {
        # 英文停用词
        'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has', 'were', 'are', 'was', 'been', 'being',
        'which', 'what', 'when', 'where', 'why', 'how', 'who', 'whom', 'whose', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'cannot', 'able', 'about', 'above', 'after', 'again', 'against', 'all', 'any',
        'because', 'before', 'below', 'between', 'both', 'but', 'by', 'during', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'than', 'then', 'there', 'these', 'they', 'those', 'through', 'until', 'very', 'while', 'within',
        'without', 'based', 'using', 'study', 'research', 'paper', 'article', 'analysis', 'method', 'result', 'conclusion',
        'introduction', 'abstract', 'background', 'objective', 'purpose', 'aim', 'goal', 'find', 'found', 'show', 'showed',
        'demonstrate', 'demonstrated', 'indicate', 'indicated', 'suggest', 'suggested', 'reveal', 'revealed', 'provide',
        'provided', 'present', 'presented', 'discuss', 'discussed', 'explain', 'explained', 'describe', 'described',
        'examine', 'examined', 'investigate', 'investigated', 'explore', 'explored', 'assess', 'assessed', 'evaluate',
        'evaluated', 'measure', 'measured', 'test', 'tested', 'model', 'models', 'data', 'dataset', 'sample', 'samples',
        
        # 中文停用词
        '的', '在', '是', '了', '和', '与', '及', '等', '其中', '通过', '基于', '使用', '采用', '进行', '具有', '包括',
        '包含', '涉及', '关于', '对于', '因此', '所以', '然而', '但是', '虽然', '尽管', '如果', '那么', '因为', '所以',
        '本文', '本研究', '我们', '作者', '论文', '文章', '研究', '分析', '方法', '结果', '结论', '引言', '摘要', '背景',
        '目的', '目标', '发现', '表明', '证明', '显示', '揭示', '提供', '提出', '讨论', '解释', '描述', '考察', '调查',
        '探讨', '评估', '测量', '测试', '模型', '数据', '样本', '资料'
    }
    
    # 过滤停用词和短词
    meaningful_words = [
        word for word in content_words 
        if (word not in stop_words and 
            len(word) > 2 and 
            not word.isdigit() and  # 排除纯数字
            not re.match(r'^[0-9\.]+$', word))  # 排除数字和点组成的字符串
    ]
    
    # 统计词频，但给予标题词更高权重
    word_freq = Counter(meaningful_words)
    
    # 给文件名中的词增加权重
    for file_word in file_words:
        if file_word.lower() not in stop_words and len(file_word) > 2:
            word_freq[file_word.lower()] += 5  # 给标题词更高权重
    
    # 选择最有意义的关键词
    top_keywords = []
    for word, freq in word_freq.most_common(20):  # 先取前20个
        # 进一步过滤：排除太常见的学术词汇
        common_academic_words = {
            'analysis', 'method', 'results', 'study', 'research', 'model', 'data', 'effect', 'impact',
            'analysis', 'method', 'results', 'study', 'research', 'model', 'data', 'effect', 'impact',
            'analysis', 'method', 'results', 'study', 'research', 'model', 'data', 'effect', 'impact'
        }
        if word not in common_academic_words:
            top_keywords.append(word)
    
    return top_keywords[:max_keywords]

def analyze_research_domains(contents: Dict[str, str]) -> Dict:
    """
    综合分析所有文件的研究领域和主题
    """
    print("🔍 正在分析研究领域和主题...")
    
    all_domains = []
    all_themes = []
    all_keywords = []
    
    for file_name, content in contents.items():
        content_lower = content.lower()
        
        # 领域映射 - 更精确的匹配
        domain_mapping = {
            '经济学': ['economic', 'economy', 'gdp', 'market', 'financial', 'investment', 'price', 'cost', 'income', 'revenue', 'profit', 'trade'],
            '气候科学': ['climate', 'temperature', 'weather', 'precipitation', 'emission', 'carbon', 'warming', 'greenhouse', 'atmospheric'],
            '农业': ['agriculture', 'crop', 'farm', 'food', 'yield', 'harvest', 'rural', 'farming', 'irrigation', 'fertilizer'],
            '环境科学': ['environment', 'pollution', 'sustainability', 'ecology', 'conservation', 'ecosystem', 'biodiversity', 'environmental'],
            '风险管理': ['risk', 'management', 'mitigation', 'uncertainty', 'vulnerability', 'resilience', 'exposure', 'hazard'],
            '政策分析': ['policy', 'regulation', 'governance', 'intervention', 'strategy', 'measure', 'implementation', 'enforcement']
        }
        
        # 主题映射 - 更精确的匹配
        theme_mapping = {
            '影响评估': ['impact', 'effect', 'evaluation', 'assessment', 'consequence', 'outcome', 'result'],
            '机制分析': ['mechanism', 'pathway', 'channel', 'transmission', 'causal', 'causality', 'mediation'],
            '实证研究': ['empirical', 'evidence', 'data analysis', 'statistical', 'regression', 'estimation', 'empirically'],
            '政策建议': ['policy', 'recommendation', 'suggestion', 'implication', 'application', 'recommend', 'suggest'],
            '模型构建': ['model', 'framework', 'theoretical', 'conceptual', 'simulation', 'modeling', 'theoretical']
        }
        
        # 识别领域 - 要求至少匹配2个关键词
        for domain, keywords in domain_mapping.items():
            match_count = sum(1 for keyword in keywords if keyword in content_lower)
            if match_count >= 2:  # 至少匹配2个关键词才认为是该领域
                all_domains.append(domain)
        
        # 识别主题 - 要求至少匹配2个关键词
        for theme, keywords in theme_mapping.items():
            match_count = sum(1 for keyword in keywords if keyword in content_lower)
            if match_count >= 2:
                all_themes.append(theme)
        
        # 收集关键词
        keywords = extract_paper_keywords(content, file_name)
        all_keywords.extend(keywords)
    
    # 如果没有识别到领域，使用默认值
    if not all_domains:
        all_domains = ['学术研究']
    
    # 确定主要领域和主题
    domain_counter = Counter(all_domains)
    theme_counter = Counter(all_themes)
    keyword_counter = Counter(all_keywords)
    
    primary_domains = [domain for domain, count in domain_counter.most_common(2)]
    primary_themes = [theme for theme, count in theme_counter.most_common(3)] if theme_counter else ['综合研究']
    top_keywords = [keyword for keyword, count in keyword_counter.most_common(10)]
    
    return {
        'primary_domains': primary_domains,
        'primary_themes': primary_themes,
        'top_keywords': top_keywords,
        'file_count': len(contents)
    }
def generate_single_domain_prompt(content: str, file_name: str, num_questions: int, language: str, domain_analysis: Dict) -> str:
    """
    为单个文件生成领域深度问题的提示词 - 修复版
    """
    domains_text = '、'.join(domain_analysis['primary_domains'])
    themes_text = '、'.join(domain_analysis['primary_themes'])
    
    if language == "chinese":
        prompt_content = f"""
请基于以下{domains_text}领域的研究内容，生成{num_questions}个具有学术深度的专业问题。

研究领域：{domains_text}
核心主题：{themes_text}
关键词：{', '.join(domain_analysis['top_keywords'][:5])}

研究内容摘要：
{content[:4000]}

请生成涵盖以下深度学术维度的问题：
1. 理论机制探讨：分析核心理论框架和因果机制
2. 方法论批判：评价研究方法的科学性和创新性
3. 实证发现解读：深入解读数据发现的理论意义
4. 政策实践关联：探讨研究成果的政策含义和实践价值
5. 学科前沿定位：分析该研究在领域发展中的位置
6. 未来研究方向：提出有见地的后续研究建议

重要要求：
- 使用专业学术语言，避免提及具体作者姓名
- 避免使用"本文"、"本研究"、"在XXX的研究中"等表述
- 问题要体现{domains_text}领域的专业深度和普适性
- 强调理论分析、机制探讨和批判性思考
- 答案应展示严谨的学术推理过程，不依赖特定研究
- 避免简单的事实陈述，注重解释和分析

请严格按照以下JSON格式输出：
[{{"question": "问题1", "answer": "答案1"}}, {{"question": "问题2", "answer": "答案2"}}]

生成{num_questions}个具有普适性的深度学术问答对：
"""
    elif language == "english":
        prompt_content = f"""
Please generate {num_questions} academically profound questions based on the following research content in the {domains_text} field.

Research Domains: {domains_text}
Core Themes: {themes_text}
Keywords: {', '.join(domain_analysis['top_keywords'][:5])}

Content Summary:
{content[:4000]}

Generate questions covering these deep academic dimensions:
1. Theoretical mechanisms: Analyze core theoretical frameworks and causal pathways
2. Methodological critique: Evaluate scientific rigor and innovation of research approaches
3. Empirical findings interpretation: Deeply interpret theoretical significance of data discoveries
4. Policy-practice connections: Discuss policy implications and practical applications
5. Disciplinary positioning: Analyze the research's contribution to field development
6. Future research directions: Propose insightful suggestions for subsequent investigations

CRITICAL REQUIREMENTS:
- Use professional academic discourse, avoid mentioning specific author names
- Refrain from using expressions like "this paper", "this study", "in XXX's research"
- Questions should demonstrate both professional depth in {domains_text} and universal applicability
- Emphasize theoretical analysis, mechanism exploration, and critical thinking
- Answers should exhibit rigorous academic reasoning processes, not relying on specific studies
- Avoid simple factual statements, focus on explanatory and analytical depth

Output in strict JSON format:
[{{"question": "question1", "answer": "answer1"}}, {{"question": "question2", "answer": "answer2"}}]

Generate {num_questions} universally applicable deep academic Q&A pairs:
"""
    else:
        prompt_content = f"""
Please generate {num_questions} bilingual deep academic questions based on this {domains_text} research.

Domains: {domains_text}
Themes: {themes_text}
Keywords: {', '.join(domain_analysis['top_keywords'][:5])}

Content:
{content[:4000]}

Generate profound questions covering:
- Theoretical frameworks and causal mechanisms
- Methodological rigor and innovation
- Interpretation of empirical findings
- Policy implications and practical applications
- Positioning in disciplinary development
- Future research trajectories

CRITICAL REQUIREMENTS:
- Generate both English and Chinese versions
- Use professional academic discourse, avoid author-specific references
- Refrain from "this paper/study" expressions in both languages
- Demonstrate deep domain expertise with universal applicability
- Show analytical reasoning in answers, not study-specific facts
- Focus on explanatory depth rather than simple statements

Output in strict JSON format:
[{{"question_en": "q1", "answer_en": "a1", "question_zh": "q1中文", "answer_zh": "a1中文"}}]

Generate {num_questions} bilingual universally applicable academic pairs:
"""
    
    return prompt_content

def generate_cross_domain_prompt(contents: Dict[str, str], num_questions: int, language: str, domain_analysis: Dict) -> str:
    """
    为多个文件生成跨领域深度问题的提示词 - 英文版
    """
    domains_text = '、'.join(domain_analysis['primary_domains'])
    themes_text = '、'.join(domain_analysis['primary_themes'])
    
    # 构建合并内容
    combined_content = ""
    for i, (file_name, content) in enumerate(contents.items(), 1):
        combined_content += f"Research Material {i}:\n{content[:2000]}\n\n"
    
    if language == "chinese":
        prompt_content = f"""
基于以下多份{domains_text}领域的研究资料，生成{num_questions}个具有学术深度的综合性问题。

研究领域：{domains_text}
核心主题：{themes_text}
关键词：{', '.join(domain_analysis['top_keywords'][:8])}
研究资料数量：{domain_analysis['file_count']}份

研究资料摘要：
{combined_content[:6000]}

请生成涵盖以下综合性学术维度的问题：
1. 理论整合探讨：分析不同研究在理论框架上的共性与差异
2. 方法论比较反思：综合评价各种研究方法的优劣和适用性
3. 实证证据汇聚：整合多个研究的发现，探讨其理论含义
4. 政策实践综合：基于多源证据提出综合性的政策建议
5. 学科发展研判：分析该领域研究的现状、挑战和前沿方向
6. 跨领域启示：探讨研究成果对其他相关领域的借鉴意义

重要要求：
- 使用专业学术语言，避免提及具体的研究资料编号和作者姓名
- 问题要体现{domains_text}领域的综合性和深度
- 强调理论整合、方法反思和前沿展望
- 答案应基于多份研究资料进行综合推理
- 避免简单的比较陈述，注重深度分析和综合判断

请严格按照以下JSON格式输出：
[{{"question": "问题1", "answer": "答案1"}}, {{"question": "问题2", "answer": "答案2"}}]

生成{num_questions}个综合性深度学术问答对：
"""
    elif language == "english":
        prompt_content = f"""
Based on the following multiple research materials in the {domains_text} field, generate {num_questions} comprehensive and profound academic questions.

Research Domains: {domains_text}
Core Themes: {themes_text}
Keywords: {', '.join(domain_analysis['top_keywords'][:8])}
Number of Research Materials: {domain_analysis['file_count']}

Research Materials Summary:
{combined_content[:6000]}

Generate questions covering these comprehensive academic dimensions:
1. Theoretical integration: Analyze commonalities and differences in theoretical frameworks across studies
2. Methodological reflection: Comprehensively evaluate strengths and applicability of various research approaches
3. Empirical evidence synthesis: Integrate findings from multiple studies to explore theoretical implications
4. Policy-practice integration: Propose comprehensive policy recommendations based on multiple evidence sources
5. Disciplinary development assessment: Analyze current status, challenges, and frontiers in the field
6. Cross-domain implications: Discuss referential significance of research findings to other related fields

CRITICAL REQUIREMENTS:
- Use professional academic discourse, avoid mentioning specific material numbers or author names
- Questions should demonstrate both comprehensiveness and depth in {domains_text}
- Emphasize theoretical integration, methodological reflection, and frontier outlook
- Answers should be based on comprehensive reasoning from multiple research materials
- Avoid simple comparative statements, focus on deep analysis and synthetic judgment
- Maintain universal applicability while demonstrating domain-specific expertise

Output in strict JSON format:
[{{"question": "question1", "answer": "answer1"}}, {{"question": "question2", "answer": "answer2"}}]

Generate {num_questions} comprehensive deep academic Q&A pairs with universal relevance:
"""
    else:
        prompt_content = f"""
Based on multiple research materials in {domains_text}, generate {num_questions} bilingual comprehensive academic questions.

Domains: {domains_text}
Themes: {themes_text}
Keywords: {', '.join(domain_analysis['top_keywords'][:8])}
Materials: {domain_analysis['file_count']}

Content Summary:
{combined_content[:6000]}

Generate comprehensive questions covering:
- Theoretical integration across studies
- Methodological evaluation and reflection
- Synthesis of empirical evidence
- Integrated policy recommendations
- Assessment of disciplinary development
- Cross-domain implications and insights

CRITICAL REQUIREMENTS:
- Generate both English and Chinese versions
- Use professional integrated academic discourse
- Avoid specific references to materials or authors
- Demonstrate comprehensive domain expertise with universal relevance
- Show synthetic reasoning in answers across multiple sources
- Focus on analytical depth rather than descriptive comparisons

Output in strict JSON format:
[{{"question_en": "q1", "answer_en": "a1", "question_zh": "q1中文", "answer_zh": "a1中文"}}]

Generate {num_questions} bilingual comprehensive academic pairs with cross-study relevance:
"""
    
    return prompt_content

def enhance_academic_language(qa_pairs: list, language: str, domain_info: Dict = None) -> list:
    """
    增强学术语言，移除所有人名引用和模糊指代
    """
    enhanced_pairs = []
    
    # 获取领域信息用于具体化表述
    domains_text = '、'.join(domain_info['primary_domains']) if domain_info else '相关领域'
    
    # 中文人名模式 - 更全面的匹配
    chinese_name_patterns = [
        # 双字姓 + 双字名
        r'[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴鬱胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍卻璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公万俟司马上官欧阳夏侯诸葛闻人东方赫连皇甫尉迟公羊澹台公冶宗政濮阳淳于单于太叔申屠公孙仲孙轩辕令狐钟离宇文长孙慕容鲜于闾丘司徒司空亓官司寇仉督子车颛孙端木巫马公西漆雕乐正壤驷公良拓跋夹谷宰父谷梁晋楚闫法汝鄢涂钦段干百里东郭南门呼延归海羊舌微生岳帅缑亢况后有琴梁丘左丘东门西门商牟佘佴伯赏南宫墨哈谯笪年爱阳佟第五言福][\u4e00-\u9fff]{1,3}',
        # 常见双字名模式
        r'[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文][\u4e00-\u9fff]{1,2}',
        # 英文名中译
        r'(史密斯|约翰逊|威廉斯|布朗|琼斯|米勒|戴维斯|加西亚|罗德里格斯|威尔逊|马丁|安德森|泰勒|托马斯|杰克逊|怀特|哈里斯|马丁内斯|汤普森|罗宾逊|克拉克|刘易斯|李|沃克|霍尔|艾伦|金|赖特|洛佩斯|希尔|斯科特|格林|亚当斯|贝克|冈萨雷斯|纳尔逊|卡特|米切尔|佩雷斯|罗伯茨|特纳|菲利普斯|坎贝尔|帕克|埃文斯|爱德华兹|柯林斯)',
    ]
    
    # 英文人名模式
    english_name_patterns = [
        # 常见英文名 + 姓氏
        r'\b(James|John|Robert|Michael|William|David|Richard|Charles|Joseph|Thomas|Christopher|Daniel|Paul|Mark|Donald|George|Kenneth|Steven|Edward|Brian|Ronald|Anthony|Kevin|Jason|Matthew|Gary|Timothy|Jose|Larry|Jeffrey|Frank|Scott|Eric|Stephen|Andrew|Raymond|Gregory|Joshua|Jerry|Dennis|Walter|Patrick|Peter|Harold|Douglas|Henry|Carl|Arthur|Ryan|Roger|Joe|Juan|Jack|Albert|Jonathan|Justin|Terry|Gerald|Keith|Samuel|Willie|Ralph|Lawrence|Nicholas|Roy|Benjamin|Bruce|Brandon|Adam|Harry|Fred|Wayne|Billy|Steve|Louis|Jeremy|Aaron|Randy|Howard|Eugene|Carlos|Russell|Bobby|Victor|Martin|Ernest|Phillip|Todd|Jesse|Craig|Alan|Shawn|Clarence|Sean|Philip|Chris|Johnny|Earl|Jimmy|Antonio|Danny|Bryan|Tony|Luis|Mike|Stanley|Leonard|Nathan|Dale|Manuel|Rodney|Curtis|Norman|Allen|Marvin|Vincent|Glenn|Jeffery|Travis|Jeff|Chad|Jacob|Lee|Melvin|Alfred|Kyle|Francis|Bradley|Jesus|Herbert|Frederick|Ray|Joel|Edwin|Don|Eddie|Ricky|Troy|Randall|Barry|Alexander|Bernard|Mario|Leroy|Francisco|Marcus|Micheal|Theodore|Clifford|Miguel|Oscar|Jay|Jim|Tom|Calvin|Alex|Jon|Ronnie|Bill|Lloyd|Tommy|Leon|Derek|Warren|Darrell|Jerome|Floyd|Leo|Alvin|Tim|Wesley|Gordon|Dean|Greg|Jorge|Dustin|Pedro|Derrick|Dan|Lewis|Zachary|Corey|Herman|Maurice|Vernon|Roberto|Clyde|Glen|Hector|Shane|Ricardo|Sam|Rick|Lester|Brent|Ramon|Charlie|Tyler|Gilbert|Gene|Marc|Reginald|Ruben|Brett|Angel|Nathaniel|Rafael|Leslie|Edgar|Milton|Raul|Ben|Chester|Cecil|Duane|Franklin|Andre|Elmer|Brad|Gabriel|Ron|Mitchell|Roland|Arnold|Harvey|Jared|Adrian|Karl|Cory|Claude|Erik|Darryl|Jamie|Neil|Jessie|Christian|Javier|Fernando|Clinton|Ted|Mathew|Tyrone|Darren|Lonnie|Lance|Cody|Julio|Kelly|Kurt|Allan|Nelson|Guy|Clayton|Hugh|Max|Dwayne|Dwight|Armando|Felix|Jimmie|Everett|Jordan|Ian|Wallace|Ken|Bob|Jaime|Casey|Alfredo|Alberto|Dave|Ivan|Johnnie|Sidney|Byron|Julian|Isaac|Morris|Clifton|Willard|Daryl|Ross|Virgil|Andy|Marshall|Salvador|Perry|Kirk|Sergio|Marion|Tracy|Seth|Kent|Terrance|Rene|Eduardo|Terrence|Enrique|Freddie|Wade|Austin|Stuart|Fredrick|Arturo|Alejandro|Jackie|Joey|Nick|Luther|Wendell|Jeremiah|Evan|Julius|Dana|Donnie|Otis|Shannon|Trevor|Oliver|Luke|Homer|Gerard|Doug|Kenny|Hubert|Angelo|Shaun|Lyle|Matt|Lynn|Alfonso|Orlando|Rex|Carlton|Ernesto|Cameron|Neal|Pablo|Lorenzo|Omar|Wilbur|Blake|Grant|Horace|Roderick|Kerry|Abraham|Willis|Rickey|Jean|Ira|Andres|Cesar|Johnathan|Malcolm|Rudolph|Damon|Kelvin|Rudy|Preston|Alton|Archie|Marco|Wm|Pete|Randolph|Garry|Geoffrey|Jonathon|Felipe|Bennie|Gerardo|Ed|Dominic|Robin|Loren|Delbert|Colin|Guillermo|Earnest|Lucas|Benny|Noel|Spencer|Rodolfo|Myron|Edmund|Garrett|Salvatore|Cedric|Lowell|Gregg|Sherman|Wilson|Devin|Sylvester|Kim|Roosevelt|Israel|Jermaine|Forrest|Wilbert|Leland|Simon|Guadalupe|Clark|Irving|Carroll|Bryant|Owen|Rufus|Woodrow|Sammy|Kristopher|Mack|Levi|Marcos|Gustavo|Jake|Lionel|Marty|Taylor|Ellis|Dallas|Gilberto|Clint|Nicolas|Laurence|Ismael|Orville|Drew|Jody|Ervin|Dewey|Al|Wilfred|Josh|Hugo|Ignacio|Caleb|Tomas|Sheldon|Erick|Frankie|Stewart|Doyle|Darrel|Rogelio|Terence|Santiago|Alonzo|Elias|Bert|Elbert|Ramiro|Conrad|Pat|Noah|Grady|Phil|Cornelius|Lamar|Rolando|Clay|Percy|Dexter|Bradford|Merle|Darin|Amos|Terrell|Moses|Irvin|Saul|Roman|Darnell|Randal|Tommie|Timmy|Darrin|Winston|Brendan|Toby|Van|Abel|Dominick|Boyd|Courtney|Jan|Emilio|Elijah|Cary|Domingo|Santos|Aubrey|Emmett|Marlon|Emanuel|Jerald|Edmond|Emil|Dewayne|Will|Otto|Teddy|Reynaldo|Bret|Morgan|Jess|Trent|Humberto|Emmanuel|Stephan|Louie|Vicente|Lamont|Stacy|Garland|Miles|Micah|Efrain|Billie|Logan|Heath|Rodger|Harley|Demetrius|Ethan|Eldon|Rocky|Pierre|Junior|Freddy|Eli|Bryce|Antoine|Robbie|Kendall|Royce|Sterling|Mickey|Chase|Grover|Elton|Cleveland|Dylan|Chuck|Damian|Reuben|Stan|August|Leonardo|Jasper|Russel|Erwin|Benito|Hans|Monte|Blaine|Ernie|Curt|Quentin|Agustin|Murray|Jamal|Devon|Adolfo|Harrison|Tyson|Burton|Brady|Elliott|Wilfredo|Bart|Jarrod|Vance|Denis|Damien|Joaquin|Harlan|Desmond|Elliot|Darwin|Ashley|Gregorio|Buddy|Xavier|Kermit|Roscoe|Esteban|Anton|Solomon|Scotty|Norbert|Elvin|Williams|Nolan|Carey|Rod|Quinton|Hal|Brain|Rob|Elwood|Kendrick|Darius|Moises|Son|Marlin|Fidel|Thaddeus|Cliff|Marcel|Ali|Jackson|Raphael|Bryon|Armand|Alvaro|Jeffry|Dane|Joesph|Thurman|Ned|Sammie|Rusty|Michel|Monty|Rory|Fabian|Reggie|Mason|Graham|Kris|Isaiah|Vaughn|Gus|Avery|Loyd|Diego|Alexis|Adolph|Norris|Millard|Rocco|Gonzalo|Derick|Rodrigo|Gerry|Stacey|Carmen|Wiley|Rigoberto|Alphonso|Ty|Shelby|Rickie|Noe|Vern|Bobbie|Reed|Jefferson|Elvis|Bernardo|Mauricio|Hiram|Donovan|Basil|Riley|Ollie|Nickolas|Maynard|Scot|Vince|Quincy|Eddy|Sebastian|Federico|Ulysses|Heriberto|Donnell|Cole|Denny|Davis|Gavin|Emery|Ward|Romeo|Jayson|Dion|Dante|Clement|Coy|Odell|Maxwell|Jarvis|Bruno|Issac|Mary|Patricia|Linda|Barbara|Elizabeth|Jennifer|Maria|Susan|Margaret|Dorothy|Lisa|Nancy|Karen|Betty|Helen|Sandra|Donna|Carol|Ruth|Sharon|Michelle|Laura|Sarah|Kimberly|Deborah|Jessica|Shirley|Cynthia|Angela|Melissa|Brenda|Amy|Anna|Rebecca|Virginia|Kathleen|Pamela|Martha|Debra|Amanda|Stephanie|Carolyn|Christine|Marie|Janet|Catherine|Frances|Ann|Joyce|Diane|Alice|Julie|Heather|Teresa|Doris|Gloria|Evelyn|Jean|Cheryl|Mildred|Katherine|Joan|Ashley|Judith|Rose|Janice|Kelly|Nicole|Judy|Christina|Kathy|Theresa|Beverly|Denise|Tammy|Irene|Jane|Lori|Rachel|Marilyn|Andrea|Kathryn|Louise|Sara|Anne|Jacqueline|Wanda|Bonnie|Julia|Ruby|Lois|Tina|Phyllis|Norma|Paula|Diana|Annie|Lillian|Emily|Robin|Peggy|Crystal|Gladys|Rita|Dawn|Connie|Florence|Tracy|Edna|Tiffany|Carmen|Rosa|Cindy|Grace|Wendy|Victoria|Michele|Tamara|Jessica|Sherry|Sylvia|Josephine|Thelma|Shannon|Sheila|Ethel|Ellen|Elaine|Marjorie|Carrie|Charlotte|Monica|Esther|Pauline|Emma|Juanita|Anita|Rhonda|Hazel|Amber|Eva|Debbie|April|Leslie|Clara|Lucille|Jamie|Joanne|Eleanor|Valerie|Danielle|Megan|Alicia|Suzanne|Michele|Gail|Bertha|Darlene|Veronica|Jill|Erin|Geraldine|Lauren|Cathy|Joann|Lorraine|Lynn|Sally|Regina|Erica|Beatrice|Dolores|Bernice|Audrey|Yvonne|Annette|June|Samantha|Marion|Dana|Stacy|Ana|Renee|Ida|Vivian|Roberta|Holly|Brittany|Melanie|Loretta|Yolanda|Jeanette|Laurie|Katie|Kristen|Vanessa|Alma|Sue|Elsie|Beth|Jeanne|Vicki|Carla|Tara|Rosemary|Eileen|Terri|Gertrude|Lucy|Tonya|Ella|Stacey|Wilma|Gina|Kristin|Jessie|Natalie|Agnes|Vera|Willie|Charlene|Bessie|Delores|Melinda|Pearl|Arlene|Maureen|Colleen|Allison|Tamara|Joy|Georgia|Constance|Lillie|Claudia|Jackie|Marcia|Tanya|Nellie|Minnie|Marlene|Heidi|Glenda|Lydia|Viola|Courtney|Marian|Stella|Caroline|Dora|Jo|Vickie|Mattie|Terry|Maxine|Irma|Mabel|Marsha|Myrtle|Lena|Christy|Deanna|Patsy|Hilda|Gwendolyn|Jennie|Nora|Margie|Nina|Cassandra|Leah|Penny|Kay|Priscilla|Naomi|Carole|Brandy|Olga|Billie|Dianne|Tracey|Leona|Jenny|Felicia|Sonia|Miriam|Velma|Becky|Bobbie|Violet|Kristina|Toni|Misty|Mae|Shelly|Daisy|Ramona|Sherri|Erika|Katrina|Claire|Lindsey|Lindsay|Geneva|Guadalupe|Belinda|Margarita|Sheryl|Cora|Faye|Ada|Natasha|Sabrina|Isabel|Marguerite|Hattie|Harriet|Molly|Cecilia|Kristi|Brandi|Blanche|Sandy|Rosie|Joanna|Iris|Eunice|Angie|Inez|Lynda|Madeline|Amelia|Alberta|Genevieve|Monique|Jodi|Janie|Maggie|Kayla|Sonya|Jan|Lee|Kristine|Candace|Fannie|Maryann|Opal|Alison|Yvette|Melody|Luz|Susie|Olivia|Flora|Shelley|Kristy|Mamie|Lula|Lola|Verna|Beulah|Antoinette|Candice|Juana|Jeannette|Pam|Kelli|Hannah|Whitney|Bridget|Karla|Celia|Latoya|Patty|Shelia|Gayle|Della|Vicky|Lynne|Sheri|Marianne|Kara|Jacquelyn|Erma|Blanca|Myra|Leticia|Pat|Krista|Roxanne|Angelica|Johnnie|Robyn|Francis|Adrienne|Rosalie|Alexandra|Brooke|Bethany|Sadie|Bernadette|Traci|Jody|Kendra|Jasmine|Nichole|Rachael|Chelsea|Mable|Ernestine|Muriel|Marcella|Elena|Krystal|Angelina|Nadine|Kari|Estelle|Dianna|Paulette|Lora|Mona|Doreen|Rosemarie|Angel|Desiree|Antonia|Hope|Ginger|Janis|Betsy|Christie|Freda|Mercedes|Meredith|Lynette|Teri|Cristina|Eula|Leigh|Meghan|Sophia|Eloise|Rochelle|Gretchen|Cecelia|Raquel|Henrietta|Alyssa|Jana|Kelley|Gwen|Kerry|Jenna|Tricia|Laverne|Olive|Alexis|Tasha|Silvia|Elvira|Casey|Delia|Sophie|Kate|Patti|Lorena|Kellie|Sonja|Lila|Lana|Darla|May|Mindy|Essie|Mandy|Lorene|Elsa|Josefina|Jeannie|Miranda|Dixie|Lucia|Marta|Faith|Lela|Johanna|Shari|Camille|Tami|Shawna|Elisa|Ebony|Melba|Ora|Nettie|Tabitha|Ollie|Jaime|Winifred|Kristie)\s+[A-Z][a-z]+',
        # 姓氏单独出现
        r'\b(Smith|Johnson|Williams|Brown|Jones|Garcia|Miller|Davis|Rodriguez|Martinez|Hernandez|Lopez|Gonzalez|Wilson|Anderson|Thomas|Taylor|Moore|Jackson|Martin|Lee|Perez|Thompson|White|Harris|Sanchez|Clark|Ramirez|Lewis|Robinson|Walker|Young|Allen|King|Wright|Scott|Torres|Nguyen|Hill|Flores|Green|Adams|Nelson|Baker|Hall|Rivera|Campbell|Mitchell|Carter|Roberts|Gomez|Phillips|Evans|Turner|Diaz|Parker|Cruz|Edwards|Collins|Reyes|Stewart|Morris|Morales|Murphy|Cook|Rogers|Gutierrez|Ortiz|Morgan|Cooper|Peterson|Bailey|Reed|Kelly|Howard|Ramos|Kim|Cox|Ward|Richardson|Watson|Brooks|Chavez|Wood|James|Bennett|Gray|Mendoza|Ruiz|Hughes|Price|Alvarez|Castillo|Sanders|Patel|Myers|Long|Ross|Foster|Jimenez|Powell|Jenkins|Perry|Russell|Sullivan|Bell|Coleman|Butler|Henderson|Barnes|Gonzales|Fisher|Vasquez|Simmons|Romero|Jordan|Patterson|Alexander|Hamilton|Graham|Reynolds|Griffin|Wallace|Moreno|West|Cole|Hayes|Bryant|Herrera|Gibson|Ellis|Tran|Medina|Aguilar|Stevens|Murray|Ford|Castro|Marshall|Owens|Harrison|Fernandez|Mcdonald|Woods|Washington|Kennedy|Wells|Vargas|Henry|Chen|Freeman|Webb|Tucker|Guzman|Burns|Crawford|Olson|Simpson|Porter|Hunter|Gordon|Mendez|Silva|Shaw|Snyder|Mason|Dixon|Munoz|Hunt|Hicks|Holmes|Palmer|Wagner|Black|Robertson|Boyd|Rose|Stone|Salazar|Fox|Warren|Mills|Meyer|Rice|Schmidt|Garza|Daniels|Ferguson|Nichols|Stephens|Soto|Weaver|Ryan|Gardner|Payne|Grant|Dunn|Kelley|Spencer|Hawkins|Arnold|Pierce|Austin|Peters|Hawkins|Hoffman|Johnston|Matthews|Wagner|Willis|Ray|Watkins|Olson|Carroll|Duncan|Snyder|Hart|Cunningham|Bradley|Lane|Andrews|Ruiz|Harper|Fox|Riley|Armstrong|Carpenter|Weaver|Greene|Lawrence|Elliott|Chavez|Sims|Austin|Peters|Kelley|Franklin|Lawson|Fields|Gutierrez|Ryan|Schmidt|Carr|Vasquez|Castillo|Wheeler|Chapman|Oliver|Montgomery|Richards|Williamson|Johnston|Banks|Meyer|Bishop|Mccoy|Howell|Alvarez|Morrison|Hansen|Fernandez|Garza|Harvey|Little|Burton|Stanley|Nguyen|George|Jacobs|Reid|Kim|Fuller|Lynch|Dean|Gilbert|Garrett|Romero|Welch|Larson|Frazier|Burke|Hanson|Day|Mendoza|Moreno|Bowman|Medina|Fowler|Brewer|Hoffman|Carlson|Silva|Pearson|Holland|Douglas|Fleming|Jensen|Vargas|Byrd|Davidson|Hopkins|May|Terry|Herrera|Wade|Soto|Walters|Curtis|Neal|Caldwell|Lowe|Jennings|Barnett|Graves|Jimenez|Horton|Shelton|Barrett|Obrien|Castro|Sutton|Gregory|Mckinney|Lucas|Miles|Craig|Rodriquez|Chambers|Holt|Lambert|Fletcher|Watts|Bates|Hale|Rhodes|Pena|Beck|Newman|Haynes|Mcdaniel|Mendez|Bush|Vaughn|Parks|Dawson|Santiago|Norris|Hardy|Love|Steele|Curry|Powers|Schultz|Barker|Guzman|Page|Munoz|Ball|Keller|Chandler|Weber|Leonard|Walsh|Lyons|Ramsey|Wolfe|Schneider|Mullins|Benson|Sharp|Bowen|Daniel|Barber|Cummings|Hines|Baldwin|Griffith|Valdez|Hubbard|Salazar|Reeves|Warner|Stevenson|Burgess|Santos|Tate|Cross|Garner|Mann|Mack|Moss|Thornton|Dennis|Mcgee|Farmer|Delgado|Aguilar|Vega|Glover|Manning|Cohen|Harmon|Rodgers|Robbins|Newton|Todd|Blair|Higgins|Ingram|Reese|Cannon|Strickland|Townsend|Potter|Goodwin|Walton|Rowe|Hampton|Ortega|Patton|Swanson|Joseph|Francis|Goodman|Maldonado|Yates|Becker|Erickson|Hodges|Rios|Conner|Adkins|Webster|Norman|Malone|Hammond|Flowers|Cobb|Moody|Quinn|Blake|Maxwell|Pope|Floyd|Osborne|Paul|Mccarthy|Guerrero|Lindsey|Estrada|Sandoval|Gibbs|Tyler|Gross|Fitzgerald|Stokes|Doyle|Sherman|Saunders|Wise|Colon|Gill|Alvarado|Greer|Padilla|Simon|Waters|Nunez|Ballard|Schwartz|Mcbride|Houston|Christensen|Klein|Pratt|Briggs|Parsons|Mclaughlin|Zimmerman|French|Buchanan|Moran|Copeland|Roy|Pittman|Brady|Mccormick|Holloway|Brock|Poole|Frank|Logan|Owen|Bass|Marsh|Drake|Wong|Jefferson|Park|Morton|Abbott|Sparks|Patrick|Norton|Huff|Clayton|Massey|Lloyd|Figueroa|Carson|Bowers|Roberson|Barton|Tran|Lamb|Harrington|Casey|Boone|Cortez|Clarke|Mathis|Singleton|Wilkins|Cain|Bryan|Underwood|Hogan|Mckenzie|Collier|Luna|Phelps|Mcguire|Allison|Bridges|Wilkerson|Nash|Summers|Atkins|Wilcox|Pitts|Conley|Marquez|Burnett|Richard|Cochran|Chase|Davenport|Hood|Gates|Clay|Ayala|Sawyer|Roman|Vazquez|Dickerson|Hodge|Acosta|Flynn|Espinoza|Nicholson|Monroe|Wolf|Morrow|Kirk|Randall|Anthony|Whitaker|Oconnor|Skinner|Ware|Molina|Kirby|Huffman|Bradford|Charles|Gilmore|Dominguez|Oneal|Bruce|Lang|Combs|Kramer|Heath|Hancock|Gallagher|Gaines|Shaffer|Short|Wiggins|Mathews|Mcclain|Fischer|Wall|Small|Melton|Hensley|Bond|Dyer|Cameron|Grimes|Contreras|Christian|Wyatt|Baxter|Snow|Mosley|Shepherd|Larsen|Hoover|Beasley|Glenn|Petersen|Whitehead|Meyers|Keith|Garrison|Vincent|Shields|Horn|Savage|Olsen|Schroeder|Hartman|Woodard|Mueller|Kemp|Deleon|Booth|Patel|Calhoun|Wiley|Eaton|Cline|Navarro|Harrell|Lester|Humphrey|Parrish|Duran|Hutchinson|Hess|Dorsey|Bullock|Robles|Beard|Dalton|Avila|Vance|Rich|Blackwell|York|Johns|Blankenship|Trevino|Salinas|Campos|Pruitt|Moses|Callahan|Golden|Montoya|Hardin|Guerra|Mcdowell|Carey|Stafford|Gallegos|Henson|Wilkinson|Booker|Merritt|Miranda|Atkinson|Orr|Decker|Hobbs|Preston|Tanner|Knox|Pacheco|Stephenson|Glass|Rojas|Serrano|Marks|Hickman|English|Sweeney|Strong|Prince|Mcclure|Conway|Walter|Roth|Maynard|Farrell|Lowery|Hurst|Nixon|Weiss|Trujillo|Ellison|Sloan|Juarez|Winters|Mclean|Randolph|Leon|Boyer|Villarreal|Mccall|Gentry|Carrillo|Kent|Ayers|Lara|Shannon|Sexton|Pace|Hull|Leblanc|Browning|Velasquez|Leach|Chang|House|Sellers|Herring|Noble|Foley|Bartlett|Mercado|Landry|Durham|Walls|Barr|Mckee|Bauer|Rivers|Everett|Bradshaw|Pugh|Velez|Rush|Estes|Dodson|Morse|Sheppard|Weeks|Camacho|Bean|Barron|Livingston|Middleton|Spears|Branch|Blevins|Chen|Kerr|Mcconnell|Hatfield|Harding|Ashley|Solis|Herman|Frost|Giles|Blackburn|William|Pennington|Woodward|Finley|Mcintosh|Koch|Best|Solomon|Mccullough|Dudley|Nolan|Blanchard|Rivas|Brennan|Mejia|Kane|Benton|Joyce|Buckley|Haley|Valentine|Maddox|Russo|Mcknight|Buck|Moon|Mcmillan|Crosby|Berg|Dotson|Mays|Roach|Cochran|Craft|Cleveland|Clemons|Wynn|Nielsen|Baird|Stanton|Snider|Rosales|Bright|Witt|Stuart|Hays|Holden|Rutledge|Kinney|Clements|Castaneda|Slater|Hahn|Emerson|Conrad|Burks|Delaney|Pate|Lancaster|Sweet|Justice|Tyson|Sharpe|Whitfield|Talley|Macias|Irwin|Burris|Ratliff|Mccray|Madden|Kaufman|Beach|Goff|Cash|Bolton|Mcfadden|Levine|Good|Byers|Kirkland|Kidd|Workman|Carney|Dale|Mcleod|Holcomb|England|Finch|Head|Burt|Hendrix|Sosa|Haney|Franks|Sargent|Nieves|Downs|Rasmussen|Bird|Hewitt|Cardenas|Fulton|Lynn|Lott|Calderon|Rosa|Pollard|Hooper|Burch|Mullen|Fry|Riddle|Levy|David|Duke|Odonnell|Guy|Michael|Britt|Frederick|Daugherty|Berger|Dillard|Alston|Jarvis|Frye|Riggs|Chaney|Odom|Duffy|Fitzpatrick|Valenzuela|Merrill|Mayer|Alford|Mcpherson|Acevedo|Donovan|Barrera|Albert|Cote|Reilly|Compton|Raymond|Mooney|Mcgowan|Craft|Cleveland|Clemons|Wynn|Nielsen|Baird|Stanton|Snider|Rosales|Bright|Witt|Stuart|Hays|Holden|Rutledge|Kinney|Clements|Castaneda|Slater|Hahn|Emerson|Conrad|Burks|Delaney|Pate|Lancaster|Sweet|Justice|Tyson|Sharpe|Whitfield|Talley|Macias|Irwin|Burris|Ratliff|Mccray|Madden|Kaufman|Beach|Goff|Cash|Bolton|Mcfadden|Levine|Good|Byers|Kirkland|Kidd|Workman|Carney|Dale|Mcleod|Holcomb|England|Finch|Head|Burt|Hendrix|Sosa|Haney|Franks|Sargent|Nieves|Downs|Rasmussen|Bird|Hewitt|Cardenas|Fulton|Lynn|Lott|Calderon|Rosa|Pollard|Hooper|Burch|Mullen|Fry|Riddle|Levy|David|Duke|Odonnell|Guy|Michael|Britt|Frederick|Daugherty|Berger|Dillard|Alston|Jarvis|Frye|Riggs|Chaney|Odom|Duffy|Fitzpatrick|Valenzuela|Merrill|Mayer|Alford|Mcpherson|Acevedo|Donovan|Barrera|Albert|Cote|Reilly|Compton|Raymond|Mooney|Mcgowan|Craft|Cleveland|Clemons|Wynn|Nielsen|Baird|Stanton|Snider|Rosales|Bright|Witt|Stuart|Hays|Holden|Rutledge|Kinney|Clements|Castaneda|Slater|Hahn|Emerson|Conrad|Burks|Delaney|Pate|Lancaster|Sweet|Justice|Tyson|Sharpe|Whitfield|Talley|Macias|Irwin|Burris|Ratliff|Mccray|Madden|Kaufman|Beach|Goff|Cash|Bolton|Mcfadden|Levine|Good|Byers|Kirkland|Kidd|Workman|Carney|Dale|Mcleod|Holcomb|England|Finch|Head|Burt|Hendrix|Sosa|Haney|Franks|Sargent|Nieves|Downs|Rasmussen|Bird|Hewitt|Cardenas|Fulton|Lynn|Lott|Calderon|Rosa|Pollard|Hooper|Burch|Mullen|Fry|Riddle|Levy|David|Duke|Odonnell|Guy|Michael|Britt|Frederick|Daugherty|Berger|Dillard|Alston|Jarvis|Frye|Riggs|Chaney|Odom|Duffy|Fitzpatrick|Valenzuela|Merrill|Mayer|Alford|Mcpherson|Acevedo|Donovan|Barrera|Albert|Cote|Reilly|Compton|Raymond|Mooney|Mcgowan)\b',
    ]
    
    for pair in qa_pairs:
        if language == "english":
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            
            # 移除英文人名引用
            for pattern in english_name_patterns:
                question = re.sub(pattern, 'research', question, flags=re.IGNORECASE)
                answer = re.sub(pattern, 'studies', answer, flags=re.IGNORECASE)
            
            # 移除"in XXX and XXX's research"等模式
            question = re.sub(r'\b(in|according to|based on)\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+(\'s)?\s+(research|study|work)\b', 
                            'in the research literature', question, flags=re.IGNORECASE)
            question = re.sub(r'\b[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+found\b', 'research has found', question, flags=re.IGNORECASE)
            
            answer = re.sub(r'\b(in|according to|based on)\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+(\'s)?\s+(research|study|work)\b', 
                          'based on research', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\b[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+(found|showed|demonstrated)\b', 'studies have shown', answer, flags=re.IGNORECASE)
            
            # 使用具体领域术语替代模糊指代
            question = re.sub(r'\bthis (paper|study|research)\b', f'the {domains_text} literature', question, flags=re.IGNORECASE)
            question = re.sub(r'\bthe (paper|study)\b', f'the {domains_text} research', question, flags=re.IGNORECASE)
            
            answer = re.sub(r'\bthis (paper|study|research)\b', f'the {domains_text} literature', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\bthe (paper|study)\b', f'the {domains_text} research', answer, flags=re.IGNORECASE)
            
            enhanced_pair = {
                "question": question,
                "answer": answer,
                "metadata": pair.get("metadata", {})
            }
            
        elif language == "chinese":
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            
            # 移除中文人名引用
            for pattern in chinese_name_patterns:
                question = re.sub(pattern, '相关研究', question)
                answer = re.sub(pattern, '实证分析', answer)
            
            # 移除具体的作者引用模式
            question = re.sub(r'在[^，。]*?(研究|分析|探讨|考察)中', '在相关研究中', question)
            question = re.sub(r'[^，。]*?等?(?:发现|表明|证明|指出)', '研究显示', question)
            question = re.sub(r'[^，。]*?的(?:研究|分析|探讨)', '相关研究', question)
            
            answer = re.sub(r'在[^，。]*?(研究|分析|探讨|考察)中', '研究显示', answer)
            answer = re.sub(r'[^，。]*?等?(?:发现|表明|证明|指出)', '实证分析表明', answer)
            answer = re.sub(r'[^，。]*?的(?:研究|分析|探讨)', '相关研究', answer)
            
            # 使用具体领域术语替代模糊指代
            question = re.sub(r'本文', f'{domains_text}文献', question)
            question = re.sub(r'本研究', f'{domains_text}研究', question)
            question = re.sub(r'该研究', f'{domains_text}领域', question)
            
            answer = re.sub(r'本文', f'{domains_text}文献', answer)
            answer = re.sub(r'本研究', f'{domains_text}研究', answer)
            answer = re.sub(r'该研究', f'{domains_text}领域', answer)
            
            enhanced_pair = {
                "question": question,
                "answer": answer,
                "metadata": pair.get("metadata", {})
            }
        else:  # bilingual
            question_en = pair.get('question_en', '')
            question_zh = pair.get('question_zh', '')
            answer_en = pair.get('answer_en', '')
            answer_zh = pair.get('answer_zh', '')
            
            # 英文清理
            for pattern in english_name_patterns:
                question_en = re.sub(pattern, 'research', question_en, flags=re.IGNORECASE)
                answer_en = re.sub(pattern, 'studies', answer_en, flags=re.IGNORECASE)
            
            # 中文清理
            for pattern in chinese_name_patterns:
                question_zh = re.sub(pattern, '相关研究', question_zh)
                answer_zh = re.sub(pattern, '实证分析', answer_zh)
            
            # 移除中英文混合的人名引用
            question_en = re.sub(r'\b(in|according to|based on)\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+(\'s)?\s+(research|study|work)\b', 
                               'in the research literature', question_en, flags=re.IGNORECASE)
            answer_en = re.sub(r'\b(in|according to|based on)\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+(\'s)?\s+(research|study|work)\b', 
                             'based on research', answer_en, flags=re.IGNORECASE)
            
            question_zh = re.sub(r'在[^，。]*?(研究|分析|探讨|考察)中', '在相关研究中', question_zh)
            answer_zh = re.sub(r'在[^，。]*?(研究|分析|探讨|考察)中', '研究显示', answer_zh)
            
            enhanced_pair = {
                "question_en": question_en,
                "answer_en": answer_en,
                "question_zh": question_zh,
                "answer_zh": answer_zh,
                "metadata": pair.get("metadata", {})
            }
        
        enhanced_pairs.append(enhanced_pair)
    
    return enhanced_pairs

def generate_single_file_questions(content: str, file_name: str, num_questions: int, language: str, domain_analysis: Dict) -> list:
    """
    为单个文件生成深度学术问题
    """
    print(f"📖 为 {file_name} 生成 {num_questions} 个深度学术问题")
    
    # 生成单个文件的提示词
    prompt = generate_single_domain_prompt(content, file_name, num_questions, language, domain_analysis)
    
    qa_pairs = _call_deepseek_api_with_retry(prompt, language)
    
    if qa_pairs:
        # 增强学术语言，传入领域信息
        enhanced_pairs = enhance_academic_language(qa_pairs, language, domain_analysis)
        print(f"✅ 成功生成 {len(enhanced_pairs)} 个深度学术问答对")
        return enhanced_pairs[:num_questions]
    else:
        print("❌ 问题生成失败")
        return []

def generate_cross_file_questions(contents: Dict[str, str], num_questions: int, language: str, domain_analysis: Dict) -> list:
    """
    为多个文件生成综合性深度问题
    """
    print(f"🔗 基于 {len(contents)} 个文件生成 {num_questions} 个综合性深度问题")
    
    # 生成跨文件提示词
    prompt = generate_cross_domain_prompt(contents, num_questions, language, domain_analysis)
    
    qa_pairs = _call_deepseek_api_with_retry(prompt, language)
    
    if qa_pairs:
        # 增强学术语言，传入领域信息
        enhanced_pairs = enhance_academic_language(qa_pairs, language, domain_analysis)
        print(f"✅ 成功生成 {len(enhanced_pairs)} 个综合性深度问答对")
        return enhanced_pairs[:num_questions]
    else:
        print("❌ 问题生成失败")
        return []

def adaptive_question_generation(contents: Dict[str, str], num_questions: int, language: str, domain_analysis: Dict) -> list:
    """
    自适应问题生成：根据文件数量选择生成策略
    """
    all_qa_pairs = []
    
    if len(contents) == 1:
        # 单文件：只生成深度问题
        file_name, content = list(contents.items())[0]
        single_qa_pairs = generate_single_file_questions(content, file_name, num_questions, language, domain_analysis)
        all_qa_pairs.extend(single_qa_pairs)
    else:
        # 多文件：为每个文件生成深度问题 + 综合性问题
        # 单个文件问题（每个文件生成较少问题）
        single_questions_per_file = max(2, num_questions // (len(contents) * 2))
        for file_name, content in contents.items():
            single_qa_pairs = generate_single_file_questions(content, file_name, single_questions_per_file, language, domain_analysis)
            all_qa_pairs.extend(single_qa_pairs)
        
        # 综合性问题
        cross_questions = max(3, num_questions - len(all_qa_pairs))
        if cross_questions > 0:
            cross_qa_pairs = generate_cross_file_questions(contents, cross_questions, language, domain_analysis)
            all_qa_pairs.extend(cross_qa_pairs)
    
    return all_qa_pairs[:num_questions]

def save_qa_dataset(qa_pairs: list, output_file: str, instruction: str, language: str = "both"):
    """
    保存问答数据集
    """
    total_records = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for pair in qa_pairs:
            metadata = pair.get("metadata", {})
            
            if language == "both":
                # 双语版本
                if all(key in pair for key in ["question_en", "answer_en", "question_zh", "answer_zh"]):
                    # 英文版本
                    output_record_en = {
                        "instruction": instruction,
                        "input": pair["question_en"],
                        "output": pair["answer_en"],
                        "metadata": metadata
                    }
                    f_out.write(json.dumps(output_record_en, ensure_ascii=False) + "\n")
                    
                    # 中文版本
                    output_record_zh = {
                        "instruction": instruction,
                        "input": pair["question_zh"],
                        "output": pair["answer_zh"],
                        "metadata": metadata
                    }
                    f_out.write(json.dumps(output_record_zh, ensure_ascii=False) + "\n")
                    total_records += 2
            
            elif language in ["english", "chinese"]:
                if "question" in pair and "answer" in pair:
                    output_record = {
                        "instruction": instruction,
                        "input": pair["question"],
                        "output": pair["answer"],
                        "metadata": metadata
                    }
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    total_records += 1
    
    return total_records

def main():
    """
    主程序 - 深度学术问题生成器
    """
    print("=== 深度学术问题生成器 ===")
    print("✨ 核心特点：")
    print("   - 用户自定义专家角色")
    print("   - 深度领域专业问题")
    print("   - 综合性学术推理")
    print("   - 专业学术语言表达")
    
    # 配置
    txt_folder = input("请输入txt文件所在文件夹路径 (默认: txt_files): ").strip() or "txt_files"
    
    # 用户输入专家角色
    instruction = input("请输入instruction内容 (例如: '你是一位经济学专家'): ").strip()
    if not instruction:
        instruction = "你是一位专业研究人员"
    
    language_choice = input("选择语言版本 (1=英文, 2=中文, 3=中英双语): ").strip()
    if language_choice == "1":
        language = "english"
        output_file = "qa_academic_english.jsonl"
    elif language_choice == "2":
        language = "chinese"
        output_file = "qa_academic_chinese.jsonl"
    else:
        language = "both"
        output_file = "qa_academic_bilingual.jsonl"
    
    num_questions = input("要生成多少个问题？ (默认20): ").strip()
    if not num_questions:
        num_questions = 20
    else:
        num_questions = int(num_questions)
    
    # 读取文件
    print("\n正在读取文件...")
    contents = read_multiple_txt_files(txt_folder)
    
    if not contents:
        print("❌ 没有找到文件，请检查文件夹路径")
        return
    
    # 分析研究领域
    domain_analysis = analyze_research_domains(contents)
    
    print(f"\n📊 综合分析结果:")
    print(f"   主要领域: {', '.join(domain_analysis['primary_domains'])}")
    print(f"   核心主题: {', '.join(domain_analysis['primary_themes'])}")
    print(f"   关键词: {', '.join(domain_analysis['top_keywords'][:8])}")
    print(f"   文件数量: {domain_analysis['file_count']}")
    
    # 生成问题
    print(f"\n🚀 开始生成深度学术问题...")
    qa_pairs = adaptive_question_generation(contents, num_questions, language, domain_analysis)
    
    print(f"\n✅ 生成完成!")
    print(f"   成功生成: {len(qa_pairs)} 个深度学术问答对")
    
    # 保存结果
    total_records = save_qa_dataset(qa_pairs, output_file, instruction, language)
    print(f"   保存到: {output_file}")
    print(f"   总记录数: {total_records} 条")
    
    # 显示示例
    if qa_pairs:
        print(f"\n🎓 示例深度学术问题:")
        print("-" * 60)
        for i, pair in enumerate(qa_pairs[:3]):
            if language == "both":
                print(f"{i+1}. [EN] {pair.get('question_en', 'N/A')}")
                print(f"    [ZH] {pair.get('question_zh', 'N/A')}")
            else:
                print(f"{i+1}. {pair.get('question', 'N/A')}")
            print()

if __name__ == "__main__":
    main()