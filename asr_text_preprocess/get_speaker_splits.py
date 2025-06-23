import os
import json
import re
from bs4 import BeautifulSoup
import time
from prompts import SPEAKER_SPLIT_SYS,SPEAKER_SPLIT_USER,SPEAKER_SPLIT_EXAMPLES,FORMAT_CORRECTION,SPEAKER_SPLIT_FORMAT
from utils import call_llm, get_all_txt_files
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
from text2sentence import split_text_into_sentences

def parse_segments_xml(text: str) -> List[Dict[str, str]]:
    """
    解析XML格式的说话人分割结果
    """
    try:
        # 包装所有段落在根元素中
        wrapped_text = f"<ROOT>{text}</ROOT>"
        soup = BeautifulSoup(wrapped_text, "xml")
        
        segments = []
        for segment in soup.find_all("SEGMENT"):
            try:
                segment_data = {
                    "id": segment.find("ID").text.strip() if segment.find("ID") else "",
                    "analysis": segment.find("ANALYSIS").text.strip() if segment.find("ANALYSIS") else "",
                    "speaker": segment.find("SPEAKER").text.strip() if segment.find("SPEAKER") else "",
                    "content": segment.find("CONTENT").text.strip() if segment.find("CONTENT") else ""
                }
                segments.append(segment_data)
            except Exception as e:
                print(f"解析单个段落时出错: {e}")
                continue
        
        return segments
    except Exception as e:
        print(f"XML解析失败: {e}")
        return []
    
def count_tokens(text: str, tokenizer) -> int:
    """
    计算文本的token数量
    
    Args:
        text: 要计算的文本
        tokenizer: 用于分词的tokenizer
        
    Returns:
        token数量
    """
    tokens = tokenizer.encode(text)
    return len(tokens)

def split_speakers(
    txt_path_list: List[str], 
    output_dir: str,
    model_path: str = "/data4/liangyaozhen/model/Qwen2-7B-Instruct",
    split_rules: str = "根据语气变化、话题转换、代词使用等线索进行说话人分割",
    max_tokens_per_batch: int = 2148,
) -> Dict[str, Any]:
    """
    批量处理ASR文本的说话人分割，基于token数量限制分批处理
    
    Args:
        txt_path_list: ASR文本文件路径列表
        output_dir: 输出目录
        model_path: 模型路径，用于加载tokenizer
        split_rules: 说话人分割规则
        max_tokens_per_batch: 每批最大token数    
    Returns:
        处理结果统计
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer
    print(f"加载tokenizer: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"加载tokenizer失败: {e}")
        print("使用默认的token计数方法（按字符数估算）")
        tokenizer = None
    
    # 处理结果统计
    results = {
        "total_files": len(txt_path_list),
        "processed_files": 0,
        "failed_files": 0,
        "failed_file_list": [],
        "processing_details": []
    }
    
    def estimate_tokens(text):
        """估算token数量，当tokenizer不可用时使用"""
        # 中文字符大约是1个token，英文单词大约是1.3个token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + int(english_words * 1.3)
    
    def count_tokens(text, tokenizer):
        """计算文本的token数量"""
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            return estimate_tokens(text)
    
    def format_corrector(response_text, format_example=SPEAKER_SPLIT_FORMAT):
        """尝试修正格式不正确的输出"""
        try:
            prompt = FORMAT_CORRECTION.format(format_example, response_text)
            messages = [
                {"role":'system','content':'你是有用的助手'},
                {"role": "user", "content": prompt}
            ]
            correction_response = call_llm(messages)
            if correction_response:
                return correction_response.choices[0].message.content
        except Exception as e:
            print(f"格式修正失败: {e}")
        return None
    
    def generate_summary(segments, batch_text):
        """生成当前批次的摘要"""
        try:
            summary_prompt = f"""
            请根据以下对话片段，生成一个简短的摘要（不超过100字），概括主要说话人和讨论的话题：
            
            {batch_text}
            
            根据分割结果，共有{len(segments)}个说话片段。
            """
            
            messages = [
                {"role": "user", "content": summary_prompt}
            ]
            
            summary_response = call_llm(messages)
            if summary_response:
                return summary_response.choices[0].message.content
            return "无法生成摘要"
        except Exception as e:
            print(f"生成摘要失败: {e}")
            return "生成摘要过程中发生错误"
    
    def process_batch(batch_text, batch_id, history_summary=""):
        """处理单个批次的文本"""
        print(f"处理批次 {batch_id}，约 {count_tokens(batch_text, tokenizer)} tokens")
        
        # 构建消息
        user_prompt = SPEAKER_SPLIT_USER.format(
            split_rules=split_rules,
            asr_raw_text=batch_text,
            speaker_split_examples=SPEAKER_SPLIT_EXAMPLES,
            asr_history=history_summary
        )
        
        messages = [
            {"role": "system", "content": SPEAKER_SPLIT_SYS},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用LLM
        try:
            response = call_llm(messages)
            
            if response is None:
                print(f"批次 {batch_id} LLM调用失败")
                return None, history_summary
            
            # 获取响应内容
            response_text = response.choices[0].message.content
            
            # 解析分割结果
            segments = parse_segments_xml(response_text)
            
            # 如果解析失败，尝试修正格式
            if not segments:
                print(f"批次 {batch_id} 分割结果解析失败，尝试修正格式")
                corrected_text = format_corrector(response_text, SPEAKER_SPLIT_EXAMPLES)
                if corrected_text:
                    segments = parse_segments_xml(corrected_text)
                    if segments:
                        print(f"格式修正成功，成功解析出 {len(segments)} 个片段")
                    else:
                        print(f"格式修正后仍然解析失败")
            
            # 生成新的历史摘要
            new_history_summary = history_summary
            if segments:
                new_history_summary = generate_summary(segments, batch_text)
            
            return segments, new_history_summary
        
        except Exception as e:
            print(f"处理批次 {batch_id} 时出错: {e}")
            return None, history_summary
    
    for i, txt_path in enumerate(txt_path_list):
        print(f"处理第 {i+1}/{len(txt_path_list)} 个文件: {txt_path}")
        
        file_processing_detail = {
            "file": txt_path,
            "segments_count": 0,
            "batch_count": 0,
            "failed_segments": [],
            "status": "processing"
        }
        
        try:
            # 读取ASR文本
            with open(txt_path, 'r', encoding='utf-8') as f:
                asr_text = f.read().strip()
            
            if not asr_text:
                print(f"文件 {txt_path} 为空，跳过处理")
                results["failed_files"] += 1
                results["failed_file_list"].append({"file": txt_path, "error": "文件为空"})
                file_processing_detail["status"] = "failed"
                file_processing_detail["error"] = "文件为空"
                results["processing_details"].append(file_processing_detail)
                continue
            
            # 分割为句子
            sentences, _ = split_text_into_sentences(asr_text)
            
            if not sentences:
                print(f"文件 {txt_path} 分割句子失败，跳过处理")
                results["failed_files"] += 1
                results["failed_file_list"].append({"file": txt_path, "error": "分割句子失败"})
                file_processing_detail["status"] = "failed"
                file_processing_detail["error"] = "分割句子失败"
                results["processing_details"].append(file_processing_detail)
                continue
            
            # 生成输出文件名
            file_stem = Path(txt_path).stem
            
            # 准备保存结果
            all_segments = []
            batch_id = 0
            
            # 初始化历史摘要
            history_summary = "这是音频文本的开头。"
            
            empty_prompt_tokens = 0  # 这里可以计算prompt模板的token数
            
            # 设置实际可用的token数量
            available_tokens = max_tokens_per_batch - empty_prompt_tokens - 100  # 留一些余量
            
            # 分批处理
            current_batch = []
            current_tokens = 0
            
            for sentence in sentences:
                # 计算当前句子的token数
                sentence_tokens = count_tokens(sentence, tokenizer)
                
                # 如果加入当前句子会超过限制，先处理当前批次
                if current_tokens + sentence_tokens > available_tokens and current_batch:
                    batch_id += 1
                    file_processing_detail["batch_count"] = batch_id
                    
                    # 构建当前批次的文本
                    batch_text = "".join(current_batch)
                    
                    # 处理当前批次
                    segments, history_summary = process_batch(batch_text, batch_id, history_summary)
                    
                    if segments:
                        # 更新ID以保持连续性
                        start_id = len(all_segments) + 1
                        for j, segment in enumerate(segments):
                            segment["id"] = str(start_id + j)
                            segment["batch"] = batch_id
                            all_segments.append(segment)
                        
                        file_processing_detail["segments_count"] = len(all_segments)
                    else:
                        print(f"批次 {batch_id} 处理失败")
                        file_processing_detail["failed_segments"].append({
                            "batch_id": batch_id,
                            "error": "分割结果解析失败"
                        })
                    
                    # 重置当前批次
                    current_batch = []
                    current_tokens = 0
                
                # 添加当前句子到批次
                current_batch.append(sentence)
                current_tokens += sentence_tokens
            
            # 处理最后一个批次
            if current_batch:
                batch_id += 1
                file_processing_detail["batch_count"] = batch_id
                
                # 构建当前批次的文本
                batch_text = "".join(current_batch)
                
                # 处理当前批次
                segments, history_summary = process_batch(batch_text, batch_id, history_summary)
                
                if segments:
                    # 更新ID以保持连续性
                    start_id = len(all_segments) + 1
                    for j, segment in enumerate(segments):
                        segment["id"] = str(start_id + j)
                        segment["batch"] = batch_id
                        all_segments.append(segment)
                    
                    file_processing_detail["segments_count"] = len(all_segments)
                else:
                    print(f"批次 {batch_id} 处理失败")
                    file_processing_detail["failed_segments"].append({
                        "batch_id": batch_id,
                        "error": "分割结果解析失败"
                    })
            
            # 保存所有结果
            if all_segments:
                # 保存为JSONL格式
                jsonl_output_path = output_path / f"{file_stem}_speaker_split.jsonl"
                with open(jsonl_output_path, 'w', encoding='utf-8') as f:
                    for segment in all_segments:
                        f.write(json.dumps(segment, ensure_ascii=False) + '\n')
                print(f"JSONL结果已保存到: {jsonl_output_path}")
                
                results["processed_files"] += 1
                file_processing_detail["status"] = "success"
                
                print(f"成功处理 {txt_path}，分割出 {len(all_segments)} 个段落，共 {batch_id} 个批次")
                
                # 检查是否有失败的段落
                if file_processing_detail["failed_segments"]:
                    file_processing_detail["status"] = "partial_success"
                    print(f"部分批次处理失败: {len(file_processing_detail['failed_segments'])} 个批次")
            else:
                print(f"文件 {txt_path} 处理失败，未获取到有效分割结果")
                results["failed_files"] += 1
                results["failed_file_list"].append({"file": txt_path, "error": "未获取到有效分割结果"})
                file_processing_detail["status"] = "failed"
                file_processing_detail["error"] = "未获取到有效分割结果"
            
            results["processing_details"].append(file_processing_detail)
            
        except Exception as e:
            print(f"处理文件 {txt_path} 时出错: {e}")
            results["failed_files"] += 1
            results["failed_file_list"].append({"file": txt_path, "error": str(e)})
            file_processing_detail["status"] = "failed"
            file_processing_detail["error"] = str(e)
            results["processing_details"].append(file_processing_detail)
    
    # 保存处理结果统计
    summary_path = output_path / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n批量处理完成!")
    print(f"总文件数: {results['total_files']}")
    print(f"成功处理: {results['processed_files']}")
    print(f"处理失败: {results['failed_files']}")
    print(f"处理结果统计已保存到: {summary_path}")
    
    return results


# 使用示例
if __name__ == "__main__":
    # 分割规则示例
    # split_rules = """
    # 1. 识别未明子本人的语言风格和习惯用语，如特定的口头禅、表达方式
    # 2. 注意代词变化，"我"和"你"的切换通常表示说话人变化
    # 3. 观察问答模式，连麦用户通常会提问，未明子会回答
    # 4. 注意语气和情绪变化，未明子通常更有权威感和解释性语气
    # 5. 关注话题突然转换，可能意味着说话人的变化
    # 6. 识别连贯性中断，如话题突然转换或语调变化
    # 7. 注意表达风格的差异，连麦用户通常语言更简短、疑问较多
    # """
    with open('/data3/liangyaozhen/vvmz/text_v0/speaker_split_rules/speaker_split_rules_20250615_234627/final_rules_summary.txt','r') as f:
        split_rules = '\n'.join(f.readlines())

    # txt_files = [
    #     '/data3/liangyaozhen/vvmz/raw_text/创伤性分离txt/BV11i7FzzEmK_【未明子】随便聊聊 2025.05.31录播_2025-05-31-04-14-42.txt'
    # ]
    txt_files = get_all_txt_files('/data3/liangyaozhen/vvmz/raw_text')

    # 输出目录
    output_directory = "/data3/liangyaozhen/vvmz/text_v0/splited_text_0"
    
    # 执行分割
    results = split_speakers(
        txt_files,
        output_directory,
        split_rules=split_rules,
    )
