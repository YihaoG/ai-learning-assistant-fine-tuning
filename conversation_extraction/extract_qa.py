import json
import os
from typing import List, Dict, Any, Tuple
import re
import logging
from dataclasses import dataclass
from utils import call_llm,count_tokens
from prompts import EXTRACTION_V5
# 安装依赖: pip install tiktoken

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """处理配置参数"""
    input_folder: str = "input_asr_files"
    output_folder: str = "output_qa_results" 
    window_length: int = 1024
    max_iterations_per_window: int = 15
    model_name: str = "gpt-4"  # 用于确定正确的tokenizer

def load_asr_files(input_folder: str) -> Dict[str, List[Dict]]:
    """
    加载输入文件夹中的所有JSON文件
    """
    asr_data = {}
    
    if not os.path.exists(input_folder):
        logger.error(f"输入文件夹不存在: {input_folder}")
        return asr_data
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(input_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    asr_data[filename] = data
            except Exception as e:
                logger.error(f"加载文件 {filename} 时出错: {e}")
    
    return asr_data

def create_windows(asr_text: List[Dict], window_length: int, model_name: str = "gpt-4") -> List[List[Dict]]:
    """
    将ASR文本分割为窗口，保持完整的发言项
    """
    windows = []
    current_window = []
    current_length = 0
    
    for item in asr_text:
        # 格式化单个发言项
        item_text = f"{item['speaker']}: {item['content']}"
        item_tokens = count_tokens(item_text, model_name)
        
        # 如果添加这个项目会超过窗口长度，并且当前窗口不为空，则开始新窗口
        if current_length + item_tokens > window_length and current_window:
            windows.append(current_window)
            current_window = [item]
            current_length = item_tokens
        else:
            current_window.append(item)
            current_length += item_tokens
    
    # 添加最后一个窗口
    if current_window:
        windows.append(current_window)
    
    logger.info(f"创建了 {len(windows)} 个窗口")
    return windows

def format_context_window(window: List[Dict]) -> str:
    """
    格式化上下文窗口为字符串
    """
    formatted = []
    for item in window:
        formatted.append(f"{item['speaker']}: {item['content']}")
    return "\n".join(formatted)

def build_prompt(history_actions: List[Dict], qa_list: List[Dict], context_window: str) -> str:
    """
    构建发送给LLM的完整prompt
    """
    # 格式化历史动作
    history_str = json.dumps(history_actions, ensure_ascii=False, indent=2) if history_actions else "[]"
    
    # 格式化QA列表
    qa_str = json.dumps(qa_list, ensure_ascii=False, indent=2) if qa_list else "[]"
    
    prompt_template = EXTRACTION_V5

    return prompt_template.format(
        history_actions=history_str,
        qa_list=qa_str,
        context_window=context_window
    )


def extract_response_parts(response: str) -> Tuple[str, Dict]:
    """
    从LLM响应中提取思考过程和JSON动作
    返回: (思考过程, 动作JSON字典)
    """
    # 提取思考过程
    thinking_pattern = r'##\s*思考过程[：:](.*?)##\s*最终输出[：:]'
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)
    thinking_process = ""
    if thinking_match:
        thinking_process = thinking_match.group(1).strip()
    else:
        # 如果没有找到标准格式，尝试其他模式
        thinking_alt_pattern = r'##\s*思考过程[：:](.*?)```json'
        thinking_alt_match = re.search(thinking_alt_pattern, response, re.DOTALL)
        if thinking_alt_match:
            thinking_process = thinking_alt_match.group(1).strip()
        else:
            # 最后尝试从开始到第一个json块
            json_start = response.find('```json')
            if json_start > 0:
                thinking_process = response[:json_start].strip()
    
    # 提取JSON
    json_patterns = [
        r'```json\s*(.*?)\s*```',  # 标准markdown json块
        r'```\s*(.*?)\s*```',      # 普通代码块
        r'\{.*\}',                 # 直接的JSON对象
    ]
    
    action_data = None
    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                action_data = json.loads(json_str)
                break
            except json.JSONDecodeError:
                continue
    
    if action_data is None:
        logger.error("未能从响应中提取有效的JSON")
        logger.debug(f"原始响应: {response[:500]}...")
    
    return thinking_process, action_data

def update_data_structures(action_data: Dict, thinking_process: str, history_actions: List[Dict], 
                         qa_list: List[Dict], full_actions: List[Dict]) -> bool:
    """
    根据动作更新数据结构
    返回是否应该继续当前窗口的处理
    """
    if not action_data:
        return False
    
    # 添加到完整动作历史（包含思考过程和完整的action信息）
    full_action_record = {
        "thinking_process": thinking_process,
        "action_data": action_data.copy(),
        "timestamp": None  # 可以在这里添加时间戳如果需要的话
    }
    full_actions.append(full_action_record)
    
    # 创建历史动作记录（不包含question和answer，用于prompt）
    history_action = {
        "action": action_data.get("action"),
        "reasoning": action_data.get("reasoning", "")
    }
    
    action_type = action_data.get("action")
    
    if action_type == "enrich_existing":
        target_index = action_data.get("target_qa_index")
        history_action["target_qa_index"] = target_index
        
        # 更新对应的QA
        if target_index is not None:
            try:
                target_index = int(target_index)
                for qa in qa_list:
                    if qa["qa_index"] == target_index:
                        if action_data.get("question"):
                            qa["question"] = action_data["question"]
                        if action_data.get("answer"):
                            qa["answer"] = action_data["answer"]
                        break
                else:
                    logger.warning(f"未找到目标QA索引: {target_index}")
            except (ValueError, TypeError):
                logger.error(f"无效的QA索引: {target_index}")
        
        history_actions.append(history_action)
        return True  # 继续处理当前窗口
        
    elif action_type == "add_new_QA":
        # 添加新的QA对
        new_qa_index = max([qa["qa_index"] for qa in qa_list], default=-1) + 1
        new_qa = {
            "qa_index": new_qa_index,
            "question": action_data.get("question", ""),
            "answer": action_data.get("answer", "")
        }
        qa_list.append(new_qa)
        
        history_actions.append(history_action)
        logger.info(f"添加新的QA对，索引: {new_qa_index}")
        return True  # 继续处理当前窗口
        
    elif action_type == "window_complete":
        history_actions.append(history_action)
        logger.info("窗口处理完成")
        return False  # 跳转到下一个窗口
    
    logger.warning(f"未知的动作类型: {action_type}")
    return True

def process_single_file(filename: str, asr_text: List[Dict], config: ProcessingConfig, call_llm) -> Dict:
    """
    处理单个ASR文件
    """
    logger.info(f"开始处理文件: {filename}")
    
    # 创建窗口
    windows = create_windows(asr_text, config.window_length, config.model_name)
    
    # 初始化数据结构
    qa_list = []
    full_actions = []
    
    # 处理每个窗口
    for window_idx, window in enumerate(windows):
        logger.info(f"处理窗口 {window_idx + 1}/{len(windows)}")
        
        history_actions = []  # 每个窗口的历史动作重置
        context_window = format_context_window(window)
        
        # 窗口内循环处理
        iteration = 0
        while iteration < config.max_iterations_per_window:
            iteration += 1
            logger.info(f"  窗口内第 {iteration} 次处理")
            
            # 构建prompt
            prompt = build_prompt(history_actions, qa_list, context_window)
            
            # 调用LLM
            try:
                messages = [{"role": "system", "content": "你是一个有用的助手。"},{"role": "user", "content": prompt}]
                response = call_llm(messages).choices[0].message.content
                logger.debug(f"LLM输出：{response}")
            except Exception as e:
                logger.error(f"  调用LLM出错: {e}")
                break
            
            # 解析响应，同时提取思考过程和动作
            thinking_process, action_data = extract_response_parts(response)
            if action_data is None:
                logger.warning("  无法解析LLM响应，跳到下一个窗口")
                # 即使解析失败，也要保存原始响应以便调试
                error_record = {
                    "thinking_process": thinking_process if thinking_process else response,
                    "action_data": None,
                    "parsing_error": True,
                    "raw_response": response
                }
                full_actions.append(error_record)
                break
            
            action_type = action_data.get("action", "unknown")
            logger.info(f"  动作: {action_type}")
            
            # 更新数据结构（现在包含思考过程）
            continue_window = update_data_structures(
                action_data, thinking_process, history_actions, qa_list, full_actions
            )
            
            if not continue_window:
                break
        
        if iteration >= config.max_iterations_per_window:
            logger.warning(f"窗口 {window_idx + 1} 达到最大迭代次数限制")
    
    result = {
        "filename": filename,
        "total_windows": len(windows),
        "total_iterations": len(full_actions),
        "qa_count": len(qa_list),
        "qa_list": qa_list,
        "full_actions": full_actions,  # 现在包含思考过程
        "processing_config": {
            "window_length": config.window_length,
            "max_iterations_per_window": config.max_iterations_per_window,
            "model_name": config.model_name
        }
    }
        
    logger.info(f"文件处理完成，共提取 {len(qa_list)} 个QA对，执行了 {len(full_actions)} 次动作")
    return result

def save_results(result: Dict, output_folder: str):
    """
    保存处理结果
    """
    os.makedirs(output_folder, exist_ok=True)
    
    filename = result["filename"]
    output_filename = filename.replace('.json', '_processed.json')
    output_path = os.path.join(output_folder, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果保存到: {output_path}")

def main(config: ProcessingConfig):
    """
    主函数
    """
    logger.info("开始ASR文本处理任务")
    logger.info(f"配置: {config}")
    
    # 检查call_llm函数是否可用
    if 'call_llm' not in globals():
        logger.error("请确保已定义 call_llm 函数")
        return
    
    # 加载所有ASR文件
    asr_data = load_asr_files(config.input_folder)
    if not asr_data:
        logger.error("没有找到可处理的ASR文件")
        return
    
    logger.info(f"加载了 {len(asr_data)} 个文件")
    
    # 处理每个文件
    for filename, asr_text in asr_data.items():
        try:
            result = process_single_file(filename, asr_text, config, call_llm)
            save_results(result, config.output_folder)
        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {e}")
            continue

if __name__ == "__main__":
    # 配置参数
    config = ProcessingConfig(
        input_folder="/data3/liangyaozhen/vvmz/text_v0/doc_official_slice",
        output_folder="/data3/liangyaozhen/vvmz/text_v1/extraction_v0", 
        window_length=1536,
        max_iterations_per_window=10,
        model_name="deepseek-chat"
    )
    
    # 运行主程序
    main(config)