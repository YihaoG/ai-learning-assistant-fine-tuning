import os
import random
import re
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from prompts import TEXT2SPEAKER_SPLIT_RULE_SYS,TEXT2SPEAKER_SPLIT_RULE_USER,AGGREGATE_RULES_SYS,AGGREGATE_RULES_USER
from utils import call_llm

def extract_rules_from_response(response_text):
    """从LLM响应中提取分割规则"""
    # 使用正则表达式提取<分割规则>标签中的内容
    pattern = r'<分割规则>(.*?)</分割规则>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # 如果找不到标签，尝试其他可能的模式
        # 例如，查找"分割规则："后面的内容
        alt_pattern = r'分割规则：\s*(.*?)(?:\n\n|$)'
        alt_match = re.search(alt_pattern, response_text, re.DOTALL)
        if alt_match:
            return alt_match.group(1).strip()
    
    # 如果仍然找不到，返回整个响应
    return "未能提取到明确的规则，原始响应：\n" + response_text

def process_folder(folder_path, output_path, samples_per_folder=5):
    """处理指定文件夹中的txt文件"""
    folder_path = Path(folder_path)
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = list(folder_path.glob("*.txt"))
    
    if not txt_files:
        print(f"警告: 文件夹 {folder_path} 中没有找到txt文件")
        return []
    
    # 随机选择指定数量的文件
    selected_files = random.sample(txt_files, min(samples_per_folder, len(txt_files)))
    
    results = []
    
    # 处理每个选定的文件
    for file_path in selected_files:
        print(f"处理文件: {file_path}")
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                asr_text = f.read()
            
            # 构建消息
            user_prompt = TEXT2SPEAKER_SPLIT_RULE_USER.replace("{{asr_text}}", asr_text)
            messages = [
                {"role": "system", "content": TEXT2SPEAKER_SPLIT_RULE_SYS},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用LLM
            response = call_llm(messages)
            
            if response and response.choices:
                # 提取响应内容
                llm_response = response.choices[0].message.content
                
                # 提取规则
                rules = extract_rules_from_response(llm_response)
                
                # 保存结果
                result = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "timestamp": datetime.now().isoformat(),
                    "rules": rules,
                    "full_response": llm_response
                }
                
                results.append(result)
                
                # 生成唯一的输出文件名
                output_file = output_path / f"rules_{folder_path.name}_{file_path.stem}.json"
                
                # 保存结果到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"已保存规则到: {output_file}")
            
            else:
                print(f"处理文件失败: {file_path}")
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return results

def process_multiple_folders(folder_paths, output_base_path, samples_per_folder=5):
    """处理多个文件夹"""
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 为此次运行创建一个输出目录
    run_output_path = Path(output_base_path) / f"speaker_split_rules_{timestamp}"
    run_output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理每个文件夹
    for folder_path in folder_paths:
        folder = Path(folder_path)
        folder_name = folder.name
        
        print(f"\n开始处理文件夹: {folder_path}")
        
        # 为每个文件夹创建子目录
        folder_output_path = run_output_path / folder_name
        folder_output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理文件夹
        results = process_folder(folder_path, folder_output_path, samples_per_folder)
        
        all_results[folder_name] = results
    
    # 保存所有结果的汇总
    summary_path = run_output_path / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成，结果汇总保存在: {summary_path}")
    
    return run_output_path, all_results

def aggregate_rules(output_path):
    """汇总所有规则并创建总结"""
    all_rules = []
    output_path = Path(output_path)
    
    # 收集所有规则文件
    rule_files = list(output_path.glob("**/*rules_*.json"))
    
    for file_path in rule_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_rules.append(data["rules"])
        except Exception as e:
            print(f"读取规则文件 {file_path} 时出错: {e}")
    
    # 将所有规则组合成一个文本
    combined_rules = "\n\n---\n\n".join(all_rules)
    
    # 构建提示，让LLM总结规则
    messages = [
        {"role": "system", "content": AGGREGATE_RULES_SYS},
        {"role": "user", "content": AGGREGATE_RULES_USER.replace('{{combined_rules}}',combined_rules)}
    ]
    
    # 调用LLM
    response = call_llm(messages)
    
    if response and response.choices:
        summary = response.choices[0].message.content
        
        # 保存总结
        summary_path = output_path / "final_rules_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"规则总结已保存到: {summary_path}")
        return summary
    else:
        print("生成规则总结失败")
        return None

# 主函数
def main(folder_paths, output_base_path="./speaker_split_results", samples_per_folder=5):
    """主函数"""
    # 处理所有文件夹
    # output_path, results = process_multiple_folders(
    #     folder_paths, 
    #     output_base_path, 
    #     samples_per_folder
    # )
    
    # # 汇总所有规则
    output_path='/data3/liangyaozhen/vvmz/text_v0/speaker_split_rules/speaker_split_rules_20250615_234627'
    final_rules = aggregate_rules(output_path)
    
    print("\n处理完成!")
    return output_path, final_rules

# 使用示例
if __name__ == "__main__":
    # 配置参数
    folder_paths = [
        "/data3/liangyaozhen/vvmz/raw_text",
        "/data3/liangyaozhen/vvmz/raw_text/TYTtxt",
        "/data3/liangyaozhen/vvmz/raw_text/鱼折txt",
        '/data3/liangyaozhen/vvmz/raw_text/空-BaldrSkytxt',
        '/data3/liangyaozhen/vvmz/raw_text/创伤性分离txt',
    ]
    output_base_path = "/data3/liangyaozhen/vvmz/text_v0/speaker_split_rules"
    samples_per_folder = 4
    
    # 运行主函数
    output_path, final_rules = main(folder_paths, output_base_path, samples_per_folder)