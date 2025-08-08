import os
import random
import re
import json
import yaml
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from typing import List

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Using simple character-based estimation.")

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    计算文本的token数量
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except KeyError:
            # 如果模型不存在，使用默认编码
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    else:
        # 简单估算：平均每4个字符约等于1个token
        return len(text) // 4


def load_config(config_path="./llm_api_config.yaml"):
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"警告: 配置文件 {config_path} 不存在，将使用默认配置")
        return {
            "BASE_URL": "",
            "DEFAULT_MODEL": "deepseek-r1",
            "REQUEST_TIMEOUT": 1000,
            "TEMPERATURE": 0.0,
            "API_KEY": "sk-demo-key"  # 默认值，实际使用时应替换
        }
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return {}

# 全局客户端实例
_client = None

def get_client(config=None):
    """获取或初始化OpenAI客户端"""
    global _client
    
    # 如果提供了新配置或客户端不存在，则初始化
    if _client is None or config is not None:
        # 如果没有提供配置，从YAML加载
        if config is None:
            config = load_config()
            
        _client = OpenAI(
            base_url=config.get("BASE_URL", ""),
            api_key=config.get("API_KEY", "sk-demo-key"),
            timeout=config.get("REQUEST_TIMEOUT", 1000)
        )
    
    return _client

def call_llm(messages, model=None, temperature=None, config=None):
    """调用LLM API的简单封装"""
    # 更新或获取配置
    if config is None:
        config = load_config()
    
    # 获取客户端
    client = get_client(config)
    
    # 设置参数
    model = model or config.get("DEFAULT_MODEL", "deepseek-r1")
    temperature = temperature if temperature is not None else config.get("TEMPERATURE", 0.0)
    
    # 调用API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=config.get("REQUEST_TIMEOUT", 1000),
            max_tokens=8192,
        )
        return response
    except Exception as e:
        print(f"LLM API调用失败: {e}")
        return None

# 其余代码保持不变...

    
def loadjson(filepath, encoding='utf-8', default=None):
    """
    从文件中加载JSON数据
    
    参数:
        filepath (str): JSON文件路径
        encoding (str): 文件编码，默认为utf-8
        default: 如果文件不存在或读取失败时返回的默认值
        
    返回:
        加载的JSON数据，或者默认值
    """
    try:
        if not os.path.exists(filepath):
            return default
        
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {filepath}, 错误: {str(e)}")
        return default

def savejson(data, filepath, encoding='utf-8', indent=2, ensure_dir=True):
    """
    将数据保存为JSON文件
    
    参数:
        data: 要保存的数据
        filepath (str): 保存的文件路径
        encoding (str): 文件编码，默认为utf-8
        indent (int): JSON缩进，默认为2
        ensure_dir (bool): 是否确保目录存在，默认为True
        
    返回:
        bool: 保存成功返回True，否则返回False
    """
    try:
        if ensure_dir:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {filepath}, 错误: {str(e)}")
        return False

def get_all_txt_files(directory: str) -> List[str]:
    """
    递归获取指定目录及其子目录下的所有txt文件
    
    Args:
        directory: 要搜索的目录路径
        
    Returns:
        所有txt文件的绝对路径列表
    """
    txt_files = []
    
    # 使用Path对象来处理路径，更加跨平台和安全
    base_dir = Path(directory)
    
    # 确保目录存在
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"错误: 目录 '{directory}' 不存在或不是一个目录")
        return []
    
    # 递归遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 筛选出所有txt文件
        for file in files:
            if file.lower().endswith('.txt'):
                # 构建完整路径并添加到列表
                full_path = Path(root) / file
                txt_files.append(str(full_path))
    
    # 按照路径排序，让输出更有序
    txt_files.sort()
    
    print(f"共找到 {len(txt_files)} 个txt文件")
    return txt_files

# 使用示例
if __name__ == "__main__":
    resp = call_llm([
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "Python有哪些优点？"}
    ])

    if resp:
        print(resp.choices[0].message.content)
