import os
import random
import re
import json
import yaml
from pathlib import Path
from datetime import datetime
from openai import OpenAI

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


# 使用示例
if __name__ == "__main__":
    resp = call_llm([
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "Python有哪些优点？"}
    ])

    if resp:
        print(resp.choices[0].message.content)
