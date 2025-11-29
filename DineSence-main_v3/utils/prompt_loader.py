# utils/prompt_loader.py

import os
from functools import lru_cache

# 取得 prompts 資料夾的絕對路徑
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts')

@lru_cache(maxsize=32) # 快取讀取過的 prompt，避免重複讀取檔案
def load_prompt_template(feature_name: str, prompt_type: str = "user") -> str:
    """
    從 prompts 資料夾載入指定的 prompt 模板。

    Args:
        feature_name (str): 功能名稱 (對應到資料夾名稱，例如 'summarize_session')
        prompt_type (str): 'user' 或 'system'

    Returns:
        str: 讀取到的 prompt 文字內容
    """
    file_path = os.path.join(PROMPTS_DIR, feature_name, f"{prompt_type}.txt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 Prompt 檔案: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()