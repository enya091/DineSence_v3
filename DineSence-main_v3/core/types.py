# core/types.py
"""
本檔案定義了整個專案中可以共用的數據結構。
使用 dataclasses 可以讓結構更清晰，並提供自動的初始化等方法
這有助於減少因為打錯字典鍵 (key) 而造成的錯誤。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

# --- 舊的 FrameResult 已被移除 ---

@dataclass
class AnalysisResult:
    def __init__(self):
        # 既有的欄位
        self.nod_event = False
        self.shake_event = False
        self.emotion_event = None
        self.plate_event = None
        
        # ★★★ 新增這兩個缺少的欄位 ★★★
        self.plate_insight = None      # 修正 AttributeError
        self.token_usage_event = None  # 避免未來報錯
        
        self.display_info = {
            "people_count": 0,
            "plate_label": "",
            "plate_circle": None,
            "food_detections": []
        }