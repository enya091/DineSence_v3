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
    """
    【新版】用於存放單次背景分析結果的「純數據」結構。
    這個物件不包含影像，只包含分析出的事件和資訊，
    會在背景的分析執行緒和主執行緒之間傳遞。
    """
    # --- 分析出的「事件」(Event) ---
    # 這些是用於觸發主執行緒中計數器更新的信號
    nod_event: bool = False
    emotion_event: str = ""
    plate_event: str = ""
    token_usage_event: Optional[Dict[str, Any]] = None
    
    # --- 顯示在畫面上的「資訊」(Info) ---
    # 這些是純粹用來繪製在畫面上的文字或圖形座標
    display_info: Dict = field(default_factory=dict)