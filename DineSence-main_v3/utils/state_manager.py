# utils/state_manager.py

"""
本模組專門用於管理 Streamlit 的 session_state。
"""

import streamlit as st
from collections import Counter

def initialize_state():
    """
    初始化應用程式所有分頁都會用到的 session_state。
    此函式應在主程式 app.py 的最開始被呼叫一次。
    """
    states = {
        # --- 登入與核心物件 ---
        "auth": False,
        "analyzer": None,  # 用於存放 LiveAnalyzer 引擎的實例
        
        # --- [新增] Session 歷史紀錄相關狀態 ---
        "session_history": [],          # 存放所有已結束的 session 記錄
        "session_start_time": None,     # 記錄當前 session 的開始時間
        "session_token_usage": Counter(), # 累加當前 session 的 token 消耗
        "live_toggle_last_state": False, # 用於偵測 toggle 按鈕的狀態變化
        
        # --- 即時統計計數器 (用於當前 session) ---
        "nod_count": 0,
        "emotion_counter": Counter(),
        "leftover_counter": Counter(),
    }

    for key, default_value in states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value