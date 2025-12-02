# ui/live_view.py

import streamlit as st
import cv2
import time
from collections import Counter
from datetime import datetime
import numpy as np

from core.live_analyzer import LiveAnalyzer
from services.database import save_session 

def display(model_pack: dict, backend_config: dict, db_manager, t=None):
    """
    Live 監控介面 - 雙鏡頭戰情室 (Dual Camera Command Center)
    """
    if t is None:
        def t(k): return k
    
    # --- 1. Session State 初始化 ---
    if "current_raw_session_id" not in st.session_state:
        st.session_state.current_raw_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    if "live_toggle_last_state" not in st.session_state: st.session_state.live_toggle_last_state = False
    if "analyzer" not in st.session_state: st.session_state.analyzer = None
    
    # 統計變數
    if "nod_count" not in st.session_state: st.session_state.nod_count = 0
    if "shake_count" not in st.session_state: st.session_state.shake_count = 0
    if "emotion_counter" not in st.session_state: st.session_state.emotion_counter = Counter()
    if "leftover_counter" not in st.session_state: st.session_state.leftover_counter = Counter()
    
    # 顯示變數
    if "last_plate_insight" not in st.session_state: st.session_state.last_plate_insight = "Waiting for VLM Analysis..."
    if "session_start_time" not in st.session_state: st.session_state.session_start_time = None
    if "last_display_emotion" not in st.session_state: st.session_state.last_display_emotion = "---"
    
    # ★★★ Log Buffer 初始化 (找回原本的 Log) ★★★
    if "live_log_buffer" not in st.session_state: st.session_state.live_log_buffer = ["System Ready..."]

    # --- 2. 頂部 HUD ---
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1:
            st.markdown(f"<span style='color:var(--primary-color)'><h3>🔴 LIVE MONITORING</h3></span>", unsafe_allow_html=True)
            st.caption(f"SESSION ID: {st.session_state.current_raw_session_id}")
        
        metric_people_ph = c2.empty()
        metric_sat_ph = c3.empty()
        metric_event_ph = c4.empty()
        
        metric_people_ph.metric(t("metric_people"), "0")
        metric_sat_ph.metric("Nods/Shakes", "0 / 0")
        metric_event_ph.metric("Emotion", "---")

    st.write("") 

    # --- 3. 主畫面佈局：雙螢幕 ---
    col_face, col_plate = st.columns(2)

    with col_face:
        st.markdown("#### 👤 Customer (Cam 0)")
        face_video_ph = st.empty()
        face_video_ph.markdown(
            f"""<div style='background-color:#000; height:360px; display:flex; 
            align-items:center; justify-content:center; color:#666; border-radius:10px;'>
            <h3>Waiting for Camera 0...</h3></div>""", unsafe_allow_html=True)
        
        # ★★★ 恢復日誌顯示區 ★★★
        with st.container(border=True):
            st.markdown("**System Log:**")
            log_placeholder = st.empty()
            log_placeholder.code("\n".join(st.session_state.live_log_buffer), language="bash")

    with col_plate:
        st.markdown("#### 🍽️ Plate (Cam 1)")
        plate_video_ph = st.empty()
        plate_video_ph.markdown(
            f"""<div style='background-color:#000; height:360px; display:flex; 
            align-items:center; justify-content:center; color:#666; border-radius:10px;'>
            <h3>Waiting for Camera 1...</h3></div>""", unsafe_allow_html=True)
        
        # AI Insight 區域
        with st.container(border=True):
            st.markdown("**AI Insight (VLM):**")
            vlm_insight_ph = st.empty()
            vlm_insight_ph.info(st.session_state.last_plate_insight)

    # --- 4. 控制區 ---
    with st.container(border=True):
        col_btn1, col_btn2, col_status = st.columns([1, 1, 4])
        with col_btn1:
            start_btn = st.button("▶ START", type="primary", use_container_width=True)
        with col_btn2:
            stop_btn = st.button("⏹ STOP", type="secondary", use_container_width=True)
        with col_status:
            status_ph = st.empty()

    # --- 5. 邏輯控制 ---
    current_toggle_state = st.session_state.get('live_toggle_active', False)
    
    if start_btn:
        current_toggle_state = True
        st.session_state['live_toggle_active'] = True
    
    if stop_btn:
        current_toggle_state = False
        st.session_state['live_toggle_active'] = False

    # (A) 啟動 Analyzer
    if current_toggle_state and st.session_state.analyzer is None:
        status_ph.success("Dual Camera System Active")
        
        st.session_state.nod_count = 0
        st.session_state.shake_count = 0
        st.session_state.emotion_counter = Counter()
        st.session_state.leftover_counter = Counter()
        st.session_state.last_plate_insight = "Waiting for AI Analysis..." 
        st.session_state.live_log_buffer = ["System Initialized."]
        
        analysis_options = { "opt_nod": True, "opt_emote": True, "opt_plate": True }
        
        st.session_state.analyzer = LiveAnalyzer(model_pack, [], analysis_options, db_manager)
        st.session_state.analyzer.start()
        st.session_state.current_raw_session_id = st.session_state.analyzer.raw_session_id
        
        st.session_state.session_start_time = datetime.now()

    # (B) 停止 Analyzer
    if not current_toggle_state and st.session_state.analyzer:
        status_ph.warning("Saving Session Data...")
        st.session_state.analyzer.stop()
        st.session_state.analyzer = None
        
        end_time = datetime.now()
        start_time = st.session_state.session_start_time
        duration = (end_time - start_time).total_seconds() if start_time else 0
        
        save_session(
            raw_session_id=st.session_state.current_raw_session_id,
            mode="Dual_Camera_Mode",
            duration=int(duration),
            nod=st.session_state.nod_count,
            shake=st.session_state.shake_count,
            emotion_dict=dict(st.session_state.emotion_counter),
            leftover_dict=dict(st.session_state.leftover_counter),
            insight=st.session_state.last_plate_insight
        )
        status_ph.info(f"Session Saved. Duration: {int(duration)}s")

    # --- 6. 更新迴圈 ---
    MAX_LOG_LINES = 6
    log_buffer = st.session_state.live_log_buffer

    if current_toggle_state and st.session_state.analyzer:
        while True:
            f_frame, p_frame = st.session_state.analyzer.get_latest_frames()
            result = st.session_state.analyzer.get_latest_analysis_result()
            
            if f_frame is None and p_frame is None:
                time.sleep(0.05); continue

            # 更新畫面 (Cam 0)
            if f_frame is not None:
                cv2.putText(f_frame, f"Nod: {st.session_state.nod_count}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(f_frame, f"Emo: {st.session_state.last_display_emotion}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                face_video_ph.image(cv2.cvtColor(f_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 更新畫面 (Cam 1)
            if p_frame is not None:
                # 安全讀取 plate_event
                p_label = getattr(result, "plate_event", None)
                if p_label:
                     cv2.putText(p_frame, f"{p_label}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                plate_video_ph.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 處理數據與更新 Log
            if result:
                event_str = ""
                ts = datetime.now().strftime("%H:%M:%S")
                
                if result.nod_event: 
                    st.session_state.nod_count += 1
                    event_str += " [NOD]"
                if getattr(result, "shake_event", False): 
                    st.session_state.shake_count += 1
                    event_str += " [SHAKE]"
                if result.emotion_event:
                    st.session_state.emotion_counter[result.emotion_event] += 1
                    st.session_state.last_display_emotion = result.emotion_event
                    event_str += f" [{result.emotion_event}]" # 情緒太頻繁，不一定要寫入Log
                
                # 安全讀取 plate_insight (這裡已經用 getattr 防呆)
                insight = getattr(result, "plate_insight", None)
                if insight:
                    st.session_state.last_plate_insight = insight
                    event_str += " [VLM Report]"

                # 如果有事件，寫入 Log Buffer
                if event_str:
                    log_buffer.append(f"[{ts}]{event_str}")
                    if len(log_buffer) > MAX_LOG_LINES: log_buffer.pop(0)
                    st.session_state.live_log_buffer = log_buffer
                    log_placeholder.code("\n".join(log_buffer), language="bash")
                
                vlm_insight_ph.info(st.session_state.last_plate_insight)

                # 更新 HUD
                display_info = result.display_info
                metric_people_ph.metric(t("metric_people"), display_info.get("people_count", 0))
                metric_sat_ph.metric("Nods / Shakes", f"{st.session_state.nod_count} / {st.session_state.shake_count}")
                metric_event_ph.metric("Emotion", st.session_state.last_display_emotion)

            time.sleep(0.03)