# ui/live_view.py

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from services import vision_analysis as va
from core import live_analyzer

def display(model_pack, config, db_manager, t=None):
    """
    Live 監控介面 - 戰情室風格 (Command Center Layout)
    """
    # 為了相容性，若沒傳入 t (翻譯函式)，給一個預設的
    if t is None:
        def t(k): return k

    # --- 上方控制列 (HUD) ---
    # 使用容器包裝，創造儀表板頂部的感覺
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1:
            st.markdown(f"### 🥂 {t('live_status_active')}")
            st.caption(f"SESSION ID: {int(time.time())}")
        
        # 預留位置給即時數據 (這些會在迴圈中更新)
        with c2:
            metric_people_ph = st.empty()
            metric_people_ph.metric(t("metric_people"), "0", border=True)
        with c3:
            metric_sat_ph = st.empty()
            metric_sat_ph.metric(t("metric_satisfaction"), "0%", border=True)
        with c4:
            metric_event_ph = st.empty()
            metric_event_ph.metric(t("metric_events"), "0", border=True)

    st.write("") # 間距

    # --- 主佈局：左側影像 (3) vs 右側資訊流 (1.2) ---
    main_col, info_col = st.columns([3, 1.2])

    with main_col:
        # 影像顯示區塊
        with st.container(border=True):
            video_placeholder = st.empty()
            # 預設顯示一張待機圖或黑底
            video_placeholder.markdown(
                f"""
                <div style='background-color:#000; height:450px; display:flex; 
                align-items:center; justify-content:center; color:#c18440; border-radius:8px;'>
                    <h3>{t("waiting")}</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )

    with info_col:
        # 右側控制與日誌區
        with st.container(border=True):
            st.markdown(f"#### ⚙️ {t('settings')}")
            camera_source = st.radio(
                t("cam_input"), 
                options=[0, 1, "RTSP"], 
                horizontal=True,
                label_visibility="collapsed"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                start_btn = st.button(t("start_btn"), type="primary", use_container_width=True)
            with col_btn2:
                stop_btn = st.button(t("stop_btn"), type="secondary", use_container_width=True)

        st.write("")
        
        # 即時日誌 (模擬終端機效果)
        with st.container(border=True):
            st.markdown(f"#### {t('log_title')}")
            log_placeholder = st.empty()
            log_placeholder.code("System Ready...\nWaiting for input...", language="bash")
            
        # AI 洞察區
        with st.container(border=True):
            st.markdown(f"#### 🧠 {t('ai_insight')}")
            insight_placeholder = st.empty()
            insight_placeholder.info("AI Analysis Module Standby")

    # --- 邏輯處理 (維持原本邏輯，對接 UI Placeholder) ---
    if start_btn:
        st.session_state['is_running'] = True
    if stop_btn:
        st.session_state['is_running'] = False

    if st.session_state.get('is_running'):
        cap = cv2.VideoCapture(camera_source)
        
        # 用於累積 Log 的列表
        log_buffer = []
        MAX_LOG_LINES = 8
        event_count = 0

        while cap.isOpened() and st.session_state['is_running']:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read camera feed.")
                break

            # 1. 分析畫面 (調用 core 邏輯)
            processed_frame, frame_data = live_analyzer.process_frame(
                frame, 
                model_pack, 
                st.session_state['db_manager']
            )
            
            # 2. 更新畫面 (左側大圖)
            # 將 BGR 轉 RGB 以供 Streamlit 顯示
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True, channels="RGB")

            # 3. 更新 HUD 數據 (上方)
            ppl_count = frame_data.get('people_count', 0)
            metric_people_ph.metric(t("metric_people"), f"{ppl_count}")
            
            # 模擬滿意度計算 (這裡可用 frame_data 裡的真實數據)
            nods = frame_data.get('nod_detected', False)
            shakes = frame_data.get('shake_detected', False)
            
            if nods or shakes:
                event_count += 1
                metric_event_ph.metric(t("metric_events"), f"{event_count}")
                
                # 更新日誌
                timestamp = datetime.now().strftime("%H:%M:%S")
                event_type = "NOD (Positive)" if nods else "SHAKE (Negative)"
                log_msg = f"[{timestamp}] DETECTED: {event_type}"
                log_buffer.append(log_msg)
                if len(log_buffer) > MAX_LOG_LINES:
                    log_buffer.pop(0)
                
                # 刷新 Log 顯示
                log_text = "\n".join(log_buffer)
                log_placeholder.code(log_text if log_text else "Monitoring...", language="bash")

            # 4. 更新 AI 洞察 (如果有)
            if 'ai_insight' in frame_data and frame_data['ai_insight']:
                 insight_placeholder.success(frame_data['ai_insight'])

            time.sleep(0.03) # 控制 FPS

        cap.release()
        st.info("System Stopped.")