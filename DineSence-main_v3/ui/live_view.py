# ui/live_view.py

import streamlit as st
import cv2
import time
from collections import Counter
from datetime import datetime

from core.live_analyzer import LiveAnalyzer
# 確保導入的是模組級別的 wrapper 函式，以便直接呼叫
from services.database import save_session 
from services import llm_handler as llm

def display(model_pack: dict, backend_config: dict, db_manager, t=None):
    """
    Live 監控介面 - 戰情室風格 (Command Center Layout)
    """
    if t is None:
        def t(k): return k
    
    # --- 1. Session State 初始化防呆 (保持原樣) ---
    if "current_raw_session_id" not in st.session_state:
        st.session_state.current_raw_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    if "live_toggle_last_state" not in st.session_state: st.session_state.live_toggle_last_state = False
    if "analyzer" not in st.session_state: st.session_state.analyzer = None
    if "nod_count" not in st.session_state: st.session_state.nod_count = 0
    if "shake_count" not in st.session_state: st.session_state.shake_count = 0
    if "emotion_counter" not in st.session_state: st.session_state.emotion_counter = Counter()
    if "leftover_counter" not in st.session_state: st.session_state.leftover_counter = Counter()
    if "last_plate_insight" not in st.session_state: st.session_state.last_plate_insight = t("live_status_inactive")
    if "session_start_time" not in st.session_state: st.session_state.session_start_time = None
    if "last_display_emotion" not in st.session_state: st.session_state.last_display_emotion = "---"
    if "session_history" not in st.session_state: st.session_state.session_history = [] 

    # --- 2. 佈局定義與 HUD Metrics ---
    # 使用 container 打造頂部 HUD 樣式
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1:
            st.markdown(f"<span style='color:var(--primary-color)'><h3>{t('live_status_active')}</h3></span>", unsafe_allow_html=True)
            st.caption(f"SESSION ID: {st.session_state.current_raw_session_id}")
        
        metric_people_ph = c2.empty(); metric_sat_ph = c3.empty(); metric_event_ph = c4.empty()
        metric_people_ph.metric(t("metric_people"), "0"); metric_sat_ph.metric(t("metric_satisfaction"), "0"); metric_event_ph.metric(t("metric_events"), "0")

    st.write("") 

    # --- 3. 主佈局：左側影像 vs 右側資訊流 ---
    main_col, info_col = st.columns([3, 1.2])
    
    # 預留右側即時資訊的 Placeholder
    with info_col:
        # 3.1. 控制面板
        with st.container(border=True):
            st.markdown(f"#### ⚙️ {t('settings')}")
            # ★ 修正：將 radio 放在這裡，避免衝突 ★
            # camera_source = st.radio(
            #     t("cam_input"), 
            #     options=[" "], 
            #     horizontal=True,
            #     key="cam_input_radio",
            #     label_visibility="collapsed"
            # )
            
            # 模式選擇
            analysis_mode = st.radio(
                "ANALYSIS MODE",
                ["🙂 顧客行為", "🍽️ 餐盤剩食"],
                index=1,
                label_visibility="collapsed"
            )
            
            # 模式切換邏輯 (保留原邏輯)
            opt_nod = False; opt_emote = False; opt_plate = False
            if analysis_mode == "🙂 顧客行為":
                st.caption("DETECTS: NOD, SHAKE, EMOTION")
                opt_nod = True; opt_emote = True
            else:
                st.caption("DETECTS: PLATE LEFTOVER, VLM INSIGHT")
                opt_plate = True
            analysis_options = { "opt_nod": opt_nod, "opt_emote": opt_emote, "opt_plate": opt_plate }

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                start_btn = st.button(t("start_btn"), type="primary", use_container_width=True)
            with col_btn2:
                stop_btn = st.button(t("stop_btn"), type="secondary", use_container_width=True)

        st.write("")
        
        # 3.2. 即時日誌 (Log)
        with st.container(border=True):
            st.markdown(f"#### {t('log_title')}")
            log_placeholder = st.empty()
            log_placeholder.code("System Ready...", language="bash")
            
        # 3.3. AI 洞察區
        with st.container(border=True):
            st.markdown(f"#### 🧠 {t('ai_insight')}")
            insight_placeholder = st.empty()
            insight_placeholder.info("AI Analysis Module Standby")


    # --- 4. Session Start/Stop 邏輯 ---
    current_toggle_state = st.session_state.get('live_toggle_active', False)
    
    # 如果點擊 START
    if start_btn:
        current_toggle_state = True
        st.session_state['live_toggle_active'] = True
    
    # 如果點擊 STOP
    if stop_btn:
        current_toggle_state = False
        st.session_state['live_toggle_active'] = False

    # (A) 啟動/重啟 Analyzer
    if current_toggle_state and st.session_state.analyzer is None:
        st.toast(f"Session Started: {analysis_mode}", icon="▶️")
        
        st.session_state.nod_count = 0; st.session_state.shake_count = 0
        st.session_state.emotion_counter = Counter()
        st.session_state.leftover_counter = Counter()
        st.session_state.last_plate_insight = "Waiting for AI Analysis..." 
        st.session_state.last_display_emotion = "---"
        
        st.session_state.analyzer = LiveAnalyzer(model_pack, [], analysis_options, db_manager)
        st.session_state.analyzer.start()
        st.session_state.current_raw_session_id = st.session_state.analyzer.raw_session_id
        st.session_state.session_start_time = datetime.now()
        log_buffer = ["System Initialized.", f"Mode: {analysis_mode}"]

    # (B) 停止 Analyzer
    if not current_toggle_state and st.session_state.analyzer:
        st.session_state.analyzer.stop()
        st.session_state.analyzer = None
        st.toast("Analysis Terminated. Saving data...", icon="💾")
        
        # 儲存 Session 紀錄 (原邏輯)
        end_time = datetime.now()
        start_time = st.session_state.session_start_time
        duration = (end_time - start_time).total_seconds() if start_time else 0
        final_insight = st.session_state.get("last_plate_insight", "No insight")
        
        save_session(
            raw_session_id=st.session_state.current_raw_session_id,
            mode=analysis_mode,
            duration=int(duration),
            nod=st.session_state.nod_count,
            shake=st.session_state.shake_count,
            emotion_dict=dict(st.session_state.emotion_counter),
            leftover_dict=dict(st.session_state.leftover_counter),
            insight=final_insight
        )


    # --- 5. 主畫面更新迴圈 (正確邏輯) ---
    with main_col:
        video_placeholder = st.empty()
        # 預設顯示黑色背景 (保持 Cyberpunk 風格)
        video_placeholder.markdown(
            f"""
            <div style='background-color:#000; height:450px; display:flex; 
            align-items:center; justify-content:center; color:var(--text-muted); border-radius:12px;'>
                <h3>{t("waiting")}</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )

    log_buffer = st.session_state.get("live_log_buffer", ["System Ready..."])
    MAX_LOG_LINES = 8
    
    if current_toggle_state and st.session_state.analyzer:
        while True:
            # ★★★ 修正的關鍵：從 Analyzer 實例中獲取結果 ★★★
            frame = st.session_state.analyzer.get_latest_frame()
            result = st.session_state.analyzer.get_latest_analysis_result()
            
            if frame is None:
                time.sleep(0.05); continue

            # --- 處理分析結果 ---
            if result:
                # 累計計數器 (原邏輯)
                if result.nod_event: st.session_state.nod_count += 1
                if getattr(result, "shake_event", False): st.session_state.shake_count += 1
                if result.emotion_event:
                    st.session_state.emotion_counter[result.emotion_event] += 1
                    st.session_state.last_display_emotion = result.emotion_event
                if result.plate_event:
                    st.session_state.leftover_counter[result.plate_event] += 1
                if getattr(result, "plate_insight", None):
                    st.session_state.last_plate_insight = result.plate_insight

                # 更新日誌 (Log)
                if result.nod_event or getattr(result, "shake_event", False) or result.plate_event:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    event_type = f"NOD:{st.session_state.nod_count}" if result.nod_event else ""
                    event_type += f" SHAKE:{st.session_state.shake_count}" if getattr(result, "shake_event", False) else ""
                    event_type += f" PLATE:{result.plate_event[:10]}" if result.plate_event else ""
                    log_msg = f"[{timestamp}] DETECTED: {event_type}"
                    log_buffer.append(log_msg)
                    if len(log_buffer) > MAX_LOG_LINES: log_buffer.pop(0)

            # --- 繪圖與數據顯示 ---
            display_info = result.display_info if result else {}
            
            # 繪製 HUD 數據
            total_events = st.session_state.nod_count + st.session_state.shake_count
            metric_people_ph.metric(t("metric_people"), display_info.get("people_count", 0))
            metric_event_ph.metric(t("metric_events"), total_events)
            
            # 刷新 Log
            log_placeholder.code("\n".join(log_buffer), language="bash")
            st.session_state["live_log_buffer"] = log_buffer # 存回 session
            
            # 繪製 CV 結果到畫面上
            # 由於我們現在是深色模式，使用白色字體 (255, 255, 255)
            cv2.putText(frame, f"Nod: {st.session_state.nod_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Shake: {st.session_state.shake_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {st.session_state.last_display_emotion}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True, channels="RGB")
            
            # 更新 AI Insight
            insight_placeholder.info(st.session_state.last_plate_insight)

            time.sleep(0.03)