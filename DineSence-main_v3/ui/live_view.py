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
    if "live_toggle_last_state" not in st.session_state:
        st.session_state.live_toggle_last_state = False
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    
    # 統計變數
    if "nod_count" not in st.session_state:
        st.session_state.nod_count = 0
    if "shake_count" not in st.session_state:
        st.session_state.shake_count = 0
    if "emotion_counter" not in st.session_state:
        st.session_state.emotion_counter = Counter()
    if "leftover_counter" not in st.session_state:
        st.session_state.leftover_counter = Counter()
    
    # 顯示變數
    if "last_plate_insight" not in st.session_state:
        # 初始顯示：等待 AI 分析
        st.session_state.last_plate_insight = "等待 AI 分析結果……"
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = None
    if "last_display_emotion" not in st.session_state:
        st.session_state.last_display_emotion = "---"
    
    # ★★★ Log Buffer 初始化 ★★★
    if "live_log_buffer" not in st.session_state:
        st.session_state.live_log_buffer = ["系統就緒……"]

    # --- 2. 頂部 HUD ---
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1:
            st.markdown(
                f"<span style='color:var(--primary-color)'><h3>{t('live_title')}</h3></span>",
                unsafe_allow_html=True
            )
            st.caption(f"{t('session_id_label')}：{st.session_state.current_raw_session_id}")
        
        metric_people_ph = c2.empty()
        metric_sat_ph = c3.empty()
        metric_event_ph = c4.empty()
        
        metric_people_ph.metric(t("metric_people"), "0")
        metric_sat_ph.metric(t("metric_nods_shakes"), "0 / 0")
        metric_event_ph.metric(t("metric_emotion"), "---")

    st.write("") 

    # --- 3. 主畫面佈局：雙螢幕 ---
    col_face, col_plate = st.columns(2)

    with col_face:
        st.markdown(f"#### {t('live_cam_face')}")
        face_video_ph = st.empty()
        face_video_ph.markdown(
            f"""
            <div style='background-color:#000; height:360px; display:flex; 
                 align-items:center; justify-content:center; color:#666; border-radius:10px;'>
                <h3>{t('waiting_cam')}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # ★★★ 日誌顯示區 ★★★
        with st.container(border=True):
            st.markdown(f"**{t('log_title')}**")
            log_placeholder = st.empty()
            log_placeholder.code("\n".join(st.session_state.live_log_buffer), language="bash")

    with col_plate:
        st.markdown(f"#### {t('live_cam_plate')}")
        plate_video_ph = st.empty()
        plate_video_ph.markdown(
            f"""
            <div style='background-color:#000; height:360px; display:flex; 
                 align-items:center; justify-content:center; color:#666; border-radius:10px;'>
                <h3>{t('waiting_cam')}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # AI Insight 區域
        with st.container(border=True):
            st.markdown(f"**{t('ai_insight')}**")
            vlm_insight_ph = st.empty()
            vlm_insight_ph.info(st.session_state.last_plate_insight)

    # --- 4. 控制區 ---
    with st.container(border=True):
        col_btn1, col_btn2, col_status = st.columns([1, 1, 4])
        with col_btn1:
            start_btn = st.button(f"▶ {t('start_btn')}", type="primary", use_container_width=True)
        with col_btn2:
            stop_btn = st.button(f"⏹ {t('stop_btn')}", type="secondary", use_container_width=True)
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
        status_ph.success("雙鏡頭系統已啟動")
        
        st.session_state.nod_count = 0
        st.session_state.shake_count = 0
        st.session_state.emotion_counter = Counter()
        st.session_state.leftover_counter = Counter()
        st.session_state.last_plate_insight = "等待 AI 分析……"
        st.session_state.live_log_buffer = ["系統初始化完成。"]
        
        analysis_options = { "opt_nod": True, "opt_emote": True, "opt_plate": True }
        
        st.session_state.analyzer = LiveAnalyzer(model_pack, [], analysis_options, db_manager)
        st.session_state.analyzer.start()
        st.session_state.current_raw_session_id = st.session_state.analyzer.raw_session_id
        
        st.session_state.session_start_time = datetime.now()

    # (B) 停止 Analyzer
    if not current_toggle_state and st.session_state.analyzer:
        
        # ★★★ [新增] 離席前的強制快照流程 ★★★
        # 使用 spinner 告訴使用者系統正在運作，請勿關閉
        with st.spinner("📸 顧客離席中：正在拍攝最終餐盤狀態並進行 AI 判斷..."):
            
            # 1. 呼叫後端執行拍攝與分析
            final_path, final_label, final_ratio = st.session_state.analyzer.snapshot_final_plate()
            
            if final_path:
                # 2. 顯示成功訊息與判斷結果
                st.success(f"✅ 已紀錄離席畫面！最終判斷：{final_label}", icon="🍽️")
                
                # 3. [重要] 將這個最終判斷加入統計
                # 這是為了讓這張照片的權重被計入 Dashboard
                st.session_state.leftover_counter[final_label] += 1
                
                # 4. 更新介面上的 Insight 顯示
                st.session_state.last_plate_insight = f"【結帳離席】最終狀態判定為：{final_label}"
            else:
                st.error("⚠️ 無法拍攝最終畫面，請檢查鏡頭連線")
            
            # 稍微停頓一下讓使用者看到訊息 (0.5秒)
            time.sleep(0.5)

        # --- 以下是原本的停止與存檔邏輯 (保持不變) ---
        status_ph.warning("正在儲存此次紀錄……")
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
        status_ph.info(f"紀錄已儲存，總時長：{int(duration)} 秒")

    # --- 6. 更新迴圈 ---
    MAX_LOG_LINES = 6
    log_buffer = st.session_state.live_log_buffer

    if current_toggle_state and st.session_state.analyzer:
        while True:
            f_frame, p_frame = st.session_state.analyzer.get_latest_frames()
            result = st.session_state.analyzer.get_latest_analysis_result()
            
            if f_frame is None and p_frame is None:
                time.sleep(0.05)
                continue

            # 更新畫面 (鏡頭 0：顧客)
            if f_frame is not None:
                cv2.putText(
                    f_frame,
                    f"點頭: {st.session_state.nod_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    f_frame,
                    f"情緒: {st.session_state.last_display_emotion}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
                face_video_ph.image(cv2.cvtColor(f_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 更新畫面 (鏡頭 1：餐盤)
            if p_frame is not None:
                # 安全讀取 plate_event
                p_label = getattr(result, "plate_event", None) if result is not None else None
                if p_label:
                    cv2.putText(
                        p_frame,
                        f"{p_label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                plate_video_ph.image(cv2.cvtColor(p_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 處理數據與更新 Log
            if result:
                event_str = ""
                ts = datetime.now().strftime("%H:%M:%S")
                
                if result.nod_event: 
                    st.session_state.nod_count += 1
                    event_str += " [點頭]"
                if getattr(result, "shake_event", False): 
                    st.session_state.shake_count += 1
                    event_str += " [搖頭]"
                if result.emotion_event:
                    st.session_state.emotion_counter[result.emotion_event] += 1
                    st.session_state.last_display_emotion = result.emotion_event
                    event_str += f" [{result.emotion_event}]"
                
                # 安全讀取 plate_insight
                insight = getattr(result, "plate_insight", None)
                if insight:
                    st.session_state.last_plate_insight = insight
                    event_str += " [餐盤報告]"

                # 如果有事件，寫入 Log Buffer
                if event_str:
                    log_buffer.append(f"[{ts}]{event_str}")
                    if len(log_buffer) > MAX_LOG_LINES:
                        log_buffer.pop(0)
                    st.session_state.live_log_buffer = log_buffer
                    log_placeholder.code("\n".join(log_buffer), language="bash")
                
                vlm_insight_ph.info(st.session_state.last_plate_insight)

                # 更新 HUD
                display_info = result.display_info
                metric_people_ph.metric(t("metric_people"), display_info.get("people_count", 0))
                metric_sat_ph.metric(
                    t("metric_nods_shakes"),
                    f"{st.session_state.nod_count} / {st.session_state.shake_count}"
                )
                metric_event_ph.metric(t("metric_emotion"), st.session_state.last_display_emotion)

            time.sleep(0.03)
