# ui/dashboard_view.py

import streamlit as st
import pandas as pd
import json
import asyncio
import datetime
import ast
import os
from collections import Counter
from services import llm_handler as llm
from services.database import DatabaseManager 
import plotly.express as px

EVIDENCE_DIR = "session_evidence"

# 輔助函式：圖片 Grid
def _render_evidence_grid(db_manager, session_id, event_type):
    """
    負責讀取並顯示圖片 Grid，不包含外層的 Expander。
    """
    evidence_df = db_manager.get_event_evidence(session_id, event_type)
    
    if evidence_df.empty:
        st.info("NO EVIDENCE FOUND.", icon="ℹ️")
        return

    # 限制顯示數量，避免一次載入太多當機
    cols = st.columns(4) 
    
    for i, row in evidence_df.iterrows():
        filename = os.path.basename(row['local_path'])
        path = os.path.join(EVIDENCE_DIR, filename)
        
        evidence_id = row['id']
        is_correct = row['human_corrected']
        
        # 決定圖片放在哪一欄
        col = cols[i % 4]
        
        if os.path.exists(path):
            with col:
                st.image(path, use_container_width=True)
                
                checkbox_key = f"img_feedback_{evidence_id}_{session_id}"
                
                def update_feedback(eid=evidence_id, key=checkbox_key):
                    new_state = st.session_state[key]
                    db_manager.update_evidence_feedback(eid, new_state)
                    if new_state:
                        st.toast(f"✅ ID {eid} CONFIRMED", icon="👍")

                st.checkbox(
                    f"#{evidence_id} CONFIRM", 
                    value=(is_correct == 1), 
                    key=checkbox_key,
                    on_change=update_feedback
                )
        else:
            with col:
                st.warning(f"MISSING {evidence_id}")

def _render_comparison_gallery(db_manager, session_id):
    """
    [NEW] 強烈情緒交叉比對畫廊
    邏輯：找出同一時間點的 Face 與 Plate 照片，並排顯示。
    """
    # 1. 撈出該 Session 所有強烈情緒相關的證據
    df_face = db_manager.get_event_evidence(session_id, "strong_emotion_face")
    df_plate = db_manager.get_event_evidence(session_id, "strong_emotion_plate")
    
    if df_face.empty and df_plate.empty:
        st.info("尚未偵測到強烈情緒事件 (Confidence > 50%)")
        return

    # 2. 進行配對 (Pairing)
    # 我們利用檔名中的時間戳記 (例如 "11月30日_12點01分05秒") 來配對
    pairs = {} 
    
    # 處理臉部照片
    for _, row in df_face.iterrows():
        path = row['local_path']
        filename = os.path.basename(path)
        
        # 修正後的解析邏輯
        parts = filename.split('_')
        # 取前兩段當作唯一的時間 Key (月日_時分秒)
        key = f"{parts[0]}_{parts[1]}"
        
        # 取得主要情緒名稱 (移除分數)
        raw_emo = parts[2] # "開心-98"
        emo_label = raw_emo.split('-')[0] # "開心"

        if key not in pairs: pairs[key] = {}
        pairs[key]['face'] = path
        pairs[key]['emotion'] = emo_label
        pairs[key]['time'] = parts[1] # 顯示時間

    # 處理餐盤照片
    for _, row in df_plate.iterrows():
        path = row['local_path']
        filename = os.path.basename(path)
        parts = filename.split('_')
        key = f"{parts[0]}_{parts[1]}"
        
        if key not in pairs: pairs[key] = {}
        pairs[key]['plate'] = path

    # 3. 渲染 UI (由新到舊排序)
    sorted_keys = sorted(pairs.keys(), reverse=True)
    
    for key in sorted_keys:
        item = pairs[key]
        face_path = item.get('face')
        plate_path = item.get('plate')
        emotion_label = item.get('emotion', 'Unknown')
        time_label = item.get('time', '')

        # 卡片式佈局
        with st.container(border=True):
            # 標題列：顯示情緒與時間
            st.markdown(f"#### 🔥 {emotion_label} <span style='font-size:0.8em; color:gray'>({time_label})</span>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            
            # 左邊：表情
            with c1:
                st.caption("👤 顧客表情")
                if face_path and os.path.exists(face_path):
                    st.image(face_path, use_container_width=True)
                else:
                    st.warning("影像遺失")
            
            # 右邊：餐盤
            with c2:
                st.caption("🍽️ 當下餐盤")
                if plate_path and os.path.exists(plate_path):
                    st.image(plate_path, use_container_width=True)
                else:
                    st.warning("影像遺失")

def _render_all_emotions_gallery(db_manager, session_id):
    """
    [NEW] 顯示所有偵測到的情緒照片 (含 Top 2 分數)
    """
    df = db_manager.get_event_evidence(session_id, "strong_emotion_face")
    
    if df.empty:
        st.info("尚無情緒紀錄")
        return

    # 使用 Grid 佈局
    cols = st.columns(4)
    
    for i, row in df.iterrows():
        path = row['local_path']
        if not os.path.exists(path): continue
            
        filename = os.path.basename(path)
        # 解析檔名: 時間_情緒1-分數_情緒2-分數_Face.jpg
        try:
            parts = filename.split('_')
            # parts[0]: 日期, parts[1]: 時間
            time_str = f"{parts[1]}" 
            
            # 解析情緒 1 (例如 "開心-98")
            e1_part = parts[2].split('-')
            e1_label = e1_part[0]
            e1_score = e1_part[1]
            
            # 解析情緒 2 (例如 "驚艷-02")
            # 舊的檔案可能沒有第二情緒，要做防呆
            if len(parts) >= 5:
                e2_part = parts[3].split('-')
                e2_label = e2_part[0]
                e2_score = e2_part[1]
                caption_text = f"🥇{e1_label}({e1_score}%) | 🥈{e2_label}({e2_score}%)"
            else:
                caption_text = f"🥇{e1_label}({e1_score}%)"
                
        except:
            # 解析失敗 (可能是舊檔案)
            time_str = "Unknown"
            caption_text = "Legacy Data"

        col = cols[i % 4]
        with col:
            st.image(path, use_container_width=True)
            st.caption(f"🕒 {time_str}")
            st.markdown(f"**{caption_text}**")

def display(client, db_manager, t=None): 
    # 防呆：如果沒傳 t (翻譯函式)，給一個預設的
    if t is None: 
        def t(k): return k
    
    db = db_manager 

    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.subheader(t("dash_title"))
    with col_refresh:
        if st.button(t("btn_refresh"), use_container_width=True):
            st.rerun()

    # ==========================================
    # 1. 篩選器 (Filter)
    # ==========================================
# 1. 篩選器
    with st.container(border=True):
        # ★★★ 修改這裡：將 color 改為 var(--primary-color) ★★★
        st.markdown(f"<h5 style='color:var(--primary-color); font-weight:bold;'>{t('filter_title')}</h5>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            today = datetime.date.today()
            date_range = st.date_input(
                t("date_range"), 
                value=[today, today],
                format="YYYY/MM/DD"
            )
            
        with col2:
            time_range_option = st.selectbox(
                t("time_period"),
                [t("opt_all_day"), t("opt_custom")]
            )

        with col3:
            # 這裡簡單映射選項
            source_option = st.selectbox(
                t("data_source"),
                ["All", "Live", "Video"]
            )

        if len(date_range) != 2:
            st.warning("Please select end date.")
            st.stop()

        start_date, end_date = date_range
        
        # 判斷是否為全日
        if "All Day" in time_range_option or "全日" in time_range_option:
            start_dt_str = f"{start_date} 00:00:00"
            end_dt_str = f"{end_date} 23:59:59"
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                s_time = st.time_input("Start", datetime.time(9, 0))
            with col_end:
                e_time = st.time_input("End", datetime.time(21, 0))
            start_dt_str = f"{start_date} {s_time.strftime('%H:%M:%S')}"
            end_dt_str = f"{end_date} {e_time.strftime('%H:%M:%S')}"
            
    # 2. 數據獲取邏輯
    if source_option == "Live":
        # [修改] 加入 'live_dual_cam'
        selected_sources = ['live_stream', 'live_session_summary', 'live_dual_cam']
    elif source_option == "Video":
        selected_sources = ['uploaded_video']
    else:
        # [修改] 加入 'live_dual_cam'
        selected_sources = ['live_stream', 'live_session_summary', 'uploaded_video', 'live_dual_cam']

    df_logs = db.get_logs_by_range(start_dt_str, end_dt_str, source_types=selected_sources)
    num_groups, groups_df = db.get_customer_groups_analysis(start_dt_str, end_dt_str, gap_minutes=0.6)
    df_sessions_all = db.get_all_session_records()
    
    df_sessions = pd.DataFrame()
    if not df_sessions_all.empty:
        df_sessions_all['timestamp'] = pd.to_datetime(df_sessions_all['timestamp'])
        mask = (df_sessions_all['timestamp'] >= pd.to_datetime(start_dt_str)) & \
               (df_sessions_all['timestamp'] <= pd.to_datetime(end_dt_str))
        df_sessions = df_sessions_all.loc[mask].copy()
        df_sessions = df_sessions.sort_values('timestamp', ascending=False)

    if df_logs.empty and df_sessions.empty and num_groups == 0:
        st.info("NO DATA AVAILABLE.")
        return

    # ==========================================
    # 3. 分頁顯示 (Tabs)
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        t("tab_traffic"), 
        t("tab_satisfaction"), 
        t("tab_plate"), 
        t("tab_report")
    ])

    # --- TAB 1: Traffic ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        avg_ppl = 0
        if not df_logs.empty:
            valid_ppl = df_logs[df_logs['people_count'] > 0]['people_count']
            if not valid_ppl.empty:
                avg_ppl = valid_ppl.mean()

        c1.metric(t("metric_groups"), f"{num_groups}")
        c2.metric(t("metric_avg_size"), f"{avg_ppl:.1f}")
        c3.metric(t("metric_sessions"), len(df_sessions))

        st.markdown(f"#### {t('chart_traffic')}")
        with st.container(border=True):
            if not df_logs.empty:
                df_chart = df_logs.copy()
                df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'])
                df_chart = df_chart.set_index('timestamp')
                flow_data = df_chart['people_count'].resample('5T').max().fillna(0)
                st.area_chart(flow_data, color="#06b6d4", use_container_width=True)
            else:
                st.info("NO TRAFFIC DATA")
        
        if num_groups > 0:
            with st.expander("DETAILS"):
                st.dataframe(groups_df, use_container_width=True, hide_index=True)

    # --- TAB 2: Satisfaction ---
    with tab2:
        total_nods = df_sessions['nod_count'].sum() if not df_sessions.empty else 0
        total_shakes = df_sessions['shake_count'].sum() if not df_sessions.empty else 0
        
        waste_count = 0
        if not df_sessions.empty:
            for _, row in df_sessions.iterrows():
                try:
                    data = json.loads(row['leftover_data'])
                    if data and len(data) > 0: 
                        waste_count += 1
                except:
                    pass
        waste_rate = (waste_count / len(df_sessions) * 100) if not df_sessions.empty else 0

        k1, k2, k3 = st.columns(3)
        k1.metric(t("metric_nods"), int(total_nods))
        k2.metric(t("metric_shakes"), int(total_shakes))
        k3.metric(t("metric_waste"), f"{waste_rate:.1f}%")

        st.divider()

        st.markdown("#### 😊 EMOTION DISTRIBUTION")
        
        df_emotions = df_logs[df_logs['source_type'].isin(['live_session_summary', 'uploaded_video'])]
        if df_emotions.empty:
            # [修改] 這裡也要加入 'live_dual_cam'
            df_emotions = df_logs[df_logs['source_type'].isin(['live_stream', 'live_dual_cam'])]

        with st.container(border=True):
            if not df_emotions.empty:
                all_emotions = Counter()
                data_found = False

                for _, row in df_emotions.iterrows():
                    e_raw = row.get('emotions')
                    if pd.isna(e_raw) or e_raw == "":
                        continue
                    try:
                        e_dict = ast.literal_eval(str(e_raw)) if isinstance(e_raw, str) else e_raw
                        if isinstance(e_dict, dict):
                            for k, v in e_dict.items():
                                if k not in ['Meal_Status', 'status']:
                                    try:
                                        val = float(v) 
                                        if val > 0:
                                            all_emotions[k] += val
                                            data_found = True
                                    except (ValueError, TypeError):
                                        continue
                    except Exception:
                        continue 

                if data_found and all_emotions:
                    e_df = pd.DataFrame(all_emotions.items(), columns=['Emotion', 'Count'])
                    # 使用 Plotly 繪製，並設定 X 軸文字角度
                    fig = px.bar(
                        e_df, 
                        x='Emotion', 
                        y='Count', 
                        color_discrete_sequence=['#8b5cf6'],
                        text_auto=True # 顯示數值在柱狀圖上
                    )
                    # ★ 強制 X 軸文字水平顯示 (0度)
                    fig.update_layout(xaxis_tickangle=0)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("NO DETAILED EMOTIONS")
            else:
                st.info("NO DATA")

    # --- TAB 3: Plate Insights ---
    with tab3:
        m1, m2 = st.columns(2)
        m1.metric("WASTE COUNT", f"{waste_count}")
        m2.metric(t("metric_waste"), f"{waste_rate:.1f}%")

        st.write("")
        if not df_sessions.empty:
            insight_df = df_sessions[df_sessions['ai_insight'].notna() & (df_sessions['ai_insight'] != "")]
            if not insight_df.empty:
                for _, row in insight_df.iterrows():
                    ts_str = row['timestamp'].strftime('%H:%M')
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f"<span style='color:#06b6d4'>**[{ts_str} {row['mode']}]**</span>", unsafe_allow_html=True)
                        st.write(row['ai_insight'])
                        try:
                            l_data = json.loads(row['leftover_data'])
                            if l_data:
                                st.caption(f"Data: {l_data}")
                        except:
                            pass
            else:
                st.info("NO INSIGHTS")
        else:
            st.info("NO DATA")

    # --- TAB 4: Report ---
    with tab4:
        st.subheader(t("tab_report"))
        
        if not df_sessions.empty:
            st.markdown(f"**PERIOD**: `{start_dt_str}` ~ `{end_dt_str}`")
            
            period_stats = {
                "total_sessions": len(df_sessions),
                "total_nods": int(total_nods),
                "total_shakes": int(total_shakes),
                "waste_rate": f"{waste_rate:.1f}%"
            }

            if st.button(t("btn_gen_report"), type="primary", use_container_width=True):
                with st.spinner("AI Generating..."):
                    prompt = f"Analyze: {period_stats}"
                    async def run_rep():
                        try: 
                            BACKEND_CONFIG = {"store_type": "Buffet", "tone": "Pro", "tips_style": "Strategy"}
                            resp, _ = await llm.summarize_session(period_stats, client=client, custom_instructions=prompt, **BACKEND_CONFIG)
                            return resp
                        except Exception as e:
                            return f"Error: {e}"
                    report = asyncio.run(run_rep())
                    st.markdown(report)

            st.divider()
            st.subheader(f"{t('header_evidence')} ({len(df_sessions)})")
            
            if 'session_id_raw' not in df_sessions.columns:
                 df_sessions['session_id_raw'] = df_sessions['timestamp'].dt.strftime('%Y%m%d%H%M%S')

            for _, row in df_sessions.iterrows():
                ts = row['timestamp']
                time_str = ts.strftime('%m/%d %H:%M')
                
                nods = int(row.get('nod_count', 0))
                shakes = int(row.get('shake_count', 0))
                label = f"📍 {time_str} | 😊 {nods} vs 😟 {shakes}"
                
                unique_session_id = row['session_id_raw']

                with st.expander(label, expanded=False):
                    # ★★★ [修改] 增加第 5 個 Tab: ALL EMOTIONS ★★★
                    t1, t2, t3, t4, t5 = st.tabs(["🎥 NOD", "🎥 SHAKE", "🍽️ WASTE", "🔥 CROSS-CHECK", "😊 ALL EMOTIONS"])
                    
                    with t1: _render_evidence_grid(db, unique_session_id, 'nod')
                    with t2: _render_evidence_grid(db, unique_session_id, 'shake')
                    with t3: _render_evidence_grid(db, unique_session_id, 'plate_vlm')
                    with t4: _render_comparison_gallery(db, unique_session_id)
                    
                    # [新增] 呼叫新函式
                    with t5:
                        _render_all_emotions_gallery(db, unique_session_id)
        else:
            st.info("NO DATA")