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

def _render_global_insights(client, db_manager, df_sessions, t):
    """
    [NEW] 總體數據洞察：跨 Session 的菜色情緒統計與 LLM 報告
    """
    st.info("此頁面統計範圍為上方「篩選器」所選定之時間段內的數據。")

    if df_sessions.empty:
        st.warning("⚠️ 目前選定的時間範圍內無 Session 資料。")
        return

    # 1. 跨 Session 資料聚合 (Data Aggregation)
    all_food_data = []
    
    # 遍歷篩選出的所有 Session
    for _, session_row in df_sessions.iterrows():
        sid = session_row['session_id_raw']
        
        # 撈取該 Session 的餐點證據
        df_evidence = db_manager.get_event_evidence(sid, "strong_emotion_plate")
        
        if df_evidence.empty: continue
            
        for _, row in df_evidence.iterrows():
            # 排除人工否決的資料
            if row['human_corrected'] == 0: continue
                
            food_name = row['food_label'] if row['food_label'] else "Unknown"
            
            # 解析情緒 (從檔名)
            # 檔名格式: 時間_情緒-分_...
            try:
                fname = os.path.basename(row['local_path'])
                parts = fname.split('_')
                emotion_tag = parts[2].split('-')[0] # 取出 "開心"
                
                all_food_data.append({
                    "session_id": sid,
                    "evidence_id": row['id'],
                    "food": food_name,
                    "emotion": emotion_tag,
                    "path": row['local_path'],
                    "timestamp": row['session_timestamp'] # 這裡可能是 session_id，需注意顯示格式
                })
            except:
                continue

    if not all_food_data:
        st.warning("⚠️ 在此時間範圍內，尚未偵測到任何有效的餐點情緒數據。")
        return

    df_analysis = pd.DataFrame(all_food_data)

    # 2. 統計數據準備 (給 LLM 用)
    # 格式: {'漢堡': {'開心': 5, '嫌棄': 1}, '薯條': ...}
    food_stats = {}
    for food in df_analysis['food'].unique():
        sub_df = df_analysis[df_analysis['food'] == food]
        counts = sub_df['emotion'].value_counts().to_dict()
        food_stats[food] = counts

    # ==========================================
    # 區塊 A: LLM 總體洞察報告
    # ==========================================
    with st.container(border=True):
        st.subheader("🤖 AI 營運洞察報告")
        st.markdown("讓 AI 為您分析本時段內，各項餐點的顧客情緒表現。")
        
        if st.button(t("btn_gen_insight_report"), type="primary", use_container_width=True):
            if not client:
                st.error("未設定 OpenAI API Key")
            else:
                with st.spinner("AI 正在分析大數據..."):
                    # 組建 Prompt
                    stats_str = json.dumps(food_stats, ensure_ascii=False, indent=2)
                    system_prompt = (
                        "你是一位專業的餐廳數據分析師。使用者會提供一份 JSON 數據，"
                        "內容是不同菜色對應的顧客情緒統計 (例如: 漢堡 -> 開心:5, 嫌棄:2)。\n"
                        "請根據數據生成一份繁體中文報告，包含：\n"
                        "1. 🏆 **明星菜色**：哪道菜的正面情緒(開心/驚艷)比例最高？\n"
                        "2. ⚠️ **改進建議**：哪道菜出現了負面情緒(嫌棄/失望/不滿)？可能原因？\n"
                        "3. 💡 **總結洞察**：整體菜單的表現評價。\n"
                        "請用專業、簡潔的條列式語氣回答。"
                    )
                    user_prompt = f"請分析以下餐點情緒數據：\n{stats_str}"

                    async def run_gpt():
                        try:
                            resp = await client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.7
                            )
                            return resp.choices[0].message.content
                        except Exception as e:
                            return f"Error: {e}"
                            
                    report_text = asyncio.run(run_gpt())
                    st.markdown("---")
                    st.markdown(report_text)

    st.divider()

    # ==========================================
    # 區塊 B: 單品項詳細分析
    # ==========================================
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 🍔 菜色細節查詢")
        food_list = sorted(list(food_stats.keys()))
        selected_food = st.selectbox("選擇要鑽研的菜色", food_list)
        
        # 顯示該菜色的基本數據
        if selected_food:
            stats = food_stats[selected_food]
            total = sum(stats.values())
            st.caption(f"共蒐集到 {total} 筆反應")
            st.json(stats)

    with c2:
        if selected_food:
            # 畫圖
            df_target = df_analysis[df_analysis['food'] == selected_food]
            emo_counts = df_target['emotion'].value_counts().reset_index()
            emo_counts.columns = ['Emotion', 'Count']
            
            fig = px.bar(
                emo_counts, x='Emotion', y='Count',
                title=f"「{selected_food}」情緒分佈圖",
                color='Emotion', text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 區塊 C: 證據驗證與修正
    # ==========================================
    st.subheader(f"✅ 資料驗證 ({selected_food})")
    
    target_records = df_analysis[df_analysis['food'] == selected_food]
    
    # Grid 顯示
    cols = st.columns(4)
    for i, row in target_records.iterrows():
        col = cols[i % 4]
        with col:
            with st.container(border=True):
                # 顯示圖片
                if os.path.exists(row['path']):
                    st.image(row['path'], use_container_width=True)
                else:
                    st.warning("影像遺失")
                
                # 情緒標籤
                st.markdown(f"**{row['emotion']}**")
                
                # 勾選框
                chk_key = f"g_chk_{row['evidence_id']}"
                
                def update_cb(eid=row['evidence_id'], k=chk_key):
                    val = st.session_state[k]
                    db_manager.update_evidence_feedback(eid, val)
                    if not val: st.toast(f"已從統計中移除 (ID: {eid})")

                st.checkbox("確認無誤", value=True, key=chk_key, on_change=update_cb)

def _render_comparison_gallery(db_manager, session_id):
    """
    [NEW] 強烈情緒交叉比對畫廊
    邏輯：找出同一時間點的 Face 與 Plate 照片，並排顯示。
    """
    # 1. 撈出該 Session 所有強烈情緒相關的證據
    df_face = db_manager.get_event_evidence(session_id, "strong_emotion_face")
    df_plate = db_manager.get_event_evidence(session_id, "strong_emotion_plate")
    
    if df_face.empty and df_plate.empty:
        st.info("尚未偵測到強烈情緒事件 (Confidence > 40%)")
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
def _render_food_insights(db_manager, session_id):
    """
    [NEW] 餐點洞察模式：以食物為中心，統計顧客的情緒反應
    """
    # 1. 撈取該 Session 所有「強烈情緒的餐盤照」
    # 這些照片已經經過 LLM 辨識，帶有 food_label
    df_plate = db_manager.get_event_evidence(session_id, "strong_emotion_plate")
    
    if df_plate.empty:
        st.info("尚無 AI 辨識的餐點數據")
        return

    # 2. 資料前處理：解析檔名中的情緒，並過濾無效數據
    data_list = []
    
    for _, row in df_plate.iterrows():
        # 如果使用者已經手動取消勾選 (human_corrected=0)，就排除這筆資料
        if row['human_corrected'] == 0:
            continue
            
        food_name = row['food_label'] if row['food_label'] else "Unknown"
        path = row['local_path']
        evidence_id = row['id']
        
        # 從檔名解析情緒
        # 格式: {時間}_{情緒1-分}_{情緒2-分}_Plate.jpg
        # 範例: 12月01日_..._開心-98_驚艷-02_Plate.jpg
        try:
            filename = os.path.basename(path)
            parts = filename.split('_')
            
            # 取出第一高分的情緒
            e1_tag = parts[2] # "開心-98"
            emotion_label = e1_tag.split('-')[0] # "開心"
            
            # 為了顯示方便，我們也嘗試找對應的臉部照片
            # 只要把檔名結尾的 Plate.jpg 改成 Face.jpg 即可
            face_path = path.replace("_Plate.jpg", "_Face.jpg")
            
            data_list.append({
                "id": evidence_id,
                "food": food_name,
                "emotion": emotion_label,
                "plate_path": path,
                "face_path": face_path,
                "timestamp": parts[1]
            })
        except:
            continue

    if not data_list:
        st.warning("沒有有效的餐點數據 (可能都被取消勾選了)")
        return

    df_analysis = pd.DataFrame(data_list)

   # 3. UI 佈局
    all_foods = sorted(df_analysis['food'].unique().tolist())
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("### 🍔 選擇餐點")
        # ★★★ [修正] 加上 key 參數，綁定 session_id ★★★
        selected_food = st.selectbox(
            "請選擇要分析的菜色", 
            all_foods, 
            key=f"food_select_{session_id}" 
        )
    
    # 篩選出該食物的資料
    df_target = df_analysis[df_analysis['food'] == selected_food]
    
    with c2:
        # ==========================================
        # UI 區塊 B: 統計直方圖
        # ==========================================
        if not df_target.empty:
            # 統計各種情緒的出現次數
            emo_counts = df_target['emotion'].value_counts().reset_index()
            emo_counts.columns = ['Emotion', 'Count']
            
            fig = px.bar(
                emo_counts, x='Emotion', y='Count',
                title=f"顧客對「{selected_food}」的情緒反應分佈",
                color='Emotion',
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("此餐點無數據")

    st.divider()

    # ==========================================
    # UI 區塊 C: 詳細佐證與人工驗證
    # ==========================================
    st.markdown(f"### ✅ 資料驗證 ({len(df_target)} 筆)")
    st.caption("如果您發現 AI 判斷錯誤 (例如：這不是漢堡，或表情判斷錯誤)，請取消勾選，上方的統計圖表會自動扣除該筆數據。")

    # 使用 Grid 顯示
    cols = st.columns(3)
    
    for i, row in df_target.iterrows():
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                # 標題
                st.markdown(f"**{row['emotion']}** <span style='color:gray'>({row['timestamp']})</span>", unsafe_allow_html=True)
                
                # 左右並排顯示圖
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    if os.path.exists(row['face_path']):
                        st.image(row['face_path'], use_container_width=True)
                    else: st.text("No Face")
                with img_c2:
                    st.image(row['plate_path'], use_container_width=True)

                # 勾選框 (互動核心)
                # 當使用者改變勾選狀態時，會呼叫 db_manager 更新資料庫，然後 Streamlit 會自動重跑 (Rerun)
                checkbox_key = f"chk_food_{row['id']}"
                
                def on_change_callback(eid=row['id'], k=checkbox_key):
                    # 取得最新狀態
                    new_val = st.session_state[k]
                    # 更新資料庫
                    db_manager.update_evidence_feedback(eid, new_val)
                    # 提示
                    if not new_val:
                        st.toast(f"已移除 ID {eid}，圖表將重新計算")

                st.checkbox(
                    "資料正確 (納入統計)", 
                    value=True, 
                    key=checkbox_key,
                    on_change=on_change_callback
                )

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
                    
                    # 🔴 原本是 5 個 Tabs
                    # t1, t2, t3, t4, t5 = st.tabs([...])

                    # 🟢 請改成 6 個 Tabs (加入 FOOD INSIGHTS)
                    t1, t2, t3, t4, t5, t6 = st.tabs([
                        "🎥 NOD", 
                        "🎥 SHAKE", 
                        "🍽️ WASTE", 
                        "🔥 CROSS-CHECK", 
                        "😊 ALL EMOTIONS", 
                        "🍽️ FOOD INSIGHTS"  # <--- 新增這個
                    ])
                    
                    with t1: _render_evidence_grid(db, unique_session_id, 'nod')
                    with t2: _render_evidence_grid(db, unique_session_id, 'shake')
                    with t3: _render_evidence_grid(db, unique_session_id, 'plate_vlm')
                    with t4: _render_comparison_gallery(db, unique_session_id)
                    with t5: _render_global_insights(client, db, df_sessions, t)
                    
                    # 🟢 加入第 6 個分頁的內容
                    with t6:
                        _render_food_insights(db, unique_session_id)
        else:
            st.info("NO DATA")