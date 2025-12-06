# utils/localization.py

TRANSLATIONS = {
    "zh": {
        # 通用
        "app_title": "DineSence智慧餐飲分析",
        "settings": "系統設定",
        "select_lang": " ",
        
        # Tabs
        "tab_live": "🌟 現場監控 (Live)",
        "tab_video": "🎞️ 影像分析 (Video)",
        "tab_dashboard": "📈 營運儀表板 (Dashboard)",

        # Live View
        "live_panel_title": "看見，未被訴說的美味",
        "live_status_active": "系統運作中",
        "metric_people": "即時人數",
        "metric_satisfaction": "滿意度指數",
        "metric_events": "事件偵測",
        "cam_input": "攝影機來源",
        "start_btn": "啟動監控",
        "stop_btn": "停止監控",
        "waiting": "等待影像輸入...",
        "log_title": "📋 即時事件日誌",
        "ai_insight": "AI 現場洞察",
        "live_title": "🔴 即時監控",

        # 🔹 Live View 追加 key（給 ui/live_view.py 用）
        "metric_nods_shakes": "點頭 / 搖頭",
        "metric_emotion": "情緒",
        "live_cam_face": "👤 顧客（鏡頭 0）",
        "live_cam_plate": "🍽️ 餐盤（鏡頭 1）",
        "session_id_label": "紀錄編號",
        "waiting_cam": "等待鏡頭畫面……",
        

        # Video View
        "video_header": "🎞️ 上傳影片進行離線分析",
        "video_upload_label": "支援 .mp4 / .avi 格式",
        "video_uploaded": "已上傳影片",
        "sample_interval": "抽樣間隔 (秒)",
        "chk_plate": "分析餐盤殘留",
        "chk_emote": "分析表情",
        "chk_food": "分析食物 (YOLO)",
        "chk_debug": "開啟 Debug 視覺化",
        "btn_start_video": "🚀 開始分析影片",
        "msg_init": "初始化模型中...",
        "msg_analyzing": "分析中",
        "msg_done": "分析完成！",
        "expander_raw": "查看原始數據 (JSON)",
        "header_summary": "🎯 影片分析摘要",


        # Dashboard View
        "dash_title": "📊 營運數據儀表板",
        "btn_refresh": "🔄 刷新數據",
        "filter_title": "🔍 數據篩選條件",
        "date_range": "📅 日期範圍",
        "time_period": "🕒 查詢時段",
        "data_source": "📹 數據來源",
        "opt_all_day": "全日 (00:00 - 23:59)",
        "opt_custom": "自訂時段",
        "tab_traffic": "👥 人流與翻桌",
        "tab_satisfaction": "😊 滿意度分析",
        "tab_plate": "🍽️ 餐盤洞察",
        "tab_report": "🤖 總結報告",
        "metric_groups": "總客組數",
        # ★ [NEW] 新增第五個 Tab 的標題
        "tab_global_insight": "📊 總體數據洞察", 
        "tab_overview": "📊 營運數據概觀",       # [NEW] 合併後的首頁
        "tab_evidence": "📸 區間影像佐證紀錄",   # [NEW] 獨立出來的證據
        "tab_menu_insight": "🍔 菜色整體洞察",   # [RENAME] 原本的總體數據洞察
        "tab_ai_agent": "🤖 AI Agent 智慧洞察",  # [NEW] 新增的空 Tab
        
        "btn_gen_insight_report": "✨ 生成總體菜色洞察報告 (LLM)",
        "metric_avg_size": "平均單組人數",
        "metric_sessions": "分析場次",
        "metric_nods": "😊 滿意點頭",
        "metric_shakes": "😟 不滿搖頭",
        "metric_waste": "⚠️ 剩食比例",
        "chart_traffic": "📈 時段人流趨勢",
        "btn_gen_report": "✨ 生成營運分析報告",
        "header_evidence": "📸 區間影像佐證紀錄",
    },
    "en": {
        # General
        "app_title": "DineSence AI Analytics",
        "settings": "System settings",
        "select_lang": " ",

        # Tabs
        "tab_live": "🌟 Live Monitor",
        "tab_video": "🎞️ Video Analysis",
        "tab_dashboard": "📈 Dashboard",

        # Live View
        "live_panel_title": "Seeing the Unspoken Deliciousness",
        "live_status_active": "SYSTEM ACTIVE",
        "metric_people": "Real-time Occupancy",
        "metric_satisfaction": "Satisfaction Index",
        "metric_events": "Events Detected",
        "cam_input": "Camera Source",
        "start_btn": "Initialize System",
        "stop_btn": "Terminate Sequence",
        "waiting": "Awaiting Video Feed...",
        "log_title": "📋 Event Log",
        "ai_insight": "AI Live Insights",
        "live_title": "🔴 LIVE MONITORING",


        # 🔹 Live View extra keys (for ui/live_view.py)
        "metric_nods_shakes": "Nods / Shakes",
        "metric_emotion": "Emotion",
        "live_cam_face": "👤 Customer (Cam 0)",
        "live_cam_plate": "🍽️ Plate (Cam 1)",
        "session_id_label": "Session ID",
        "waiting_cam": "Waiting for camera feed...",

        # Video View
        "video_header": "🎞️ Offline Video Analysis",
        "video_upload_label": "Supports .mp4 / .avi",
        "video_uploaded": "Video Uploaded",
        "sample_interval": "Interval (sec)",
        "chk_plate": "Analyze Plate",
        "chk_emote": "Analyze Emotion",
        "chk_food": "Analyze Food (YOLO)",
        "chk_debug": "Debug Overlay",
        "btn_start_video": "🚀 Start Analysis",
        "msg_init": "Initializing...",
        "msg_analyzing": "Analyzing",
        "msg_done": "Analysis Complete!",
        "expander_raw": "View Raw Data (JSON)",
        "header_summary": "🎯 Analysis Summary",

        # Dashboard View
        "dash_title": "📊 Analytics Dashboard",
        "btn_refresh": "🔄 Refresh",
        "filter_title": "🔍 Data Filters",
        "date_range": "📅 Date Range",
        "time_period": "🕒 Time Period",
        "data_source": "📹 Data Source",
        "opt_all_day": "All Day (00:00 - 23:59)",
        "opt_custom": "Custom Range",
        "tab_traffic": "👥 Traffic",
        "tab_satisfaction": "😊 Satisfaction",
        "tab_plate": "🍽️ Plate Insights",
        "tab_report": "🤖 AI Report",
        "tab_global_insight": "📊 Global Insights",
        "btn_gen_insight_report": "✨ Generate Global Food Report",
        "metric_groups": "Total Groups",
        "metric_avg_size": "Avg Group Size",
        "metric_sessions": "Total Sessions",
        "metric_nods": "😊 Positive (Nod)",
        "metric_shakes": "😟 Negative (Shake)",
        "metric_waste": "⚠️ Waste Rate",
        "chart_traffic": "📈 Traffic Trend",
        "btn_gen_report": "✨ Generate Intelligence Report",
        "header_evidence": "📸 Evidence Feed",
    }
}

def get_text(key, lang="zh"):
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)
