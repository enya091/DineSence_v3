# ui/video_view.py

import streamlit as st
import os
import cv2
import asyncio
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from services import vision_analysis as va
from services import llm_handler as llm

# 輔助繪圖函式 (保持不變)
def draw_debug_info(frame, face_res, food_res, plate_info):
    debug_frame = frame.copy()
    if face_res and face_res.detections:
        for detection in face_res.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, c = debug_frame.shape
            x = int(bbox.xmin * w); y = int(bbox.ymin * h)
            rw = int(bbox.width * w); rh = int(bbox.height * h)
            cv2.rectangle(debug_frame, (x, y), (x + rw, y + rh), (0, 0, 255), 2)
            cv2.putText(debug_frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if food_res:
        for item in food_res:
            x1, y1, x2, y2 = item["xyxy"]
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"{item['label']} {item['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if plate_info:
        if len(plate_info) == 3:
            x, y, r = plate_info
            cv2.circle(debug_frame, (x, y), r, (255, 0, 0), 3)
        elif len(plate_info) == 4:
            x, y, w, h = plate_info
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return debug_frame

def display(client, menu_items, llm_preferences, t=None):
    if t is None: 
        def t(k): return k

    st.subheader(t("video_header"))
    up = st.file_uploader(t("video_upload_label"), type=["mp4", "avi"])

    # --- 設定區 ---
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_sec = st.number_input(t("sample_interval"), min_value=1, max_value=30, value=2, step=1)
        with col2:
            do_plate_v = st.checkbox(t("chk_plate"), value=True)
            do_food_v = st.checkbox(t("chk_food"), value=True)
        with col3:
            do_emote_v = st.checkbox(t("chk_emote"), value=True)
            # [優化] 預設關閉 Debug 畫面以加速，或減少渲染頻率
            show_debug_video = st.toggle(t("chk_debug"), value=True)

    if up is not None:
        tmp_path = os.path.join(".", f"tmp_{up.name}")
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())

        st.success(f"{t('video_uploaded')}: {up.name}")
        
        # [優化] 使用 ThreadPool 來處理耗時的 LLM 請求
        executor = ThreadPoolExecutor(max_workers=4)
        
        if st.button(t("btn_start_video"), type="primary", use_container_width=True, disabled=not client):
            
            col_video, col_result = st.columns([1.5, 1])
            
            with col_video:
                image_placeholder = st.empty()
                progress_bar = st.progress(0, text=t("msg_init"))
            
            # 初始化偵測器
            pose_detector = va.get_pose_detector()
            face_detector = va.mp_face.FaceDetection(min_detection_confidence=0.2) 
            nod_detector = va.HeadGestureDetector()
            
            leftover_counter = Counter()
            emotion_counter = Counter()
            food_counter = Counter()
            nod_total = 0
            timeline = []

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(fps * sample_sec))

            # [優化] UI 更新頻率控制 (每處理幾幀才更新一次畫面)
            ui_update_rate = 3 
            processed_count = 0

            # 用來存放還沒跑完的 LLM 任務
            future_to_frame = []

            try:
                for fr in range(0, total_frames, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ok, frame = cap.read()
                    if not ok: break
                    
                    processed_count += 1

                    # [優化] 縮小影像尺寸：大幅提升 Vision Model (YOLO/MediaPipe) 處理速度
                    # 將寬度固定為 640px，高度等比例縮放
                    h, w = frame.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (640, new_h))

                    timestamp_s = int(fr / fps)
                    mm_ss = f"{timestamp_s//60:02d}:{timestamp_s%60:02d}"

                    # 更新進度條 (不需要每次都更新，減少 overhead)
                    if processed_count % 5 == 0:
                        progress_bar.progress(fr / total_frames, text=f"{t('msg_analyzing')}: {fr}/{total_frames}")

                    # 1. 餐盤分析
                    plate_label = "-"
                    plate_circle = None
                    if do_plate_v:
                        label, _, circle = va.estimate_plate_leftover(frame)
                        plate_circle = circle
                        if label in ["剩餘50%以上", "無剩餘"]:
                            leftover_counter[label] += 1
                            plate_label = label

                    # 2. 食物分析
                    food_label = "-"
                    food_regions = []
                    if do_food_v:
                        # conf 稍微調高可以減少誤判並加速
                        food_regions = va.detect_food_regions_yolo(frame, conf=0.25, min_area_ratio=0.01)
                        if food_regions:
                            top_food = food_regions[0]
                            food_label = top_food["label"]
                            food_counter[food_label] += 1

                    # 3. 動作偵測
                    nod_flag = 0
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_res = pose_detector.process(rgb)
                    if pose_res.pose_landmarks:
                        lm = pose_res.pose_landmarks.landmark
                        gesture = nod_detector.update_and_classify(lm[0].x - 0.5, lm[0].y - 0.5) 
                        if gesture == "nod":
                            nod_total += 1
                            nod_flag = 1
                    
                    # 4. 表情分析 [重點優化：非同步執行]
                    emo_label = "Analyzing..." # 暫時狀態
                    face_res = None
                    if do_emote_v:
                        face_res = face_detector.process(rgb)
                        if face_res.detections:
                            face_crop = va.crop_face_with_mediapipe(frame, face_detector, min_conf=0.2)
                            if face_crop is not None:
                                # 將耗時的 API 呼叫丟給 ThreadPool
                                # 注意：這會讓 emotion_counter 的統計稍微延遲，但影片處理速度會快非常多
                                future = executor.submit(llm.sync_gpt_image_classify_3cls, face_crop, client)
                                future_to_frame.append(future)

                    # 紀錄時間軸 (注意：這裡的 emotion 可能是空的，因為是非同步)
                    timeline.append({
                        "t": mm_ss, "leftover": plate_label, "food": food_label, 
                        "nod": "✔" if nod_flag else " ", "emotion": None 
                    })
                    
                    # [優化] 只有當開關開啟且達到更新頻率時才更新畫面
                    if show_debug_video and (processed_count % ui_update_rate == 0):
                        debug_frame = draw_debug_info(frame, face_res, food_regions, plate_circle)
                        # 顯示時將圖片縮小以加快傳輸
                        image_placeholder.image(debug_frame, caption=f"Time: {mm_ss}", use_container_width=True, channels="BGR")
                    
                    # 右側數據 (降低更新頻率)
                    if processed_count % ui_update_rate == 0:
                        with col_result:
                            st.metric("Detected Nod", nod_total)

                # --- 影片處理結束，收集所有 API 結果 ---
                progress_bar.progress(0.9, text="Waiting for AI API responses...")
                
                # 收集 ThreadPool 的結果
                for i, future in enumerate(future_to_frame):
                    try:
                        # 等待結果 (這裡會一次收回來)
                        emo = future.result(timeout=10) 
                        if isinstance(emo, tuple): emo = emo[0]
                        if emo in ["喜歡", "中性", "討厭"]:
                            emotion_counter[emo] += 1
                            # 回填 timeline (雖然時間軸順序可能對不上 crop 的確切 index，但做總結夠用了)
                            # 若要精確對應，需要更複雜的結構，這邊做簡單統計
                    except Exception as e:
                        print(f"API Error: {e}")

                progress_bar.progress(1.0, text=t("msg_done"))

                # 整理最終數據
                stats = {
                    "leftover": dict(leftover_counter), "food": dict(food_counter),
                    "nod": nod_total, "emotion": dict(emotion_counter),
                    "timeline": timeline
                }
                
                with st.expander(t("expander_raw"), expanded=False):
                    st.json(stats)
                
                # 產生摘要
                result_tuple = asyncio.run(llm.summarize_session(
                    stats,
                    store_type=llm_preferences.get("store_type", "餐廳"),
                    tone=llm_preferences.get("tone", "專業"),
                    tips_style=llm_preferences.get("tips_style", "預設"),
                    client=client
                ))
                
                st.subheader(t("header_summary"))
                st.markdown(result_tuple[0])

            finally:
                cap.release()
                executor.shutdown(wait=False) # 關閉執行緒池
                if os.path.exists(tmp_path): os.remove(tmp_path)
                progress_bar.empty()