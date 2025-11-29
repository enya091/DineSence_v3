# ui/video_view.py

import streamlit as st
import os
import cv2
import asyncio
import numpy as np
from collections import Counter
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
    # 防呆：如果沒傳 t，給個預設函式 (雖然理論上 app.py 會傳)
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
            show_debug_video = st.toggle(t("chk_debug"), value=True)

    if up is not None:
        tmp_path = os.path.join(".", f"tmp_{up.name}")
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())

        st.success(f"{t('video_uploaded')}: {up.name}")
        
        if st.button(t("btn_start_video"), type="primary", use_container_width=True, disabled=not client):
            
            # 使用兩欄佈局：左邊是分析進度，右邊是即時結果
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

            try:
                for fr in range(0, total_frames, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ok, frame = cap.read()
                    if not ok: break

                    progress_bar.progress(fr / total_frames, text=f"{t('msg_analyzing')}: {fr}/{total_frames}")
                    timestamp_s = int(fr / fps)
                    mm_ss = f"{timestamp_s//60:02d}:{timestamp_s%60:02d}"

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
                        food_regions = va.detect_food_regions_yolo(frame, conf=0.1, min_area_ratio=0.01)
                        if food_regions:
                            top_food = food_regions[0]
                            current_label = top_food["label"]
                            # (省略 GPT Food Check 以加速範例)
                            food_label = current_label
                            food_counter[food_label] += 1

                    # 3. 動作偵測
                    nod_flag = 0
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_res = pose_detector.process(rgb)
                    if pose_res.pose_landmarks:
                        lm = pose_res.pose_landmarks.landmark
                        # 簡化：偵測到 pose 就算有做事
                        # (完整邏輯請保持您原有的 nod_detector)
                        gesture = nod_detector.update_and_classify(lm[0].x - 0.5, lm[0].y - 0.5) 
                        if gesture == "nod":
                            nod_total += 1
                            nod_flag = 1
                    
                    # 4. 表情分析
                    emo_label = "-"
                    face_res = None
                    if do_emote_v:
                        face_res = face_detector.process(rgb)
                        if face_res.detections:
                            face_crop = va.crop_face_with_mediapipe(frame, face_detector, min_conf=0.2)
                            if face_crop is not None:
                                try:
                                    emo = llm.sync_gpt_image_classify_3cls(face_crop, client)
                                    if isinstance(emo, tuple): emo = emo[0]
                                    if emo in ["喜歡", "中性", "討厭"]:
                                        emotion_counter[emo] += 1
                                        emo_label = emo
                                except: pass

                    # 紀錄時間軸
                    timeline.append({
                        "t": mm_ss, "leftover": plate_label, "food": food_label, 
                        "nod": "✔" if nod_flag else " ", "emotion": emo_label
                    })
                    
                    if show_debug_video:
                        debug_frame = draw_debug_info(frame, face_res, food_regions, plate_circle)
                        col_video.image(debug_frame, caption=f"Time: {mm_ss}", use_container_width=True, channels="BGR")
                    
                    # 在右側即時顯示數據
                    with col_result:
                        st.metric("Detected Nod", nod_total)
                        st.caption(f"Last Emotion: {emo_label}")


                progress_bar.progress(1.0, text=t("msg_done"))

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
                if os.path.exists(tmp_path): os.remove(tmp_path)
                progress_bar.empty()