# core/live_analyzer.py

"""
即時分析引擎 (LiveAnalyzer) - VLM 整合版
功能：
1. DeepFace 本地情緒 (行為模式)。
2. GPT-4o VLM 餐盤洞察 (餐盤模式)。
3. 雙模式分流與快取機制。
4. ★ [NEW] 影像佐證儲存與 DB 連結。
"""

import cv2
import time
import threading
import asyncio
from queue import Queue, Empty, Full
import numpy as np
import os # [NEW]
from datetime import datetime # [NEW]

# [MODIFIED] 導入 DB 寫入函式 (保留您原本的 insert_log，新增 save_event_evidence)
from services.database import insert_log 

from services.vision_analysis import (
    HeadGestureDetector,
    crop_face_with_mediapipe,
    estimate_plate_leftover,
    detect_food_regions_yolo, 
)
from services import llm_handler as llm 

# 防呆讀取 config
import config as _cfg

EMOTE_INTERVAL_SECONDS   = getattr(_cfg, "EMOTE_INTERVAL_SECONDS", 2.0)
CAMERA_RESOLUTION_WIDTH  = getattr(_cfg, "CAMERA_RESOLUTION_WIDTH", 1280)
CAMERA_RESOLUTION_HEIGHT = getattr(_cfg, "CAMERA_RESOLUTION_HEIGHT", 720)
CAMERA_BUFFER_SIZE       = getattr(_cfg, "CAMERA_BUFFER_SIZE", 1)
CAMERA_INDEX             = getattr(_cfg, "CAMERA_INDEX", 0)

from .types import AnalysisResult

# VLM 專用冷卻時間
VLM_INTERVAL_SECONDS = 10.0
# Log 寫入時間間隔
LOG_INTERVAL_SECONDS = 5.0 

# ★★★ 新增：影像佐證儲存目錄 ★★★
EVIDENCE_DIR = "session_evidence" 

class LiveAnalyzer:
    # [MODIFIED] 接受 db_manager 參數
    def __init__(self, model_pack: dict, menu_items: list, analysis_options: dict, db_manager):
        self.model_pack = model_pack
        self.menu_items = menu_items
        self.analysis_options = analysis_options

        self._frame_display_queue = Queue(maxsize=1)
        self._frame_analysis_queue = Queue(maxsize=1)
        self._analysis_result_queue = Queue(maxsize=1)

        self.gesture_detector = HeadGestureDetector()

        self._stop_event = threading.Event()
        self._camera_thread = None
        self._worker_thread = None
        
        # ★★★ 新增：DB 實例與 Session ID ★★★
        self.db_manager = db_manager
        # 初始 ID (start 時會更新)
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S") 
        os.makedirs(EVIDENCE_DIR, exist_ok=True) 
        
        # 情緒分析狀態
        self._llm_busy = False 
        self._last_emote_ts = 0.0
        self._cached_emotion = "中性"      
        self._new_emotion_arrived = False 

        # VLM 餐盤分析狀態
        self._vlm_busy = False
        self._last_vlm_ts = 0.0
        self._cached_plate_insight = None 
        self._new_insight_arrived = False
        
        self._last_log_ts = 0.0
       
        # Token 統計
        self._cached_token_usage = None
        
        # YOLO 食物偵測快取
        self._cached_food_detections = [] 

        # Latch
        self._latched_nod = False
        self._latched_shake = False
        self._latch_lock = threading.Lock()

    # -------------------------------------------------
    #  執行緒 1：攝影機
    # -------------------------------------------------
    def _camera_loop(self):
        # [保留您原本的邏輯，但建議使用 AVFoundation 以防萬一]
        import cv2
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(1) # 退回內建
            if not cap.isOpened():
                print("❌ 無法開啟攝影機")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_HEIGHT)

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            if self._frame_display_queue.full():
                try: self._frame_display_queue.get_nowait()
                except Empty: pass
            self._frame_display_queue.put_nowait(frame)

            if self._frame_analysis_queue.full():
                try: self._frame_analysis_queue.get_nowait()
                except Empty: pass
            self._frame_analysis_queue.put_nowait(frame)
            
            time.sleep(0.005) 
        cap.release()

    # ★★★ 新增：儲存影像證據輔助函式 ★★★
    def _save_evidence(self, event_type, frame, frame_count):
        """將影像佐證儲存到磁碟並記錄到資料庫"""
        try:
            filename = f"{self.session_id}_{event_type}_{frame_count}.jpg"
            path = os.path.join(EVIDENCE_DIR, filename)
            
            if not os.path.exists(EVIDENCE_DIR):
                os.makedirs(EVIDENCE_DIR)

            cv2.imwrite(path, frame)
            
            # 使用 db_manager 寫入
            self.db_manager.save_event_evidence(
                session_id=self.session_id, 
                event_type=event_type, 
                local_path=path
            )
        except Exception as e:
            print(f"Evidence Save Error: {e}")

    # -------------------------------------------------
    #  執行緒 2：CV 分析
    # -------------------------------------------------
    def _analysis_worker(self):
        client = self.model_pack.get("client")
        pose_detector = self.model_pack.get("pose_detector")
        face_detector = self.model_pack.get("face_detector")
        detector = self.gesture_detector

        frame_count = 0
        
        # 狀態快取變數
        cached_plate_label = None
        cached_plate_ratio = None 
        cached_plate_circle = None
        cached_food_dets = [] 

        last_debug_print_ts = 0 

        while not self._stop_event.is_set():
            try:
                frame = self._frame_analysis_queue.get(timeout=1)
            except Empty:
                continue

            result = AnalysisResult()
            frame_count += 1

            # 人數計算
            current_people_count = 0
            if face_detector:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = face_detector.process(rgb_frame)
                    if face_results.detections:
                        current_people_count = len(face_results.detections)
                except Exception as e:
                    pass # 忽略錯誤
            
            result.display_info["people_count"] = current_people_count

            # (A) 點頭/搖頭
            opt_nod = self.analysis_options.get("opt_nod", True)
            if opt_nod and pose_detector:
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    res = pose_detector.process(rgb)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        nose_x, nose_y = lm[0].x, lm[0].y
                        ref_x = (lm[7].x + lm[8].x + lm[11].x + lm[12].x) / 4.0
                        ref_y = (lm[7].y + lm[8].y + lm[11].y + lm[12].y) / 4.0
                        dx = nose_x - ref_x
                        dy = nose_y - ref_y
                        event = detector.update_and_classify(dx, dy)
                        
                        if event == "nod":
                            with self._latch_lock: self._latched_nod = True
                            # ★ 觸發點：儲存點頭佐證
                            self._save_evidence("nod", frame.copy(), frame_count)
                        elif event == "shake":
                            with self._latch_lock: self._latched_shake = True
                            # ★ 觸發點：儲存搖頭佐證
                            self._save_evidence("shake", frame.copy(), frame_count)
                except Exception:
                    pass

            # =========================================
            #  (B) 餐盤偵測 + YOLO + VLM (餐盤模式)
            # =========================================
            if self.analysis_options.get("opt_plate"):
                # 1. OpenCV 快速篩選 & YOLO 偵測 (每 15 幀一次)
                if frame_count % 15 == 0:
                    try:
                        label, ratio, circle = estimate_plate_leftover(frame)
                        
                        if label in ["剩餘50%以上", "無剩餘"]:
                            cached_plate_label = label
                            cached_plate_ratio = ratio 
                        else:
                            cached_plate_label = None 
                            cached_plate_ratio = None
                        
                        cached_plate_circle = circle
                        
                        # YOLO 偵測
                        cached_food_dets = detect_food_regions_yolo(frame)
                        self._cached_food_detections = cached_food_dets
                        
                    except Exception as e:
                        print(f"CV/YOLO error: {e}")
                        pass
                
                # 更新結果給 UI
                if cached_plate_label:
                    result.plate_event = cached_plate_label 
                    
                    if cached_plate_ratio is not None:
                        display_text = f"{cached_plate_label} ({cached_plate_ratio:.0%})"
                    else:
                        display_text = cached_plate_label
                    
                    result.display_info["plate_label"] = display_text

                if cached_plate_circle:
                    result.display_info["plate_circle"] = cached_plate_circle
                
                result.display_info["food_detections"] = cached_food_dets


                # 2. VLM 深度分析觸發
                now = time.time()
                cond_cv = (cached_plate_label is not None)
                cond_yolo = (len(cached_food_dets) > 0)
                should_trigger = (cond_cv or cond_yolo)
                is_cooldown = (now - self._last_vlm_ts) < VLM_INTERVAL_SECONDS
                
                if should_trigger and (now - last_debug_print_ts > 3.0):
                    if not client:
                        print("⚠️ [VLM Warning] 未設定 OpenAI API Key")
                    elif self._vlm_busy:
                        print("⏳ [VLM Skip] 系統忙碌中")
                    last_debug_print_ts = now

                if (should_trigger and 
                    client and 
                    not self._vlm_busy and 
                    not is_cooldown):
                    
                    self._vlm_busy = True 
                    self._last_vlm_ts = now 
                    
                    print(f"🚀 VLM 觸發成功!")
                    
                    # ★ 觸發點：儲存 VLM 觸發佐證
                    self._save_evidence("plate_vlm", frame.copy(), frame_count)
                    
                    # [MODIFIED] 傳入 cached_food_dets 給 VLM (對應您的 llm_handler)
                    threading.Thread(target=self._run_vlm_background, 
                                     args=(frame.copy(), client, cached_food_dets)).start()

            # (C) DeepFace 表情
            now = time.time()
            if (self.analysis_options.get("opt_emote") and 
                not self._llm_busy and (now - self._last_emote_ts) > EMOTE_INTERVAL_SECONDS):
                
                self._llm_busy = True
                self._last_emote_ts = now
                threading.Thread(target=self._run_deepface_background, 
                                 args=(frame.copy(), face_detector)).start()

            # 填入結果
            if self._new_emotion_arrived:
                result.emotion_event = self._cached_emotion
                self._new_emotion_arrived = False 

            if self._cached_plate_insight:
                result.plate_insight = self._cached_plate_insight

            if self._cached_token_usage:
                result.token_usage_event = self._cached_token_usage
                self._cached_token_usage = None 

            # 自動記錄 Log
            now = time.time()
            if (now - self._last_log_ts) > LOG_INTERVAL_SECONDS: 
                if current_people_count > 0 or cached_plate_label:
                    emotions_data = {"primary": self._cached_emotion} if self._cached_emotion else {}
                    try:
                        insert_log(
                            source_type="live_stream",
                            people_count=current_people_count,
                            emotions=emotions_data,
                            food_detected=cached_plate_label if cached_plate_label else "無"
                        )
                        self._last_log_ts = now
                    except Exception as e:
                        print(f"Log save failed: {e}")

            if self._analysis_result_queue.full():
                try: self._analysis_result_queue.get_nowait()
                except Empty: pass
            self._analysis_result_queue.put_nowait(result)

    # -------------------------------------------------
    #  背景任務：VLM 餐盤分析
    # -------------------------------------------------
    def _run_vlm_background(self, frame, client, food_detections):
        # [MODIFIED] 這裡假設您的 llm_handler.analyze_plate_vlm 尚未更新為接受 food_detections
        # 如果您還沒改 llm_handler，我們只傳前兩個參數，忽略 food_detections
        # 如果您已經改了，請改為: await llm.analyze_plate_vlm(frame, client, food_detections=food_detections)
        try:
            async def task():
                # 使用標準呼叫，忽略 food_detections 以防錯誤
                insight, usage = await llm.analyze_plate_vlm(frame, client) 
                return insight, usage

            insight, usage = asyncio.run(task())
            
            if insight:
                print(f"✨ AI 洞察回傳: {insight}")
                self._cached_plate_insight = insight 
                self._new_insight_arrived = True 
            else:
                print("⚠️ AI 洞察回傳為空 (可能 API 錯誤)")
            
            if usage:
                current = self._cached_token_usage or {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                current["total_tokens"] += usage.total_tokens
                current["prompt_tokens"] += usage.prompt_tokens
                current["completion_tokens"] += usage.completion_tokens
                self._cached_token_usage = current

        except Exception as e:
            print(f"VLM Error: {e}")
        finally:
            self._vlm_busy = False 

    # -------------------------------------------------
    #  背景任務：DeepFace 本地分析 (維持不變)
    # -------------------------------------------------
    def _run_deepface_background(self, frame, face_detector):
        try:
            from deepface import DeepFace 
            face_crop = crop_face_with_mediapipe(frame, face_detector)
            if face_crop is None: return

            analysis = DeepFace.analyze(
                img_path=face_crop, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='skip',
                silent=True
            )
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_map = {
                "happy": "滿意/開心",
                "surprise": "驚艷",
                "neutral": "平淡/進食",
                "sad": "失望",
                "disgust": "嫌棄/難吃",
                "angry": "不滿",
                "fear": "困惑" 
            }
            final_emotion = emotion_map.get(dominant_emotion, dominant_emotion)
            
            self._cached_emotion = final_emotion
            self._new_emotion_arrived = True

        except Exception:
            pass
        finally:
            self._llm_busy = False

    def start(self):
        # [MODIFIED] 啟動時更新 Session ID，確保與 UI 同步
        #self.session_id = datetime.now().strftime("%Y%m%d%H%M%S") 
        if self._camera_thread and self._camera_thread.is_alive(): return
        self._stop_event.clear()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self._camera_thread.start()
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._camera_thread: self._camera_thread.join(timeout=2)
        if self._worker_thread: self._worker_thread.join(timeout=2)
        self._camera_thread = None
        self._worker_thread = None

    def get_latest_frame(self):
        try: return self._frame_display_queue.get_nowait()
        except Empty: return None

    # [NEW] 提供屬性給 UI 讀取
    @property
    def raw_session_id(self):
        return self.session_id

    def get_latest_analysis_result(self):
        try:
            result = self._analysis_result_queue.get_nowait()
        except Empty:
            return None

        with self._latch_lock:
            if self._latched_nod:
                result.nod_event = True
                self._latched_nod = False
            
            if self._latched_shake:
                result.shake_event = True 
                self._latched_shake = False

        return result