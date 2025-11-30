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
import os 
from datetime import datetime
import platform # 用於判斷作業系統

# ★★★ 優化 1：將 DeepFace 移至最上方載入，避免執行緒內重複載入造成卡頓 ★★★
try:
    from deepface import DeepFace
    print("✅ DeepFace 模組載入成功")
except ImportError:
    DeepFace = None
    print("⚠️ DeepFace 模組未安裝，情緒分析將無法使用 (請執行 pip install deepface tf-keras)")

from services.database import insert_log 

from services.vision_analysis import (
    HeadGestureDetector,
    crop_face_with_mediapipe,
    estimate_plate_leftover,
    detect_food_regions_yolo, 
)
from services import llm_handler as llm 

import config as _cfg
from .types import AnalysisResult

EMOTE_INTERVAL_SECONDS   = getattr(_cfg, "EMOTE_INTERVAL_SECONDS", 2.0)
CAMERA_RESOLUTION_WIDTH  = getattr(_cfg, "CAMERA_RESOLUTION_WIDTH", 1280)
CAMERA_RESOLUTION_HEIGHT = getattr(_cfg, "CAMERA_RESOLUTION_HEIGHT", 720)
CAMERA_BUFFER_SIZE       = getattr(_cfg, "CAMERA_BUFFER_SIZE", 1)
CAMERA_INDEX             = getattr(_cfg, "CAMERA_INDEX", 0)

VLM_INTERVAL_SECONDS = 10.0
LOG_INTERVAL_SECONDS = 5.0 
EVIDENCE_DIR = "session_evidence" 

class LiveAnalyzer:
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
        
        self.db_manager = db_manager
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
        self._cached_token_usage = None
        self._cached_food_detections = [] 

        # Latch 鎖定機制
        self._latched_nod = False
        self._latched_shake = False
        self._latched_emotion = None
        self._latch_lock = threading.Lock()

    # -------------------------------------------------
    #  執行緒 1：攝影機
    # -------------------------------------------------
    def _camera_loop(self):
        system_os = platform.system()
        print(f"📷 正在啟動攝影機... (偵測系統: {system_os})")

        cap = None
        # 1. 根據系統選擇開啟方式
        if system_os == "Darwin": # macOS
             cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        elif system_os == "Windows": # Windows
             # Windows 優先使用 DSHOW
             cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        
        # 2. 如果失敗，退回預設
        if cap is None or not cap.isOpened():
            print("⚠️ 專屬模式開啟失敗，嘗試預設模式...")
            cap = cv2.VideoCapture(CAMERA_INDEX)

        if cap is None or not cap.isOpened():
            print("❌ 無法開啟攝影機 (請檢查連接或是被其他程式佔用)")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_HEIGHT)

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # 放入顯示佇列
            if self._frame_display_queue.full():
                try: self._frame_display_queue.get_nowait()
                except Empty: pass
            self._frame_display_queue.put_nowait(frame)

            # 放入分析佇列
            if self._frame_analysis_queue.full():
                try: self._frame_analysis_queue.get_nowait()
                except Empty: pass
            self._frame_analysis_queue.put_nowait(frame)
            
            time.sleep(0.005) 
            
        cap.release()
        print("📷 攝影機已釋放")

    def _save_evidence(self, event_type, frame, frame_count):
        try:
            filename = f"{self.session_id}_{event_type}_{frame_count}.jpg"
            path = os.path.join(EVIDENCE_DIR, filename)
            if not os.path.exists(EVIDENCE_DIR):
                os.makedirs(EVIDENCE_DIR)
            cv2.imwrite(path, frame)
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
        
        cached_plate_label = None
        cached_plate_ratio = None 
        cached_plate_circle = None
        cached_food_dets = [] 

        last_debug_print_ts = 0 

        while not self._stop_event.is_set():
            try:
                frame = self._frame_analysis_queue.get(timeout=0.5) # 縮短 timeout
            except Empty:
                continue

            result = AnalysisResult()
            frame_count += 1

            # 人數計算
            current_people_count = 0
            if face_detector:
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_results = face_detector.process(rgb_frame)
                    if face_results.detections:
                        current_people_count = len(face_results.detections)
                except Exception: pass
            
            result.display_info["people_count"] = current_people_count

            # (A) 點頭/搖頭
            if self.analysis_options.get("opt_nod") and pose_detector:
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    res = pose_detector.process(rgb)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        dx = lm[0].x - (lm[7].x + lm[8].x + lm[11].x + lm[12].x) / 4.0
                        dy = lm[0].y - (lm[7].y + lm[8].y + lm[11].y + lm[12].y) / 4.0
                        event = detector.update_and_classify(dx, dy)
                        
                        if event == "nod":
                            with self._latch_lock: self._latched_nod = True
                            self._save_evidence("nod", frame.copy(), frame_count)
                        elif event == "shake":
                            with self._latch_lock: self._latched_shake = True
                            self._save_evidence("shake", frame.copy(), frame_count)
                except Exception: pass

            # (B) 餐盤偵測
            if self.analysis_options.get("opt_plate"):
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
                        cached_food_dets = detect_food_regions_yolo(frame)
                        self._cached_food_detections = cached_food_dets
                    except Exception: pass
                
                if cached_plate_label:
                    result.plate_event = cached_plate_label 
                    display_text = f"{cached_plate_label} ({cached_plate_ratio:.0%})" if cached_plate_ratio else cached_plate_label
                    result.display_info["plate_label"] = display_text

                if cached_plate_circle: result.display_info["plate_circle"] = cached_plate_circle
                result.display_info["food_detections"] = cached_food_dets

                # VLM 觸發
                now = time.time()
                should_trigger = (cached_plate_label is not None or len(cached_food_dets) > 0)
                is_cooldown = (now - self._last_vlm_ts) < VLM_INTERVAL_SECONDS
                
                if should_trigger and (now - last_debug_print_ts > 3.0):
                    if not client: print("⚠️ [VLM Warning] 未設定 OpenAI API Key")
                    elif self._vlm_busy: print("⏳ [VLM Skip] 系統忙碌中")
                    last_debug_print_ts = now

                if should_trigger and client and not self._vlm_busy and not is_cooldown:
                    self._vlm_busy = True 
                    self._last_vlm_ts = now 
                    print(f"🚀 VLM 觸發成功!")
                    self._save_evidence("plate_vlm", frame.copy(), frame_count)
                    threading.Thread(target=self._run_vlm_background, 
                                     args=(frame.copy(), client, cached_food_dets)).start()

            # (C) DeepFace 表情 (使用全域 DeepFace)
            now = time.time()
            if (self.analysis_options.get("opt_emote") and 
                DeepFace is not None and  # 確保模組有載入
                not self._llm_busy and (now - self._last_emote_ts) > EMOTE_INTERVAL_SECONDS):
                
                self._llm_busy = True
                self._last_emote_ts = now
                threading.Thread(target=self._run_deepface_background, 
                                 args=(frame.copy(), face_detector)).start()

            if self._cached_plate_insight: result.plate_insight = self._cached_plate_insight
            if self._cached_token_usage:
                result.token_usage_event = self._cached_token_usage
                self._cached_token_usage = None 

            # 自動記錄 Log
            now = time.time()
            if (now - self._last_log_ts) > LOG_INTERVAL_SECONDS: 
                if current_people_count > 0 or cached_plate_label:
                    emotions_data = {self._cached_emotion: 1.0} if self._cached_emotion else {}
                    try:
                        insert_log(
                            source_type="live_stream",
                            people_count=current_people_count,
                            emotions=emotions_data,
                            food_detected=cached_plate_label if cached_plate_label else "無"
                        )
                        self._last_log_ts = now
                    except Exception: pass
            
            # 推送結果
            if self._analysis_result_queue.full():
                try: self._analysis_result_queue.get_nowait()
                except Empty: pass
            self._analysis_result_queue.put_nowait(result)
            
            # ★★★ 優化 2：微小延遲讓出 CPU，解決畫面卡頓 ★★★
            time.sleep(0.005)

    def _run_vlm_background(self, frame, client, food_detections):
        try:
            async def task():
                insight, usage = await llm.analyze_plate_vlm(frame, client) 
                return insight, usage
            insight, usage = asyncio.run(task())
            
            if insight:
                self._cached_plate_insight = insight 
                self._new_insight_arrived = True 
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

    def _run_deepface_background(self, frame, face_detector):
        # 這裡不需再 import DeepFace，直接使用全域變數
        try:
            face_crop = crop_face_with_mediapipe(frame, face_detector)
            if face_crop is None: return

            analysis = DeepFace.analyze(
                img_path=face_crop, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='skip', 
                silent=True
            )
            dominant = analysis[0]['dominant_emotion']
            mapping = {
                "happy": "開心", "neutral": "平淡", "sad": "失望", 
                "angry": "不滿", "surprise": "驚艷", "fear": "困惑", "disgust": "嫌棄"
            }
            final_emotion = mapping.get(dominant, dominant)

            # 同步更新快取 (給 DB/圖表) 和 鎖定 (給 UI 日誌)
            self._cached_emotion = final_emotion 
            with self._latch_lock:
                self._latched_emotion = final_emotion 
                
            print(f"✅ 情緒偵測: {final_emotion}")

        except Exception as e:
            print(f"DeepFace Error: {e}")
        finally:
            self._llm_busy = False

    def start(self):
        if self._camera_thread and self._camera_thread.is_alive(): return
        self._stop_event.clear()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self._camera_thread.start()
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        time.sleep(0.5)
        self._camera_thread = None
        self._worker_thread = None

    def get_latest_frame(self):
        try: return self._frame_display_queue.get_nowait()
        except Empty: return None

    @property
    def raw_session_id(self): return self.session_id

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
            
            # 取出情緒事件
            if self._latched_emotion:
                result.emotion_event = self._latched_emotion
                self._latched_emotion = None 

        return result