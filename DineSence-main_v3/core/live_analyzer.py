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

from services.llm_handler import (
    analyze_plate_vlm, 
    identify_food_item  # ★ [NEW] 記得引入這個！
)

EMOTE_INTERVAL_SECONDS   = getattr(_cfg, "EMOTE_INTERVAL_SECONDS", 2.0)
CAMERA_RESOLUTION_WIDTH  = getattr(_cfg, "CAMERA_RESOLUTION_WIDTH", 1280)
CAMERA_RESOLUTION_HEIGHT = getattr(_cfg, "CAMERA_RESOLUTION_HEIGHT", 720)
CAMERA_BUFFER_SIZE       = getattr(_cfg, "CAMERA_BUFFER_SIZE", 1)
FACE_CAM_INDEX  = getattr(_cfg, "FACE_CAM_INDEX", 1)  # 預設 0 = Mac 筆電鏡頭
PLATE_CAM_INDEX = getattr(_cfg, "PLATE_CAM_INDEX", 0)  # 預設 1 = 手機鏡頭（Camo 等）


VLM_INTERVAL_SECONDS = 10.0
LOG_INTERVAL_SECONDS = 5.0 
EVIDENCE_DIR = "session_evidence" 

class LiveAnalyzer:
    def __init__(self, model_pack: dict, menu_items: list, analysis_options: dict, db_manager):
        self.model_pack = model_pack
        self.menu_items = menu_items
        self.analysis_options = analysis_options
        
        self._face_display_queue = Queue(maxsize=1)
        self._face_analysis_queue = Queue(maxsize=1)
        
        self._plate_display_queue = Queue(maxsize=1)
        self._plate_analysis_queue = Queue(maxsize=1)
        
        # 結果佇列維持一個，因為我們要合併結果傳給 UI
        self._analysis_result_queue = Queue(maxsize=1)

        self.gesture_detector = HeadGestureDetector()

        self._stop_event = threading.Event()
        
        # [修改] 準備兩個相機執行緒
        self._face_cam_thread = None
        self._plate_cam_thread = None
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

        # 人臉相關狀態
        self._current_people_count = 0
        
        # 餐盤相關狀態
        self._cached_plate_label = None
        self._cached_plate_ratio = None
        self._cached_plate_circle = None
        self._cached_food_detections = []
        
        # 輔助變數
        self._frame_count = 0
        self._last_debug_print_ts = 0
        self._cross_capture_signal = None
        

    # -------------------------------------------------
    #  執行緒 1：攝影機
    # -------------------------------------------------
    def _open_camera(self, index, width, height):
        system_os = platform.system()
        print(f"📷 正在開啟鏡頭 ID {index} (系統: {system_os})...")
        cap = None
        
        if system_os == "Darwin": # macOS
             cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        elif system_os == "Windows": # Windows
             cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(index) # 失敗退回預設

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
        return cap

    # [修改] 人臉鏡頭迴圈 (筆電鏡頭)
    def _face_cam_loop(self):
        print(f"[DEBUG] Face camera using index = {FACE_CAM_INDEX}")
        # 假設 0 是筆電鏡頭，解析度 1280x720
        cap = self._open_camera(FACE_CAM_INDEX, 1280, 720) 
        
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1); continue

            # 放入 Face 的佇列
            if self._face_display_queue.full(): 
                try: self._face_display_queue.get_nowait()
                except Empty: pass
            self._face_display_queue.put_nowait(frame)

            if self._face_analysis_queue.full():
                try: self._face_analysis_queue.get_nowait()
                except Empty: pass
            self._face_analysis_queue.put_nowait(frame)
            
            time.sleep(0.005)
        if cap: cap.release()

    # [修改] 餐盤鏡頭迴圈 (外接鏡頭)
    def _plate_cam_loop(self):
        print(f"[DEBUG] Plate camera using index = {PLATE_CAM_INDEX}")
        # 假設 1 是外接鏡頭，解析度可以用高一點例如 1920x1080 看清楚食物
        cap = self._open_camera(PLATE_CAM_INDEX, 1920, 1080)
        
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1); continue

            # 放入 Plate 的佇列
            if self._plate_display_queue.full():
                try: self._plate_display_queue.get_nowait()
                except Empty: pass
            self._plate_display_queue.put_nowait(frame)

            if self._plate_analysis_queue.full():
                try: self._plate_analysis_queue.get_nowait()
                except Empty: pass
            self._plate_analysis_queue.put_nowait(frame)

            time.sleep(0.005)
        if cap: cap.release()

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

    def _process_face_task(self, frame, result):
        """處理人臉鏡頭的邏輯：人數、動作、表情"""
        face_detector = self.model_pack.get("face_detector")
        pose_detector = self.model_pack.get("pose_detector")
        
        # (A) 計算人數
        if face_detector:
            try:
                small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_results = face_detector.process(rgb_frame)
                self._current_people_count = len(face_results.detections) if face_results.detections else 0
            except Exception: pass
        
        result.display_info["people_count"] = self._current_people_count

        # (B) 點頭/搖頭偵測
        if self.analysis_options.get("opt_nod") and pose_detector:
            try:
                small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                res = pose_detector.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    dx = lm[0].x - (lm[7].x + lm[8].x + lm[11].x + lm[12].x) / 4.0
                    dy = lm[0].y - (lm[7].y + lm[8].y + lm[11].y + lm[12].y) / 4.0
                    
                    event = self.gesture_detector.update_and_classify(dx, dy)
                    
                    if event == "nod":
                        with self._latch_lock: self._latched_nod = True
                        self._save_evidence("nod", frame.copy(), self._frame_count)
                    elif event == "shake":
                        with self._latch_lock: self._latched_shake = True
                        self._save_evidence("shake", frame.copy(), self._frame_count)
            except Exception: pass

        # (C) 情緒偵測 (觸發背景執行緒)
        now = time.time()
        if (self.analysis_options.get("opt_emote") and 
            DeepFace is not None and 
            not self._llm_busy and 
            (now - self._last_emote_ts) > EMOTE_INTERVAL_SECONDS):
            
            self._llm_busy = True
            self._last_emote_ts = now
            threading.Thread(target=self._run_deepface_background, 
                             args=(frame.copy(), face_detector)).start()
    # -------------------------------------------------

    def _process_plate_task(self, frame, result, client):
        """處理餐盤鏡頭的邏輯：剩食計算、VLM 觸發"""
        if not self.analysis_options.get("opt_plate"):
            return

        # (A) 基礎演算法 (每 15 幀更新一次快取)
        if self._frame_count % 15 == 0:
            try:
                label, ratio, circle = estimate_plate_leftover(frame)
                if label in ["剩餘50%以上", "無剩餘"]:
                    self._cached_plate_label = label
                    self._cached_plate_ratio = ratio 
                else:
                    self._cached_plate_label = None 
                    self._cached_plate_ratio = None
                self._cached_plate_circle = circle
                # self._cached_food_detections = detect_food_regions_yolo(frame)
            except Exception: pass
        
        # 填入 Result
        if self._cached_plate_label:
            result.plate_event = self._cached_plate_label 
            display_text = f"{self._cached_plate_label} ({self._cached_plate_ratio:.0%})" \
                           if self._cached_plate_ratio else self._cached_plate_label
            result.display_info["plate_label"] = display_text

        if self._cached_plate_circle: 
            result.display_info["plate_circle"] = self._cached_plate_circle
        
        result.display_info["food_detections"] = self._cached_food_detections

        # (B) VLM 觸發判斷
        now = time.time()
        should_trigger = (self._cached_plate_label is not None or len(self._cached_food_detections) > 0)
        is_cooldown = (now - self._last_vlm_ts) < VLM_INTERVAL_SECONDS
        
        # Debug 訊息
        if should_trigger and (now - self._last_debug_print_ts > 3.0):
            if not client: print("⚠️ [VLM Warning] 未設定 OpenAI API Key")
            elif self._vlm_busy: print("⏳ [VLM Skip] 系統忙碌中")
            self._last_debug_print_ts = now

        if should_trigger and client and not self._vlm_busy and not is_cooldown:
            self._vlm_busy = True 
            self._last_vlm_ts = now 
            print(f"🚀 VLM 觸發成功!")
            self._save_evidence("plate_vlm", frame.copy(), self._frame_count)
            threading.Thread(target=self._run_vlm_background, 
                             args=(frame.copy(), client, self._cached_food_detections)).start()
            
    def _sync_log_task(self):
        """檢查並執行資料同步儲存"""
        now = time.time()
        if (now - self._last_log_ts) > LOG_INTERVAL_SECONDS: 
            # 只有當有人或有餐盤狀態時才紀錄
            if self._current_people_count > 0 or self._cached_plate_label:
                
                emotions_data = {self._cached_emotion: 1.0} if self._cached_emotion else {}
                food_data = self._cached_plate_label if self._cached_plate_label else "無"

                try:
                    insert_log(
                        source_type="live_dual_cam",
                        people_count=self._current_people_count,
                        emotions=emotions_data,
                        food_detected=food_data
                    )
                    self._last_log_ts = now
                except Exception as e: 
                    print(f"Log Error: {e}")

    # [新增] 輔助函式：存檔用 (放在類別內)
    def _save_custom_file(self, filename, frame):
        try:
            path = os.path.join(EVIDENCE_DIR, filename)
            cv2.imwrite(path, frame)
            return path
        except Exception: return None

    #  執行緒 2：CV 分析
    # -------------------------------------------------
    def _analysis_worker(self):
        client = self.model_pack.get("client")
        
        while not self._stop_event.is_set():
            # 1. 獲取畫面
            face_frame = None
            plate_frame = None
            try: face_frame = self._face_analysis_queue.get_nowait()
            except Empty: pass
            try: plate_frame = self._plate_analysis_queue.get_nowait()
            except Empty: pass

            if face_frame is None and plate_frame is None:
                time.sleep(0.005); continue

            result = AnalysisResult() 
            self._frame_count += 1

            # 2. 執行任務 (模組化)
            if face_frame is not None:
                self._process_face_task(face_frame, result)
            
            if plate_frame is not None:
                self._process_plate_task(plate_frame, result, client)

            # =========================================================
            # ★★★ [新增] 處理強烈情緒的「雙鏡頭連拍」 ★★★
            # =========================================================
            if self._cross_capture_signal:
                signal = self._cross_capture_signal
                self._cross_capture_signal = None 
                
                if face_frame is not None and plate_frame is not None:
                    try:
                        # (A) 準備檔名資訊
                        now = datetime.now()
                        readable_ts = now.strftime("%m月%d日_%H點%M分%S秒")
                        e1_name, e1_score = signal["top1"]
                        e2_name, e2_score = signal["top2"]
                        emo_tag_1 = f"{e1_name}-{int(e1_score)}"
                        emo_tag_2 = f"{e2_name}-{int(e2_score)}"
                        
                        # (B) 處理人臉 (Face) - 保持簡單，直接存
                        face_filename = f"{readable_ts}_{emo_tag_1}_{emo_tag_2}_Face.jpg"
                        face_path = self._save_custom_file(face_filename, face_frame)
                        if face_path:
                            self.db_manager.save_event_evidence(
                                self.session_id, "strong_emotion_face", face_path
                            )
                        
                        # (C) 處理餐盤 (Plate) - ★ [修改] 啟動背景辨識
                        plate_filename = f"{readable_ts}_{emo_tag_1}_{emo_tag_2}_Plate.jpg"
                        plate_path = self._save_custom_file(plate_filename, plate_frame)
                        
                        if plate_path:
                            print(f"📸 雙鏡頭快照完成，正在背景辨識食物...")
                            # 啟動一條新執行緒去跑 LLM，避免卡住主畫面
                            threading.Thread(
                                target=self._background_identify_and_save,
                                args=(plate_frame.copy(), plate_path, self.session_id)
                            ).start()
                            
                    except Exception as e:
                        print(f"Snapshot Error: {e}")
            # =========================================================

            # 3. 處理異步回傳的資料
            if self._cached_plate_insight: 
                result.plate_insight = self._cached_plate_insight
                self._cached_plate_insight = None
                
            if self._cached_token_usage:
                result.token_usage_event = self._cached_token_usage
                self._cached_token_usage = None 

            # 4. 同步儲存 Log
            self._sync_log_task()
            
            # 5. 推送結果
            if self._analysis_result_queue.full():
                try: self._analysis_result_queue.get_nowait()
                except Empty: pass
            self._analysis_result_queue.put_nowait(result)
            
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

    def _background_identify_and_save(self, frame, local_path, session_id):
        """
        [NEW] 背景任務：辨識食物並存入 DB
        """
        try:
            client = self.model_pack.get("client")
            # 定義候選菜單 (您可以從 config 或外部傳入)
            menu_list = self.menu_items if self.menu_items else ["漢堡", "雞塊", "薯條"]
            
            # 1. 呼叫 LLM 辨識 (同步執行 async 函式)
            # 因為這是在獨立的 Thread 跑，所以用 asyncio.run 是安全的
            food_name = asyncio.run(identify_food_item(frame, menu_list, client))
            
            print(f"🍔 [AI 辨識結果] {food_name}")

            # 2. 存入資料庫 (帶有 food_label)
            self.db_manager.save_event_evidence(
                session_id=session_id, 
                event_type="strong_emotion_plate", 
                local_path=local_path,
                food_label=food_name  # ★ 把辨識結果存進去
            )
            
        except Exception as e:
            print(f"Food ID Task Error: {e}")
            # 失敗也要存，但 label 是 None
            self.db_manager.save_event_evidence(session_id, "strong_emotion_plate", local_path, "Unknown")

    def _run_deepface_background(self, frame, face_detector):
        try:
            face_crop = crop_face_with_mediapipe(frame, face_detector)
            if face_crop is None: return

            # 1. 執行分析
            analysis = DeepFace.analyze(
                img_path=face_crop, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='skip', 
                silent=True
            )
            
            result = analysis[0]
            emotions_dict = result['emotion'] # 取得所有情緒的分數字典
            
            # 排序：由高到低 [(emotion, score), ...]
            sorted_emotions = sorted(emotions_dict.items(), key=lambda item: item[1], reverse=True)
            
            # 第一名
            top1_name = sorted_emotions[0][0]
            top1_score = sorted_emotions[0][1]
            
            # 第二名 (以防萬一只有一個，做個檢查)
            top2_name = sorted_emotions[1][0] if len(sorted_emotions) > 1 else "neutral"
            top2_score = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0.0

            # 3. 中文映射 (Mapping)
            mapping = {
                "happy": "開心", "neutral": "平淡", "sad": "失望", 
                "angry": "不滿", "surprise": "驚艷", "fear": "困惑", "disgust": "嫌棄"
            }
            top1_zh = mapping.get(top1_name, top1_name)
            top2_zh = mapping.get(top2_name, top2_name)

            # ★★★ [重點 1] 更新快取 (給 DB 用)：保持純文字 ★★★
            self._cached_emotion = top1_zh 
            
            # 4. 強烈情緒觸發邏輯
            # 條件：第一名不是平淡，且分數 > 40% (您設定的值)
            INTENSITY_THRESHOLD = 40.0 
            
            if top1_name != "neutral" and top1_score > INTENSITY_THRESHOLD:
                print(f"🔥 強烈情緒: {top1_zh}({top1_score:.0f}%) / {top2_zh}({top2_score:.0f}%)")
                
                # 發送訊號：傳遞更完整的資訊
                self._cross_capture_signal = {
                    "top1": (top1_zh, top1_score), 
                    "top2": (top2_zh, top2_score) 
                }

            # ★★★ [重點 2] Log 鎖定 (給 UI 用)：加上分數 ★★★
            with self._latch_lock:
                # 這裡改成 formatted string，讓介面顯示 "開心 (98%)"
                self._latched_emotion = f"{top1_zh} ({top1_score:.0f}%)"
                
        except Exception as e:
            print(f"DeepFace Error: {e}")
        finally:
            self._llm_busy = False

    def start(self):
        if self._face_cam_thread and self._face_cam_thread.is_alive(): return
        self._stop_event.clear()
        
        # [修改] 啟動兩個相機執行緒 + 一個分析執行緒
        self._face_cam_thread = threading.Thread(target=self._face_cam_loop, daemon=True)
        self._plate_cam_thread = threading.Thread(target=self._plate_cam_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        
        self._face_cam_thread.start()
        self._plate_cam_thread.start()
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        time.sleep(0.5)
        self._face_cam_thread = None
        self._plate_cam_thread = None
        self._worker_thread = None

    # [修改] 回傳兩張圖 (Face, Plate)
    def get_latest_frames(self):
        f_frame = None
        p_frame = None
        try: f_frame = self._face_display_queue.get_nowait()
        except Empty: pass
        try: p_frame = self._plate_display_queue.get_nowait()
        except Empty: pass
        return f_frame, p_frame

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