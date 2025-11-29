# services/detectors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import time
from collections import deque
import numpy as np
import cv2

# --------------------------------------------------
#  Config Defaults (Updated to Strict Version)
# --------------------------------------------------

# 通用冷卻時間
COOLDOWN_SECONDS = 0.8        # 原本 1.0 -> 改短一點，測試比較方便

# 頭部動作偵測參數
GESTURE_BUFFER_LEN = 20

# ★★★ 關鍵修改區 ★★★
NOD_AMP_THRESH = 0.008        # 原本 0.015 -> 改 0.008 (讓小幅度點頭也能觸發)
SHAKE_AMP_THRESH = 0.010      # 原本 0.020 -> 改 0.010
MAX_SECONDARY_AMP = 0.020     # 原本 0.010 -> 改 0.020 (這是重點！容許頭部不自覺的左右晃動)
MIN_OSC_COUNT = 2             # 原本 3 -> 改 2 (來回一次就觸發)

GESTURE_COOLDOWN = 0.8
GESTURE_MIN_OFFSET = 0.002

# 其他偵測參數
HAND_RAISE_MARGIN = 30
LEAN_FWD_TORSO_DEG = 12
LEAN_BACK_TORSO_DEG = -8
MIN_FACE_CONF = 0.6


@dataclass
class BehaviorEvent:
    ts: float
    event_type: str
    subject_id: Optional[int]
    zone: Optional[str]
    score: Optional[float]
    meta: Optional[Dict[str, Any]] = None


# --------------------------------------------------
#  New: Unified Head Gesture Detector
# --------------------------------------------------
class HeadGestureDetector:
    """
    嚴格版頭部動作偵測 (取代原本分開的 Nod/Shake 偵測器)
    同時監控 X 與 Y 軸，確保動作單純才觸發。
    """
    def __init__(
        self,
        buf_len=GESTURE_BUFFER_LEN,
        nod_amp_thresh=NOD_AMP_THRESH,
        shake_amp_thresh=SHAKE_AMP_THRESH,
        max_secondary_amp=MAX_SECONDARY_AMP,
        min_osc=MIN_OSC_COUNT,
        cooldown=GESTURE_COOLDOWN,
        min_offset=GESTURE_MIN_OFFSET,
    ):
        self.buf_len = buf_len
        self.nod_amp_thresh = nod_amp_thresh
        self.shake_amp_thresh = shake_amp_thresh
        self.max_secondary_amp = max_secondary_amp
        self.min_osc = min_osc
        self.cooldown = cooldown
        self.min_offset = min_offset

        self.x_hist = deque(maxlen=buf_len)
        self.y_hist = deque(maxlen=buf_len)
        self.last_event_ts = 0.0

    def _osc_features(self, arr: np.ndarray):
        if arr.size < 3:
            return 0.0, 0
        amp = float(arr.max() - arr.min())
        diff1 = np.diff(arr)
        sign_changes = int(np.sum(np.diff(np.sign(diff1)) != 0))
        return amp, sign_changes

    def update_and_classify(self, dx, dy):
        """
        輸入: dx, dy (相對位移)
        輸出: "nod", "shake" 或 None
        """
        # 1. 去抖動
        if abs(dx) < self.min_offset: dx = 0.0
        if abs(dy) < self.min_offset: dy = 0.0

        self.x_hist.append(dx)
        self.y_hist.append(dy)

        # 2. 資料不足或冷卻中則跳過
        if len(self.x_hist) < self.x_hist.maxlen:
            return None

        now = time.time()
        if (now - self.last_event_ts) < self.cooldown:
            return None

        # 3. 訊號處理
        arr_x = np.array(self.x_hist, dtype=np.float32)
        arr_y = np.array(self.y_hist, dtype=np.float32)

        # 去掉平均值
        arr_x = arr_x - arr_x.mean()
        arr_y = arr_y - arr_y.mean()

        # 平滑化
        arr_x = cv2.GaussianBlur(arr_x.reshape(-1, 1), (5, 1), 0).flatten()
        arr_y = cv2.GaussianBlur(arr_y.reshape(-1, 1), (5, 1), 0).flatten()

        # 4. 提取特徵
        amp_x, osc_x = self._osc_features(arr_x)
        amp_y, osc_y = self._osc_features(arr_y)

        event = None
        
        # 5. 判定邏輯
        # --- 檢查點頭條件 ---
        if (amp_y >= self.nod_amp_thresh and 
            amp_x <= self.max_secondary_amp and 
            osc_y >= self.min_osc):
            event = "nod"

        # --- 檢查搖頭條件 ---
        elif (amp_x >= self.shake_amp_thresh and 
              amp_y <= self.max_secondary_amp and 
              osc_x >= self.min_osc):
            event = "shake"

        # 6. Debug 輸出
        if amp_y > 0.005 or amp_x > 0.005:
            # 這裡可以幫你檢查為何沒觸發
            print(f"[Debug] 判定:{event} | Y:{amp_y:.4f}/{self.nod_amp_thresh} (次數:{osc_y}) | X:{amp_x:.4f}/{self.max_secondary_amp}")

        if event:
            self.last_event_ts = now
            print(f"🚀 觸發動作: {event} !!!")
        
        return event


# --------------------------------------------------
#  Other Detectors
# --------------------------------------------------

def crop_face_with_mediapipe(bgr_frame, detector, min_conf:float=MIN_FACE_CONF):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if not res.detections:
        return None, None
    det = res.detections[0]
    score = det.score[0] if hasattr(det, "score") else getattr(det, "confidence", 0.0)
    if score < min_conf:
        return None, None
    h, w = bgr_frame.shape[:2]
    bbox = det.location_data.relative_bounding_box
    x1 = max(0, int(bbox.xmin * w)); y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, int((bbox.xmin + bbox.width) * w)); y2 = min(h, int((bbox.ymin + bbox.height) * h))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return bgr_frame[y1:y2, x1:x2], (x1, y1, x2, y2)

class PostureClassifier:
    def __init__(self, fwd_deg:float=LEAN_FWD_TORSO_DEG, back_deg:float=LEAN_BACK_TORSO_DEG):
        self.fwd_deg = fwd_deg
        self.back_deg = back_deg

    @staticmethod
    def _deg(a, b):
        a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
        dot = float(np.dot(a,b)); na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        cosv = np.clip(dot/(na*nb), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosv)))

    def classify(self, landmarks, img_h:int, img_w:int):
        LSH, RSH, LHP, RHP = 11,12,23,24
        try:
            Ls = landmarks[LSH]; Rs = landmarks[RSH]; Lh = landmarks[LHP]; Rh = landmarks[RHP]
        except Exception:
            return "neutral", 0.0
        sh_mid = ((Ls.x+Rs.x)/2*img_w, (Ls.y+Rs.y)/2*img_h)
        hp_mid = ((Lh.x+Rh.x)/2*img_w, (Lh.y+Rh.y)/2*img_h)
        v = (hp_mid[0]-sh_mid[0], hp_mid[1]-sh_mid[1])
        angle = self._deg(v, (0,1))
        signed = (1 if v[0] > 0 else -1) * angle
        if signed >= self.fwd_deg: return "lean_forward", signed
        if signed <= self.back_deg: return "lean_back", signed
        return "neutral", signed

class HandRaiseDetector:
    # 這裡現在可以正確抓到全域的 COOLDOWN_SECONDS 了
    def __init__(self, margin:int=HAND_RAISE_MARGIN, cooldown:float=COOLDOWN_SECONDS):
        self.margin = margin
        self.last_ts = 0.0
        self.cooldown = cooldown

    def update_and_check(self, landmarks, img_h:int, img_w:int):
        LWR, RWR, LSH, RSH = 15,16,11,12
        try:
            lw_y = landmarks[LWR].y * img_h; rw_y = landmarks[RWR].y * img_h
            ls_y = landmarks[LSH].y * img_h; rs_y = landmarks[RSH].y * img_h
        except Exception:
            return False, None
        now = time.time()
        if (now - self.last_ts) <= self.cooldown:
            return False, None
        if lw_y + self.margin < ls_y:
            self.last_ts = now; return True, "left"
        if rw_y + self.margin < rs_y:
            self.last_ts = now; return True, "right"
        return False, None

class ExpressionEstimator:
    def __init__(self, external_clf=None, use_heuristic:bool=True):
        self.external = external_clf; self.use_heur = use_heuristic

    def predict(self, face_bgr, facemesh_landmarks=None):
        if self.external is not None:
            return self.external.predict(face_bgr)
        # fallback heuristic
        return {"label":"neutral","score":0.5}


# --------------------------------------------------
#  Main Aggregator Class
# --------------------------------------------------
class BodyEmotionDetector:
    def __init__(self):
        # [Updated] Use the single HeadGestureDetector instead of separate ones
        self.head_gesture = HeadGestureDetector()
        
        self.posture = PostureClassifier()
        self.hand_raise = HandRaiseDetector()
        self.expr = ExpressionEstimator()

    def process_frame(self, frame_bgr, *, face_detector, pose_landmarks=None,
                      nose_xy: Optional[Tuple[float,float]]=None, ears_xy: Optional[Tuple[Tuple[float,float],Tuple[float,float]]]=None,
                      zone: Optional[str]=None, face_mesh_points: Optional[dict]=None, subject_id: Optional[int]=None):
        h, w = frame_bgr.shape[:2]
        events: List[BehaviorEvent] = []

        # 1. Face Crop & Expression
        face_roi, bbox = crop_face_with_mediapipe(frame_bgr, face_detector, MIN_FACE_CONF)
        expr = None
        if face_roi is not None:
            expr = self.expr.predict(face_roi, facemesh_landmarks=face_mesh_points)
            if expr and expr.get("label") in ("smile","frown"):
                events.append(BehaviorEvent(ts=time.time(), event_type=expr["label"], subject_id=subject_id, zone=zone, score=expr.get("score")))

        # 2. [Updated] Head Gesture (Nod / Shake)
        if nose_xy and ears_xy:
            nose_x, nose_y = nose_xy
            
            ref_x = (ears_xy[0][0] + ears_xy[1][0]) / 2.0
            ref_y = (ears_xy[0][1] + ears_xy[1][1]) / 2.0
            
            dx = nose_x - ref_x
            dy = nose_y - ref_y
            
            gesture = self.head_gesture.update_and_classify(dx, dy)
            
            if gesture == "nod":
                events.append(BehaviorEvent(ts=time.time(), event_type="nod", subject_id=subject_id, zone=zone, score=1.0))
            elif gesture == "shake":
                events.append(BehaviorEvent(ts=time.time(), event_type="shake", subject_id=subject_id, zone=zone, score=1.0))

        # 3. Posture & Hand Raise
        if pose_landmarks is not None:
            label, ang = self.posture.classify(pose_landmarks, h, w)
            if label in ("lean_forward","lean_back"):
                events.append(BehaviorEvent(ts=time.time(), event_type=label, subject_id=subject_id, zone=zone, score=ang))
            raised, side = self.hand_raise.update_and_check(pose_landmarks, h, w)
            if raised:
                events.append(BehaviorEvent(ts=time.time(), event_type="hand_raise", subject_id=subject_id, zone=zone, score=1.0, meta={"side":side}))

        # 4. Logic: If shake detected and not smiling, imply "dislike"
        if any(e.event_type == "shake" for e in events):
            lbl = (expr or {}).get("label","neutral")
            if lbl != "smile":
                events.append(BehaviorEvent(ts=time.time(), event_type="dislike", subject_id=subject_id, zone=zone, score=0.8))

        return events, bbox, (expr or {"label":"neutral","score":0.5})