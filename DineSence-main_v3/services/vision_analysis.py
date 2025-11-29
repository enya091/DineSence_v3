# services/vision_analysis.py
"""
所有電腦視覺相關演算法集中於此：
- 餐盤殘留偵測
- 頭部動作偵測（點頭 / 搖頭）
- 臉部擷取
- YOLO 食物偵測
"""

import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from ultralytics import YOLO

import config as _cfg


# --------------------------------------------------
#  讀取 config 參數 (這裡我直接把預設值改成寬鬆版，確保你就算 config 沒改也能動)
# --------------------------------------------------

YOLO_MODEL_PATH  = getattr(_cfg, "YOLO_MODEL_PATH", "yolov8n.pt")
FOODISH_CLASSES  = getattr(_cfg, "FOODISH_CLASSES", set())

# Head Gesture 參數 (使用寬鬆版 Defaults)
GESTURE_BUFFER_LEN = getattr(_cfg, "GESTURE_BUFFER_LEN", 20)
GESTURE_COOLDOWN   = getattr(_cfg, "GESTURE_COOLDOWN_SECONDS", 0.8)
GESTURE_MIN_OFFSET = getattr(_cfg, "GESTURE_MIN_OFFSET", 0.002)

# ★★★ 這裡直接改寬鬆，不用怕 config 沒設對 ★★★
NOD_AMP_THRESH     = getattr(_cfg, "NOD_AMP_THRESH", 0.008)       # 很容易觸發
SHAKE_AMP_THRESH   = getattr(_cfg, "SHAKE_AMP_THRESH", 0.010)
MAX_SECONDARY_AMP  = getattr(_cfg, "MAX_SECONDARY_AMP", 0.020)    # 容許頭晃動
MIN_OSC_COUNT      = getattr(_cfg, "MIN_OSC_COUNT", 2)            # 來回兩次就算

MIN_FACE_CONF = getattr(_cfg, "MIN_FACE_CONF", 0.6)


# --------------------------------------------------
#  模型初始化
# --------------------------------------------------

try:
    _yolo_food = YOLO(YOLO_MODEL_PATH)
    _yolo_ok = True
except Exception as e:
    print(f"[YOLO 載入失敗] {e}")
    _yolo_food = None
    _yolo_ok = False

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection


def get_pose_detector():
    return mp_pose.Pose(
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def get_face_detector():
    return mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

# --------------------------------------------------
#  (A) 餐盤殘留偵測
# --------------------------------------------------

def estimate_plate_leftover(bgr_frame):
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
        param1=100, param2=30, minRadius=60, maxRadius=0
    )

    if circles is None:
        return "未偵測到餐盤", None, None

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = max(circles, key=lambda c: c[2])

    h, w = bgr_frame.shape[:2]
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return "餐盤不完整", None, (x, y, r)

    roi = bgr_frame[y-r:y+r, x-r:x+r].copy()

    mask = np.zeros((2*r, 2*r), dtype=np.uint8)
    cv2.circle(mask, (r, r), r - 2, 255, -1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_mask = (S < 50) & (V > 200)
    food_mask = (~white_mask) & (mask > 0)

    total = np.count_nonzero(mask)
    food_pixels = np.count_nonzero(food_mask)

    if total == 0:
        return "餐盤區域無效", None, (x, y, r)

    ratio = food_pixels / total
    label = "剩餘50%以上" if ratio >= 0.5 else "無剩餘"

    return label, float(ratio), (x, y, r)


# --------------------------------------------------
#  (B) 頭部動作偵測 (HeadGestureDetector)
# --------------------------------------------------

class HeadGestureDetector:
    """
    嚴格版頭部動作偵測：
    - 同時看 X / Y 相對位移的波形
    - 點頭：Y 振幅夠大，而且 X 幾乎沒動
    - 搖頭：X 振幅夠大，而且 Y 幾乎沒動
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

    def reset(self):
        self.x_hist.clear()
        self.y_hist.clear()
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
        【比例判定版】
        不再使用固定的 MAX_SECONDARY_AMP 卡死，
        而是比較 X 與 Y 的相對大小。
        """
        # 1. 去抖動
        if abs(dx) < self.min_offset: dx = 0.0
        if abs(dy) < self.min_offset: dy = 0.0

        self.x_hist.append(dx)
        self.y_hist.append(dy)

        if len(self.x_hist) < self.x_hist.maxlen:
            return None

        now = time.time()
        if (now - self.last_event_ts) < self.cooldown:
            return None

        # 3. 訊號處理
        arr_x = np.array(self.x_hist, dtype=np.float32)
        arr_y = np.array(self.y_hist, dtype=np.float32)

        arr_x = arr_x - arr_x.mean()
        arr_y = arr_y - arr_y.mean()

        arr_x = cv2.GaussianBlur(arr_x.reshape(-1, 1), (5, 1), 0).flatten()
        arr_y = cv2.GaussianBlur(arr_y.reshape(-1, 1), (5, 1), 0).flatten()

        # 4. 提取特徵
        amp_x, osc_x = self._osc_features(arr_x)
        amp_y, osc_y = self._osc_features(arr_y)

        event = None
        
        # ★★★ 這裡改成了比例判定 ★★★
        
        # 定義一個比率，例如主軸必須是副軸的 1.5 倍大
        RATIO = 1.2 

        # --- 判定點頭 (Y 為主) ---
        # 條件1: Y 振幅夠大
        # 條件2: Y 振幅 明顯大於 X 振幅 (不再用固定值卡)
        # 條件3: 有來回動作
        if (amp_y >= self.nod_amp_thresh and 
            amp_y > (amp_x * RATIO) and 
            osc_y >= self.min_osc):
            event = "nod"

        # --- 判定搖頭 (X 為主) ---
        elif (amp_x >= self.shake_amp_thresh and 
              amp_x > (amp_y * RATIO) and 
              osc_x >= self.min_osc):
            event = "shake"

        # Debug 輸出
        if amp_y > 0.01 or amp_x > 0.01:
            print(f"[Debug] 判定:{event} | Y:{amp_y:.4f} vs X:{amp_x:.4f} | (比率檢查: {'Pass' if event else 'Fail'})")

        if event:
            self.last_event_ts = now
            print(f"🚀 觸發動作: {event} !!!")
        
        return event


# --------------------------------------------------
#  (C) 臉部擷取
# --------------------------------------------------

def crop_face_with_mediapipe(bgr_frame, detector, min_conf=0.6):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    if not res.detections:
        return None

    det = res.detections[0]
    if det.score[0] < min_conf:
        return None

    h, w = bgr_frame.shape[:2]
    bbox = det.location_data.relative_bounding_box

    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, int((bbox.xmin + bbox.width) * w))
    y2 = min(h, int((bbox.ymin + bbox.height) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    return bgr_frame[y1:y2, x1:x2]


# --------------------------------------------------
#  (D) YOLO 食物偵測
# --------------------------------------------------

def detect_food_regions_yolo(bgr, conf=0.3, min_area_ratio=0.01):
    if not _yolo_ok:
        return []

    res = _yolo_food(bgr, conf=conf, iou=0.45, verbose=False)[0]
    h, w = bgr.shape[:2]

    out = []
    for b in res.boxes:
        name = res.names.get(int(b.cls.item()), "")
        if name not in FOODISH_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if area / (w * h) < min_area_ratio:
            continue

        out.append({
            "xyxy": (x1, y1, x2, y2),
            "label": name,
            "conf": float(b.conf.item()),
        })

    return out


def has_big_cup(bgr, min_area_ratio=0.04):
    if not _yolo_ok:
        return False

    res = _yolo_food(bgr, conf=0.3, iou=0.45, verbose=False)[0]
    h, w = bgr.shape[:2]

    for b in res.boxes:
        name = res.names.get(int(b.cls.item()), "")
        if name in ["cup", "wine glass", "bottle"]:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            if ((x2 - x1) * (y2 - y1)) / (w * h) >= min_area_ratio:
                return True

    return False