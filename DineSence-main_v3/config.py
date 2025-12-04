# config.py

import os
from dotenv import load_dotenv
load_dotenv()

DASH_USER = "admin"
DASH_PASS = "1234"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------
# LLM Setting
# ---------------------
LLM_MODEL_EMOTION = "gpt-4o-mini"
LLM_MODEL_SUMMARY = "gpt-4o-mini"

# ---------------------
# Camera Setting
# ---------------------
# 鏡頭設定
# 建議：0 通常是內建鏡頭 (Face)，1 通常是外接 USB 鏡頭 (Plate)
CAMERA_INDEX_FACE = 1   
CAMERA_INDEX_PLATE = 0

# 給 LiveAnalyzer 用的索引（保持一致）
FACE_CAM_INDEX = CAMERA_INDEX_FACE   # = 0
PLATE_CAM_INDEX = CAMERA_INDEX_PLATE # = 1

# 解析度可以分開設定，例如餐盤需要高畫質看細節，人臉只需要 720p
FACE_CAM_RES = (1280, 720)
PLATE_CAM_RES = (1920, 1080)

CAMERA_RESOLUTION_WIDTH = 1280
CAMERA_RESOLUTION_HEIGHT = 720
CAMERA_BUFFER_SIZE = 2

EMOTE_INTERVAL_SECONDS = 1.5

# ---------------------
# Head Gesture Detection (Nod & Shake)
# ---------------------
# 採用嚴格版偵測參數
GESTURE_BUFFER_LEN = 20           # 緩衝長度
GESTURE_COOLDOWN_SECONDS = 1.0    # 冷卻時間
GESTURE_MIN_OFFSET = 0.002        # 去抖動門檻

# 判定參數
NOD_AMP_THRESH = 0.015            # Y 軸振幅門檻 (點頭)
SHAKE_AMP_THRESH = 0.020          # X 軸振幅門檻 (搖頭)
MAX_SECONDARY_AMP = 0.010         # 非主軸的最大容許振幅 (超過則視為模糊動作)
MIN_OSC_COUNT = 3                 # 最小來回次數

# --------------------
# Plate Detection
# --------------------
YOLO_MODEL_PATH = "yolov8n.pt"
FOODISH_CLASSES = {
    "cake", "pizza", "hot dog", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "bowl", "cup", "wine glass", "bottle", "spoon",
    "fork", "knife", "plate"
}