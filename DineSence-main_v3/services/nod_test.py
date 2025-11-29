import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque


class HeadGestureDetectorStrict:
    """
    嚴格版頭部動作偵測：
    - 同時看 X / Y 相對位移的波形
    - 點頭：Y 振幅夠大，而且 X 幾乎沒動
    - 搖頭：X 振幅夠大，而且 Y 幾乎沒動
    - 只要兩邊一起動得很大，就當「模糊動作」→ 不判
    """

    def __init__(
        self,
        buf_len=20,
        nod_amp_thresh=0.01,       # Y 振幅至少要到這個才算點頭
        shake_amp_thresh=0.015,     # X 振幅至少要到這個才算搖頭
        max_secondary_amp=0.007,    # 另一個軸的振幅若超過這個，就不判（太斜）
        min_osc=3,                  # 至少有幾次方向反轉（來回）
        cooldown=1.0,
        min_offset=0.002,           # 小於這個就視為 0（去抖動）
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

    def update_and_classify(self, dx: float, dy: float):
        """
        dx, dy：鼻子相對頭部中心的位移 (nose - ref)，0~1 空間
        回傳：
          - "nod"   → 點頭
          - "shake" → 搖頭
          - None    → 沒有明確動作
        """
        # 去掉極小抖動
        if abs(dx) < self.min_offset:
            dx = 0.0
        if abs(dy) < self.min_offset:
            dy = 0.0

        self.x_hist.append(dx)
        self.y_hist.append(dy)

        if len(self.x_hist) < self.x_hist.maxlen:
            return None

        now = time.time()
        if (now - self.last_event_ts) < self.cooldown:
            return None

        arr_x = np.array(self.x_hist, dtype=np.float32)
        arr_y = np.array(self.y_hist, dtype=np.float32)

        # 移除視窗內平均值，避免慢慢飄移
        arr_x = arr_x - arr_x.mean()
        arr_y = arr_y - arr_y.mean()

        # 簡單平滑
        arr_x = cv2.GaussianBlur(arr_x.reshape(-1, 1), (5, 1), 0).flatten()
        arr_y = cv2.GaussianBlur(arr_y.reshape(-1, 1), (5, 1), 0).flatten()

        amp_x, osc_x = self._osc_features(arr_x)
        amp_y, osc_y = self._osc_features(arr_y)

        # Debug 想看可以解開這行：
        # print(f"amp_x={amp_x:.4f}, amp_y={amp_y:.4f}, osc_x={osc_x}, osc_y={osc_y}")

        event = None

        # ✅ 點頭判斷：Y 很大，而且 X 很小，而且 Y 有足夠的來回次數
        if (
            amp_y >= self.nod_amp_thresh
            and amp_x <= self.max_secondary_amp
            and osc_y >= self.min_osc
        ):
            event = "nod"

        # ✅ 搖頭判斷：X 很大，而且 Y 很小，而且 X 有足夠的來回次數
        elif (
            amp_x >= self.shake_amp_thresh
            and amp_y <= self.max_secondary_amp
            and osc_x >= self.min_osc
        ):
            event = "shake"

        # 否則都視為模糊動作：不判
        if event is not None:
            self.last_event_ts = now

        return event


def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    detector = HeadGestureDetectorStrict(
        buf_len=20,
        nod_amp_thresh=0.015,
        shake_amp_thresh=0.020,
        max_secondary_amp=0.010,
        min_osc=3,
        cooldown=1.0,
        min_offset=0.002,
    )

    nod_count = 0
    shake_count = 0
    last_label = ""
    last_event_ts = 0.0

    print("啟動中：請面向鏡頭保持自然姿勢幾秒，先收一點 baseline...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取畫面")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        status_text = "Waiting for pose..."
        status_color = (0, 0, 255)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            nose = lm[0]
            left_ear = lm[7]
            right_ear = lm[8]
            left_shoulder = lm[11]
            right_shoulder = lm[12]

            nose_x = nose.x
            nose_y = nose.y

            ref_x = (left_ear.x + right_ear.x + left_shoulder.x + right_shoulder.x) / 4.0
            ref_y = (left_ear.y + right_ear.y + left_shoulder.y + right_shoulder.y) / 4.0

            dx = nose_x - ref_x
            dy = nose_y - ref_y

            # 畫點 + 線方便你肉眼看
            nose_px = (int(nose_x * w), int(nose_y * h))
            ref_px = (int(ref_x * w), int(ref_y * h))
            cv2.circle(frame, nose_px, 6, (0, 255, 0), -1)
            cv2.circle(frame, ref_px, 6, (255, 0, 0), -1)
            cv2.line(frame, nose_px, ref_px, (255, 255, 0), 2)

            cv2.putText(
                frame,
                f"dx: {dx:.4f}",
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                frame,
                f"dy: {dy:.4f}",
                (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            status_text = "Tracking..."
            status_color = (0, 255, 0)

            event = detector.update_and_classify(dx, dy)
            if event == "nod":
                nod_count += 1
                last_label = "NOD"
                last_event_ts = time.time()
                print(f"✅ 偵測到【點頭】！目前累計：{nod_count}")
            elif event == "shake":
                shake_count += 1
                last_label = "SHAKE"
                last_event_ts = time.time()
                print(f"❎ 偵測到【搖頭】！目前累計：{shake_count}")

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        cv2.putText(
            frame,
            f"Nods: {nod_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Shakes: {shake_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

        if time.time() - last_event_ts < 1.0 and last_label:
            color = (0, 0, 255) if last_label == "SHAKE" else (0, 255, 0)
            cv2.putText(
                frame,
                last_label,
                (w // 2 - 80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                color,
                4,
            )

        cv2.imshow("Strict Head Gesture Detection - q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
