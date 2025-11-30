import numpy as np
from deepface import DeepFace

print("🚀 正在啟動 DeepFace 測試...")
print("如果是第一次執行，這裡會開始下載權重檔 (約 500MB+)，請耐心等待...")

try:
    # 1. 建立一個純黑色的假圖片 (避免因為找不到圖片檔而報錯)
    # 格式: (高度, 寬度, 3色版)
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

    # 2. 強制執行一次分析
    # enforce_detection=False 讓它就算沒看到臉也必須載入模型跑一次
    result = DeepFace.analyze(
        img_path=dummy_img, 
        actions=['emotion'], 
        enforce_detection=False,
        silent=False
    )

    print("\n✅ 測試成功！模型已下載完成且可正常運作。")
    print(f"回傳結果範例: {result[0]['dominant_emotion']}")

except ImportError:
    print("\n❌ 錯誤：找不到 deepface 模組。")
    print("請執行: pip install deepface tf-keras")
except Exception as e:
    print(f"\n❌ 發生其他錯誤: {e}")