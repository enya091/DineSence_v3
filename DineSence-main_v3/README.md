🍽️ DineSence 智慧餐飲顧客分析平台
AI 驅動的餐飲決策核心，讀懂顧客無聲的反饋

📖 專案簡介 (Introduction)
DineSence 是一個結合 電腦視覺 (Computer Vision) 與 大型語言模型 (LLM) 的智慧餐飲營運輔助系統。

現行餐飲業缺乏即時且客觀的顧客反饋，傳統問卷容易有偏差，銷售數據也無法完全反映顧客喜好。DineSence 透過非接觸式的鏡頭分析，捕捉顧客用餐時的微表情、肢體動作以及餐盤剩食狀況，並將這些非結構化數據轉化為可執行的商業洞察，協助餐廳優化菜單與服務流程。

✨ 核心功能 (Key Features)
🟢 1. 多模態即時分析 (Real-time Multimodal Analysis)
系統能透過店內鏡頭即時捕捉並分析顧客狀態：

情緒辨識：利用 GPT-4o Vision API 精準判斷顧客情緒光譜（喜歡 / 中性 / 討厭）。

行為解讀：結合 MediaPipe Pose 偵測顧客的肢體語言（如：點頭肯定、搖頭否定）。

非干擾式體驗：無需顧客填寫問卷，大幅降低干擾並獲得最真實的反應。

🍛 2. 智慧剩食洞察 (Smart Leftover Analysis)
餐盤偵測：運用 YOLOv8 物件偵測技術，自動識別餐盤區域。

滿意度量化：透過 OpenCV 演算法計算剩食比例，推估顧客對該菜色的真實滿意度（吃光 = 高滿意度）。

📊 3. 商業智慧儀表板 (Business Intelligence Dashboard)
AI 營運報告：整合所有數據，由 LLM 自動生成每日/每週營運建議報告。

視覺化數據：提供來客數、情緒趨勢、熱門/地雷菜色分析圖表。

🛠️ 技術架構 (Tech Stack)
前端框架: Streamlit

核心 AI 模型:

LLM & VLM: OpenAI GPT-4o / GPT-4o-mini (情緒分析、報告生成)

Object Detection: YOLOv8 (食物與餐盤偵測)

Pose Estimation: Google MediaPipe (臉部網格與骨架追蹤)

影像處理: OpenCV, PIL

資料管理: SQLite (輕量級資料庫)

🚀 快速開始 (Quick Start)
1. 環境設定
本專案需要 Python 3.8 或更高版本。

Bash

# Clone 專案
git clone [您的 Git Repo URL]
cd DineSence-main_v3

# 建立虛擬環境
python -m venv venv

# 啟用虛擬環境 (Windows)
.\venv\Scripts\activate
# 啟用虛擬環境 (Mac/Linux)
source venv/bin/activate
2. 安裝依賴
Bash

pip install -r requirements.txt
3. 設定環境變數
請在專案根目錄建立 .env 檔案，並填入您的 OpenAI API Key：

Code snippet

OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
4. 啟動系統
Bash

streamlit run app.py
系統將自動開啟瀏覽器，預設登入帳號密碼為：admin / admin123。

📁 專案結構
DineSence/
├── app.py                  # 程式進入點 (Streamlit Main App)
├── config.py               # 全域設定檔
├── services/               # 核心服務邏輯
│   ├── vision_analysis.py  # CV 演算法 (YOLO, MediaPipe)
│   ├── llm_handler.py      # LLM 介面 (OpenAI API)
│   └── ...
├── ui/                     # 介面模組 (Dashboard, Live View)
└── ...
📝 關於我們
本專案由 資科四A (蔡加恩、楊沛潔、曾鈺涵、黃薇庭、洪文蕙) 開發，指導老師為 吳政隆教授。 我們致力於利用 AI 技術解決真實世界的商業難題。

Disclaimer: 本系統目前僅供教育與研究用途。實際場域應用時，請確保符合當地隱私權法規。
