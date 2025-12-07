# DineSence_v3

1. 將整個專案 git clone 下來
2. 打開整個資料夾
3. 在終端機中輸入 touch .env 在env檔中放入OpenAPI Key
4. 在終端機中輸入mkdir -p .streamlit && echo '[theme]
base = "dark"
primaryColor = "#c18440"
backgroundColor = "#031e27"
secondaryBackgroundColor = "#1e293b"
textColor = "#ffffff"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false' > .streamlit/config.tom
5. 輸入streamlit run app.py 
