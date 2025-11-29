# app.py

import streamlit as st
import os
import config
from services import llm_handler, vision_analysis as va
from services.detectors import BodyEmotionDetector
from services.analyzer import EmotionSatisfactionAnalyzer
from services import database
from ui import live_view, video_view, dashboard_view, login_view
from utils import state_manager
from utils import localization
from services.database import DatabaseManager

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="DineSence",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "🍽️", # 網頁分頁也用 Logo
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ★★★ 新增：左上角常駐 Logo ★★★
if os.path.exists("assets/logo.png"):
    st.logo("assets/logo.png", icon_image="assets/logo.png")

# --- 2. 樣式注入 (智慧配色版) ---
# 修改 app.py 中的 load_custom_css 函式

def load_custom_css():
    st.markdown("""
    <style>
        /* 1. 引用字體 */
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');
        
        /* 2. 全局設定 - 讓 Streamlit 使用 config.toml 的設定 */
        html, body, [class*="css"] {
            font-family: 'Noto Sans TC', sans-serif;
            /* 移除強制背景色，讓 config.toml 生效 */
        }
        
        /* 3. 定義變數 - 從 config.toml 自動抓取顏色 */
        :root {
            --primary-color: #c18440; /* 這裡可以對應您的設定 */
            --card-bg: rgba(30, 41, 59, 0.5); /* 半透明卡片背景 */
            --text-color: #ffffff;
            --text-muted: #94a3b8;
        }

        /* 4. 標題特效 - 金屬光澤感 */
        h1, h2, h3 {
            font-family: 'Rajdhani', 'Noto Sans TC', sans-serif;
            font-weight: 700;
            color: var(--text-color);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        
        /* 5. Metric 卡片 - 毛玻璃特效 */
        [data-testid="stMetric"] {
            background-color: var(--card-bg); /* 使用半透明 */
            backdrop-filter: blur(10px);      /* 毛玻璃模糊 */
            border: 1px solid rgba(193, 132, 64, 0.3); /* 使用主色當微弱邊框 */
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: var(--primary-color);
            box-shadow: 0 0 15px rgba(193, 132, 64, 0.4); /* 發光特效 */
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: var(--primary-color) !important; /* 強制使用金色 */
            font-weight: 700;
            font-family: 'Rajdhani', sans-serif;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted);
        }

        /* 6. Tabs 分頁 - 膠囊樣式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
            padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: rgba(255,255,255,0.05);
            border-radius: 8px;
            color: var(--text-muted);
            border: 1px solid transparent;
            padding: 0 20px;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(193, 132, 64, 0.2) !important;
            color: var(--primary-color) !important;
            border: 1px solid var(--primary-color) !important;
            box-shadow: 0 0 10px rgba(193, 132, 64, 0.2);
        }

        /* 7. 按鈕 - 金色漸層 */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #c18440 0%, #a06030 100%);
            color: white;
            border: none;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            font-weight: bold;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 0 15px rgba(193, 132, 64, 0.6);
            transform: scale(1.02);
        }

        /* 8. 修正輸入框背景 */
        div[data-baseweb="input"] {
            background-color: rgba(0, 0, 0, 0.2) !important;
            color: white !important;
            border-color: #334155 !important;
        }
        
        /* 9. 隱藏不必要的元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --- 3. 初始化狀態與語言 ---
state_manager.initialize_state()
if 'language' not in st.session_state:
    st.session_state['language'] = 'zh'

def t(key):
    return localization.get_text(key, st.session_state['language'])

# --- 4. 初始化資源 ---
@st.cache_resource
def load_models():
    db_manager = DatabaseManager() 
    openai_client = llm_handler.get_openai_client(config.OPENAI_API_KEY)
    pose_detector = va.get_pose_detector()
    face_detector = va.get_face_detector()
    _ = va.get_food_model() if hasattr(va, 'get_food_model') else None
    body_emotion_detector = BodyEmotionDetector()
    emotion_analyzer = EmotionSatisfactionAnalyzer()
    return openai_client, pose_detector, face_detector, body_emotion_detector, emotion_analyzer, db_manager

client, pose_detector, face_detector, body_emotion_detector, emotion_analyzer, db_manager = load_models()

st.session_state.setdefault('emotion_analyzer', emotion_analyzer)
st.session_state.setdefault('body_emotion_detector', body_emotion_detector)
st.session_state.setdefault('db_manager', db_manager)

# --- 5. 登入閘門 ---
if not st.session_state.auth:
    login_view.display()
    st.stop()

# --- 6. 後端配置 ---
BACKEND_CONFIG = {
    "store_type": "Buffet", 
    "menu_items": ["Main Course", "Vegetables", "Dessert", "Drinks"] 
}

model_pack = {
    "client": client,
    "pose_detector": pose_detector,
    "face_detector": face_detector
}

# --- 7. 主頁面 UI ---
top_col1, top_col2 = st.columns([6, 1])

with top_col1:
    # 標題這裡也讓它使用主色
    st.markdown(f"<h1>DineSence <span style='color:var(--primary-color); text-shadow:0 0 15px var(--primary-color);'></span> <span style='font-size:0.5em; opacity:0.7;'>// {t('live_panel_title')}</span></h1>", unsafe_allow_html=True)

with top_col2:
    with st.popover(t("settings"), use_container_width=True):
        st.markdown(f"##### 🌐 {t('select_lang')}")
        lang_choice = st.radio(
            "Language",
            options=["中文", "English"],
            index=0 if st.session_state['language'] == 'zh' else 1,
            label_visibility="collapsed",
            key="lang_radio"
        )
        new_lang = "zh" if lang_choice == "中文" else "en"
        if new_lang != st.session_state['language']:
            st.session_state['language'] = new_lang
            st.rerun()

st.markdown("---")

if not client:
    st.error("⚠️ SYSTEM ALERT: OpenAI API Key Missing.")
else:
    tab_live, tab_video, tab_dashboard = st.tabs([
        t("tab_live"), 
        t("tab_video"),
        t("tab_dashboard")
    ])

    with tab_live:
        st.write("") 
        live_view.display(model_pack, BACKEND_CONFIG, db_manager, t=t)

    with tab_video:
        st.write("")
        video_view.display(client, BACKEND_CONFIG["menu_items"], BACKEND_CONFIG, t=t)
        
    with tab_dashboard:
        st.write("")
        dashboard_view.display(client, db_manager, t=t)