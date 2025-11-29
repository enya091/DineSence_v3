# ui/login_view.py

import streamlit as st
import os
import config

def display():
    """
    顯示登入頁面並處理驗證邏輯。
    """
    lcol, ccol, rcol = st.columns([1, 0.8, 1])

    with ccol:
        st.write("") 
        st.write("") 
        
        # 使用容器創造毛玻璃卡片
        with st.container(border=True):
            # --- 修改重點開始：替換 Emoji 為圖片 ---
            logo_path = "assets/logo.png"
            
            # 檢查圖片是否存在，存在就顯示圖片，不存在就顯示備用 Emoji
            if os.path.exists(logo_path):
                # 這裡設定 width=180，您可以根據 Logo 的實際長寬比調整
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(logo_path, use_container_width=True)
            else:
                st.markdown("<div style='text-align: center; font-size: 4rem;'>🧬</div>", unsafe_allow_html=True)
            # --- 修改重點結束 ---

            # 發光的標題文字 (保留文字，讓使用者知道這是什麼系統)
            st.markdown("<h2 style='text-align: center; color: #FFFFFF; letter-spacing: 2px; margin-top:10px;'>DINESENCE <span style='color:var(--primary-color);'>AI</span></h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 30px; font-size: 0.8rem; font-family: monospace;'>ACCESS RESTRICTED // AUTHORIZED PERSONNEL ONLY</p>", unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                st.markdown("<p style='color:var(--primary-color); font-size:0.8rem; margin-bottom:5px;'>USER ID</p>", unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="admin", key="login_username", label_visibility="collapsed")
                
                st.markdown("<p style='color:var(--primary-color); font-size:0.8rem; margin-bottom:5px; margin-top:15px;'>PASSWORD</p>", unsafe_allow_html=True)
                password = st.text_input("Password", type="password", placeholder="••••••", key="login_password", label_visibility="collapsed")
                
                st.write("") 
                submitted = st.form_submit_button("INITIALIZE SESSION ➤", use_container_width=True, type="primary")

                if submitted:
                    correct_username = config.DASH_USER
                    correct_password = config.DASH_PASS

                    if username == correct_username and password == correct_password:
                        st.session_state['auth'] = True
                        st.toast("ACCESS GRANTED. WELCOME BACK.", icon="🔓")
                        st.rerun()
                    else:
                        st.error("ACCESS DENIED. INVALID CREDENTIALS.")
            
            st.markdown(
                "<div style='text-align: center; margin-top: 25px; font-size: 10px; color: #475569; font-family: monospace;'>SYSTEM VERSION 3.0.0 | SECURE CONNECTION</div>", 
                unsafe_allow_html=True
            )