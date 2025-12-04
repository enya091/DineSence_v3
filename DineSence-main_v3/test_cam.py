import streamlit as st
import cv2
import time

st.title("Dual Camera Display (Face + Food)")

col1, col2 = st.columns(2)

frame_placeholder1 = col1.empty()
frame_placeholder2 = col2.empty()

# 你的兩個鏡頭
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)  # 如果是 Camo (iPhone)，通常是 1

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if ret1:
        frame_placeholder1.image(f1, channels="BGR")

    if ret2:
        frame_placeholder2.image(f2, channels="BGR")

    time.sleep(0.03)  # 避免爆 CPU（30 FPS）

