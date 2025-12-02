# services/llm_handler.py

import base64
import io
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Any

# --- 導入我們建立的 prompt 讀取工具 ---
from utils.prompt_loader import load_prompt_template

# 使用 AsyncClient 進行非同步請求
aclient = httpx.AsyncClient()

# 圖片最大邊長限制
MAX_VLM_IMAGE_DIM = 768

def get_openai_client(api_key):
    """根據 API Key 初始化並返回異步的 OpenAI 客戶端物件。"""
    if not api_key:
        return None
    try:
        return AsyncOpenAI(api_key=api_key)
    except Exception as e:
        print(f"初始化 OpenAI Client 失敗: {e}")
        return None

def _image_to_base64(pil_image: Image.Image) -> str:
    """將 PIL.Image 物件轉換為 base64 字串，並自動縮放以加速 VLM 分析。"""
    width, height = pil_image.size
    
    if max(width, height) > MAX_VLM_IMAGE_DIM:
        if width > height:
            new_width = MAX_VLM_IMAGE_DIM
            new_height = int(height * (new_width / width))
        else:
            new_height = MAX_VLM_IMAGE_DIM
            new_width = int(width * (new_height / height))
        
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

async def gpt_image_classify_3cls(face_bgr, client: AsyncOpenAI, model="gpt-4o-mini"):
    """
    (非同步) 使用 GPT-4o-mini 進行三分類表情辨識。
    """
    if face_bgr is None: return "無臉", None
    if client is None: return "（未設定 API）", None

    pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)
    
    prompt = (
        "請根據臉部表情，在三類中擇一輸出（請只輸出一個詞）：\n"
        "『喜歡』（正向/微笑）、『中性』、或『討厭』（厭惡/皺眉）。\n"
        "只輸出：喜歡 / 中性 /討厭。"
    )
    
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            temperature=0, max_tokens=10
        )
        text = resp.choices[0].message.content.strip()
        usage = resp.usage

        emotion = "中性"
        if "喜歡" in text: emotion = "喜歡"
        if "討厭" in text: emotion = "討厭"
        
        return emotion, usage

    except Exception as e:
        print(f"表情分類 API 錯誤: {e}")
        return "API 錯誤", None


async def gpt_food_from_menu(plate_bgr, menu_items, client: AsyncOpenAI, model="gpt-4o-mini"):
    """(非同步) 根據提供的菜單，使用 GPT-4o-mini 辨識畫面中的餐點。"""
    if plate_bgr is None: return {"label": "未知", "confidence": 0.0, "rationale": "無ROI"}
    if client is None: return {"label": "（未設定API）", "confidence": 0.0, "rationale": ""}

    pil_img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)

    options = ", ".join(menu_items)
    prompt = (
        "請只根據提供的菜單清單判斷畫面中的餐點屬於哪一項，並以 JSON 格式輸出：\n"
        '{ "label": "<從菜單中擇一>", "confidence": 0.0~1.0 的小數, "rationale": "一句簡短的判斷原因" }\n'
        f"菜單清單：[{options}]\n"
        "如果畫面中的餐點與任何清單項目都不太相符，請選擇最相近的一項，但給予較低的信心度（<=0.4）。\n"
        "請務必只輸出 JSON 物件，不要包含任何前後的文字或 markdown 標籤。"
    )

    try:
        resp = await client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        return {
            "label": str(data.get("label", "未知")),
            "confidence": float(data.get("confidence", 0.0)),
            "rationale": str(data.get("rationale", ""))[:120]
        }
    except Exception as e:
        print(f"食物分類 API 錯誤: {e}")
        return {"label": "解析失敗", "confidence": 0.0, "rationale": str(e)[:120]}


async def analyze_plate_vlm(plate_bgr: Any, client: AsyncOpenAI, 
                            food_detections: List[Dict[str, Any]] = None, 
                            model="gpt-4o-mini"):
    """
    (非同步) 使用 VLM (GPT-4o-mini) 分析餐盤剩食原因。
    """
    if plate_bgr is None: return None, None
    if client is None: return None, None

    pil_img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)

    yolo_info = ""
    if food_detections:
        formatted_dets = [
            f"{det['label']} (信心度: {det['conf']:.2f})"
            for det in food_detections
        ]
        if formatted_dets:
            yolo_info = "\n[系統提示] YOLO 演算法在畫面中偵測到了： " + ", ".join(formatted_dets)

    system_prompt = (
        "你是一位專業的餐廳營運顧問。請觀察這張餐盤回收的照片。"
        "請先判斷餐盤狀態，並依據情況擇一回答 (繁體中文，50字以內)：\n\n"
        "情況 A：如果是『空盤』或『吃得很乾淨』\n"
        "請回答：「顧客對餐點接受度高，完食無浪費。」這類正面評價，不要捏造剩食原因。\n\n"
        "情況 B：如果有『明顯剩食』\n"
        "請回答：\n"
        "1. 剩下了什麼具體食物？(例如：青椒、澱粉、肉類)\n"
        "2. 推測剩食原因？(例如：完全未動可能是不愛吃、剩一半可能是份量太大)"
    )
    
    user_prompt = f"請分析這張剩食照片。{yolo_info}"

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            ],
            max_tokens=150, 
            temperature=0.3 
        )
        content = resp.choices[0].message.content.strip()
        usage = resp.usage
        return content, usage
    except Exception as e:
        print(f"VLM 餐盤分析錯誤: {e}")
        return f"API 錯誤: {str(e)[:50]}", None


# ★★★ [修正重點] 新增 custom_instructions 參數，並設為預設 None ★★★
async def summarize_session(stats: dict, 
                            client: AsyncOpenAI, 
                            store_type: str = "餐廳", 
                            tone: str = "專業", 
                            tips_style: str = "預設",
                            custom_instructions: str = None,  # <--- 新增這個
                            model="gpt-4o-mini"):
    """
    (非同步) 根據統計數據，生成客製化的顧客體驗摘要報告。
    支援 custom_instructions 以覆寫預設 Prompt。
    """
    if client is None:
        return "（未設定 OPENAI_API_KEY，無法產生摘要）", None

    try:
        system_template = load_prompt_template('summarize_session', 'system')
        # 如果沒有自訂指令，才去讀取預設的 user template
        if not custom_instructions:
            user_template = load_prompt_template('summarize_session', 'user')
    except FileNotFoundError as e:
        error_msg = f"找不到 Prompt 模板檔案：{e}。請確認 'prompts/summarize_session/' 資料夾及內部的 .txt 檔案是否存在。"
        print(error_msg)
        return error_msg, None

    # 1. 處理 System Prompt
    system_prompt = system_template.replace('{store_type}', str(store_type)) \
                                   .replace('{tone}', str(tone)) \
                                   .replace('{tips_style}', str(tips_style))

    # 2. 處理 User Prompt (判斷是否使用自訂指令)
    if custom_instructions:
        # ★ 如果有自訂指令，直接使用它 (Dashboard 模式)
        user_prompt = custom_instructions
    else:
        # ★ 否則使用預設模板並填入數據 (Video/Live 模式)
        json_str = json.dumps(stats, indent=2, ensure_ascii=False)
        user_prompt = user_template.replace('{stats_json}', json_str)
    
    try:
        r: ChatCompletion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7, max_tokens=800,
        )
        summary_text = r.choices[0].message.content
        usage_data = r.usage
        return summary_text, usage_data
        
    except Exception as e:
        error_msg = f"生成摘要時發生錯誤：{e}"
        print(f"摘要生成 API 錯誤: {e}")
        return error_msg, None
    
async def identify_food_item(plate_bgr, menu_items: list, client: AsyncOpenAI, model="gpt-4o"):
    """
    針對強烈情緒觸發的快照，進行高精準度的食物辨識。
    
    Args:
        plate_bgr: OpenCV 格式的影像 (BGR)
        menu_items: 候選菜單列表 (例如 ["漢堡", "雞塊", "薯條"])
        client: OpenAI Client
        model: 建議使用 gpt-4o 以獲得最佳視覺辨識能力
        
    Returns:
        str: 辨識出的食物名稱 (例如 "漢堡")，若失敗則回傳 "Unknown"
    """
    if plate_bgr is None: return "Unknown"
    if client is None: return "No_API_Key"

    # 1. 圖片轉 Base64
    pil_img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)

    # 2. 準備選單字串 (確保有 Other 選項)
    safe_menu = menu_items.copy()
    if "其他" not in safe_menu and "Other" not in safe_menu:
        safe_menu.append("Other")
    
    options_str = ", ".join(safe_menu)
    
    # 3. 建構 Prompt (要求精簡回答)
    prompt = (
        f"請觀察這張餐盤照片，並從以下清單中選出最符合的食物名稱：\n"
        f"清單：[{options_str}]\n"
        "請直接回傳該名稱，不要加任何標點符號或額外文字(例如不要回傳 '是漢堡'，只要回傳 '漢堡')。\n"
        "如果完全看不出來、空盤或不在清單中，請回傳 'Other'。"
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            temperature=0.1, # 低隨機性，確保答案穩定
            max_tokens=20
        )
        
        # 4. 取得結果
        text = resp.choices[0].message.content.strip()
        
        # 簡單防呆：確保回傳的文字真的在我們的清單裡
        # (有時候 LLM 會回傳 "應該是漢堡"，我們只要 "漢堡")
        for item in safe_menu:
            if item in text:
                return item
                
        return text # 如果都沒對到，就回傳原始答案 (通常是 Other)

    except Exception as e:
        print(f"Food ID Error: {e}")
        return "Error"