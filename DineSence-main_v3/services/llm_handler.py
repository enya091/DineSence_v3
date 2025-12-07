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
# 請確保 utils/prompt_loader.py 存在，若無可暫時註解並將 Prompt 寫死在函式內
try:
    from utils.prompt_loader import load_prompt_template
except ImportError:
    # 簡單的 fallback，避免如果沒有這個檔案時報錯
    def load_prompt_template(name, type):
        return "" 

# 使用 AsyncClient 進行非同步請求
aclient = httpx.AsyncClient()

# 圖片最大邊長限制 (加速 VLM 分析用)
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

# ==========================================
# 1. 影像辨識相關 (Vision / VLM)
# ==========================================

async def gpt_image_classify_3cls(face_bgr, client: AsyncOpenAI, model="gpt-4o-mini"):
    """
    (非同步) 使用 GPT-4o-mini 進行三分類表情辨識。
    輸入: OpenCV BGR 影像
    輸出: (情緒字串, usage_data)
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
    safe_menu = menu_items.copy() if menu_items else []
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


# ==========================================
# 2. 文本報告與摘要相關 (Report Generation)
# ==========================================

async def summarize_session(stats: dict, 
                            client: AsyncOpenAI, 
                            store_type: str = "餐廳", 
                            tone: str = "專業", 
                            tips_style: str = "預設",
                            custom_instructions: str = None, 
                            model="gpt-4o-mini"):
    """
    (非同步) 根據統計數據，生成客製化的顧客體驗摘要報告。
    支援 custom_instructions 參數，若有傳入則直接使用該指令 (用於 Dashboard 營運報告)。
    """
    if client is None:
        return "（未設定 OPENAI_API_KEY，無法產生摘要）", None

    # 1. 嘗試讀取 Prompt 模板，若失敗則使用預設值
    try:
        system_template = load_prompt_template('summarize_session', 'system')
        if not system_template: raise FileNotFoundError
    except:
        system_template = "你是一位專業的餐廳營運顧問。請根據數據生成報告。"
        
    try:
        user_template = load_prompt_template('summarize_session', 'user')
        if not user_template: raise FileNotFoundError
    except:
        user_template = "數據如下：{stats_json}。請分析。"

    # 2. 處理 System Prompt
    system_prompt = system_template.replace('{store_type}', str(store_type)) \
                                   .replace('{tone}', str(tone)) \
                                   .replace('{tips_style}', str(tips_style))

    # 3. 處理 User Prompt
    if custom_instructions:
        # ★ 如果有自訂指令 (Dashboard Tab 1)，直接使用它
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


async def generate_menu_report(food_stats: dict, client: AsyncOpenAI, model="gpt-4o-mini"):
    """
    (非同步) 專門針對「菜色」生成研發與調整建議報告 (Tab 2 專用)。
    """
    if not client: return "（未設定 API Key）", None

    # 將統計數據轉為 JSON 字串
    stats_str = json.dumps(food_stats, ensure_ascii=False, indent=2)

    system_prompt = (
        "你是一位擁有 20 年經驗的『餐飲菜色研發顧問』。你的工作是根據顧客對特定菜色的情緒反應數據，"
        "為內場主廚提供具體的菜單調整建議。\n\n"
        "【數據說明】\n"
        "輸入的 JSON 格式為：{'菜名': {'開心': 次數, '嫌棄': 次數, ...}}\n\n"
        "【報告結構要求】\n"
        "1. 🏆 **明星菜色 (Star Dishes)**：正面情緒佔比最高的菜。分析其成功可能的因素（口味、賣相）。\n"
        "2. 💣 **問題菜色 (Problem Dishes)**：負面情緒（嫌棄/失望/不滿）較高的菜。請大膽推測可能原因（如：調味過重、冷掉、食材搭配怪異）。\n"
        "3. 🔪 **主廚行動建議 (Action Plan)**：針對問題菜色，提出 2-3 個具體的改良方向（例如：調整醬汁比例、更換盛盤方式）。\n"
        "請用專業、直白且建設性的語氣撰寫，不要講空話。"
    )

    user_prompt = f"請分析本週的菜色情緒數據：\n{stats_str}"

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return resp.choices[0].message.content, resp.usage
    except Exception as e:
        return f"生成菜色報告失敗: {e}", None

# ==========================================
# 3. 同步輔助函式 (Sync Helper for ThreadPool)
# ==========================================

def sync_gpt_image_classify_3cls(face_img, client):
    """
    同步版本的 GPT-4o 圖片情緒辨識 (專供 ThreadPoolExecutor 使用)
    輸入: OpenCV 影像 (BGR)
    輸出: '喜歡', '中性', '討厭' 其中之一
    """
    if face_img is None or face_img.size == 0:
        return "中性"

    try:
        # 1. 影像編碼 (BGR -> JPEG -> Base64)
        _, buffer = cv2.imencode('.jpg', face_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 2. 呼叫 GPT-4o (同步模式，不加 await)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an emotion classifier. Classify the face image into exactly one of these 3 classes: '喜歡', '中性', '討厭'. Return ONLY the class name."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Classify this face."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        # 簡單防呆清洗
        for valid in ["喜歡", "中性", "討厭"]:
            if valid in result:
                return valid
                
        return "中性" # 預設值

    except Exception as e:
        print(f"LLM Sync Error: {e}")
        return "中性"