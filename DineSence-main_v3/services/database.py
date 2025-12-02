# services/database.py

import sqlite3
import datetime
import pandas as pd
import os
import json 

DB_NAME = "dinesence.db"

class DatabaseManager:
    def __init__(self, db_path=DB_NAME):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        """建立並回傳資料庫連線"""
        return sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=10.0
        )

    def init_db(self):
        """初始化資料庫結構"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. 即時分析日誌表 (您原本的)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT,   
                people_count INTEGER,
                emotions TEXT,      
                food_detected TEXT, 
                raw_data TEXT       
            )
        ''')

        # 2. 分析總結紀錄表 (★ 修改：增加 session_id_raw 用於連結影像)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id_raw TEXT UNIQUE,  -- [NEW] 連結影像的唯一ID
                timestamp TEXT,
                mode TEXT,
                duration_seconds INTEGER,
                nod_count INTEGER,
                shake_count INTEGER,
                emotion_data TEXT,
                leftover_data TEXT,
                ai_insight TEXT
            )
        ''')

        # 3. [修改] 事件影像佐證表 (增加 food_label)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_timestamp TEXT,
                event_type TEXT,
                local_path TEXT UNIQUE,
                human_corrected INTEGER DEFAULT 1,
                food_label TEXT  -- ★ [NEW] 新增這個欄位來存 "漢堡"
            )
        ''')
        
        conn.commit()
        conn.close()

    # ==========================================
    # Part 1: 即時日誌功能 (保留您原本的)
    # ==========================================

    def insert_log(self, source_type, people_count, emotions, food_detected):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            emotions_str = str(emotions)
            food_str = str(food_detected)

            cursor.execute('''
                INSERT INTO analysis_logs (timestamp, source_type, people_count, emotions, food_detected)
                VALUES (?, ?, ?, ?, ?)
            ''', (current_time, source_type, people_count, emotions_str, food_str))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"DB Log Error: {e}") 
            return False

    def get_recent_logs(self, limit=100):
        conn = self.get_connection()
        query = "SELECT * FROM analysis_logs ORDER BY timestamp DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df

    def clear_logs(self, source_type=None):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            if source_type:
                cursor.execute("DELETE FROM analysis_logs WHERE source_type = ?", (source_type,))
            else:
                cursor.execute("DELETE FROM analysis_logs")
                cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name = 'analysis_logs'")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"DB Error: {e}")
            return False

    def get_logs_filtered(self, start_time, end_time, source_types=None):
        if not source_types:
            source_types = ['live_stream', 'uploaded_video'] 
        conn = self.get_connection()
        source_placeholders = ','.join('?' for _ in source_types)
        query = f"""
            SELECT * FROM analysis_logs 
            WHERE timestamp BETWEEN ? AND ? 
            AND source_type IN ({source_placeholders})
            ORDER BY timestamp DESC
        """
        params = [start_time, end_time] + source_types
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
        
    def get_logs_by_range(self, start_time, end_time, source_types=None):
        return self.get_logs_filtered(start_time, end_time, source_types)

    def get_logs_by_date(self, target_date, source_types=None):
        start_time = f"{target_date} 00:00:00"
        end_time = f"{target_date} 23:59:59"
        return self.get_logs_filtered(start_time, end_time, source_types)

    # ==========================================
    # Part 2: 總結紀錄功能 (增強)
    # ==========================================

    # [MODIFIED] 增加 raw_session_id 參數
    def save_session(self, raw_session_id, mode, duration, nod, shake, emotion_dict, leftover_dict, insight):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            emotion_json = json.dumps(emotion_dict, ensure_ascii=False)
            leftover_json = json.dumps(leftover_dict, ensure_ascii=False)
            
            display_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # [MODIFIED] 寫入 session_id_raw
            cursor.execute('''
                INSERT INTO analysis_records 
                (session_id_raw, timestamp, mode, duration_seconds, nod_count, shake_count, emotion_data, leftover_data, ai_insight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (raw_session_id, display_timestamp, mode, duration, nod, shake, emotion_json, leftover_json, insight))
            
            conn.commit()
            conn.close()
            print(f"✅ Session 資料已寫入資料庫: {display_timestamp}")
            return True
        except Exception as e:
            print(f"DB Session Error: {e}")
            return False

    def get_all_session_records(self):
        conn = self.get_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM analysis_records ORDER BY timestamp DESC", conn)
        except Exception as e:
            print(f"Error reading session records: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()
        return df

    # ==========================================
    # Part 3: 影像佐證功能 (新增)
    # ==========================================
    
    def save_event_evidence(self, session_id, event_type, local_path, food_label=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # [修改] 寫入 food_label
            cursor.execute('''
                INSERT INTO event_evidence (session_timestamp, event_type, local_path, food_label)
                VALUES (?, ?, ?, ?)
            ''', (session_id, event_type, local_path, food_label))
            conn.commit()
        except Exception as e:
            print(f"DB Save Evidence Error: {e}") # 加印錯誤訊息方便除錯
        finally:
            conn.close()

    def get_event_evidence(self, session_id, event_type=None):
        conn = self.get_connection()
        params = [session_id]
        
        # ★★★ [修改] 在 SELECT 中加入 food_label ★★★
        query = "SELECT id, event_type, local_path, human_corrected, food_label FROM event_evidence WHERE session_timestamp = ?"
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
            
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def update_evidence_feedback(self, evidence_id, is_correct: bool):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            corrected_value = 1 if is_correct else 0
            cursor.execute('UPDATE event_evidence SET human_corrected = ? WHERE id = ?', (corrected_value, evidence_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    # ==========================================
    # Part 4: 進階分析功能 (保留您原本的)
    # ==========================================

    def get_people_flow_trend(self, start_time, end_time, resample_rule='5min'):
        df = self.get_logs_filtered(start_time, end_time)
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'avg_people', 'max_people'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        trend_df = df['people_count'].resample(resample_rule).agg(['mean', 'max']).fillna(0)
        trend_df.columns = ['avg_people', 'max_people']
        trend_df['avg_people'] = trend_df['avg_people'].round(1)
        return trend_df.reset_index()

    def get_hourly_busy_stats(self, date_str):
        start_time = f"{date_str} 00:00:00"
        end_time = f"{date_str} 23:59:59"
        df = self.get_logs_filtered(start_time, end_time)
        if df.empty:
            hours = list(range(24))
            return pd.DataFrame({'hour': hours, 'avg_people': 0})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        hourly_stats = df.groupby('hour')['people_count'].mean().reset_index()
        hourly_stats.columns = ['hour', 'avg_people']
        hourly_stats['avg_people'] = hourly_stats['avg_people'].round(1)
        all_hours = pd.DataFrame({'hour': range(24)})
        result = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)
        return result

    def get_customer_groups_analysis(self, start_time, end_time, gap_minutes=3):
        conn = self.get_connection()
        query = """
            SELECT timestamp, people_count 
            FROM analysis_logs 
            WHERE timestamp BETWEEN ? AND ? 
            AND source_type IN ('live_stream', 'uploaded_video')
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(start_time, end_time))
        conn.close()

        if df.empty: return 0, pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_active = df[df['people_count'] > 0].copy()
        
        if df_active.empty: return 0, pd.DataFrame()

        df_active['time_diff'] = df_active['timestamp'].diff()
        threshold = pd.Timedelta(minutes=gap_minutes)
        df_active['group_id'] = (df_active['time_diff'] > threshold).cumsum().fillna(0)
        
        groups = df_active.groupby('group_id').agg(
            start_time=('timestamp', 'min'),
            end_time=('timestamp', 'max'),
            avg_people=('people_count', 'mean'),
            max_people=('people_count', 'max')
        ).reset_index(drop=True)
        
        groups['duration_minutes'] = (groups['end_time'] - groups['start_time']).dt.total_seconds() / 60
        groups['duration_minutes'] = groups['duration_minutes'].round(1)
        groups['avg_people'] = groups['avg_people'].round(1)
        groups['start_time_str'] = groups['start_time'].dt.strftime('%H:%M:%S')
        groups['end_time_str'] = groups['end_time'].dt.strftime('%H:%M:%S')
        groups = groups[['start_time_str', 'end_time_str', 'duration_minutes', 'avg_people', 'max_people']]
        groups.columns = ['開始時間', '結束時間', '用餐時長(分)', '平均人數', '最大人數']
        
        return len(groups), groups

# --- Wrapper 函式 ---
_db_instance = DatabaseManager()

def save_session(raw_session_id, mode, duration, nod, shake, emotion_dict, leftover_dict, insight):
    return _db_instance.save_session(raw_session_id, mode, duration, nod, shake, emotion_dict, leftover_dict, insight)

def insert_log(source_type, people_count, emotions, food_detected):
    return _db_instance.insert_log(source_type, people_count, emotions, food_detected)

def get_customer_groups_analysis(start_time, end_time, gap_minutes=3):
    return _db_instance.get_customer_groups_analysis(start_time, end_time, gap_minutes)

def get_logs_by_range(start_time, end_time, source_types=None):
    return _db_instance.get_logs_by_range(start_time, end_time, source_types)

def get_event_evidence(session_id, event_type=None):
    return _db_instance.get_event_evidence(session_id, event_type)

def update_evidence_feedback(evidence_id, is_correct: bool):
    return _db_instance.update_evidence_feedback(evidence_id, is_correct)