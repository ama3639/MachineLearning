#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اجزای مشترک برای سیستم استخراج داده‌های ساده‌شده
نسخه Simplified v2.0 - برای استفاده مشترک بین فایل‌های 01 و 01A

🎯 هدف: کاهش پیچیدگی و تمرکز بر کیفیت
📊 استراتژی: Binance (قیمت) + RSS (اخبار) = 99% کارایی
🚀 مزایا: سریع، قابل اعتماد، آسان نگهداری

این ماژول شامل:
- مدیریت وضعیت ساده‌شده
- توابع کمکی مشترک
- تنظیمات لاگ‌گیری
- مدیریت خطا
"""

import os
import time
import sqlite3
import logging
import configparser
import requests
from datetime import datetime
from typing import Dict, List, Optional
import threading

# --- خواندن تنظیمات ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'

try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    RAW_DATA_PATH = config.get('Paths', 'raw')
    LOG_PATH = config.get('Paths', 'logs')
    
    # تنظیمات ساده‌شده
    BINANCE_DELAY = config.getfloat('Rate_Limits', 'binance_delay', fallback=0.1)
    RSS_DELAY = config.getfloat('Rate_Limits', 'rss_delay', fallback=0.5)
    REQUEST_TIMEOUT = config.getint('Rate_Limits', 'request_timeout', fallback=30)
    MAX_RETRIES = config.getint('Rate_Limits', 'max_retries', fallback=3)
    
except Exception as e:
    print(f"خطا در خواندن config.ini: {e}")
    print("از تنظیمات پیش‌فرض استفاده می‌شود...")
    RAW_DATA_PATH = './data/raw'
    LOG_PATH = './data/logs'
    BINANCE_DELAY = 0.1
    RSS_DELAY = 0.5
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

# ایجاد پوشه‌های مورد نیاز
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

class SimpleStateManager:
    """
    مدیریت وضعیت ساده‌شده برای tracking پیشرفت
    
    ویژگی‌ها:
    - SQLite database برای ذخیره وضعیت
    - tracking سشن‌های قیمت و اخبار
    - آمار موفقیت/شکست
    - بازیابی آسان وضعیت
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(RAW_DATA_PATH, 'simple_extraction_state.db')
        
        self.db_path = db_path
        self.setup_database()
        logging.info(f"💾 مدیریت وضعیت ساده اولیه‌سازی شد: {db_path}")
    
    def setup_database(self):
        """ایجاد جداول مورد نیاز"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                -- جدول سشن‌ها
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    session_type TEXT CHECK(session_type IN ('price', 'news', 'both')),
                    status TEXT DEFAULT 'active',
                    total_items INTEGER DEFAULT 0,
                    completed_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- جدول پیشرفت قیمت
                CREATE TABLE IF NOT EXISTS price_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    status TEXT DEFAULT 'pending',
                    file_path TEXT,
                    records_count INTEGER,
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                
                -- جدول پیشرفت اخبار
                CREATE TABLE IF NOT EXISTS news_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol TEXT,
                    status TEXT DEFAULT 'pending',
                    file_path TEXT,
                    news_count INTEGER,
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            ''')
        logging.info("✅ پایگاه داده وضعیت آماده شد")
    
    def create_session(self, session_type: str, total_items: int) -> str:
        """ایجاد سشن جدید"""
        session_id = f"{session_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO sessions (session_id, session_type, total_items)
                VALUES (?, ?, ?)
            ''', (session_id, session_type, total_items))
        
        logging.info(f"🆕 سشن جدید ایجاد شد: {session_id} ({session_type}) - {total_items} آیتم")
        return session_id
    
    def update_price_progress(self, session_id: str, symbol: str, timeframe: str, 
                            status: str, **kwargs):
        """بروزرسانی پیشرفت قیمت"""
        with sqlite3.connect(self.db_path) as conn:
            # بررسی وجود رکورد
            cursor = conn.execute('''
                SELECT id FROM price_progress 
                WHERE session_id = ? AND symbol = ? AND timeframe = ?
            ''', (session_id, symbol, timeframe))
            
            if cursor.fetchone():
                # بروزرسانی
                conn.execute('''
                    UPDATE price_progress 
                    SET status = ?, file_path = ?, records_count = ?, 
                        error_message = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE session_id = ? AND symbol = ? AND timeframe = ?
                ''', (status, kwargs.get('file_path'), kwargs.get('records_count'),
                     kwargs.get('error_message'), session_id, symbol, timeframe))
            else:
                # درج جدید
                conn.execute('''
                    INSERT INTO price_progress 
                    (session_id, symbol, timeframe, status, file_path, records_count, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, symbol, timeframe, status, kwargs.get('file_path'),
                     kwargs.get('records_count'), kwargs.get('error_message')))
            
            # بروزرسانی آمار سشن
            if status == 'completed':
                self._update_session_stats(session_id, 'completed')
            elif status == 'failed':
                self._update_session_stats(session_id, 'failed')
    
    def update_news_progress(self, session_id: str, symbol: str, status: str, **kwargs):
        """بروزرسانی پیشرفت اخبار"""
        with sqlite3.connect(self.db_path) as conn:
            # بررسی وجود رکورد
            cursor = conn.execute('''
                SELECT id FROM news_progress 
                WHERE session_id = ? AND symbol = ?
            ''', (session_id, symbol))
            
            if cursor.fetchone():
                # بروزرسانی
                conn.execute('''
                    UPDATE news_progress 
                    SET status = ?, file_path = ?, news_count = ?, 
                        error_message = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE session_id = ? AND symbol = ?
                ''', (status, kwargs.get('file_path'), kwargs.get('news_count'),
                     kwargs.get('error_message'), session_id, symbol))
            else:
                # درج جدید
                conn.execute('''
                    INSERT INTO news_progress 
                    (session_id, symbol, status, file_path, news_count, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, symbol, status, kwargs.get('file_path'),
                     kwargs.get('news_count'), kwargs.get('error_message')))
            
            # بروزرسانی آمار سشن
            if status == 'completed':
                self._update_session_stats(session_id, 'completed')
            elif status == 'failed':
                self._update_session_stats(session_id, 'failed')
    
    def _update_session_stats(self, session_id: str, result_type: str):
        """بروزرسانی آمار سشن"""
        with sqlite3.connect(self.db_path) as conn:
            if result_type == 'completed':
                conn.execute('''
                    UPDATE sessions 
                    SET completed_items = completed_items + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (session_id,))
            elif result_type == 'failed':
                conn.execute('''
                    UPDATE sessions 
                    SET failed_items = failed_items + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (session_id,))
    
    def get_session_stats(self, session_id: str) -> Dict:
        """دریافت آمار سشن"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT session_type, status, total_items, completed_items, failed_items, created_at
                FROM sessions WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'session_id': session_id,
                    'session_type': row[0],
                    'status': row[1],
                    'total_items': row[2],
                    'completed_items': row[3],
                    'failed_items': row[4],
                    'created_at': row[5],
                    'success_rate': round((row[3] / row[2]) * 100, 1) if row[2] > 0 else 0
                }
            return None
    
    def complete_session(self, session_id: str):
        """اتمام سشن"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE sessions 
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (session_id,))
        logging.info(f"✅ سشن تکمیل شد: {session_id}")

class SimpleRateLimiter:
    """
    مدیریت ساده نرخ درخواست
    
    ویژگی‌ها:
    - تأخیر بین درخواست‌ها
    - تلاش مجدد در صورت خطا
    - آمار استفاده
    """
    
    def __init__(self):
        self.last_request_time = {}
        self.request_counts = {
            'binance': 0,
            'rss': 0
        }
        self.lock = threading.Lock()
        logging.info("🚦 مدیریت ساده نرخ درخواست آماده شد")
    
    def wait_if_needed(self, source: str):
        """اعمال تأخیر در صورت نیاز"""
        with self.lock:
            current_time = time.time()
            
            # تعیین تأخیر بر اساس منبع
            if source == 'binance':
                required_delay = BINANCE_DELAY
            elif source == 'rss':
                required_delay = RSS_DELAY
            else:
                required_delay = 1.0
            
            # محاسبه تأخیر مورد نیاز
            if source in self.last_request_time:
                elapsed = current_time - self.last_request_time[source]
                if elapsed < required_delay:
                    wait_time = required_delay - elapsed
                    time.sleep(wait_time)
            
            # ثبت زمان درخواست
            self.last_request_time[source] = time.time()
            self.request_counts[source] += 1
    
    def get_stats(self) -> Dict:
        """دریافت آمار استفاده"""
        return {
            'binance_requests': self.request_counts['binance'],
            'rss_requests': self.request_counts['rss'],
            'total_requests': sum(self.request_counts.values())
        }

def setup_logging(script_name: str) -> str:
    """
    تنظیم سیستم لاگ‌گیری
    
    Args:
        script_name: نام اسکریپت برای ایجاد پوشه جداگانه
    
    Returns:
        مسیر فایل لاگ
    """
    # ایجاد پوشه لاگ برای اسکریپت
    log_subfolder_path = os.path.join(LOG_PATH, script_name)
    os.makedirs(log_subfolder_path, exist_ok=True)
    
    # نام فایل لاگ با timestamp
    log_filename = os.path.join(
        log_subfolder_path, 
        f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    # تنظیم logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"📝 سیستم لاگ‌گیری آماده شد: {log_filename}")
    return log_filename

def safe_request(url: str, params: Dict = None, headers: Dict = None, 
                timeout: int = None, max_retries: int = None) -> requests.Response:
    """
    ارسال درخواست ایمن با تلاش مجدد
    
    Args:
        url: آدرس درخواست
        params: پارامترهای URL
        headers: هدرهای HTTP
        timeout: مهلت زمانی (ثانیه)
        max_retries: حداکثر تلاش مجدد
    
    Returns:
        Response object
    
    Raises:
        requests.RequestException: در صورت شکست نهایی
    """
    if timeout is None:
        timeout = REQUEST_TIMEOUT
    if max_retries is None:
        max_retries = MAX_RETRIES
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                params=params, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logging.error(f"❌ درخواست ناموفق بعد از {max_retries} تلاش: {e}")
                raise
            
            wait_time = 2 ** attempt  # Exponential backoff
            logging.warning(f"⚠️ تلاش {attempt + 1} ناموفق. انتظار {wait_time} ثانیه...")
            time.sleep(wait_time)

def sanitize_filename(filename: str) -> str:
    """
    پاکسازی نام فایل از کاراکترهای غیرمجاز
    
    Args:
        filename: نام فایل اصلی
    
    Returns:
        نام فایل پاکسازی شده
    """
    # جایگزینی کاراکترهای غیرمجاز
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # جایگزینی / با -
    filename = filename.replace('/', '-')
    
    return filename

def get_user_selection(options: List[str], title: str, 
                      allow_manual: bool = False, 
                      allow_multi: bool = False, 
                      allow_all: bool = False) -> List[str]:
    """
    منوی انتخاب برای کاربر
    
    Args:
        options: لیست گزینه‌ها
        title: عنوان منو
        allow_manual: امکان ورود دستی
        allow_multi: امکان انتخاب چندگانه
        allow_all: امکان انتخاب همه
    
    Returns:
        لیست موارد انتخاب شده
    """
    print(f"\n--- {title} ---")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    if allow_all:
        print(f"{len(options) + 1}. همه موارد بالا")
    if allow_manual:
        print(f"{len(options) + 2 if allow_all else len(options) + 1}. ورود دستی")
    
    if allow_multi:
        prompt = "شماره موارد مورد نظر (با کاما جدا کنید): "
    else:
        prompt = "شماره مورد نظر: "
    
    choice_str = input(prompt).strip()
    
    if not choice_str:
        logging.error("ورودی خالی.")
        return []
    
    try:
        if allow_all and choice_str == str(len(options) + 1):
            return options
        
        manual_entry_num = len(options) + 2 if allow_all else len(options) + 1
        if allow_manual and choice_str == str(manual_entry_num):
            manual_input = input("لطفاً مورد نظر خود را تایپ کنید (برای چند مورد، با کاما جدا کنید): ").upper()
            return [item.strip() for item in manual_input.split(',')]
        
        if allow_multi:
            selected_indices = [int(i.strip()) - 1 for i in choice_str.split(',')]
            return [options[i] for i in selected_indices if 0 <= i < len(options)]
        else:
            idx = int(choice_str) - 1
            if 0 <= idx < len(options):
                return [options[idx]]
    
    except (ValueError, IndexError):
        logging.error("ورودی نامعتبر.")
    
    return []

# پارامترهای مشترک
COMMON_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", 
    "ADA/USDT", "DOGE/USDT", "SHIB/USDT", "TRX/USDT", "MATIC/USDT",
    "LTC/USDT", "DOT/USDT", "AVAX/USDT", "LINK/USDT", "BCH/USDT",
    "UNI/USDT", "FIL/USDT", "ETC/USDT", "ATOM/USDT", "ICP/USDT",
    "VET/USDT", "OP/USDT", "ARB/USDT", "APT/USDT", "NEAR/USDT",
    "FTM/USDT", "RNDR/USDT", "GRT/USDT", "MANA/USDT", "SAND/USDT"
]

COMMON_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

DEFAULT_LIMIT = 2000
MAX_NEWS_PER_SYMBOL = 10

# اولیه‌سازی اجزای مشترک
state_manager = SimpleStateManager()
rate_limiter = SimpleRateLimiter()

logging.info("🚀 اجزای مشترک ساده‌شده آماده شد")
logging.info(f"📊 پیش‌فرض: {len(COMMON_SYMBOLS)} نماد، {len(COMMON_TIMEFRAMES)} تایم‌فریم")
