#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت یکپارچه استخراج داده‌های قیمت و اخبار (نسخه اصلاح شده نهایی)

این اسکریپت ادغامی از fetch_historical_data_01.py و fetch_news_01a.py است
با قابلیت استخراج هماهنگ داده‌های قیمت و اخبار برای نمادهای انتخابی

ویژگی‌ها:
- State Management یکپارچه برای قیمت و اخبار
- استخراج هماهنگ بر اساس نماد و بازه زمانی
- مدیریت Rate Limit مشترک
- منوی تعاملی کامل
- استخراج همه نمادها در همه تایم‌فریم‌ها
- اخبار فقط به زبان انگلیسی (برای کاهش مصرف API)
- حلقه اصلی برای نگه‌داشتن برنامه فعال
- Backfill کامل برای تکمیل داده‌های از دست رفته
- منابع خبری چندگانه: GNews + NewsAPI + CoinGecko + RSS (جدید)
"""

import os
import time
import pandas as pd
import requests
import logging
import configparser
from datetime import datetime, timezone, timedelta
import threading
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
import threading
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === imports جدید برای منابع خبری اضافی ===
try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    logging.warning("feedparser not available. RSS feeds disabled.")

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False
    logging.warning("concurrent.futures not available. Parallel processing disabled.")

# --- بخش خواندن پیکربندی ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    RAW_DATA_PATH = config.get('Paths', 'raw')
    LOG_PATH = config.get('Paths', 'logs')
    
    # کلیدهای API موجود
    CRYPTOCOMPARE_API_KEY = config.get('API_Keys', 'cryptocompare_api_key', fallback=None)
    GNEWS_API_KEY = config.get('API_Keys', 'gnews_api_key', fallback=None)
    
    # === کلیدهای API جدید ===
    NEWSAPI_KEY = config.get('API_Keys', 'newsapi_key', fallback=None)
    ALPHA_VANTAGE_KEY = config.get('API_Keys', 'alpha_vantage_key', fallback=None)
    
    # === تنظیمات فعال‌سازی منابع ===
    GNEWS_ENABLED = config.getboolean('News_Sources', 'gnews_enabled', fallback=True)
    NEWSAPI_ENABLED = config.getboolean('News_Sources', 'newsapi_enabled', fallback=True)
    COINGECKO_ENABLED = config.getboolean('News_Sources', 'coingecko_enabled', fallback=True)
    RSS_ENABLED = config.getboolean('News_Sources', 'rss_enabled', fallback=True)
    PARALLEL_FETCHING = config.getboolean('News_Sources', 'parallel_fetching', fallback=True)
    REMOVE_DUPLICATES = config.getboolean('News_Sources', 'remove_duplicates', fallback=True)
    
    # Rate Limits موجود
    CRYPTOCOMPARE_DELAY = config.getfloat('Rate_Limits', 'cryptocompare_delay', fallback=0.6)
    BINANCE_DELAY = config.getfloat('Rate_Limits', 'binance_delay', fallback=0.1)
    KRAKEN_DELAY = config.getfloat('Rate_Limits', 'kraken_delay', fallback=1.5)
    GNEWS_DELAY = config.getfloat('Rate_Limits', 'gnews_delay', fallback=1.0)
    
    # === Rate Limits جدید ===
    NEWSAPI_DELAY = config.getfloat('Rate_Limits', 'newsapi_delay', fallback=2.0)
    COINGECKO_DELAY = config.getfloat('Rate_Limits', 'coingecko_delay', fallback=1.0)
    RSS_DELAY = config.getfloat('Rate_Limits', 'rss_delay', fallback=0.5)
    
    # محدودیت‌های موجود
    DAILY_LIMIT = config.getint('Rate_Limits', 'cryptocompare_daily_limit', fallback=3200)
    HOURLY_LIMIT = config.getint('Rate_Limits', 'cryptocompare_hourly_limit', fallback=135)
    GNEWS_DAILY_LIMIT = config.getint('Rate_Limits', 'gnews_daily_limit', fallback=100)
    GNEWS_HOURLY_LIMIT = config.getint('Rate_Limits', 'gnews_hourly_limit', fallback=10)
    
    # === محدودیت‌های جدید ===
    NEWSAPI_DAILY_LIMIT = config.getint('Rate_Limits', 'newsapi_daily_limit', fallback=33)
    NEWSAPI_MONTHLY_LIMIT = config.getint('Rate_Limits', 'newsapi_monthly_limit', fallback=1000)
    
    MAX_REQUESTS_PER_SESSION = config.getint('Data_Settings', 'max_requests_per_session', fallback=500)
    
    # === تنظیمات RSS ===
    RSS_CACHE_MINUTES = config.getint('RSS_Feeds', 'rss_cache_minutes', fallback=5)
    MAX_ARTICLES_PER_FEED = config.getint('RSS_Feeds', 'max_articles_per_feed', fallback=20)
    RSS_TIMEOUT = config.getint('RSS_Feeds', 'rss_timeout', fallback=10)
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Error: {e}")
    exit()

# تنظیمات logging
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# استفاده مستقیم از مسیر raw بدون ایجاد زیرپوشه
price_data_path = RAW_DATA_PATH
news_data_path = RAW_DATA_PATH
os.makedirs(price_data_path, exist_ok=True)
os.makedirs(news_data_path, exist_ok=True)

log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# پارامترهای پیش‌فرض - همه نمادهای مهم
COMMON_PAIRS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT", 
    "DOGE/USDT", "SHIB/USDT", "TRX/USDT", "MATIC/USDT", "LTC/USDT", "DOT/USDT", 
    "AVAX/USDT", "LINK/USDT", "BCH/USDT", "UNI/USDT", "FIL/USDT", "ETC/USDT", 
    "ATOM/USDT", "ICP/USDT", "VET/USDT", "OP/USDT", "ARB/USDT", "APT/USDT", 
    "NEAR/USDT", "FTM/USDT", "RNDR/USDT", "GRT/USDT", "MANA/USDT", "SAND/USDT"
]
COMMON_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
LIMIT = 2000
MAX_NEWS_PER_SYMBOL = 10

# --- کلاس State Management یکپارچه ---
class UnifiedStateManager:
    """مدیریت state یکپارچه برای داده‌های قیمت و اخبار"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(RAW_DATA_PATH, 'unified_extraction_state.db')
        self.db_path = db_path
        self.setup_database()
        logging.info(f"💾 Unified State Manager اولیه‌سازی شد: {db_path}")
    
    def setup_database(self):
        """ایجاد جداول مورد نیاز"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                -- جدول اصلی sessions
                CREATE TABLE IF NOT EXISTS extraction_sessions (
                    session_id TEXT PRIMARY KEY,
                    session_type TEXT CHECK(session_type IN ('price', 'news', 'unified')),
                    status TEXT DEFAULT 'active',
                    total_symbols INTEGER,
                    completed_symbols INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- جدول پیشرفت داده‌های قیمت
                CREATE TABLE IF NOT EXISTS price_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    exchange TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    status TEXT DEFAULT 'pending',
                    file_path TEXT,
                    records_count INTEGER,
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES extraction_sessions(session_id)
                );
                
                -- جدول پیشرفت اخبار
                CREATE TABLE IF NOT EXISTS news_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol TEXT,
                    language TEXT,
                    status TEXT DEFAULT 'pending',
                    file_path TEXT,
                    news_count INTEGER,
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES extraction_sessions(session_id)
                );
                
                -- جدول rate limits
                CREATE TABLE IF NOT EXISTS rate_limits (
                    api_name TEXT PRIMARY KEY,
                    daily_count INTEGER DEFAULT 0,
                    hourly_count INTEGER DEFAULT 0,
                    last_daily_reset TEXT,
                    last_hourly_reset TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- جدول failed items
                CREATE TABLE IF NOT EXISTS failed_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_type TEXT CHECK(item_type IN ('price', 'news')),
                    exchange TEXT,
                    symbol TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
        logging.info("✅ Unified Database تنظیم شد")
    
    def create_unified_session(self, symbols: List[str], include_price: bool = True, 
                              include_news: bool = True) -> str:
        """ایجاد session یکپارچه جدید"""
        session_id = f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_type = 'unified' if include_price and include_news else ('price' if include_price else 'news')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO extraction_sessions 
                (session_id, session_type, total_symbols, status)
                VALUES (?, ?, ?, 'active')
            ''', (session_id, session_type, len(symbols)))
        
        logging.info(f"🆕 Unified Session جدید: {session_id} (نوع: {session_type})")
        return session_id
    
    def update_price_progress(self, session_id: str, exchange: str, symbol: str, 
                            timeframe: str, status: str, **kwargs):
        """بروزرسانی پیشرفت داده‌های قیمت"""
        with sqlite3.connect(self.db_path) as conn:
            # چک وجود رکورد
            cursor = conn.execute('''
                SELECT id FROM price_progress 
                WHERE session_id = ? AND exchange = ? AND symbol = ? AND timeframe = ?
            ''', (session_id, exchange, symbol, timeframe))
            
            if cursor.fetchone():
                # بروزرسانی
                update_fields = ['status = ?']
                update_values = [status]
                
                if 'file_path' in kwargs:
                    update_fields.append('file_path = ?')
                    update_values.append(kwargs['file_path'])
                if 'records_count' in kwargs:
                    update_fields.append('records_count = ?')
                    update_values.append(kwargs['records_count'])
                if 'error_message' in kwargs:
                    update_fields.append('error_message = ?')
                    update_values.append(kwargs['error_message'])
                
                if status == 'completed':
                    update_fields.append('completed_at = CURRENT_TIMESTAMP')
                
                update_values.extend([session_id, exchange, symbol, timeframe])
                
                conn.execute(f'''
                    UPDATE price_progress 
                    SET {', '.join(update_fields)}
                    WHERE session_id = ? AND exchange = ? AND symbol = ? AND timeframe = ?
                ''', update_values)
            else:
                # درج جدید
                conn.execute('''
                    INSERT INTO price_progress 
                    (session_id, exchange, symbol, timeframe, status, file_path, records_count, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, exchange, symbol, timeframe, status, 
                     kwargs.get('file_path'), kwargs.get('records_count'), kwargs.get('error_message')))
            
            # بروزرسانی session
            if status == 'completed':
                conn.execute('''
                    UPDATE extraction_sessions 
                    SET completed_symbols = completed_symbols + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (session_id,))
    
    def update_news_progress(self, session_id: str, symbol: str, language: str, 
                           status: str, **kwargs):
        """بروزرسانی پیشرفت اخبار"""
        with sqlite3.connect(self.db_path) as conn:
            # چک وجود رکورد
            cursor = conn.execute('''
                SELECT id FROM news_progress 
                WHERE session_id = ? AND symbol = ? AND language = ?
            ''', (session_id, symbol, language))
            
            if cursor.fetchone():
                # بروزرسانی
                update_fields = ['status = ?']
                update_values = [status]
                
                if 'file_path' in kwargs:
                    update_fields.append('file_path = ?')
                    update_values.append(kwargs['file_path'])
                if 'news_count' in kwargs:
                    update_fields.append('news_count = ?')
                    update_values.append(kwargs['news_count'])
                if 'error_message' in kwargs:
                    update_fields.append('error_message = ?')
                    update_values.append(kwargs['error_message'])
                
                if status == 'completed':
                    update_fields.append('completed_at = CURRENT_TIMESTAMP')
                
                update_values.extend([session_id, symbol, language])
                
                conn.execute(f'''
                    UPDATE news_progress 
                    SET {', '.join(update_fields)}
                    WHERE session_id = ? AND symbol = ? AND language = ?
                ''', update_values)
            else:
                # درج جدید
                conn.execute('''
                    INSERT INTO news_progress 
                    (session_id, symbol, language, status, file_path, news_count, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, symbol, language, status, 
                     kwargs.get('file_path'), kwargs.get('news_count'), kwargs.get('error_message')))
    
    def get_session_status(self, session_id: str) -> Dict:
        """دریافت وضعیت کامل session"""
        with sqlite3.connect(self.db_path) as conn:
            # اطلاعات session
            cursor = conn.execute('''
                SELECT * FROM extraction_sessions WHERE session_id = ?
            ''', (session_id,))
            session_info = cursor.fetchone()
            
            if not session_info:
                return None
            
            # پیشرفت قیمت
            cursor = conn.execute('''
                SELECT COUNT(*) total, 
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) completed
                FROM price_progress WHERE session_id = ?
            ''', (session_id,))
            price_progress = cursor.fetchone()
            
            # پیشرفت اخبار
            cursor = conn.execute('''
                SELECT COUNT(*) total, 
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) completed
                FROM news_progress WHERE session_id = ?
            ''', (session_id,))
            news_progress = cursor.fetchone()
            
            return {
                'session_id': session_info[0],
                'session_type': session_info[1],
                'status': session_info[2],
                'total_symbols': session_info[3],
                'completed_symbols': session_info[4],
                'price_progress': {
                    'total': price_progress[0] if price_progress else 0,
                    'completed': price_progress[1] if price_progress else 0
                },
                'news_progress': {
                    'total': news_progress[0] if news_progress else 0,
                    'completed': news_progress[1] if news_progress else 0
                }
            }
    
    def add_failed_item(self, item_type: str, symbol: str, error_msg: str, exchange: str = None):
        """اضافه کردن آیتم شکست خورده"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO failed_items (item_type, exchange, symbol, error_message)
                VALUES (?, ?, ?, ?)
            ''', (item_type, exchange, symbol, error_msg))
    
    def is_failed_item(self, item_type: str, symbol: str, exchange: str = None) -> bool:
        """بررسی آیا آیتم قبلاً شکست خورده"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 1 FROM failed_items 
                WHERE item_type = ? AND symbol = ? AND (exchange = ? OR exchange IS NULL)
            ''', (item_type, symbol, exchange))
            return cursor.fetchone() is not None

# --- کلاس مدیریت Rate Limit یکپارچه (بهبود یافته) ---
class UnifiedRateLimiter:
    """مدیریت هوشمند نرخ درخواست برای همه API ها"""
    
    def __init__(self, state_manager: UnifiedStateManager):
        self.state_manager = state_manager
        self.last_request_time = {}
        self.request_counters = {
            'CryptoCompare': {'daily': 0, 'hourly': 0, 'session': 0},
            'Binance': {'session': 0},
            'Kraken': {'session': 0},
            'GNews': {'daily': 0, 'hourly': 0, 'session': 0},
            # === منابع جدید ===
            'NewsAPI': {'daily': 0, 'monthly': 0, 'session': 0},
            'CoinGecko': {'session': 0},
            'RSS': {'session': 0}
        }
        
        self.min_intervals = {
            'CryptoCompare': CRYPTOCOMPARE_DELAY,
            'Binance': BINANCE_DELAY,
            'Kraken': KRAKEN_DELAY,
            'GNews': GNEWS_DELAY,
            # === منابع جدید ===
            'NewsAPI': NEWSAPI_DELAY,
            'CoinGecko': COINGECKO_DELAY,
            'RSS': RSS_DELAY
        }
        
        self.limits = {
            'CryptoCompare': {
                'daily': DAILY_LIMIT,
                'hourly': HOURLY_LIMIT,
                'session': MAX_REQUESTS_PER_SESSION
            },
            'GNews': {
                'daily': GNEWS_DAILY_LIMIT,
                'hourly': GNEWS_HOURLY_LIMIT,
                'session': MAX_REQUESTS_PER_SESSION
            },
            # === منابع جدید ===
            'NewsAPI': {
                'daily': NEWSAPI_DAILY_LIMIT,
                'monthly': NEWSAPI_MONTHLY_LIMIT,
                'session': MAX_REQUESTS_PER_SESSION
            }
        }
        
        self.lock = threading.Lock()
        self.load_persisted_state()
        self.hour_start = time.time()
        self.day_start = time.time()
        logging.info(f"🔧 Enhanced Rate Limiter اولیه‌سازی شد")
    
    def load_persisted_state(self):
        """بارگذاری وضعیت از database"""
        with sqlite3.connect(self.state_manager.db_path) as conn:
            for api_name in ['CryptoCompare', 'GNews', 'NewsAPI']:
                cursor = conn.execute('SELECT * FROM rate_limits WHERE api_name = ?', (api_name,))
                row = cursor.fetchone()
                
                if row and api_name in self.request_counters:
                    self.request_counters[api_name]['daily'] = row[1]
                    self.request_counters[api_name]['hourly'] = row[2]
                    
                    # بررسی نیاز به ریست
                    if row[3]:  # last_daily_reset
                        last_daily = datetime.fromisoformat(row[3])
                        if (datetime.now() - last_daily).days >= 1:
                            self.reset_daily_counter(api_name)
                    
                    if row[4]:  # last_hourly_reset  
                        last_hourly = datetime.fromisoformat(row[4])
                        if (datetime.now() - last_hourly).total_seconds() >= 3600:
                            self.reset_hourly_counter(api_name)
    
    def save_state(self, api_name: str):
        """ذخیره وضعیت در database"""
        if api_name not in ['CryptoCompare', 'GNews', 'NewsAPI']:
            return
            
        counters = self.request_counters[api_name]
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.state_manager.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO rate_limits 
                (api_name, daily_count, hourly_count, last_daily_reset, last_hourly_reset)
                VALUES (?, ?, ?, ?, ?)
            ''', (api_name, counters.get('daily', 0), counters.get('hourly', 0), now, now))
    
    def reset_daily_counter(self, api_name: str):
        """ریست شمارنده روزانه"""
        if api_name in self.request_counters:
            self.request_counters[api_name]['daily'] = 0
            logging.info(f"🔄 شمارنده روزانه {api_name} ریست شد")
            self.save_state(api_name)
    
    def reset_hourly_counter(self, api_name: str):
        """ریست شمارنده ساعتی"""
        if api_name in self.request_counters:
            self.request_counters[api_name]['hourly'] = 0
            logging.info(f"🔄 شمارنده ساعتی {api_name} ریست شد")
            self.save_state(api_name)
    
    def check_and_wait_for_reset(self, api_name: str) -> bool:
        """بررسی محدودیت و انتظار برای ریست در صورت نیاز"""
        if api_name not in self.limits:
            return True
        
        counters = self.request_counters[api_name]
        limits = self.limits[api_name]
        
        # بررسی محدودیت ساعتی
        if 'hourly' in limits and counters.get('hourly', 0) >= limits['hourly']:
            logging.warning(f"⏳ محدودیت ساعتی {api_name} رسیده - انتظار تا ریست...")
            
            # محاسبه زمان تا ریست
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_seconds = (next_hour - now).total_seconds()
            
            logging.info(f"⏰ انتظار {wait_seconds:.0f} ثانیه تا ریست ساعتی...")
            
            # انتظار با نمایش پیشرفت
            for remaining in range(int(wait_seconds), 0, -60):
                minutes = remaining // 60
                logging.info(f"⏳ {minutes} دقیقه تا ریست ساعتی...")
                time.sleep(min(60, remaining))
            
            # ریست شمارنده
            self.reset_hourly_counter(api_name)
            logging.info("✅ محدودیت ساعتی ریست شد - ادامه کار...")
            return True
        
        # بررسی محدودیت روزانه
        if 'daily' in limits and counters.get('daily', 0) >= limits['daily']:
            logging.warning(f"⏳ محدودیت روزانه {api_name} رسیده - انتظار تا ریست...")
            
            # محاسبه زمان تا ریست
            now = datetime.now()
            next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            wait_seconds = (next_day - now).total_seconds()
            
            logging.info(f"⏰ انتظار {wait_seconds:.0f} ثانیه تا ریست روزانه...")
            
            # انتظار طولانی
            time.sleep(wait_seconds)
            self.reset_daily_counter(api_name)
            return True
        
        return True
    
    def wait_if_needed(self, api_name: str) -> bool:
        """اعمال تأخیر و بررسی محدودیت"""
        with self.lock:
            # بررسی و انتظار برای ریست در صورت نیاز
            if not self.check_and_wait_for_reset(api_name):
                return False
            
            # اعمال تأخیر معمول
            current_time = time.time()
            if api_name in self.last_request_time:
                elapsed = current_time - self.last_request_time[api_name]
                required_interval = self.min_intervals.get(api_name, 1.0)
                
                if elapsed < required_interval:
                    wait_time = required_interval - elapsed
                    time.sleep(wait_time)
            
            # ثبت زمان و بروزرسانی شمارنده
            self.last_request_time[api_name] = time.time()
            
            if api_name in self.request_counters:
                self.request_counters[api_name]['session'] += 1
                if api_name in ['CryptoCompare', 'GNews', 'NewsAPI']:
                    if 'daily' in self.request_counters[api_name]:
                        self.request_counters[api_name]['daily'] += 1
                    if 'hourly' in self.request_counters[api_name]:
                        self.request_counters[api_name]['hourly'] += 1
                    self.save_state(api_name)
            
            return True
    
    def get_stats(self, api_name: str) -> dict:
        """دریافت آمار استفاده"""
        if api_name not in self.request_counters:
            return {}
            
        counters = self.request_counters[api_name]
        if api_name in self.limits:
            limits = self.limits[api_name]
            result = {}
            if 'daily' in limits:
                result['daily'] = f"{counters.get('daily', 0)}/{limits['daily']}"
            if 'hourly' in limits:
                result['hourly'] = f"{counters.get('hourly', 0)}/{limits['hourly']}"
            result['session'] = f"{counters['session']}/{limits.get('session', 'N/A')}"
            return result
        else:
            return {'session': counters['session']}

# --- توابع کمکی ---
def safe_request(url: str, params: dict = None, headers: dict = None, 
                api_name: str = None, max_retries: int = 3) -> requests.Response:
    """ارسال درخواست ایمن با تلاش مجدد"""
    for retry in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if retry == max_retries - 1:
                logging.error(f"خطا در درخواست پس از {max_retries} تلاش: {e}")
                raise
            
            wait_time = 2 ** retry
            logging.warning(f"خطا در درخواست. انتظار {wait_time} ثانیه قبل از تلاش مجدد...")
            time.sleep(wait_time)

# --- توابع استخراج داده قیمت (بدون تغییر) ---
def fetch_from_cryptocompare(symbol: str, timeframe: str, limit: int, to_ts: int = None) -> pd.DataFrame:
    """استخراج داده از CryptoCompare API"""
    if not CRYPTOCOMPARE_API_KEY:
        logging.warning("کلید API برای CryptoCompare تنظیم نشده است.")
        return pd.DataFrame()
    
    BASE_URL = "https://min-api.cryptocompare.com/data/v2/"
    endpoint_map = {'m': 'histominute', 'h': 'histohour', 'd': 'histoday'}
    
    try:
        tf_unit = timeframe.lower()[-1]
        tf_agg = int(timeframe[:-1])
        endpoint = endpoint_map.get(tf_unit)
        if not endpoint: raise ValueError("Timeframe unit not recognized.")
        base_sym, quote_sym = symbol.upper().split('/')
    except Exception:
        logging.error(f"[CryptoCompare] تایم‌فریم یا نماد نامعتبر: '{timeframe}', '{symbol}'")
        return pd.DataFrame()
    
    params = {"fsym": base_sym, "tsym": quote_sym, "limit": limit, "aggregate": tf_agg}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    if to_ts:
        params['toTs'] = to_ts
    
    logging.info(f"[CryptoCompare] در حال استخراج داده برای {symbol} | {timeframe}...")
    
    try:
        response = safe_request(f"{BASE_URL}{endpoint}", params=params, api_name="CryptoCompare")
        data = response.json()
        
        if data.get('Response') == 'Error':
            error_msg = data.get('Message', 'Unknown error')
            logging.error(f"[CryptoCompare] خطا از API: {error_msg}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['Data']['Data'])
        if df.empty: return pd.DataFrame()
        
        df.rename(columns={'volumefrom': 'volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['exchange'] = 'CryptoCompare'
        
        return df
        
    except Exception as e:
        logging.error(f"[CryptoCompare] خطای پیش‌بینی نشده: {e}")
        return pd.DataFrame()

def fetch_from_binance(symbol: str, timeframe: str, limit: int, **kwargs) -> pd.DataFrame:
    """استخراج داده از Binance API"""
    try:
        binance_symbol = symbol.replace('/', '').upper()
        timeframe_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
        binance_interval = timeframe_map.get(timeframe, timeframe)
        params = {
            'symbol': binance_symbol,
            'interval': binance_interval,
            'limit': min(limit, 1000)
        }
        
        logging.info(f"[Binance] در حال استخراج داده برای {symbol} | {timeframe}...")
        
        response = safe_request("https://api.binance.com/api/v3/klines", params=params, api_name="Binance")
        data = response.json()
        
        if not data:
            logging.warning(f"[Binance] داده‌ای برای {symbol} | {timeframe} دریافت نشد.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['exchange'] = 'Binance'
        
        return df
        
    except Exception as e:
        logging.error(f"[Binance] خطا در دریافت داده برای {symbol} | {timeframe}: {e}")
        return pd.DataFrame()

def fetch_from_kraken(symbol: str, timeframe: str, limit: int, **kwargs) -> pd.DataFrame:
    """استخراج داده از Kraken API"""
    try:
        symbol_map = {
            'BTC/USDT': 'XBTUSD', 'BTC/USD': 'XBTUSD',
            'ETH/USDT': 'ETHUSD', 'ETH/USD': 'ETHUSD',
            'XRP/USDT': 'XRPUSD', 'XRP/USD': 'XRPUSD',
            'LTC/USDT': 'LTCUSD', 'LTC/USD': 'LTCUSD',
            'ADA/USDT': 'ADAUSD', 'ADA/USD': 'ADAUSD',
            'DOT/USDT': 'DOTUSD', 'DOT/USD': 'DOTUSD',
            'SOL/USDT': 'SOLUSD', 'SOL/USD': 'SOLUSD'
        }
        
        kraken_symbol = symbol_map.get(symbol.upper(), symbol.replace('/', '').upper())
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
        }
        
        interval = timeframe_minutes.get(timeframe, 60)
        end_date = datetime.now()
        days_back = min(365, limit * interval // 1440) if interval < 1440 else 365
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'pair': kraken_symbol,
            'interval': interval,
            'since': int(start_date.timestamp())
        }
        
        logging.info(f"[Kraken] در حال استخراج داده برای {symbol} | {timeframe}...")
        
        response = safe_request("https://api.kraken.com/0/public/OHLC", params=params, api_name="Kraken")
        result = response.json()
        
        if 'error' in result and result['error']:
            logging.error(f"[Kraken] خطای API: {result['error']}")
            return pd.DataFrame()
        
        if 'result' not in result:
            logging.warning(f"[Kraken] ساختار پاسخ غیرمنتظره برای {symbol}")
            return pd.DataFrame()
        
        data_key = list(result['result'].keys())[0]
        data = result['result'][data_key]
        
        if not data:
            logging.warning(f"[Kraken] داده‌ای برای {symbol} | {timeframe} دریافت نشد.")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['exchange'] = 'Kraken'
        
        return df
        
    except Exception as e:
        logging.error(f"[Kraken] خطا در دریافت داده برای {symbol} | {timeframe}: {e}")
        return pd.DataFrame()

# === کلاس‌های جدید برای منابع خبری اضافی ===

class NewsAPIFetcher:
    """استخراج اخبار از NewsAPI.org - 1000 درخواست/ماه رایگان"""
    
    def __init__(self, api_key: str, rate_limiter: UnifiedRateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.base_url = "https://newsapi.org/v2/everything"
        
    def fetch_crypto_news(self, symbol: str, max_news: int = 10) -> List[Dict]:
        """دریافت اخبار برای یک نماد"""
        if not self.api_key:
            return []
            
        crypto_name = symbol.split('/')[0]
        
        params = {
            'q': f'{crypto_name} cryptocurrency',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(max_news, 20),
            'apiKey': self.api_key
        }
        
        try:
            # اعمال rate limit
            self.rate_limiter.wait_if_needed('NewsAPI')
            
            response = safe_request(self.base_url, params=params, api_name='NewsAPI')
            data = response.json()
            
            if data.get('status') != 'ok':
                error_msg = data.get('message', 'Unknown NewsAPI error')
                logging.warning(f"NewsAPI خطا: {error_msg}")
                return []
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'timestamp': article.get('publishedAt', ''),
                    'symbol': symbol,
                    'title': article.get('title', ''),
                    'content': article.get('content', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'url': article.get('url', ''),
                    'language': 'en',
                    'image': article.get('urlToImage', ''),
                    'api_source': 'NewsAPI'
                })
            
            logging.info(f"📰 NewsAPI: {len(articles)} اخبار برای {symbol}")
            return articles
            
        except Exception as e:
            logging.error(f"خطا در NewsAPI برای {symbol}: {e}")
            return []

class CoinGeckoNewsFetcher:
    """استخراج اخبار از CoinGecko - رایگان و نامحدود (با rate limiting بهبود یافته)"""
    
    def __init__(self, rate_limiter: UnifiedRateLimiter):
        self.rate_limiter = rate_limiter
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # === Circuit breaker برای مدیریت خطاهای متوالی ===
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.circuit_open = False
        self.circuit_reset_time = None
        
        # نقشه تبدیل نماد به coin_id
        self.symbol_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
            'XRP': 'ripple', 'SOL': 'solana', 'ADA': 'cardano',
            'DOGE': 'dogecoin', 'MATIC': 'polygon', 'LTC': 'litecoin',
            'DOT': 'polkadot', 'AVAX': 'avalanche-2', 'LINK': 'chainlink',
            'BCH': 'bitcoin-cash', 'UNI': 'uniswap', 'ATOM': 'cosmos',
            'FIL': 'filecoin', 'VET': 'vechain', 'ICP': 'internet-computer',
            'TRX': 'tron', 'ETC': 'ethereum-classic', 'NEAR': 'near',
            'FTM': 'fantom', 'SAND': 'the-sandbox', 'MANA': 'decentraland',
            'SHIB': 'shiba-inu', 'OP': 'optimism', 'ARB': 'arbitrum',
            'APT': 'aptos', 'RNDR': 'render-token', 'GRT': 'the-graph'
        }
    
    def is_circuit_open(self) -> bool:
        """بررسی آیا circuit breaker باز است"""
        if not self.circuit_open:
            return False
        
        # اگر ۱۰ دقیقه گذشته، circuit را ریست کن
        if self.circuit_reset_time and time.time() - self.circuit_reset_time > 600:
            self.circuit_open = False
            self.consecutive_errors = 0
            self.circuit_reset_time = None
            logging.info("🔄 CoinGecko circuit breaker ریست شد")
            return False
        
        return True
    
    def record_error(self):
        """ثبت خطا و مدیریت circuit breaker"""
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.circuit_open = True
            self.circuit_reset_time = time.time()
            logging.warning(f"⚠️ CoinGecko circuit breaker فعال شد - ۱۰ دقیقه انتظار")
    
    def record_success(self):
        """ثبت موفقیت و ریست خطاها"""
        self.consecutive_errors = 0
        if self.circuit_open:
            self.circuit_open = False
            self.circuit_reset_time = None
            logging.info("✅ CoinGecko circuit breaker ریست شد")
    
    def get_coin_id(self, symbol: str) -> str:
        """تبدیل نماد به coin_id کوین‌گکو"""
        crypto_name = symbol.split('/')[0].upper()
        return self.symbol_map.get(crypto_name, crypto_name.lower())
    
    def fetch_crypto_news(self, symbol: str, max_news: int = 10) -> List[Dict]:
        """دریافت اخبار برای یک نماد"""
        
        # بررسی circuit breaker
        if self.is_circuit_open():
            logging.warning(f"🚫 CoinGecko circuit breaker فعال - رد کردن {symbol}")
            return []
        
        try:
            # اعمال rate limit با تاخیر بیشتر
            self.rate_limiter.wait_if_needed('CoinGecko')
            
            # تاخیر اضافی برای CoinGecko (3 ثانیه)
            time.sleep(3.0)
            
            # استفاده از trending news (بیشتر در دسترس)
            url = f"{self.base_url}/news"
            response = safe_request(url, api_name='CoinGecko', max_retries=2)
            data = response.json()
            
            # ثبت موفقیت
            self.record_success()
            
            articles = []
            news_items = data.get('data', [])
            crypto_name = symbol.split('/')[0].lower()
            
            # فیلتر کردن اخبار مرتبط با نماد
            relevant_count = 0
            for item in news_items:
                title = item.get('title', '').lower()
                description = item.get('description', '').lower()
                
                # اگر نام ارز در عنوان یا توضیحات باشد، یا تعداد کم باشد
                if (crypto_name in title or crypto_name in description or 
                    'crypto' in title or 'bitcoin' in title or relevant_count < 3):
                    
                    articles.append({
                        'timestamp': item.get('updated_at', ''),
                        'symbol': symbol,
                        'title': item.get('title', ''),
                        'content': item.get('description', ''),
                        'description': item.get('description', ''),
                        'source': item.get('news_site', 'CoinGecko'),
                        'url': item.get('url', ''),
                        'language': 'en',
                        'image': item.get('thumb_2x', ''),
                        'api_source': 'CoinGecko'
                    })
                    
                    relevant_count += 1
                    if relevant_count >= max_news:
                        break
            
            logging.info(f"🦎 CoinGecko: {len(articles)} اخبار برای {symbol}")
            return articles
            
        except requests.exceptions.RequestException as e:
            # ثبت خطا
            self.record_error()
            
            if "429" in str(e) or "Too Many Requests" in str(e):
                logging.error(f"🚫 CoinGecko rate limit: {symbol} - {e}")
            else:
                logging.error(f"خطا در CoinGecko برای {symbol}: {e}")
            return []
        except Exception as e:
            # ثبت خطا
            self.record_error()
            logging.error(f"خطا در CoinGecko برای {symbol}: {e}")
            return []

class RSSNewsFetcher:
    """استخراج اخبار از RSS feeds - رایگان و نامحدود"""
    
    def __init__(self, rate_limiter: UnifiedRateLimiter):
        self.rate_limiter = rate_limiter
        
        # فیدهای RSS معتبر کریپتو
        self.rss_feeds = {
            'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'CoinTelegraph': 'https://cointelegraph.com/rss',
            'Decrypt': 'https://decrypt.co/feed',
            'CryptoNews': 'https://cryptonews.com/news/feed'
        }
        
        # کش برای RSS feeds
        self._feed_cache = {}
        self._last_fetch = {}
        
    def fetch_rss_feed(self, feed_name: str, feed_url: str) -> List[Dict]:
        """دریافت و کش RSS feed"""
        current_time = time.time()
        
        # کش برای مدت زمان تنظیم شده
        cache_seconds = RSS_CACHE_MINUTES * 60
        if (feed_name in self._last_fetch and 
            current_time - self._last_fetch[feed_name] < cache_seconds):
            return self._feed_cache.get(feed_name, [])
        
        if not RSS_AVAILABLE:
            logging.warning(f"feedparser not available. Skipping {feed_name}")
            return []
        
        try:
            # اعمال rate limit
            self.rate_limiter.wait_if_needed('RSS')
            
            logging.info(f"📡 بارگذاری RSS: {feed_name}")
            feed = feedparser.parse(feed_url)
            
            articles = []
            max_articles = min(MAX_ARTICLES_PER_FEED, len(feed.entries))
            
            for entry in feed.entries[:max_articles]:
                articles.append({
                    'timestamp': entry.get('published', ''),
                    'title': entry.get('title', ''),
                    'content': entry.get('summary', ''),
                    'description': entry.get('summary', ''),
                    'source': feed_name,
                    'url': entry.get('link', ''),
                    'language': 'en',
                    'image': '',
                    'api_source': 'RSS'
                })
            
            # ذخیره در کش
            self._feed_cache[feed_name] = articles
            self._last_fetch[feed_name] = current_time
            
            logging.info(f"📡 {feed_name}: {len(articles)} خبر کش شد")
            return articles
            
        except Exception as e:
            logging.error(f"خطا در RSS {feed_name}: {e}")
            return []
    
    def fetch_crypto_news(self, symbol: str, max_news: int = 10) -> List[Dict]:
        """جستجو در اخبار RSS برای نماد مشخص"""
        crypto_name = symbol.split('/')[0].lower()
        relevant_articles = []
        
        # جمع‌آوری از همه RSS feeds
        all_articles = []
        for feed_name, feed_url in self.rss_feeds.items():
            feed_articles = self.fetch_rss_feed(feed_name, feed_url)
            all_articles.extend(feed_articles)
        
        # فیلتر کردن اخبار مرتبط
        for article in all_articles:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            # جستجو برای نام ارز یا کلمات کلیدی کریپتو
            if (crypto_name in title or crypto_name in content or 
                'crypto' in title or 'bitcoin' in title or 'blockchain' in title):
                article['symbol'] = symbol
                relevant_articles.append(article)
                
                if len(relevant_articles) >= max_news:
                    break
        
        logging.info(f"📡 RSS: {len(relevant_articles)} اخبار مرتبط برای {symbol}")
        return relevant_articles

class MultiSourceNewsFetcher:
    """مدیریت موازی چندین منبع خبری (با timeout handling بهبود یافته)"""
    
    def __init__(self, rate_limiter: UnifiedRateLimiter):
        self.rate_limiter = rate_limiter
        self.sources = {}
        
        # اولیه‌سازی منابع فعال
        if GNEWS_ENABLED and GNEWS_API_KEY:
            # GNews با interface متفاوت
            self.sources['GNews'] = 'gnews_special'
            
        if NEWSAPI_ENABLED and NEWSAPI_KEY:
            self.sources['NewsAPI'] = NewsAPIFetcher(NEWSAPI_KEY, rate_limiter)
            
        if COINGECKO_ENABLED:
            self.sources['CoinGecko'] = CoinGeckoNewsFetcher(rate_limiter)
            
        if RSS_ENABLED and RSS_AVAILABLE:
            self.sources['RSS'] = RSSNewsFetcher(rate_limiter)
        
        logging.info(f"🔗 MultiSource تشکیل شد: {list(self.sources.keys())}")
    
    def fetch_from_single_source(self, source_name: str, fetcher, 
                                symbols: List[str], max_news: int) -> List[Dict]:
        """استخراج از یک منبع (با timeout و error handling بهتر)"""
        all_articles = []
        
        try:
            if source_name == 'GNews':
                # GNews با interface خاص
                df = fetch_crypto_news(GNEWS_API_KEY, symbols, max_news, self.rate_limiter)
                if not df.empty:
                    articles_dict = df.to_dict('records')
                    # اضافه کردن api_source
                    for article in articles_dict:
                        article['api_source'] = 'GNews'
                    all_articles = articles_dict
            else:
                # سایر منابع با محدودیت زمان برای هر symbol
                for i, symbol in enumerate(symbols):
                    try:
                        articles = fetcher.fetch_crypto_news(symbol, max_news)
                        all_articles.extend(articles)
                        
                        # محدودیت تعداد کل برای جلوگیری از حجم زیاد
                        if len(all_articles) > len(symbols) * max_news:
                            break
                            
                        # نمایش پیشرفت برای منابع آهسته
                        if source_name == 'CoinGecko' and (i + 1) % 3 == 0:
                            logging.info(f"🦎 CoinGecko پیشرفت: {i + 1}/{len(symbols)} نماد")
                            
                    except Exception as symbol_error:
                        logging.warning(f"خطا در {source_name} برای {symbol}: {symbol_error}")
                        continue
            
            logging.info(f"✅ {source_name}: {len(all_articles)} خبر کل")
            return all_articles
            
        except Exception as e:
            logging.error(f"❌ خطا در {source_name}: {e}")
            return []
    
    def fetch_parallel(self, symbols: List[str], max_news: int = 10) -> pd.DataFrame:
        """استخراج موازی از همه منابع (با timeout management بهتر)"""
        all_articles = []
        
        logging.info(f"🚀 شروع استخراج از {len(self.sources)} منبع...")
        
        if PARALLEL_FETCHING and CONCURRENT_AVAILABLE and len(self.sources) > 1:
            # استخراج موازی با timeout management بهتر
            with ThreadPoolExecutor(max_workers=min(4, len(self.sources))) as executor:
                # ارسال tasks
                futures = {}
                for source_name, fetcher in self.sources.items():
                    future = executor.submit(
                        self.fetch_from_single_source, 
                        source_name, fetcher, symbols, max_news
                    )
                    futures[future] = source_name
                
                # جمع‌آوری نتایج با timeout مرحله‌ای
                completed_sources = []
                
                try:
                    # timeout اولیه: 120 ثانیه برای منابع سریع
                    for future in as_completed(futures, timeout=120):
                        source_name = futures[future]
                        try:
                            articles = future.result(timeout=30)  # timeout per source
                            all_articles.extend(articles)
                            completed_sources.append(source_name)
                            logging.info(f"✅ {source_name} تکمیل شد")
                        except Exception as e:
                            logging.error(f"❌ {source_name} ناموفق: {e}")
                
                except Exception as timeout_error:
                    logging.warning(f"⏰ Timeout در parallel processing: {timeout_error}")
                    
                    # تلاش برای دریافت نتایج منابع باقیمانده
                    remaining_futures = [f for f in futures.keys() if futures[f] not in completed_sources]
                    
                    if remaining_futures:
                        logging.info(f"🔄 در حال دریافت نتایج {len(remaining_futures)} منبع باقیمانده...")
                        
                        for future in remaining_futures:
                            source_name = futures[future]
                            try:
                                if future.done():
                                    articles = future.result(timeout=10)
                                    all_articles.extend(articles)
                                    logging.info(f"✅ {source_name} (تاخیری) تکمیل شد")
                                else:
                                    logging.warning(f"⏰ {source_name} همچنان در حال اجرا - رد شد")
                                    future.cancel()
                            except Exception as e:
                                logging.error(f"❌ {source_name} (تاخیری) ناموفق: {e}")
        else:
            # استخراج متوالی
            for source_name, fetcher in self.sources.items():
                articles = self.fetch_from_single_source(source_name, fetcher, symbols, max_news)
                all_articles.extend(articles)
        
        if not all_articles:
            logging.warning("❌ هیچ خبری از هیچ منبعی دریافت نشد")
            return pd.DataFrame()
        
        # تبدیل به DataFrame
        df = pd.DataFrame(all_articles)
        
        # استانداردسازی timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # حذف duplicates بر اساس title
        if REMOVE_DUPLICATES:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['title'], keep='first')
            final_count = len(df)
            
            if initial_count > final_count:
                logging.info(f"🧹 حذف {initial_count - final_count} خبر تکراری")
        
        # افزودن sentiment_score
        analyzer = SentimentIntensityAnalyzer()
        
        def analyze_sentiment(row):
            text = f"{row.get('title', '')} {row.get('description', '')}"
            try:
                return analyzer.polarity_scores(text)['compound']
            except:
                return 0
        
        df['sentiment_score'] = df.apply(analyze_sentiment, axis=1)
        
        # مرتب‌سازی بر اساس timestamp
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        logging.info(f"🎉 مجموع نهایی: {len(df)} خبر منحصر از {len(self.sources)} منبع")
        
        # آمار به تفکیک منبع
        if 'api_source' in df.columns:
            source_stats = df['api_source'].value_counts()
            for source, count in source_stats.items():
                logging.info(f"   📊 {source}: {count} خبر")
        
        return df

# --- توابع استخراج اخبار اصلی (بدون تغییر GNews) ---
def fetch_crypto_news(api_key: str, symbols: List[str], max_news: int = 10, 
                     rate_limiter: UnifiedRateLimiter = None) -> pd.DataFrame:
    """
    دریافت اخبار مرتبط با ارزهای دیجیتال از GNews API
    توجه: فقط به زبان انگلیسی برای کاهش مصرف API
    """
    logging.info("شروع جمع‌آوری داده‌های خبری GNews (فقط انگلیسی)...")
    all_articles = []
    base_url = "https://gnews.io/api/v4/search"
    
    total_requests = len(symbols)  # فقط یک زبان
    current_request = 0
    skipped_due_to_limit = 0
    
    for symbol in symbols:
        # استخراج نام ارز از جفت ارز
        crypto_name = symbol.split('/')[0]
        
        current_request += 1
        
        # بررسی محدودیت قبل از درخواست
        if rate_limiter:
            if rate_limiter.request_counters['GNews']['daily'] >= GNEWS_DAILY_LIMIT:
                skipped_due_to_limit += 1
                logging.warning(f"⏭️ رد شدن {symbol} به دلیل محدودیت روزانه")
                continue
        
        query = f"{crypto_name} cryptocurrency"
        
        params = {
            'q': query,
            'lang': 'en',  # فقط انگلیسی
            'max': max_news,
            'apikey': api_key,
            'sortby': 'publishedAt'  # جدیدترین اخبار
        }
        
        try:
            logging.info(f"[{current_request}/{total_requests}] دریافت اخبار GNews {symbol}...")
            
            # اعمال rate limit
            if rate_limiter:
                rate_limiter.wait_if_needed('GNews')
            
            response = safe_request(base_url, params=params, api_name='GNews')
            data = response.json()
            
            if 'articles' not in data:
                logging.warning(f"پاسخ غیرمنتظره GNews برای {symbol}: {data}")
                continue
            
            articles = data.get('articles', [])
            
            for article in articles:
                all_articles.append({
                    'timestamp': article.get('publishedAt', ''),
                    'symbol': symbol,
                    'title': article.get('title', ''),
                    'content': article.get('content', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'language': 'en',
                    'image': article.get('image', '')
                })
            
            logging.info(f"✅ GNews: تعداد {len(articles)} خبر برای {symbol} دریافت شد.")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"خطا در دریافت اخبار GNews برای {symbol}: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"خطا در پردازش JSON برای {symbol}: {e}")
        except Exception as e:
            logging.error(f"خطای غیرمنتظره برای {symbol}: {e}")
    
    if skipped_due_to_limit > 0:
        logging.warning(f"⚠️ تعداد {skipped_due_to_limit} درخواست به دلیل محدودیت نادیده گرفته شد.")
    
    if not all_articles:
        logging.warning("هیچ خبری از GNews دریافت نشد.")
        return pd.DataFrame()
    
    # تبدیل به DataFrame
    df = pd.DataFrame(all_articles)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    logging.info(f"📊 مجموع اخبار GNews دریافت شده: {len(df)}")
    
    return df

# --- توابع دریافت لیست جفت‌ارزها (بدون تغییر) ---
def fetch_all_tradable_pairs_from_exchange(exchange_name: str, quote_currency="USDT"):
    """دریافت لیست تمام جفت‌ارزهای قابل معامله از صرافی انتخابی"""
    if exchange_name == 'CryptoCompare':
        return fetch_all_tradable_pairs_cryptocompare(quote_currency)
    elif exchange_name == 'Binance':
        return fetch_all_tradable_pairs_binance(quote_currency)
    elif exchange_name == 'Kraken':
        return fetch_all_tradable_pairs_kraken(quote_currency)
    else:
        logging.error(f"دریافت لیست جفت‌ارز برای صرافی '{exchange_name}' پیاده‌سازی نشده است.")
        return []

def fetch_all_tradable_pairs_cryptocompare(quote_currency="USDT"):
    """دریافت لیست از CryptoCompare"""
    logging.info(f"[CryptoCompare] در حال دریافت لیست تمام جفت‌ارزها با مرجع {quote_currency}...")
    try:
        url = "https://min-api.cryptocompare.com/data/all/coinlist"
        params = {}
        if CRYPTOCOMPARE_API_KEY:
            params["api_key"] = CRYPTOCOMPARE_API_KEY
            
        response = safe_request(url, params=params, api_name="CryptoCompare")
        data = response.json()['Data']
        
        pairs = [f"{symbol_data['Symbol']}/{quote_currency}" for symbol, symbol_data in data.items() 
                 if symbol_data.get('IsTrading', False) and symbol.isalpha()]
        
        # فیلتر کردن نمادهای نامعتبر
        valid_pairs = []
        for pair in pairs:
            if len(pair.split('/')[0]) <= 10:  # حذف نمادهای خیلی طولانی
                valid_pairs.append(pair)
        
        logging.info(f"[CryptoCompare] تعداد {len(valid_pairs)} جفت ارز معتبر یافت شد.")
        return valid_pairs[:100]  # محدود کردن به 100 جفت برتر
    except Exception as e:
        logging.error(f"[CryptoCompare] خطا در دریافت لیست جفت‌ارزها: {e}")
        return []

def fetch_all_tradable_pairs_binance(quote_currency="USDT"):
    """دریافت لیست از Binance"""
    logging.info(f"[Binance] در حال دریافت لیست تمام جفت‌ارزها با مرجع {quote_currency}...")
    try:
        response = safe_request("https://api.binance.com/api/v3/exchangeInfo", api_name="Binance")
        data = response.json()
        
        pairs = []
        for symbol_info in data['symbols']:
            if (symbol_info['status'] == 'TRADING' and 
                symbol_info['quoteAsset'] == quote_currency):
                pair = f"{symbol_info['baseAsset']}/{symbol_info['quoteAsset']}"
                pairs.append(pair)
        
        logging.info(f"[Binance] تعداد {len(pairs)} جفت ارز معتبر یافت شد.")
        return pairs
    except Exception as e:
        logging.error(f"[Binance] خطا در دریافت لیست جفت‌ارزها: {e}")
        return []

def fetch_all_tradable_pairs_kraken(quote_currency="USD"):
    """دریافت لیست از Kraken"""
    logging.info(f"[Kraken] در حال دریافت لیست تمام جفت‌ارزها با مرجع {quote_currency}...")
    try:
        response = safe_request("https://api.kraken.com/0/public/AssetPairs", api_name="Kraken")
        data = response.json()
        
        if 'error' in data and data['error']:
            logging.error(f"[Kraken] خطای API: {data['error']}")
            return []
        
        pairs = []
        for pair_name, pair_info in data['result'].items():
            if quote_currency.upper() in pair_info.get('quote', '').upper():
                base = pair_info.get('base', '')
                quote = pair_info.get('quote', '')
                if base and quote:
                    pairs.append(f"{base}/{quote}")
        
        logging.info(f"[Kraken] تعداد {len(pairs)} جفت ارز معتبر یافت شد.")
        return pairs
    except Exception as e:
        logging.error(f"[Kraken] خطا در دریافت لیست جفت‌ارزها: {e}")
        return []

# --- کلاس اصلی Unified Data Fetcher (بهبود یافته) ---
class UnifiedDataFetcher:
    """کلاس اصلی برای استخراج یکپارچه داده‌های قیمت و اخبار"""
    
    def __init__(self):
        self.state_manager = UnifiedStateManager()
        self.rate_limiter = UnifiedRateLimiter(self.state_manager)
        
        self.exchange_functions = {
            'CryptoCompare': fetch_from_cryptocompare,
            'Binance': fetch_from_binance,
            'Kraken': fetch_from_kraken,
        }
        
        # ایجاد sentiment analyzer برای پردازش مقدماتی اخبار
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logging.info("🚀 Enhanced Unified Data Fetcher آماده شد")
    
    def fetch_price_data(self, symbol: str, timeframe: str, limit: int, 
                        exchange_name: str, session_id: str) -> bool:
        """استخراج داده قیمت با مدیریت state"""
        
        # بررسی آیتم شکست خورده
        if self.state_manager.is_failed_item('price', symbol, exchange_name):
            logging.info(f"⏭️ رد شدن جفت ارز شکست خورده: {symbol}")
            return True
        
        # بررسی و انتظار rate limit
        if not self.rate_limiter.wait_if_needed(exchange_name):
            logging.error(f"❌ Rate limit رسیده برای {exchange_name}")
            return False
        
        try:
            # بروزرسانی وضعیت به در حال پردازش
            self.state_manager.update_price_progress(
                session_id, exchange_name, symbol, timeframe, 'processing'
            )
            
            # استخراج داده
            fetch_function = self.exchange_functions[exchange_name]
            df = fetch_function(symbol, timeframe, limit)
            
            if df.empty:
                error_msg = f"Empty data for {symbol}"
                logging.warning(f"⚠️ {error_msg}")
                self.state_manager.add_failed_item('price', symbol, error_msg, exchange_name)
                self.state_manager.update_price_progress(
                    session_id, exchange_name, symbol, timeframe, 'failed',
                    error_message=error_msg
                )
                return True
            
            # ذخیره داده
            filename = self.save_price_data(df, exchange_name, session_id)
            
            # بروزرسانی وضعیت به تکمیل شده
            self.state_manager.update_price_progress(
                session_id, exchange_name, symbol, timeframe, 'completed',
                file_path=filename, records_count=len(df)
            )
            
            logging.info(f"✅ قیمت موفق: {symbol}|{timeframe} - {len(df)} سطر")
            return True
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            logging.error(f"❌ خطا در {symbol}|{timeframe}: {error_msg}")
            
            # اگر خطای API است، به failed items اضافه کن
            if any(x in str(e).lower() for x in ['market does not exist', 'unknown asset pair', 'invalid symbol']):
                self.state_manager.add_failed_item('price', symbol, error_msg, exchange_name)
            
            self.state_manager.update_price_progress(
                session_id, exchange_name, symbol, timeframe, 'failed',
                error_message=error_msg
            )
            
            return True
    
    def fetch_news_data(self, symbols: List[str], max_news: int, session_id: str) -> bool:
        """استخراج اخبار با مدیریت state - چندمنبعه بهبود یافته"""
        
        logging.info("\n--- شروع استخراج اخبار چندمنبعه بهبود یافته ---")
        logging.info(f"تعداد نمادها: {len(symbols)}")
        logging.info(f"تعداد اخبار برای هر نماد: {max_news}")
        
        # ایجاد multi-source fetcher
        multi_fetcher = MultiSourceNewsFetcher(self.rate_limiter)
        
        if not multi_fetcher.sources:
            logging.error("❌ هیچ منبع خبری فعالی یافت نشد")
            return False
        
        # استخراج موازی/متوالی
        df_news = multi_fetcher.fetch_parallel(symbols, max_news)
        
        if df_news.empty:
            logging.warning("❌ هیچ خبری دریافت نشد")
            return False
        
        # ذخیره بر اساس نماد (مشابه کد قبلی)
        for symbol, group in df_news.groupby('symbol'):
            try:
                # بروزرسانی وضعیت
                self.state_manager.update_news_progress(
                    session_id, symbol, 'en', 'processing'
                )
                
                # ذخیره داده
                filename = self.save_news_data(group, symbol, 'en', session_id)
                
                # بروزرسانی به تکمیل شده
                self.state_manager.update_news_progress(
                    session_id, symbol, 'en', 'completed',
                    file_path=filename, news_count=len(group)
                )
                
                logging.info(f"✅ اخبار موفق: {symbol} - {len(group)} خبر")
                
            except Exception as e:
                error_msg = f"Error saving news for {symbol}: {str(e)}"
                logging.error(error_msg)
                self.state_manager.update_news_progress(
                    session_id, symbol, 'en', 'failed',
                    error_message=error_msg
                )
        
        # نمایش آمار کلی
        if 'api_source' in df_news.columns:
            total_by_source = df_news['api_source'].value_counts()
            logging.info("📊 خلاصه نهایی:")
            for source, count in total_by_source.items():
                logging.info(f"   {source}: {count} خبر")
        
        return True
    
    def analyze_sentiment(self, text: str) -> float:
        """تحلیل احساسات مقدماتی برای یک متن"""
        try:
            if not text or not isinstance(text, str):
                return 0
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound']
        except Exception:
            return 0
    
    def save_price_data(self, df: pd.DataFrame, exchange_name: str, session_id: str) -> str:
        """ذخیره داده قیمت"""
        if df.empty:
            return ""
        
        # استانداردسازی ستون‌ها
        required_columns = ['timestamp', 'symbol', 'timeframe', 'exchange', 'open', 'high', 'low', 'close', 'volume']
        df_final = df[required_columns].copy()
        
        # نام‌گذاری فایل
        symbol = df_final['symbol'].iloc[0]
        timeframe = df_final['timeframe'].iloc[0]
        
        symbol_sanitized = symbol.replace('/', '-')
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{exchange_name}_{symbol_sanitized}_{timeframe}_{timestamp_str}.csv"
        output_path = os.path.join(RAW_DATA_PATH, filename)        
        # ذخیره
        df_final.to_csv(output_path, index=False, float_format='%.8f')
        
        return filename
    
    def save_news_data(self, df: pd.DataFrame, symbol: str, language: str, session_id: str) -> str:
        """ذخیره داده اخبار"""
        if df.empty:
            return ""
        
        # نام‌گذاری فایل
        symbol_sanitized = symbol.replace('/', '-')
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"news_{symbol_sanitized}_{language}_{timestamp_str}.csv"
        output_path = os.path.join(RAW_DATA_PATH, filename)        
        # ذخیره
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return filename
    
    def run_price_extraction(self, exchange_name: str, symbols: List[str], 
                           timeframes: List[str], session_id: str) -> int:
        """اجرای استخراج قیمت برای همه نمادها و تایم‌فریم‌ها"""
        success_count = 0
        
        logging.info(f"\n--- شروع استخراج داده‌های قیمت از {exchange_name} ---")
        logging.info(f"تعداد نمادها: {len(symbols)}")
        logging.info(f"تایم‌فریم‌ها: {', '.join(timeframes)}")
        
        for symbol in symbols:
            for timeframe in timeframes:
                success = self.fetch_price_data(
                    symbol, timeframe, LIMIT, exchange_name, session_id
                )
                if success:
                    success_count += 1
                
                # نمایش آمار
                stats = self.rate_limiter.get_stats(exchange_name)
                if stats:
                    logging.info(f"📊 آمار {exchange_name}: {stats}")
        
        return success_count
    
    def run_news_extraction(self, symbols: List[str], max_news: int, session_id: str) -> bool:
        """اجرای استخراج اخبار چندمنبعه بهبود یافته"""
        success = self.fetch_news_data(symbols, max_news, session_id)
        
        # نمایش آمار تمام منابع
        for source_name in ['GNews', 'NewsAPI', 'CoinGecko', 'RSS']:
            stats = self.rate_limiter.get_stats(source_name)
            if stats:
                logging.info(f"📊 آمار {source_name}: {stats}")
        
        return success

# --- توابع کمکی UI (بدون تغییر) ---
def get_user_selection(options: list, title: str, allow_manual=False, allow_multi=False, allow_all=False):
    """منوی شماره‌گذاری شده بهبود یافته"""
    print(f"\n--- {title} ---")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    if allow_all: 
        print(f"{len(options)+1}. همه موارد بالا")
    if allow_manual: 
        print(f"{len(options)+2 if allow_all else len(options)+1}. ورود دستی")
    
    if allow_multi:
        prompt = "شماره موارد مورد نظر (با کاما جدا کنید): "
    else:
        prompt = "شماره مورد نظر: "
    
    choice_str = input(prompt).strip()
    
    if not choice_str:
        logging.error("ورودی خالی.")
        return []
    
    try:
        if allow_all and choice_str == str(len(options)+1):
            return options
            
        manual_entry_num = len(options)+2 if allow_all else len(options)+1
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

def get_exchange_selection():
    """انتخاب صرافی توسط کاربر"""
    exchanges = ['Binance', 'CryptoCompare', 'Kraken']
    
    print("\n🏦 انتخاب صرافی:")
    print("💡 توصیه: Binance برای سرعت و عدم محدودیت")
    
    exchange_list = get_user_selection(exchanges, "انتخاب صرافی", allow_multi=False)
    return exchange_list[0] if exchange_list else None

# --- تابع اصلی (اصلاح شده کامل) ---
def main():
    """تابع اصلی منو محور - مطابق با fetch_historical_data_01.py اصلی"""
    logging.info("🚀 شروع اسکریپت Enhanced Unified Data Fetcher")
    
    # اولیه‌سازی
    fetcher = UnifiedDataFetcher()
    
    # نمایش خلاصه تنظیمات
    print("\n" + "="*80)
    print("🔐 تنظیمات Enhanced Unified Data Fetcher:")
    print(f"📊 CryptoCompare: حداکثر {DAILY_LIMIT}/روز، {HOURLY_LIMIT}/ساعت")
    print(f"📰 GNews: حداکثر {GNEWS_DAILY_LIMIT}/روز، {GNEWS_HOURLY_LIMIT}/ساعت")
    
    # === نمایش منابع جدید ===
    print("=== منابع خبری جدید ===")
    if NEWSAPI_ENABLED and NEWSAPI_KEY:
        print(f"📰 NewsAPI: حداکثر {NEWSAPI_DAILY_LIMIT}/روز - فعال")
    else:
        print("📰 NewsAPI: غیرفعال")
        
    if COINGECKO_ENABLED:
        print("🦎 CoinGecko: نامحدود - فعال")
    else:
        print("🦎 CoinGecko: غیرفعال")
        
    if RSS_ENABLED and RSS_AVAILABLE:
        print("📡 RSS Feeds: نامحدود - فعال")
    else:
        print("📡 RSS Feeds: غیرفعال")
    
    print(f"⚡ Binance: بدون محدودیت، delay {BINANCE_DELAY}s")
    print(f"🔄 Kraken: delay {KRAKEN_DELAY}s")
    print("💾 State Management: یکپارچه برای قیمت و اخبار")
    print("🚀 اجرای موازی: " + ("فعال" if PARALLEL_FETCHING and CONCURRENT_AVAILABLE else "غیرفعال"))
    print("="*80)
    
    # حلقه اصلی برای نگه‌داشتن برنامه
    while True:
        print("\n" + "="*80)
        print("   منوی اصلی Enhanced Unified Data Fetcher")
        print("="*80)
        print("1. استخراج سفارشی (انتخاب از لیست)")
        print("2. دریافت تمام جفت ارزها (Production Mode)")
        print("3. تکمیل داده‌های تاریخی (Backfill)")
        print("4. نمایش آمار و وضعیت")
        print("5. مدیریت State (Database)")
        print("6. خروج")
        
        main_choice = input("\nلطفاً شماره عملیات مورد نظر را وارد کنید: ")
        
        if main_choice == '1':
            # استخراج سفارشی
            print("\n🎯 نوع داده برای استخراج:")
            print("1. فقط قیمت")
            print("2. فقط اخبار")
            print("3. هر دو (قیمت و اخبار)")
            
            data_type = input("انتخاب کنید: ")
            
            include_price = data_type in ['1', '3']
            include_news = data_type in ['2', '3']
            
            # اگر اخبار انتخاب شده، تعداد خبر را بپرس
            max_news = 10  # پیش‌فرض
            if include_news:
                print("\n📰 تعداد اخبار برای هر نماد:")
                print("1. 5 خبر")
                print("2. 10 خبر (پیش‌فرض)")
                print("3. 20 خبر")
                
                news_choice = input("انتخاب کنید: ")
                news_counts = {'1': 5, '2': 10, '3': 20}
                max_news = news_counts.get(news_choice, 10)
            
            # انتخاب نمادها
            pairs = get_user_selection(COMMON_PAIRS, "انتخاب جفت ارز", 
                                     allow_manual=True, allow_multi=True, allow_all=True)
            if not pairs:
                input("Enter برای ادامه...")
                continue
            
            # ایجاد session
            session_id = fetcher.state_manager.create_unified_session(
                pairs, include_price, include_news
            )
            
            if include_price:
                # انتخاب صرافی
                exchange = get_exchange_selection()
                if not exchange:
                    input("Enter برای ادامه...")
                    continue
                
                # انتخاب تایم‌فریم
                timeframes = get_user_selection(COMMON_TIMEFRAMES, "انتخاب تایم فریم", 
                                              allow_multi=True, allow_all=True)
                if not timeframes:
                    input("Enter برای ادامه...")
                    continue
                
                # استخراج قیمت
                success_count = fetcher.run_price_extraction(
                    exchange, pairs, timeframes, session_id
                )
                
                logging.info(f"✅ قیمت: {success_count} درخواست موفق")
            
            if include_news:
                # استخراج اخبار چندمنبعه
                success = fetcher.run_news_extraction(pairs, max_news, session_id)
                
                if success:
                    logging.info("✅ اخبار چندمنبعه: استخراج موفق")
            
            # دریافت وضعیت نهایی
            final_status = fetcher.state_manager.get_session_status(session_id)
            
            logging.info("\n✅ استخراج تکمیل شد")
            logging.info(f"📊 خلاصه نتایج:")
            if include_price:
                logging.info(f"   قیمت: {final_status['price_progress']['completed']}/{final_status['price_progress']['total']}")
            if include_news:
                logging.info(f"   اخبار: {final_status['news_progress']['completed']}/{final_status['news_progress']['total']}")
            
            input("\nEnter برای بازگشت به منوی اصلی...")

        elif main_choice == '2':
            # دریافت تمام جفت ارزها
            print("\n🚨 حالت Production - دریافت تمام جفت‌ارزها")
            
            # انتخاب نوع داده
            print("\n🎯 نوع داده برای استخراج:")
            print("1. فقط قیمت")
            print("2. فقط اخبار")
            print("3. هر دو (قیمت و اخبار)")
            
            data_type = input("انتخاب کنید: ")
            
            include_price = data_type in ['1', '3']
            include_news = data_type in ['2', '3']
            
            # اگر اخبار انتخاب شده، تعداد خبر را بپرس
            max_news = 10  # پیش‌فرض
            if include_news:
                print("\n📰 تعداد اخبار برای هر نماد:")
                print("1. 5 خبر")
                print("2. 10 خبر (پیش‌فرض)")
                print("3. 20 خبر")
                
                news_choice = input("انتخاب کنید: ")
                news_counts = {'1': 5, '2': 10, '3': 20}
                max_news = news_counts.get(news_choice, 10)
            
            # انتخاب صرافی
            exchange = get_exchange_selection()
            if not exchange:
                input("Enter برای ادامه...")
                continue
            
            # دریافت لیست جفت ارزها
            quote_currency = "USDT" if exchange != 'Kraken' else "USD"
            all_pairs = fetch_all_tradable_pairs_from_exchange(exchange, quote_currency)
            
            if not all_pairs:
                logging.error("❌ خطا در دریافت لیست جفت‌ارزها")
                input("Enter برای ادامه...")
                continue
            
            print(f"\n✅ تعداد {len(all_pairs)} جفت ارز یافت شد")
            
            # محاسبه تخمینی
            if include_price:
                price_requests = len(all_pairs) * len(COMMON_TIMEFRAMES)
                print(f"📊 تخمین درخواست‌های قیمت: {price_requests}")
            
            if include_news:
                news_requests = len(all_pairs)
                remaining_daily = max(0, GNEWS_DAILY_LIMIT - fetcher.rate_limiter.request_counters['GNews']['daily'])
                print(f"📰 تخمین درخواست‌های اخبار: {news_requests}")
                print(f"📰 باقیمانده روزانه GNews: {remaining_daily}")
                
                # نمایش منابع فعال
                active_sources = []
                if GNEWS_ENABLED and GNEWS_API_KEY:
                    active_sources.append("GNews")
                if NEWSAPI_ENABLED and NEWSAPI_KEY:
                    active_sources.append("NewsAPI")
                if COINGECKO_ENABLED:
                    active_sources.append("CoinGecko")
                if RSS_ENABLED and RSS_AVAILABLE:
                    active_sources.append("RSS")
                
                print(f"📡 منابع خبری فعال: {', '.join(active_sources)}")
            
            confirm = input("\nآیا می‌خواهید ادامه دهید؟ (y/n): ")
            if confirm.lower() != 'y':
                input("Enter برای ادامه...")
                continue
            
            # ایجاد session
            session_id = fetcher.state_manager.create_unified_session(
                all_pairs, include_price, include_news
            )
            
            if include_price:
                # استخراج قیمت برای همه تایم‌فریم‌ها
                success_count = fetcher.run_price_extraction(
                    exchange, all_pairs, COMMON_TIMEFRAMES, session_id
                )
                logging.info(f"✅ قیمت: {success_count} درخواست موفق")
            
            if include_news:
                # استخراج اخبار چندمنبعه
                success = fetcher.run_news_extraction(all_pairs, max_news, session_id)
                if success:
                    logging.info("✅ اخبار چندمنبعه: استخراج موفق")
            
            input("\nEnter برای بازگشت به منوی اصلی...")
        
        elif main_choice == '3':
            # تکمیل داده‌های تاریخی (Backfill)
            print("\n🔄 تکمیل داده‌های تاریخی (Backfill)")
            print("="*50)
            print("1. تکمیل تایم‌فریم‌های از دست رفته")
            print("2. تکمیل نمادهای از دست رفته")
            print("3. تکمیل آیتم‌های شکست خورده")
            print("4. تکمیل اخبار برای نمادهای موجود")
            print("5. بازگشت به منوی اصلی")
            
            backfill_choice = input("انتخاب کنید: ")
            
            if backfill_choice == '1':
                # تکمیل تایم‌فریم‌های از دست رفته
                print("\n📊 تکمیل تایم‌فریم‌های از دست رفته")
                
                # دریافت لیست فایل‌های موجود
                price_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv') and not f.startswith('news_')]
                
                if not price_files:
                    print("❌ هیچ فایل قیمت موجودی یافت نشد")
                    input("Enter برای ادامه...")
                    continue
                
                # تحلیل نمادها و تایم‌فریم‌های موجود
                existing_combinations = set()
                symbols_found = set()
                
                for filename in price_files:
                    try:
                        parts = filename.replace('.csv', '').split('_')
                        if len(parts) >= 3:
                            exchange = parts[0]
                            symbol = parts[1].replace('-', '/')
                            timeframe = parts[2]
                            existing_combinations.add((exchange, symbol, timeframe))
                            symbols_found.add(symbol)
                    except:
                        continue
                
                print(f"✅ یافت شد: {len(symbols_found)} نماد در {len(existing_combinations)} ترکیب")
                
                # انتخاب صرافی
                exchange = get_exchange_selection()
                if not exchange:
                    input("Enter برای ادامه...")
                    continue
                
                # پیدا کردن تایم‌فریم‌های از دست رفته
                missing_combinations = []
                for symbol in symbols_found:
                    for timeframe in COMMON_TIMEFRAMES:
                        if (exchange, symbol, timeframe) not in existing_combinations:
                            missing_combinations.append((symbol, timeframe))
                
                if not missing_combinations:
                    print("✅ همه تایم‌فریم‌ها برای نمادهای موجود کامل است")
                    input("Enter برای ادامه...")
                    continue
                
                print(f"📋 تعداد {len(missing_combinations)} ترکیب از دست رفته یافت شد")
                
                # نمایش نمونه
                print("\nنمونه موارد از دست رفته:")
                for i, (symbol, timeframe) in enumerate(missing_combinations[:10]):
                    print(f"  - {symbol} | {timeframe}")
                if len(missing_combinations) > 10:
                    print(f"  ... و {len(missing_combinations) - 10} مورد دیگر")
                
                confirm = input("\nآیا می‌خواهید این موارد را تکمیل کنید؟ (y/n): ")
                if confirm.lower() != 'y':
                    input("Enter برای ادامه...")
                    continue
                
                # ایجاد session برای backfill
                unique_symbols = list(set([combo[0] for combo in missing_combinations]))
                session_id = fetcher.state_manager.create_unified_session(unique_symbols, True, False)
                
                # اجرای backfill
                success_count = 0
                total_items = len(missing_combinations)
                
                for i, (symbol, timeframe) in enumerate(missing_combinations):
                    success = fetcher.fetch_price_data(symbol, timeframe, LIMIT, exchange, session_id)
                    if success:
                        success_count += 1
                    
                    # نمایش پیشرفت
                    progress = ((i + 1) / total_items) * 100
                    print(f"⚡ پیشرفت: {progress:.1f}% ({i + 1}/{total_items}) - موفق: {success_count}")
                
                print(f"\n✅ Backfill تکمیل شد: {success_count} مورد موفق از {total_items}")
                
            elif backfill_choice == '2':
                # تکمیل نمادهای از دست رفته
                print("\n💰 تکمیل نمادهای از دست رفته")
                
                # دریافت نمادهای موجود
                price_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv') and not f.startswith('news_')]
                existing_symbols = set()
                
                for filename in price_files:
                    try:
                        parts = filename.replace('.csv', '').split('_')
                        if len(parts) >= 2:
                            symbol = parts[1].replace('-', '/')
                            existing_symbols.add(symbol)
                    except:
                        continue
                
                # پیدا کردن نمادهای از دست رفته
                missing_symbols = []
                for symbol in COMMON_PAIRS:
                    if symbol not in existing_symbols:
                        missing_symbols.append(symbol)
                
                if not missing_symbols:
                    print("✅ همه نمادهای مهم موجود است")
                    input("Enter برای ادامه...")
                    continue
                
                print(f"📋 تعداد {len(missing_symbols)} نماد از دست رفته:")
                for symbol in missing_symbols:
                    print(f"  - {symbol}")
                
                confirm = input("\nآیا می‌خواهید این نمادها را اضافه کنید؟ (y/n): ")
                if confirm.lower() != 'y':
                    input("Enter برای ادامه...")
                    continue
                
                # انتخاب صرافی
                exchange = get_exchange_selection()
                if not exchange:
                    input("Enter برای ادامه...")
                    continue
                
                # ایجاد session
                session_id = fetcher.state_manager.create_unified_session(missing_symbols, True, False)
                
                # اجرای استخراج
                success_count = fetcher.run_price_extraction(exchange, missing_symbols, COMMON_TIMEFRAMES, session_id)
                print(f"\n✅ تکمیل نمادها: {success_count} درخواست موفق")
                
            elif backfill_choice == '3':
                # تکمیل آیتم‌های شکست خورده
                print("\n🔄 تکمیل آیتم‌های شکست خورده")
                
                # دریافت لیست آیتم‌های شکست خورده
                with sqlite3.connect(fetcher.state_manager.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT item_type, exchange, symbol, error_message 
                        FROM failed_items 
                        WHERE item_type = 'price'
                        ORDER BY created_at DESC
                    ''')
                    failed_items = cursor.fetchall()
                
                if not failed_items:
                    print("✅ هیچ آیتم شکست خورده‌ای یافت نشد")
                    input("Enter برای ادامه...")
                    continue
                
                print(f"📋 تعداد {len(failed_items)} آیتم شکست خورده:")
                for item in failed_items[:10]:
                    print(f"  - {item[2]} ({item[1]}) | {item[3][:50]}...")
                
                if len(failed_items) > 10:
                    print(f"  ... و {len(failed_items) - 10} مورد دیگر")
                
                confirm = input("\nآیا می‌خواهید دوباره تلاش کنید؟ (y/n): ")
                if confirm.lower() != 'y':
                    input("Enter برای ادامه...")
                    continue
                
                # پاک کردن failed items برای تلاش مجدد
                with sqlite3.connect(fetcher.state_manager.db_path) as conn:
                    conn.execute("DELETE FROM failed_items WHERE item_type = 'price'")
                
                print("🧹 لیست آیتم‌های شکست خورده پاک شد")
                
                # تلاش مجدد
                failed_symbols = list(set([item[2] for item in failed_items]))
                session_id = fetcher.state_manager.create_unified_session(failed_symbols, True, False)
                
                # انتخاب صرافی (ممکن است تغییر صرافی کمک کند)
                exchange = get_exchange_selection()
                if not exchange:
                    input("Enter برای ادامه...")
                    continue
                
                success_count = fetcher.run_price_extraction(exchange, failed_symbols, COMMON_TIMEFRAMES, session_id)
                print(f"\n✅ تلاش مجدد: {success_count} درخواست موفق")
                
            elif backfill_choice == '4':
                # تکمیل اخبار برای نمادهای موجود
                print("\n📰 تکمیل اخبار برای نمادهای موجود")
                
                # دریافت نمادهای موجود از فایل‌های قیمت
                price_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv') and not f.startswith('news_')]
                symbols_with_price = set()
                
                for filename in price_files:
                    try:
                        parts = filename.replace('.csv', '').split('_')
                        if len(parts) >= 2:
                            symbol = parts[1].replace('-', '/')
                            symbols_with_price.add(symbol)
                    except:
                        continue
                
                # دریافت نمادهایی که اخبار دارند
                news_files = [f for f in os.listdir(RAW_DATA_PATH) if f.startswith('news_') and f.endswith('.csv')]
                symbols_with_news = set()
                
                for filename in news_files:
                    try:
                        parts = filename.replace('.csv', '').split('_')
                        if len(parts) >= 2:
                            symbol = parts[1].replace('-', '/')
                            symbols_with_news.add(symbol)
                    except:
                        continue
                
                # پیدا کردن نمادهایی که قیمت دارند ولی اخبار ندارند
                symbols_need_news = symbols_with_price - symbols_with_news
                
                if not symbols_need_news:
                    print("✅ همه نمادهای موجود اخبار دارند")
                    input("Enter برای ادامه...")
                    continue
                
                print(f"📋 تعداد {len(symbols_need_news)} نماد نیاز به اخبار:")
                symbols_list = list(symbols_need_news)
                for symbol in symbols_list[:10]:
                    print(f"  - {symbol}")
                if len(symbols_need_news) > 10:
                    print(f"  ... و {len(symbols_need_news) - 10} مورد دیگر")
                
                # انتخاب تعداد اخبار
                print("\n📰 تعداد اخبار برای هر نماد:")
                print("1. 5 خبر")
                print("2. 10 خبر (پیش‌فرض)")
                print("3. 20 خبر")
                
                news_choice = input("انتخاب کنید: ")
                news_counts = {'1': 5, '2': 10, '3': 20}
                max_news = news_counts.get(news_choice, 10)
                
                # بررسی محدودیت GNews
                remaining_daily = max(0, GNEWS_DAILY_LIMIT - fetcher.rate_limiter.request_counters['GNews']['daily'])
                
                if len(symbols_need_news) > remaining_daily:
                    print(f"⚠️ تعداد نمادها ({len(symbols_need_news)}) بیشتر از محدودیت روزانه باقیمانده ({remaining_daily}) است")
                    symbols_list = symbols_list[:remaining_daily]
                    print(f"📊 محدود شده به {len(symbols_list)} نماد اول")
                else:
                    symbols_list = list(symbols_need_news)
                
                # نمایش منابع فعال
                active_sources = []
                if GNEWS_ENABLED and GNEWS_API_KEY:
                    active_sources.append("GNews")
                if NEWSAPI_ENABLED and NEWSAPI_KEY:
                    active_sources.append("NewsAPI")
                if COINGECKO_ENABLED:
                    active_sources.append("CoinGecko")
                if RSS_ENABLED and RSS_AVAILABLE:
                    active_sources.append("RSS")
                
                print(f"📡 منابع خبری فعال: {', '.join(active_sources)}")
                
                confirm = input("\nآیا می‌خواهید اخبار را تکمیل کنید؟ (y/n): ")
                if confirm.lower() != 'y':
                    input("Enter برای ادامه...")
                    continue
                
                # ایجاد session برای اخبار
                session_id = fetcher.state_manager.create_unified_session(symbols_list, False, True)
                
                # استخراج اخبار چندمنبعه
                success = fetcher.run_news_extraction(symbols_list, max_news, session_id)
                
                if success:
                    print("✅ تکمیل اخبار چندمنبعه موفق بود")
                else:
                    print("❌ خطا در تکمیل اخبار")
                    
            elif backfill_choice == '5':
                pass  # بازگشت به منوی اصلی
            else:
                print("انتخاب نامعتبر.")
                
            if backfill_choice != '5':
                input("\nEnter برای بازگشت به منوی اصلی...")
        
        elif main_choice == '4':
            # نمایش آمار
            print("\n📊 آمار وضعیت:")
            
            # آمار منابع قیمت
            for api_name in ['CryptoCompare', 'Binance', 'Kraken']:
                stats = fetcher.rate_limiter.get_stats(api_name)
                if stats:
                    print(f"\n🔑 {api_name}:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
            
            # آمار منابع خبری
            print("\n📰 منابع خبری:")
            for api_name in ['GNews', 'NewsAPI', 'CoinGecko', 'RSS']:
                stats = fetcher.rate_limiter.get_stats(api_name)
                if stats:
                    print(f"\n📡 {api_name}:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                else:
                    # نمایش وضعیت فعال/غیرفعال
                    status = "غیرفعال"
                    if api_name == 'GNews' and GNEWS_ENABLED and GNEWS_API_KEY:
                        status = "فعال"
                    elif api_name == 'NewsAPI' and NEWSAPI_ENABLED and NEWSAPI_KEY:
                        status = "فعال"
                    elif api_name == 'CoinGecko' and COINGECKO_ENABLED:
                        status = "فعال"
                    elif api_name == 'RSS' and RSS_ENABLED and RSS_AVAILABLE:
                        status = "فعال"
                    print(f"\n📡 {api_name}: {status}")
            
            # آمار database
            with sqlite3.connect(fetcher.state_manager.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM price_progress WHERE status = "completed"')
                price_completed = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM news_progress WHERE status = "completed"')
                news_completed = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM failed_items')
                failed_count = cursor.fetchone()[0]
                
                # آمار فایل‌ها
                price_files = len([f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv') and not f.startswith('news_')])
                news_files = len([f for f in os.listdir(RAW_DATA_PATH) if f.startswith('news_') and f.endswith('.csv')])
                
                print(f"\n💾 آمار Database:")
                print(f"   ✅ قیمت‌های تکمیل شده: {price_completed}")
                print(f"   ✅ اخبار تکمیل شده: {news_completed}")
                print(f"   ❌ آیتم‌های شکست خورده: {failed_count}")
                
                print(f"\n📁 آمار فایل‌ها:")
                print(f"   📊 فایل‌های قیمت: {price_files}")
                print(f"   📰 فایل‌های اخبار: {news_files}")
            
            input("\nEnter برای ادامه...")
            
        elif main_choice == '5':
            # مدیریت State
            print("\n💾 مدیریت State Database:")
            print("1. نمایش Sessions فعال")
            print("2. نمایش آیتم‌های شکست خورده")
            print("3. پاک کردن Database")
            print("4. نمایش جزئیات Session")
            print("5. بازگشت")
            
            state_choice = input("انتخاب: ")
            
            if state_choice == '1':
                with sqlite3.connect(fetcher.state_manager.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT session_id, session_type, status, total_symbols, completed_symbols, created_at
                        FROM extraction_sessions ORDER BY created_at DESC LIMIT 10
                    ''')
                    
                    print("\n📋 Sessions اخیر:")
                    sessions = cursor.fetchall()
                    if sessions:
                        for row in sessions:
                            print(f"   {row[0]} | {row[1]} | {row[2]} | {row[4]}/{row[3]} | {row[5]}")
                    else:
                        print("   هیچ session‌ای یافت نشد")
            
            elif state_choice == '2':
                with sqlite3.connect(fetcher.state_manager.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT item_type, exchange, symbol, error_message 
                        FROM failed_items ORDER BY created_at DESC LIMIT 20
                    ''')
                    
                    print("\n❌ آیتم‌های شکست خورده:")
                    failed_items = cursor.fetchall()
                    if failed_items:
                        for row in failed_items:
                            print(f"   {row[0]} | {row[1]} | {row[2]} | {row[3][:50]}...")
                    else:
                        print("   هیچ آیتم شکست خورده‌ای یافت نشد")
            
            elif state_choice == '3':
                confirm = input("⚠️ آیا مطمئن به پاک کردن Database هستید؟ (yes/no): ")
                if confirm == 'yes':
                    with sqlite3.connect(fetcher.state_manager.db_path) as conn:
                        conn.executescript('''
                            DELETE FROM extraction_sessions;
                            DELETE FROM price_progress;
                            DELETE FROM news_progress;
                            DELETE FROM rate_limits;
                            DELETE FROM failed_items;
                        ''')
                    print("✅ Database پاک شد")
                    logging.info("✅ Database پاک شد")
            
            elif state_choice == '4':
                # نمایش جزئیات Session
                session_id = input("Session ID را وارد کنید: ").strip()
                if session_id:
                    status = fetcher.state_manager.get_session_status(session_id)
                    if status:
                        print(f"\n📊 جزئیات Session: {session_id}")
                        print(f"   نوع: {status['session_type']}")
                        print(f"   وضعیت: {status['status']}")
                        print(f"   کل نمادها: {status['total_symbols']}")
                        print(f"   نمادهای تکمیل شده: {status['completed_symbols']}")
                        print(f"   قیمت: {status['price_progress']['completed']}/{status['price_progress']['total']}")
                        print(f"   اخبار: {status['news_progress']['completed']}/{status['news_progress']['total']}")
                    else:
                        print("❌ Session یافت نشد")
            
            if state_choice != '5':
                input("\nEnter برای ادامه...")
        
        elif main_choice == '6':
            print("\n👋 خداحافظ! Enhanced Unified Data Fetcher بسته شد.")
            logging.info("--- Enhanced Unified Data Fetcher به پایان رسید ---")
            break
        else:
            print("\n❌ انتخاب نامعتبر. لطفاً شماره‌ای بین 1 تا 6 وارد کنید.")
            input("Enter برای ادامه...")

if __name__ == '__main__':
    main()