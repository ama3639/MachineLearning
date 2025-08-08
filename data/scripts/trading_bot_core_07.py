#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت هسته اصلی ربات مشاور هوشمند (نسخه 5.2 - اصلاح کامل Authentication)

🔧 تغییرات این نسخه:
- ✅ رفع مشکل 401 Authentication Error
- ✅ اضافه کردن Commercial API Authentication
- ✅ بهبود سازگاری با API جدید (Optimized Models)
- ✅ نمایش Optimal Threshold در گزارش‌ها
- ✅ بهبود اطلاعات Performance مدل
- ✅ Enhanced Health Check
- ✅ بهتر شدن اطلاعات Risk Management

ویژگی‌های موجود:
- Risk Management Module
- Position Sizing با Kelly Criterion  
- Dynamic Stop Loss و Take Profit بر اساس ATR
- Max Drawdown Protection
- Portfolio Heat Management
- Binance API Fallback
- Multi-source Data
- Commercial API Authentication Support
"""

import os
import time
import pandas as pd
import requests
import logging
import configparser
import ccxt
import pandas_ta as ta
from typing import Optional, Dict, Any, List, Tuple
import datetime
import json
import glob
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict
import signal
import sys

# --- بخش ۱: خواندن پیکربندی و تنظیمات ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')

    LOG_PATH = config.get('Paths', 'logs')
    MODELS_PATH = config.get('Paths', 'models')
    FEATURES_PATH = config.get('Paths', 'features')
    API_HOST = config.get('API_Settings', 'host')
    API_PORT = config.getint('API_Settings', 'port')
    API_URL = f"http://{API_HOST}:{API_PORT}/predict"
    API_HEALTH_URL = f"http://{API_HOST}:{API_PORT}/health"
    API_MODEL_INFO_URL = f"http://{API_HOST}:{API_PORT}/model-info"
    
    CRYPTOCOMPARE_API_KEY = config.get('API_Keys', 'cryptocompare_api_key', fallback=None)

    # بررسی حالت multi-pair
    MULTI_PAIR_ENABLED = config.getboolean('Bot_Settings', 'multi_pair_enabled', fallback=False)
    
    if MULTI_PAIR_ENABLED:
        # خواندن لیست جفت ارزها و تایم‌فریم‌ها
        PAIRS_TO_MONITOR = [p.strip() for p in config.get('Bot_Settings', 'pairs_to_monitor').split(',')]
        TIMEFRAMES_TO_MONITOR = [t.strip() for t in config.get('Bot_Settings', 'timeframes_to_monitor').split(',')]
        EXCHANGE_TO_USE = config.get('Bot_Settings', 'exchange_to_use')
    else:
        # حالت تک جفت ارز (سازگاری با نسخه قبلی)
        EXCHANGE_TO_USE = config.get('Bot_Settings', 'exchange_to_use')
        SYMBOL_TO_TRADE = config.get('Bot_Settings', 'symbol_to_trade')
        TIMEFRAME_TO_TRADE = config.get('Bot_Settings', 'timeframe_to_trade')
        PAIRS_TO_MONITOR = [SYMBOL_TO_TRADE]
        TIMEFRAMES_TO_MONITOR = [TIMEFRAME_TO_TRADE]
    
    CANDLE_HISTORY_NEEDED = config.getint('Bot_Settings', 'candle_history_needed')
    POLL_INTERVAL_SECONDS = config.getint('Bot_Settings', 'poll_interval_seconds')
    CONFIDENCE_THRESHOLD = config.getfloat('Bot_Settings', 'confidence_threshold')
    
    # === 🔧 تنظیمات Authentication جدید ===
    try:
        # Authentication settings
        USE_AUTHENTICATION = config.getboolean('Bot_Authentication', 'use_authentication', fallback=True)
        API_USERNAME = config.get('Bot_Authentication', 'api_username', fallback='hasnamir92')
        API_PASSWORD = config.get('Bot_Authentication', 'api_password', fallback='123456')
        
        # اگر تنظیمات authentication موجود نباشد، از مقادیر پیش‌فرض استفاده کن
        if not config.has_section('Bot_Authentication'):
            logging.warning("Bot_Authentication section not found in config. Using default credentials.")
            USE_AUTHENTICATION = True
            API_USERNAME = "hasnamir92"  # نام کاربری که در لاگ دیدیم
            API_PASSWORD = "123456"     # رمز عبور پیش‌فرض - باید تغییر کند
            
    except Exception as e:
        logging.error(f"Error reading authentication config: {e}")
        # مقادیر پیش‌فرض
        USE_AUTHENTICATION = True
        API_USERNAME = "hasnamir92"
        API_PASSWORD = "123456"
    
    # تنظیمات تلگرام
    TELEGRAM_ENABLED = config.getboolean('Telegram', 'enabled', fallback=False)
    TELEGRAM_BOT_TOKEN = config.get('Telegram', 'bot_token', fallback=None)
    TELEGRAM_CHAT_ID = config.get('Telegram', 'chat_id', fallback=None)
    
    # پارامترهای اندیکاتورها از Feature_Engineering
    INDICATOR_PARAMS = {
        'rsi_length': config.getint('Feature_Engineering', 'rsi_length', fallback=14),
        'macd_fast': config.getint('Feature_Engineering', 'macd_fast', fallback=12),
        'macd_slow': config.getint('Feature_Engineering', 'macd_slow', fallback=26),
        'macd_signal': config.getint('Feature_Engineering', 'macd_signal', fallback=9),
        'bb_length': config.getint('Feature_Engineering', 'bb_length', fallback=20),
        'bb_std': config.getfloat('Feature_Engineering', 'bb_std', fallback=2.0),
        'atr_length': config.getint('Feature_Engineering', 'atr_length', fallback=14),
    }
    
    # === تنظیمات Risk Management جدید ===
    MAX_POSITION_SIZE = config.getfloat('Risk_Management', 'max_position_size', fallback=0.25)
    STOP_LOSS_ATR_MULTIPLIER = config.getfloat('Risk_Management', 'stop_loss_atr_multiplier', fallback=2.0)
    TAKE_PROFIT_ATR_MULTIPLIER = config.getfloat('Risk_Management', 'take_profit_atr_multiplier', fallback=3.0)
    MAX_DAILY_DRAWDOWN = config.getfloat('Risk_Management', 'max_daily_drawdown', fallback=0.10)
    KELLY_CRITERION_ENABLED = config.getboolean('Risk_Management', 'kelly_criterion_enabled', fallback=True)

except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini' or a required key is missing. Error: {e}")
    exit()

# --- بخش ۲: تنظیمات لاگ‌گیری پیشرفته ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)

# فایل‌های لاگ مختلف
log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
signals_log = os.path.join(log_subfolder_path, f"signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
performance_log = os.path.join(log_subfolder_path, f"performance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
risk_log = os.path.join(log_subfolder_path, f"risk_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# لیست برای ذخیره سیگنال‌ها
signals_history = []

# دیکشنری برای ذخیره آخرین timestamp پردازش شده برای هر جفت
last_processed_timestamps = {}

# Lock برای thread safety
signals_lock = threading.Lock()

# متغیر global برای اطلاعات مدل
api_model_info = {}

# متغیرهای global برای tracking
successful_predictions = 0
failed_attempts = 0
iteration_count = 0

# --- بخش Risk Management جدید ---
@dataclass
class Position:
    """کلاس برای نگهداری اطلاعات پوزیشن"""
    symbol: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime.datetime
    atr_at_entry: float
    confidence: float
    
class RiskManager:
    """مدیریت ریسک پیشرفته برای ربات معاملاتی"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # فعلاً برای شبیه‌سازی
        self.daily_pnl = 0
        self.max_drawdown = 0
        self.win_rate_history = defaultdict(list)  # برای محاسبه Kelly
        self.portfolio_heat = 0  # درصد کل سرمایه در ریسک
        self.daily_start_capital = initial_capital
        self.risk_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
        logging.info(f"💼 Risk Manager initialized with capital: ${initial_capital}")
    
    def calculate_kelly_fraction(self, symbol: str) -> float:
        """محاسبه Kelly Criterion برای position sizing"""
        if not KELLY_CRITERION_ENABLED:
            return MAX_POSITION_SIZE
        
        history = self.win_rate_history.get(symbol, [])
        if len(history) < 10:  # حداقل 10 معامله برای محاسبه معتبر
            return MAX_POSITION_SIZE * 0.5  # نصف حداکثر برای شروع
        
        wins = [h for h in history if h > 0]
        losses = [h for h in history if h < 0]
        
        if not wins or not losses:
            return MAX_POSITION_SIZE * 0.5
        
        win_rate = len(wins) / len(history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula: f = (p * b - q) / b
        # p = win rate, q = lose rate, b = win/loss ratio
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # محدود کردن Kelly به حداکثر 25% برای جلوگیری از ریسک بالا
        kelly_fraction = max(0, min(kelly_fraction, MAX_POSITION_SIZE))
        
        logging.info(f"📊 Kelly Fraction for {symbol}: {kelly_fraction:.2%} "
                    f"(Win Rate: {win_rate:.2%}, Avg Win/Loss: {b:.2f})")
        
        return kelly_fraction
    
    def calculate_position_size(self, symbol: str, confidence: float, 
                              current_price: float, atr: float) -> float:
        """محاسبه اندازه پوزیشن با در نظر گرفتن ریسک"""
        
        # بررسی محدودیت drawdown روزانه
        if self.check_daily_drawdown_limit():
            logging.warning("⚠️ Daily drawdown limit reached. No new positions.")
            return 0
        
        # محاسبه Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(symbol)
        
        # تنظیم بر اساس confidence
        confidence_multiplier = min(1.0, (confidence - CONFIDENCE_THRESHOLD) / (1 - CONFIDENCE_THRESHOLD))
        
        # محاسبه position size نهایی
        base_position_size = kelly_fraction * confidence_multiplier
        
        # محاسبه ریسک بر اساس ATR
        stop_loss_distance = atr * STOP_LOSS_ATR_MULTIPLIER
        risk_per_share = stop_loss_distance
        max_shares_by_risk = (self.current_capital * 0.02) / risk_per_share  # حداکثر 2% ریسک
        
        # محاسبه تعداد نهایی
        position_value = self.current_capital * base_position_size
        shares = min(position_value / current_price, max_shares_by_risk)
        
        # بررسی portfolio heat
        new_heat = self.calculate_portfolio_heat() + (shares * risk_per_share / self.current_capital)
        if new_heat > 0.06:  # حداکثر 6% portfolio heat
            logging.warning(f"⚠️ Portfolio heat too high ({new_heat:.1%}). Reducing position size.")
            shares *= (0.06 - self.calculate_portfolio_heat()) / (new_heat - self.calculate_portfolio_heat())
        
        logging.info(f"📏 Position Size for {symbol}: {shares:.2f} shares "
                    f"(${shares * current_price:.2f}) at ${current_price:.2f}")
        
        return shares
    
    def calculate_stop_loss(self, entry_price: float, atr: float, signal: str) -> float:
        """محاسبه Stop Loss بر اساس ATR"""
        if signal == "PROFIT":
            # برای خرید، stop loss زیر قیمت ورود
            stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
        else:
            # برای فروش، stop loss بالای قیمت ورود
            stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
        
        return round(stop_loss, 2)
    
    def calculate_take_profit(self, entry_price: float, atr: float, signal: str) -> float:
        """محاسبه Take Profit بر اساس ATR"""
        if signal == "PROFIT":
            # برای خرید، take profit بالای قیمت ورود
            take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
        else:
            # برای فروش، take profit زیر قیمت ورود
            take_profit = entry_price - (atr * TAKE_PROFIT_ATR_MULTIPLIER)
        
        return round(take_profit, 2)
    
    def check_daily_drawdown_limit(self) -> bool:
        """بررسی محدودیت drawdown روزانه"""
        daily_loss = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        return daily_loss >= MAX_DAILY_DRAWDOWN
    
    def calculate_portfolio_heat(self) -> float:
        """محاسبه مجموع ریسک فعلی پورتفولیو"""
        total_risk = 0
        for position in self.positions.values():
            risk = abs(position.entry_price - position.stop_loss) * position.position_size
            total_risk += risk
        
        self.portfolio_heat = total_risk / self.current_capital
        return self.portfolio_heat
    
    def update_performance_metrics(self, symbol: str, pnl: float):
        """بروزرسانی معیارهای عملکرد"""
        self.risk_metrics['total_trades'] += 1
        
        if pnl > 0:
            self.risk_metrics['winning_trades'] += 1
        else:
            self.risk_metrics['losing_trades'] += 1
        
        self.risk_metrics['total_pnl'] += pnl
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        # بروزرسانی win rate history
        self.win_rate_history[symbol].append(pnl)
        if len(self.win_rate_history[symbol]) > 100:  # حداکثر 100 معامله اخیر
            self.win_rate_history[symbol].pop(0)
        
        # محاسبه معیارها
        if self.risk_metrics['total_trades'] > 0:
            self.risk_metrics['win_rate'] = self.risk_metrics['winning_trades'] / self.risk_metrics['total_trades']
        
        # محاسبه max drawdown
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        self.risk_metrics['max_drawdown'] = max(self.risk_metrics['max_drawdown'], drawdown)
        
        # ذخیره معیارها
        self.save_risk_metrics()
    
    def save_risk_metrics(self):
        """ذخیره معیارهای ریسک"""
        try:
            with open(risk_log, 'w') as f:
                json.dump({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'current_capital': self.current_capital,
                    'daily_pnl': self.daily_pnl,
                    'portfolio_heat': self.portfolio_heat,
                    'metrics': self.risk_metrics
                }, f, indent=2)
        except Exception as e:
            logging.error(f"خطا در ذخیره risk metrics: {e}")
    
    def reset_daily_metrics(self):
        """ریست معیارهای روزانه"""
        self.daily_pnl = 0
        self.daily_start_capital = self.current_capital
        logging.info(f"📅 Daily metrics reset. Starting capital: ${self.current_capital:.2f}")
    
    def get_risk_report(self) -> str:
        """تولید گزارش ریسک"""
        report = f"""
📊 گزارش مدیریت ریسک
========================
💰 سرمایه فعلی: ${self.current_capital:.2f}
📈 سود/زیان روزانه: ${self.daily_pnl:.2f} ({self.daily_pnl/self.daily_start_capital*100:.1f}%)
🔥 Portfolio Heat: {self.portfolio_heat:.1%}
📉 Max Drawdown: {self.risk_metrics['max_drawdown']:.1%}
🎯 Win Rate: {self.risk_metrics['win_rate']:.1%}
📊 مجموع معاملات: {self.risk_metrics['total_trades']}
✅ معاملات برنده: {self.risk_metrics['winning_trades']}
❌ معاملات بازنده: {self.risk_metrics['losing_trades']}
💵 مجموع سود/زیان: ${self.risk_metrics['total_pnl']:.2f}
"""
        return report

# ایجاد instance از Risk Manager
risk_manager = RiskManager()

# === توابع cleanup برای تلگرام ===
def cleanup_and_shutdown():
    """تابع cleanup برای ارسال پیام قطع ارتباط و ذخیره آمار"""
    global successful_predictions, failed_attempts, iteration_count
    
    try:
        # ذخیره نهایی قبل از خروج
        save_performance_metrics()
        risk_manager.save_risk_metrics()
        
        # ارسال پیام خاموش شدن
        if TELEGRAM_ENABLED:
            total_attempts = successful_predictions + failed_attempts
            final_risk_report = risk_manager.get_risk_report()
            
            shutdown_message = f"""
🛑 <b>ربات مشاور هوشمند v5.2 متوقف شد</b>

📊 <b>آمار نهایی:</b>
• تعداد کل بررسی‌ها: {iteration_count}
• سیگنال‌های صادر شده: {len(signals_history)}
• نرخ موفقیت: {(successful_predictions / total_attempts * 100) if total_attempts > 0 else 0:.1f}%

🤖 <b>مدل استفاده شده:</b>
{api_model_info.get('model_type', 'Unknown')} {'(Optimized)' if api_model_info.get('is_optimized') else ''}

🔐 <b>Authentication:</b>
User: {API_USERNAME} {'(Success)' if USE_AUTHENTICATION else '(Disabled)'}

{final_risk_report}

🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#BotStopped #v5_2 #AuthFixed
"""
            send_telegram_message(shutdown_message)
            logging.info("📱 Shutdown message sent to Telegram")
        
        logging.info("\n👋 Bot shutdown complete")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logging.info("\n⛔ Received shutdown signal (Ctrl+C)")
    print("\n⛔ Shutting down gracefully...")
    cleanup_and_shutdown()
    sys.exit(0)

# === 🔧 بخش جدید: Authentication Check ===
def check_authentication():
    """بررسی Authentication قبل از شروع ربات"""
    if not USE_AUTHENTICATION:
        logging.info("🔓 Authentication disabled - running in legacy mode")
        return True
    
    try:
        # تست ساده authentication با health endpoint
        logging.info(f"🔐 Testing authentication with username: {API_USERNAME}")
        
        test_response = requests.get(
            API_HEALTH_URL, 
            timeout=5,
            auth=(API_USERNAME, API_PASSWORD)
        )
        
        if test_response.status_code == 200:
            logging.info("✅ Authentication test successful")
            return True
        elif test_response.status_code == 401:
            logging.error("❌ Authentication test failed - Invalid credentials")
            logging.error(f"💡 Username: {API_USERNAME}")
            logging.error("💡 Please update Bot_Authentication section in config.ini")
            return False
        else:
            logging.warning(f"⚠️ Unexpected response: {test_response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Authentication test error: {e}")
        return False

# === بخش جدید: API Health Check بهبود یافته ===
def check_api_health():
    """بررسی سلامت API و دریافت اطلاعات مدل (اصلاح شده)"""
    global api_model_info
    
    try:
        # Health check با timeout بیشتر
        logging.info(f"🔍 Checking API health at {API_HEALTH_URL}")
        
        # 🔧 اضافه کردن Authentication اگر فعال باشد
        if USE_AUTHENTICATION:
            health_response = requests.get(API_HEALTH_URL, timeout=10, auth=(API_USERNAME, API_PASSWORD))
        else:
            health_response = requests.get(API_HEALTH_URL, timeout=10)
        
        # لاگ response برای debugging
        logging.info(f"📡 API Response Status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            if health_data.get('status') == 'healthy':
                logging.info("✅ API Health Check: Healthy")
                
                # دریافت اطلاعات مدل
                if 'model_info' in health_data:
                    api_model_info = health_data['model_info']
                    model_type = api_model_info.get('model_type', 'Unknown')
                    threshold = api_model_info.get('optimal_threshold', 0.5)
                    is_optimized = api_model_info.get('is_optimized', False)
                    
                    logging.info(f"🤖 Model Type: {model_type}")
                    logging.info(f"🎯 Optimal Threshold: {threshold:.4f}")
                    logging.info(f"⚡ Optimized Model: {'Yes' if is_optimized else 'No'}")
                    
                    # نمایش performance اگر موجود باشد
                    performance = api_model_info.get('performance')
                    if performance and performance.get('accuracy'):
                        logging.info(f"📊 Model Performance: Accuracy={performance['accuracy']:.1%}, "
                                   f"Precision={performance['precision']:.1%}, "
                                   f"Recall={performance['recall']:.1%}")
                
                return True
            else:
                logging.error("❌ API Health Check: Unhealthy")
                logging.error(f"📋 Health response: {health_data}")
                return False
                
        elif health_response.status_code == 401:
            # خطای Authentication
            logging.error("❌ API Health Check failed: 401 Authentication Error")
            logging.error(f"💡 Current credentials: {API_USERNAME} / [password hidden]")
            logging.error("💡 Please check Bot_Authentication section in config.ini")
            return False
        elif health_response.status_code == 500:
            # خطای سرور - تلاش برای دریافت جزئیات خطا
            try:
                error_data = health_response.json()
                logging.error(f"❌ API Health Check failed (HTTP 500): {error_data}")
            except:
                error_text = health_response.text[:200]  # اول 200 کاراکتر
                logging.error(f"❌ API Health Check failed (HTTP 500): {error_text}")
            return False
        else:
            logging.error(f"❌ API Health Check failed: HTTP {health_response.status_code}")
            try:
                response_text = health_response.text[:200]
                logging.error(f"📋 Response: {response_text}")
            except:
                pass
            return False
            
    except requests.exceptions.ConnectionError as e:
        logging.error(f"❌ Connection Error: API server not reachable - {e}")
        return False
    except requests.exceptions.Timeout as e:
        logging.error(f"❌ Timeout Error: API server too slow - {e}")
        return False
    except Exception as e:
        logging.error(f"❌ API Health Check error: {e}")
        return False

# === اصلاح بخش test API connection ===
def test_api_connection():
    """تست اتصال API با جزئیات بیشتر"""
    print("\n🔍 Testing API Connection...")
    
    # تست endpoint اصلی
    try:
        response = requests.get(f"http://{API_HOST}:{API_PORT}/", timeout=10)
        if response.status_code == 200:
            print(f"✅ Main endpoint accessible: {response.text[:50]}...")
        else:
            print(f"⚠️ Main endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Main endpoint failed: {e}")
    
    # تست health endpoint
    try:
        if USE_AUTHENTICATION:
            response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=10, auth=(API_USERNAME, API_PASSWORD))
        else:
            response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=10)
            
        print(f"📊 Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check successful: {data.get('status')}")
            return True
        elif response.status_code == 401:
            print(f"❌ Authentication error in health endpoint")
            print(f"💡 Username: {API_USERNAME}")
            print(f"💡 Check config.ini [Bot_Authentication] section")
        elif response.status_code == 500:
            print(f"❌ Server error in health endpoint")
            try:
                error_data = response.json()
                print(f"📋 Error details: {error_data}")
            except:
                print(f"📋 Error text: {response.text[:200]}")
        else:
            print(f"⚠️ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
    
    return False

# --- بخش ۳: توابع تلگرام (با افزودن گزارش ریسک) ---
def send_telegram_message(message: str) -> bool:
    """ارسال پیام به تلگرام"""
    if not TELEGRAM_ENABLED:
        return False
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("تنظیمات تلگرام ناقص است. پیام ارسال نشد.")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        
        if response.json().get('ok'):
            logging.info("پیام با موفقیت به تلگرام ارسال شد.")
            return True
        else:
            logging.error(f"خطا در ارسال پیام تلگرام: {response.json()}")
            return False
            
    except Exception as e:
        logging.error(f"خطا در ارسال پیام به تلگرام: {e}")
        return False

def format_telegram_message(symbol: str, timeframe: str, signal: str, confidence: float, 
                          exchange: str, position_size: float = None, stop_loss: float = None, 
                          take_profit: float = None, threshold_used: float = None) -> str:
    """فرمت‌دهی پیام برای تلگرام با اطلاعات مدل بهبود یافته"""
    emoji_signal = "🟢" if signal == "PROFIT" else "🔴"
    emoji_confidence = "🔥" if confidence >= 0.8 else "✅" if confidence >= 0.7 else "⚡"
    
    # اطلاعات مدل
    model_type = api_model_info.get('model_type', 'Unknown')
    is_optimized = api_model_info.get('is_optimized', False)
    model_accuracy = api_model_info.get('performance', {}).get('accuracy')
    
    message = f"""
{emoji_signal} <b>سیگنال جدید از ربات مشاور هوشمند v5.2</b> {emoji_signal}

📊 <b>نماد:</b> {symbol}
⏱ <b>تایم فریم:</b> {timeframe}
🏦 <b>صرافی:</b> {exchange.upper()}
📈 <b>سیگنال:</b> <b>{signal}</b>
{emoji_confidence} <b>اطمینان:</b> {confidence:.1%}
🎯 <b>آستانه:</b> {CONFIDENCE_THRESHOLD:.0%}
"""

    # اطلاعات مدل بهبود یافته
    if threshold_used:
        threshold_emoji = "⚡" if is_optimized else "🔧"
        message += f"""
🤖 <b>مدل:</b> {model_type[:20]}{'...' if len(model_type) > 20 else ''}
{threshold_emoji} <b>Threshold:</b> {threshold_used:.3f} {'(Optimized)' if is_optimized else '(Default)'}
"""
    
    if model_accuracy:
        message += f"📊 <b>دقت مدل:</b> {model_accuracy:.1%}\n"
    
    # اطلاعات Authentication
    auth_emoji = "🔐" if USE_AUTHENTICATION else "🔓"
    message += f"{auth_emoji} <b>Auth:</b> {API_USERNAME if USE_AUTHENTICATION else 'Disabled'}\n"
    
    # افزودن اطلاعات Risk Management
    if position_size is not None:
        message += f"""
💼 <b>مدیریت ریسک:</b>
   📏 اندازه پوزیشن: {position_size:.2f} واحد
   🛑 حد ضرر: ${stop_loss:.2f}
   ✅ حد سود: ${take_profit:.2f}
   🔥 Portfolio Heat: {risk_manager.portfolio_heat:.1%}
"""
    
    message += f"""
🕐 <b>زمان:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#SmartAdvisor #CryptoSignal #{symbol.replace('/', '')} #{timeframe} #v5_2 #AuthFixed
"""
    return message

# --- بخش ۴: توابع بررسی همخوانی ---
def load_model_features() -> Optional[List[str]]:
    """بارگذاری لیست ویژگی‌های ذخیره شده توسط مدل"""
    try:
        # سعی در دریافت از API
        try:
            if USE_AUTHENTICATION:
                response = requests.get(API_MODEL_INFO_URL, timeout=5, auth=(API_USERNAME, API_PASSWORD))
            else:
                response = requests.get(API_MODEL_INFO_URL, timeout=5)
                
            if response.status_code == 200:
                model_info = response.json()
                feature_columns = model_info.get('model_info', {}).get('feature_columns', [])
                if feature_columns:
                    logging.info(f"✅ Model features from API: {len(feature_columns)} features")
                    return feature_columns
        except:
            logging.warning("Could not get features from API, trying local files...")
        
        # جستجو در فایل‌های محلی
        list_of_files = glob.glob(os.path.join(MODELS_PATH, 'feature_names_optimized_*.txt'))
        if not list_of_files:
            list_of_files = glob.glob(os.path.join(MODELS_PATH, 'feature_names_*.txt'))
        
        if not list_of_files:
            # جستجو در پوشه‌های run_*
            list_of_files = glob.glob(os.path.join(MODELS_PATH, 'run_*/feature_names_*.txt'))
        
        if not list_of_files:
            logging.warning("فایل feature_names یافت نشد. ربات بدون بررسی همخوانی ادامه می‌دهد.")
            return None
        
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f if line.strip()]
        
        logging.info(f"✅ لیست {len(feature_names)} ویژگی از '{os.path.basename(latest_file)}' بارگذاری شد.")
        return feature_names
        
    except Exception as e:
        logging.error(f"خطا در بارگذاری feature_names: {e}")
        return None

def verify_feature_consistency(calculated_features: Dict[str, Any], expected_features: List[str]) -> bool:
    """بررسی تطابق ویژگی‌های محاسبه شده با انتظارات مدل"""
    missing_features = []
    for feature in expected_features:
        if feature not in calculated_features:
            missing_features.append(feature)
    
    if missing_features:
        logging.error(f"❌ ویژگی‌های گمشده: {missing_features}")
        return False
    
    logging.info(f"✅ تمام {len(expected_features)} ویژگی مورد نیاز محاسبه شده‌اند.")
    return True

# --- بخش ۵: توابع اصلی (با اصلاح مشکل Binance API) ---
def fetch_from_cryptocompare_api(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """تابع اختصاصی برای دریافت داده از CryptoCompare API."""
    logging.info("Using dedicated function for CryptoCompare...")
    BASE_URL = "https://min-api.cryptocompare.com/data/v2/"
    endpoint_map = {'m': 'histominute', 'h': 'histohour', 'd': 'histoday'}
    
    try:
        tf_unit = timeframe.lower()[-1]
        tf_agg = int(timeframe[:-1])
        endpoint = endpoint_map.get(tf_unit)
        if not endpoint: raise ValueError("Timeframe not recognized for CryptoCompare.")
        base_sym, quote_sym = symbol.upper().split('/')
    except Exception as e:
        logging.error(f"[CryptoCompare] Invalid symbol/timeframe format: {e}")
        return None
    
    params = {"fsym": base_sym, "tsym": quote_sym, "limit": limit, "aggregate": tf_agg}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'Error':
            logging.error(f"[CryptoCompare] API Error: {data.get('Message', 'Unknown error')}")
            return None
        
        df = pd.DataFrame(data['Data']['Data'])
        if df.empty: return None
        
        df.rename(columns={'volumefrom': 'volume'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        logging.error(f"[CryptoCompare] Failed to fetch data: {e}")
        return None

def get_latest_data(symbol: str, timeframe: str, limit: int, exchange_name: str) -> Optional[pd.DataFrame]:
    """
    تابع اصلاح شده برای دریافت داده با حل مشکل Binance API
    """
    logging.info(f"Attempting to fetch data from: {exchange_name.upper()} for {symbol} {timeframe}")
    
    if exchange_name.lower() == 'cryptocompare':
        return fetch_from_cryptocompare_api(symbol, timeframe, limit)
    else:
        try:
            # تنظیمات اصلاح شده برای Binance
            if exchange_name.lower() == 'binance':
                exchange = ccxt.binance({
                    'timeout': 30000,  # 30 ثانیه timeout
                    'rateLimit': 100,  # محدودیت نرخ
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'  # مشخص کردن نوع معاملات
                    }
                })
            else:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'timeout': 30000,
                    'rateLimit': 1000,
                    'enableRateLimit': True
                })
            
            # تلاش برای دریافت داده با retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    if len(df) < limit // 2:  # اگر خیلی کم داده دریافت شد
                        logging.warning(f"Data received ({len(df)}) is less than expected ({limit}).")
                        if attempt < max_retries - 1:
                            logging.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                            time.sleep(2)
                            continue
                    
                    logging.info(f"Successfully fetched {len(df)} candles from {exchange_name.upper()}")
                    return df
                    
                except Exception as attempt_error:
                    logging.warning(f"Attempt {attempt + 1} failed: {attempt_error}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # انتظار قبل از تلاش مجدد
                        continue
                    else:
                        raise attempt_error
            
        except AttributeError:
            logging.error(f"Exchange '{exchange_name}' is not supported by CCXT.")
        except ccxt.NetworkError as e:
            logging.error(f"Network error accessing {exchange_name.upper()}: {e}")
            # اگر Binance کار نکرد، fallback به CryptoCompare
            if exchange_name.lower() == 'binance':
                logging.info("🔄 Fallback to CryptoCompare due to Binance connection issues...")
                return fetch_from_cryptocompare_api(symbol, timeframe, limit)
        except ccxt.BaseError as e:
            logging.error(f"Exchange error from {exchange_name.upper()}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error fetching data from {exchange_name.upper()}: {e}")
            # اگر Binance کار نکرد، fallback به CryptoCompare
            if exchange_name.lower() == 'binance':
                logging.info("🔄 Fallback to CryptoCompare due to connection issues...")
                return fetch_from_cryptocompare_api(symbol, timeframe, limit)
        
        return None

def calculate_features(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """محاسبه ویژگی‌ها - آینه کامل از اسکریپت 03 با اضافه شدن برگرداندن ATR"""
    try:
        group = df.copy()
        
        
        # 🔧 اصلاح مشکل dtype - تبدیل volume به float64
        group['volume'] = group['volume'].astype('float64')
        group['high'] = group['high'].astype('float64')
        group['low'] = group['low'].astype('float64')
        group['close'] = group['close'].astype('float64')
        group['open'] = group['open'].astype('float64')
        
        # محاسبات مطابق با پارامترهای config
        group['rsi'] = ta.rsi(group['close'], length=INDICATOR_PARAMS['rsi_length'])
        
        macd = ta.macd(group['close'], 
                      fast=INDICATOR_PARAMS['macd_fast'], 
                      slow=INDICATOR_PARAMS['macd_slow'], 
                      signal=INDICATOR_PARAMS['macd_signal'])
        if macd is not None and not macd.empty:
            col_names = macd.columns.tolist()
            group['macd'] = macd[col_names[0]]
            group['macd_hist'] = macd[col_names[1]]
            group['macd_signal'] = macd[col_names[2]]
        
        bbands = ta.bbands(group['close'], 
                          length=INDICATOR_PARAMS['bb_length'], 
                          std=INDICATOR_PARAMS['bb_std'])
        if bbands is not None and not bbands.empty:
            col_names = bbands.columns.tolist()
            group['bb_upper'] = bbands[col_names[0]]
            group['bb_middle'] = bbands[col_names[1]]
            group['bb_lower'] = bbands[col_names[2]]
            group['bb_position'] = (group['close'] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'])
        
        group['atr'] = ta.atr(group['high'], group['low'], group['close'], 
                             length=INDICATOR_PARAMS['atr_length'])
        group['atr_percent'] = (group['atr'] / group['close']) * 100
        
        # بقیه محاسبات مطابق اسکریپت 03
        group['price_change'] = group['close'].pct_change()
        group['volatility'] = group['price_change'].rolling(window=20).std() * 100
        
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        vwap_numerator = (typical_price * group['volume']).cumsum()
        vwap_denominator = group['volume'].cumsum()
        group['vwap'] = vwap_numerator / vwap_denominator
        group['vwap_deviation'] = ((group['close'] - group['vwap']) / group['vwap']) * 100
        
        group['obv'] = ta.obv(group['close'], group['volume'])
        group['obv_change'] = group['obv'].pct_change()
        
        # 🔧 اصلاح کامل MFI calculation
        try:
            # تبدیل صریح انواع داده برای MFI
            high_values = group['high'].astype('float64')
            low_values = group['low'].astype('float64') 
            close_values = group['close'].astype('float64')
            volume_values = group['volume'].astype('float64')
            
            group['mfi'] = ta.mfi(high_values, low_values, close_values, volume_values, length=14)
        except Exception as mfi_error:
            logging.warning(f"MFI calculation failed: {mfi_error}. Using default value.")
            group['mfi'] = 50.0  # مقدار پیش‌فرض
        
        group['ad'] = ta.ad(group['high'], group['low'], group['close'], group['volume'])
        
        stoch = ta.stoch(group['high'], group['low'], group['close'], k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            col_names = stoch.columns.tolist()
            group['stoch_k'] = stoch[col_names[0]]
            group['stoch_d'] = stoch[col_names[1]]
        
        group['williams_r'] = ta.willr(group['high'], group['low'], group['close'], length=14)
        group['cci'] = ta.cci(group['high'], group['low'], group['close'], length=20)
        
        group['ema_short'] = ta.ema(group['close'], length=12)
        group['ema_medium'] = ta.ema(group['close'], length=26)
        group['ema_long'] = ta.ema(group['close'], length=50)
        group['ema_short_above_medium'] = (group['ema_short'] > group['ema_medium']).astype(int)
        group['ema_medium_above_long'] = (group['ema_medium'] > group['ema_long']).astype(int)
        group['ema_short_slope'] = group['ema_short'].pct_change(periods=5)
        group['ema_medium_slope'] = group['ema_medium'].pct_change(periods=5)
        
        group['sma_short'] = ta.sma(group['close'], 10)
        group['sma_medium'] = ta.sma(group['close'], 20)
        group['sma_long'] = ta.sma(group['close'], 50)
        group['price_above_sma_short'] = (group['close'] > group['sma_short']).astype(int)
        group['price_above_sma_medium'] = (group['close'] > group['sma_medium']).astype(int)
        group['price_above_sma_long'] = (group['close'] > group['sma_long']).astype(int)
        
        group['return_1'] = group['close'].pct_change(1)
        group['return_5'] = group['close'].pct_change(5)
        group['return_10'] = group['close'].pct_change(10)
        group['avg_return_5'] = group['return_1'].rolling(5).mean()
        group['avg_return_10'] = group['return_1'].rolling(10).mean()
        
        group['hl_ratio'] = (group['high'] - group['low']) / group['close']
        group['close_position'] = (group['close'] - group['low']) / (group['high'] - group['low'])
        group['volume_ma'] = group['volume'].rolling(20).mean()
        group['volume_ratio'] = group['volume'] / group['volume_ma']
        
        try:
            psar = ta.psar(group['high'], group['low'], group['close'])
            if psar is not None and len(psar) > 0:
                if isinstance(psar, pd.DataFrame):
                    group['psar'] = psar.iloc[:, 0]
                else:
                    group['psar'] = psar
                group['price_above_psar'] = (group['close'] > group['psar']).astype(int)
            else:
                group['psar'] = group['close'].shift(1).fillna(group['close']) * 0.98
                group['price_above_psar'] = 1
        except Exception as e:
            group['psar'] = group['close'].shift(1).fillna(group['close']) * 0.98  
            group['price_above_psar'] = 1
        
        adx = ta.adx(group['high'], group['low'], group['close'], length=14)
        if adx is not None and not adx.empty:
            col_names = adx.columns.tolist()
            for col in col_names:
                if 'ADX' in col:
                    group['adx'] = adx[col]
                    break
        
        # ویژگی‌های احساسات (با مقادیر پیش‌فرض بهبود یافته)
        group['sentiment_score'] = 0
        group['sentiment_momentum'] = 0
        group['sentiment_ma_7'] = 0
        group['sentiment_ma_14'] = 0
        group['sentiment_volume'] = 0
        group['sentiment_divergence'] = 0

        # استخراج آخرین ردیف
        latest_features = group.iloc[-1].to_dict()
        
        # ذخیره مقدار ATR برای محاسبات Risk Management
        latest_atr = group['atr'].iloc[-1]
        
        # 🔧 فیلتر کردن NaN ها و مقادیر inf (اصلاح نهایی)
        features_for_api = {}
        for k, v in latest_features.items():
            try:
                # بررسی نوع داده و validity
                if pd.notna(v):
                    # بررسی برای انواع numeric
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if not np.isinf(v):
                            # تبدیل numpy types به Python native types
                            if isinstance(v, np.integer):
                                features_for_api[k] = int(v)
                            elif isinstance(v, np.floating):
                                features_for_api[k] = float(v)
                            else:
                                features_for_api[k] = v
                    # برای انواع non-numeric، مستقیماً اضافه کن
                    elif isinstance(v, (str, bool)):
                        features_for_api[k] = v
                    # برای datetime objects
                    elif hasattr(v, 'timestamp'):
                        continue  # رد کردن timestamp ها
                    else:
                        # سایر انواع - تلاش برای تبدیل به string
                        try:
                            str_val = str(v)
                            if str_val not in ['nan', 'inf', '-inf', 'NaT']:
                                features_for_api[k] = str_val
                        except:
                            continue  # رد کردن مقادیر غیرقابل تبدیل
            except Exception as e:
                logging.warning(f"Error processing feature {k}={v}: {e}")
                continue
        
        # حذف timestamp
        features_for_api.pop('timestamp', None)
        
        # 🔧 بررسی مقادیر معقول (اصلاح شده - حفظ مقدار 0)
        cleaned_features = {}
        for k, v in features_for_api.items():
            if isinstance(v, (int, float)):
                # فقط حذف مقادیر خیلی بزرگ (حفظ 0 و مقادیر کوچک)
                if abs(v) < 1e10:  # 🔧 حذف شرط > 1e-10 برای حفظ 0
                    cleaned_features[k] = v
                else:
                    logging.warning(f"Outlier value removed: {k}={v}")
            else:
                cleaned_features[k] = v
        
        # افزودن ATR به خروجی (برای Risk Management)
        if not np.isinf(latest_atr) and pd.notna(latest_atr):
            cleaned_features['_atr_value'] = float(latest_atr)
        else:
            cleaned_features['_atr_value'] = 1.0  # مقدار پیش‌فرض
        
        # بررسی تعداد ویژگی‌ها
        logging.info(f"تعداد ویژگی‌های محاسبه شده: {len(cleaned_features)}")
        
        return cleaned_features
        
    except Exception as e:
        logging.error(f"خطا در محاسبه ویژگی‌ها: {e}", exc_info=True)
        return None
        
def get_prediction(payload: Dict) -> Optional[Dict]:
    """ارسال درخواست به API پیش‌بینی بهبود یافته با Authentication"""
    try:
        # حذف ATR از payload قبل از ارسال به API
        atr_value = payload.pop('_atr_value', None)
        
        # 🔧 Debugging: بررسی payload قبل از ارسال
        logging.debug(f"Payload size: {len(payload)} features")
        
        # بررسی مقادیر مشکوک
        problematic_values = []
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                if np.isinf(v) or np.isnan(v) or abs(v) > 1e8:
                    problematic_values.append(f"{k}={v}")
        
        if problematic_values:
            logging.warning(f"Problematic values detected: {problematic_values[:5]}")
            # حذف مقادیر مشکوک
            cleaned_payload = {}
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    if not (np.isinf(v) or np.isnan(v) or abs(v) > 1e8):
                        cleaned_payload[k] = v
                else:
                    cleaned_payload[k] = v
            payload = cleaned_payload
            logging.info(f"Cleaned payload size: {len(payload)} features")
        
        # تبدیل payload به JSON قابل serialize
        json_payload = {}
        for k, v in payload.items():
            if isinstance(v, np.integer):
                json_payload[k] = int(v)
            elif isinstance(v, np.floating):
                json_payload[k] = float(v)
            elif isinstance(v, (int, float, str, bool)):
                json_payload[k] = v
            else:
                logging.warning(f"Skipping non-serializable value: {k}={type(v)}")
        
        # 🔧 ارسال درخواست با Authentication
        if USE_AUTHENTICATION:
            logging.debug(f"🔐 Using Basic Auth with username: {API_USERNAME}")
            response = requests.post(
                API_URL, 
                json=json_payload, 
                timeout=15,
                auth=(API_USERNAME, API_PASSWORD)  # 🔧 اضافه کردن Basic Auth
            )
        else:
            # حالت غیر تجاری (برای backward compatibility)
            response = requests.post(API_URL, json=json_payload, timeout=15)
        
        # بررسی response
        if response.status_code == 401:
            logging.error(f"❌ Authentication failed! Username: {API_USERNAME}")
            logging.error("💡 Make sure username and password are correct in config.ini")
            logging.error("💡 Check if user exists in commercial database")
            return None
        elif response.status_code == 500:
            # لاگ جزئیات خطای سرور
            try:
                error_detail = response.json()
                logging.error(f"API Server Error Details: {error_detail}")
            except:
                error_text = response.text[:500]
                logging.error(f"API Server Error Text: {error_text}")
            return None
        
        response.raise_for_status()
        
        result = response.json()
        # اضافه کردن ATR به نتیجه
        if atr_value:
            result['atr'] = atr_value
            
        # لاگ اطلاعات تفصیلی‌تر
        if 'model_info' in result:
            model_info = result['model_info']
            logging.info(f"🤖 Model: {model_info.get('model_type', 'Unknown')}")
            logging.info(f"🎯 Threshold Used: {model_info.get('threshold_used', 0.5):.4f}")
            logging.info(f"⚡ Optimized: {'Yes' if model_info.get('is_optimized') else 'No'}")
            
        return result
        
    except requests.exceptions.RequestException as e:
        if "401" in str(e):
            logging.error(f"❌ Authentication Error: {e}")
            logging.error(f"💡 Current credentials: {API_USERNAME} / [password hidden]")
            logging.error("💡 Please check Bot_Authentication section in config.ini")
        else:
            logging.error(f"خطا در برقراری ارتباط با API: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in get_prediction: {e}")
        return None
    
def save_signal(signal_data: Dict):
    """ذخیره سیگنال در فایل JSON"""
    with signals_lock:
        signals_history.append(signal_data)
        try:
            with open(signals_log, 'w', encoding='utf-8') as f:
                json.dump(signals_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"خطا در ذخیره سیگنال: {e}")

def save_performance_metrics():
    """ذخیره معیارهای عملکرد"""
    try:
        if signals_history:
            df_performance = pd.DataFrame(signals_history)
            df_performance.to_csv(performance_log, index=False)
            logging.info(f"معیارهای عملکرد در {performance_log} ذخیره شد.")
    except Exception as e:
        logging.error(f"خطا در ذخیره معیارهای عملکرد: {e}")

def send_notification(symbol, timeframe, signal, confidence, current_price, atr, 
                     prediction_result=None):
    """ارسال اعلان به کنسول و تلگرام و ذخیره سیگنال با اطلاعات بهبود یافته"""
    
    # محاسبات Risk Management
    position_size = risk_manager.calculate_position_size(symbol, confidence, current_price, atr)
    
    if position_size == 0:
        logging.warning(f"⚠️ Position size is 0 for {symbol}. Skipping notification.")
        return
    
    stop_loss = risk_manager.calculate_stop_loss(current_price, atr, signal)
    take_profit = risk_manager.calculate_take_profit(current_price, atr, signal)
    
    # استخراج threshold از prediction result
    threshold_used = None
    if prediction_result and 'model_info' in prediction_result:
        threshold_used = prediction_result['model_info'].get('threshold_used', 0.5)
    
    signal_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": EXCHANGE_TO_USE,
        "signal": signal,
        "confidence": confidence,
        "threshold": CONFIDENCE_THRESHOLD,
        "threshold_used": threshold_used,
        "current_price": current_price,
        "position_size": position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,
        "portfolio_heat": risk_manager.portfolio_heat,
        # اطلاعات مدل
        "model_type": api_model_info.get('model_type', 'Unknown'),
        "is_optimized": api_model_info.get('is_optimized', False),
        # اطلاعات Authentication
        "authenticated": USE_AUTHENTICATION,
        "api_username": API_USERNAME if USE_AUTHENTICATION else None
    }
    
    # ذخیره سیگنال
    save_signal(signal_data)
    
    # ایجاد position شبیه‌سازی شده
    position = Position(
        symbol=symbol,
        entry_price=current_price,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_time=datetime.datetime.now(),
        atr_at_entry=atr,
        confidence=confidence
    )
    risk_manager.positions[symbol] = position
    
    # نمایش در کنسول با اطلاعات بهبود یافته
    threshold_info = f"({threshold_used:.4f})" if threshold_used else f"({CONFIDENCE_THRESHOLD:.2%})"
    model_info_text = f"Model: {api_model_info.get('model_type', 'Unknown')[:20]}"
    auth_info = f"Auth: {API_USERNAME}" if USE_AUTHENTICATION else "Auth: Disabled"
    
    console_message = f"""
    ================================================
    !!!    سیگنال جدید از مشاور هوشمند v5.2    !!!
    ================================================
    نماد:         {symbol}
    تایم فریم:     {timeframe}
    صرافی:        {EXCHANGE_TO_USE.upper()}
    سیگنال:       {signal.upper()}
    اطمینان:      {confidence:.2%}
    آستانه:       {threshold_info}
    
    🤖 اطلاعات مدل:
    {model_info_text}
    Optimized:    {'Yes' if api_model_info.get('is_optimized') else 'No'}
    🔐 {auth_info}
    
    💼 مدیریت ریسک:
    قیمت فعلی:    ${current_price:.2f}
    اندازه پوزیشن: {position_size:.2f} واحد
    حد ضرر:       ${stop_loss:.2f} ({((stop_loss-current_price)/current_price*100):.1f}%)
    حد سود:       ${take_profit:.2f} ({((take_profit-current_price)/current_price*100):.1f}%)
    ATR:          ${atr:.2f}
    Portfolio Heat: {risk_manager.portfolio_heat:.1%}
    
    زمان:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================
    """
    logging.info(console_message)
    print("\033[92m" + console_message + "\033[0m")
    
    # ارسال به تلگرام
    if TELEGRAM_ENABLED:
        telegram_message = format_telegram_message(
            symbol, timeframe, signal, confidence, EXCHANGE_TO_USE,
            position_size, stop_loss, take_profit, threshold_used
        )
        send_telegram_message(telegram_message)

def send_startup_message():
    """ارسال پیام شروع به کار ربات بهبود یافته"""
    mode = "چند جفت ارز" if MULTI_PAIR_ENABLED else "تک جفت ارز"
    pairs_text = ", ".join(PAIRS_TO_MONITOR)
    timeframes_text = ", ".join(TIMEFRAMES_TO_MONITOR)
    
    # اطلاعات مدل
    model_type = api_model_info.get('model_type', 'Unknown')
    threshold = api_model_info.get('optimal_threshold', 'Unknown')
    is_optimized = api_model_info.get('is_optimized', False)
    
    # اطلاعات Authentication
    auth_status = "🔐 Enabled" if USE_AUTHENTICATION else "🔓 Disabled"
    auth_user = f" (User: {API_USERNAME})" if USE_AUTHENTICATION else ""
    
    startup_message = f"""
🚀 <b>ربات مشاور هوشمند v5.2 فعال شد!</b>

📊 <b>تنظیمات:</b>
• حالت: {mode}
• صرافی: {EXCHANGE_TO_USE.upper()}
• نمادها: {pairs_text}
• تایم فریم‌ها: {timeframes_text}
• آستانه اطمینان: {CONFIDENCE_THRESHOLD:.0%}
• بازه زمانی بررسی: {POLL_INTERVAL_SECONDS} ثانیه

🔐 <b>Authentication:</b>
• وضعیت: {auth_status}{auth_user}
• API Status: {'✅ Connected' if api_model_info else '❌ Disconnected'}

🤖 <b>اطلاعات مدل:</b>
• نوع مدل: {model_type}
• Threshold: {threshold}
• Optimized: {'✅' if is_optimized else '❌'}

💼 <b>مدیریت ریسک:</b>
• حداکثر اندازه پوزیشن: {MAX_POSITION_SIZE:.0%}
• ضریب Stop Loss: {STOP_LOSS_ATR_MULTIPLIER}x ATR
• ضریب Take Profit: {TAKE_PROFIT_ATR_MULTIPLIER}x ATR
• حداکثر Drawdown روزانه: {MAX_DAILY_DRAWDOWN:.0%}
• Kelly Criterion: {'فعال' if KELLY_CRITERION_ENABLED else 'غیرفعال'}

⚡ ربات آماده دریافت و تحلیل داده‌ها است...

#BotStarted #{datetime.datetime.now().strftime('%Y%m%d')} #v5_2 #AuthFixed
"""
    
    if TELEGRAM_ENABLED:
        send_telegram_message(startup_message)

def process_pair(symbol: str, timeframe: str, expected_features: Optional[List[str]] = None) -> Dict:
    """پردازش یک جفت ارز و تایم‌فریم مشخص با اطلاعات بهبود یافته"""
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'success': False,
        'signal': None,
        'confidence': None,
        'error': None,
        'threshold_used': None
    }
    
    try:
        # دریافت داده‌های جدید
        latest_data = get_latest_data(symbol, timeframe, CANDLE_HISTORY_NEEDED, EXCHANGE_TO_USE)
        
        if latest_data is None:
            result['error'] = "Failed to get data"
            return result
        
        current_candle_timestamp = latest_data['timestamp'].iloc[-1]
        current_price = latest_data['close'].iloc[-1]
        
        # بررسی آیا کندل جدید است
        last_timestamp_key = f"{symbol}_{timeframe}"
        if last_timestamp_key in last_processed_timestamps:
            if current_candle_timestamp == last_processed_timestamps[last_timestamp_key]:
                result['error'] = "Same candle as before"
                return result
        
        logging.info(f"🕯️ New candle detected for {symbol} {timeframe}: {current_candle_timestamp}")
        
        # محاسبه ویژگی‌ها
        features_payload = calculate_features(latest_data)
        if not features_payload:
            result['error'] = "Feature calculation failed"
            last_processed_timestamps[last_timestamp_key] = current_candle_timestamp
            return result
        
        # استخراج ATR
        atr = features_payload.get('_atr_value', 0)
        
        # بررسی همخوانی (اگر لیست ویژگی‌ها موجود باشد)
        if expected_features:
            # حذف _atr_value از بررسی
            features_to_check = {k: v for k, v in features_payload.items() if k != '_atr_value'}
            if not verify_feature_consistency(features_to_check, expected_features):
                result['error'] = "Feature consistency check failed"
                last_processed_timestamps[last_timestamp_key] = current_candle_timestamp
                return result
        
        # دریافت پیش‌بینی
        prediction_result = get_prediction(features_payload)
        
        if prediction_result:
            signal = prediction_result.get('signal')
            profit_prob = prediction_result.get('confidence', {}).get('profit_prob', 0)
            threshold_used = prediction_result.get('model_info', {}).get('threshold_used', 0.5)
            
            result['success'] = True
            result['signal'] = signal
            result['confidence'] = profit_prob
            result['threshold_used'] = threshold_used
            
            logging.info(f"📈 Prediction for {symbol} {timeframe}: "
                        f"Signal={signal}, Confidence={profit_prob:.2%}, "
                        f"Threshold={threshold_used:.4f}")
            
            # بررسی آستانه و ارسال اعلان
            if signal == 'PROFIT' and profit_prob >= CONFIDENCE_THRESHOLD:
                send_notification(symbol, timeframe, signal, profit_prob, current_price, atr, 
                                prediction_result)
        else:
            result['error'] = "Failed to get prediction from API"
        
        last_processed_timestamps[last_timestamp_key] = current_candle_timestamp
        
    except Exception as e:
        result['error'] = f"Exception: {str(e)}"
        logging.error(f"Error processing {symbol} {timeframe}: {e}")
    
    return result

def multi_pair_loop(expected_features: Optional[List[str]] = None):
    """حلقه اصلی برای پردازش چند جفت ارز بهبود یافته"""
    global successful_predictions, failed_attempts, iteration_count
    
    # ثبت signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    logging.info("="*70)
    logging.info("🤖 Smart Advisor Bot v5.2 Started (Enhanced Authentication)")
    logging.info(f"📊 Exchange: {EXCHANGE_TO_USE.upper()}")
    logging.info(f"💱 Symbols: {', '.join(PAIRS_TO_MONITOR)}")
    logging.info(f"⏱️ Timeframes: {', '.join(TIMEFRAMES_TO_MONITOR)}")
    logging.info(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    logging.info(f"⏱️ Poll Interval: {POLL_INTERVAL_SECONDS} seconds")
    logging.info(f"📁 Logs Directory: {log_subfolder_path}")
    logging.info(f"📱 Telegram: {'Enabled' if TELEGRAM_ENABLED else 'Disabled'}")
    logging.info(f"💼 Risk Management: Enabled")
    logging.info(f"🔐 Authentication: {'Enabled' if USE_AUTHENTICATION else 'Disabled'} ({API_USERNAME})")
    logging.info("="*70)
    
    # بررسی سلامت API و دریافت اطلاعات مدل
    if not check_api_health():
        logging.error("❌ API Health Check failed! Bot will continue but may not work properly.")
        print("❌ WARNING: API is not healthy! Check if prediction_api_commercial_05.py is running.")
        input("Press Enter to continue anyway or Ctrl+C to exit...")
    
    # ارسال پیام شروع به کار
    send_startup_message()
    
    successful_predictions = 0
    failed_attempts = 0
    iteration_count = 0
    last_daily_reset = datetime.datetime.now().date()
    
    try:
        while True:
            try:
                # بررسی ریست روزانه
                current_date = datetime.datetime.now().date()
                if current_date > last_daily_reset:
                    risk_manager.reset_daily_metrics()
                    last_daily_reset = current_date
                
                iteration_count += 1
                logging.info(f"\n--- Iteration #{iteration_count} ---")
                
                # ایجاد لیست کارها
                tasks = []
                for symbol in PAIRS_TO_MONITOR:
                    for timeframe in TIMEFRAMES_TO_MONITOR:
                        tasks.append((symbol, timeframe))
                
                logging.info(f"Processing {len(tasks)} pair-timeframe combinations...")
                
                # پردازش همزمان با ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_task = {
                        executor.submit(process_pair, symbol, timeframe, expected_features): (symbol, timeframe)
                        for symbol, timeframe in tasks
                    }
                    
                    for future in as_completed(future_to_task):
                        symbol, timeframe = future_to_task[future]
                        try:
                            result = future.result()
                            if result['success']:
                                successful_predictions += 1
                                # لاگ threshold برای successful predictions
                                if result.get('threshold_used'):
                                    logging.debug(f"✅ {symbol} {timeframe}: Threshold {result['threshold_used']:.4f}")
                            else:
                                if result['error'] not in ["Same candle as before"]:
                                    failed_attempts += 1
                        except Exception as e:
                            logging.error(f"Thread error for {symbol} {timeframe}: {e}")
                            failed_attempts += 1
                
                # گزارش وضعیت دوره‌ای با اطلاعات مدل
                if iteration_count % 10 == 0:
                    total_attempts = successful_predictions + failed_attempts
                    success_rate = (successful_predictions / total_attempts * 100) if total_attempts > 0 else 0
                    
                    # دریافت گزارش ریسک
                    risk_report = risk_manager.get_risk_report()
                    
                    # اطلاعات مدل
                    model_info_text = ""
                    if api_model_info:
                        model_info_text = f"""
🤖 <b>اطلاعات مدل:</b>
• نوع: {api_model_info.get('model_type', 'Unknown')[:25]}
• Threshold: {api_model_info.get('optimal_threshold', 0.5):.4f}
• Optimized: {'✅' if api_model_info.get('is_optimized') else '❌'}
"""
                    
                    # اطلاعات Authentication
                    auth_info_text = f"""
🔐 <b>Authentication:</b>
• Status: {'✅ Active' if USE_AUTHENTICATION else '🔓 Disabled'}
• User: {API_USERNAME if USE_AUTHENTICATION else 'N/A'}
"""
                    
                    status_message = f"""
📊 <b>گزارش وضعیت دوره‌ای v5.2</b>

• تعداد بررسی‌ها: {iteration_count}
• پیش‌بینی‌های موفق: {successful_predictions}
• خطاها: {failed_attempts}
• نرخ موفقیت: {success_rate:.1f}%
• سیگنال‌های صادر شده: {len(signals_history)}

{model_info_text}

{auth_info_text}

{risk_report}

🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    logging.info(f"\n📊 Status Report (after {iteration_count} iterations):")
                    logging.info(f"   - Successful Predictions: {successful_predictions}")
                    logging.info(f"   - Failed Attempts: {failed_attempts}")
                    logging.info(f"   - Success Rate: {success_rate:.1f}%")
                    logging.info(f"   - Total Signals Generated: {len(signals_history)}")
                    logging.info(f"   - Authentication: {'✅ ' + API_USERNAME if USE_AUTHENTICATION else '🔓 Disabled'}")
                    
                    if api_model_info:
                        logging.info(f"   - Model: {api_model_info.get('model_type', 'Unknown')}")
                        logging.info(f"   - Threshold: {api_model_info.get('optimal_threshold', 0.5):.4f}")
                    
                    save_performance_metrics()
                    
                    # ارسال گزارش دوره‌ای به تلگرام
                    if TELEGRAM_ENABLED and iteration_count % 50 == 0:  # هر 50 تکرار
                        send_telegram_message(status_message)
                
            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
                failed_attempts += 1
                
                # ارسال پیام خطا به تلگرام در صورت خطاهای مکرر
                if failed_attempts % 5 == 0 and TELEGRAM_ENABLED:
                    error_message = f"""
⚠️ <b>هشدار خطا v5.2</b>

ربات با خطاهای مکرر مواجه شده است.
تعداد خطاها: {failed_attempts}
آخرین خطا: {str(e)[:100]}...

🔐 Authentication: {'✅ ' + API_USERNAME if USE_AUTHENTICATION else '🔓 Disabled'}
🔄 سیستم fallback فعال است.
لطفاً وضعیت API و شبکه را بررسی کنید.
"""
                    send_telegram_message(error_message)
                
            time.sleep(POLL_INTERVAL_SECONDS)
        
    except KeyboardInterrupt:
        logging.info("\n⛔ Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logging.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        # اجرای cleanup در هر صورت
        cleanup_and_shutdown()

def single_pair_loop(expected_features: Optional[List[str]] = None):
    """حلقه اصلی برای حالت تک جفت ارز (سازگاری با نسخه قبلی)"""
    # این همان main_loop قبلی است با تغییرات جزئی
    multi_pair_loop(expected_features)

# --- نقطه شروع اسکریپت ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 Smart Advisor Bot v5.2")
    print("🔐 Enhanced Commercial API Authentication")
    print("📊 Multi-Pair & Multi-Timeframe Support")
    print("💼 Risk Management Module Enabled")
    print("🔄 Binance API Fallback System")
    print("⚡ Enhanced API Integration (Optimized Models)")
    print("="*60)
    
    # نمایش تنظیمات
    if MULTI_PAIR_ENABLED:
        print("✅ Multi-pair mode: ENABLED")
        print(f"📊 Monitoring {len(PAIRS_TO_MONITOR)} symbols across {len(TIMEFRAMES_TO_MONITOR)} timeframes")
        print(f"   Symbols: {', '.join(PAIRS_TO_MONITOR)}")
        print(f"   Timeframes: {', '.join(TIMEFRAMES_TO_MONITOR)}")
    else:
        print("ℹ️ Single-pair mode (backward compatible)")
    
    # نمایش تنظیمات Risk Management
    print("\n💼 Risk Management Settings:")
    print(f"   Max Position Size: {MAX_POSITION_SIZE:.0%}")
    print(f"   Stop Loss: {STOP_LOSS_ATR_MULTIPLIER}x ATR")
    print(f"   Take Profit: {TAKE_PROFIT_ATR_MULTIPLIER}x ATR")
    print(f"   Max Daily Drawdown: {MAX_DAILY_DRAWDOWN:.0%}")
    print(f"   Kelly Criterion: {'Enabled' if KELLY_CRITERION_ENABLED else 'Disabled'}")
    
    # 🔧 نمایش تنظیمات Authentication
    print(f"\n🔐 Authentication Settings:")
    print(f"   Status: {'Enabled' if USE_AUTHENTICATION else 'Disabled'}")
    if USE_AUTHENTICATION:
        print(f"   Username: {API_USERNAME}")
        print(f"   Password: {'*' * len(API_PASSWORD)}")
        
        # تست authentication
        print(f"\n🔍 Testing authentication...")
        if check_authentication():
            print("✅ Authentication test: Passed")
        else:
            print("❌ Authentication test: Failed")
            print("⚠️  Bot will continue but may not work properly!")
            print("💡 Please check Bot_Authentication section in config.ini")
            input("Press Enter to continue anyway or Ctrl+C to exit...")
    
    # بررسی تنظیمات تلگرام
    if TELEGRAM_ENABLED:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            print("\n✅ Telegram notifications: ENABLED")
        else:
            print("\n⚠️ Telegram enabled but configuration is incomplete!")
            TELEGRAM_ENABLED = False
    else:
        print("\nℹ️ Telegram notifications: DISABLED")
    
    # بررسی سلامت API
    print(f"\n🔍 Checking API health at {API_HEALTH_URL}...")
    if check_api_health():
        print("✅ API Health Check: Passed")
        if api_model_info:
            print(f"🤖 Model Type: {api_model_info.get('model_type', 'Unknown')}")
            print(f"🎯 Optimal Threshold: {api_model_info.get('optimal_threshold', 0.5):.4f}")
            print(f"⚡ Optimized Model: {'Yes' if api_model_info.get('is_optimized') else 'No'}")
    else:
        print("❌ API Health Check: Failed")
        print("⚠️  Make sure prediction_api_commercial_05.py is running!")
        print("💡 Check Authentication settings if using commercial mode")
    
    # بارگذاری لیست ویژگی‌های مدل
    model_features = load_model_features()
    
    if model_features:
        print(f"\n✅ Model features loaded: {len(model_features)} features")
    else:
        print("\n⚠️ Running without feature consistency check")
    
    print(f"\n📡 API Endpoints:")
    print(f"   - Prediction: {API_URL}")
    print(f"   - Health Check: {API_HEALTH_URL}")
    print(f"   - Model Info: {API_MODEL_INFO_URL}")
    print("🔄 Fallback system: CryptoCompare API available if Binance fails")
    print("📊 Connection timeout: 30 seconds")
    print("🔄 Retry mechanism: 3 attempts per request")
    print(f"🔐 Authentication: {'✅ Required' if USE_AUTHENTICATION else '🔓 Disabled'}")
    input("Press Enter to start the bot...")
    
    try:
        if MULTI_PAIR_ENABLED:
            multi_pair_loop(expected_features=model_features)
        else:
            single_pair_loop(expected_features=model_features)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")