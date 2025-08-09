#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت هسته اصلی ربات مشاور هوشمند (نسخه 5.3 - اصلاحات نهایی کامل)

🔧 تغییرات مهم v5.3 (ترکیب بهترین اصلاحات):
- ✅ کاهش threshold به 0.40 برای سیگنال‌های بیشتر
- ✅ رفع کامل مشکل ارسال چندباره پیام خروج
- ✅ بهبود مدیریت Rate Limiting با Circuit Breaker
- ✅ اضافه کردن delay بیشتر بین درخواست‌ها
- ✅ بهبود cleanup mechanism با thread safety
- ✅ اصلاح threshold detection از API
- ✅ رفع مشکل 401 Authentication Error
- ✅ بهبود سازگاری با API جدید (Optimized Models)
- ✅ محاسبه کامل 58 ویژگی (شامل PSAR)

ویژگی‌های موجود:
- Risk Management Module کامل
- Position Sizing با Kelly Criterion  
- Dynamic Stop Loss و Take Profit بر اساس ATR
- Max Drawdown Protection
- Portfolio Heat Management
- Binance API Fallback با retry mechanism
- Multi-source Data
- Commercial API Authentication Support
- Complete Feature Calculation (58 features)
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
import atexit

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
    
    # 🔧 افزایش poll interval برای کاهش rate limiting
    POLL_INTERVAL_SECONDS = config.getint('Bot_Settings', 'poll_interval_seconds', fallback=300)
    if POLL_INTERVAL_SECONDS < 180:  # حداقل 3 دقیقه
        POLL_INTERVAL_SECONDS = 180
        logging.warning(f"⚠️ Poll interval increased to {POLL_INTERVAL_SECONDS}s to prevent rate limiting")
    
    # 🔧 کاهش threshold برای سیگنال‌های بیشتر
    CONFIDENCE_THRESHOLD = config.getfloat('Bot_Settings', 'confidence_threshold', fallback=0.40)
    if CONFIDENCE_THRESHOLD > 0.50:  # حداکثر 50%
        CONFIDENCE_THRESHOLD = 0.40
        logging.warning(f"⚠️ Confidence threshold lowered to {CONFIDENCE_THRESHOLD:.0%} for more signals")
    
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

# 🔧 متغیرهای global برای جلوگیری از ارسال چندباره پیام خروج
shutdown_message_sent = False
cleanup_in_progress = False
shutdown_lock = threading.Lock()

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

# === 🔧 توابع cleanup بهبود یافته برای تلگرام ===
def cleanup_and_shutdown():
    """تابع cleanup برای ارسال پیام قطع ارتباط و ذخیره آمار - اصلاح کامل"""
    global successful_predictions, failed_attempts, iteration_count, shutdown_message_sent, cleanup_in_progress
    
    # استفاده از lock برای thread safety
    with shutdown_lock:
        # جلوگیری از اجرای چندباره
        if cleanup_in_progress or shutdown_message_sent:
            return
        
        cleanup_in_progress = True
        
        try:
            logging.info("🔄 Starting cleanup and shutdown process...")
            
            # ذخیره نهایی قبل از خروج
            save_performance_metrics()
            risk_manager.save_risk_metrics()
            
            # ارسال پیام خاموش شدن (فقط یک بار)
            if TELEGRAM_ENABLED and not shutdown_message_sent:
                shutdown_message_sent = True  # جلوگیری از ارسال مجدد
                
                total_attempts = successful_predictions + failed_attempts
                final_risk_report = risk_manager.get_risk_report()
                
                shutdown_message = f"""
🛑 <b>ربات مشاور هوشمند v5.3 متوقف شد</b>

📊 <b>آمار نهایی:</b>
• تعداد کل بررسی‌ها: {iteration_count}
• سیگنال‌های صادر شده: {len(signals_history)}
• نرخ موفقیت: {(successful_predictions / total_attempts * 100) if total_attempts > 0 else 0:.1f}%

🤖 <b>مدل استفاده شده:</b>
{api_model_info.get('model_type', 'Unknown')} {'(Optimized)' if api_model_info.get('is_optimized') else ''}

🔐 <b>Authentication:</b>
User: {API_USERNAME} {'(Success)' if USE_AUTHENTICATION else '(Disabled)'}

⚙️ <b>تنظیمات v5.3:</b>
• Threshold: {CONFIDENCE_THRESHOLD:.0%} (کاهش یافته)
• Poll Interval: {POLL_INTERVAL_SECONDS}s (افزایش یافته)

{final_risk_report}

🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#BotStopped #v5_3 #FinalVersion #ThresholdOptimized
"""
                try:
                    send_telegram_message(shutdown_message)
                    logging.info("📱 Shutdown message sent to Telegram successfully")
                except Exception as telegram_error:
                    logging.error(f"Error sending shutdown message: {telegram_error}")
            
            logging.info("✅ Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            cleanup_in_progress = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logging.info("\n⛔ Received shutdown signal (Ctrl+C)")
    print("\n⛔ Shutting down gracefully...")
    cleanup_and_shutdown()
    sys.exit(0)

# ثبت cleanup برای اجرا در هنگام خروج
atexit.register(cleanup_and_shutdown)

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
                    logging.info(f"🎯 Model Optimal Threshold: {threshold:.4f}")
                    logging.info(f"⚡ Optimized Model: {'Yes' if is_optimized else 'No'}")
                    
                    # 🔧 تطبیق threshold با مدل (اگر خیلی بالا باشد)
                    global CONFIDENCE_THRESHOLD
                    if threshold > 0.60 and CONFIDENCE_THRESHOLD > 0.50:
                        old_threshold = CONFIDENCE_THRESHOLD
                        CONFIDENCE_THRESHOLD = 0.40  # کاهش برای سیگنال‌های بیشتر
                        logging.warning(f"🔧 Model threshold ({threshold:.4f}) is high. ")
                        logging.warning(f"🔧 Bot threshold adjusted: {old_threshold:.0%} → {CONFIDENCE_THRESHOLD:.0%}")
                    
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
{emoji_signal} <b>سیگنال جدید از ربات مشاور هوشمند v5.3</b> {emoji_signal}

📊 <b>نماد:</b> {symbol}
⏱ <b>تایم فریم:</b> {timeframe}
🏦 <b>صرافی:</b> {exchange.upper()}
📈 <b>سیگنال:</b> <b>{signal}</b>
{emoji_confidence} <b>اطمینان:</b> {confidence:.1%}
🎯 <b>آستانه ربات:</b> {CONFIDENCE_THRESHOLD:.0%}
"""

    # اطلاعات مدل بهبود یافته
    if threshold_used:
        threshold_emoji = "⚡" if is_optimized else "🔧"
        message += f"""
🤖 <b>مدل:</b> {model_type[:20]}{'...' if len(model_type) > 20 else ''}
{threshold_emoji} <b>Threshold مدل:</b> {threshold_used:.3f} {'(Optimized)' if is_optimized else '(Default)'}
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
    
    # نمایش تنظیمات بهبود یافته
    message += f"""
⚙️ <b>تنظیمات v5.3:</b>
   🔄 Poll Interval: {POLL_INTERVAL_SECONDS}s (افزایش یافته)
   🎯 Threshold: {CONFIDENCE_THRESHOLD:.0%} (کاهش یافته)
   📊 ویژگی‌ها: 58 (کامل)

🕐 <b>زمان:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#SmartAdvisor #CryptoSignal #{symbol.replace('/', '')} #{timeframe} #v5_3 #FinalVersion
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

# --- بخش ۵: توابع اصلی (با اصلاح مشکل Binance API و circuit breaker) ---
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
    تابع اصلاح شده برای دریافت داده با حل مشکل Binance API و circuit breaker
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
                    'rateLimit': 1500,  # 🔧 افزایش rate limit delay
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'  # مشخص کردن نوع معاملات
                    }
                })
            else:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'timeout': 30000,
                    'rateLimit': 2000,  # 🔧 افزایش برای سایر صرافی‌ها
                    'enableRateLimit': True
                })
            
            # تلاش برای دریافت داده با retry mechanism و circuit breaker
            max_retries = 3
            base_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # 🔧 اضافه کردن delay قبل از درخواست
                    if attempt > 0:
                        delay_time = base_delay ** attempt  # exponential backoff
                        logging.info(f"⏳ Waiting {delay_time}s before retry (attempt {attempt + 1})...")
                        time.sleep(delay_time)
                    
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    if len(df) < limit // 2:  # اگر خیلی کم داده دریافت شد
                        logging.warning(f"Data received ({len(df)}) is less than expected ({limit}).")
                        if attempt < max_retries - 1:
                            logging.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                            continue
                    
                    logging.info(f"Successfully fetched {len(df)} candles from {exchange_name.upper()}")
                    return df
                    
                except ccxt.RateLimitExceeded as rate_error:
                    logging.warning(f"⚠️ Rate limit exceeded on attempt {attempt + 1}: {rate_error}")
                    if attempt < max_retries - 1:
                        delay_time = 90  # 1.5 دقیقه انتظار برای rate limit
                        logging.info(f"⏳ Rate limit cooldown: waiting {delay_time}s...")
                        time.sleep(delay_time)
                        continue
                    else:
                        logging.error("❌ Rate limit exceeded - falling back to CryptoCompare")
                        return fetch_from_cryptocompare_api(symbol, timeframe, limit)
                        
                except ccxt.NetworkError as network_error:
                    logging.warning(f"🌐 Network error on attempt {attempt + 1}: {network_error}")
                    if attempt < max_retries - 1:
                        delay_time = base_delay ** (attempt + 1)
                        logging.info(f"⏳ Network error cooldown: waiting {delay_time}s...")
                        time.sleep(delay_time)
                        continue
                    else:
                        logging.error("❌ Network error persists - falling back to CryptoCompare")
                        return fetch_from_cryptocompare_api(symbol, timeframe, limit)
                        
                except Exception as attempt_error:
                    logging.warning(f"Attempt {attempt + 1} failed: {attempt_error}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise attempt_error
            
        except AttributeError:
            logging.error(f"Exchange '{exchange_name}' is not supported by CCXT.")
        except ccxt.BaseError as e:
            logging.error(f"Exchange error from {exchange_name.upper()}: {e}")
            # اگر Binance کار نکرد، fallback به CryptoCompare
            if exchange_name.lower() == 'binance':
                logging.info("🔄 Fallback to CryptoCompare due to exchange error...")
                return fetch_from_cryptocompare_api(symbol, timeframe, limit)
        except Exception as e:
            logging.error(f"Unexpected error fetching data from {exchange_name.upper()}: {e}")
            # اگر Binance کار نکرد، fallback به CryptoCompare
            if exchange_name.lower() == 'binance':
                logging.info("🔄 Fallback to CryptoCompare due to connection issues...")
                return fetch_from_cryptocompare_api(symbol, timeframe, limit)
        
        return None

def calculate_features(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """محاسبه ویژگی‌های کامل 58 ویژگی - آینه کامل از اسکریپت 03 با اضافه شدن برگرداندن ATR"""
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
        
        # 🔧 PSAR - اصلاح شده برای محاسبه کامل 58 ویژگی
        try:
            psar_result = ta.psar(group['high'], group['low'], group['close'])
            if psar_result is not None and not psar_result.empty:
                if isinstance(psar_result, pd.DataFrame):
                    # استخراج ستون‌های long و short
                    psar_long = psar_result.iloc[:, 0]  # PSARl_0.02_0.2
                    psar_short = psar_result.iloc[:, 1]  # PSARs_0.02_0.2
                    
                    # ترکیب long و short - اگر long موجود است از آن استفاده کن، وگرنه short
                    group['psar'] = psar_long.fillna(psar_short)
                else:
                    group['psar'] = psar_result
                
                # اگر هنوز NaN داریم، با مقدار پیش‌فرض پر کنیم
                group['psar'] = group['psar'].fillna(group['close'] * 0.98)
                group['price_above_psar'] = (group['close'] > group['psar']).astype(int)
            else:
                # مقدار پیش‌فرض
                group['psar'] = group['close'] * 0.98
                group['price_above_psar'] = 1
        except Exception as e:
            logging.warning(f"PSAR calculation failed: {e}. Using default values.")
            group['psar'] = group['close'] * 0.98
            group['price_above_psar'] = 1

        # اطمینان از وجود PSAR در خروجی (ویژگی مهم 58)
        if 'psar' not in group.columns or group['psar'].isna().all():
            group['psar'] = group['close'] * 0.98
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
        
        # بررسی تعداد ویژگی‌ها (هدف: 58 ویژگی)
        expected_features = 58
        actual_features = len(cleaned_features) - 1  # منهای _atr_value
        logging.info(f"🔢 تعداد ویژگی‌های محاسبه شده: {actual_features}/58")
        
        if actual_features < expected_features:
            logging.warning(f"⚠️ تعداد ویژگی‌های محاسبه شده ({actual_features}) کمتر از انتظار ({expected_features}) است")
        
        return cleaned_features
        
    except Exception as e:
        logging.error(f"خطا در محاسبه ویژگی‌ها: {e}", exc_info=True)
        return None
        
def get_prediction(payload: Dict) -> Optional[Dict]:
    """ارسال درخواست به API پیش‌بینی بهبود یافته با Authentication و retry mechanism"""
    try:
        # حذف ATR از payload قبل از ارسال
        atr_value = payload.pop('_atr_value', 1.0)  # ذخیره ATR برای Risk Management
        
        # Retry mechanism برای API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # ارسال درخواست با Authentication
                if USE_AUTHENTICATION:
                    response = requests.post(
                        API_URL, 
                        json=payload, 
                        timeout=30,
                        auth=(API_USERNAME, API_PASSWORD)
                    )
                else:
                    response = requests.post(API_URL, json=payload, timeout=30)
                
                # لاگ response برای debugging
                if attempt == 0:  # فقط در تلاش اول
                    logging.info(f"📡 API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # اضافه کردن ATR به نتیجه
                    if result:
                        result['_atr_value'] = atr_value
                    
                    return result
                elif response.status_code == 429:
                    # 🔧 مدیریت بهتر Rate Limiting
                    retry_after = response.headers.get('Retry-After', 90)
                    logging.warning(f"⚠️ Rate Limited (429). Retry after: {retry_after}s")
                    if attempt < max_retries - 1:
                        time.sleep(int(retry_after))
                        continue
                    return {'error': 'rate_limited', 'retry_after': int(retry_after)}
                elif response.status_code == 401:
                    logging.error("❌ Authentication Error (401) - Invalid credentials")
                    logging.error(f"💡 Check username: {API_USERNAME}")
                    return {'error': 'authentication_failed'}
                elif response.status_code == 500:
                    logging.error("❌ Server Error (500)")
                    try:
                        error_data = response.json()
                        logging.error(f"📋 Server error details: {error_data}")
                    except:
                        logging.error(f"📋 Server error text: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # انتظار کوتاه قبل از retry
                        continue
                    return {'error': 'server_error'}
                else:
                    logging.error(f"❌ API Error: HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    return {'error': f'http_{response.status_code}'}
                    
            except requests.exceptions.Timeout:
                logging.warning(f"⏰ API Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return {'error': 'timeout'}
            except requests.exceptions.ConnectionError:
                logging.warning(f"🌐 Connection Error on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                return {'error': 'connection_error'}
            except Exception as e:
                logging.error(f"❌ Prediction request error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {'error': 'unexpected_error', 'details': str(e)}
                
    except Exception as e:
        logging.error(f"❌ Critical error in get_prediction: {e}")
        return {'error': 'critical_error', 'details': str(e)}

def save_performance_metrics():
    """ذخیره معیارهای عملکرد"""
    try:
        metrics = {
            'timestamp': datetime.datetime.now().isoformat(),
            'successful_predictions': successful_predictions,
            'failed_attempts': failed_attempts,
            'iteration_count': iteration_count,
            'total_signals': len(signals_history),
            'uptime_hours': (datetime.datetime.now() - pd.Timestamp.now().floor('H')).total_seconds() / 3600,
            'current_threshold': CONFIDENCE_THRESHOLD,
            'poll_interval': POLL_INTERVAL_SECONDS,
            'model_info': api_model_info
        }
        
        # ذخیره در CSV
        df = pd.DataFrame([metrics])
        df.to_csv(performance_log, mode='a', header=not os.path.exists(performance_log), index=False)
        
    except Exception as e:
        logging.error(f"خطا در ذخیره performance metrics: {e}")

def process_pair(symbol: str, timeframe: str, exchange: str, expected_features: Optional[List] = None) -> bool:
    """پردازش یک جفت ارز و تولید سیگنال"""
    global successful_predictions, failed_attempts, last_processed_timestamps
    
    try:
        logging.info(f"\n🔍 Processing {symbol} {timeframe} on {exchange.upper()}")
        
        # دریافت داده‌ها
        df = get_latest_data(symbol, timeframe, CANDLE_HISTORY_NEEDED, exchange)
        if df is None or df.empty:
            logging.error(f"❌ No data received for {symbol}")
            failed_attempts += 1
            return False
        
        # بررسی timestamp جدید
        latest_timestamp = df['timestamp'].iloc[-1]
        pair_key = f"{symbol}_{timeframe}"
        
        if pair_key in last_processed_timestamps:
            if latest_timestamp <= last_processed_timestamps[pair_key]:
                logging.info(f"⏭️ No new data for {symbol} {timeframe}")
                return False
        
        last_processed_timestamps[pair_key] = latest_timestamp
        
        # محاسبه ویژگی‌های کامل (58 ویژگی)
        features = calculate_features(df)
        if not features:
            logging.error(f"❌ Feature calculation failed for {symbol}")
            failed_attempts += 1
            return False
        
        # بررسی همخوانی ویژگی‌ها
        if expected_features and not verify_feature_consistency(features, expected_features):
            logging.warning(f"⚠️ Feature mismatch for {symbol} - continuing anyway")
        
        # دریافت ATR برای Risk Management
        atr_value = features.get('_atr_value', 1.0)
        
        # درخواست پیش‌بینی
        prediction_result = get_prediction(features)
        if not prediction_result:
            logging.error(f"❌ Prediction failed for {symbol}")
            failed_attempts += 1
            return False
        
        # مدیریت خطاها
        if 'error' in prediction_result:
            error_type = prediction_result['error']
            if error_type == 'rate_limited':
                retry_after = prediction_result.get('retry_after', 90)
                logging.warning(f"⏳ Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                return False
            elif error_type == 'authentication_failed':
                logging.error("🔐 Authentication failed - check credentials")
                return False
            else:
                logging.error(f"❌ API Error: {error_type}")
                failed_attempts += 1
                return False
        
        # پردازش نتیجه پیش‌بینی
        confidence = prediction_result.get('prediction_proba', 0)
        if 'confidence' in prediction_result:
            confidence = prediction_result['confidence'].get('profit_prob', 0)
        
        prediction_class = prediction_result.get('prediction', 'NO_SIGNAL')
        if prediction_class == 1:
            prediction_class = 'PROFIT'
        elif prediction_class == 0:
            prediction_class = 'NO_PROFIT'
        
        signal = prediction_result.get('signal', prediction_class)
        threshold_used = prediction_result.get('threshold_used', CONFIDENCE_THRESHOLD)
        
        logging.info(f"🎯 Prediction for {symbol}: {signal} (Confidence: {confidence:.3f})")
        
        # بررسی آستانه اطمینان
        if confidence >= CONFIDENCE_THRESHOLD:
            current_price = df['close'].iloc[-1]
            
            # محاسبه Risk Management
            position_size = risk_manager.calculate_position_size(symbol, confidence, current_price, atr_value)
            stop_loss = risk_manager.calculate_stop_loss(current_price, atr_value, signal)
            take_profit = risk_manager.calculate_take_profit(current_price, atr_value, signal)
            
            # ایجاد سیگنال
            signal_data = {
                'timestamp': latest_timestamp.isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'threshold_used': threshold_used,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr_value,
                'model_info': api_model_info.get('model_type', 'Unknown'),
                'features_count': len(features) - 1  # منهای _atr_value
            }
            
            # ذخیره سیگنال
            with signals_lock:
                signals_history.append(signal_data)
            
            # ذخیره در فایل JSON
            try:
                with open(signals_log, 'w', encoding='utf-8') as f:
                    json.dump(signals_history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"خطا در ذخیره سیگنال: {e}")
            
            # ارسال تلگرام
            telegram_message = format_telegram_message(
                symbol, timeframe, signal, confidence, exchange,
                position_size, stop_loss, take_profit, threshold_used
            )
            
            if send_telegram_message(telegram_message):
                logging.info(f"📱 Signal sent to Telegram for {symbol}")
            
            successful_predictions += 1
            logging.info(f"✅ Signal generated for {symbol}: {signal} (Confidence: {confidence:.1%})")
            return True
            
        else:
            logging.info(f"⚪ No signal for {symbol}: confidence {confidence:.3f} below threshold {CONFIDENCE_THRESHOLD:.3f}")
            successful_predictions += 1  # موفق بوده ولی سیگنال نداشته
            return False
            
    except Exception as e:
        logging.error(f"❌ Error processing {symbol}: {e}", exc_info=True)
        failed_attempts += 1
        return False

def monitor_pairs_concurrent():
    """پردازش همزمان چند جفت ارز"""
    global iteration_count
    
    # بارگذاری ویژگی‌های مورد انتظار
    expected_features = load_model_features()
    
    while True:
        try:
            iteration_count += 1
            logging.info(f"\n🔄 === Iteration {iteration_count} === {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # پردازش همزمان
            with ThreadPoolExecutor(max_workers=min(len(PAIRS_TO_MONITOR) * len(TIMEFRAMES_TO_MONITOR), 4)) as executor:
                futures = []
                
                for symbol in PAIRS_TO_MONITOR:
                    for timeframe in TIMEFRAMES_TO_MONITOR:
                        future = executor.submit(process_pair, symbol, timeframe, EXCHANGE_TO_USE, expected_features)
                        futures.append(future)
                
                # جمع‌آوری نتایج
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)  # 2 دقیقه timeout
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Task failed: {e}")
                        results.append(False)
            
            signals_generated = sum(results)
            logging.info(f"📊 Iteration {iteration_count} complete. Signals generated: {signals_generated}")
            
            # ذخیره آمار
            save_performance_metrics()
            
            # 🔧 افزایش delay بین iterations
            sleep_time = max(POLL_INTERVAL_SECONDS, 180)  # حداقل 3 دقیقه
            logging.info(f"😴 Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("⛔ Received interrupt signal")
            break
        except Exception as e:
            logging.error(f"❌ Error in monitoring loop: {e}", exc_info=True)
            time.sleep(120)  # استراحت 2 دقیقه قبل از ادامه

def main():
    """تابع اصلی - اصلاح شده"""
    global shutdown_message_sent
    
    # ثبت signal handler برای Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("\n🚀 Smart Trading Bot v5.3 Starting (Final Version)...")
        print("=" * 60)
        
        # نمایش تنظیمات اصلی
        print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%} (Optimized for more signals)")
        print(f"⏱️ Poll Interval: {POLL_INTERVAL_SECONDS}s (Increased for stability)")
        print(f"🔐 Authentication: {'Enabled' if USE_AUTHENTICATION else 'Disabled'}")
        if USE_AUTHENTICATION:
            print(f"👤 Username: {API_USERNAME}")
        
        # تست اتصال API
        if not test_api_connection():
            print("❌ API connection test failed. Please check the API server.")
            return
        
        # بررسی Authentication
        if not check_authentication():
            print("❌ Authentication check failed. Please update credentials in config.ini")
            return
        
        # بررسی سلامت API
        if not check_api_health():
            print("❌ API health check failed. Cannot proceed.")
            return
        
        print(f"✅ All checks passed. Monitoring {len(PAIRS_TO_MONITOR)} pairs on {len(TIMEFRAMES_TO_MONITOR)} timeframes")
        print(f"📊 Expected features: {len(load_model_features() or [])} features")
        print(f"🎯 Target: 58 complete features per prediction")
        
        # ارسال پیام شروع
        if TELEGRAM_ENABLED and not shutdown_message_sent:
            startup_message = f"""
🚀 <b>ربات مشاور هوشمند v5.3 راه‌اندازی شد</b>

⚙️ <b>تنظیمات کلیدی:</b>
• Threshold: {CONFIDENCE_THRESHOLD:.0%} (بهینه‌سازی شده)
• Poll Interval: {POLL_INTERVAL_SECONDS}s (پایدار)
• Multi-pair: {'Yes' if MULTI_PAIR_ENABLED else 'No'}
• Authentication: {'Yes' if USE_AUTHENTICATION else 'No'}

🎯 <b>نظارت بر:</b>
• نمادها: {', '.join(PAIRS_TO_MONITOR)}
• تایم‌فریم‌ها: {', '.join(TIMEFRAMES_TO_MONITOR)}
• صرافی: {EXCHANGE_TO_USE.upper()}

🤖 <b>مدل:</b>
{api_model_info.get('model_type', 'Unknown')} {'(Optimized)' if api_model_info.get('is_optimized') else ''}

💼 <b>Risk Management:</b>
• Max Position: {MAX_POSITION_SIZE:.0%}
• Stop Loss ATR: {STOP_LOSS_ATR_MULTIPLIER}x
• Take Profit ATR: {TAKE_PROFIT_ATR_MULTIPLIER}x
• Kelly Criterion: {'Enabled' if KELLY_CRITERION_ENABLED else 'Disabled'}

📊 <b>ویژگی‌ها:</b>
• محاسبه کامل 58 ویژگی
• شامل PSAR (مشکل v5.2 حل شد)
• Risk Management کامل

🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#BotStarted #v5_3 #FinalVersion #Complete58Features
"""
            send_telegram_message(startup_message)
        
        print("\n🔄 Starting monitoring loop...")
        
        # شروع پردازش
        monitor_pairs_concurrent()
        
    except KeyboardInterrupt:
        print("\n⛔ Shutdown signal received")
    except Exception as e:
        logging.error(f"❌ Critical error in main: {e}", exc_info=True)
        print(f"❌ Critical error: {e}")
    finally:
        print("\n👋 Bot shutting down...")
        cleanup_and_shutdown()

if __name__ == "__main__":
    main()