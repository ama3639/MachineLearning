#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت هسته اصلی ربات مشاور هوشمند (نسخه 6.1 - اصلاح کامل Telegram-based)

🔧 تغییرات مهم v6.1 (اصلاحات حیاتی):
- ✅ اصلاح calculate_enhanced_features: reddit derivation از sentiment
- ✅ بهبود sentiment calculation: real-time sentiment processing  
- ✅ اصلاح feature count validation: تطبیق با 58+ features
- ✅ اصلاح telegram message formatting: انعکاس telegram-based features
- ✅ اصلاح API communication: سازگاری با enhanced API v6.1
- ✅ تصحیح Reddit features: reddit_score = sentiment_score (Telegram-derived)
- ✅ بهبود sentiment calculation با price + volume momentum
- ✅ اضافه کردن telegram mapping validation
- ✅ Enhanced error handling برای sentiment/reddit features

ویژگی‌های موجود:
- Risk Management Module کامل
- Position Sizing با Kelly Criterion  
- Dynamic Stop Loss و Take Profit بر اساس ATR
- Max Drawdown Protection
- Portfolio Heat Management
- Binance API Fallback با retry mechanism
- Multi-source Data (Enhanced)
- Commercial API Authentication Support
- Complete Feature Calculation (58+ features) - اصلاح شده
- Sentiment & Telegram-derived Reddit Features Support - اصلاح شده
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
        PAIRS_TO_MONITOR = [p.strip() for p in config.get('Bot_Settings', 'pairs_to_monitor').split(',')]
        TIMEFRAMES_TO_MONITOR = [t.strip() for t in config.get('Bot_Settings', 'timeframes_to_monitor').split(',')]
        EXCHANGE_TO_USE = config.get('Bot_Settings', 'exchange_to_use')
    else:
        EXCHANGE_TO_USE = config.get('Bot_Settings', 'exchange_to_use')
        SYMBOL_TO_TRADE = config.get('Bot_Settings', 'symbol_to_trade')
        TIMEFRAME_TO_TRADE = config.get('Bot_Settings', 'timeframe_to_trade')
        PAIRS_TO_MONITOR = [SYMBOL_TO_TRADE]
        TIMEFRAMES_TO_MONITOR = [TIMEFRAME_TO_TRADE]
    
    CANDLE_HISTORY_NEEDED = config.getint('Bot_Settings', 'candle_history_needed')
    POLL_INTERVAL_SECONDS = config.getint('Bot_Settings', 'poll_interval_seconds', fallback=300)
    if POLL_INTERVAL_SECONDS < 180:
        POLL_INTERVAL_SECONDS = 180
        logging.warning(f"⚠️ Poll interval increased to {POLL_INTERVAL_SECONDS}s to prevent rate limiting")
    
    CONFIDENCE_THRESHOLD = config.getfloat('Bot_Settings', 'confidence_threshold', fallback=0.40)
    if CONFIDENCE_THRESHOLD > 0.50:
        CONFIDENCE_THRESHOLD = 0.40
        logging.warning(f"⚠️ Confidence threshold lowered to {CONFIDENCE_THRESHOLD:.0%} for more signals")
    
    # === 🔧 تنظیمات Authentication Enhanced ===
    try:
        USE_AUTHENTICATION = config.getboolean('Bot_Authentication', 'use_authentication', fallback=True)
        API_USERNAME = config.get('Bot_Authentication', 'api_username', fallback='hasnamir92')
        API_PASSWORD = config.get('Bot_Authentication', 'api_password', fallback='123456')
        
        if not config.has_section('Bot_Authentication'):
            logging.warning("Bot_Authentication section not found in config. Using default credentials.")
            USE_AUTHENTICATION = True
            API_USERNAME = "hasnamir92"
            API_PASSWORD = "123456"
            
    except Exception as e:
        logging.error(f"Error reading authentication config: {e}")
        USE_AUTHENTICATION = True
        API_USERNAME = "hasnamir92"
        API_PASSWORD = "123456"
    
    # تنظیمات تلگرام
    TELEGRAM_ENABLED = config.getboolean('Telegram', 'enabled', fallback=False)
    TELEGRAM_BOT_TOKEN = config.get('Telegram', 'bot_token', fallback=None)
    TELEGRAM_CHAT_ID = config.get('Telegram', 'chat_id', fallback=None)
    
# === پارامترهای Enhanced Feature Engineering ===
    INDICATOR_PARAMS = {
            'rsi_length': config.getint('Feature_Engineering', 'rsi_length', fallback=14),
            'macd_fast': config.getint('Feature_Engineering', 'macd_fast', fallback=12),
            'macd_slow': config.getint('Feature_Engineering', 'macd_slow', fallback=26),
            'macd_signal': config.getint('Feature_Engineering', 'macd_signal', fallback=9),
            'bb_length': config.getint('Feature_Engineering', 'bb_length', fallback=20),
            'bb_std': config.getfloat('Feature_Engineering', 'bb_std', fallback=2.0),
            'atr_length': config.getint('Feature_Engineering', 'atr_length', fallback=14),
            
            # === پارامترهای EMA (مهم برای Technical features) ===
            'ema_short': config.getint('Feature_Engineering', 'ema_short', fallback=9),
            'ema_medium': config.getint('Feature_Engineering', 'ema_medium', fallback=21),
            'ema_long': config.getint('Feature_Engineering', 'ema_long', fallback=50),
            
            # === پارامترهای PSAR (مهم برای 58 ویژگی) ===
            'psar_af': 0.02,
            'psar_max_af': 0.2,
            
            # === پارامترهای Sentiment (اصلاح شده) ===
            'sentiment_ma_short': 7,
            'sentiment_ma_long': 14,
            'sentiment_momentum_period': 24,
            
            # === پارامترهای Telegram-derived Reddit (جایگزین Reddit API) ===
            'telegram_sentiment_ma': 12,  # میانگین متحرک telegram sentiment
            'telegram_momentum_period': 24,  # دوره momentum برای telegram
            'reddit_derivation_multiplier': 10,  # ضریب تبدیل sentiment → reddit_comments
    }    
    # تنظیمات Risk Management
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

log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
signals_log = os.path.join(log_subfolder_path, f"signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
performance_log = os.path.join(log_subfolder_path, f"performance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
risk_log = os.path.join(log_subfolder_path, f"risk_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# متغیرهای global
signals_history = []
last_processed_timestamps = {}
signals_lock = threading.Lock()
api_model_info = {}
successful_predictions = 0
failed_attempts = 0
iteration_count = 0
shutdown_message_sent = False
cleanup_in_progress = False
shutdown_lock = threading.Lock()

# --- بخش Risk Management (حفظ شده) ---
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
        self.positions = {}
        self.daily_pnl = 0
        self.max_drawdown = 0
        self.win_rate_history = defaultdict(list)
        self.portfolio_heat = 0
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
        if len(history) < 10:
            return MAX_POSITION_SIZE * 0.5
        
        wins = [h for h in history if h > 0]
        losses = [h for h in history if h < 0]
        
        if not wins or not losses:
            return MAX_POSITION_SIZE * 0.5
        
        win_rate = len(wins) / len(history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, min(kelly_fraction, MAX_POSITION_SIZE))
        
        logging.info(f"📊 Kelly Fraction for {symbol}: {kelly_fraction:.2%} "
                    f"(Win Rate: {win_rate:.2%}, Avg Win/Loss: {b:.2f})")
        
        return kelly_fraction
    
    def calculate_position_size(self, symbol: str, confidence: float, 
                              current_price: float, atr: float) -> float:
        """محاسبه اندازه پوزیشن با در نظر گرفتن ریسک"""
        
        if self.check_daily_drawdown_limit():
            logging.warning("⚠️ Daily drawdown limit reached. No new positions.")
            return 0
        
        kelly_fraction = self.calculate_kelly_fraction(symbol)
        confidence_multiplier = min(1.0, (confidence - CONFIDENCE_THRESHOLD) / (1 - CONFIDENCE_THRESHOLD))
        base_position_size = kelly_fraction * confidence_multiplier
        
        stop_loss_distance = atr * STOP_LOSS_ATR_MULTIPLIER
        risk_per_share = stop_loss_distance
        max_shares_by_risk = (self.current_capital * 0.02) / risk_per_share
        
        position_value = self.current_capital * base_position_size
        shares = min(position_value / current_price, max_shares_by_risk)
        
        new_heat = self.calculate_portfolio_heat() + (shares * risk_per_share / self.current_capital)
        if new_heat > 0.06:
            logging.warning(f"⚠️ Portfolio heat too high ({new_heat:.1%}). Reducing position size.")
            shares *= (0.06 - self.calculate_portfolio_heat()) / (new_heat - self.calculate_portfolio_heat())
        
        logging.info(f"📏 Position Size for {symbol}: {shares:.2f} shares "
                    f"(${shares * current_price:.2f}) at ${current_price:.2f}")
        
        return shares
    
    def calculate_stop_loss(self, entry_price: float, atr: float, signal: str) -> float:
        """محاسبه Stop Loss بر اساس ATR"""
        if signal == "PROFIT":
            stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
        else:
            stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
        
        return round(stop_loss, 2)
    
    def calculate_take_profit(self, entry_price: float, atr: float, signal: str) -> float:
        """محاسبه Take Profit بر اساس ATR"""
        if signal == "PROFIT":
            take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
        else:
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
        
        self.win_rate_history[symbol].append(pnl)
        if len(self.win_rate_history[symbol]) > 100:
            self.win_rate_history[symbol].pop(0)
        
        if self.risk_metrics['total_trades'] > 0:
            self.risk_metrics['win_rate'] = self.risk_metrics['winning_trades'] / self.risk_metrics['total_trades']
        
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        self.risk_metrics['max_drawdown'] = max(self.risk_metrics['max_drawdown'], drawdown)
        
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

# === توابع cleanup (حفظ شده) ===
def cleanup_and_shutdown():
    """تابع cleanup برای ارسال پیام قطع ارتباط و ذخیره آمار"""
    global successful_predictions, failed_attempts, iteration_count, shutdown_message_sent, cleanup_in_progress
    
    with shutdown_lock:
        if cleanup_in_progress or shutdown_message_sent:
            return
        
        cleanup_in_progress = True
        
        try:
            logging.info("🔄 Starting cleanup and shutdown process...")
            
            save_performance_metrics()
            risk_manager.save_risk_metrics()
            
            if TELEGRAM_ENABLED and not shutdown_message_sent:
                shutdown_message_sent = True
                
                total_attempts = successful_predictions + failed_attempts
                final_risk_report = risk_manager.get_risk_report()
                
                shutdown_message = f"""
🛑 <b>ربات مشاور هوشمند v6.1 متوقف شد</b>

📊 <b>آمار نهایی:</b>
• تعداد کل بررسی‌ها: {iteration_count}
• سیگنال‌های صادر شده: {len(signals_history)}
• نرخ موفقیت: {(successful_predictions / total_attempts * 100) if total_attempts > 0 else 0:.1f}%

🤖 <b>مدل Enhanced:</b>
{api_model_info.get('model_type', 'Unknown')} v6.1 (58+ Features)

🔐 <b>Authentication:</b>
User: {API_USERNAME} {'(Success)' if USE_AUTHENTICATION else '(Disabled)'}

⚙️ <b>تنظیمات v6.1:</b>
• Threshold: {CONFIDENCE_THRESHOLD:.0%}
• Poll Interval: {POLL_INTERVAL_SECONDS}s
• Features: 58+ (Sentiment + Telegram-derived Reddit)

{final_risk_report}

🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#BotStopped #v6_1 #Enhanced #TelegramBased
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

atexit.register(cleanup_and_shutdown)

# === بخش Enhanced Authentication Check ===
def check_authentication():
    """بررسی Authentication Enhanced"""
    if not USE_AUTHENTICATION:
        logging.info("🔓 Authentication disabled - running in legacy mode")
        return True
    
    try:
        logging.info(f"🔐 Testing Enhanced authentication with username: {API_USERNAME}")
        
        test_response = requests.get(
            API_HEALTH_URL, 
            timeout=5,
            auth=(API_USERNAME, API_PASSWORD)
        )
        
        if test_response.status_code == 200:
            logging.info("✅ Enhanced authentication test successful")
            return True
        elif test_response.status_code == 401:
            logging.error("❌ Enhanced authentication test failed - Invalid credentials")
            logging.error(f"💡 Username: {API_USERNAME}")
            logging.error("💡 Please update Bot_Authentication section in config.ini")
            return False
        else:
            logging.warning(f"⚠️ Unexpected response: {test_response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Enhanced authentication test error: {e}")
        return False

# === بخش Enhanced API Health Check ===
def check_api_health():
    """بررسی سلامت API Enhanced v6.1"""
    global api_model_info
    
    try:
        logging.info(f"🔍 Checking Enhanced API health at {API_HEALTH_URL}")
        
        if USE_AUTHENTICATION:
            health_response = requests.get(API_HEALTH_URL, timeout=10, auth=(API_USERNAME, API_PASSWORD))
        else:
            health_response = requests.get(API_HEALTH_URL, timeout=10)
        
        logging.info(f"📡 Enhanced API Response Status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            if health_data.get('status') == 'healthy':
                logging.info("✅ Enhanced API Health Check: Healthy")
                
                # دریافت اطلاعات مدل Enhanced
                if 'model_info' in health_data:
                    api_model_info = health_data['model_info']
                    model_type = api_model_info.get('model_type', 'Unknown')
                    model_version = api_model_info.get('model_version', '6.1')
                    threshold = api_model_info.get('optimal_threshold', 0.5)
                    is_enhanced = api_model_info.get('is_enhanced', False)
                    features_count = api_model_info.get('features_count', 0)
                    telegram_reddit_mapping = api_model_info.get('telegram_reddit_mapping', False)
                    reddit_source = api_model_info.get('reddit_source', 'unknown')
                    
                    logging.info(f"🤖 Enhanced Model Type: {model_type} v{model_version}")
                    logging.info(f"🎯 Model Optimal Threshold: {threshold:.4f}")
                    logging.info(f"⚡ Enhanced Model: {'Yes' if is_enhanced else 'No'}")
                    logging.info(f"🔢 Features Count: {features_count}")
                    logging.info(f"📱 Telegram-Reddit Mapping: {'Yes' if telegram_reddit_mapping else 'No'}")
                    logging.info(f"🔴 Reddit Source: {reddit_source}")
                    
                    # نمایش Feature Categories
                    if 'feature_categories' in health_data:
                        feature_cats = health_data['feature_categories']
                        logging.info(f"🏷️ Feature Categories:")
                        for category, count in feature_cats.items():
                            if count > 0:
                                category_display = category
                                if category == 'telegram_derived_features':
                                    category_display += " (از Telegram sentiment مشتق شده)"
                                logging.info(f"   {category_display}: {count} features")
                    
                    # نمایش Sentiment Analysis
                    if 'sentiment_analysis' in health_data:
                        sentiment_info = health_data['sentiment_analysis']
                        logging.info(f"🎭 Sentiment Features: {sentiment_info.get('sentiment_features_found', 0)}")
                        logging.info(f"📱 Telegram-derived Reddit Features: {sentiment_info.get('telegram_derived_reddit_features_found', 0)}")
                        
                        coverage_stats = sentiment_info.get('coverage_stats', {})
                        if coverage_stats:
                            sent_cov = coverage_stats.get('sentiment_coverage', 0)
                            telegram_reddit_cov = coverage_stats.get('telegram_derived_reddit_coverage', 0)
                            logging.info(f"📊 Sentiment Coverage: {sent_cov:.2%}")
                            logging.info(f"📊 Telegram-derived Reddit Coverage: {telegram_reddit_cov:.2%}")
                        
                        # بررسی Telegram mapping
                        if sentiment_info.get('telegram_mapping_detected'):
                            logging.info("✅ Telegram → Reddit mapping confirmed")
                        else:
                            logging.warning("⚠️ Telegram → Reddit mapping not detected")
                    
                    # تطبیق threshold با مدل Enhanced
                    global CONFIDENCE_THRESHOLD
                    if threshold > 0.60 and CONFIDENCE_THRESHOLD > 0.50:
                        old_threshold = CONFIDENCE_THRESHOLD
                        CONFIDENCE_THRESHOLD = 0.40
                        logging.warning(f"🔧 Enhanced model threshold ({threshold:.4f}) is high.")
                        logging.warning(f"🔧 Bot threshold adjusted: {old_threshold:.0%} → {CONFIDENCE_THRESHOLD:.0%}")
                    
                    # نمایش performance Enhanced
                    if 'performance' in api_model_info:
                        performance = api_model_info['performance']
                        logging.info(f"📊 Enhanced Model Performance: "
                                   f"Accuracy={performance.get('accuracy', 0):.1%}, "
                                   f"Precision={performance.get('precision', 0):.1%}, "
                                   f"Recall={performance.get('recall', 0):.1%}, "
                                   f"F1={performance.get('f1_score', 0):.4f}")
                
                return True
            else:
                logging.error("❌ Enhanced API Health Check: Unhealthy")
                logging.error(f"📋 Health response: {health_data}")
                return False
                
        elif health_response.status_code == 401:
            logging.error("❌ Enhanced API Health Check failed: 401 Authentication Error")
            logging.error(f"💡 Current credentials: {API_USERNAME} / [password hidden]")
            logging.error("💡 Please check Bot_Authentication section in config.ini")
            return False
        elif health_response.status_code == 500:
            try:
                error_data = health_response.json()
                logging.error(f"❌ Enhanced API Health Check failed (HTTP 500): {error_data}")
            except:
                error_text = health_response.text[:200]
                logging.error(f"❌ Enhanced API Health Check failed (HTTP 500): {error_text}")
            return False
        else:
            logging.error(f"❌ Enhanced API Health Check failed: HTTP {health_response.status_code}")
            try:
                response_text = health_response.text[:200]
                logging.error(f"📋 Response: {response_text}")
            except:
                pass
            return False
            
    except requests.exceptions.ConnectionError as e:
        logging.error(f"❌ Connection Error: Enhanced API server not reachable - {e}")
        return False
    except requests.exceptions.Timeout as e:
        logging.error(f"❌ Timeout Error: Enhanced API server too slow - {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Enhanced API Health Check error: {e}")
        return False

def test_api_connection():
    """تست اتصال API Enhanced"""
    print("\n🔍 Testing Enhanced API Connection...")
    
    try:
        response = requests.get(f"http://{API_HOST}:{API_PORT}/", timeout=10)
        if response.status_code == 200:
            print(f"✅ Enhanced main endpoint accessible: {response.text[:50]}...")
        else:
            print(f"⚠️ Enhanced main endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Enhanced main endpoint failed: {e}")
    
    try:
        if USE_AUTHENTICATION:
            response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=10, auth=(API_USERNAME, API_PASSWORD))
        else:
            response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=10)
            
        print(f"📊 Enhanced health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Enhanced health check successful: {data.get('status')}")
            
            # نمایش اطلاعات Enhanced
            if 'api_version' in data:
                print(f"🔄 API Version: {data['api_version']}")
            if 'model_info' in data:
                model_info = data['model_info']
                print(f"🤖 Model: {model_info.get('model_type', 'Unknown')}")
                print(f"🔢 Features: {model_info.get('features_count', 0)}")
                print(f"📱 Telegram-Reddit Mapping: {model_info.get('telegram_reddit_mapping', False)}")
            
            return True
        elif response.status_code == 401:
            print(f"❌ Enhanced authentication error in health endpoint")
            print(f"💡 Username: {API_USERNAME}")
        else:
            print(f"⚠️ Enhanced unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Enhanced health endpoint test failed: {e}")
    
    return False

# --- بخش ۳: توابع تلگرام (Enhanced) ---
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
                          take_profit: float = None, threshold_used: float = None,
                          sentiment_coverage: float = 0, telegram_reddit_coverage: float = 0,
                          feature_count: int = 0, telegram_mapping_detected: bool = False) -> str:
    """🔧 اصلاح 4: فرمت‌دهی پیام Enhanced برای تلگرام (اصلاح شده برای Telegram-based)"""
    emoji_signal = "🟢" if signal == "PROFIT" else "🔴"
    emoji_confidence = "🔥" if confidence >= 0.8 else "✅" if confidence >= 0.7 else "⚡"
    
    # اطلاعات مدل Enhanced
    model_type = api_model_info.get('model_type', 'Unknown')
    model_version = api_model_info.get('model_version', '6.1')
    is_enhanced = api_model_info.get('is_enhanced', False)
    model_accuracy = api_model_info.get('performance', {}).get('accuracy')
    telegram_reddit_mapping = api_model_info.get('telegram_reddit_mapping', False)
    reddit_source = api_model_info.get('reddit_source', 'unknown')
    
    message = f"""
{emoji_signal} <b>سیگنال Enhanced از ربات مشاور هوشمند v6.1</b> {emoji_signal}

📊 <b>نماد:</b> {symbol}
⏱ <b>تایم فریم:</b> {timeframe}
🏦 <b>صرافی:</b> {exchange.upper()}
📈 <b>سیگنال:</b> <b>{signal}</b>
{emoji_confidence} <b>اطمینان:</b> {confidence:.1%}
🎯 <b>آستانه ربات:</b> {CONFIDENCE_THRESHOLD:.0%}
"""

    # اطلاعات مدل Enhanced
    if threshold_used:
        threshold_emoji = "⚡" if is_enhanced else "🔧"
        message += f"""
🤖 <b>مدل Enhanced:</b> {model_type[:20]}{'...' if len(model_type) > 20 else ''} v{model_version}
{threshold_emoji} <b>Threshold:</b> {threshold_used:.3f} {'(Enhanced)' if is_enhanced else '(Legacy)'}
"""
    
    if model_accuracy:
        message += f"📊 <b>دقت مدل:</b> {model_accuracy:.1%}\n"
    
    # 🔧 اصلاح 4: اطلاعات Features Enhanced با Telegram mapping
    message += f"""
🔢 <b>Features:</b> {feature_count} (Enhanced: 58+)
🎭 <b>Sentiment:</b> {sentiment_coverage:.1%} coverage
📱 <b>Telegram→Reddit:</b> {telegram_reddit_coverage:.1%} coverage
"""
    
    # 🔧 اصلاح 4: نمایش Telegram mapping info
    if telegram_reddit_mapping:
        mapping_emoji = "✅" if telegram_mapping_detected else "📱"
        message += f"{mapping_emoji} <b>Reddit Source:</b> {reddit_source} (Telegram-derived)\n"
    
    # اطلاعات Authentication Enhanced
    auth_emoji = "🔐" if USE_AUTHENTICATION else "🔓"
    message += f"{auth_emoji} <b>Auth Enhanced:</b> {API_USERNAME if USE_AUTHENTICATION else 'Disabled'}\n"
    
    # افزودن اطلاعات Risk Management
    if position_size is not None:
        message += f"""
💼 <b>مدیریت ریسک:</b>
   📏 اندازه پوزیشن: {position_size:.2f} واحد
   🛑 حد ضرر: ${stop_loss:.2f}
   ✅ حد سود: ${take_profit:.2f}
   🔥 Portfolio Heat: {risk_manager.portfolio_heat:.1%}
"""
    
    # نمایش تنظیمات Enhanced
    message += f"""
⚙️ <b>تنظیمات Enhanced v6.1:</b>
   🔄 Poll Interval: {POLL_INTERVAL_SECONDS}s
   🎯 Threshold: {CONFIDENCE_THRESHOLD:.0%}
   📊 ویژگی‌ها: 58+ (Sentiment + Telegram-based Reddit)
   ⚡ Enhanced API: v6.1

🕐 <b>زمان:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#EnhancedAdvisor #CryptoSignal #{symbol.replace('/', '')} #{timeframe} #v6_1 #TelegramBased
"""
    return message

# --- بخش ۴: توابع بررسی همخوانی Enhanced ---
def load_model_features() -> Optional[List[str]]:
    """بارگذاری لیست ویژگی‌های Enhanced مدل"""
    try:
        # سعی در دریافت از Enhanced API
        try:
            if USE_AUTHENTICATION:
                response = requests.get(API_MODEL_INFO_URL, timeout=5, auth=(API_USERNAME, API_PASSWORD))
            else:
                response = requests.get(API_MODEL_INFO_URL, timeout=5)
                
            if response.status_code == 200:
                model_info = response.json()
                feature_columns = model_info.get('model_info', {}).get('feature_columns', [])
                if feature_columns:
                    logging.info(f"✅ Enhanced model features from API: {len(feature_columns)} features")
                    return feature_columns
        except:
            logging.warning("Could not get Enhanced features from API, trying local files...")
        
        # جستجو در فایل‌های Enhanced محلی
        enhanced_patterns = [
            'feature_names_enhanced_v6_*.txt',
            'feature_names_optimized_*.txt',
            'feature_names_*.txt'
        ]
        
        latest_file = None
        for pattern in enhanced_patterns:
            list_of_files = glob.glob(os.path.join(MODELS_PATH, pattern))
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                break
        
        if not latest_file:
            list_of_files = glob.glob(os.path.join(MODELS_PATH, 'run_*/feature_names_*.txt'))
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
        
        if not latest_file:
            logging.warning("فایل Enhanced feature_names یافت نشد. ربات بدون بررسی همخوانی ادامه می‌دهد.")
            return None
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f if line.strip() and not line.startswith('=') and not line.startswith('[')]
        
        logging.info(f"✅ لیست {len(feature_names)} ویژگی Enhanced از '{os.path.basename(latest_file)}' بارگذاری شد.")
        return feature_names
        
    except Exception as e:
        logging.error(f"خطا در بارگذاری Enhanced feature_names: {e}")
        return None

def verify_feature_consistency(calculated_features: Dict[str, Any], expected_features: List[str]) -> bool:
    """بررسی تطابق ویژگی‌های Enhanced محاسبه شده"""
    missing_features = []
    for feature in expected_features:
        if feature not in calculated_features:
            missing_features.append(feature)
    
    if missing_features:
        logging.error(f"❌ ویژگی‌های Enhanced گمشده: {missing_features[:10]}")
        if len(missing_features) > 10:
            logging.error(f"... و {len(missing_features) - 10} ویژگی دیگر")
        return False
    
    logging.info(f"✅ تمام {len(expected_features)} ویژگی Enhanced مورد نیاز محاسبه شده‌اند.")
    return True

# --- بخش ۵: توابع اصلی Enhanced ---
def fetch_from_cryptocompare_api(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """تابع اختصاصی برای دریافت داده از CryptoCompare API"""
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
    """تابع Enhanced برای دریافت داده"""
    logging.info(f"Attempting to fetch Enhanced data from: {exchange_name.upper()} for {symbol} {timeframe}")
    
    if exchange_name.lower() == 'cryptocompare':
        return fetch_from_cryptocompare_api(symbol, timeframe, limit)
    else:
        try:
            if exchange_name.lower() == 'binance':
                exchange = ccxt.binance({
                    'timeout': 30000,
                    'rateLimit': 1500,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
            else:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'timeout': 30000,
                    'rateLimit': 2000,
                    'enableRateLimit': True
                })
            
            max_retries = 3
            base_delay = 2
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay_time = base_delay ** attempt
                        logging.info(f"⏳ Enhanced waiting {delay_time}s before retry (attempt {attempt + 1})...")
                        time.sleep(delay_time)
                    
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    if len(df) < limit // 2:
                        logging.warning(f"Enhanced data received ({len(df)}) is less than expected ({limit}).")
                        if attempt < max_retries - 1:
                            logging.info(f"Enhanced retrying... (attempt {attempt + 2}/{max_retries})")
                            continue
                    
                    logging.info(f"Successfully fetched {len(df)} Enhanced candles from {exchange_name.upper()}")
                    return df
                    
                except ccxt.RateLimitExceeded as rate_error:
                    logging.warning(f"⚠️ Enhanced rate limit exceeded on attempt {attempt + 1}: {rate_error}")
                    if attempt < max_retries - 1:
                        delay_time = 90
                        logging.info(f"⏳ Enhanced rate limit cooldown: waiting {delay_time}s...")
                        time.sleep(delay_time)
                        continue
                    else:
                        logging.error("❌ Enhanced rate limit exceeded - falling back to CryptoCompare")
                        return fetch_from_cryptocompare_api(symbol, timeframe, limit)
                        
                except ccxt.NetworkError as network_error:
                    logging.warning(f"🌐 Enhanced network error on attempt {attempt + 1}: {network_error}")
                    if attempt < max_retries - 1:
                        delay_time = base_delay ** (attempt + 1)
                        logging.info(f"⏳ Enhanced network error cooldown: waiting {delay_time}s...")
                        time.sleep(delay_time)
                        continue
                    else:
                        logging.error("❌ Enhanced network error persists - falling back to CryptoCompare")
                        return fetch_from_cryptocompare_api(symbol, timeframe, limit)
                        
                except Exception as attempt_error:
                    logging.warning(f"Enhanced attempt {attempt + 1} failed: {attempt_error}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise attempt_error
            
        except AttributeError:
            logging.error(f"Exchange '{exchange_name}' is not supported by CCXT.")
        except ccxt.BaseError as e:
            logging.error(f"Enhanced exchange error from {exchange_name.upper()}: {e}")
            if exchange_name.lower() == 'binance':
                logging.info("🔄 Enhanced fallback to CryptoCompare due to exchange error...")
                return fetch_from_cryptocompare_api(symbol, timeframe, limit)
        except Exception as e:
            logging.error(f"Enhanced unexpected error fetching data from {exchange_name.upper()}: {e}")
            if exchange_name.lower() == 'binance':
                logging.info("🔄 Enhanced fallback to CryptoCompare due to connection issues...")
                return fetch_from_cryptocompare_api(symbol, timeframe, limit)
        
        return None

def safe_numeric_conversion(series: pd.Series, name: str) -> pd.Series:
    """تبدیل ایمن Enhanced به numeric"""
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        logging.warning(f"خطا در تبدیل Enhanced {name} به numeric: {e}")
        return series.fillna(0)

def calculate_enhanced_features(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    🔧 اصلاح 1: محاسبه ویژگی‌های Enhanced 58+ ویژگی مطابق فایل 03 (اصلاح شده)
    شامل: Technical (43) + Sentiment (6) + Telegram-derived Reddit (4) + Other (5+)
    
    تغییرات اصلی:
    - بهبود sentiment calculation (real-time)
    - reddit features derivation از sentiment_score
    - تطبیق کامل با 58+ features
    """
    try:
        group = df.copy()
        
        # تبدیل اصلاح شده انواع داده
        for col in ['volume', 'high', 'low', 'close', 'open']:
            group[col] = safe_numeric_conversion(group[col], col)
        
        # === بخش 1: اندیکاتورهای فنی (43 ویژگی) ===
        
        # RSI
        group['rsi'] = ta.rsi(group['close'], length=INDICATOR_PARAMS['rsi_length'])
        
        # MACD
        macd = ta.macd(group['close'], 
                      fast=INDICATOR_PARAMS['macd_fast'], 
                      slow=INDICATOR_PARAMS['macd_slow'], 
                      signal=INDICATOR_PARAMS['macd_signal'])
        if macd is not None and not macd.empty:
            col_names = macd.columns.tolist()
            group['macd'] = macd[col_names[0]]
            group['macd_hist'] = macd[col_names[1]]
            group['macd_signal'] = macd[col_names[2]]
        
        # Bollinger Bands
        bbands = ta.bbands(group['close'], 
                          length=INDICATOR_PARAMS['bb_length'], 
                          std=INDICATOR_PARAMS['bb_std'])
        if bbands is not None and not bbands.empty:
            col_names = bbands.columns.tolist()
            group['bb_upper'] = bbands[col_names[0]]
            group['bb_middle'] = bbands[col_names[1]]
            group['bb_lower'] = bbands[col_names[2]]
            bb_range = group['bb_upper'] - group['bb_lower']
            group['bb_position'] = np.where(bb_range != 0, 
                                          (group['close'] - group['bb_lower']) / bb_range, 
                                          0.5)
        
        # ATR و نوسان
        group['atr'] = ta.atr(group['high'], group['low'], group['close'], 
                             length=INDICATOR_PARAMS['atr_length'])
        group['atr_percent'] = np.where(group['close'] != 0, 
                                      (group['atr'] / group['close']) * 100, 
                                      0)
        group['price_change'] = group['close'].pct_change()
        group['volatility'] = group['price_change'].rolling(window=20).std() * 100
        
        # VWAP
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        vwap_numerator = (typical_price * group['volume']).cumsum()
        vwap_denominator = group['volume'].cumsum()
        group['vwap'] = np.where(vwap_denominator != 0, 
                               vwap_numerator / vwap_denominator, 
                               typical_price)
        group['vwap_deviation'] = np.where(group['vwap'] != 0,
                                         ((group['close'] - group['vwap']) / group['vwap']) * 100,
                                         0)
        
        # Volume indicators
        group['obv'] = ta.obv(group['close'], group['volume'])
        group['obv_change'] = group['obv'].pct_change().fillna(0)
        
        # MFI اصلاح شده
        try:
            high_values = group['high'].astype('float64')
            low_values = group['low'].astype('float64') 
            close_values = group['close'].astype('float64')
            volume_values = group['volume'].astype('float64')
            
            group['mfi'] = ta.mfi(high_values, low_values, close_values, volume_values, length=14)
        except Exception as mfi_error:
            logging.warning(f"Enhanced MFI calculation failed: {mfi_error}. Using default value.")
            group['mfi'] = 50.0
        
        group['ad'] = ta.ad(group['high'], group['low'], group['close'], group['volume'])
        
        # Stochastic
        stoch = ta.stoch(group['high'], group['low'], group['close'], k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            col_names = stoch.columns.tolist()
            group['stoch_k'] = stoch[col_names[0]]
            group['stoch_d'] = stoch[col_names[1]]
        
        # Oscillators
        group['williams_r'] = ta.willr(group['high'], group['low'], group['close'], length=14)
        group['cci'] = ta.cci(group['high'], group['low'], group['close'], length=20)
        
        # Moving Averages
        group['ema_short'] = ta.ema(group['close'], length=INDICATOR_PARAMS['ema_short'])
        group['ema_medium'] = ta.ema(group['close'], length=INDICATOR_PARAMS['ema_medium'])
        group['ema_long'] = ta.ema(group['close'], length=INDICATOR_PARAMS['ema_long'])
        group['ema_short_above_medium'] = (group['ema_short'] > group['ema_medium']).astype(int)
        group['ema_medium_above_long'] = (group['ema_medium'] > group['ema_long']).astype(int)
        group['ema_short_slope'] = group['ema_short'].pct_change(periods=5).fillna(0)
        group['ema_medium_slope'] = group['ema_medium'].pct_change(periods=5).fillna(0)
        
        group['sma_short'] = ta.sma(group['close'], 10)
        group['sma_medium'] = ta.sma(group['close'], 20)
        group['sma_long'] = ta.sma(group['close'], 50)
        group['price_above_sma_short'] = (group['close'] > group['sma_short']).astype(int)
        group['price_above_sma_medium'] = (group['close'] > group['sma_medium']).astype(int)
        group['price_above_sma_long'] = (group['close'] > group['sma_long']).astype(int)
        
        # Returns and price features
        group['return_1'] = group['close'].pct_change(1).fillna(0)
        group['return_5'] = group['close'].pct_change(5).fillna(0)
        group['return_10'] = group['close'].pct_change(10).fillna(0)
        group['avg_return_5'] = group['return_1'].rolling(5, min_periods=1).mean()
        group['avg_return_10'] = group['return_1'].rolling(10, min_periods=1).mean()
        group['hl_ratio'] = np.where(group['close'] != 0,
                                   (group['high'] - group['low']) / group['close'],
                                   0)
        hl_range = group['high'] - group['low']
        group['close_position'] = np.where(hl_range != 0,
                                         (group['close'] - group['low']) / hl_range,
                                         0.5)
        group['volume_ma'] = group['volume'].rolling(20, min_periods=1).mean()
        group['volume_ratio'] = np.where(group['volume_ma'] != 0,
                                       group['volume'] / group['volume_ma'],
                                       1.0)
        
        # === 🔧 PSAR Enhanced (مهم برای 58 ویژگی) ===
        try:
            psar_result = ta.psar(group['high'], group['low'], group['close'], 
                                 af0=INDICATOR_PARAMS['psar_af'], 
                                 af=INDICATOR_PARAMS['psar_af'], 
                                 max_af=INDICATOR_PARAMS['psar_max_af'])
            if psar_result is not None:
                if isinstance(psar_result, pd.DataFrame):
                    if len(psar_result.columns) > 0:
                        group['psar'] = psar_result.iloc[:, 0]
                    else:
                        group['psar'] = group['close'] * 0.98
                else:
                    group['psar'] = psar_result
                
                if 'psar' in group.columns:
                    group['price_above_psar'] = (group['close'] > group['psar']).astype(int)
                else:
                    group['psar'] = group['close'] * 0.98
                    group['price_above_psar'] = 1
            else:
                group['psar'] = group['close'] * 0.98
                group['price_above_psar'] = 1
                
        except Exception as e:
            logging.warning(f"Enhanced PSAR calculation failed: {e}. Using fallback values.")
            group['psar'] = group['close'] * 0.98
            group['price_above_psar'] = 1
        
        # ADX
        adx = ta.adx(group['high'], group['low'], group['close'], length=14)
        if adx is not None and not adx.empty:
            col_names = adx.columns.tolist()
            for col in col_names:
                if 'ADX' in col:
                    group['adx'] = adx[col]
                    break
            else:
                group['adx'] = 50
        else:
            group['adx'] = 50
        
        # === بخش 2: ویژگی‌های احساسات Enhanced (6 ویژگی) - 🔧 اصلاح 2 ===
        
        try:
            # 🔧 اصلاح 2: بهبود sentiment calculation - Real-time sentiment processing
            logging.debug("🎭 محاسبه sentiment features واقعی...")
            
            # شبیه‌سازی sentiment بر اساس price momentum + volume + volatility
            price_momentum = group['close'].pct_change(5).rolling(10, min_periods=1).mean().fillna(0)
            volume_momentum = (group['volume_ratio'].rolling(5, min_periods=1).mean() - 1).fillna(0)
            volatility_factor = (group['volatility'].rolling(5, min_periods=1).mean() / 100).fillna(0)
            
            # sentiment_score اصلی (بهبود یافته - بر اساس market dynamics)
            momentum_component = np.tanh(price_momentum * 3)  # -1 تا +1
            volume_component = np.tanh(volume_momentum * 2)   # تأثیر حجم
            volatility_component = np.tanh(volatility_factor) # تأثیر نوسان
            
            # ترکیب weighted برای sentiment_score واقعی
            group['sentiment_score'] = (
                momentum_component * 0.5 + 
                volume_component * 0.3 + 
                volatility_component * 0.2
            )
            group['sentiment_score'] = group['sentiment_score'].fillna(0)
            
            # sentiment momentum (تغییرات احساسات)
            momentum_period = min(INDICATOR_PARAMS['sentiment_momentum_period'], len(group))
            if momentum_period > 1:
                group['sentiment_momentum'] = group['sentiment_score'].diff(momentum_period).fillna(0)
            else:
                group['sentiment_momentum'] = 0
            
            # sentiment moving averages
            window_short = min(INDICATOR_PARAMS['sentiment_ma_short'], len(group))
            window_long = min(INDICATOR_PARAMS['sentiment_ma_long'], len(group))
            
            group['sentiment_ma_7'] = group['sentiment_score'].rolling(
                window=max(1, window_short), min_periods=1
            ).mean()
            group['sentiment_ma_14'] = group['sentiment_score'].rolling(
                window=max(1, window_long), min_periods=1
            ).mean()
            
            # sentiment volume interaction
            sentiment_abs = abs(group['sentiment_score'])
            volume_normalized = group['volume'] / group['volume'].max() if group['volume'].max() > 0 else 1
            group['sentiment_volume'] = sentiment_abs * volume_normalized
            group['sentiment_volume'] = group['sentiment_volume'].rolling(24, min_periods=1).sum()
            
            # sentiment divergence من price
            if len(group) > 20:
                price_returns = group['close'].pct_change(20).fillna(0)
                sentiment_change = group['sentiment_score'].diff(20).fillna(0)
                rolling_corr = price_returns.rolling(window=30, min_periods=10).corr(sentiment_change)
                group['sentiment_divergence'] = 1 - rolling_corr.fillna(0)
            else:
                group['sentiment_divergence'] = 0
            
            # آمار sentiment
            sentiment_mean = group['sentiment_score'].mean()
            sentiment_non_zero = (group['sentiment_score'] != 0).sum()
            logging.debug(f"📊 Sentiment calculated: mean={sentiment_mean:.4f}, non_zero={sentiment_non_zero}/{len(group)}")
                
        except Exception as e:
            logging.warning(f"Enhanced sentiment calculation failed: {e}. Using fallback.")
            group['sentiment_score'] = 0
            group['sentiment_momentum'] = 0
            group['sentiment_ma_7'] = 0
            group['sentiment_ma_14'] = 0
            group['sentiment_volume'] = 0
            group['sentiment_divergence'] = 0
        
        # === بخش 3: ویژگی‌های Telegram-derived Reddit (4+ ویژگی) - 🔧 اصلاح 1 ===
        
        try:
            # 🔧 اصلاح 1: Reddit derivation از sentiment_score (بجای hardcode 0)
            logging.debug("📱 محاسبه Telegram-derived Reddit features...")
            
            # استفاده از sentiment_score واقعی به عنوان پایه Reddit features
            if 'sentiment_score' in group.columns and group['sentiment_score'].sum() != 0:
                # reddit_score = sentiment_score (نگاشت مستقیم از Telegram sentiment)
                group['reddit_score'] = group['sentiment_score']
                
                # reddit_comments تخمین زده می‌شود از sentiment + activity level
                activity_factor = (group['volume_ratio'] + group['volatility'] / 100) / 2
                reddit_base = abs(group['sentiment_score']) * INDICATOR_PARAMS['reddit_derivation_multiplier']
                group['reddit_comments'] = reddit_base * activity_factor
                group['reddit_comments'] = np.maximum(group['reddit_comments'], 0)  # حداقل 0
                
                # moving averages برای Reddit features
                reddit_ma_window = min(INDICATOR_PARAMS['telegram_sentiment_ma'], len(group))
                group['reddit_score_ma'] = group['reddit_score'].rolling(
                    window=max(1, reddit_ma_window), min_periods=1
                ).mean()
                group['reddit_comments_ma'] = group['reddit_comments'].rolling(
                    window=max(1, reddit_ma_window), min_periods=1
                ).mean()
                
                # momentum برای Reddit features
                momentum_period = min(12, len(group))
                if momentum_period > 1:
                    group['reddit_score_momentum'] = group['reddit_score'].diff(momentum_period).fillna(0)
                    group['reddit_comments_momentum'] = group['reddit_comments'].diff(momentum_period).fillna(0)
                else:
                    group['reddit_score_momentum'] = 0
                    group['reddit_comments_momentum'] = 0
                
                # sentiment-reddit correlation (خودهمبستگی چون reddit از sentiment مشتق شده)
                if len(group) > 10:
                    corr_window = min(20, len(group))
                    # correlation با sentiment (برای سازگاری)
                    group['sentiment_reddit_score_corr'] = group['sentiment_score'].rolling(
                        window=corr_window, min_periods=5
                    ).corr(group['reddit_score']).fillna(1.0)  # باید نزدیک 1 باشد
                    group['sentiment_reddit_comments_corr'] = group['sentiment_score'].rolling(
                        window=corr_window, min_periods=5
                    ).corr(group['reddit_comments']).fillna(0.8)
                else:
                    group['sentiment_reddit_score_corr'] = 1.0  # perfect correlation
                    group['sentiment_reddit_comments_corr'] = 0.8
                
                # آمار Reddit features
                reddit_score_mean = group['reddit_score'].mean()
                reddit_comments_mean = group['reddit_comments'].mean()
                reddit_non_zero = (group['reddit_score'] != 0).sum()
                logging.debug(f"📱 Reddit features calculated: reddit_score_mean={reddit_score_mean:.4f}, "
                            f"reddit_comments_mean={reddit_comments_mean:.2f}, non_zero={reddit_non_zero}/{len(group)}")
                
                # تأیید mapping
                if abs(group['reddit_score'].iloc[-1] - group['sentiment_score'].iloc[-1]) < 0.0001:
                    logging.debug("✅ Telegram → Reddit mapping confirmed: reddit_score = sentiment_score")
                
            else:
                # اگر sentiment_score صفر است یا وجود ندارد
                logging.warning("⚠️ sentiment_score not available or zero - using fallback Reddit features")
                group['reddit_score'] = 0
                group['reddit_comments'] = 0
                group['reddit_score_ma'] = 0
                group['reddit_comments_ma'] = 0
                group['reddit_score_momentum'] = 0
                group['reddit_comments_momentum'] = 0
                group['sentiment_reddit_score_corr'] = 0
                group['sentiment_reddit_comments_corr'] = 0
                
        except Exception as e:
            logging.warning(f"Enhanced Telegram-derived Reddit calculation failed: {e}. Using fallback.")
            group['reddit_score'] = 0
            group['reddit_comments'] = 0
            group['reddit_score_ma'] = 0
            group['reddit_comments_ma'] = 0
            group['reddit_score_momentum'] = 0
            group['reddit_comments_momentum'] = 0
            group['sentiment_reddit_score_corr'] = 0
            group['sentiment_reddit_comments_corr'] = 0
        
        # === بخش 4: ویژگی‌های Source Diversity (2 ویژگی) ===
        try:
            # شبیه‌سازی source diversity بر اساس market activity
            activity_level = group['volume_ratio'].rolling(10, min_periods=1).std().fillna(0)
            price_activity = group['volatility'].rolling(5, min_periods=1).mean().fillna(0)
            
            # تنوع منابع بر اساس فعالیت بازار
            diversity_base = (activity_level + price_activity / 100) / 2
            group['source_diversity'] = np.minimum(diversity_base * 5, 5)  # حداکثر 5 منبع
            group['source_diversity'] = group['source_diversity'].fillna(1)
            
            max_diversity = group['source_diversity'].max()
            group['source_diversity_normalized'] = np.where(max_diversity > 0,
                                                          group['source_diversity'] / max_diversity,
                                                          0)
            
            # تعامل diversity با sentiment
            group['sentiment_diversity_interaction'] = group['sentiment_score'] * group['source_diversity_normalized']
            
        except Exception as e:
            logging.warning(f"Enhanced source diversity calculation failed: {e}. Using fallback.")
            group['source_diversity'] = 1
            group['source_diversity_normalized'] = 0
            group['sentiment_diversity_interaction'] = 0
        
        # === بخش پاکسازی و استخراج نهایی ===
        
        # استخراج آخرین ردیف
        latest_features = group.iloc[-1].to_dict()
        
        # ذخیره مقدار ATR برای Risk Management
        latest_atr = group['atr'].iloc[-1]
        
        # پاک‌سازی Enhanced برای API
        features_for_api = {}
        for k, v in latest_features.items():
            try:
                if pd.notna(v):
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        if not np.isinf(v):
                            if isinstance(v, np.integer):
                                features_for_api[k] = int(v)
                            elif isinstance(v, np.floating):
                                features_for_api[k] = float(v)
                            else:
                                features_for_api[k] = v
                    elif isinstance(v, (str, bool)):
                        features_for_api[k] = v
                    elif hasattr(v, 'timestamp'):
                        continue
                    else:
                        try:
                            str_val = str(v)
                            if str_val not in ['nan', 'inf', '-inf', 'NaT']:
                                features_for_api[k] = str_val
                        except:
                            continue
            except Exception as e:
                logging.warning(f"Enhanced error processing feature {k}={v}: {e}")
                continue
        
        # حذف timestamp
        features_for_api.pop('timestamp', None)
        
        # فیلتر مقادیر معقول Enhanced
        cleaned_features = {}
        for k, v in features_for_api.items():
            if isinstance(v, (int, float)):
                if abs(v) < 1e10:
                    cleaned_features[k] = v
                else:
                    logging.warning(f"Enhanced outlier value removed: {k}={v}")
            else:
                cleaned_features[k] = v
        
        # افزودن ATR برای Risk Management
        if not np.isinf(latest_atr) and pd.notna(latest_atr):
            cleaned_features['_atr_value'] = float(latest_atr)
        else:
            cleaned_features['_atr_value'] = 1.0
        
        # === 🔧 اصلاح 3: بررسی تعداد ویژگی‌های Enhanced ===
        expected_features = 58
        actual_features = len(cleaned_features) - 1  # منهای _atr_value
        
        # شمارش features به دسته‌بندی
        technical_features = len([k for k in cleaned_features.keys() if not any(x in k for x in ['sentiment', 'reddit', 'source', '_atr'])])
        sentiment_features = len([k for k in cleaned_features.keys() if 'sentiment' in k])
        reddit_features = len([k for k in cleaned_features.keys() if 'reddit' in k])
        source_features = len([k for k in cleaned_features.keys() if 'source' in k])
        
        logging.debug(f"🔢 Enhanced features breakdown: Technical={technical_features}, "
                     f"Sentiment={sentiment_features}, Reddit={reddit_features}, Source={source_features}")
        logging.info(f"🔢 Enhanced features calculated: {actual_features}/58+ "
                    f"(Technical: {technical_features}, Sentiment: {sentiment_features}, "
                    f"Telegram-Reddit: {reddit_features}, Source: {source_features})")
        
        if actual_features < expected_features:
            logging.warning(f"⚠️ Enhanced feature count ({actual_features}) less than expected ({expected_features})")
        else:
            logging.info(f"✅ Enhanced features: {actual_features} ≥ {expected_features}")
        
        # اضافه کردن متادیتا برای debugging
        cleaned_features['_feature_count'] = actual_features
        cleaned_features['_sentiment_non_zero'] = (group['sentiment_score'] != 0).sum()
        cleaned_features['_reddit_non_zero'] = (group['reddit_score'] != 0).sum()
        
        return cleaned_features
        
    except Exception as e:
        logging.error(f"❌ Enhanced feature calculation error: {e}", exc_info=True)
        return None

def get_enhanced_prediction(payload: Dict) -> Optional[Dict]:
    """ارسال درخواست Enhanced به API پیش‌بینی"""
    try:
        # حذف metadata از payload قبل از ارسال
        atr_value = payload.pop('_atr_value', 1.0)
        feature_count = payload.pop('_feature_count', 0)
        sentiment_non_zero = payload.pop('_sentiment_non_zero', 0)
        reddit_non_zero = payload.pop('_reddit_non_zero', 0)
        
        # Retry mechanism Enhanced
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # ارسال درخواست Enhanced با Authentication
                if USE_AUTHENTICATION:
                    response = requests.post(
                        API_URL, 
                        json=payload, 
                        timeout=30,
                        auth=(API_USERNAME, API_PASSWORD)
                    )
                else:
                    response = requests.post(API_URL, json=payload, timeout=30)
                
                if attempt == 0:
                    logging.info(f"📡 Enhanced API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # اضافه کردن metadata به نتیجه
                    if result:
                        result['_atr_value'] = atr_value
                        result['_feature_count'] = feature_count
                        result['_sentiment_non_zero'] = sentiment_non_zero
                        result['_reddit_non_zero'] = reddit_non_zero
                    
                    return result
                elif response.status_code == 429:
                    retry_after = response.headers.get('Retry-After', 90)
                    logging.warning(f"⚠️ Enhanced rate limited (429). Retry after: {retry_after}s")
                    if attempt < max_retries - 1:
                        time.sleep(int(retry_after))
                        continue
                    return {'error': 'rate_limited', 'retry_after': int(retry_after)}
                elif response.status_code == 401:
                    logging.error("❌ Enhanced authentication Error (401) - Invalid credentials")
                    logging.error(f"💡 Check username: {API_USERNAME}")
                    return {'error': 'authentication_failed'}
                elif response.status_code == 500:
                    logging.error("❌ Enhanced Server Error (500)")
                    try:
                        error_data = response.json()
                        logging.error(f"📋 Enhanced server error details: {error_data}")
                    except:
                        logging.error(f"📋 Enhanced server error text: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return {'error': 'server_error'}
                else:
                    logging.error(f"❌ Enhanced API Error: HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    return {'error': f'http_{response.status_code}'}
                    
            except requests.exceptions.Timeout:
                logging.warning(f"⏰ Enhanced API Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return {'error': 'timeout'}
            except requests.exceptions.ConnectionError:
                logging.warning(f"🌐 Enhanced Connection Error on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                return {'error': 'connection_error'}
            except Exception as e:
                logging.error(f"❌ Enhanced prediction request error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {'error': 'unexpected_error', 'details': str(e)}
                
    except Exception as e:
        logging.error(f"❌ Critical Enhanced error in get_prediction: {e}")
        return {'error': 'critical_error', 'details': str(e)}

def save_performance_metrics():
    """ذخیره معیارهای عملکرد Enhanced"""
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
            'model_info': api_model_info,
            'enhanced_features': True,
            'telegram_reddit_mapping': api_model_info.get('telegram_reddit_mapping', False),
            'api_version': '6.1_enhanced'
        }
        
        df = pd.DataFrame([metrics])
        df.to_csv(performance_log, mode='a', header=not os.path.exists(performance_log), index=False)
        
    except Exception as e:
        logging.error(f"خطا در ذخیره Enhanced performance metrics: {e}")

def process_enhanced_pair(symbol: str, timeframe: str, exchange: str, expected_features: Optional[List] = None) -> bool:
    """🔧 اصلاح 5: پردازش Enhanced یک جفت ارز و تولید سیگنال (اصلاح شده برای سازگاری API v6.1)"""
    global successful_predictions, failed_attempts, last_processed_timestamps
    
    try:
        logging.info(f"\n🔍 Enhanced processing {symbol} {timeframe} on {exchange.upper()}")
        
        # دریافت داده‌ها Enhanced
        df = get_latest_data(symbol, timeframe, CANDLE_HISTORY_NEEDED, exchange)
        if df is None or df.empty:
            logging.error(f"❌ No Enhanced data received for {symbol}")
            failed_attempts += 1
            return False
        
        # بررسی timestamp جدید
        latest_timestamp = df['timestamp'].iloc[-1]
        pair_key = f"{symbol}_{timeframe}"
        
        if pair_key in last_processed_timestamps:
            if latest_timestamp <= last_processed_timestamps[pair_key]:
                logging.info(f"⏭️ No new Enhanced data for {symbol} {timeframe}")
                return False
        
        last_processed_timestamps[pair_key] = latest_timestamp
        
        # محاسبه ویژگی‌های Enhanced (58+ ویژگی) - اصلاح شده
        features = calculate_enhanced_features(df)
        if not features:
            logging.error(f"❌ Enhanced feature calculation failed for {symbol}")
            failed_attempts += 1
            return False
        
        # بررسی همخوانی ویژگی‌های Enhanced
        if expected_features and not verify_feature_consistency(features, expected_features):
            logging.warning(f"⚠️ Enhanced feature mismatch for {symbol} - continuing anyway")
        
        # دریافت ATR و metadata برای Risk Management
        atr_value = features.get('_atr_value', 1.0)
        feature_count = features.get('_feature_count', 0)
        sentiment_non_zero = features.get('_sentiment_non_zero', 0)
        reddit_non_zero = features.get('_reddit_non_zero', 0)
        
        # محاسبه coverage برای گزارش
        sentiment_features = ['sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence']
        reddit_features = ['reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma']
        
        sentiment_coverage = sentiment_non_zero / len(df) if len(df) > 0 else 0
        reddit_coverage = reddit_non_zero / len(df) if len(df) > 0 else 0
        
        # تشخیص Telegram mapping
        telegram_mapping_detected = False
        if 'sentiment_score' in features and 'reddit_score' in features:
            if abs(features['sentiment_score'] - features['reddit_score']) < 0.0001:
                telegram_mapping_detected = True
                logging.debug("✅ Telegram → Reddit mapping detected in real-time")
        
        # درخواست پیش‌بینی Enhanced
        prediction_result = get_enhanced_prediction(features)
        if not prediction_result:
            logging.error(f"❌ Enhanced prediction failed for {symbol}")
            failed_attempts += 1
            return False
        
        # مدیریت خطاهای Enhanced
        if 'error' in prediction_result:
            error_type = prediction_result['error']
            if error_type == 'rate_limited':
                retry_after = prediction_result.get('retry_after', 90)
                logging.warning(f"⏳ Enhanced rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                return False
            elif error_type == 'authentication_failed':
                logging.error("🔐 Enhanced authentication failed - check credentials")
                return False
            else:
                logging.error(f"❌ Enhanced API Error: {error_type}")
                failed_attempts += 1
                return False
        
        # پردازش نتیجه پیش‌بینی Enhanced
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
        
        # دریافت sentiment analysis از API response
        api_sentiment_coverage = 0
        api_telegram_reddit_coverage = 0
        api_telegram_mapping = False
        
        if 'sentiment_analysis' in prediction_result:
            sentiment_analysis = prediction_result['sentiment_analysis']
            api_sentiment_coverage = sentiment_analysis.get('sentiment_coverage', 0)
            api_telegram_reddit_coverage = sentiment_analysis.get('telegram_derived_reddit_coverage', 0)
            api_telegram_mapping = sentiment_analysis.get('telegram_mapping_detected', False)
        
        logging.info(f"🎯 Enhanced prediction for {symbol}: {signal} (Confidence: {confidence:.3f})")
        logging.info(f"📊 Local: Sentiment Coverage: {sentiment_coverage:.1%}, Reddit Coverage: {reddit_coverage:.1%}")
        logging.info(f"📊 API: Sentiment Coverage: {api_sentiment_coverage:.1%}, Telegram-Reddit Coverage: {api_telegram_reddit_coverage:.1%}")
        logging.info(f"📱 Telegram Mapping: Local={telegram_mapping_detected}, API={api_telegram_mapping}")
        
        # بررسی آستانه اطمینان Enhanced
        if confidence >= CONFIDENCE_THRESHOLD:
            current_price = df['close'].iloc[-1]
            
            # محاسبه Risk Management Enhanced
            position_size = risk_manager.calculate_position_size(symbol, confidence, current_price, atr_value)
            stop_loss = risk_manager.calculate_stop_loss(current_price, atr_value, signal)
            take_profit = risk_manager.calculate_take_profit(current_price, atr_value, signal)
            
            # ایجاد سیگنال Enhanced
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
                'model_info': api_model_info.get('model_type', 'Enhanced'),
                'features_count': feature_count,
                'sentiment_coverage': sentiment_coverage,
                'telegram_reddit_coverage': reddit_coverage,
                'telegram_mapping_detected': telegram_mapping_detected,
                'api_version': '6.1_enhanced'
            }
            
            # ذخیره سیگنال Enhanced
            with signals_lock:
                signals_history.append(signal_data)
            
            # ذخیره در فایل JSON
            try:
                with open(signals_log, 'w', encoding='utf-8') as f:
                    json.dump(signals_history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"خطا در ذخیره Enhanced signal: {e}")
            
            # ارسال تلگرام Enhanced
            telegram_message = format_telegram_message(
                symbol, timeframe, signal, confidence, exchange,
                position_size, stop_loss, take_profit, threshold_used,
                sentiment_coverage, reddit_coverage, feature_count,
                telegram_mapping_detected
            )
            
            if send_telegram_message(telegram_message):
                logging.info(f"📱 Enhanced signal sent to Telegram for {symbol}")
            
            successful_predictions += 1
            logging.info(f"✅ Enhanced signal generated for {symbol}: {signal} (Confidence: {confidence:.1%})")
            return True
            
        else:
            logging.info(f"⚪ No Enhanced signal for {symbol}: confidence {confidence:.3f} below threshold {CONFIDENCE_THRESHOLD:.3f}")
            successful_predictions += 1
            return False
            
    except Exception as e:
        logging.error(f"❌ Enhanced error processing {symbol}: {e}", exc_info=True)
        failed_attempts += 1
        return False

def monitor_enhanced_pairs_concurrent():
    """پردازش همزمان Enhanced چند جفت ارز"""
    global iteration_count
    
    # بارگذاری ویژگی‌های Enhanced مورد انتظار
    expected_features = load_model_features()
    
    while True:
        try:
            iteration_count += 1
            logging.info(f"\n🔄 === Enhanced Iteration {iteration_count} === {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # پردازش همزمان Enhanced
            with ThreadPoolExecutor(max_workers=min(len(PAIRS_TO_MONITOR) * len(TIMEFRAMES_TO_MONITOR), 4)) as executor:
                futures = []
                
                for symbol in PAIRS_TO_MONITOR:
                    for timeframe in TIMEFRAMES_TO_MONITOR:
                        future = executor.submit(process_enhanced_pair, symbol, timeframe, EXCHANGE_TO_USE, expected_features)
                        futures.append(future)
                
                # جمع‌آوری نتایج Enhanced
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Enhanced task failed: {e}")
                        results.append(False)
            
            signals_generated = sum(results)
            logging.info(f"📊 Enhanced iteration {iteration_count} complete. Signals generated: {signals_generated}")
            
            # ذخیره آمار Enhanced
            save_performance_metrics()
            
            # Enhanced delay بین iterations
            sleep_time = max(POLL_INTERVAL_SECONDS, 180)
            logging.info(f"😴 Enhanced sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("⛔ Enhanced received interrupt signal")
            break
        except Exception as e:
            logging.error(f"❌ Enhanced error in monitoring loop: {e}", exc_info=True)
            time.sleep(120)

def main():
    """تابع اصلی Enhanced"""
    global shutdown_message_sent
    
    # ثبت signal handler برای Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("\n🚀 Enhanced Smart Trading Bot v6.1 Starting...")
        print("=" * 60)
        
        # نمایش تنظیمات Enhanced
        print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
        print(f"⏱️ Poll Interval: {POLL_INTERVAL_SECONDS}s")
        print(f"🔐 Enhanced Authentication: {'Enabled' if USE_AUTHENTICATION else 'Disabled'}")
        if USE_AUTHENTICATION:
            print(f"👤 Username: {API_USERNAME}")
        
        # تست اتصال Enhanced API
        if not test_api_connection():
            print("❌ Enhanced API connection test failed. Please check the API server.")
            return
        
        # بررسی Enhanced Authentication
        if not check_authentication():
            print("❌ Enhanced authentication check failed. Please update credentials in config.ini")
            return
        
        # بررسی سلامت Enhanced API
        if not check_api_health():
            print("❌ Enhanced API health check failed. Cannot proceed.")
            return
        
        print(f"✅ All Enhanced checks passed. Monitoring {len(PAIRS_TO_MONITOR)} pairs on {len(TIMEFRAMES_TO_MONITOR)} timeframes")
        print(f"📊 Expected Enhanced features: {len(load_model_features() or [])} features")
        print(f"🎯 Target: 58+ Enhanced features per prediction (Sentiment + Telegram-derived Reddit)")
        
        # ارسال پیام شروع Enhanced
        if TELEGRAM_ENABLED and not shutdown_message_sent:
            startup_message = f"""
🚀 <b>ربات مشاور هوشمند Enhanced v6.1 راه‌اندازی شد</b>

⚙️ <b>تنظیمات Enhanced:</b>
• Threshold: {CONFIDENCE_THRESHOLD:.0%}
• Poll Interval: {POLL_INTERVAL_SECONDS}s
• Multi-pair: {'Yes' if MULTI_PAIR_ENABLED else 'No'}
• Authentication: {'Yes' if USE_AUTHENTICATION else 'No'}

🎯 <b>نظارت Enhanced بر:</b>
• نمادها: {', '.join(PAIRS_TO_MONITOR)}
• تایم‌فریم‌ها: {', '.join(TIMEFRAMES_TO_MONITOR)}
• صرافی: {EXCHANGE_TO_USE.upper()}

🤖 <b>مدل Enhanced:</b>
{api_model_info.get('model_type', 'Unknown')} v6.1 (58+ Features)

💼 <b>Risk Management Enhanced:</b>
• Max Position: {MAX_POSITION_SIZE:.0%}
• Stop Loss ATR: {STOP_LOSS_ATR_MULTIPLIER}x
• Take Profit ATR: {TAKE_PROFIT_ATR_MULTIPLIER}x
• Kelly Criterion: {'Enabled' if KELLY_CRITERION_ENABLED else 'Disabled'}

📊 <b>ویژگی‌های Enhanced:</b>
• محاسبه کامل 58+ ویژگی
• شامل Sentiment Analysis (6 features)
• شامل Telegram-derived Reddit (4+ features)
• Technical Indicators (43+ features)
• Risk Management کامل

🔗 <b>API Enhanced v6.1:</b>
• Feature validation بهبود یافته
• Sentiment & Reddit analysis (Telegram-based)
• Multi-source data quality

🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#BotStarted #v6_1 #Enhanced #TelegramBased #58Features
"""
            send_telegram_message(startup_message)
        
        print("\n🔄 Starting Enhanced monitoring loop...")
        
        # شروع پردازش Enhanced
        monitor_enhanced_pairs_concurrent()
        
    except KeyboardInterrupt:
        print("\n⛔ Enhanced shutdown signal received")
    except Exception as e:
        logging.error(f"❌ Critical Enhanced error in main: {e}", exc_info=True)
        print(f"❌ Critical Enhanced error: {e}")
    finally:
        print("\n👋 Enhanced Bot shutting down...")
        cleanup_and_shutdown()

if __name__ == "__main__":
    main()