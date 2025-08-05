#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت نهایی مهندسی ویژگی (فاز ۳، گام الف) - نسخه اصلاح شده

تغییرات اصلی:
- اصلاح خواندن فایل‌های sentiment جدید (symbol-level aggregation)
- حل مشکل KeyError در ستون‌های sentiment
- سازگاری با ساختار Broadcasting sentiment
- محاسبه صحیح ویژگی‌های sentiment بر اساس داده‌های موجود
- بهبود error handling و fallback mechanisms
"""
import os
import glob
import pandas as pd
import pandas_ta as ta
import logging
import configparser
from typing import Optional, Dict, Any
import numpy as np
import gc
from datetime import datetime

# بخش خواندن پیکربندی
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    PROCESSED_DATA_PATH = config.get('Paths', 'processed')
    FEATURES_PATH = config.get('Paths', 'features')
    LOG_PATH = config.get('Paths', 'logs')
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Error: {e}")
    exit()

# ایجاد پوشه‌های مورد نیاز
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(FEATURES_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# --- تنظیمات لاگ‌گیری پویا ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)

log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# === پارامترهای پویا و قابل تنظیم ===
INDICATOR_PARAMS = {
    # پارامترهای اصلی
    'rsi_length': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_length': 20,
    'bb_std': 2.0,
    
    # اندیکاتورهای جدید
    'atr_length': 14,
    'vwap_anchor': None,  # None برای VWAP روزانه
    
    # اندیکاتورهای نوسان و ترند
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_smooth': 3,
    'williams_r_length': 14,
    'cci_length': 20,
    
    # میانگین‌های متحرک
    'ema_short': 12,
    'ema_medium': 26,
    'ema_long': 50,
    'sma_short': 10,
    'sma_medium': 20,
    'sma_long': 50,
    
    # اندیکاتورهای حجم
    'obv_enabled': True,
    'mfi_length': 14,
    'ad_enabled': True,
    
    # پارامترهای احساسات جدید
    'sentiment_ma_short': 7,
    'sentiment_ma_long': 14,
    'sentiment_momentum_period': 24,  # 24 ساعت
    
    # حداقل داده مورد نیاز
    'min_data_points': 100,
}

# پارامترهای مهندسی ویژگی و هدف
TARGET_FUTURE_PERIODS = 24
TARGET_PROFIT_PERCENT = 0.02

# کانتر global برای tracking
GLOBAL_COUNTER = 0
TOTAL_GROUPS = 0

def log_indicator_error(indicator_name: str, group_name: Any, error: Exception):
    """تابع یکپارچه برای لاگ کردن خطاهای اندیکاتورها"""
    logging.warning(f"خطا در محاسبه {indicator_name} برای گروه {group_name}: {error}")

def log_progress(current: int, total: int, group_name: str = ""):
    """نمایش پیشرفت پردازش"""
    if total > 0:
        progress = (current / total) * 100
        if current % max(1, total // 20) == 0:  # هر 5% گزارش
            logging.info(f"🔄 پیشرفت: {progress:.1f}% ({current}/{total}) - {group_name}")

def apply_features(group: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    این تابع تمام اندیکاتورها و ویژگی‌های پیشرفته را برای یک گروه داده محاسبه می‌کند.
    """
    global GLOBAL_COUNTER, TOTAL_GROUPS
    GLOBAL_COUNTER += 1
    
    # نمایش پیشرفت
    log_progress(GLOBAL_COUNTER, TOTAL_GROUPS, str(group.name))
    
    # اطمینان از اینکه داده کافی برای محاسبه وجود دارد
    if len(group) < INDICATOR_PARAMS['min_data_points']:
        logging.debug(f"گروه {group.name} داده کافی ندارد ({len(group)} < {INDICATOR_PARAMS['min_data_points']})")
        return None

    # === بخش ۱: اندیکاتورهای ترند و قیمت ===
    try:
        group['rsi'] = ta.rsi(group['close'], length=INDICATOR_PARAMS['rsi_length'])
    except Exception as e:
        log_indicator_error('RSI', group.name, e)

    try:
        macd = ta.macd(group['close'], 
                      fast=INDICATOR_PARAMS['macd_fast'], 
                      slow=INDICATOR_PARAMS['macd_slow'], 
                      signal=INDICATOR_PARAMS['macd_signal'])
        if macd is not None and not macd.empty:
            group['macd'] = macd[f'MACD_{INDICATOR_PARAMS["macd_fast"]}_{INDICATOR_PARAMS["macd_slow"]}_{INDICATOR_PARAMS["macd_signal"]}']
            group['macd_hist'] = macd[f'MACDh_{INDICATOR_PARAMS["macd_fast"]}_{INDICATOR_PARAMS["macd_slow"]}_{INDICATOR_PARAMS["macd_signal"]}']
            group['macd_signal'] = macd[f'MACDs_{INDICATOR_PARAMS["macd_fast"]}_{INDICATOR_PARAMS["macd_slow"]}_{INDICATOR_PARAMS["macd_signal"]}']
    except Exception as e:
        log_indicator_error('MACD', group.name, e)

    try:
        bbands = ta.bbands(group['close'], 
                          length=INDICATOR_PARAMS['bb_length'], 
                          std=INDICATOR_PARAMS['bb_std'])
        if bbands is not None and not bbands.empty:
            group['bb_upper'] = bbands[f'BBU_{INDICATOR_PARAMS["bb_length"]}_{INDICATOR_PARAMS["bb_std"]}']
            group['bb_middle'] = bbands[f'BBM_{INDICATOR_PARAMS["bb_length"]}_{INDICATOR_PARAMS["bb_std"]}']
            group['bb_lower'] = bbands[f'BBL_{INDICATOR_PARAMS["bb_length"]}_{INDICATOR_PARAMS["bb_std"]}']
            # محاسبه موقعیت قیمت در کانال Bollinger Bands
            group['bb_position'] = (group['close'] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'])
    except Exception as e:
        log_indicator_error('Bollinger Bands', group.name, e)

    # === بخش ۲: اندیکاتورهای نوسان (Volatility) ===
    try:
        group['atr'] = ta.atr(group['high'], group['low'], group['close'], 
                             length=INDICATOR_PARAMS['atr_length'])
        # محاسبه ATR نرمال شده (ATR به نسبت قیمت)
        group['atr_percent'] = (group['atr'] / group['close']) * 100
    except Exception as e:
        log_indicator_error('ATR', group.name, e)

    try:
        # محاسبه نوسان تاریخی (Historical Volatility)
        group['price_change'] = group['close'].pct_change()
        group['volatility'] = group['price_change'].rolling(window=20).std() * 100
    except Exception as e:
        log_indicator_error('Historical Volatility', group.name, e)

    # === بخش ۳: اندیکاتورهای مبتنی بر حجم (Volume-Based) ===
    try:
        # محاسبه VWAP دستی برای جلوگیری از خطای MultiIndex
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        vwap_numerator = (typical_price * group['volume']).cumsum()
        vwap_denominator = group['volume'].cumsum()
        group['vwap'] = vwap_numerator / vwap_denominator
        # محاسبه انحراف قیمت از VWAP
        group['vwap_deviation'] = ((group['close'] - group['vwap']) / group['vwap']) * 100
    except Exception as e:
        log_indicator_error('VWAP', group.name, e)

    if INDICATOR_PARAMS['obv_enabled']:
        try:
            group['obv'] = ta.obv(group['close'], group['volume'])
            # محاسبه تغییرات OBV
            group['obv_change'] = group['obv'].pct_change()
        except Exception as e:
            log_indicator_error('OBV', group.name, e)

    try:
        group['mfi'] = ta.mfi(group['high'], group['low'], group['close'], group['volume'], 
                             length=INDICATOR_PARAMS['mfi_length'])
    except Exception as e:
        log_indicator_error('MFI', group.name, e)

    if INDICATOR_PARAMS['ad_enabled']:
        try:
            group['ad'] = ta.ad(group['high'], group['low'], group['close'], group['volume'])
        except Exception as e:
            log_indicator_error('A/D Line', group.name, e)

    # === بخش ۴: اندیکاتورهای اوسیلاتور ===
    try:
        stoch = ta.stoch(group['high'], group['low'], group['close'], 
                        k=INDICATOR_PARAMS['stoch_k'], 
                        d=INDICATOR_PARAMS['stoch_d'], 
                        smooth_k=INDICATOR_PARAMS['stoch_smooth'])
        if stoch is not None and not stoch.empty:
            group['stoch_k'] = stoch[f'STOCHk_{INDICATOR_PARAMS["stoch_k"]}_{INDICATOR_PARAMS["stoch_d"]}_{INDICATOR_PARAMS["stoch_smooth"]}']
            group['stoch_d'] = stoch[f'STOCHd_{INDICATOR_PARAMS["stoch_k"]}_{INDICATOR_PARAMS["stoch_d"]}_{INDICATOR_PARAMS["stoch_smooth"]}']
    except Exception as e:
        log_indicator_error('Stochastic', group.name, e)

    try:
        group['williams_r'] = ta.willr(group['high'], group['low'], group['close'], 
                                      length=INDICATOR_PARAMS['williams_r_length'])
    except Exception as e:
        log_indicator_error('Williams %R', group.name, e)

    try:
        group['cci'] = ta.cci(group['high'], group['low'], group['close'], 
                             length=INDICATOR_PARAMS['cci_length'])
    except Exception as e:
        log_indicator_error('CCI', group.name, e)

    # === بخش ۵: میانگین‌های متحرک ===
    try:
        group['ema_short'] = ta.ema(group['close'], length=INDICATOR_PARAMS['ema_short'])
        group['ema_medium'] = ta.ema(group['close'], length=INDICATOR_PARAMS['ema_medium'])
        group['ema_long'] = ta.ema(group['close'], length=INDICATOR_PARAMS['ema_long'])
        
        # سیگنال‌های cross over
        group['ema_short_above_medium'] = (group['ema_short'] > group['ema_medium']).astype(int)
        group['ema_medium_above_long'] = (group['ema_medium'] > group['ema_long']).astype(int)
        
        # محاسبه شیب EMA (ترند)
        group['ema_short_slope'] = group['ema_short'].pct_change(periods=5)
        group['ema_medium_slope'] = group['ema_medium'].pct_change(periods=5)
    except Exception as e:
        log_indicator_error('EMA', group.name, e)

    try:
        group['sma_short'] = ta.sma(group['close'], length=INDICATOR_PARAMS['sma_short'])
        group['sma_medium'] = ta.sma(group['close'], length=INDICATOR_PARAMS['sma_medium'])
        group['sma_long'] = ta.sma(group['close'], length=INDICATOR_PARAMS['sma_long'])
        
        # موقعیت قیمت نسبت به SMA
        group['price_above_sma_short'] = (group['close'] > group['sma_short']).astype(int)
        group['price_above_sma_medium'] = (group['close'] > group['sma_medium']).astype(int)
        group['price_above_sma_long'] = (group['close'] > group['sma_long']).astype(int)
    except Exception as e:
        log_indicator_error('SMA', group.name, e)

    # === بخش ۶: ویژگی‌های قیمت خام ===
    try:
        # محاسبه بازده‌های مختلف
        group['return_1'] = group['close'].pct_change(1)
        group['return_5'] = group['close'].pct_change(5)
        group['return_10'] = group['close'].pct_change(10)
        
        # محاسبه میانگین بازده
        group['avg_return_5'] = group['return_1'].rolling(window=5).mean()
        group['avg_return_10'] = group['return_1'].rolling(window=10).mean()
        
        # محاسبه High-Low ratio
        group['hl_ratio'] = (group['high'] - group['low']) / group['close']
        
        # محاسبه موقعیت close در محدوده high-low
        group['close_position'] = (group['close'] - group['low']) / (group['high'] - group['low'])
        
        # حجم نرمال شده
        group['volume_ma'] = group['volume'].rolling(window=20).mean()
        group['volume_ratio'] = group['volume'] / group['volume_ma']
        
    except Exception as e:
        log_indicator_error('Price Features', group.name, e)

    # === بخش ۷: اندیکاتورهای پیشرفته (اختیاری) ===
    try:
        # Parabolic SAR با بررسی نوع نتیجه
        psar_result = ta.psar(group['high'], group['low'], group['close'])
        if psar_result is not None:
            if isinstance(psar_result, pd.DataFrame):
                # اگر DataFrame است، ستون اول را انتخاب می‌کنیم
                group['psar'] = psar_result.iloc[:, 0]
            else:
                # اگر Series است
                group['psar'] = psar_result
            group['price_above_psar'] = (group['close'] > group['psar']).astype(int)
    except Exception as e:
        log_indicator_error('Parabolic SAR', group.name, e)

    try:
        # ADX با بررسی نوع نتیجه
        adx_result = ta.adx(group['high'], group['low'], group['close'], length=14)
        if adx_result is not None:
            if isinstance(adx_result, pd.DataFrame):
                # اگر DataFrame است، ستون ADX را انتخاب می‌کنیم
                if 'ADX_14' in adx_result.columns:
                    group['adx'] = adx_result['ADX_14']
                else:
                    group['adx'] = adx_result.iloc[:, 0]  # ستون اول
            else:
                # اگر Series است
                group['adx'] = adx_result

    except Exception as e:
        log_indicator_error('ADX', group.name, e)

    # پاکسازی حافظه
    if GLOBAL_COUNTER % 50 == 0:  # هر 50 گروه
        gc.collect()

    return group

def enhance_sentiment_features(df_features: pd.DataFrame, processed_data_path: str) -> pd.DataFrame:
    """
    تابع اصلاح شده برای اضافه کردن ویژگی‌های پیشرفته احساسات
    سازگار با ساختار جدید فایل‌های sentiment (Broadcasting)
    """
    logging.info("🎭 شروع بهبود ویژگی‌های احساسات (نسخه اصلاح شده)...")
    
    try:
        # بررسی اینکه آیا احساسات از قبل در فایل موجود است
        existing_sentiment_cols = [col for col in df_features.columns if 'sentiment' in col]
        
        if existing_sentiment_cols:
            logging.info(f"✅ احساسات از قبل موجود است: {existing_sentiment_cols}")
            
            # استخراج sentiment_score از ستون‌های موجود
            if 'sentiment_compound_mean' in df_features.columns:
                df_features['sentiment_score'] = df_features['sentiment_compound_mean']
                logging.info("✅ sentiment_score از sentiment_compound_mean استخراج شد")
            else:
                df_features['sentiment_score'] = 0
                logging.warning("⚠️ sentiment_compound_mean یافت نشد، sentiment_score = 0 تنظیم شد")
        else:
            logging.warning("⚠️ هیچ ستون احساساتی یافت نشد. تلاش برای خواندن از فایل...")
            
            # جستجو مستقیم در مسیر processed برای فایل‌های احساسات
            sentiment_raw_files = glob.glob(os.path.join(PROCESSED_DATA_PATH, 'sentiment_scores_raw_*.parquet'))
            sentiment_daily_files = glob.glob(os.path.join(PROCESSED_DATA_PATH, 'sentiment_scores_daily_*.parquet'))
            sentiment_hourly_files = glob.glob(os.path.join(PROCESSED_DATA_PATH, 'sentiment_scores_hourly_*.parquet'))
            
            logging.info(f"📁 فایل‌های احساسات یافت شده: Raw={len(sentiment_raw_files)}, Daily={len(sentiment_daily_files)}, Hourly={len(sentiment_hourly_files)}")
            
            if not (sentiment_raw_files or sentiment_daily_files or sentiment_hourly_files):
                logging.warning("⚠️ هیچ فایل احساساتی یافت نشد. اضافه کردن مقادیر پیش‌فرض.")
                df_features['sentiment_score'] = 0
            else:
                # تلاش برای خواندن و ادغام احساسات
                logging.info("🔄 تلاش برای خواندن فایل‌های احساسات...")
                df_features['sentiment_score'] = 0  # پیش‌فرض
                
                # اگر فایل hourly موجود است
                if sentiment_hourly_files:
                    try:
                        latest_hourly_file = max(sentiment_hourly_files, key=os.path.getctime)
                        logging.info(f"📊 خواندن فایل احساسات ساعتی: {os.path.basename(latest_hourly_file)}")
                        sentiment_hourly_df = pd.read_parquet(latest_hourly_file)
                        
                        # محاولة ادغام ساده
                        if 'symbol' in sentiment_hourly_df.columns and 'sentiment_compound_mean' in sentiment_hourly_df.columns:
                            # گروه‌بندی بر اساس symbol و میانگین‌گیری
                            symbol_sentiment = sentiment_hourly_df.groupby('symbol')['sentiment_compound_mean'].mean().to_dict()
                            
                            # اطمینان از وجود ستون symbol در df_features
                            if df_features.index.names and 'symbol' in df_features.index.names:
                                df_features = df_features.reset_index()
                                df_features['sentiment_score'] = df_features['symbol'].map(symbol_sentiment).fillna(0)
                                df_features.set_index(['symbol', 'timeframe', 'timestamp'], inplace=True)
                                logging.info("✅ احساسات با موفقیت ادغام شد")
                            else:
                                logging.warning("⚠️ ساختار index مناسب نیست")
                    except Exception as e:
                        logging.error(f"❌ خطا در خواندن فایل hourly: {e}")
        
        # محاسبه ویژگی‌های پیشرفته احساسات بر اساس sentiment_score
        logging.info("🧮 محاسبه ویژگی‌های پیشرفته احساسات...")
        
        def calculate_advanced_sentiment_features(group):
            """محاسبه ویژگی‌های پیشرفته برای یک گروه"""
            # مرتب‌سازی بر اساس زمان
            group = group.sort_values('timestamp') if 'timestamp' in group.columns else group.sort_index()
            
            # محاسبه sentiment_momentum (تغییرات 24 ساعته)
            if len(group) >= INDICATOR_PARAMS['sentiment_momentum_period']:
                group['sentiment_momentum'] = group['sentiment_score'].diff(
                    INDICATOR_PARAMS['sentiment_momentum_period']
                )
            else:
                group['sentiment_momentum'] = 0
            
            # محاسبه میانگین متحرک احساسات
            window_short = min(INDICATOR_PARAMS['sentiment_ma_short'] * 24, len(group))  # 7 روز * 24 ساعت
            window_long = min(INDICATOR_PARAMS['sentiment_ma_long'] * 24, len(group))   # 14 روز * 24 ساعت
            
            if window_short > 0:
                group['sentiment_ma_7'] = group['sentiment_score'].rolling(
                    window=window_short, min_periods=1
                ).mean()
            else:
                group['sentiment_ma_7'] = group['sentiment_score']
            
            if window_long > 0:
                group['sentiment_ma_14'] = group['sentiment_score'].rolling(
                    window=window_long, min_periods=1
                ).mean()
            else:
                group['sentiment_ma_14'] = group['sentiment_score']
            
            # محاسبه sentiment_volume (تعداد اخبار - شبیه‌سازی شده)
            # از آنجایی که داده‌های خام اخبار در دسترس نیست، از یک metric تقریبی استفاده می‌کنیم
            group['sentiment_volume'] = abs(group['sentiment_score']).rolling(window=24, min_periods=1).sum()
            
            # محاسبه واگرایی احساسات از قیمت
            if 'close' in group.columns and group['sentiment_score'].std() > 0:
                try:
                    price_normalized = (group['close'] - group['close'].mean()) / group['close'].std()
                    sentiment_normalized = (group['sentiment_score'] - group['sentiment_score'].mean()) / group['sentiment_score'].std()
                    group['sentiment_divergence'] = price_normalized - sentiment_normalized
                except:
                    group['sentiment_divergence'] = 0
            else:
                group['sentiment_divergence'] = 0
            
            # پر کردن مقادیر NaN
            for col in ['sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence']:
                if col in group.columns:
                    group[col] = group[col].fillna(0)
            
            return group
        
        # اعمال محاسبات به هر گروه
        if isinstance(df_features.index, pd.MultiIndex):
            if 'symbol' in df_features.index.names and 'timeframe' in df_features.index.names:
                df_features = df_features.reset_index()
                unique_groups = df_features.groupby(['symbol', 'timeframe']).ngroups
                logging.info(f"🔄 پردازش {unique_groups} گروه برای محاسبه احساسات...")
                
                df_features = df_features.groupby(['symbol', 'timeframe']).apply(
                    calculate_advanced_sentiment_features
                ).reset_index(drop=True)
                
                # بازگرداندن index
                df_features.set_index(['symbol', 'timeframe', 'timestamp'], inplace=True)
            else:
                # اگر structure مناسب نیست، کل داده را پردازش کن
                df_features = calculate_advanced_sentiment_features(df_features)
        else:
            # اگر MultiIndex نیست
            df_features = calculate_advanced_sentiment_features(df_features)
        
        # اطمینان از وجود همه ویژگی‌های احساسات
        required_sentiment_features = ['sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence']
        for feature in required_sentiment_features:
            if feature not in df_features.columns:
                df_features[feature] = 0
                logging.warning(f"⚠️ {feature} اضافه شد با مقدار پیش‌فرض 0")
        
        # نمایش آمار ویژگی‌های احساسات
        logging.info("📈 آمار ویژگی‌های احساسات:")
        for feature in required_sentiment_features:
            if feature in df_features.columns:
                stats = df_features[feature].describe()
                non_zero = (df_features[feature] != 0).sum()
                logging.info(f"   {feature}: میانگین={stats['mean']:.4f}, انحراف معیار={stats['std']:.4f}, غیرصفر={non_zero}")
        
        logging.info("✅ بهبود ویژگی‌های احساسات با موفقیت انجام شد.")
        
    except Exception as e:
        logging.error(f"❌ خطا در بهبود ویژگی‌های احساسات: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # اضافه کردن ویژگی‌های پیش‌فرض در صورت خطا
        logging.info("🔄 اضافه کردن ویژگی‌های پیش‌فرض احساسات...")
        required_features = ['sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence']
        for feature in required_features:
            if feature not in df_features.columns:
                df_features[feature] = 0
    
    return df_features

def run_feature_engineering(input_path: str, output_path: str):
    """تابع اصلی اجرای مهندسی ویژگی"""
    global GLOBAL_COUNTER, TOTAL_GROUPS
    
    start_time = datetime.now()
    logging.info("🚀 شروع مهندسی ویژگی (نسخه اصلاح شده با احساسات Broadcasting)...")
    logging.info(f"📋 پارامترهای اندیکاتور: {INDICATOR_PARAMS}")
    
    # یافتن آخرین فایل داده
    list_of_files = glob.glob(os.path.join(input_path, 'master_*_data_*.parquet'))
    if not list_of_files:
        logging.error(f"❌ هیچ فایل داده اصلی در مسیر '{input_path}' یافت نشد.")
        return
    latest_file = max(list_of_files, key=os.path.getctime)
    logging.info(f"📂 در حال خواندن فایل داده اصلی: {os.path.basename(latest_file)}")
    
    # خواندن داده
    df = pd.read_parquet(latest_file)
    logging.info(f"📊 تعداد ردیف‌های اولیه: {len(df):,}")
    
    # بررسی ساختار فایل
    logging.info(f"🔍 ساختار فایل: Index={df.index.names}, Columns={list(df.columns)}")
    
    # محاسبه تعداد کل گروه‌ها
    if isinstance(df.index, pd.MultiIndex) and 'symbol' in df.index.names and 'timeframe' in df.index.names:
        TOTAL_GROUPS = df.groupby(level=['symbol', 'timeframe']).ngroups
    else:
        TOTAL_GROUPS = 1
    logging.info(f"🔢 تعداد کل گروه‌ها برای پردازش: {TOTAL_GROUPS:,}")
    
    # اعمال ویژگی‌ها
    logging.info("⚙️ شروع اعمال ویژگی‌ها به صورت گروه‌بندی شده...")
    logging.info(f"💡 حداقل داده مورد نیاز برای هر گروه: {INDICATOR_PARAMS['min_data_points']}")
    
    if isinstance(df.index, pd.MultiIndex) and 'symbol' in df.index.names and 'timeframe' in df.index.names:
        df_features = df.groupby(level=['symbol', 'timeframe'], group_keys=False).apply(apply_features)
    else:
        # اگر ساختار MultiIndex درست نیست، کل داده را پردازش کن
        logging.warning("⚠️ ساختار MultiIndex مناسب نیست، پردازش کل داده...")
        df_features = apply_features(df)
        if df_features is None:
            logging.error("❌ خطا در محاسبه ویژگی‌ها")
            return
        df_features = pd.DataFrame([df_features])  # تبدیل به DataFrame
    
    # پاکسازی حافظه
    del df
    gc.collect()
    
    # بررسی نتیجه
    if df_features is None or (isinstance(df_features, pd.DataFrame) and df_features.empty):
        logging.error("❌ خطا در محاسبه ویژگی‌ها. عملیات متوقف شد.")
        return
    
    # حذف گروه‌هایی که به دلیل نداشتن داده کافی، None برگردانده‌اند
    if isinstance(df_features, pd.DataFrame):
        initial_rows = len(df_features)
        df_features.dropna(how='all', inplace=True)
        final_rows_after_filter = len(df_features)
        logging.info(f"📊 تعداد ردیف‌ها پس از محاسبه ویژگی‌ها: {final_rows_after_filter:,} (حذف شده: {initial_rows - final_rows_after_filter:,})")
    
    # --- بخش بهبود یافته: ادغام داده‌های احساسات ---
    df_features = enhance_sentiment_features(df_features, input_path)

    logging.info("🎯 محاسبه ویژگی‌ها تمام شد. در حال ایجاد متغیر هدف...")
    
    # اطمینان از وجود index صحیح
    if not isinstance(df_features.index, pd.MultiIndex):
        if 'symbol' in df_features.columns and 'timeframe' in df_features.columns and 'timestamp' in df_features.columns:
            df_features.set_index(['symbol', 'timeframe', 'timestamp'], inplace=True)
        else:
            logging.warning("⚠️ نمی‌توان index صحیح تنظیم کرد")
    
    # محاسبه متغیر هدف با روش پویا
    logging.info(f"🔮 محاسبه target با {TARGET_FUTURE_PERIODS} پریود آینده و {TARGET_PROFIT_PERCENT*100}% سود")
    
    if isinstance(df_features.index, pd.MultiIndex) and 'symbol' in df_features.index.names and 'timeframe' in df_features.index.names:
        df_features['future_close'] = df_features.groupby(level=['symbol', 'timeframe'])['close'].shift(-TARGET_FUTURE_PERIODS)
    else:
        df_features['future_close'] = df_features['close'].shift(-TARGET_FUTURE_PERIODS)
    
    df_features['target'] = (df_features['future_close'] > df_features['close'] * (1 + TARGET_PROFIT_PERCENT)).astype(int)
    
    # حذف ستون کمکی و ردیف‌های ناقص
    df_features.drop(columns=['future_close'], inplace=True)
    
    # شمارش کل ویژگی‌ها
    exclude_cols = ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'target']
    if isinstance(df_features.index, pd.MultiIndex):
        feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    else:
        feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    
    logging.info(f"🔢 تعداد ویژگی‌های محاسبه شده: {len(feature_columns)}")
    
    # حذف ردیف‌های دارای مقدار NaN
    initial_rows = len(df_features)
    df_features.dropna(inplace=True)
    final_rows = len(df_features)
    logging.info(f"🧹 تعداد ردیف‌های حذف شده به دلیل NaN: {initial_rows - final_rows:,}")
    
    logging.info(f"✅ دیتاست نهایی با {final_rows:,} ردیف و {len(feature_columns)} ویژگی آماده شد.")
    
    # نمایش آمار کلی ویژگی‌ها (با تاکید بر ویژگی‌های احساسات)
    logging.info("📈 === آمار کلی ویژگی‌ها ===")
    important_features = ['sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 
                         'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence',
                         'rsi', 'macd', 'bb_position', 'atr_percent', 'volume_ratio']
    
    for col in important_features:
        if col in df_features.columns:
            mean_val = df_features[col].mean()
            std_val = df_features[col].std()
            logging.info(f"   {col}: میانگین={mean_val:.4f}, انحراف معیار={std_val:.4f}")
    
    # بررسی توزیع target
    target_distribution = df_features['target'].value_counts()
    logging.info(f"🎯 توزیع متغیر هدف: {target_distribution.to_dict()}")
    target_percentage = (target_distribution.get(1, 0) / len(df_features)) * 100
    logging.info(f"📊 درصد نمونه‌های مثبت: {target_percentage:.2f}%")
    
    # ذخیره دیتاست نهایی
    logging.info("💾 شروع ذخیره‌سازی...")
    os.makedirs(output_path, exist_ok=True)
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # ذخیره فایل اصلی
    output_filename = f'final_dataset_for_training_{timestamp_str}.parquet'
    output_file_path = os.path.join(output_path, output_filename)
    df_features.to_parquet(output_file_path)
    logging.info(f"✅ دیتاست نهایی آماده آموزش در مسیر '{output_file_path}' (فرمت Parquet) ذخیره شد.")
    
    # ذخیره فایل CSV برای بررسی دستی
    csv_output_filename = f'final_dataset_for_training_{timestamp_str}.csv'
    csv_output_file_path = os.path.join(output_path, csv_output_filename)
    sample_size = min(1000, len(df_features))
    df_features.head(sample_size).to_csv(csv_output_file_path, index=True)
    logging.info(f"📄 نمونه CSV ({sample_size:,} ردیف اول) در مسیر '{csv_output_file_path}' ذخیره شد.")
    
    # محاسبه زمان اجرا
    end_time = datetime.now()
    execution_time = end_time - start_time
    logging.info(f"⏱️ زمان اجرای کل: {execution_time}")
    
    # گزارش نهایی
    print("\n" + "="*80)
    print("🎉 === گزارش نهایی مهندسی ویژگی (نسخه اصلاح شده) ===")
    print(f"📊 تعداد کل ردیف‌ها: {final_rows:,}")
    print(f"🔢 تعداد ویژگی‌ها: {len(feature_columns)}")
    print(f"🎯 درصد نمونه‌های مثبت: {target_percentage:.2f}%")
    print(f"⏱️ زمان اجرا: {execution_time}")
    print(f"📁 فایل خروجی: {output_filename}")
    print("\n🆕 ویژگی‌های احساسات (Broadcasting):")
    print("  ✅ sentiment_score (امتیاز پایه از Broadcasting)")
    print("  ✅ sentiment_momentum (تغییرات محاسبه شده)")
    print("  ✅ sentiment_ma_7 (میانگین متحرک 7 واحد)")
    print("  ✅ sentiment_ma_14 (میانگین متحرک 14 واحد)")
    print("  ✅ sentiment_volume (حجم تقریبی)")
    print("  ✅ sentiment_divergence (واگرایی از قیمت)")
    print("="*80)
    
    # نمایش نمونه داده نهایی
    if final_rows > 0:
        print("\n--- نمونه ۵ ردیف آخر از دیتاست نهایی ---")
        display_cols = ['open', 'high', 'low', 'close', 'volume', 'target'] + \
                      [col for col in ['sentiment_score', 'rsi', 'macd', 'bb_position'] if col in df_features.columns][:4]
        print(df_features[display_cols].tail())
        
        print(f"\n--- اطلاعات کلی دیتاست نهایی ---")
        print(f"Shape: {df_features.shape}")
        print(f"Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

if __name__ == '__main__':
    run_feature_engineering(PROCESSED_DATA_PATH, FEATURES_PATH)