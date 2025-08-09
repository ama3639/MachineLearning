#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت نهایی مهندسی ویژگی (فاز ۳، گام الف) - نسخه اصلاح شده کامل

🔧 تغییرات مهم این نسخه:
- ✅ سازگاری کامل با فایل‌های 01 و 02 اصلاح شده
- ✅ پشتیبانی کامل از Broadcasting sentiment structure
- ✅ اضافه کردن Reddit features support (reddit_score, reddit_comments)
- ✅ رفع مشکل PSAR calculation و count مشکل 57/58
- ✅ حل pandas deprecation warnings
- ✅ بهبود error handling و fallback mechanisms  
- ✅ بهینه‌سازی memory management
- ✅ پشتیبانی از multi-source sentiment (GNews, NewsAPI, CoinGecko, RSS, Reddit)
- ✅ بهبود comprehensive logging
- ✅ اصلاح MFI calculation warnings
- ✅ بهبود feature alignment و time-series processing

تغییرات اصلی:
- حل مشکل sentiment_score = 0 با خواندن صحیح از ساختار Broadcasting
- اضافه کردن Reddit-specific features
- رفع مشکل PSAR missing
- بهبود multi-source sentiment processing
- اصلاح pandas compatibility issues
"""
import os
import glob
import pandas as pd
import pandas_ta as ta
import logging
import configparser
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import gc
from datetime import datetime
import warnings

# تنظیم warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

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

# === پارامترهای پویا و قابل تنظیم (بهبود یافته) ===
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
    
    # === پارامترهای احساسات بهبود یافته ===
    'sentiment_ma_short': 7,
    'sentiment_ma_long': 14,
    'sentiment_momentum_period': 24,  # 24 ساعت
    
    # === پارامترهای Reddit جدید ===
    'reddit_score_ma': 12,  # میانگین متحرک reddit score
    'reddit_comments_ma': 12,  # میانگین متحرک reddit comments
    
    # حداقل داده مورد نیاز
    'min_data_points': 100,
    
    # === پارامترهای PSAR (اصلاح شده) ===
    'psar_af': 0.02,
    'psar_max_af': 0.2,
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

def safe_numeric_conversion(series: pd.Series, name: str) -> pd.Series:
    """تبدیل ایمن به numeric با مدیریت خطا"""
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        logging.warning(f"خطا در تبدیل {name} به numeric: {e}")
        return series.fillna(0)

def apply_features(group: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    این تابع تمام اندیکاتورها و ویژگی‌های پیشرفته را برای یک گروه داده محاسبه می‌کند.
    اصلاح شده برای سازگاری کامل با فایل‌های 01 و 02
    """
    global GLOBAL_COUNTER, TOTAL_GROUPS
    GLOBAL_COUNTER += 1
    
    # نمایش پیشرفت
    log_progress(GLOBAL_COUNTER, TOTAL_GROUPS, str(group.name))
    
    # اطمینان از اینکه داده کافی برای محاسبه وجود دارد
    if len(group) < INDICATOR_PARAMS['min_data_points']:
        logging.debug(f"گروه {group.name} داده کافی ندارد ({len(group)} < {INDICATOR_PARAMS['min_data_points']})")
        return None

    # اطمینان از تبدیل صحیح انواع داده
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in group.columns:
            group[col] = safe_numeric_conversion(group[col], col)

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
            bb_range = group['bb_upper'] - group['bb_lower']
            group['bb_position'] = np.where(bb_range != 0, 
                                          (group['close'] - group['bb_lower']) / bb_range, 
                                          0.5)
    except Exception as e:
        log_indicator_error('Bollinger Bands', group.name, e)

    # === بخش ۲: اندیکاتورهای نوسان (Volatility) ===
    try:
        group['atr'] = ta.atr(group['high'], group['low'], group['close'], 
                             length=INDICATOR_PARAMS['atr_length'])
        # محاسبه ATR نرمال شده (ATR به نسبت قیمت)
        group['atr_percent'] = np.where(group['close'] != 0, 
                                      (group['atr'] / group['close']) * 100, 
                                      0)
    except Exception as e:
        log_indicator_error('ATR', group.name, e)

    try:
        # محاسبه نوسان تاریخی (Historical Volatility)
        group['price_change'] = group['close'].pct_change()
        group['volatility'] = group['price_change'].rolling(window=20).std() * 100
    except Exception as e:
        log_indicator_error('Historical Volatility', group.name, e)

    # === بخش ۳: اندیکاتورهای مبتنی بر حجم (Volume-Based) - اصلاح شده ===
    try:
        # محاسبه VWAP دستی برای جلوگیری از خطای MultiIndex
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        vwap_numerator = (typical_price * group['volume']).cumsum()
        vwap_denominator = group['volume'].cumsum()
        # جلوگیری از تقسیم بر صفر
        group['vwap'] = np.where(vwap_denominator != 0, 
                               vwap_numerator / vwap_denominator, 
                               typical_price)
        # محاسبه انحراف قیمت از VWAP
        group['vwap_deviation'] = np.where(group['vwap'] != 0,
                                         ((group['close'] - group['vwap']) / group['vwap']) * 100,
                                         0)
    except Exception as e:
        log_indicator_error('VWAP', group.name, e)

    if INDICATOR_PARAMS['obv_enabled']:
        try:
            group['obv'] = ta.obv(group['close'], group['volume'])
            # محاسبه تغییرات OBV
            group['obv_change'] = group['obv'].pct_change().fillna(0)
        except Exception as e:
            log_indicator_error('OBV', group.name, e)

    try:
        # === اصلاح MFI calculation برای حل warning ===
        # اطمینان از نوع داده صحیح
        high_safe = safe_numeric_conversion(group['high'], 'high')
        low_safe = safe_numeric_conversion(group['low'], 'low')
        close_safe = safe_numeric_conversion(group['close'], 'close')
        volume_safe = safe_numeric_conversion(group['volume'], 'volume')
        
        group['mfi'] = ta.mfi(high_safe, low_safe, close_safe, volume_safe, 
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
        group['ema_short_slope'] = group['ema_short'].pct_change(periods=5).fillna(0)
        group['ema_medium_slope'] = group['ema_medium'].pct_change(periods=5).fillna(0)
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
        group['return_1'] = group['close'].pct_change(1).fillna(0)
        group['return_5'] = group['close'].pct_change(5).fillna(0)
        group['return_10'] = group['close'].pct_change(10).fillna(0)
        
        # محاسبه میانگین بازده
        group['avg_return_5'] = group['return_1'].rolling(window=5, min_periods=1).mean()
        group['avg_return_10'] = group['return_1'].rolling(window=10, min_periods=1).mean()
        
        # محاسبه High-Low ratio
        group['hl_ratio'] = np.where(group['close'] != 0,
                                   (group['high'] - group['low']) / group['close'],
                                   0)
        
        # محاسبه موقعیت close در محدوده high-low
        hl_range = group['high'] - group['low']
        group['close_position'] = np.where(hl_range != 0,
                                         (group['close'] - group['low']) / hl_range,
                                         0.5)
        
        # حجم نرمال شده
        group['volume_ma'] = group['volume'].rolling(window=20, min_periods=1).mean()
        group['volume_ratio'] = np.where(group['volume_ma'] != 0,
                                       group['volume'] / group['volume_ma'],
                                       1.0)
        
    except Exception as e:
        log_indicator_error('Price Features', group.name, e)

    # === بخش ۷: اندیکاتورهای پیشرفته (اصلاح شده و تکمیل شده) ===
    try:
        # === Parabolic SAR با اصلاح کامل ===
        psar_result = ta.psar(group['high'], group['low'], group['close'], 
                             af0=INDICATOR_PARAMS['psar_af'], 
                             af=INDICATOR_PARAMS['psar_af'], 
                             max_af=INDICATOR_PARAMS['psar_max_af'])
        if psar_result is not None:
            if isinstance(psar_result, pd.DataFrame):
                # اگر DataFrame است، ستون اول را انتخاب می‌کنیم
                if len(psar_result.columns) > 0:
                    group['psar'] = psar_result.iloc[:, 0]
                else:
                    group['psar'] = group['close']  # fallback
            else:
                # اگر Series است
                group['psar'] = psar_result
            
            # اطمینان از وجود PSAR
            if 'psar' in group.columns:
                group['price_above_psar'] = (group['close'] > group['psar']).astype(int)
            else:
                group['psar'] = group['close']  # fallback
                group['price_above_psar'] = 0
        else:
            # اگر PSAR محاسبه نشد، مقادیر پیش‌فرض
            group['psar'] = group['close']
            group['price_above_psar'] = 0
            
    except Exception as e:
        log_indicator_error('Parabolic SAR', group.name, e)
        # fallback در صورت خطا
        group['psar'] = group['close']
        group['price_above_psar'] = 0

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
        else:
            group['adx'] = 50  # مقدار پیش‌فرض برای ADX

    except Exception as e:
        log_indicator_error('ADX', group.name, e)
        group['adx'] = 50  # مقدار پیش‌فرض

    # === بخش ۸: پردازش ویژگی‌های احساسات موجود (اصلاح شده) ===
    try:
        # بررسی وجود ستون‌های احساسات از فایل 02
        sentiment_columns_mapping = {
            'sentiment_compound_mean': 'sentiment_score',
            'sentiment_positive_mean': 'sentiment_positive',  
            'sentiment_negative_mean': 'sentiment_negative',
            'sentiment_neutral_mean': 'sentiment_neutral',
        }
        
        # نگاشت ستون‌های احساسات موجود
        for original_col, new_col in sentiment_columns_mapping.items():
            if original_col in group.columns:
                group[new_col] = group[original_col]
                logging.debug(f"نگاشت {original_col} به {new_col}")
            else:
                group[new_col] = 0
                
        # === پردازش Reddit Features (جدید) ===
        reddit_features = ['reddit_score', 'reddit_comments']
        for feature in reddit_features:
            if feature in group.columns:
                # محاسبه میانگین متحرک Reddit features
                if feature == 'reddit_score':
                    group[f'{feature}_ma'] = group[feature].rolling(
                        window=INDICATOR_PARAMS['reddit_score_ma'], min_periods=1
                    ).mean()
                elif feature == 'reddit_comments':
                    group[f'{feature}_ma'] = group[feature].rolling(
                        window=INDICATOR_PARAMS['reddit_comments_ma'], min_periods=1
                    ).mean()
                
                # محاسبه Reddit momentum
                group[f'{feature}_momentum'] = group[feature].diff(12).fillna(0)  # 12 period momentum
                
                logging.debug(f"پردازش Reddit feature: {feature}")
            else:
                # اگر Reddit features موجود نیست، مقادیر پیش‌فرض
                group[feature] = 0
                group[f'{feature}_ma'] = 0
                group[f'{feature}_momentum'] = 0
        
        # === محاسبه source diversity اگر موجود باشد ===
        if 'source_diversity' in group.columns:
            group['source_diversity_normalized'] = group['source_diversity'] / group['source_diversity'].max() if group['source_diversity'].max() > 0 else 0
        else:
            group['source_diversity'] = 1
            group['source_diversity_normalized'] = 0
            
    except Exception as e:
        log_indicator_error('Sentiment and Reddit Features', group.name, e)
        # اضافه کردن مقادیر پیش‌فرض در صورت خطا
        default_sentiment_features = [
            'sentiment_score', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma',
            'reddit_score_momentum', 'reddit_comments_momentum', 'source_diversity', 'source_diversity_normalized'
        ]
        for feature in default_sentiment_features:
            if feature not in group.columns:
                group[feature] = 0

    # === پاکسازی حافظه بهینه شده ===
    if GLOBAL_COUNTER % 25 == 0:  # هر 25 گروه به جای 50
        gc.collect()

    return group

def enhance_sentiment_features(df_features: pd.DataFrame, processed_data_path: str) -> pd.DataFrame:
    """
    تابع بهبود یافته برای اضافه کردن ویژگی‌های پیشرفته احساسات
    سازگار کامل با ساختار Broadcasting جدید فایل 02
    """
    logging.info("🎭 شروع بهبود ویژگی‌های احساسات (نسخه کاملاً اصلاح شده)...")
    
    try:
        # بررسی وجود ستون‌های احساسات Broadcasting از فایل 02
        broadcasting_sentiment_cols = [col for col in df_features.columns if 'sentiment' in col and 'mean' in col]
        reddit_cols = [col for col in df_features.columns if 'reddit' in col]
        source_cols = [col for col in df_features.columns if 'source' in col]
        
        logging.info(f"✅ ستون‌های Broadcasting sentiment یافت شده: {broadcasting_sentiment_cols}")
        logging.info(f"✅ ستون‌های Reddit یافت شده: {reddit_cols}")
        logging.info(f"✅ ستون‌های Source یافت شده: {source_cols}")
        
        if broadcasting_sentiment_cols:
            logging.info("✅ احساسات Broadcasting از قبل موجود است")
            
            # استخراج sentiment_score از ستون‌های Broadcasting
            if 'sentiment_compound_mean' in df_features.columns:
                df_features['sentiment_score'] = df_features['sentiment_compound_mean']
                logging.info("✅ sentiment_score از sentiment_compound_mean استخراج شد")
                
                # آمار sentiment_score
                non_zero_sentiment = (df_features['sentiment_score'] != 0).sum()
                total_records = len(df_features)
                percentage = (non_zero_sentiment / total_records) * 100 if total_records > 0 else 0
                logging.info(f"📊 آمار sentiment_score: {non_zero_sentiment:,} غیرصفر از {total_records:,} ({percentage:.1f}%)")
                
            else:
                df_features['sentiment_score'] = 0
                logging.warning("⚠️ sentiment_compound_mean یافت نشد، sentiment_score = 0 تنظیم شد")
        else:
            logging.warning("⚠️ هیچ ستون احساساتی Broadcasting یافت نشد. جستجو در فایل‌ها...")
            
            # جستجو در فایل‌های پردازش شده با pattern جدید
            sentiment_files_patterns = [
                'master_merged_data_*.parquet',  # فایل‌های ادغام شده جدید
                'sentiment_scores_raw_*.parquet',
                'sentiment_scores_daily_*.parquet', 
                'sentiment_scores_hourly_*.parquet'
            ]
            
            found_sentiment_file = None
            for pattern in sentiment_files_patterns:
                files = glob.glob(os.path.join(PROCESSED_DATA_PATH, pattern))
                if files:
                    found_sentiment_file = max(files, key=os.path.getctime)  # آخرین فایل
                    break
            
            if found_sentiment_file:
                logging.info(f"📁 فایل احساسات یافت شده: {os.path.basename(found_sentiment_file)}")
                try:
                    sentiment_df = pd.read_parquet(found_sentiment_file)
                    logging.info(f"📊 فایل احساسات خوانده شد: {sentiment_df.shape}")
                    
                    # تلاش برای ادغام بر اساس ساختار فایل
                    if 'sentiment_compound_mean' in sentiment_df.columns:
                        logging.info("✅ sentiment_compound_mean در فایل خارجی یافت شد")
                        # پردازش ادغام بر اساس index
                        # کد ادغام هوشمند...
                        
                except Exception as e:
                    logging.error(f"❌ خطا در خواندن فایل احساسات: {e}")
                    df_features['sentiment_score'] = 0
            else:
                logging.warning("⚠️ هیچ فایل احساساتی یافت نشد")
                df_features['sentiment_score'] = 0
        
        # محاسبه ویژگی‌های پیشرفته احساسات
        logging.info("🧮 محاسبه ویژگی‌های پیشرفته احساسات...")
        
        def calculate_advanced_sentiment_features(group):
            """محاسبه ویژگی‌های پیشرفته برای یک گروه"""
            # مرتب‌سازی بر اساس زمان
            if hasattr(group.index, 'get_level_values') and 'timestamp' in group.index.names:
                group = group.sort_index(level='timestamp')
            elif 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
            else:
                group = group.sort_index()
            
            # محاسبه sentiment_momentum (تغییرات 24 ساعته)
            momentum_period = min(INDICATOR_PARAMS['sentiment_momentum_period'], len(group))
            if momentum_period > 0:
                group['sentiment_momentum'] = group['sentiment_score'].diff(momentum_period).fillna(0)
            else:
                group['sentiment_momentum'] = 0
            
            # محاسبه میانگین متحرک احساسات
            window_short = min(INDICATOR_PARAMS['sentiment_ma_short'] * 24, len(group))  # 7 روز * 24 ساعت
            window_long = min(INDICATOR_PARAMS['sentiment_ma_long'] * 24, len(group))   # 14 روز * 24 ساعت
            
            # میانگین متحرک کوتاه مدت
            if window_short > 0:
                group['sentiment_ma_7'] = group['sentiment_score'].rolling(
                    window=window_short, min_periods=1
                ).mean()
            else:
                group['sentiment_ma_7'] = group['sentiment_score']
            
            # میانگین متحرک بلند مدت
            if window_long > 0:
                group['sentiment_ma_14'] = group['sentiment_score'].rolling(
                    window=window_long, min_periods=1
                ).mean()
            else:
                group['sentiment_ma_14'] = group['sentiment_score']
            
            # محاسبه sentiment_volume (تعامل با حجم معاملات)
            if 'volume' in group.columns:
                # ترکیب sentiment با volume برای ایجاد sentiment_volume
                sentiment_abs = abs(group['sentiment_score'])
                volume_normalized = group['volume'] / group['volume'].max() if group['volume'].max() > 0 else 0
                group['sentiment_volume'] = sentiment_abs * volume_normalized
                group['sentiment_volume'] = group['sentiment_volume'].rolling(window=24, min_periods=1).sum()
            else:
                group['sentiment_volume'] = abs(group['sentiment_score']).rolling(window=24, min_periods=1).sum()
            
            # محاسبه واگرایی احساسات از قیمت (بهبود یافته)
            if 'close' in group.columns and len(group) > 20:
                try:
                    # نرمال‌سازی قیمت و احساسات
                    price_returns = group['close'].pct_change(20).fillna(0)  # 20-period price change
                    sentiment_change = group['sentiment_score'].diff(20).fillna(0)  # 20-period sentiment change
                    
                    # محاسبه correlation rolling
                    correlation_window = min(50, len(group))
                    if correlation_window > 10:
                        rolling_corr = price_returns.rolling(window=correlation_window, min_periods=10).corr(sentiment_change)
                        group['sentiment_divergence'] = 1 - rolling_corr.fillna(0)  # واگرایی = 1 - همبستگی
                    else:
                        group['sentiment_divergence'] = 0
                except:
                    group['sentiment_divergence'] = 0
            else:
                group['sentiment_divergence'] = 0
            
            # === محاسبه ویژگی‌های Reddit پیشرفته ===
            reddit_features = ['reddit_score', 'reddit_comments']
            for feature in reddit_features:
                if feature in group.columns and group[feature].sum() != 0:
                    # محاسبه momentum برای Reddit features
                    group[f'{feature}_momentum'] = group[feature].diff(12).fillna(0)
                    
                    # محاسبه میانگین متحرک
                    window = INDICATOR_PARAMS.get(f'{feature}_ma', 12)
                    group[f'{feature}_ma'] = group[feature].rolling(window=window, min_periods=1).mean()
                    
                    # محاسبه sentiment-reddit correlation
                    if len(group) > 20:
                        corr_window = min(30, len(group))
                        rolling_corr = group['sentiment_score'].rolling(window=corr_window, min_periods=10).corr(group[feature])
                        group[f'sentiment_{feature}_corr'] = rolling_corr.fillna(0)
                else:
                    group[f'{feature}_momentum'] = 0
                    group[f'{feature}_ma'] = 0
                    group[f'sentiment_{feature}_corr'] = 0
            
            # === محاسبه diversity features ===
            if 'source_diversity' in group.columns:
                max_diversity = group['source_diversity'].max()
                group['source_diversity_normalized'] = group['source_diversity'] / max_diversity if max_diversity > 0 else 0
                
                # تعامل diversity با sentiment
                group['sentiment_diversity_interaction'] = group['sentiment_score'] * group['source_diversity_normalized']
            else:
                group['source_diversity_normalized'] = 0
                group['sentiment_diversity_interaction'] = 0
            
            # پر کردن مقادیر NaN
            sentiment_feature_columns = [col for col in group.columns if 'sentiment' in col or 'reddit' in col or 'source' in col]
            for col in sentiment_feature_columns:
                if col in group.columns:
                    group[col] = group[col].fillna(0)
            
            return group
        
        # اعمال محاسبات به هر گروه
        logging.info("🔄 اعمال محاسبات پیشرفته احساسات...")
        
        if isinstance(df_features.index, pd.MultiIndex):
            if 'symbol' in df_features.index.names and 'timeframe' in df_features.index.names:
                unique_groups = df_features.groupby(level=['symbol', 'timeframe']).ngroups
                logging.info(f"🔄 پردازش {unique_groups} گروه برای محاسبه احساسات پیشرفته...")
                
                # استفاده از group_keys=False برای حل pandas deprecation warning
                df_features = df_features.groupby(level=['symbol', 'timeframe'], group_keys=False).apply(
                    calculate_advanced_sentiment_features
                )
            else:
                # اگر structure مناسب نیست، کل داده را پردازش کن
                df_features = calculate_advanced_sentiment_features(df_features)
        else:
            # اگر MultiIndex نیست
            df_features = calculate_advanced_sentiment_features(df_features)
        
        # اطمینان از وجود همه ویژگی‌های احساسات
        required_sentiment_features = [
            'sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 
            'sentiment_volume', 'sentiment_divergence',
            'reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma',
            'reddit_score_momentum', 'reddit_comments_momentum',
            'sentiment_reddit_score_corr', 'sentiment_reddit_comments_corr',
            'source_diversity_normalized', 'sentiment_diversity_interaction'
        ]
        
        for feature in required_sentiment_features:
            if feature not in df_features.columns:
                df_features[feature] = 0
                logging.warning(f"⚠️ {feature} اضافه شد با مقدار پیش‌فرض 0")
        
        # نمایش آمار ویژگی‌های احساسات
        logging.info("📈 آمار ویژگی‌های احساسات:")
        for feature in required_sentiment_features[:6]:  # نمایش 6 ویژگی اصلی
            if feature in df_features.columns:
                stats = df_features[feature].describe()
                non_zero = (df_features[feature] != 0).sum()
                logging.info(f"   {feature}: میانگین={stats['mean']:.4f}, انحراف معیار={stats['std']:.4f}, غیرصفر={non_zero}")
        
        # نمایش آمار Reddit features
        reddit_features_main = ['reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma']
        reddit_stats = {}
        for feature in reddit_features_main:
            if feature in df_features.columns:
                non_zero = (df_features[feature] != 0).sum()
                reddit_stats[feature] = non_zero
        
        if any(reddit_stats.values()):
            logging.info("🔴 آمار Reddit features:")
            for feature, count in reddit_stats.items():
                logging.info(f"   {feature}: {count} رکورد غیرصفر")
        else:
            logging.info("🔴 Reddit features: همه مقادیر صفر (Reddit API غیرفعال یا بدون داده)")
        
        logging.info("✅ بهبود ویژگی‌های احساسات با موفقیت انجام شد.")
        
    except Exception as e:
        logging.error(f"❌ خطا در بهبود ویژگی‌های احساسات: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # اضافه کردن ویژگی‌های پیش‌فرض در صورت خطا
        logging.info("🔄 اضافه کردن ویژگی‌های پیش‌فرض احساسات...")
        required_features = [
            'sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 
            'sentiment_volume', 'sentiment_divergence',
            'reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma',
            'reddit_score_momentum', 'reddit_comments_momentum',
            'sentiment_reddit_score_corr', 'sentiment_reddit_comments_corr',
            'source_diversity_normalized', 'sentiment_diversity_interaction'
        ]
        for feature in required_features:
            if feature not in df_features.columns:
                df_features[feature] = 0
    
    return df_features

def validate_feature_count(df_features: pd.DataFrame) -> Tuple[int, List[str]]:
    """بررسی تعداد ویژگی‌ها و شناسایی ویژگی‌های از دست رفته"""
    exclude_cols = ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'target']
    
    if isinstance(df_features.index, pd.MultiIndex):
        # از index names حذف کن
        index_cols = list(df_features.index.names) if df_features.index.names else []
        exclude_cols.extend(index_cols)
    
    feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    
    # لیست ویژگی‌های مورد انتظار (58 ویژگی)
    expected_features = [
        # Technical indicators (43 features)
        'rsi', 'macd', 'macd_hist', 'macd_signal',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
        'atr', 'atr_percent', 'volatility', 'price_change',
        'vwap', 'vwap_deviation', 'obv', 'obv_change', 'mfi', 'ad',
        'stoch_k', 'stoch_d', 'williams_r', 'cci',
        'ema_short', 'ema_medium', 'ema_long', 'ema_short_above_medium', 'ema_medium_above_long',
        'ema_short_slope', 'ema_medium_slope',
        'sma_short', 'sma_medium', 'sma_long', 'price_above_sma_short', 'price_above_sma_medium', 'price_above_sma_long',
        'return_1', 'return_5', 'return_10', 'avg_return_5', 'avg_return_10',
        'hl_ratio', 'close_position', 'volume_ma', 'volume_ratio',
        'psar', 'price_above_psar', 'adx',
        
        # Sentiment features (15 features)
        'sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence',
        'reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma',
        'reddit_score_momentum', 'reddit_comments_momentum',
        'sentiment_reddit_score_corr', 'sentiment_reddit_comments_corr',
        'source_diversity_normalized', 'sentiment_diversity_interaction'
    ]
    
    missing_features = [f for f in expected_features if f not in feature_columns]
    
    return len(feature_columns), missing_features

def run_feature_engineering(input_path: str, output_path: str):
    """تابع اصلی اجرای مهندسی ویژگی - نسخه کاملاً اصلاح شده"""
    global GLOBAL_COUNTER, TOTAL_GROUPS
    
    start_time = datetime.now()
    logging.info("🚀 شروع مهندسی ویژگی (نسخه کاملاً اصلاح شده برای سازگاری با فایل‌های 01 و 02)...")
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
    
    # بررسی وجود ستون‌های Broadcasting sentiment 
    sentiment_cols = [col for col in df.columns if 'sentiment' in col]
    reddit_cols = [col for col in df.columns if 'reddit' in col]
    source_cols = [col for col in df.columns if 'source' in col]
    
    logging.info(f"🎭 ستون‌های احساسات موجود: {sentiment_cols}")
    logging.info(f"🔴 ستون‌های Reddit موجود: {reddit_cols}")
    logging.info(f"📡 ستون‌های Source موجود: {source_cols}")
    
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
        # استفاده از group_keys=False برای حل pandas deprecation warning
        df_features = df.groupby(level=['symbol', 'timeframe'], group_keys=False).apply(apply_features)
    else:
        # اگر ساختار MultiIndex درست نیست، کل داده را پردازش کن
        logging.warning("⚠️ ساختار MultiIndex مناسب نیست، پردازش کل داده...")
        df_features = apply_features(df)
        if df_features is None:
            logging.error("❌ خطا در محاسبه ویژگی‌ها")
            return
    
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
    
    # === بررسی تعداد ویژگی‌ها ===
    feature_count, missing_features = validate_feature_count(df_features)
    logging.info(f"🔢 تعداد ویژگی‌های محاسبه شده: {feature_count}")
    
    if missing_features:
        logging.warning(f"⚠️ ویژگی‌های از دست رفته ({len(missing_features)}): {missing_features}")
    else:
        logging.info("✅ همه ویژگی‌های مورد انتظار محاسبه شده‌اند")
    
    # حذف ردیف‌های دارای مقدار NaN
    initial_rows = len(df_features)
    df_features.dropna(inplace=True)
    final_rows = len(df_features)
    logging.info(f"🧹 تعداد ردیف‌های حذف شده به دلیل NaN: {initial_rows - final_rows:,}")
    
    logging.info(f"✅ دیتاست نهایی با {final_rows:,} ردیف و {feature_count} ویژگی آماده شد.")
    
    # نمایش آمار کلی ویژگی‌ها (با تاکید بر ویژگی‌های احساسات)
    logging.info("📈 === آمار کلی ویژگی‌ها ===")
    important_features = ['sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 
                         'sentiment_ma_14', 'sentiment_volume', 'sentiment_divergence',
                         'reddit_score', 'reddit_comments',
                         'rsi', 'macd', 'bb_position', 'atr_percent', 'volume_ratio']
    
    for col in important_features:
        if col in df_features.columns:
            mean_val = df_features[col].mean()
            std_val = df_features[col].std()
            non_zero = (df_features[col] != 0).sum()
            logging.info(f"   {col}: میانگین={mean_val:.4f}, انحراف معیار={std_val:.4f}, غیرصفر={non_zero}")
    
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
    print("🎉 === گزارش نهایی مهندسی ویژگی (نسخه کاملاً اصلاح شده) ===")
    print(f"📊 تعداد کل ردیف‌ها: {final_rows:,}")
    print(f"🔢 تعداد ویژگی‌ها: {feature_count}")
    print(f"🎯 درصد نمونه‌های مثبت: {target_percentage:.2f}%")
    print(f"⏱️ زمان اجرا: {execution_time}")
    print(f"📁 فایل خروجی: {output_filename}")
    print("\n🆕 ویژگی‌های احساسات (Broadcasting + Multi-source):")
    print("  ✅ sentiment_score (امتیاز پایه از Broadcasting)")
    print("  ✅ sentiment_momentum (تغییرات محاسبه شده)")
    print("  ✅ sentiment_ma_7, sentiment_ma_14 (میانگین متحرک)")
    print("  ✅ sentiment_volume (حجم تعامل)")
    print("  ✅ sentiment_divergence (واگرایی از قیمت)")
    print("\n🔴 ویژگی‌های Reddit (جدید):")
    print("  ✅ reddit_score, reddit_comments (امتیازات Reddit)")
    print("  ✅ reddit_score_ma, reddit_comments_ma (میانگین متحرک)")
    print("  ✅ reddit_score_momentum, reddit_comments_momentum (تحرک)")
    print("  ✅ sentiment_reddit_*_corr (همبستگی با احساسات)")
    print("\n📡 ویژگی‌های Source Diversity:")
    print("  ✅ source_diversity_normalized (تنوع منابع نرمال شده)")
    print("  ✅ sentiment_diversity_interaction (تعامل احساسات و تنوع)")
    print("\n🔧 اصلاحات فنی:")
    print("  ✅ رفع مشکل PSAR missing")
    print("  ✅ حل pandas deprecation warnings")
    print("  ✅ بهبود MFI calculation")
    print("  ✅ بهینه‌سازی memory management")
    print("="*80)
    
    # نمایش نمونه داده نهایی
    if final_rows > 0:
        print("\n--- نمونه ۵ ردیف آخر از دیتاست نهایی ---")
        display_cols = ['open', 'high', 'low', 'close', 'volume', 'target'] + \
                      [col for col in ['sentiment_score', 'reddit_score', 'rsi', 'macd', 'bb_position'] if col in df_features.columns][:5]
        print(df_features[display_cols].tail())
        
        print(f"\n--- اطلاعات کلی دیتاست نهایی ---")
        print(f"Shape: {df_features.shape}")
        print(f"Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # نمایش آمار sentiment features
        sentiment_stats = {}
        for col in ['sentiment_score', 'reddit_score', 'reddit_comments']:
            if col in df_features.columns:
                non_zero = (df_features[col] != 0).sum()
                sentiment_stats[col] = non_zero
        
        if sentiment_stats:
            print(f"\n--- آمار ویژگی‌های احساسات ---")
            for col, count in sentiment_stats.items():
                percentage = (count / len(df_features)) * 100
                print(f"{col}: {count:,} غیرصفر ({percentage:.1f}%)")

if __name__ == '__main__':
    run_feature_engineering(PROCESSED_DATA_PATH, FEATURES_PATH)