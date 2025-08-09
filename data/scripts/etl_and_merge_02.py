#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت یکپارچه ETL و پردازش احساسات (نسخه اصلاح شده نهایی)

🔧 تغییرات مهم این نسخه:
- ✅ سازگاری کامل با فایل‌های جدید fetch_01_fixed_clean.py
- ✅ پشتیبانی از منابع خبری جدید (Reddit, NewsAPI, RSS, CoinGecko)
- ✅ تشخیص نام‌گذاری جدید فایل‌ها
- ✅ پردازش API sources مختلف
- ✅ حل مشکل عدم تطبیق زمانی بین اخبار و قیمت‌ها
- ✅ استفاده از Broadcasting احساسات برای کل دوره
- ✅ اصلاح منطق ادغام برای حفظ احساسات
- ✅ افزودن fallback برای داده‌های بدون احساسات

تغییرات اصلی:
- حل مشکل عدم تطبیق زمانی بین اخبار و قیمت‌ها
- استفاده از Broadcasting احساسات برای کل دوره
- اصلاح منطق ادغام برای حفظ احساسات
- افزودن fallback برای داده‌های بدون احساسات
- پشتیبانی کامل از منابع خبری جدید
"""

import os
import re
import glob
import pandas as pd
import logging
import configparser
from typing import List, Dict, Optional, Tuple, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import numpy as np

# بخش خواندن پیکربندی
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    RAW_DATA_PATH = config.get('Paths', 'raw')
    PROCESSED_DATA_PATH = config.get('Paths', 'processed')
    LOG_PATH = config.get('Paths', 'logs')
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Error: {e}")
    exit()

# --- تنظیمات لاگ‌گیری ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)

log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# ایجاد پوشه‌های مورد نیاز
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# استفاده مستقیم از مسیرهای کانفیگ بدون زیرپوشه
price_raw_path = RAW_DATA_PATH
news_raw_path = RAW_DATA_PATH
processed_price_path = PROCESSED_DATA_PATH
processed_sentiment_path = PROCESSED_DATA_PATH

# --- کلاس پردازش یکپارچه داده (بهبود یافته) ---
class UnifiedDataProcessor:
    """کلاس یکپارچه برای پردازش داده‌های قیمت و احساسات - نسخه بهبود یافته"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.price_data = None
        self.sentiment_data = None
        
        # === منابع خبری شناسایی شده ===
        self.known_news_sources = {
            'GNews': 'gnews',
            'NewsAPI': 'newsapi', 
            'CoinGecko': 'coingecko',
            'RSS': 'rss',
            'Reddit': 'reddit'
        }
        
        logging.info("🚀 Enhanced Unified Data Processor اولیه‌سازی شد")
        logging.info(f"🔗 پشتیبانی از منابع خبری: {list(self.known_news_sources.keys())}")
    
    def debug_timestamp_column(self, df: pd.DataFrame, context: str = ""):
        """تابع کمکی برای debug کردن مشکلات timestamp"""
        logging.info(f"🔍 Debug timestamp {context}:")
        logging.info(f"   نوع: {df['timestamp'].dtype}")
        logging.info(f"   تعداد کل: {len(df)}")
        logging.info(f"   تعداد null: {df['timestamp'].isnull().sum()}")
        
        if len(df) > 0:
            logging.info(f"   نمونه مقادیر: {df['timestamp'].head(3).tolist()}")
            logging.info(f"   محدوده: {df['timestamp'].min()} تا {df['timestamp'].max()}")
        
        # بررسی انواع مختلف داده در ستون
        unique_types = df['timestamp'].apply(type).value_counts()
        logging.info(f"   انواع داده: {unique_types.to_dict()}")
    
    def extract_metadata_from_filename(self, filename: str) -> Tuple[str, str]:
        """استخراج نماد و تایم‌فریم از نام فایل - نسخه بهبود یافته"""
        basename = os.path.basename(filename).upper()
        symbol, timeframe = "UNKNOWN", "UNKNOWN"
        
        # الگوی بهبود یافته برای استخراج نماد
        symbol_match = re.search(r'([A-Z0-9]{2,}[-_]?(USDT|USD|BUSD|BTC|ETH|BNB|USDC|DAI))', basename)
        if symbol_match:
            # جایگزینی جداکننده‌ها با اسلش
            symbol = symbol_match.group(1).replace('-', '/').replace('_', '/')
            if '/' not in symbol: # برای حالتی مانند BTCUSDT
                quote = symbol_match.group(2)
                base = symbol.replace(quote, '')
                symbol = f"{base}/{quote}"
        
        # الگوی بهبود یافته برای استخراج تایم‌فریم
        tf_match = re.search(r'(\d+[MHWD])|HISTO(?:MINUTE|HOUR|DAY)', basename)
        if tf_match:
            timeframe = tf_match.group(0).replace("HISTOMINUTE", "1m").replace("HISTOHOUR", "1h").replace("HISTODAY", "1d")
        
        return symbol, timeframe
    
    def is_price_file(self, filename: str) -> bool:
        """تشخیص فایل‌های قیمت - نسخه بهبود یافته"""
        basename = os.path.basename(filename).lower()
        
        # فایل‌هایی که قطعاً قیمت نیستند
        if basename.startswith('news_') or basename.startswith('sentiment_') or \
           basename.startswith('unified_extraction_state') or \
           'sentiment' in basename or 'news' in basename:
            return False
        
        # فایل‌هایی که احتمالاً قیمت هستند
        price_indicators = [
            # الگوهای صرافی
            'binance_', 'cryptocompare_', 'kraken_',
            # الگوهای تایم‌فریم
            '_1m_', '_5m_', '_15m_', '_1h_', '_4h_', '_1d_',
            # الگوهای کلی
            'ohlc', 'candle', 'kline', 'price'
        ]
        
        return any(indicator in basename for indicator in price_indicators)
    
    def is_news_file(self, filename: str) -> bool:
        """تشخیص فایل‌های خبری - نسخه بهبود یافته"""
        basename = os.path.basename(filename).lower()
        
        # الگوهای فایل‌های خبری
        news_patterns = [
            'news_',           # فرمت جدید: news_BTC-USDT_en_20241127_143022.csv
            'raw_news_',       # فرمت قدیمی
            'sentiment_',      # فایل‌های احساسات
        ]
        
        # بررسی الگوهای اصلی
        if any(basename.startswith(pattern) for pattern in news_patterns):
            return True
        
        # بررسی وجود منابع خبری در نام فایل
        news_source_indicators = [source.lower() for source in self.known_news_sources.values()]
        if any(source in basename for source in news_source_indicators):
            return True
        
        return False
    
    def standardize_price_data(self, df: pd.DataFrame, filename: str) -> Optional[pd.DataFrame]:
        """استانداردسازی و اعتبارسنجی داده‌های قیمت"""
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower().str.strip()
        
        # استخراج متادیتا
        symbol_from_col = df_copy['symbol'].iloc[0] if 'symbol' in df_copy.columns else None
        timeframe_from_col = df_copy['timeframe'].iloc[0] if 'timeframe' in df_copy.columns else None
        
        symbol_from_fname, timeframe_from_fname = self.extract_metadata_from_filename(filename)
        
        # اولویت با داده‌های داخل فایل
        final_symbol = symbol_from_col or symbol_from_fname
        final_timeframe = timeframe_from_col or timeframe_from_fname
        
        # استانداردسازی ستون‌های OHLCV
        column_map: Dict[str, str] = {
            'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp', 'unnamed: 0': 'timestamp',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'price': 'close',
            'volume': 'volume', 'volumefrom': 'volume'
        }
        df_copy.rename(columns=column_map, inplace=True)
        
        # تبدیل و پاکسازی timestamp
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce', unit='s').fillna(
                pd.to_datetime(df_copy['timestamp'], errors='coerce')
            )
            df_copy.dropna(subset=['timestamp'], inplace=True)
            df_copy.set_index('timestamp', inplace=True)
        elif isinstance(df_copy.index, pd.DatetimeIndex):
            pass
        else:
            return None
        
        # بررسی ستون‌های ضروری
        if 'close' not in df_copy.columns: 
            return None
        
        # تکمیل ستون‌های OHLC
        for col in ['open', 'high', 'low']:
            if col not in df_copy.columns: 
                df_copy[col] = df_copy['close']
        
        if 'volume' not in df_copy.columns: 
            df_copy['volume'] = 0
        
        # تبدیل به numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        df_copy.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        # افزودن متادیتا
        df_copy['symbol'] = final_symbol
        df_copy['timeframe'] = final_timeframe
        
        return df_copy[['symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]
    
    def process_price_data(self) -> pd.DataFrame:
        """پردازش همه فایل‌های قیمت - نسخه بهبود یافته"""
        logging.info("شروع پردازش داده‌های قیمت...")
        
        # یافتن فایل‌های قیمت با فیلتر بهبود یافته
        all_files = glob.glob(os.path.join(RAW_DATA_PATH, '*.*'))        
        price_files = []
        
        for f_path in all_files:
            if f_path.endswith(('.csv', '.json', '.parquet')) and self.is_price_file(f_path):
                price_files.append(f_path)
        
        logging.info(f"تعداد {len(price_files)} فایل قیمت یافت شد")
        
        # نمایش نمونه فایل‌های یافت شده
        if price_files:
            logging.info("نمونه فایل‌های قیمت یافت شده:")
            for f_path in price_files[:5]:  # نمایش 5 فایل اول
                logging.info(f"   - {os.path.basename(f_path)}")
            if len(price_files) > 5:
                logging.info(f"   ... و {len(price_files) - 5} فایل دیگر")
        
        all_dataframes = []
        
        for f_path in price_files:
            try:
                # خواندن فایل
                if f_path.endswith('.csv'):
                    df = pd.read_csv(f_path, low_memory=False)
                elif f_path.endswith('.json'):
                    df = pd.read_json(f_path, lines=True)
                elif f_path.endswith('.parquet'):
                    df = pd.read_parquet(f_path)
                else:
                    continue
                
                # استانداردسازی
                validated_df = self.standardize_price_data(df, f_path)
                
                if validated_df is not None and not validated_df.empty:
                    all_dataframes.append(validated_df)
                    logging.info(f"✅ فایل '{os.path.basename(f_path)}' پردازش شد "
                               f"(نماد: {validated_df['symbol'].iloc[0]}, "
                               f"تایم‌فریم: {validated_df['timeframe'].iloc[0]})")
            
            except Exception as e:
                logging.warning(f"خطا در پردازش فایل '{os.path.basename(f_path)}': {e}")
        
        if not all_dataframes:
            logging.error("هیچ داده قابل استفاده‌ای برای ادغام یافت نشد.")
            return pd.DataFrame()
        
        # ادغام داده‌ها
        logging.info("شروع ادغام داده‌های قیمت...")
        master_df = pd.concat(all_dataframes)
        
        # حذف داده‌های UNKNOWN
        master_df = master_df[(master_df['symbol'] != 'UNKNOWN') & (master_df['timeframe'] != 'UNKNOWN')]
        
        # تنظیم index چند سطحی
        master_df.set_index(['symbol', 'timeframe'], append=True, inplace=True)
        master_df = master_df.reorder_levels(['symbol', 'timeframe', 'timestamp'])
        master_df.sort_index(inplace=True)
        
        # حذف تکراری‌ها
        master_df = master_df[~master_df.index.duplicated(keep='last')]
        
        logging.info(f"✅ ادغام قیمت تکمیل شد. تعداد نهایی ردیف‌ها: {len(master_df)}")
        
        self.price_data = master_df
        return master_df
    
    def analyze_sentiment_with_vader(self, text: str) -> Dict[str, float]:
        """تحلیل احساسات یک متن با استفاده از VADER"""
        try:
            if not text or not isinstance(text, str):
                return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
            
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logging.warning(f"خطا در تحلیل احساسات: {e}")
            return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
    
    def detect_news_source(self, file_path: str, df: pd.DataFrame) -> str:
        """تشخیص منبع خبری از نام فایل یا محتوای فایل"""
        basename = os.path.basename(file_path).lower()
        
        # بررسی نام فایل
        for source_name, source_key in self.known_news_sources.items():
            if source_key in basename:
                return source_name
        
        # بررسی ستون api_source در داده‌ها
        if 'api_source' in df.columns:
            sources = df['api_source'].value_counts()
            if not sources.empty:
                return sources.index[0]  # بازگرداندن پرتکرارترین منبع
        
        # بررسی ستون source
        if 'source' in df.columns:
            sources = df['source'].value_counts()
            if not sources.empty:
                # تشخیص بر اساس الگوهای شناخته شده
                top_source = sources.index[0].lower()
                if 'reddit' in top_source or 'r/' in top_source:
                    return 'Reddit'
                elif 'newsapi' in top_source:
                    return 'NewsAPI'
                elif 'coingecko' in top_source:
                    return 'CoinGecko'
                elif any(rss_name in top_source for rss_name in ['coindesk', 'cointelegraph', 'decrypt', 'cryptonews']):
                    return 'RSS'
        
        return 'Unknown'
    
    def process_news_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """پردازش یک فایل خبری و تحلیل احساسات آن - نسخه بهبود یافته"""
        try:
            # خواندن فایل
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                return None
            
            # تشخیص منبع خبری
            news_source = self.detect_news_source(file_path, df)
            
            logging.info(f"پردازش فایل خبری: {os.path.basename(file_path)} با {len(df)} خبر (منبع: {news_source})")
            
            # بررسی ستون‌های مورد نیاز
            required_cols = ['timestamp', 'symbol', 'title']
            if not all(col in df.columns for col in required_cols):
                logging.error(f"ستون‌های مورد نیاز در فایل یافت نشد: {required_cols}")
                logging.error(f"ستون‌های موجود: {list(df.columns)}")
                return None
            
            # ترکیب متن‌ها برای تحلیل جامع‌تر
            df['full_text'] = (
                df['title'].fillna('') + ". " + 
                df.get('content', '').fillna('') + ". " + 
                df.get('description', '').fillna('')
            )
            
            # حذف ردیف‌های بدون متن معتبر
            df = df[df['full_text'].str.strip().str.len() > 10]
            
            if df.empty:
                return None
            
            # اضافه کردن اطلاعات منبع
            df['detected_source'] = news_source
            
            # تحلیل احساسات
            sentiment_scores = df['full_text'].apply(lambda x: self.analyze_sentiment_with_vader(x))
            
            # استخراج امتیازات جداگانه
            df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
            df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
            df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
            df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
            
            # تعیین برچسب احساسات
            df['sentiment_label'] = df['sentiment_compound'].apply(
                lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
            )
            
            # تبدیل timestamp با بررسی بیشتر
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['timestamp'])

            # اطمینان از نوع datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                logging.error(f"خطا: ستون timestamp نوع datetime ندارد در فایل {file_path}")
                return None

            # تبدیل timezone-aware به naive اگر لازم باشد
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

            logging.info(f"✅ timestamp تبدیل شد: {df['timestamp'].dtype}, محدوده: {df['timestamp'].min()} تا {df['timestamp'].max()}")
            
            # محاسبه ویژگی‌های اضافی
            df['text_length'] = df['full_text'].str.len()
            df['title_length'] = df['title'].str.len()
            
            # اگر sentiment_score از قبل وجود داشت (از اسکریپت 01 یکپارچه)
            if 'sentiment_score' in df.columns:
                # میانگین‌گیری از دو روش
                df['sentiment_compound'] = (df['sentiment_compound'] + df['sentiment_score']) / 2
            
            # === ویژگی‌های مخصوص Reddit ===
            if news_source == 'Reddit':
                # Reddit دارای ویژگی‌های خاص است
                if 'score' in df.columns:
                    df['reddit_score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
                if 'comments' in df.columns:
                    df['reddit_comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
                
                logging.info(f"✅ Reddit features اضافه شد")
            
            return df
            
        except Exception as e:
            logging.error(f"خطا در پردازش فایل {file_path}: {e}")
            return None
    
    def process_sentiment_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """پردازش همه فایل‌های خبری و محاسبه آمار احساسات - نسخه بهبود یافته"""
        logging.info("شروع پردازش داده‌های احساسات...")
        
        # یافتن فایل‌های خبری با فیلتر بهبود یافته
        all_files = glob.glob(os.path.join(RAW_DATA_PATH, '*.*'))
        news_files = []
        
        for f_path in all_files:
            if f_path.endswith(('.csv', '.parquet')) and self.is_news_file(f_path):
                news_files.append(f_path)
        
        if not news_files:
            logging.warning("هیچ فایل خبری برای پردازش یافت نشد.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        logging.info(f"تعداد {len(news_files)} فایل خبری یافت شد")
        
        # نمایش نمونه فایل‌های یافت شده
        if news_files:
            logging.info("نمونه فایل‌های خبری یافت شده:")
            for f_path in news_files[:5]:  # نمایش 5 فایل اول
                logging.info(f"   - {os.path.basename(f_path)}")
            if len(news_files) > 5:
                logging.info(f"   ... و {len(news_files) - 5} فایل دیگر")
        
        all_processed_dfs = []
        source_stats = {}
        
        for file_path in news_files:
            processed_df = self.process_news_file(file_path)
            if processed_df is not None:
                all_processed_dfs.append(processed_df)
                
                # آمار منابع
                if 'detected_source' in processed_df.columns:
                    source = processed_df['detected_source'].iloc[0]
                    source_stats[source] = source_stats.get(source, 0) + len(processed_df)
        
        if not all_processed_dfs:
            logging.error("هیچ داده‌ای برای پردازش احساسات یافت نشد.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # نمایش آمار منابع
        logging.info("\n📊 آمار منابع خبری:")
        for source, count in source_stats.items():
            logging.info(f"   📡 {source}: {count:,} خبر")
        
        # ادغام تمام داده‌های پردازش شده
        logging.info("ادغام داده‌های احساسات...")
        combined_df = pd.concat(all_processed_dfs, ignore_index=True)

        # بررسی نوع ستون‌ها بعد از concat
        logging.info(f"نوع ستون timestamp بعد از concat: {combined_df['timestamp'].dtype}")

        # اطمینان از یکسان بودن نوع timestamp در تمام ردیف‌ها
        if combined_df['timestamp'].dtype == 'object':
            logging.warning("نوع timestamp object است - تبدیل به datetime...")
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
            
            # حذف ردیف‌هایی که timestamp آن‌ها تبدیل نشده
            before_drop = len(combined_df)
            combined_df = combined_df.dropna(subset=['timestamp'])
            after_drop = len(combined_df)
            
            if before_drop != after_drop:
                logging.warning(f"حذف {before_drop - after_drop} ردیف با timestamp نامعتبر")
        
        # حذف تکراری‌ها
        before_dedup = len(combined_df)
        dedup_cols = ['timestamp', 'symbol', 'title']
        
        # اگر ستون url موجود است، آن را هم به عنوان کلید یکتا اضافه کن
        if 'url' in combined_df.columns:
            dedup_cols.append('url')
        
        combined_df = combined_df.drop_duplicates(subset=dedup_cols)
        after_dedup = len(combined_df)
        
        if before_dedup != after_dedup:
            logging.info(f"حذف {before_dedup - after_dedup} ردیف تکراری")
        
        # بررسی نوع timestamp قبل از محاسبه آمار
        logging.info(f"نوع ستون timestamp قبل از تجمیع: {combined_df['timestamp'].dtype}")
        logging.info(f"نمونه مقادیر timestamp: {combined_df['timestamp'].head()}")

        # اطمینان از نوع datetime
        if not pd.api.types.is_datetime64_any_dtype(combined_df['timestamp']):
            logging.warning("تبدیل مجدد timestamp به datetime...")
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
            combined_df = combined_df.dropna(subset=['timestamp'])
        
        # Debug timestamp قبل از تجمیع
        self.debug_timestamp_column(combined_df, "قبل از تجمیع")

        # محاسبه آمار تجمیعی
        daily_stats, hourly_stats = self.aggregate_sentiments(combined_df)
        
        # تولید گزارش
        self.generate_sentiment_report(combined_df)
        
        self.sentiment_data = combined_df
        
        return combined_df, daily_stats, hourly_stats
    
    def aggregate_sentiments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """محاسبه آمار تجمیعی احساسات"""
        
        # بررسی وجود داده
        if df.empty:
            logging.warning("DataFrame خالی برای محاسبه آمار احساسات")
            return pd.DataFrame(), pd.DataFrame()
        
        # بررسی نوع ستون timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logging.error(f"خطا: ستون timestamp نوع datetime ندارد. نوع فعلی: {df['timestamp'].dtype}")
            logging.error(f"نمونه مقادیر: {df['timestamp'].head()}")
            
            # تلاش برای تبدیل
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                logging.info("✅ timestamp با موفقیت تبدیل شد")
            except Exception as e:
                logging.error(f"خطا در تبدیل timestamp: {e}")
                return pd.DataFrame(), pd.DataFrame()
        
        # آمار روزانه برای هر نماد
        try:
            df['date'] = df['timestamp'].dt.date
        except Exception as e:
            logging.error(f"خطا در استخراج date از timestamp: {e}")
            logging.error(f"نوع timestamp: {df['timestamp'].dtype}")
            return pd.DataFrame(), pd.DataFrame()
        
        # آمار اولیه
        agg_dict = {
            'sentiment_compound': ['mean', 'std', 'min', 'max'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'text_length': 'mean'
        }
        
        # اگر ستون‌های مخصوص Reddit موجود است
        if 'reddit_score' in df.columns:
            agg_dict['reddit_score'] = 'mean'
        if 'reddit_comments' in df.columns:
            agg_dict['reddit_comments'] = 'mean'
        
        daily_stats = df.groupby(['symbol', 'date']).agg(agg_dict).round(4)
        
        # تغییر نام ستون‌ها
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
        daily_stats = daily_stats.reset_index()
        
        # افزودن تعداد اخبار و تنوع منابع
        news_count = df.groupby(['symbol', 'date']).size().reset_index(name='news_count')
        daily_stats = pd.merge(daily_stats, news_count, on=['symbol', 'date'], how='left')
        
        # تعداد منابع مختلف
        if 'detected_source' in df.columns:
            source_diversity = df.groupby(['symbol', 'date'])['detected_source'].nunique().reset_index(name='source_diversity')
            daily_stats = pd.merge(daily_stats, source_diversity, on=['symbol', 'date'], how='left')
        
        # آمار ساعتی برای هر نماد
        try:
            df['hour'] = df['timestamp'].dt.floor('H')
        except Exception as e:
            logging.error(f"خطا در محاسبه hour: {e}")
            return daily_stats, pd.DataFrame()
        
        hourly_agg = {
            'sentiment_compound': ['mean', 'count'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        }
        
        # اگر ستون‌های مخصوص Reddit موجود است
        if 'reddit_score' in df.columns:
            hourly_agg['reddit_score'] = 'mean'
        
        hourly_stats = df.groupby(['symbol', 'hour']).agg(hourly_agg).round(4)
        
        hourly_stats.columns = ['_'.join(col).strip() for col in hourly_stats.columns.values]
        hourly_stats = hourly_stats.reset_index()
        
        return daily_stats, hourly_stats
    
    def generate_sentiment_report(self, df: pd.DataFrame):
        """تولید گزارش جامع احساسات - نسخه بهبود یافته"""
        logging.info("\n" + "="*60)
        logging.info("📊 گزارش جامع تحلیل احساسات (Enhanced)")
        logging.info("="*60)
        
        # آمار کلی
        total_news = len(df)
        date_range = f"{df['timestamp'].min().date()} تا {df['timestamp'].max().date()}"
        
        logging.info(f"📅 بازه زمانی: {date_range}")
        logging.info(f"📰 تعداد کل اخبار: {total_news:,}")
        logging.info(f"🪙 تعداد نمادها: {df['symbol'].nunique()}")
        
        # آمار منابع (اگر موجود باشد)
        if 'detected_source' in df.columns:
            source_dist = df['detected_source'].value_counts()
            logging.info(f"\n📡 توزیع منابع خبری:")
            for source, count in source_dist.items():
                percentage = (count / total_news) * 100
                emoji = {'GNews': '🌐', 'NewsAPI': '📰', 'CoinGecko': '🦎', 'RSS': '📡', 'Reddit': '🔴'}.get(source, '📊')
                logging.info(f"   {emoji} {source}: {count:,} ({percentage:.1f}%)")
        
        # توزیع احساسات کلی
        sentiment_dist = df['sentiment_label'].value_counts()
        logging.info("\n🎭 توزیع احساسات:")
        for label, count in sentiment_dist.items():
            percentage = (count / total_news) * 100
            emoji = {'positive': '😊', 'negative': '😟', 'neutral': '😐'}.get(label, '')
            logging.info(f"   {emoji} {label}: {count:,} ({percentage:.1f}%)")
        
        # آمار به تفکیک نماد
        logging.info("\n📈 آمار احساسات به تفکیک نماد:")
        symbol_stats = df.groupby('symbol').agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x) * 100
        }).round(3)
        
        for symbol in symbol_stats.index[:10]:  # نمایش 10 نماد برتر
            mean_sentiment = symbol_stats.loc[symbol, ('sentiment_compound', 'mean')]
            std_sentiment = symbol_stats.loc[symbol, ('sentiment_compound', 'std')]
            count = symbol_stats.loc[symbol, ('sentiment_compound', 'count')]
            positive_pct = symbol_stats.loc[symbol, ('sentiment_label', '<lambda>')]
            
            emoji = '🟢' if mean_sentiment > 0.1 else ('🔴' if mean_sentiment < -0.1 else '🟡')
            logging.info(f"   {emoji} {symbol}: میانگین={mean_sentiment:.3f}, "
                        f"انحراف معیار={std_sentiment:.3f}, "
                        f"تعداد={count}, مثبت={positive_pct:.1f}%")
        
        # آمار ویژه Reddit (اگر موجود باشد)
        if 'reddit_score' in df.columns:
            reddit_df = df[df['detected_source'] == 'Reddit']
            if not reddit_df.empty:
                logging.info("\n🔴 آمار ویژه Reddit:")
                avg_score = reddit_df['reddit_score'].mean()
                avg_comments = reddit_df['reddit_comments'].mean()
                logging.info(f"   میانگین امتیاز پست‌ها: {avg_score:.1f}")
                logging.info(f"   میانگین تعداد کامنت‌ها: {avg_comments:.1f}")
    
    def normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """نرمال‌سازی timezone برای جلوگیری از خطای merge"""
        if df.empty:
            return df
        
        # اگر timestamp timezone-aware است، به UTC تبدیل کن و سپس timezone را حذف کن
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        # همین کار را برای ستون‌های timestamp انجام بده
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
        
        return df
    
    def merge_price_and_sentiment(self) -> pd.DataFrame:
        """
        ادغام هوشمند داده‌های قیمت و احساسات با حل مشکل عدم تطبیق زمانی
        
        راه‌حل: استفاده از Broadcasting احساسات برای کل دوره
        """
        logging.info("شروع ادغام داده‌های قیمت و احساسات...")
        
        if self.price_data is None or self.price_data.empty:
            logging.error("داده‌های قیمت موجود نیست")
            return pd.DataFrame()
        
        if self.sentiment_data is None or self.sentiment_data.empty:
            logging.warning("داده‌های احساسات موجود نیست. ادامه بدون احساسات...")
            # اضافه کردن ستون‌های احساسات با مقدار صفر
            price_data = self.price_data.reset_index()
            price_data['sentiment_compound_mean'] = 0
            price_data['sentiment_positive_mean'] = 0
            price_data['sentiment_negative_mean'] = 0
            price_data['sentiment_neutral_mean'] = 0
            price_data.set_index(['symbol', 'timeframe', 'timestamp'], inplace=True)
            return price_data
        
        # نرمال‌سازی timezone
        logging.info("نرمال‌سازی timezone...")
        price_data = self.price_data.reset_index()
        price_data = self.normalize_timezone(price_data)
        
        sentiment_data = self.sentiment_data.copy()
        sentiment_data = self.normalize_timezone(sentiment_data)
        
        # محاسبه آمار احساسات کلی برای هر نماد (بدون توجه به زمان)
        logging.info("محاسبه آمار احساسات کلی برای هر نماد...")
        
        # آمار پایه
        basic_agg = {
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        }
        
        # اگر ستون‌های Reddit موجود است
        if 'reddit_score' in sentiment_data.columns:
            basic_agg['reddit_score'] = 'mean'
        if 'reddit_comments' in sentiment_data.columns:
            basic_agg['reddit_comments'] = 'mean'
        
        sentiment_symbol_stats = sentiment_data.groupby('symbol').agg(basic_agg).round(4)
        
        # تغییر نام ستون‌ها
        sentiment_symbol_stats.columns = ['_'.join(col).strip() for col in sentiment_symbol_stats.columns.values]
        sentiment_symbol_stats = sentiment_symbol_stats.reset_index()
        
        # اضافه کردن تنوع منابع
        if 'detected_source' in sentiment_data.columns:
            source_diversity = sentiment_data.groupby('symbol')['detected_source'].nunique().reset_index(name='source_diversity')
            sentiment_symbol_stats = pd.merge(sentiment_symbol_stats, source_diversity, on='symbol', how='left')
        
        # ادغام کلی بر اساس نماد (Broadcast احساسات)
        logging.info(f"Broadcasting احساسات برای {len(price_data)} رکورد قیمت...")
        
        merged_data = pd.merge(
            price_data,
            sentiment_symbol_stats,
            on='symbol',
            how='left'
        )
        
        # پر کردن مقادیر خالی احساسات با مقادیر پیش‌فرض
        sentiment_columns = [col for col in merged_data.columns if 'sentiment' in col or 'reddit' in col or 'source_diversity' in col]
        for col in sentiment_columns:
            merged_data[col] = merged_data[col].fillna(0)
        
        # بازگرداندن index
        merged_data.set_index(['symbol', 'timeframe', 'timestamp'], inplace=True)
        merged_data.sort_index(inplace=True)
        
        logging.info(f"✅ ادغام تکمیل شد. شکل نهایی داده: {merged_data.shape}")
        
        # نمایش نمونه احساسات ادغام شده
        sentiment_cols = [col for col in merged_data.columns if 'sentiment' in col or 'reddit' in col]
        if sentiment_cols:
            logging.info(f"\n📊 آمار احساسات ادغام شده:")
            for col in sentiment_cols:
                non_zero = (merged_data[col] != 0).sum()
                mean_val = merged_data[col].mean()
                logging.info(f"   {col}: تعداد غیر صفر = {non_zero} ({non_zero/len(merged_data)*100:.1f}%), میانگین = {mean_val:.4f}")
        
        # نمایش آمار منابع
        if 'source_diversity' in merged_data.columns:
            avg_diversity = merged_data['source_diversity'].mean()
            logging.info(f"   📡 میانگین تنوع منابع: {avg_diversity:.2f}")
        
        return merged_data
    
    def save_processed_data(self, price_df: pd.DataFrame, sentiment_raw: pd.DataFrame,
                          sentiment_daily: pd.DataFrame, sentiment_hourly: pd.DataFrame,
                          merged_df: pd.DataFrame) -> Dict[str, str]:
        """ذخیره داده‌های پردازش شده"""
        timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        # اطمینان از وجود پوشه‌های مقصد
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        # ذخیره داده‌های قیمت پردازش شده
        if not price_df.empty:
            price_filename = f'master_ohlcv_data_{timestamp_str}.parquet'
            price_path = os.path.join(PROCESSED_DATA_PATH, price_filename)            
            price_df.to_parquet(price_path)
            saved_files['price'] = price_path
            logging.info(f"✅ داده‌های قیمت ذخیره شد: {price_path}")
        
        # ذخیره داده‌های احساسات
        if not sentiment_raw.empty:
            # داده‌های خام احساسات
            raw_filename = f'sentiment_scores_raw_{timestamp_str}.parquet'
            raw_path = os.path.join(PROCESSED_DATA_PATH, raw_filename)
            sentiment_raw.to_parquet(raw_path, index=False)
            saved_files['sentiment_raw'] = raw_path
            
            # آمار روزانه
            if not sentiment_daily.empty:
                daily_filename = f'sentiment_scores_daily_{timestamp_str}.parquet'
                daily_path = os.path.join(PROCESSED_DATA_PATH, daily_filename)
                sentiment_daily.to_parquet(daily_path, index=False)
                saved_files['sentiment_daily'] = daily_path
            
            # آمار ساعتی
            if not sentiment_hourly.empty:
                hourly_filename = f'sentiment_scores_hourly_{timestamp_str}.parquet'
                hourly_path = os.path.join(PROCESSED_DATA_PATH, hourly_filename)
                sentiment_hourly.to_parquet(hourly_path, index=False)
                saved_files['sentiment_hourly'] = hourly_path
            
            logging.info(f"✅ داده‌های احساسات ذخیره شد")
        
        # ذخیره داده‌های ادغام شده
        if not merged_df.empty:
            merged_filename = f'master_merged_data_{timestamp_str}.parquet'
            merged_path = os.path.join(PROCESSED_DATA_PATH, merged_filename)
            merged_df.to_parquet(merged_path)
            saved_files['merged'] = merged_path
            logging.info(f"✅ داده‌های ادغام شده ذخیره شد: {merged_path}")
            
            # ذخیره نمونه CSV برای بررسی
            sample_csv = f'merged_sample_{timestamp_str}.csv'
            sample_path = os.path.join(PROCESSED_DATA_PATH, sample_csv)
            merged_df.head(1000).to_csv(sample_path, encoding='utf-8-sig')
            saved_files['sample'] = sample_path
        
        logging.info("\n📁 فایل‌های ذخیره شده:")
        for file_type, path in saved_files.items():
            logging.info(f"   {file_type}: {path}")
        
        return saved_files

def run_unified_processing(process_price: bool = True, process_sentiment: bool = True,
                         merge_data: bool = True):
    """تابع اصلی اجرای پردازش یکپارچه - نسخه بهبود یافته"""
    logging.info("="*80)
    logging.info("🚀 شروع پردازش یکپارچه داده‌ها (Enhanced Unified ETL)")
    logging.info("="*80)
    
    processor = UnifiedDataProcessor()
    
    # متغیرهای خروجی
    price_df = pd.DataFrame()
    sentiment_raw = pd.DataFrame()
    sentiment_daily = pd.DataFrame()
    sentiment_hourly = pd.DataFrame()
    merged_df = pd.DataFrame()
    
    # پردازش داده‌های قیمت
    if process_price:
        logging.info("\n📊 مرحله 1: پردازش داده‌های قیمت")
        price_df = processor.process_price_data()
        
        if not price_df.empty:
            logging.info(f"✅ تعداد {len(price_df)} رکورد قیمت پردازش شد")
            logging.info(f"📈 تعداد نمادها: {price_df.index.get_level_values('symbol').nunique()}")
            logging.info(f"⏱️ تعداد تایم‌فریم‌ها: {price_df.index.get_level_values('timeframe').nunique()}")
        else:
            logging.warning("⚠️ هیچ داده قیمتی پردازش نشد")
    
    # پردازش داده‌های احساسات
    if process_sentiment:
        logging.info("\n🎭 مرحله 2: پردازش داده‌های احساسات (Enhanced)")
        sentiment_raw, sentiment_daily, sentiment_hourly = processor.process_sentiment_data()
        
        if not sentiment_raw.empty:
            logging.info(f"✅ تعداد {len(sentiment_raw)} خبر پردازش شد")
            logging.info(f"📰 تعداد نمادها: {sentiment_raw['symbol'].nunique()}")
            
            # آمار منابع
            if 'detected_source' in sentiment_raw.columns:
                source_counts = sentiment_raw['detected_source'].value_counts()
                logging.info(f"📡 منابع شناسایی شده: {dict(source_counts)}")
        else:
            logging.warning("⚠️ هیچ داده احساساتی پردازش نشد")
    
    # ادغام داده‌ها
    if merge_data and not price_df.empty:
        logging.info("\n🔗 مرحله 3: ادغام داده‌های قیمت و احساسات (Enhanced Broadcasting)")
        merged_df = processor.merge_price_and_sentiment()
        
        if not merged_df.empty:
            logging.info(f"✅ ادغام موفق: {merged_df.shape}")
            
            # نمایش نمونه ستون‌های احساسات
            sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col or 'reddit' in col]
            if sentiment_cols:
                logging.info(f"🎭 ستون‌های احساسات اضافه شده: {len(sentiment_cols)} ستون")
                for col in sentiment_cols[:5]:  # نمایش 5 ستون اول
                    logging.info(f"     - {col}")
                if len(sentiment_cols) > 5:
                    logging.info(f"     ... و {len(sentiment_cols) - 5} ستون دیگر")
    
    # ذخیره داده‌ها
    logging.info("\n💾 مرحله 4: ذخیره داده‌های پردازش شده")
    saved_files = processor.save_processed_data(
        price_df, sentiment_raw, sentiment_daily, sentiment_hourly, merged_df
    )
    
    # گزارش نهایی
    print("\n" + "="*80)
    print("📊 گزارش نهایی پردازش یکپارچه (Enhanced)")
    print("="*80)
    
    if process_price:
        print(f"\n💰 داده‌های قیمت:")
        print(f"   - تعداد رکورد: {len(price_df):,}")
        print(f"   - تعداد نماد: {price_df.index.get_level_values('symbol').nunique() if not price_df.empty else 0}")
        print(f"   - تعداد تایم‌فریم: {price_df.index.get_level_values('timeframe').nunique() if not price_df.empty else 0}")
    
    if process_sentiment:
        print(f"\n🎭 داده‌های احساسات:")
        print(f"   - تعداد خبر: {len(sentiment_raw):,}")
        print(f"   - تعداد نماد: {sentiment_raw['symbol'].nunique() if not sentiment_raw.empty else 0}")
        
        # آمار منابع
        if not sentiment_raw.empty and 'detected_source' in sentiment_raw.columns:
            source_counts = sentiment_raw['detected_source'].value_counts()
            print(f"   - منابع خبری:")
            for source, count in source_counts.items():
                print(f"     📡 {source}: {count:,} خبر")
    
    if merge_data and not merged_df.empty:
        print(f"\n🔗 داده‌های ادغام شده:")
        print(f"   - شکل نهایی: {merged_df.shape}")
        sentiment_features = [col for col in merged_df.columns if 'sentiment' in col or 'reddit' in col]
        print(f"   - ویژگی‌های احساسات: {len(sentiment_features)}")
        
        # نمایش درصد رکوردهایی که احساسات دارند
        non_zero_sentiment = 0
        if sentiment_features:
            for col in sentiment_features:
                if 'compound' in col and 'mean' in col:
                    non_zero_sentiment = (merged_df[col] != 0).sum()
                    mean_val = merged_df[col].mean()
                    print(f"   - رکوردهای دارای احساسات: {non_zero_sentiment:,} ({non_zero_sentiment/len(merged_df)*100:.1f}%)")
                    print(f"   - میانگین احساسات: {mean_val:.4f}")
                    break
        
        # آمار منابع در داده‌های ادغام شده
        if 'source_diversity' in merged_df.columns:
            avg_diversity = merged_df['source_diversity'].mean()
            max_diversity = merged_df['source_diversity'].max()
            print(f"   - میانگین تنوع منابع: {avg_diversity:.2f}")
            print(f"   - حداکثر تنوع منابع: {max_diversity:.0f}")
    
    print("\n📁 فایل‌های ذخیره شده:")
    for file_type, path in saved_files.items():
        print(f"   - {file_type}: {os.path.basename(path)}")
    
    print("="*80)
    
    # نمایش نمونه داده‌های ادغام شده
    if not merged_df.empty:
        print("\n--- نمونه 5 ردیف از داده‌های ادغام شده (Enhanced) ---")
        display_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # اضافه کردن ستون‌های احساسات مهم
        sentiment_display_cols = []
        for col in merged_df.columns:
            if 'sentiment_compound_mean' in col or 'source_diversity' in col:
                sentiment_display_cols.append(col)
            if len(sentiment_display_cols) >= 3:  # حداکثر 3 ستون احساسات
                break
        
        display_cols.extend(sentiment_display_cols)
        print(merged_df[display_cols].head())

def get_user_options():
    """دریافت تنظیمات از کاربر"""
    print("\n" + "="*60)
    print("⚙️ تنظیمات پردازش یکپارچه (Enhanced - سازگار با fetch_01_fixed)")
    print("="*60)
    
    print("\nانتخاب کنید چه داده‌هایی پردازش شوند:")
    print("1. فقط داده‌های قیمت")
    print("2. فقط داده‌های احساسات")
    print("3. هر دو (قیمت و احساسات) - بدون ادغام")
    print("4. هر دو + ادغام (توصیه می‌شود)")
    
    choice = input("\nانتخاب شما (پیش‌فرض: 4): ").strip() or '4'
    
    process_price = choice in ['1', '3', '4']
    process_sentiment = choice in ['2', '3', '4']
    merge_data = choice == '4'
    
    print("\n🔧 ویژگی‌های نسخه Enhanced:")
    print("✅ سازگاری کامل با fetch_01_fixed_clean.py")
    print("✅ پشتیبانی از منابع جدید: Reddit, NewsAPI, RSS, CoinGecko")
    print("✅ تشخیص هوشمند فایل‌های قیمت و خبری")
    print("✅ Broadcasting احساسات برای حل مشکل عدم تطبیق زمانی")
    print("✅ آمار تفصیلی منابع خبری")
    
    return process_price, process_sentiment, merge_data

if __name__ == '__main__':
    # دریافت تنظیمات از کاربر
    process_price, process_sentiment, merge_data = get_user_options()
    
    # اجرای پردازش
    run_unified_processing(process_price, process_sentiment, merge_data)