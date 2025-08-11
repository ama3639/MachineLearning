#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 اسکریپت یکپارچه استخراج داده‌های تلگرام (نسخه ادغام شده با پروژه)
======================================================================

این اسکریپت به عنوان همراه fetch_historical_data_01.py و etl_and_merge_02.py طراحی شده است.

🔧 تغییرات کلیدی این نسخه:
✅ ادغام کامل با config.ini پروژه (بدون تغییر ساختار موجود)
✅ خروجی CSV سازگار 100% با etl_and_merge_02.py
✅ تشخیص هوشمند نماد از متن پیام (بهبود یافته)
✅ پشتیبانی از کانال‌های قیمت، اخبار و تحلیل
✅ تشخیص خودکار نوع کانال (خبری، قیمتی، تحلیلی)
✅ اضافه کردن sentiment analysis اولیه
✅ تشخیص اعداد، قیمت‌ها و پیش‌بینی‌ها از متن
✅ حفظ تمام قابلیت‌های قبلی
✅ اصلاح مشکل خواندن config و ایجاد پوشه‌ها

نویسنده: ادغام شده با پروژه اصلی
تاریخ: 2025
"""

import asyncio
import json
import csv
import re
import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import configparser

# کتابخانه‌های تلگرام
try:
    from telethon.sync import TelegramClient
    from telethon.tl.functions.messages import GetHistoryRequest
    from telethon.errors import FloodWaitError, ChannelPrivateError
    from telethon import events
except ImportError:
    print("❌ لطفاً کتابخانه telethon را نصب کنید: pip install telethon")
    sys.exit(1)

# sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("❌ لطفاً کتابخانه vaderSentiment را نصب کنید: pip install vaderSentiment")
    sys.exit(1)

# =============================================================================
# 📊 کلاس‌های داده
# =============================================================================

@dataclass
class ChannelConfig:
    """پیکربندی کانال"""
    name: str
    url: str
    channel_type: str
    active: bool = True
    description: str = ""

@dataclass
class MessageData:
    """ساختار داده پیام"""
    id: int
    date: str
    channel: str
    message: str
    raw_text: str
    url: str
    sender_id: Optional[int] = None
    sender_username: Optional[str] = None


# =============================================================================
# 🔧 کلاس اصلی یکپارچه (ادغام شده با پروژه)
# =============================================================================

class TelegramExtractorForProject:
    """
    کلاس یکپارچه برای استخراج تلگرام - ادغام شده با پروژه اصلی
    """
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.channels: Dict[str, ChannelConfig] = {}
        self.client: Optional[TelegramClient] = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # مسیرهای پیش‌فرض (در صورت عدم وجود config)
        self.raw_data_path = "data/raw"
        self.processed_data_path = "data/processed"
        self.log_path = "logs"
        
        # ترتیب صحیح اجرای توابع
        try:
            print("🔧 بارگذاری پیکربندی...")
            self._load_config()
            
            print("📝 راه‌اندازی سیستم لاگ‌گذاری...")
            self._setup_logging()
            
            print("📁 ایجاد ساختار پوشه‌ها...")
            self._create_directory_structure()
            
            if hasattr(self, 'logger'):
                self.logger.info("🚀 Telegram Extractor for Project راه‌اندازی شد")
            else:
                print("🚀 Telegram Extractor for Project راه‌اندازی شد")
        except Exception as e:
            print(f"❌ خطا در راه‌اندازی: {e}")
            print("📂 ادامه با تنظیمات حداقلی...")
            # تنظیم logger ساده در صورت خطا
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

    def _load_config(self):
        """بارگذاری پیکربندی از فایل (ادغام شده با config.ini پروژه)"""
        try:
            if not os.path.exists(self.config_file):
                print(f"⚠️ فایل {self.config_file} یافت نشد. از تنظیمات پیش‌فرض استفاده می‌شود.")
                return
            
            self.config.read(self.config_file, encoding='utf-8')

            # خواندن مسیرها از بخش [Paths] موجود
            if self.config.has_section('Paths'):
                self.raw_data_path = self.config.get('Paths', 'raw', fallback=self.raw_data_path)
                self.processed_data_path = self.config.get('Paths', 'processed', fallback=self.processed_data_path)
                self.log_path = self.config.get('Paths', 'logs', fallback=self.log_path)
            
            # بارگذاری کانال‌ها از بخش جدید [TELEGRAM_CHANNELS]
            self._load_channels()
            
        except Exception as e:
            print(f"❌ خطا در خواندن '{self.config_file}': {e}")
            print("📁 ادامه با تنظیمات پیش‌فرض...")
            # اطمینان از اینکه کانال‌های پیش‌فرض اضافه شوند
            if len(self.channels) == 0:
                self._add_default_channels()

    def _setup_logging(self):
        """راه‌اندازی سیستم لاگ‌گذاری (سازگار با پروژه)"""
        try:
            # ایجاد پوشه‌های اصلی اگر وجود نداشت
            os.makedirs(self.log_path, exist_ok=True)

            # ایجاد زیرپوشه برای telegram extractor
            telegram_log_path = os.path.join(self.log_path, "telegram_extractor")
            os.makedirs(telegram_log_path, exist_ok=True)

            # تنظیمات فرمت لاگ و فایل لاگ
            log_format = "%(asctime)s - %(levelname)s - %(message)s"
            log_file = os.path.join(telegram_log_path, f"telegram_extractor_{datetime.now().strftime('%Y%m%d')}.log")

            # سطح لاگ از config یا پیش‌فرض
            log_level = 'INFO'
            if self.config.has_section('TELEGRAM_SETTINGS'):
                log_level = self.config.get('TELEGRAM_SETTINGS', 'log_level', fallback='INFO')

            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)
                ]
            )

            # دریافت logger
            self.logger = logging.getLogger(__name__)
            self.logger.info("📝 سیستم لاگ‌گذاری راه‌اندازی شد.")
        
        except Exception as e:
            print(f"❌ خطا در راه‌اندازی سیستم لاگ‌گذاری: {e}")
            # تنظیم logger ساده در صورت خطا
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

    def _create_directory_structure(self):
        """ایجاد پوشه‌های مورد نیاز بر اساس پیکربندی"""
        try:
            Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
            Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            
            # استفاده ایمن از logger
            if hasattr(self, 'logger'):
                self.logger.info("📁 ساختار پوشه‌ها بر اساس config.ini بررسی و ایجاد شد.")
            else:
                print("📁 ساختار پوشه‌ها بر اساس config.ini بررسی و ایجاد شد.")
        except Exception as e:
            print(f"❌ خطا در ایجاد پوشه‌ها: {e}")
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ خطا در ایجاد پوشه‌ها: {e}")

    def _load_channels(self):
        """بارگذاری لیست کانال‌ها از بخش [TELEGRAM_CHANNELS]"""
        try:
            if 'TELEGRAM_CHANNELS' in self.config:
                for name, config_str in self.config['TELEGRAM_CHANNELS'].items():
                    parts = config_str.split(',')
                    if len(parts) >= 3:
                        url, channel_type, status = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        description = parts[3].strip() if len(parts) > 3 else ""
                        self.channels[name] = ChannelConfig(
                            name=name, url=url, channel_type=channel_type,
                            active=(status.lower() == 'فعال'), description=description
                        )
                # استفاده ایمن از logger
                if hasattr(self, 'logger'):
                    self.logger.info(f"📺 {len(self.channels)} کانال از بخش TELEGRAM_CHANNELS بارگذاری شد")
                else:
                    print(f"📺 {len(self.channels)} کانال از بخش TELEGRAM_CHANNELS بارگذاری شد")
            else:
                if hasattr(self, 'logger'):
                    self.logger.warning("⚠️ بخش TELEGRAM_CHANNELS در config.ini یافت نشد. کانال‌های پیش‌فرض اضافه می‌شوند.")
                else:
                    print("⚠️ بخش TELEGRAM_CHANNELS در config.ini یافت نشد. کانال‌های پیش‌فرض اضافه می‌شوند.")
                self._add_default_channels()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ خطا در بارگذاری کانال‌ها: {e}")
            else:
                print(f"❌ خطا در بارگذاری کانال‌ها: {e}")
            self._add_default_channels()

    def _add_default_channels(self):
        """اضافه کردن کانال‌های پیش‌فرض در صورت عدم وجود تنظیمات"""
        default_channels = {
            'arzdigital': ChannelConfig(
                name='arzdigital',
                url='https://t.me/arzdigital',
                channel_type='اخبار',
                active=True,
                description='کانال ارز دیجیتال فارسی'
            ),
            'binance_announcements': ChannelConfig(
                name='binance_announcements',
                url='https://t.me/binance_announcements',
                channel_type='اخبار',
                active=True,
                description='اعلانات رسمی بایننس'
            )
        }
        
        self.channels.update(default_channels)
        # استفاده ایمن از logger
        if hasattr(self, 'logger'):
            self.logger.info(f"📺 {len(default_channels)} کانال پیش‌فرض اضافه شد")
        else:
            print(f"📺 {len(default_channels)} کانال پیش‌فرض اضافه شد")

    async def _init_telegram_client(self):
        """راه‌اندازی کلاینت تلگرام (از بخش [TELEGRAM])"""
        try:
            # خواندن تنظیمات از config یا استفاده از پیش‌فرض
            if self.config.has_section('TELEGRAM'):
                api_id = self.config.get('TELEGRAM', 'api_id')
                api_hash = self.config.get('TELEGRAM', 'api_hash')
                phone = self.config.get('TELEGRAM', 'phone_number')
                session_name = self.config.get('TELEGRAM', 'session_name', fallback='project_session')
            else:
                raise ValueError("بخش [TELEGRAM] در config.ini یافت نشد. لطفاً تنظیمات تلگرام را اضافه کنید.")
            
            self.client = TelegramClient(session_name, int(api_id), api_hash)
            await self.client.start(phone=phone)
            self.logger.info("✅ اتصال به تلگرام برقرار شد")
        except Exception as e:
            self.logger.critical(f"❌ خطا در اتصال به تلگرام: {e}")
            raise

    # =============================================================================
    # 🔍 بخش استخراج و پردازش هوشمند
    # =============================================================================

    async def extract_channel_messages(self, channel_config: ChannelConfig) -> List[MessageData]:
        """استخراج پیام‌های یک کانال (بهبود یافته)"""
        messages = []
        try:
            self.logger.info(f"🔄 شروع استخراج از کانال: {channel_config.name} (نوع: {channel_config.channel_type})")
            entity = await self.client.get_entity(channel_config.url)
            
            offset_id = 0
            batch_size = 50
            max_messages = 1000
            delay = 2.0
            
            # خواندن تنظیمات از config در صورت وجود
            if self.config.has_section('TELEGRAM_SETTINGS'):
                batch_size = int(self.config.get('TELEGRAM_SETTINGS', 'batch_size', fallback=50))
                max_messages = int(self.config.get('TELEGRAM_SETTINGS', 'max_messages_per_channel', fallback=1000))
                delay = float(self.config.get('TELEGRAM_SETTINGS', 'delay_between_requests', fallback=2.0))
            
            message_count = 0
            while message_count < max_messages:
                try:
                    history = await self.client(GetHistoryRequest(
                        peer=entity, limit=min(batch_size, max_messages - message_count),
                        offset_id=offset_id, offset_date=None, add_offset=0,
                        max_id=0, min_id=0, hash=0
                    ))
                    
                    if not history.messages:
                        break
                        
                    for msg in history.messages:
                        if msg.message and msg.sender_id:
                            sender = await msg.get_sender()
                            sender_username = sender.username if sender and hasattr(sender, 'username') else 'N/A'
                            
                            message_data = MessageData(
                                id=msg.id,
                                date=msg.date.strftime('%Y-%m-%d %H:%M:%S'),
                                channel=channel_config.name,
                                message=msg.message.replace('\n', ' ').replace('\r', ''),
                                raw_text=msg.message,
                                url=f"https://t.me/{entity.username}/{msg.id}" if hasattr(entity, 'username') else "N/A",
                                sender_id=msg.sender_id,
                                sender_username=sender_username
                            )
                            messages.append(message_data)
                            message_count += 1
                    
                    offset_id = history.messages[-1].id
                    self.logger.info(f"📥 {message_count} پیام از {channel_config.name} استخراج شد")
                    await asyncio.sleep(delay)
                    
                except FloodWaitError as e:
                    self.logger.warning(f"⏳ FloodWait در {channel_config.name}: {e.seconds} ثانیه انتظار...")
                    await asyncio.sleep(e.seconds + 5)
                except Exception as e:
                    self.logger.error(f"❌ خطا در دریافت batch از {channel_config.name}: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            self.logger.error(f"❌ خطای کلی در استخراج کانال {channel_config.name}: {e}")
            
        self.logger.info(f"✅ استخراج کانال {channel_config.name} تکمیل شد: {len(messages)} پیام")
        return messages

    def _infer_symbol_advanced(self, text: str, channel_type: str) -> Optional[str]:
        """تشخیص هوشمند نماد ارز دیجیتال (بهبود یافته)"""
        
        # الگو برای جفت‌ارزهای کامل
        pair_patterns = [
            r'([A-Z]{2,})[-_/]?(USDT|USD|BUSD|BTC|ETH|BNB|USDC|DAI|MATIC)',
            r'([A-Z]{2,})\s*[/]\s*(USDT|USD|BUSD|BTC|ETH|BNB|USDC)',
            r'([A-Z]{2,})\s*به\s*(USDT|USD|تتر|دلار)',
        ]
        
        for pattern in pair_patterns:
            match = re.search(pattern, text.upper())
            if match:
                base, quote = match.groups()
                # تبدیل quote فارسی به انگلیسی
                quote = quote.replace('تتر', 'USDT').replace('دلار', 'USD')
                return f"{base}/{quote}"
        
        # الگو برای نمادهای تکی (با در نظر گیری نوع کانال)
        common_symbols = [
            "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "SHIB", "MATIC", "LTC", 
            "DOT", "AVAX", "LINK", "BCH", "UNI", "FIL", "ATOM", "VET", "TRX", "ETC",
            "OP", "ARB", "APT", "NEAR", "FTM", "RNDR", "GRT", "MANA", "SAND", "ICP"
        ]
        
        # جستجو در متن با الگوهای مختلف
        symbol_patterns = [
            r'\b(' + '|'.join(common_symbols) + r')\b',
            r'#(' + '|'.join(common_symbols) + r')\b',
            r'(' + '|'.join(common_symbols) + r')[\s]*[:/]',
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                symbol = matches[0]
                # برای کانال‌های قیمت، احتمال بالاتری دارد که USDT باشد
                if channel_type.lower() in ['قیمت', 'price', 'تحلیل', 'analysis']:
                    return f"{symbol}/USDT"
                else:
                    return f"{symbol}/USDT"  # پیش‌فرض

        return None

    def _detect_channel_content_type(self, text: str) -> str:
        """تشخیص نوع محتوای کانال (خبری، قیمتی، تحلیلی)"""
        
        # الگوهای قیمت و عدد
        price_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,234.56
            r'[\d,]+\.\d+\s*(دلار|USD|USDT)',  # 1,234.56 دلار
            r'قیمت\s*[:\s]*[\d,]+',  # قیمت: 1234
            r'[\d,]+\s*(تومان|ریال)',  # 1234 تومان
        ]
        
        # الگوهای تحلیل
        analysis_patterns = [
            r'(تحلیل|آنالیز|بررسی|نظر)',
            r'(خرید|فروش|نگهداری|HOLD|BUY|SELL)',
            r'(پیش\s*بینی|prediction|forecast)',
            r'(هدف|target|resistance|support)',
            r'(صعودی|نزولی|bullish|bearish)',
        ]
        
        # الگوهای خبری
        news_patterns = [
            r'(خبر|اخبار|news|اعلام)',
            r'(بازار|market|ارز)',
            r'(اعلان|announcement)',
        ]
        
        text_upper = text.upper()
        
        # بررسی قیمت
        if any(re.search(pattern, text) for pattern in price_patterns):
            return 'price'
        
        # بررسی تحلیل
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in analysis_patterns):
            return 'analysis'
        
        # بررسی خبر (پیش‌فرض)
        return 'news'

    def _extract_numbers_and_prices(self, text: str) -> Dict[str, Any]:
        """استخراج اعداد، قیمت‌ها و پیش‌بینی‌ها از متن"""
        extracted_data = {
            'prices': [],
            'percentages': [],
            'volumes': [],
            'targets': [],
            'predictions': []
        }
        
        # الگوهای قیمت
        price_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,234.56
            r'[\d,]+\.?\d+\s*(دلار|USD|USDT)',  # 1,234.56 دلار
            r'قیمت\s*[:\s]*([\d,]+\.?\d*)',  # قیمت: 1234.56
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            extracted_data['prices'].extend(matches)
        
        # الگوهای درصد
        percentage_patterns = [
            r'[\d,]+\.?\d*\s*%',  # 12.5%
            r'[\d,]+\.?\d*\s*درصد',  # 12.5 درصد
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, text)
            extracted_data['percentages'].extend(matches)
        
        # الگوهای هدف قیمتی
        target_patterns = [
            r'هدف\s*[:\s]*([\d,]+\.?\d*)',  # هدف: 1234
            r'target\s*[:\s]*([\d,]+\.?\d*)',  # target: 1234
        ]
        
        for pattern in target_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted_data['targets'].extend(matches)
        
        return extracted_data

    def _detect_language(self, text: str) -> str:
        """تشخیص ساده زبان (فارسی یا انگلیسی)"""
        if re.search(r'[\u0600-\u06FF]', text):
            return 'fa'
        return 'en'

    def _analyze_sentiment(self, text: str) -> float:
        """تحلیل احساسات متن"""
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound']
        except:
            return 0.0

    # =============================================================================
    # 💾 بخش ذخیره‌سازی (سازگار 100% با ETL)
    # =============================================================================

    def save_for_etl_project(self, messages: List[MessageData], channel_name: str):
        """
        ذخیره داده‌ها در فرمت CSV سازگار کامل با etl_and_merge_02.py
        """
        if not messages:
            self.logger.warning(f"هیچ پیامی در کانال {channel_name} برای ذخیره‌سازی یافت نشد.")
            return

        try:
            etl_data = []
            processed_count = 0
            
            for msg in messages:
                # تشخیص نوع محتوا
                content_type = self._detect_channel_content_type(msg.raw_text)
                
                # تشخیص نماد
                symbol = self._infer_symbol_advanced(msg.raw_text, content_type)
                
                # اگر نماد یافت نشد، از کل متن سعی کن دوباره استخراج کنی
                if not symbol:
                    # تلاش دوم با الگوهای کلی‌تر
                    words = msg.raw_text.upper().split()
                    crypto_keywords = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE"]
                    for word in words:
                        clean_word = re.sub(r'[^\w]', '', word)
                        if clean_word in crypto_keywords:
                            symbol = f"{clean_word}/USDT"
                            break
                
                # فقط پیام‌هایی را ذخیره کن که نماد دارند
                if symbol:
                    # استخراج عنوان (خط اول) و محتوا (بقیه خطوط)
                    lines = msg.raw_text.strip().split('\n')
                    title = lines[0] if lines else ""
                    content = "\n".join(lines[1:]) if len(lines) > 1 else ""
                    
                    # استخراج اعداد و قیمت‌ها
                    extracted_numbers = self._extract_numbers_and_prices(msg.raw_text)
                    
                    # تحلیل احساسات
                    sentiment_score = self._analyze_sentiment(msg.raw_text)

                    row = {
                        'timestamp': msg.date,
                        'symbol': symbol,
                        'title': title[:200],  # محدود کردن عنوان
                        'content': content,
                        'description': msg.message[:250],  # خلاصه برای description
                        'source': msg.channel,
                        'url': msg.url,
                        'language': self._detect_language(msg.raw_text),
                        'api_source': 'Telegram',  # منبع API برای ETL
                        'sentiment_score': sentiment_score,  # سازگار با فایل 02
                        'channel_type': content_type,  # نوع کانال
                        'extracted_prices': str(extracted_numbers['prices']),  # اعداد استخراج شده
                        'extracted_targets': str(extracted_numbers['targets']),  # اهداف قیمتی
                        'extracted_percentages': str(extracted_numbers['percentages']),  # درصدها
                    }
                    etl_data.append(row)
                    processed_count += 1
            
            if not etl_data:
                self.logger.warning(f"هیچ پیامی با نماد مشخص در کانال {channel_name} برای ذخیره‌سازی ETL یافت نشد.")
                return

            df = pd.DataFrame(etl_data)
            
            # نام‌گذاری فایل سازگار با etl_and_merge_02.py
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_telegram_{channel_name}_{timestamp_str}.csv"
            output_path = os.path.join(self.raw_data_path, filename)
            
            # ذخیره با encoding سازگار
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"📊 CSV سازگار با ETL ذخیره شد: {output_path}")
            self.logger.info(f"   📈 تعداد پیام‌های پردازش شده: {processed_count} از {len(messages)}")
            self.logger.info(f"   💰 نمادهای یافت شده: {df['symbol'].nunique()} نماد منحصر به فرد")
            
            # نمایش آمار نمادها
            symbol_counts = df['symbol'].value_counts()
            self.logger.info(f"   🏆 پرتکرارترین نمادها: {dict(symbol_counts.head(5))}")
            
        except Exception as e:
            self.logger.error(f"❌ خطا در ذخیره فایل برای ETL: {e}")

    # =============================================================================
    # 🎯 عملیات اصلی
    # =============================================================================

    async def run_extraction_for_project(self):
        """
        اجرای استخراج برای ادغام با پروژه اصلی
        """
        try:
            await self._init_telegram_client()
            
            active_channels = [ch for ch in self.channels.values() if ch.active]
            self.logger.info(f"🎯 شروع استخراج از {len(active_channels)} کانال فعال برای پروژه...")
            
            total_messages = 0
            total_symbols = set()
            
            for channel_config in active_channels:
                self.logger.info(f"\n--- پردازش کانال: {channel_config.name} (نوع: {channel_config.channel_type}) ---")
                
                messages = await self.extract_channel_messages(channel_config)
                if messages:
                    self.save_for_etl_project(messages, channel_config.name)
                    total_messages += len(messages)
                    
                    # شمارش نمادهای منحصر به فرد
                    for msg in messages:
                        symbol = self._infer_symbol_advanced(msg.raw_text, channel_config.channel_type)
                        if symbol:
                            total_symbols.add(symbol)
            
            self.logger.info(f"\n🎉 استخراج کامل تکمیل شد:")
            self.logger.info(f"   📊 کل پیام‌های پردازش شده: {total_messages}")
            self.logger.info(f"   💰 کل نمادهای شناسایی شده: {len(total_symbols)}")
            self.logger.info(f"   📁 فایل‌های ذخیره شده در: {self.raw_data_path}")
            self.logger.info(f"\n✅ آماده برای پردازش با etl_and_merge_02.py")
            
        except Exception as e:
            self.logger.critical(f"❌ خطای کلی در فرآیند استخراج: {e}")
        finally:
            if self.client:
                await self.client.disconnect()
                self.logger.info("🔌 اتصال از تلگرام قطع شد.")

# =============================================================================
# 🎛️ رابط خط فرمان (سازگار با پروژه)
# =============================================================================

def show_main_menu():
    """نمایش منوی اصلی"""
    print("\n" + "="*80)
    print("🔥 Telegram Extractor برای پروژه (ادغام شده)")
    print("="*80)
    print("1️⃣  اجرای استخراج (آماده‌سازی برای etl_and_merge_02.py)")
    print("2️⃣  نمایش لیست کانال‌های تعریف شده")
    print("3️⃣  تست اتصال به تلگرام")
    print("4️⃣  نمایش آمار فایل‌های موجود")
    print("0️⃣  خروج")
    print("="*80)

def list_configured_channels(extractor: TelegramExtractorForProject):
    """نمایش لیست کانال‌ها"""
    print("\n📺 لیست کانال‌های تعریف شده:")
    print("-" * 100)
    print(f"{'نام':<25} {'نوع':<15} {'وضعیت':<10} {'URL':<50}")
    print("-" * 100)
    
    for channel in extractor.channels.values():
        status = "فعال" if channel.active else "غیرفعال"
        print(f"{channel.name:<25} {channel.channel_type:<15} {status:<10} {channel.url:<50}")
    print("-" * 100)
    print(f"📊 کل: {len(extractor.channels)} کانال ({len([c for c in extractor.channels.values() if c.active])} فعال)")

def show_existing_files_stats(extractor: TelegramExtractorForProject):
    """نمایش آمار فایل‌های موجود"""
    try:
        import glob
        
        # فایل‌های telegram
        telegram_files = glob.glob(os.path.join(extractor.raw_data_path, "news_telegram_*.csv"))
        
        print(f"\n📁 آمار فایل‌های telegram در {extractor.raw_data_path}:")
        print("-" * 60)
        
        if telegram_files:
            total_rows = 0
            symbols_found = set()
            
            for file_path in telegram_files:
                try:
                    df = pd.read_csv(file_path)
                    total_rows += len(df)
                    if 'symbol' in df.columns:
                        symbols_found.update(df['symbol'].dropna().unique())
                    
                    file_info = f"   📄 {os.path.basename(file_path)}: {len(df)} ردیف"
                    print(file_info)
                except:
                    print(f"   ❌ خطا در خواندن: {os.path.basename(file_path)}")
            
            print("-" * 60)
            print(f"📊 خلاصه:")
            print(f"   📁 تعداد فایل: {len(telegram_files)}")
            print(f"   📝 کل ردیف‌ها: {total_rows:,}")
            print(f"   💰 نمادهای منحصر به فرد: {len(symbols_found)}")
            
            if symbols_found:
                print(f"   🏆 نمونه نمادها: {', '.join(list(symbols_found)[:10])}")
        else:
            print("   📭 هیچ فایل telegram یافت نشد")
            print("   💡 ابتدا گزینه 1 را برای استخراج انتخاب کنید")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ خطا در نمایش آمار: {e}")

async def test_connection(extractor: TelegramExtractorForProject):
    """تست اتصال به تلگرام"""
    try:
        print("🔄 در حال تست اتصال به تلگرام...")
        await extractor._init_telegram_client()
        me = await extractor.client.get_me()
        print(f"✅ اتصال موفق! کاربر: {me.first_name}")
        
        # تست دسترسی به کانال‌ها
        accessible_channels = 0
        for channel in extractor.channels.values():
            if channel.active:
                try:
                    entity = await extractor.client.get_entity(channel.url)
                    accessible_channels += 1
                    print(f"   ✅ {channel.name}: قابل دسترس")
                except:
                    print(f"   ❌ {channel.name}: مشکل دسترسی")
        
        print(f"\n📊 نتیجه: {accessible_channels}/{len([c for c in extractor.channels.values() if c.active])} کانال قابل دسترس")
        
    except Exception as e:
        print(f"❌ خطا در اتصال: {e}")
    finally:
        if extractor.client:
            await extractor.client.disconnect()

async def main():
    """تابع اصلی برنامه"""
    print("🚀 در حال راه‌اندازی Telegram Extractor...")
    
    try:
        extractor = TelegramExtractorForProject()
    except SystemExit:
        print("برنامه به دلیل خطای پیکربندی متوقف شد.")
        return

    while True:
        show_main_menu()
        choice = input("انتخاب کنید: ").strip()
        
        if choice == "1":
            print("⏳ شروع فرآیند استخراج...")
            await extractor.run_extraction_for_project()
            print("\n✅ فرآیند استخراج تکمیل شد.")
            print("🔄 اکنون می‌توانید etl_and_merge_02.py را اجرا کنید.")
            input("برای ادامه Enter بزنید...")
        
        elif choice == "2":
            list_configured_channels(extractor)
            input("\nبرای ادامه Enter بزنید...")

        elif choice == "3":
            await test_connection(extractor)
            input("\nبرای ادامه Enter بزنید...")
            
        elif choice == "4":
            show_existing_files_stats(extractor)
            input("\nبرای ادامه Enter بزنید...")

        elif choice == "0":
            print("👋 خداحافظ!")
            break
            
        else:
            print("❌ انتخاب نامعتبر!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 برنامه توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\n❌ خطای کلی: {e}")