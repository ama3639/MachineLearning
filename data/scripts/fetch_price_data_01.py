#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
استخراج داده‌های قیمت ساده‌شده - فقط Binance
نسخه Simplified v2.0

🎯 هدف: استخراج سریع و قابل اعتماد داده‌های قیمت
📊 منبع: فقط Binance API (بهترین کیفیت، بدون محدودیت)
🚀 مزایا: 99% uptime، سرعت بالا، رایگان کامل

ویژگی‌ها:
- استخراج از Binance (2000+ جفت ارز)
- پشتیبانی از همه timeframe ها
- مدیریت وضعیت ساده
- خطایابی قوی
- عملکرد بهینه
"""

import pandas as pd
import logging
from datetime import datetime
from typing import List, Optional
from typing import Dict

# import اجزای مشترک
from fetch_historical_data_01 import (
    state_manager, rate_limiter, setup_logging,
    safe_request, sanitize_filename, get_user_selection,
    COMMON_SYMBOLS, COMMON_TIMEFRAMES, DEFAULT_LIMIT,
    RAW_DATA_PATH
)

# --- تنظیم لاگ‌گیری ---
setup_logging('fetch_price_data_01')

class BinancePriceFetcher:
    """
    کلاس استخراج قیمت از Binance
    
    ویژگی‌ها:
    - رایگان و بدون محدودیت
    - سرعت بالا (0.1 ثانیه delay)
    - کیفیت داده عالی
    - پشتیبانی از همه timeframe ها
    """
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.timeframe_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        logging.info("🏦 Binance Price Fetcher اولیه‌سازی شد")
    
    def get_available_symbols(self) -> List[str]:
        """
        دریافت لیست نمادهای موجود در Binance
        
        Returns:
            لیست نمادهای USDT
        """
        try:
            logging.info("📊 دریافت لیست نمادهای Binance...")
            
            response = safe_request("https://api.binance.com/api/v3/exchangeInfo")
            data = response.json()
            
            usdt_symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] == 'USDT'):
                    symbol = f"{symbol_info['baseAsset']}/USDT"
                    usdt_symbols.append(symbol)
            
            logging.info(f"✅ تعداد {len(usdt_symbols)} نماد USDT یافت شد")
            return usdt_symbols
            
        except Exception as e:
            logging.error(f"❌ خطا در دریافت لیست نمادها: {e}")
            return COMMON_SYMBOLS  # fallback به لیست پیش‌فرض
    
    def fetch_symbol_data(self, symbol: str, timeframe: str, limit: int = DEFAULT_LIMIT) -> Optional[pd.DataFrame]:
        """
        استخراج داده قیمت یک نماد
        
        Args:
            symbol: نماد ارز (مثل BTC/USDT)
            timeframe: تایم‌فریم (مثل 1h)
            limit: تعداد کندل (حداکثر 1000)
        
        Returns:
            DataFrame شامل OHLCV یا None در صورت خطا
        """
        try:
            # تبدیل نماد به فرمت Binance
            binance_symbol = symbol.replace('/', '').upper()
            
            # بررسی timeframe
            binance_timeframe = self.timeframe_map.get(timeframe, timeframe)
            
            # محدود کردن limit به حداکثر مجاز
            limit = min(limit, 1000)
            
            # پارامترهای درخواست
            params = {
                'symbol': binance_symbol,
                'interval': binance_timeframe,
                'limit': limit
            }
            
            logging.info(f"📈 استخراج {symbol} | {timeframe} | {limit} کندل...")
            
            # اعمال rate limiting
            rate_limiter.wait_if_needed('binance')
            
            # ارسال درخواست
            response = safe_request(self.base_url, params=params)
            data = response.json()
            
            if not data:
                logging.warning(f"⚠️ داده‌ای برای {symbol} دریافت نشد")
                return None
            
            # تبدیل به DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # انتخاب ستون‌های مورد نیاز
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # تبدیل انواع داده
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # تبدیل timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # افزودن متادیتا
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            df['exchange'] = 'Binance'
            
            # حذف داده‌های نامعتبر
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            
            if df.empty:
                logging.warning(f"⚠️ همه داده‌های {symbol} نامعتبر بودند")
                return None
            
            logging.info(f"✅ {symbol} | {timeframe}: {len(df)} کندل دریافت شد")
            return df
            
        except Exception as e:
            logging.error(f"❌ خطا در استخراج {symbol} | {timeframe}: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        ذخیره داده در فایل CSV
        
        Args:
            df: DataFrame داده‌ها
            symbol: نماد ارز
            timeframe: تایم‌فریم
        
        Returns:
            مسیر فایل ذخیره شده
        """
        try:
            # نام‌گذاری فایل
            symbol_clean = sanitize_filename(symbol)
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"Binance_{symbol_clean}_{timeframe}_{timestamp_str}.csv"
            file_path = f"{RAW_DATA_PATH}/{filename}"
            
            # ذخیره فایل
            df.to_csv(file_path, index=False, float_format='%.8f')
            
            logging.info(f"💾 فایل ذخیره شد: {filename}")
            return file_path
            
        except Exception as e:
            logging.error(f"❌ خطا در ذخیره {symbol}: {e}")
            return ""

class PriceDataManager:
    """
    مدیریت کلی استخراج داده‌های قیمت
    
    ویژگی‌ها:
    - مدیریت سشن‌ها
    - آمار پیشرفت
    - گزارش‌دهی جامع
    """
    
    def __init__(self):
        self.fetcher = BinancePriceFetcher()
        self.session_id = None
        logging.info("🎯 مدیریت داده‌های قیمت آماده شد")
    
    def fetch_single_symbol(self, symbol: str, timeframes: List[str], 
                          limit: int = DEFAULT_LIMIT) -> int:
        """
        استخراج داده یک نماد در همه timeframe ها
        
        Args:
            symbol: نماد ارز
            timeframes: لیست timeframe ها
            limit: تعداد کندل
        
        Returns:
            تعداد فایل‌های موفق
        """
        success_count = 0
        
        for timeframe in timeframes:
            try:
                # بروزرسانی وضعیت
                state_manager.update_price_progress(
                    self.session_id, symbol, timeframe, 'processing'
                )
                
                # استخراج داده
                df = self.fetcher.fetch_symbol_data(symbol, timeframe, limit)
                
                if df is not None and not df.empty:
                    # ذخیره فایل
                    file_path = self.fetcher.save_data(df, symbol, timeframe)
                    
                    if file_path:
                        # بروزرسانی وضعیت موفق
                        state_manager.update_price_progress(
                            self.session_id, symbol, timeframe, 'completed',
                            file_path=file_path, records_count=len(df)
                        )
                        success_count += 1
                    else:
                        # خطا در ذخیره
                        state_manager.update_price_progress(
                            self.session_id, symbol, timeframe, 'failed',
                            error_message="خطا در ذخیره فایل"
                        )
                else:
                    # داده خالی
                    state_manager.update_price_progress(
                        self.session_id, symbol, timeframe, 'failed',
                        error_message="داده خالی یا نامعتبر"
                    )
            
            except Exception as e:
                # خطای غیرمنتظره
                logging.error(f"❌ خطا در پردازش {symbol} | {timeframe}: {e}")
                state_manager.update_price_progress(
                    self.session_id, symbol, timeframe, 'failed',
                    error_message=str(e)
                )
        
        return success_count
    
    def fetch_multiple_symbols(self, symbols: List[str], timeframes: List[str], 
                             limit: int = DEFAULT_LIMIT) -> Dict:
        """
        استخراج داده چندین نماد
        
        Args:
            symbols: لیست نمادها
            timeframes: لیست timeframe ها
            limit: تعداد کندل
        
        Returns:
            آمار نتایج
        """
        logging.info(f"🚀 شروع استخراج {len(symbols)} نماد در {len(timeframes)} timeframe")
        
        # ایجاد سشن
        total_items = len(symbols) * len(timeframes)
        self.session_id = state_manager.create_session('price', total_items)
        
        # آمار
        total_success = 0
        total_failed = 0
        
        # پردازش هر نماد
        for i, symbol in enumerate(symbols, 1):
            logging.info(f"📊 [{i}/{len(symbols)}] پردازش {symbol}...")
            
            success_count = self.fetch_single_symbol(symbol, timeframes, limit)
            
            total_success += success_count
            total_failed += len(timeframes) - success_count
            
            # نمایش پیشرفت
            progress = (i / len(symbols)) * 100
            logging.info(f"⚡ پیشرفت: {progress:.1f}% | موفق: {total_success} | ناموفق: {total_failed}")
        
        # اتمام سشن
        state_manager.complete_session(self.session_id)
        
        # آمار نهایی
        stats = {
            'session_id': self.session_id,
            'total_symbols': len(symbols),
            'total_timeframes': len(timeframes),
            'total_items': total_items,
            'successful_files': total_success,
            'failed_items': total_failed,
            'success_rate': round((total_success / total_items) * 100, 1) if total_items > 0 else 0
        }
        
        return stats

def main():
    """تابع اصلی - منوی تعاملی"""
    logging.info("🚀 استخراج داده‌های قیمت ساده‌شده (Binance Only)")
    
    # نمایش تنظیمات
    print("\n" + "="*70)
    print("🏦 استخراج داده‌های قیمت ساده‌شده")
    print("="*70)
    print("📊 منبع: Binance (بهترین کیفیت، بدون محدودیت)")
    print("🚀 مزایا: سرعت بالا، 99% uptime، رایگان کامل")
    print("⚡ تأخیر: 0.1 ثانیه بین درخواست‌ها")
    print("📈 پشتیبانی: 2000+ جفت ارز، همه timeframe ها")
    
    # اولیه‌سازی مدیریت
    manager = PriceDataManager()
    
    while True:
        print("\n" + "="*50)
        print("📋 منوی اصلی استخراج قیمت")
        print("="*50)
        print("1. استخراج نمادهای محبوب (30 نماد)")
        print("2. استخراج سفارشی (انتخاب از لیست)")
        print("3. استخراج همه نمادهای Binance")
        print("4. نمایش آمار سشن‌های قبلی")
        print("5. خروج")
        
        choice = input("\nانتخاب شما: ").strip()
        
        if choice == '1':
            # نمادهای محبوب
            print(f"\n📊 استخراج {len(COMMON_SYMBOLS)} نماد محبوب...")
            
            # انتخاب timeframe ها
            timeframes = get_user_selection(
                COMMON_TIMEFRAMES, 
                "انتخاب timeframe ها",
                allow_multi=True, 
                allow_all=True
            )
            
            if not timeframes:
                print("❌ هیچ timeframe انتخاب نشد")
                continue
            
            # تنظیم تعداد کندل
            limit_choice = input(f"\nتعداد کندل (پیش‌فرض: {DEFAULT_LIMIT}): ").strip()
            limit = int(limit_choice) if limit_choice.isdigit() else DEFAULT_LIMIT
            limit = min(limit, 1000)  # محدودیت Binance
            
            # تأیید نهایی
            total_requests = len(COMMON_SYMBOLS) * len(timeframes)
            estimated_time = (total_requests * 0.1) / 60  # دقیقه
            
            print(f"\n📊 خلاصه:")
            print(f"   نمادها: {len(COMMON_SYMBOLS)}")
            print(f"   Timeframe ها: {len(timeframes)}")
            print(f"   کل درخواست‌ها: {total_requests}")
            print(f"   زمان تخمینی: {estimated_time:.1f} دقیقه")
            
            confirm = input("\nادامه می‌دهید؟ (y/n): ").lower()
            if confirm != 'y':
                continue
            
            # اجرای استخراج
            stats = manager.fetch_multiple_symbols(COMMON_SYMBOLS, timeframes, limit)
            
            # نمایش نتایج
            print(f"\n✅ استخراج تکمیل شد!")
            print(f"📊 موفق: {stats['successful_files']}/{stats['total_items']} ({stats['success_rate']}%)")
            print(f"🆔 Session ID: {stats['session_id']}")
        
        elif choice == '2':
            # انتخاب سفارشی
            print("\n📊 استخراج سفارشی...")
            
            # انتخاب نمادها
            symbols = get_user_selection(
                COMMON_SYMBOLS, 
                "انتخاب نمادها",
                allow_manual=True,
                allow_multi=True,
                allow_all=True
            )
            
            if not symbols:
                print("❌ هیچ نمادی انتخاب نشد")
                continue
            
            # انتخاب timeframe ها
            timeframes = get_user_selection(
                COMMON_TIMEFRAMES,
                "انتخاب timeframe ها", 
                allow_multi=True,
                allow_all=True
            )
            
            if not timeframes:
                print("❌ هیچ timeframe انتخاب نشد")
                continue
            
            # تنظیم تعداد کندل
            limit_choice = input(f"\nتعداد کندل (پیش‌فرض: {DEFAULT_LIMIT}): ").strip()
            limit = int(limit_choice) if limit_choice.isdigit() else DEFAULT_LIMIT
            limit = min(limit, 1000)
            
            # تأیید نهایی
            total_requests = len(symbols) * len(timeframes)
            estimated_time = (total_requests * 0.1) / 60
            
            print(f"\n📊 خلاصه:")
            print(f"   نمادها: {len(symbols)}")
            print(f"   Timeframe ها: {len(timeframes)}")
            print(f"   کل درخواست‌ها: {total_requests}")
            print(f"   زمان تخمینی: {estimated_time:.1f} دقیقه")
            
            confirm = input("\nادامه می‌دهید؟ (y/n): ").lower()
            if confirm != 'y':
                continue
            
            # اجرای استخراج
            stats = manager.fetch_multiple_symbols(symbols, timeframes, limit)
            
            # نمایش نتایج
            print(f"\n✅ استخراج تکمیل شد!")
            print(f"📊 موفق: {stats['successful_files']}/{stats['total_items']} ({stats['success_rate']}%)")
            print(f"🆔 Session ID: {stats['session_id']}")
        
        elif choice == '3':
            # همه نمادهای Binance
            print("\n🌍 استخراج همه نمادهای Binance...")
            print("⚠️ این عملیات ممکن است چندین ساعت طول بکشد")
            
            # دریافت لیست کامل
            all_symbols = manager.fetcher.get_available_symbols()
            
            if not all_symbols:
                print("❌ خطا در دریافت لیست نمادها")
                continue
            
            print(f"📊 تعداد {len(all_symbols)} نماد یافت شد")
            
            # انتخاب timeframe ها
            timeframes = get_user_selection(
                COMMON_TIMEFRAMES,
                "انتخاب timeframe ها",
                allow_multi=True,
                allow_all=True
            )
            
            if not timeframes:
                print("❌ هیچ timeframe انتخاب نشد")
                continue
            
            # محاسبه زمان تخمینی
            total_requests = len(all_symbols) * len(timeframes)
            estimated_hours = (total_requests * 0.1) / 3600
            
            print(f"\n📊 خلاصه:")
            print(f"   نمادها: {len(all_symbols)}")
            print(f"   Timeframe ها: {len(timeframes)}")
            print(f"   کل درخواست‌ها: {total_requests:,}")
            print(f"   زمان تخمینی: {estimated_hours:.1f} ساعت")
            
            confirm = input("\n⚠️ آیا مطمئن هستید؟ (yes/no): ").lower()
            if confirm != 'yes':
                continue
            
            # اجرای استخراج
            stats = manager.fetch_multiple_symbols(all_symbols, timeframes, DEFAULT_LIMIT)
            
            # نمایش نتایج
            print(f"\n🎉 استخراج کامل تکمیل شد!")
            print(f"📊 موفق: {stats['successful_files']:,}/{stats['total_items']:,} ({stats['success_rate']}%)")
            print(f"🆔 Session ID: {stats['session_id']}")
        
        elif choice == '4':
            # نمایش آمار
            print("\n📊 آمار سشن‌های قبلی:")
            
            # نمایش آمار rate limiter
            rate_stats = rate_limiter.get_stats()
            print(f"📈 کل درخواست‌ها: {rate_stats['total_requests']:,}")
            print(f"   Binance: {rate_stats['binance_requests']:,}")
            
            input("\nEnter برای ادامه...")
        
        elif choice == '5':
            print("\n👋 خداحافظ!")
            logging.info("استخراج داده‌های قیمت به پایان رسید")
            break
        
        else:
            print("❌ انتخاب نامعتبر")

if __name__ == '__main__':
    main()
