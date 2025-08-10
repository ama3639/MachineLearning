#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
استخراج داده‌های اخبار ساده‌شده - فقط RSS
نسخه Simplified v2.0

🎯 هدف: استخراج مطمئن و بدون محدودیت اخبار کریپتو
📡 منبع: فقط RSS Feeds (کیفیت بالا، بدون محدودیت)
🚀 مزایا: رایگان کامل، پایدار، اخبار معتبر

ویژگی‌ها:
- استخراج از RSS feeds معتبر (CoinDesk, CoinTelegraph)
- تحلیل احساسات با VADER
- کش محلی برای بهبود سرعت  
- مدیریت وضعیت ساده
- خطایابی قوی
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

# تلاش برای import کردن feedparser
try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False
    logging.warning("⚠️ feedparser در دسترس نیست. نصب کنید: pip install feedparser")

# تحلیل احساسات
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logging.warning("⚠️ vaderSentiment در دسترس نیست. نصب کنید: pip install vaderSentiment")

# import اجزای مشترک
from fetch_historical_data_01 import (
    state_manager, rate_limiter, setup_logging,
    safe_request, sanitize_filename, get_user_selection,
    COMMON_SYMBOLS, MAX_NEWS_PER_SYMBOL,
    RAW_DATA_PATH
)

# --- تنظیم لاگ‌گیری ---
setup_logging('fetch_news_data_01A')

class RSSNewsFetcher:
    """
    کلاس استخراج اخبار از RSS feeds
    
    ویژگی‌ها:
    - رایگان و بدون محدودیت
    - کیفیت اخبار بالا
    - سرعت مناسب (کش محلی)
    - منابع معتبر
    """
    
    def __init__(self):
        # RSS feeds معتبر کریپتو
        self.rss_feeds = {
            'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'CoinTelegraph': 'https://cointelegraph.com/rss',
            'Decrypt': 'https://decrypt.co/feed',
            'CryptoNews': 'https://cryptonews.com/news/feed'
        }
        
        # کش محلی
        self._feed_cache = {}
        self._last_fetch = {}
        self.cache_duration = 300  # 5 دقیقه
        
        # تحلیل احساسات
        if SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
        
        logging.info("📡 RSS News Fetcher اولیه‌سازی شد")
        logging.info(f"📰 منابع: {list(self.rss_feeds.keys())}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        تحلیل احساسات متن
        
        Args:
            text: متن برای تحلیل
        
        Returns:
            دیکشنری امتیازات احساسات
        """
        if not self.sentiment_analyzer or not text:
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logging.warning(f"⚠️ خطا در تحلیل احساسات: {e}")
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
    
    def fetch_rss_feed(self, feed_name: str, feed_url: str) -> List[Dict]:
        """
        دریافت محتوای یک RSS feed
        
        Args:
            feed_name: نام feed
            feed_url: آدرس RSS
        
        Returns:
            لیست مقالات
        """
        if not RSS_AVAILABLE:
            logging.error("❌ feedparser در دسترس نیست")
            return []
        
        current_time = time.time()
        
        # بررسی کش
        if (feed_name in self._last_fetch and 
            current_time - self._last_fetch[feed_name] < self.cache_duration):
            logging.info(f"📋 استفاده از کش برای {feed_name}")
            return self._feed_cache.get(feed_name, [])
        
        try:
            logging.info(f"📡 دریافت RSS: {feed_name}")
            
            # اعمال rate limiting
            rate_limiter.wait_if_needed('rss')
            
            # دریافت feed
            feed = feedparser.parse(feed_url)
            
            # بررسی موفقیت
            if hasattr(feed, 'status') and feed.status != 200:
                logging.warning(f"⚠️ RSS {feed_name} وضعیت {feed.status} برگرداند")
                return []
            
            if not hasattr(feed, 'entries') or not feed.entries:
                logging.warning(f"⚠️ RSS {feed_name} خالی است")
                return []
            
            # پردازش مقالات
            articles = []
            for entry in feed.entries[:20]:  # حداکثر 20 مقاله
                try:
                    # استخراج اطلاعات
                    title = entry.get('title', '').strip()
                    summary = entry.get('summary', '').strip()
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    # ترکیب متن برای تحلیل احساسات
                    full_text = f"{title}. {summary}"
                    
                    # تحلیل احساسات
                    sentiment = self.analyze_sentiment(full_text)
                    
                    # ایجاد رکورد مقاله
                    article = {
                        'timestamp': self._parse_published_date(published),
                        'title': title,
                        'content': summary,
                        'description': summary,
                        'source': feed_name,
                        'url': link,
                        'language': 'en',
                        'api_source': 'RSS',
                        'full_text': full_text,
                        
                        # امتیازات احساسات
                        'sentiment_compound': sentiment['compound'],
                        'sentiment_positive': sentiment['pos'],
                        'sentiment_negative': sentiment['neg'],
                        'sentiment_neutral': sentiment['neu'],
                        
                        # برچسب احساسات
                        'sentiment_label': self._get_sentiment_label(sentiment['compound']),
                        
                        # ویژگی‌های اضافی
                        'text_length': len(full_text),
                        'title_length': len(title)
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    logging.warning(f"⚠️ خطا در پردازش مقاله {feed_name}: {e}")
                    continue
            
            # ذخیره در کش
            self._feed_cache[feed_name] = articles
            self._last_fetch[feed_name] = current_time
            
            logging.info(f"✅ {feed_name}: {len(articles)} مقاله دریافت شد")
            return articles
            
        except Exception as e:
            logging.error(f"❌ خطا در دریافت RSS {feed_name}: {e}")
            return []
    
    def _parse_published_date(self, published_str: str) -> str:
        """تبدیل تاریخ انتشار به فرمت ISO"""
        try:
            if not published_str:
                return datetime.now().isoformat()
            
            # تلاش برای تبدیل با feedparser
            import time
            time_struct = feedparser._parse_date(published_str)
            if time_struct:
                dt = datetime(*time_struct[:6])
                return dt.isoformat()
            
            # fallback به زمان فعلی
            return datetime.now().isoformat()
            
        except Exception:
            return datetime.now().isoformat()
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """تعیین برچسب احساسات بر اساس امتیاز compound"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def fetch_all_feeds(self) -> List[Dict]:
        """
        دریافت همه RSS feeds
        
        Returns:
            لیست کل مقالات
        """
        all_articles = []
        
        for feed_name, feed_url in self.rss_feeds.items():
            articles = self.fetch_rss_feed(feed_name, feed_url)
            all_articles.extend(articles)
        
        logging.info(f"📰 مجموع {len(all_articles)} مقاله از {len(self.rss_feeds)} منبع")
        return all_articles
    
    def filter_relevant_news(self, articles: List[Dict], symbol: str, 
                           max_news: int = MAX_NEWS_PER_SYMBOL) -> List[Dict]:
        """
        فیلتر اخبار مرتبط با نماد
        
        Args:
            articles: لیست همه مقالات
            symbol: نماد ارز (مثل BTC/USDT)
            max_news: حداکثر تعداد اخبار
        
        Returns:
            لیست اخبار مرتبط
        """
        crypto_name = symbol.split('/')[0].lower()
        relevant_articles = []
        
        # کلمات کلیدی مرتبط
        keywords = [
            crypto_name,
            'crypto', 'cryptocurrency', 'bitcoin', 'blockchain',
            'digital currency', 'altcoin', 'defi', 'nft'
        ]
        
        for article in articles:
            title_lower = article.get('title', '').lower()
            content_lower = article.get('content', '').lower()
            
            # بررسی وجود کلمات کلیدی
            is_relevant = False
            
            # اولویت بالا: نام ارز در عنوان یا محتوا
            if crypto_name in title_lower or crypto_name in content_lower:
                is_relevant = True
            
            # اولویت متوسط: کلمات کلیدی کریپتو
            elif any(keyword in title_lower or keyword in content_lower for keyword in keywords):
                is_relevant = True
            
            if is_relevant:
                # اضافه کردن نماد به مقاله
                article_copy = article.copy()
                article_copy['symbol'] = symbol
                relevant_articles.append(article_copy)
                
                if len(relevant_articles) >= max_news:
                    break
        
        logging.info(f"🎯 {symbol}: {len(relevant_articles)} خبر مرتبط یافت شد")
        return relevant_articles

class NewsDataManager:
    """
    مدیریت کلی استخراج داده‌های اخبار
    
    ویژگی‌ها:
    - مدیریت سشن‌ها
    - آمار پیشرفت
    - گزارش‌دهی جامع
    """
    
    def __init__(self):
        self.fetcher = RSSNewsFetcher()
        self.session_id = None
        logging.info("🎯 مدیریت داده‌های اخبار آماده شد")
    
    def save_news_data(self, articles: List[Dict], symbol: str) -> str:
        """
        ذخیره اخبار در فایل CSV
        
        Args:
            articles: لیست اخبار
            symbol: نماد ارز
        
        Returns:
            مسیر فایل ذخیره شده
        """
        try:
            if not articles:
                return ""
            
            # تبدیل به DataFrame
            df = pd.DataFrame(articles)
            
            # نام‌گذاری فایل
            symbol_clean = sanitize_filename(symbol)
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"news_{symbol_clean}_en_{timestamp_str}.csv"
            file_path = f"{RAW_DATA_PATH}/{filename}"
            
            # ذخیره فایل
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            logging.info(f"💾 فایل اخبار ذخیره شد: {filename}")
            return file_path
            
        except Exception as e:
            logging.error(f"❌ خطا در ذخیره اخبار {symbol}: {e}")
            return ""
    
    def fetch_symbol_news(self, symbol: str, max_news: int = MAX_NEWS_PER_SYMBOL) -> int:
        """
        استخراج اخبار یک نماد
        
        Args:
            symbol: نماد ارز
            max_news: حداکثر تعداد اخبار
        
        Returns:
            تعداد اخبار دریافت شده
        """
        try:
            # بروزرسانی وضعیت
            state_manager.update_news_progress(
                self.session_id, symbol, 'processing'
            )
            
            # دریافت همه اخبار
            all_articles = self.fetcher.fetch_all_feeds()
            
            if not all_articles:
                logging.warning(f"⚠️ هیچ خبری برای {symbol} یافت نشد")
                state_manager.update_news_progress(
                    self.session_id, symbol, 'failed',
                    error_message="هیچ خبری یافت نشد"
                )
                return 0
            
            # فیلتر اخبار مرتبط
            relevant_news = self.fetcher.filter_relevant_news(all_articles, symbol, max_news)
            
            if not relevant_news:
                logging.warning(f"⚠️ هیچ خبر مرتبطی برای {symbol} یافت نشد")
                state_manager.update_news_progress(
                    self.session_id, symbol, 'failed',
                    error_message="هیچ خبر مرتبط یافت نشد"
                )
                return 0
            
            # ذخیره اخبار
            file_path = self.save_news_data(relevant_news, symbol)
            
            if file_path:
                # موفقیت
                state_manager.update_news_progress(
                    self.session_id, symbol, 'completed',
                    file_path=file_path, news_count=len(relevant_news)
                )
                return len(relevant_news)
            else:
                # خطا در ذخیره
                state_manager.update_news_progress(
                    self.session_id, symbol, 'failed',
                    error_message="خطا در ذخیره فایل"
                )
                return 0
            
        except Exception as e:
            logging.error(f"❌ خطا در استخراج اخبار {symbol}: {e}")
            state_manager.update_news_progress(
                self.session_id, symbol, 'failed',
                error_message=str(e)
            )
            return 0
    
    def fetch_multiple_symbols_news(self, symbols: List[str], 
                                  max_news: int = MAX_NEWS_PER_SYMBOL) -> Dict:
        """
        استخراج اخبار چندین نماد
        
        Args:
            symbols: لیست نمادها
            max_news: حداکثر اخبار به ازای هر نماد
        
        Returns:
            آمار نتایج
        """
        logging.info(f"🚀 شروع استخراج اخبار {len(symbols)} نماد")
        
        # ایجاد سشن
        self.session_id = state_manager.create_session('news', len(symbols))
        
        # آمار
        total_news = 0
        successful_symbols = 0
        failed_symbols = 0
        
        # دریافت یکبار همه اخبار (برای بهبود کارایی)
        logging.info("📡 دریافت همه اخبار RSS...")
        all_articles = self.fetcher.fetch_all_feeds()
        
        if not all_articles:
            logging.error("❌ هیچ خبری از RSS feeds دریافت نشد")
            state_manager.complete_session(self.session_id)
            return {
                'session_id': self.session_id,
                'total_symbols': len(symbols),
                'successful_symbols': 0,
                'failed_symbols': len(symbols),
                'total_news': 0,
                'success_rate': 0
            }
        
        # پردازش هر نماد
        for i, symbol in enumerate(symbols, 1):
            logging.info(f"📰 [{i}/{len(symbols)}] پردازش اخبار {symbol}...")
            
            try:
                # فیلتر اخبار مرتبط
                relevant_news = self.fetcher.filter_relevant_news(all_articles, symbol, max_news)
                
                if relevant_news:
                    # ذخیره اخبار
                    file_path = self.save_news_data(relevant_news, symbol)
                    
                    if file_path:
                        state_manager.update_news_progress(
                            self.session_id, symbol, 'completed',
                            file_path=file_path, news_count=len(relevant_news)
                        )
                        total_news += len(relevant_news)
                        successful_symbols += 1
                    else:
                        state_manager.update_news_progress(
                            self.session_id, symbol, 'failed',
                            error_message="خطا در ذخیره"
                        )
                        failed_symbols += 1
                else:
                    state_manager.update_news_progress(
                        self.session_id, symbol, 'failed',
                        error_message="اخبار مرتبط یافت نشد"
                    )
                    failed_symbols += 1
                
            except Exception as e:
                logging.error(f"❌ خطا در {symbol}: {e}")
                state_manager.update_news_progress(
                    self.session_id, symbol, 'failed',
                    error_message=str(e)
                )
                failed_symbols += 1
            
            # نمایش پیشرفت
            progress = (i / len(symbols)) * 100
            logging.info(f"⚡ پیشرفت: {progress:.1f}% | موفق: {successful_symbols} | اخبار: {total_news}")
        
        # اتمام سشن
        state_manager.complete_session(self.session_id)
        
        # آمار نهایی
        stats = {
            'session_id': self.session_id,
            'total_symbols': len(symbols),
            'successful_symbols': successful_symbols,
            'failed_symbols': failed_symbols,
            'total_news': total_news,
            'avg_news_per_symbol': round(total_news / max(successful_symbols, 1), 1),
            'success_rate': round((successful_symbols / len(symbols)) * 100, 1) if symbols else 0
        }
        
        return stats

def main():
    """تابع اصلی - منوی تعاملی"""
    logging.info("🚀 استخراج داده‌های اخبار ساده‌شده (RSS Only)")
    
    # بررسی وابستگی‌ها
    if not RSS_AVAILABLE:
        print("❌ خطا: feedparser نصب نیست")
        print("نصب کنید: pip install feedparser")
        return
    
    if not SENTIMENT_AVAILABLE:
        print("⚠️ هشدار: vaderSentiment نصب نیست")
        print("نصب کنید: pip install vaderSentiment")
        print("بدون آن، تحلیل احساسات غیرفعال خواهد بود")
    
    # نمایش تنظیمات
    print("\n" + "="*70)
    print("📰 استخراج داده‌های اخبار ساده‌شده")
    print("="*70)
    print("📡 منبع: RSS Feeds (کیفیت بالا، بدون محدودیت)")
    print("🚀 مزایا: رایگان کامل، پایدار، اخبار معتبر")
    print("⚡ کش: 5 دقیقه برای بهبود سرعت")
    print("🎭 تحلیل احساسات: VADER (خودکار)")
    
    # اولیه‌سازی مدیریت
    manager = NewsDataManager()
    
    while True:
        print("\n" + "="*50)
        print("📋 منوی اصلی استخراج اخبار")
        print("="*50)
        print("1. استخراج اخبار نمادهای محبوب (30 نماد)")
        print("2. استخراج سفارشی (انتخاب از لیست)")
        print("3. تست RSS feeds (بررسی اتصال)")
        print("4. نمایش آمار سشن‌های قبلی")
        print("5. خروج")
        
        choice = input("\nانتخاب شما: ").strip()
        
        if choice == '1':
            # نمادهای محبوب
            print(f"\n📰 استخراج اخبار {len(COMMON_SYMBOLS)} نماد محبوب...")
            
            # تنظیم تعداد اخبار
            max_news_choice = input(f"\nحداکثر اخبار به ازای هر نماد (پیش‌فرض: {MAX_NEWS_PER_SYMBOL}): ").strip()
            max_news = int(max_news_choice) if max_news_choice.isdigit() else MAX_NEWS_PER_SYMBOL
            
            # تأیید نهایی
            estimated_time = (len(COMMON_SYMBOLS) * 0.5) / 60  # دقیقه
            
            print(f"\n📊 خلاصه:")
            print(f"   نمادها: {len(COMMON_SYMBOLS)}")
            print(f"   حداکثر اخبار/نماد: {max_news}")
            print(f"   منابع RSS: {len(manager.fetcher.rss_feeds)}")
            print(f"   زمان تخمینی: {estimated_time:.1f} دقیقه")
            
            confirm = input("\nادامه می‌دهید؟ (y/n): ").lower()
            if confirm != 'y':
                continue
            
            # اجرای استخراج
            stats = manager.fetch_multiple_symbols_news(COMMON_SYMBOLS, max_news)
            
            # نمایش نتایج
            print(f"\n✅ استخراج اخبار تکمیل شد!")
            print(f"📊 موفق: {stats['successful_symbols']}/{stats['total_symbols']} نماد ({stats['success_rate']}%)")
            print(f"📰 کل اخبار: {stats['total_news']} (میانگین: {stats['avg_news_per_symbol']}/نماد)")
            print(f"🆔 Session ID: {stats['session_id']}")
        
        elif choice == '2':
            # انتخاب سفارشی
            print("\n📰 استخراج سفارشی اخبار...")
            
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
            
            # تنظیم تعداد اخبار
            max_news_choice = input(f"\nحداکثر اخبار به ازای هر نماد (پیش‌فرض: {MAX_NEWS_PER_SYMBOL}): ").strip()
            max_news = int(max_news_choice) if max_news_choice.isdigit() else MAX_NEWS_PER_SYMBOL
            
            # تأیید نهایی
            estimated_time = (len(symbols) * 0.5) / 60
            
            print(f"\n📊 خلاصه:")
            print(f"   نمادها: {len(symbols)}")
            print(f"   حداکثر اخبار/نماد: {max_news}")
            print(f"   منابع RSS: {len(manager.fetcher.rss_feeds)}")
            print(f"   زمان تخمینی: {estimated_time:.1f} دقیقه")
            
            confirm = input("\nادامه می‌دهید؟ (y/n): ").lower()
            if confirm != 'y':
                continue
            
            # اجرای استخراج
            stats = manager.fetch_multiple_symbols_news(symbols, max_news)
            
            # نمایش نتایج
            print(f"\n✅ استخراج اخبار تکمیل شد!")
            print(f"📊 موفق: {stats['successful_symbols']}/{stats['total_symbols']} نماد ({stats['success_rate']}%)")
            print(f"📰 کل اخبار: {stats['total_news']} (میانگین: {stats['avg_news_per_symbol']}/نماد)")
            print(f"🆔 Session ID: {stats['session_id']}")
        
        elif choice == '3':
            # تست RSS feeds
            print("\n🔍 تست اتصال RSS feeds...")
            
            for feed_name, feed_url in manager.fetcher.rss_feeds.items():
                print(f"\n📡 تست {feed_name}...")
                articles = manager.fetcher.fetch_rss_feed(feed_name, feed_url)
                
                if articles:
                    print(f"✅ {feed_name}: {len(articles)} مقاله دریافت شد")
                    
                    # نمایش نمونه
                    if articles:
                        sample = articles[0]
                        print(f"   📰 نمونه: {sample['title'][:50]}...")
                        print(f"   🎭 احساسات: {sample['sentiment_label']} ({sample['sentiment_compound']:.3f})")
                else:
                    print(f"❌ {feed_name}: خطا در دریافت")
            
            input("\nEnter برای ادامه...")
        
        elif choice == '4':
            # نمایش آمار
            print("\n📊 آمار سشن‌های قبلی:")
            
            # نمایش آمار rate limiter
            rate_stats = rate_limiter.get_stats()
            print(f"📈 کل درخواست‌ها: {rate_stats['total_requests']:,}")
            print(f"   RSS: {rate_stats['rss_requests']:,}")
            
            input("\nEnter برای ادامه...")
        
        elif choice == '5':
            print("\n👋 خداحافظ!")
            logging.info("استخراج داده‌های اخبار به پایان رسید")
            break
        
        else:
            print("❌ انتخاب نامعتبر")

if __name__ == '__main__':
    main()
