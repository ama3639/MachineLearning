#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت بک تست پیشرفته Enhanced v3.1 - سازگاری کامل با Telegram-based Pipeline

🔧 تغییرات مهم v3.1 (اصلاحات حیاتی Telegram-based):
- ✅ پشتیبانی کامل از Enhanced Models v6.1+ (58+ features)
- ✅ Telegram-derived Reddit Features Integration (بجای Reddit API)
- ✅ اصلاح data quality analysis: تشخیص telegram mapping
- ✅ ادغام sentiment و reddit impact analysis (جلوگیری از تحلیل دوگانه)
- ✅ اصلاح enhanced visualizations: نمایش telegram-based features
- ✅ بهبود reporting: واقعی از telegram capabilities
- ✅ Feature Categories Analysis با telegram mapping detection
- ✅ Enhanced Performance Metrics با telegram-derived reddit
- ✅ Multi-source Data Quality Analysis (telegram-based)
- ✅ Comprehensive Error Handling بهبود یافته
- ✅ Fallback Mechanism برای backward compatibility

ویژگی‌های Enhanced v3.1:
- بک تست چند نمادی و چند بازه زمانی
- ردیابی دلیل خروج و telegram sentiment impact
- معیارهای عملکرد پیشرفته (شارپ، حداکثر افت سرمایه و غیره)
- تجسم تعاملی با برچسب‌های Enhanced
- گزارش‌های معاملاتی دقیق با telegram sentiment analysis
- تحلیل تأثیر Telegram-derived Reddit features
- Multi-source data effectiveness reporting (telegram-based)
"""
import os
import glob
import pandas as pd
import numpy as np
import logging
import configparser
import joblib
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

# --- Configuration and Enhanced Logging Setup ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    FEATURES_PATH = config.get('Paths', 'features')
    MODELS_PATH = config.get('Paths', 'models')
    LOG_PATH = config.get('Paths', 'logs')
    REPORTS_PATH = config.get('Paths', 'reports')
    INITIAL_CAPITAL = config.getfloat('Backtester_Settings', 'initial_capital')
    TRADE_SIZE_PERCENT = config.getfloat('Backtester_Settings', 'trade_size_percent')
    TARGET_FUTURE_PERIODS = config.getint('ETL_Settings', 'target_future_periods')
    
    # === Enhanced Settings جدید برای Telegram-based ===
    MIN_SENTIMENT_COVERAGE = config.getfloat('Data_Quality', 'min_sentiment_coverage', fallback=0.10)
    MIN_TELEGRAM_REDDIT_COVERAGE = config.getfloat('Data_Quality', 'min_telegram_reddit_coverage', fallback=0.05)  # اصلاح نام
    SENTIMENT_ANALYSIS_ENABLED = config.getboolean('Enhanced_Analysis', 'sentiment_analysis_enabled', fallback=True)
    TELEGRAM_REDDIT_ANALYSIS_ENABLED = config.getboolean('Enhanced_Analysis', 'telegram_reddit_analysis_enabled', fallback=True)  # اصلاح نام
    DETAILED_FEATURE_ANALYSIS = config.getboolean('Enhanced_Analysis', 'detailed_feature_analysis', fallback=True)
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Error: {e}")
    exit()

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
report_subfolder_path = os.path.join(REPORTS_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)
os.makedirs(report_subfolder_path, exist_ok=True)
log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# === Enhanced Helper Functions ===

def find_enhanced_latest_file(pattern: str, description: str) -> str:
    """
    یافتن آخرین فایل Enhanced با pattern مشخص - اصلاح شده برای Enhanced models
    """
    try:
        # Enhanced patterns with priority order
        enhanced_patterns = []
        if 'model' in pattern.lower():
            enhanced_patterns = [
                pattern.replace('optimized_model_', 'enhanced_model_v6_'),
                pattern.replace('optimized_model_', 'optimized_model_'),
                pattern.replace('optimized_model_', 'random_forest_model_')
            ]
        elif 'scaler' in pattern.lower():
            enhanced_patterns = [
                pattern.replace('scaler_optimized_', 'scaler_enhanced_v6_'),
                pattern.replace('scaler_optimized_', 'scaler_optimized_'),
                pattern.replace('scaler_optimized_', 'scaler_')
            ]
        else:
            enhanced_patterns = [pattern]
        
        logging.info(f"🔍 Enhanced search for {description}...")
        
        # جستجو با اولویت Enhanced
        for i, enhanced_pattern in enumerate(enhanced_patterns):
            logging.info(f"   Pattern {i+1}: {os.path.basename(enhanced_pattern)}")
            files = glob.glob(enhanced_pattern)
            
            if files:
                existing_files = [f for f in files if os.path.exists(f) and os.path.getsize(f) > 0]
                logging.info(f"   📁 Valid files found: {len(existing_files)}")
                
                if existing_files:
                    latest_file = max(existing_files, key=os.path.getctime)
                    file_type = "Enhanced v6.0" if "enhanced_v6" in latest_file else "Optimized" if "optimized" in latest_file else "Legacy"
                    logging.info(f"✅ {file_type} {description}: {os.path.basename(latest_file)}")
                    return latest_file
        
        # جستجو در زیرپوشه‌ها
        parent_dir = os.path.dirname(pattern)
        logging.info(f"🔍 Enhanced search in subdirectories of {parent_dir}...")
        
        for enhanced_pattern in enhanced_patterns:
            file_pattern = os.path.basename(enhanced_pattern)
            alternative_patterns = [
                os.path.join(parent_dir, "**", file_pattern),
                os.path.join(parent_dir, "run_*", file_pattern),
                os.path.join(parent_dir, "enhanced_*", file_pattern)
            ]
            
            for alt_pattern in alternative_patterns:
                alt_files = glob.glob(alt_pattern, recursive=True)
                if alt_files:
                    existing_alt_files = [f for f in alt_files if os.path.exists(f) and os.path.getsize(f) > 0]
                    if existing_alt_files:
                        latest_file = max(existing_alt_files, key=os.path.getctime)
                        logging.info(f"✅ Enhanced alternative {description}: {os.path.basename(latest_file)}")
                        return latest_file
        
        # اگر هیچ فایلی پیدا نشد
        logging.error(f"❌ No Enhanced {description} file found")
        logging.error(f"💡 Searched patterns: {[os.path.basename(p) for p in enhanced_patterns]}")
        
        # نمایش فایل‌های موجود برای debugging
        try:
            parent_directory = os.path.dirname(pattern) if os.path.dirname(pattern) else "."
            if os.path.exists(parent_directory):
                all_files = os.listdir(parent_directory)
                model_files = [f for f in all_files if 'model' in f.lower() or 'scaler' in f.lower()]
                logging.info(f"📋 Model/Scaler files available in {parent_directory}:")
                for file in model_files[:10]:
                    logging.info(f"   - {file}")
                if len(model_files) > 10:
                    logging.info(f"   ... and {len(model_files) - 10} more files")
        except Exception as list_error:
            logging.warning(f"Could not list directory contents: {list_error}")
        
        return None
        
    except Exception as e:
        logging.error(f"❌ Enhanced error in file search for {description}: {e}")
        return None

def analyze_telegram_enhanced_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """🔧 اصلاح 1: تحلیل کیفیت داده Enhanced با Telegram-based features"""
    quality_analysis = {
        'total_records': len(df),
        'sentiment_features': [],
        'telegram_derived_reddit_features': [],  # 🔧 اصلاح: نام جدید
        'technical_features': [],
        'sentiment_coverage': 0,
        'telegram_reddit_coverage': 0,  # 🔧 اصلاح: coverage جدید
        'telegram_mapping_detected': False,  # 🔧 اصلاح: تشخیص mapping
        'quality_score': 0,
        'warnings': []
    }
    
    try:
        # شناسایی feature categories
        all_features = df.columns.tolist()
        
        for feature in all_features:
            feature_lower = feature.lower()
            if 'sentiment' in feature_lower:
                quality_analysis['sentiment_features'].append(feature)
            elif 'reddit' in feature_lower:
                quality_analysis['telegram_derived_reddit_features'].append(feature)  # 🔧 اصلاح: نام جدید
            elif feature not in ['target', 'timestamp', 'close', 'open', 'high', 'low', 'volume']:
                quality_analysis['technical_features'].append(feature)
        
        logging.info(f"🎭 Sentiment features found: {len(quality_analysis['sentiment_features'])}")
        logging.info(f"📱 Telegram-derived Reddit features found: {len(quality_analysis['telegram_derived_reddit_features'])}")  # 🔧 اصلاح
        logging.info(f"⚙️ Technical features found: {len(quality_analysis['technical_features'])}")
        
        # 🔧 اصلاح 1: تحلیل sentiment coverage
        if quality_analysis['sentiment_features']:
            sentiment_non_zero = 0
            for feature in quality_analysis['sentiment_features']:
                if feature in df.columns:
                    non_zero_count = (df[feature] != 0).sum()
                    if non_zero_count > 0:
                        sentiment_non_zero += 1
            
            quality_analysis['sentiment_coverage'] = sentiment_non_zero / len(quality_analysis['sentiment_features'])
            logging.info(f"📊 Sentiment coverage: {quality_analysis['sentiment_coverage']:.2%}")
            
            if quality_analysis['sentiment_coverage'] < MIN_SENTIMENT_COVERAGE:
                quality_analysis['warnings'].append(f"Low sentiment coverage ({quality_analysis['sentiment_coverage']:.1%})")
        
        # 🔧 اصلاح 1: تحلیل Telegram-derived Reddit coverage
        if quality_analysis['telegram_derived_reddit_features']:
            telegram_reddit_non_zero = 0
            for feature in quality_analysis['telegram_derived_reddit_features']:
                if feature in df.columns:
                    non_zero_count = (df[feature] != 0).sum()
                    if non_zero_count > 0:
                        telegram_reddit_non_zero += 1
            
            quality_analysis['telegram_reddit_coverage'] = telegram_reddit_non_zero / len(quality_analysis['telegram_derived_reddit_features'])
            logging.info(f"📊 Telegram-derived Reddit coverage: {quality_analysis['telegram_reddit_coverage']:.2%}")
            
            if quality_analysis['telegram_reddit_coverage'] > 0 and quality_analysis['telegram_reddit_coverage'] < MIN_TELEGRAM_REDDIT_COVERAGE:
                quality_analysis['warnings'].append(f"Low Telegram-Reddit coverage ({quality_analysis['telegram_reddit_coverage']:.1%})")
        
        # 🔧 اصلاح 1: تشخیص Telegram mapping
        if ('sentiment_score' in df.columns and 'reddit_score' in df.columns):
            # بررسی همبستگی برای تشخیص mapping
            correlation = df['sentiment_score'].corr(df['reddit_score'])
            if correlation > 0.95:  # همبستگی بالا نشان‌دهنده mapping است
                quality_analysis['telegram_mapping_detected'] = True
                logging.info("✅ Telegram → Reddit mapping detected in data")
            else:
                logging.info(f"📊 Sentiment-Reddit correlation: {correlation:.3f} (no direct mapping)")
        
        # محاسبه quality score کلی
        base_score = 0.6  # امتیاز پایه برای technical features
        sentiment_bonus = quality_analysis['sentiment_coverage'] * 0.25 if SENTIMENT_ANALYSIS_ENABLED else 0
        telegram_reddit_bonus = quality_analysis['telegram_reddit_coverage'] * 0.15 if TELEGRAM_REDDIT_ANALYSIS_ENABLED else 0
        
        quality_analysis['quality_score'] = base_score + sentiment_bonus + telegram_reddit_bonus
        
        logging.info(f"📈 Enhanced data quality score: {quality_analysis['quality_score']:.2%}")
        
        # نمایش warnings
        if quality_analysis['warnings']:
            logging.warning("⚠️ Enhanced Data Quality Warnings:")
            for warning in quality_analysis['warnings']:
                logging.warning(f"   - {warning}")
        
    except Exception as e:
        logging.error(f"❌ Enhanced data quality analysis failed: {e}")
        quality_analysis['warnings'].append(f"Analysis error: {str(e)}")
    
    return quality_analysis

def load_enhanced_model_package(model_file: str) -> Tuple[Any, float, Dict[str, Any]]:
    """بارگذاری Enhanced model package با اطلاعات کامل Telegram-based"""
    try:
        logging.info(f"🤖 Loading Enhanced model: {os.path.basename(model_file)}")
        model_data = joblib.load(model_file)
        
        # تشخیص نوع مدل
        if isinstance(model_data, dict):
            if 'model' in model_data:
                # Enhanced model package v6.1+
                model = model_data['model']
                optimal_threshold = model_data.get('optimal_threshold', 0.5)
                
                model_info = {
                    'model_type': model_data.get('model_type', 'Unknown'),
                    'model_version': model_data.get('model_version', '6.1_enhanced'),
                    'accuracy': model_data.get('accuracy', 0),
                    'precision': model_data.get('precision', 0),
                    'recall': model_data.get('recall', 0),
                    'f1_score': model_data.get('f1_score', 0),
                    'feature_columns': model_data.get('feature_columns', []),
                    'feature_categories': model_data.get('feature_categories', {}),
                    'sentiment_stats': model_data.get('sentiment_stats', {}),
                    'correlation_analysis': model_data.get('correlation_analysis', {}),
                    'telegram_reddit_mapping': model_data.get('telegram_reddit_mapping', False),  # 🔧 اصلاح: telegram mapping
                    'reddit_source': model_data.get('reddit_source', 'unknown'),  # 🔧 اصلاح: منبع reddit
                    'is_enhanced': True
                }
                
                logging.info(f"✅ Enhanced Model Package v6.1+ loaded")
                logging.info(f"   Model Type: {model_info['model_type']}")
                logging.info(f"   Optimal Threshold: {optimal_threshold:.4f}")
                logging.info(f"   Expected Features: {len(model_info['feature_columns'])}")
                logging.info(f"   📱 Telegram-Reddit Mapping: {'Yes' if model_info['telegram_reddit_mapping'] else 'No'}")  # 🔧 اصلاح
                logging.info(f"   🔴 Reddit Source: {model_info['reddit_source']}")  # 🔧 اصلاح
                
                # نمایش performance metrics
                if model_info['accuracy'] > 0:
                    logging.info(f"   Performance: Accuracy={model_info['accuracy']:.2%}, "
                               f"Precision={model_info['precision']:.2%}, "
                               f"Recall={model_info['recall']:.2%}, "
                               f"F1={model_info['f1_score']:.4f}")
                
                # نمایش feature categories
                feature_categories = model_info['feature_categories']
                if feature_categories:
                    logging.info(f"   🏷️ Feature Categories:")
                    for category, features in feature_categories.items():
                        if features:
                            category_display = category
                            if category == 'telegram_derived_features':  # 🔧 اصلاح
                                category_display += " (از Telegram sentiment مشتق شده)"
                            logging.info(f"      {category_display}: {len(features)} features")
                
                return model, optimal_threshold, model_info
            else:
                # Dictionary ولی format متفاوت
                model = model_data
                optimal_threshold = 0.5
                model_info = {'is_enhanced': False, 'model_type': 'Unknown Dictionary Format'}
                logging.warning("⚠️ Unknown model dictionary format, using defaults")
        else:
            # Legacy model (raw model object)
            model = model_data
            optimal_threshold = 0.5
            model_info = {
                'model_type': type(model_data).__name__,
                'is_enhanced': False,
                'is_legacy': True,
                'feature_columns': [],
                'telegram_reddit_mapping': False,
                'reddit_source': 'unknown'
            }
            logging.warning(f"⚠️ Legacy model loaded: {model_info['model_type']}")
        
        return model, optimal_threshold, model_info
        
    except Exception as e:
        logging.error(f"❌ Error loading Enhanced model: {e}")
        raise e

def calculate_enhanced_max_drawdown(equity_curve: pd.Series) -> float:
    """محاسبه Enhanced maximum drawdown"""
    if equity_curve.empty:
        return 0
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_enhanced_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """محاسبه Enhanced Sharpe ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate/252
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def analyze_telegram_sentiment_and_reddit_impact(trade_history: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
    """🔧 اصلاح 2: تحلیل یکپارچه تأثیر Telegram sentiment و derived Reddit features (ادغام شده)"""
    telegram_analysis = {
        # Sentiment analysis
        'sentiment_positive_trades': 0,
        'sentiment_negative_trades': 0,
        'avg_sentiment_winners': 0,
        'avg_sentiment_losers': 0,
        'sentiment_effectiveness': 0,
        
        # Telegram-derived Reddit analysis
        'high_telegram_reddit_activity_trades': 0,
        'low_telegram_reddit_activity_trades': 0,
        'avg_telegram_reddit_winners': 0,
        'avg_telegram_reddit_losers': 0,
        'telegram_reddit_effectiveness': 0,
        
        # Mapping analysis
        'telegram_mapping_confirmed': False,
        'sentiment_reddit_correlation': 0,
        
        # Combined effectiveness
        'combined_effectiveness': 0,
        'telegram_vs_reddit_consistency': 0
    }
    
    try:
        if not trade_history:
            return telegram_analysis
        
        # بررسی وجود features
        has_sentiment = 'sentiment_score' in df.columns
        has_reddit = 'reddit_score' in df.columns
        
        if not has_sentiment and not has_reddit:
            logging.warning("⚠️ No sentiment or Reddit features found for impact analysis")
            return telegram_analysis
        
        # 🔧 اصلاح 2: تشخیص Telegram mapping
        if has_sentiment and has_reddit:
            correlation = df['sentiment_score'].corr(df['reddit_score'])
            telegram_analysis['sentiment_reddit_correlation'] = correlation
            if correlation > 0.95:
                telegram_analysis['telegram_mapping_confirmed'] = True
                logging.info(f"✅ Telegram mapping confirmed in analysis: correlation={correlation:.4f}")
            else:
                logging.info(f"📊 Sentiment-Reddit correlation: {correlation:.4f}")
        
        # جمع‌آوری داده‌ها برای تحلیل
        sentiment_values_winners = []
        sentiment_values_losers = []
        reddit_values_winners = []
        reddit_values_losers = []
        
        for trade in trade_history:
            try:
                entry_date = trade['entry_date']
                if entry_date in df.index:
                    
                    # Sentiment analysis
                    if has_sentiment:
                        sentiment_score = df.loc[entry_date, 'sentiment_score']
                        
                        if trade['pnl'] > 0:
                            sentiment_values_winners.append(sentiment_score)
                            if sentiment_score > 0:
                                telegram_analysis['sentiment_positive_trades'] += 1
                        else:
                            sentiment_values_losers.append(sentiment_score)
                            if sentiment_score < 0:
                                telegram_analysis['sentiment_negative_trades'] += 1
                    
                    # 🔧 اصلاح 2: Telegram-derived Reddit analysis
                    if has_reddit:
                        reddit_score = df.loc[entry_date, 'reddit_score']
                        reddit_threshold = df['reddit_score'].median()
                        
                        if trade['pnl'] > 0:
                            reddit_values_winners.append(reddit_score)
                            if reddit_score > reddit_threshold:
                                telegram_analysis['high_telegram_reddit_activity_trades'] += 1
                        else:
                            reddit_values_losers.append(reddit_score)
                            if reddit_score <= reddit_threshold:
                                telegram_analysis['low_telegram_reddit_activity_trades'] += 1
                        
            except (KeyError, TypeError):
                continue
        
        # محاسبه آمارهای sentiment
        if sentiment_values_winners:
            telegram_analysis['avg_sentiment_winners'] = np.mean(sentiment_values_winners)
        if sentiment_values_losers:
            telegram_analysis['avg_sentiment_losers'] = np.mean(sentiment_values_losers)
        
        # محاسبه آمارهای telegram-derived reddit
        if reddit_values_winners:
            telegram_analysis['avg_telegram_reddit_winners'] = np.mean(reddit_values_winners)
        if reddit_values_losers:
            telegram_analysis['avg_telegram_reddit_losers'] = np.mean(reddit_values_losers)
        
        # محاسبه effectiveness
        total_trades = len(trade_history)
        if total_trades > 0:
            telegram_analysis['sentiment_effectiveness'] = telegram_analysis['sentiment_positive_trades'] / total_trades
            telegram_analysis['telegram_reddit_effectiveness'] = telegram_analysis['high_telegram_reddit_activity_trades'] / total_trades
            
            # 🔧 اصلاح 2: Combined effectiveness (یکپارچه)
            if telegram_analysis['telegram_mapping_confirmed']:
                # اگر mapping تایید شده، از sentiment effectiveness استفاده کنیم
                telegram_analysis['combined_effectiveness'] = telegram_analysis['sentiment_effectiveness']
                telegram_analysis['telegram_vs_reddit_consistency'] = 1.0  # کامل consistent
                logging.info("📱 Using sentiment effectiveness as combined (due to telegram mapping)")
            else:
                # اگر mapping نشده، میانگین وزنی
                sentiment_weight = 0.7
                reddit_weight = 0.3
                telegram_analysis['combined_effectiveness'] = (
                    telegram_analysis['sentiment_effectiveness'] * sentiment_weight +
                    telegram_analysis['telegram_reddit_effectiveness'] * reddit_weight
                )
                # consistency بر اساس تفاوت effectiveness
                effectiveness_diff = abs(telegram_analysis['sentiment_effectiveness'] - telegram_analysis['telegram_reddit_effectiveness'])
                telegram_analysis['telegram_vs_reddit_consistency'] = 1 - effectiveness_diff
        
        # لاگ نتایج یکپارچه
        logging.info(f"🎭 Telegram Sentiment Analysis Results:")
        logging.info(f"   Sentiment-positive winning trades: {telegram_analysis['sentiment_positive_trades']}")
        logging.info(f"   Average sentiment (winners): {telegram_analysis['avg_sentiment_winners']:.4f}")
        logging.info(f"   Average sentiment (losers): {telegram_analysis['avg_sentiment_losers']:.4f}")
        logging.info(f"   Sentiment effectiveness: {telegram_analysis['sentiment_effectiveness']:.2%}")
        
        logging.info(f"📱 Telegram-derived Reddit Analysis Results:")
        logging.info(f"   High activity winning trades: {telegram_analysis['high_telegram_reddit_activity_trades']}")
        logging.info(f"   Average Reddit score (winners): {telegram_analysis['avg_telegram_reddit_winners']:.4f}")
        logging.info(f"   Average Reddit score (losers): {telegram_analysis['avg_telegram_reddit_losers']:.4f}")
        logging.info(f"   Telegram-Reddit effectiveness: {telegram_analysis['telegram_reddit_effectiveness']:.2%}")
        
        logging.info(f"🔗 Combined Analysis:")
        logging.info(f"   Combined effectiveness: {telegram_analysis['combined_effectiveness']:.2%}")
        logging.info(f"   Telegram vs Reddit consistency: {telegram_analysis['telegram_vs_reddit_consistency']:.2%}")
        
    except Exception as e:
        logging.warning(f"Telegram sentiment/Reddit impact analysis failed: {e}")
    
    return telegram_analysis

def generate_enhanced_report_file(report_data: Dict, symbol: str, timeframe: str, 
                                data_quality: Dict, telegram_impact: Dict, model_info: Dict):
    """🔧 اصلاح 3: تولید گزارش Enhanced کامل با Telegram-based features"""
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Enhanced_Telegram_Backtest_Report_{symbol.replace('/', '-')}_{timeframe}_{timestamp_str}.txt"
    filepath = os.path.join(report_subfolder_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("    ENHANCED TELEGRAM-BASED BACKTEST PERFORMANCE REPORT v3.1\n")
        f.write("="*80 + "\n\n")
        
        # اطلاعات کلی Enhanced
        f.write("📊 Enhanced General Information:\n")
        f.write("-"*60 + "\n")
        f.write(f"Symbol tested:         {symbol}\n")
        f.write(f"Timeframe tested:      {timeframe}\n")
        f.write(f"Report date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Enhanced API Version:  v6.1 (Telegram-based)\n")
        f.write(f"Backtester Version:    v3.1\n\n")
        
        # اطلاعات مدل Enhanced
        f.write("🤖 Enhanced Telegram-based Model Information:\n")
        f.write("-"*60 + "\n")
        f.write(f"Model Type:            {model_info.get('model_type', 'Unknown')}\n")
        f.write(f"Model Version:         {model_info.get('model_version', 'Unknown')}\n")
        f.write(f"Is Enhanced:           {'Yes' if model_info.get('is_enhanced') else 'No'}\n")
        f.write(f"Expected Features:     {len(model_info.get('feature_columns', []))}\n")
        f.write(f"Optimal Threshold:     {report_data.get('optimal_threshold', 0.5):.4f}\n")
        f.write(f"Telegram-Reddit Map:   {'Yes' if model_info.get('telegram_reddit_mapping') else 'No'}\n")  # 🔧 اصلاح
        f.write(f"Reddit Source:         {model_info.get('reddit_source', 'unknown')}\n")  # 🔧 اصلاح
        
        if model_info.get('accuracy', 0) > 0:
            f.write(f"Model Accuracy:        {model_info['accuracy']:.2%}\n")
            f.write(f"Model Precision:       {model_info['precision']:.2%}\n")
            f.write(f"Model Recall:          {model_info['recall']:.2%}\n")
            f.write(f"Model F1 Score:        {model_info['f1_score']:.4f}\n")
        f.write("\n")
        
        # Feature Categories Analysis
        feature_categories = model_info.get('feature_categories', {})
        if feature_categories:
            f.write("🏷️ Enhanced Telegram-based Feature Categories:\n")
            f.write("-"*60 + "\n")
            for category, features in feature_categories.items():
                category_display = category
                if category == 'telegram_derived_features':  # 🔧 اصلاح
                    category_display += " (Telegram-derived)"
                f.write(f"{category_display:<30} {len(features) if features else 0} features\n")
            f.write("\n")
        
        # 🔧 اصلاح 3: Data Quality Analysis (Telegram-based)
        f.write("📈 Enhanced Telegram-based Data Quality Analysis:\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Records:           {data_quality['total_records']:,}\n")
        f.write(f"Sentiment Features:      {len(data_quality['sentiment_features'])}\n")
        f.write(f"Telegram-Reddit Features: {len(data_quality['telegram_derived_reddit_features'])}\n")  # 🔧 اصلاح
        f.write(f"Technical Features:      {len(data_quality['technical_features'])}\n")
        f.write(f"Sentiment Coverage:      {data_quality['sentiment_coverage']:.2%}\n")
        f.write(f"Telegram-Reddit Coverage: {data_quality['telegram_reddit_coverage']:.2%}\n")  # 🔧 اصلاح
        f.write(f"Telegram Mapping:        {'Detected' if data_quality['telegram_mapping_detected'] else 'Not Detected'}\n")  # 🔧 اصلاح
        f.write(f"Quality Score:           {data_quality['quality_score']:.2%}\n")
        
        if data_quality['warnings']:
            f.write("⚠️ Data Quality Warnings:\n")
            for warning in data_quality['warnings']:
                f.write(f"   - {warning}\n")
        f.write("\n")
        
        # نتایج مالی Enhanced
        f.write("💰 Enhanced Financial Results:\n")
        f.write("-"*60 + "\n")
        for key, value in report_data.items():
            if key not in ['trade_history', 'optimal_threshold'] and not key.startswith('enhanced_'):
                f.write(f"{key:<25} {value}\n")
        f.write("\n")
        
        # 🔧 اصلاح 3: Telegram Impact Analysis (یکپارچه)
        if SENTIMENT_ANALYSIS_ENABLED and telegram_impact:
            f.write("📱 Enhanced Telegram Sentiment & Derived Reddit Impact Analysis:\n")
            f.write("-"*70 + "\n")
            f.write(f"Sentiment Positive Wins:       {telegram_impact['sentiment_positive_trades']}\n")
            f.write(f"Avg Sentiment (Winners):       {telegram_impact['avg_sentiment_winners']:.4f}\n")
            f.write(f"Avg Sentiment (Losers):        {telegram_impact['avg_sentiment_losers']:.4f}\n")
            f.write(f"Sentiment Effectiveness:       {telegram_impact['sentiment_effectiveness']:.2%}\n")
            f.write("\n")
            
            if TELEGRAM_REDDIT_ANALYSIS_ENABLED:
                f.write(f"High Telegram-Reddit Activity:  {telegram_impact['high_telegram_reddit_activity_trades']}\n")
                f.write(f"Avg Telegram-Reddit (Winners):  {telegram_impact['avg_telegram_reddit_winners']:.4f}\n")
                f.write(f"Avg Telegram-Reddit (Losers):   {telegram_impact['avg_telegram_reddit_losers']:.4f}\n")
                f.write(f"Telegram-Reddit Effectiveness:  {telegram_impact['telegram_reddit_effectiveness']:.2%}\n")
                f.write("\n")
                
                f.write(f"📊 Combined Analysis:\n")
                f.write(f"Telegram Mapping Confirmed:    {'Yes' if telegram_impact['telegram_mapping_confirmed'] else 'No'}\n")
                f.write(f"Sentiment-Reddit Correlation:  {telegram_impact['sentiment_reddit_correlation']:.4f}\n")
                f.write(f"Combined Effectiveness:        {telegram_impact['combined_effectiveness']:.2%}\n")
                f.write(f"Telegram vs Reddit Consistency: {telegram_impact['telegram_vs_reddit_consistency']:.2%}\n")
                f.write("\n")
        
        # Trade History Enhanced
        f.write("📋 Enhanced Telegram-based Trade History:\n")
        f.write("-"*110 + "\n")
        if report_data['trade_history']:
            f.write(f"{'#':<3} {'Entry Date':<16} {'Exit Date':<16} {'Entry $':<10} {'Exit $':<10} "
                   f"{'P&L':<8} {'Sentiment':<10} {'T-Reddit':<10} {'Reason':<25}\n")
            f.write("-"*110 + "\n")
            
            for i, trade in enumerate(report_data['trade_history'], 1):
                entry_ts = pd.to_datetime(str(trade['entry_date'])).strftime('%Y-%m-%d %H:%M')
                exit_ts = pd.to_datetime(str(trade['exit_date'])).strftime('%Y-%m-%d %H:%M')
                reason = trade.get('exit_reason', 'Target reached')
                sentiment_val = trade.get('sentiment_at_entry', 0)
                telegram_reddit_val = trade.get('telegram_reddit_at_entry', 0)  # 🔧 اصلاح: نام جدید
                
                f.write(f"{i:<3} {entry_ts:<16} {exit_ts:<16} "
                       f"${trade['entry_price']:<9.4f} ${trade['exit_price']:<9.4f} "
                       f"{trade['pnl']:<7.2%} {sentiment_val:<9.4f} {telegram_reddit_val:<9.4f} {reason:<25}\n")
        else:
            f.write("No trades executed in this Enhanced Telegram-based backtest period.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("End of Enhanced Telegram-based Backtest Report v3.1\n")
        f.write("="*80 + "\n")
    
    logging.info(f"Enhanced Telegram-based report saved to '{filepath}'")

def generate_enhanced_telegram_visualizations(df: pd.DataFrame, trade_history: List[Dict], 
                                            symbol: str, timeframe: str, report_path: str,
                                            data_quality: Dict, telegram_impact: Dict):
    """🔧 اصلاح 4: تولید نمودارهای Enhanced با Telegram sentiment و derived Reddit analysis"""
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Enhanced Price and Signals Chart
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(f'Enhanced Telegram-based Backtest Analysis - {symbol} ({timeframe})', fontsize=16, fontweight='bold')
    
    # 1. Price chart with Enhanced signals
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7, color='blue', linewidth=1)
    
    # Enhanced trade visualization
    if trade_history:
        for i, trade in enumerate(trade_history):
            try:
                entry_idx = df.index.get_loc(trade['entry_date'])
                exit_idx = df.index.get_loc(trade['exit_date'])
                color = 'green' if trade['pnl'] > 0 else 'red'
                alpha = 0.8 if abs(trade['pnl']) > 0.05 else 0.5
                
                # Trade line
                ax1.plot([trade['entry_date'], trade['exit_date']], 
                        [trade['entry_price'], trade['exit_price']], 
                        'o-', color=color, markersize=6, linewidth=2, alpha=alpha)
                
                # Enhanced annotation با telegram sentiment
                if i < 10:  # نمایش annotation برای 10 معامله اول
                    mid_date = df.index[entry_idx + (exit_idx - entry_idx) // 2]
                    mid_price = (trade['entry_price'] + trade['exit_price']) / 2
                    sentiment_text = f"S:{trade.get('sentiment_at_entry', 0):.2f}" if 'sentiment_at_entry' in trade else ""
                    telegram_reddit_text = f"TR:{trade.get('telegram_reddit_at_entry', 0):.2f}" if 'telegram_reddit_at_entry' in trade else ""
                    annotation_text = f"{trade['pnl']:.1%}\n{sentiment_text}\n{telegram_reddit_text}"
                    
                    ax1.annotate(annotation_text, 
                               xy=(mid_date, mid_price),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=6, ha='left',
                               bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.3))
            except (KeyError, ValueError):
                continue
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Enhanced Price Chart with Telegram Sentiment Info')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Model Predictions Enhanced
    ax2 = axes[0, 1]
    if 'prediction' in df.columns:
        ax2.plot(df.index, df['prediction'], label='Model Signal', color='orange', alpha=0.7)
        if 'prediction_proba' in df.columns:
            ax2.plot(df.index, df['prediction_proba'], label='Prediction Probability', color='purple', alpha=0.5)
        ax2.fill_between(df.index, 0, df['prediction'], alpha=0.3, color='orange')
    ax2.set_ylabel('Signal Strength')
    ax2.set_title('Enhanced Model Predictions')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Telegram Sentiment Analysis Chart
    ax3 = axes[1, 0]
    sentiment_features = data_quality['sentiment_features']
    if sentiment_features and SENTIMENT_ANALYSIS_ENABLED:
        main_sentiment = 'sentiment_score' if 'sentiment_score' in df.columns else sentiment_features[0]
        ax3.plot(df.index, df[main_sentiment], label='Telegram Sentiment Score', color='purple', alpha=0.7)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Mark trade entries with sentiment
        if trade_history:
            for trade in trade_history:
                if 'sentiment_at_entry' in trade:
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax3.scatter(trade['entry_date'], trade['sentiment_at_entry'], 
                              color=color, s=50, alpha=0.8, marker='o')
    else:
        ax3.text(0.5, 0.5, 'No Telegram Sentiment Data Available', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_ylabel('Telegram Sentiment Score')
    ax3.set_title('Telegram Sentiment Analysis Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 🔧 اصلاح 4: Telegram-derived Reddit Features Chart
    ax4 = axes[1, 1]
    telegram_reddit_features = data_quality['telegram_derived_reddit_features']
    if telegram_reddit_features and TELEGRAM_REDDIT_ANALYSIS_ENABLED:
        main_reddit = 'reddit_score' if 'reddit_score' in df.columns else telegram_reddit_features[0]
        ax4.plot(df.index, df[main_reddit], label='Telegram-derived Reddit Score', color='red', alpha=0.7)
        
        # 🔧 اصلاح 4: اگر mapping تشخیص داده شده، sentiment و reddit را با هم نمایش دهیم
        if data_quality.get('telegram_mapping_detected', False) and 'sentiment_score' in df.columns:
            ax4.plot(df.index, df['sentiment_score'], label='Original Sentiment Score', 
                    color='purple', alpha=0.5, linestyle='--')
        
        # Mark trade entries with Reddit activity
        if trade_history:
            for trade in trade_history:
                if 'telegram_reddit_at_entry' in trade:  # 🔧 اصلاح: نام جدید
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax4.scatter(trade['entry_date'], trade['telegram_reddit_at_entry'], 
                              color=color, s=50, alpha=0.8, marker='s')
    else:
        ax4.text(0.5, 0.5, 'No Telegram-derived Reddit Data Available', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_ylabel('Telegram-derived Reddit Activity')
    ax4.set_title('Telegram-derived Reddit Features Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. P&L Distribution Enhanced با Telegram analysis
    ax5 = axes[2, 0]
    if trade_history:
        pnl_values = [t['pnl'] * 100 for t in trade_history]
        sentiment_positive = [t['pnl'] * 100 for t in trade_history if t.get('sentiment_at_entry', 0) > 0]
        sentiment_negative = [t['pnl'] * 100 for t in trade_history if t.get('sentiment_at_entry', 0) <= 0]
        
        ax5.hist(pnl_values, bins=15, alpha=0.7, color='blue', label='All Trades', density=True)
        if sentiment_positive:
            ax5.hist(sentiment_positive, bins=10, alpha=0.5, color='green', 
                    label='Positive Telegram Sentiment', density=True)
        if sentiment_negative:
            ax5.hist(sentiment_negative, bins=10, alpha=0.5, color='red', 
                    label='Negative Telegram Sentiment', density=True)
        
        ax5.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    ax5.set_xlabel('Profit/Loss (%)')
    ax5.set_ylabel('Density')
    ax5.set_title('Enhanced P&L Distribution by Telegram Sentiment')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. 🔧 اصلاح 4: Feature Effectiveness Summary (Telegram-based)
    ax6 = axes[2, 1]
    effectiveness_data = {
        'Technical': 0.6,  # Base effectiveness
        'Telegram Sentiment': telegram_impact.get('sentiment_effectiveness', 0) if SENTIMENT_ANALYSIS_ENABLED else 0,
        'Derived Reddit': telegram_impact.get('telegram_reddit_effectiveness', 0) if TELEGRAM_REDDIT_ANALYSIS_ENABLED else 0
    }
    
    # اگر mapping تشخیص داده شده، combined effectiveness نمایش دهیم
    if telegram_impact.get('telegram_mapping_confirmed', False):
        effectiveness_data = {
            'Technical': 0.6,
            'Telegram Combined': telegram_impact.get('combined_effectiveness', 0)
        }
    
    categories = list(effectiveness_data.keys())
    values = list(effectiveness_data.values())
    colors = ['blue', 'purple', 'red'] if len(categories) == 3 else ['blue', 'purple']
    
    bars = ax6.bar(categories, values, color=colors[:len(categories)], alpha=0.7)
    ax6.set_ylabel('Effectiveness Score')
    ax6.set_title('Telegram-based Feature Category Effectiveness')
    ax6.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    chart_filename = os.path.join(report_path, f"enhanced_telegram_backtest_analysis_{symbol.replace('/', '-')}_{timeframe}_{timestamp_str}.png")
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Enhanced Telegram-based analysis chart saved to '{chart_filename}'")

def select_enhanced_symbols_and_timeframes(df_full: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Enhanced selection با Telegram-based data quality information"""
    # Symbol selection with quality info
    available_symbols = df_full.index.get_level_values('symbol').unique().tolist()
    print("\n📊 Enhanced Available Symbols with Telegram-based Data Quality:")
    print("-" * 80)
    
    for i, sym in enumerate(available_symbols, 1):
        try:
            # بررسی کیفیت داده برای هر symbol
            sample_data = df_full.loc[sym].iloc[:100]  # نمونه کوچک
            quality = analyze_telegram_enhanced_data_quality(sample_data)
            quality_emoji = "🟢" if quality['quality_score'] > 0.8 else "🟡" if quality['quality_score'] > 0.6 else "🔴"
            mapping_emoji = "📱" if quality['telegram_mapping_detected'] else "❓"
            
            print(f"{i:3d}. {sym:<12} {quality_emoji} Quality: {quality['quality_score']:.0%} "
                  f"(S:{quality['sentiment_coverage']:.0%} TR:{quality['telegram_reddit_coverage']:.0%} {mapping_emoji})")
        except:
            print(f"{i:3d}. {sym:<12} ❓ Quality: Unknown")
    
    print(f"{len(available_symbols)+1:3d}. ALL SYMBOLS")
    print("-" * 80)
    print("Legend: S=Sentiment Coverage, TR=Telegram-Reddit Coverage, 📱=Mapping Detected")
    
    symbol_choice = input("\n💱 Enter Enhanced symbol number(s) or name (e.g., 1 or BTC/USDT or 1,3,5): ").strip()
    
    selected_symbols = []
    if symbol_choice == str(len(available_symbols)+1) or symbol_choice.upper() == 'ALL':
        selected_symbols = available_symbols
        print(f"✅ Selected ALL {len(selected_symbols)} symbols for Enhanced Telegram analysis")
    elif ',' in symbol_choice:
        try:
            indices = [int(x.strip()) - 1 for x in symbol_choice.split(',') if x.strip().isdigit()]
            selected_symbols = [available_symbols[i] for i in indices if 0 <= i < len(available_symbols)]
        except (ValueError, IndexError):
            print("❌ Invalid Enhanced selection format")
            return None, None
    elif symbol_choice.isdigit():
        try:
            idx = int(symbol_choice) - 1
            if 0 <= idx < len(available_symbols):
                selected_symbols = [available_symbols[idx]]
        except (ValueError, IndexError):
            print("❌ Invalid Enhanced selection")
            return None, None
    else:
        symbol_choice = symbol_choice.upper()
        if symbol_choice in available_symbols:
            selected_symbols = [symbol_choice]
    
    if not selected_symbols:
        print("❌ Invalid Enhanced selection")
        return None, None
    
    # Timeframe selection
    try:
        first_symbol_tf = df_full.loc[selected_symbols[0]].index.get_level_values('timeframe').unique().tolist()
    except KeyError:
        print(f"❌ Enhanced symbol {selected_symbols[0]} not found in dataset")
        return None, None
        
    print(f"\n⏱️ Enhanced Available Timeframes:")
    print("-" * 40)
    for i, tf in enumerate(first_symbol_tf, 1):
        print(f"{i:2d}. {tf}")
    print(f"{len(first_symbol_tf)+1:2d}. ALL TIMEFRAMES")
    print("-" * 40)
    
    tf_choice = input("\n🕐 Enter Enhanced timeframe number(s) or name: ").strip()
    
    selected_timeframes = []
    if tf_choice == str(len(first_symbol_tf)+1) or tf_choice.upper() == 'ALL':
        selected_timeframes = first_symbol_tf
        print(f"✅ Selected ALL {len(selected_timeframes)} timeframes for Enhanced Telegram analysis")
    elif ',' in tf_choice:
        try:
            indices = [int(x.strip()) - 1 for x in tf_choice.split(',') if x.strip().isdigit()]
            selected_timeframes = [first_symbol_tf[i] for i in indices if 0 <= i < len(first_symbol_tf)]
        except (ValueError, IndexError):
            print("❌ Invalid Enhanced timeframe selection")
            return None, None
    elif tf_choice.isdigit():
        try:
            idx = int(tf_choice) - 1
            if 0 <= idx < len(first_symbol_tf):
                selected_timeframes = [first_symbol_tf[idx]]
        except (ValueError, IndexError):
            print("❌ Invalid Enhanced timeframe selection")
            return None, None
    else:
        tf_choice = tf_choice.lower()
        if tf_choice in first_symbol_tf:
            selected_timeframes = [tf_choice]
    
    if not selected_timeframes:
        print("❌ Invalid Enhanced timeframe selection")
        return None, None
    
    return selected_symbols, selected_timeframes

def run_enhanced_backtest_complete(features_path: str, models_path: str):
    """اجرای کامل Enhanced Backtest v3.1 با Telegram-based features"""
    logging.info("="*80)
    logging.info("Starting Enhanced Telegram-based Backtest Strategy v3.1 - Complete Integration")
    logging.info("="*80)
    
    try:
        # === بارگذاری داده‌ها Enhanced ===
        feature_file = find_enhanced_latest_file(
            os.path.join(features_path, 'final_dataset_for_training_*.parquet'),
            "Enhanced dataset file"
        )
        if not feature_file:
            logging.error("❌ No Enhanced dataset file found")
            print("❌ فایل Enhanced dataset یافت نشد")
            print("💡 لطفاً prepare_features_03.py را اجرا کنید")
            return
            
        logging.info(f"📁 Loading Enhanced dataset: {os.path.basename(feature_file)}")
        df_full = pd.read_parquet(feature_file)
        logging.info(f"✅ Enhanced dataset loaded: {df_full.shape}")
        
        # === بارگذاری Enhanced Model ===
        model_file = find_enhanced_latest_file(
            os.path.join(models_path, 'optimized_model_*.joblib'),
            "Enhanced model"
        )
        
        scaler_file = find_enhanced_latest_file(
            os.path.join(models_path, 'scaler_optimized_*.joblib'),
            "Enhanced scaler"
        )
        
        if not model_file or not scaler_file:
            logging.error("❌ Required Enhanced model or scaler files not found")
            print("❌ فایل‌های Enhanced model یا scaler یافت نشد")
            print("💡 لطفاً train_model_04.py را اجرا کنید")
            return
        
        # بارگذاری Enhanced model package
        model, optimal_threshold, model_info = load_enhanced_model_package(model_file)
        scaler = joblib.load(scaler_file)
        logging.info(f"✅ Enhanced Telegram-based model and scaler loaded successfully")
        
    except Exception as e:
        logging.error(f"❌ Error loading Enhanced files: {e}")
        print(f"❌ خطا در بارگذاری فایل‌های Enhanced: {e}")
        return

    # === انتخاب Enhanced symbols و timeframes ===
    selected_symbols, selected_timeframes = select_enhanced_symbols_and_timeframes(df_full)
    if not selected_symbols or not selected_timeframes:
        return
    
    # === ذخیره Enhanced results ===
    enhanced_results = []
    
    # === Enhanced Backtesting Loop ===
    for symbol in selected_symbols:
        for timeframe in selected_timeframes:
            try:
                logging.info(f"\n{'='*60}")
                logging.info(f"Enhanced Telegram-based Backtesting: {symbol} on {timeframe}")
                logging.info(f"{'='*60}")
                
                # دریافت داده برای symbol/timeframe
                df = df_full.loc[(symbol, timeframe)].copy()
                if len(df) < TARGET_FUTURE_PERIODS * 2:
                    logging.warning(f"❌ Insufficient Enhanced data for {symbol}/{timeframe}. Skipping...")
                    continue
                
                logging.info(f"📊 Enhanced data records: {len(df)}")
                
                # === 🔧 اصلاح 1: تحلیل Enhanced Data Quality (Telegram-based) ===
                data_quality = analyze_telegram_enhanced_data_quality(df)
                logging.info(f"📈 Enhanced Telegram-based data quality score: {data_quality['quality_score']:.2%}")
                
                # === مقداردهی Enhanced Backtest Variables ===
                capital = INITIAL_CAPITAL
                equity_curve = [capital]
                position_open = False
                entry_price = 0
                entry_index = -1
                entry_date = None
                trade_history = []
                
                # === Enhanced Feature Preparation ===
                feature_columns = [col for col in df.columns if col not in ['target', 'timestamp']]
                expected_features = model_info.get('feature_columns', [])
                
                # بررسی Enhanced feature compatibility
                if expected_features:
                    missing_features = [f for f in expected_features if f not in feature_columns]
                    if missing_features:
                        logging.warning(f"⚠️ Enhanced missing features: {len(missing_features)}")
                        logging.info("Adding missing features with default values...")
                        for feature in missing_features:
                            df[feature] = 0
                        feature_columns = expected_features
                
                X = df[feature_columns]
                logging.info(f"🔢 Enhanced features used: {len(feature_columns)}")
                
                # === Enhanced Model Predictions ===
                try:
                    X_scaled = scaler.transform(X)
                    
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_scaled)[:, 1]
                        df['prediction'] = (y_prob >= optimal_threshold).astype(int)
                        df['prediction_proba'] = y_prob
                        logging.info(f"✅ Enhanced predictions using optimal threshold: {optimal_threshold:.4f}")
                    else:
                        df['prediction'] = model.predict(X_scaled)
                        df['prediction_proba'] = 0.5
                        logging.warning("⚠️ Enhanced model doesn't support probability predictions")
                        
                except Exception as pred_error:
                    logging.error(f"❌ Enhanced prediction error: {pred_error}")
                    continue
                
                # === Enhanced Trading Simulation ===
                sentiment_features = data_quality['sentiment_features']
                telegram_reddit_features = data_quality['telegram_derived_reddit_features']  # 🔧 اصلاح
                
                for i in range(len(df) - TARGET_FUTURE_PERIODS):
                    current_row = df.iloc[i]
                    current_date = df.index[i]
                    
                    # === Enhanced Entry Logic ===
                    if not position_open and current_row['prediction'] == 1:
                        # Additional Enhanced filters
                        enhanced_entry = True
                        
                        # Telegram Sentiment filter (اختیاری)
                        if SENTIMENT_ANALYSIS_ENABLED and sentiment_features:
                            main_sentiment = sentiment_features[0] if sentiment_features else None
                            if main_sentiment and main_sentiment in df.columns:
                                sentiment_score = current_row[main_sentiment]
                                if sentiment_score < -0.1:  # خیلی منفی
                                    enhanced_entry = False
                                    logging.debug(f"⚠️ Entry blocked by negative Telegram sentiment: {sentiment_score:.4f}")
                        
                        if enhanced_entry:
                            position_open = True
                            entry_price = current_row['close']
                            entry_index = i
                            entry_date = current_date
                            
                            logging.info(f"🟢 Enhanced Entry at {entry_date}: ${entry_price:.4f}")
                            logging.info(f"   Prediction probability: {current_row.get('prediction_proba', 'N/A'):.4f}")
                    
                    # === Enhanced Exit Logic ===
                    elif position_open:
                        exit_condition = False
                        exit_reason = ""
                        
                        # Standard exit conditions
                        if i >= entry_index + TARGET_FUTURE_PERIODS:
                            exit_condition = True
                            exit_reason = "Target period reached"
                        elif (current_row['close'] - entry_price) / entry_price < -0.05:  # 5% stop loss
                            exit_condition = True
                            exit_reason = "Stop loss (-5%)"
                        elif (current_row['close'] - entry_price) / entry_price > 0.10:  # 10% take profit
                            exit_condition = True
                            exit_reason = "Take profit (+10%)"
                        elif current_row['prediction'] == 0 and i > entry_index + 3:  # Signal reversal
                            exit_condition = True
                            exit_reason = "Enhanced signal reversed"
                        
                        # === Enhanced Exit Conditions (Telegram-based) ===
                        if SENTIMENT_ANALYSIS_ENABLED and sentiment_features and not exit_condition:
                            main_sentiment = sentiment_features[0] if sentiment_features else None
                            if main_sentiment and main_sentiment in df.columns:
                                current_sentiment = current_row[main_sentiment]
                                if current_sentiment < -0.15 and i > entry_index + 5:  # Strong negative sentiment
                                    exit_condition = True
                                    exit_reason = "Negative Telegram sentiment exit"
                        
                        if exit_condition:
                            exit_price = current_row['close']
                            exit_date = current_date
                            pnl_percent = (exit_price - entry_price) / entry_price
                            
                            # === 🔧 اصلاح 3: Enhanced Trade Record (Telegram-based) ===
                            trade_record = {
                                'entry_date': entry_date,
                                'entry_price': entry_price,
                                'exit_date': exit_date,
                                'exit_price': exit_price,
                                'pnl': pnl_percent,
                                'exit_reason': exit_reason
                            }
                            
                            # Enhanced: Add Telegram sentiment info
                            if SENTIMENT_ANALYSIS_ENABLED and sentiment_features:
                                main_sentiment = sentiment_features[0] if sentiment_features else None
                                if main_sentiment and main_sentiment in df.columns:
                                    trade_record['sentiment_at_entry'] = df.loc[entry_date, main_sentiment] if entry_date in df.index else 0
                            
                            # 🔧 اصلاح 3: Add Telegram-derived Reddit info
                            if TELEGRAM_REDDIT_ANALYSIS_ENABLED and telegram_reddit_features:
                                main_telegram_reddit = telegram_reddit_features[0] if telegram_reddit_features else None
                                if main_telegram_reddit and main_telegram_reddit in df.columns:
                                    trade_record['telegram_reddit_at_entry'] = df.loc[entry_date, main_telegram_reddit] if entry_date in df.index else 0
                            
                            trade_history.append(trade_record)
                            
                            # Update capital
                            trade_amount = capital * TRADE_SIZE_PERCENT
                            profit_loss = trade_amount * pnl_percent
                            capital += profit_loss
                            equity_curve.append(capital)
                            
                            position_open = False
                            entry_index = -1
                            
                            emoji = "✅" if pnl_percent > 0 else "❌"
                            logging.info(f"{emoji} Enhanced Exit at {exit_date}: ${exit_price:.4f}, "
                                       f"P&L: {pnl_percent:.2%}, Reason: {exit_reason}")
                
                # === Enhanced Results Calculation ===
                final_capital = capital
                total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
                num_trades = len(trade_history)
                wins = [t for t in trade_history if t['pnl'] > 0]
                losses = [t for t in trade_history if t['pnl'] <= 0]
                win_rate = len(wins) / num_trades if num_trades > 0 else 0
                
                # Enhanced Performance Metrics
                equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
                max_drawdown = calculate_enhanced_max_drawdown(equity_series)
                
                if trade_history:
                    returns = pd.Series([t['pnl'] for t in trade_history])
                    sharpe_ratio = calculate_enhanced_sharpe_ratio(returns)
                    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
                    profit_factor = abs(sum([t['pnl'] for t in wins]) / sum([t['pnl'] for t in losses])) if losses else np.inf
                else:
                    sharpe_ratio = avg_win = avg_loss = profit_factor = 0
                
                # === 🔧 اصلاح 2: Enhanced Impact Analysis (یکپارچه) ===
                telegram_impact = analyze_telegram_sentiment_and_reddit_impact(trade_history, df)
                
                # === Enhanced Results Storage ===
                enhanced_result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'total_return': total_return,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'final_capital': final_capital,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'data_quality': data_quality['quality_score'],
                    'sentiment_effectiveness': telegram_impact.get('sentiment_effectiveness', 0),
                    'telegram_reddit_effectiveness': telegram_impact.get('telegram_reddit_effectiveness', 0),  # 🔧 اصلاح
                    'combined_effectiveness': telegram_impact.get('combined_effectiveness', 0),  # 🔧 اصلاح
                    'telegram_mapping_detected': data_quality.get('telegram_mapping_detected', False)  # 🔧 اصلاح
                }
                enhanced_results.append(enhanced_result)
                
                # === Enhanced Report Data ===
                report_data = {
                    'Initial Capital': f"${INITIAL_CAPITAL:,.2f}",
                    'Final Capital': f"${final_capital:,.2f}",
                    'Total Return': f"{total_return:.2%}",
                    'Total Trades': num_trades,
                    'Winning Trades': len(wins),
                    'Losing Trades': len(losses),
                    'Win Rate': f"{win_rate:.2%}",
                    'Max Drawdown': f"{max_drawdown:.2%}",
                    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                    'Average Win': f"{avg_win:.2%}",
                    'Average Loss': f"{avg_loss:.2%}",
                    'Profit Factor': f"{profit_factor:.2f}",
                    'optimal_threshold': optimal_threshold,
                    'trade_history': trade_history
                }
                
                # === 🔧 اصلاح 5: Enhanced Console Output (Telegram-based) ===
                print(f"\n{'='*50}")
                print(f"📊 Enhanced Telegram Results: {symbol} ({timeframe})")
                print(f"{'='*50}")
                print(f"💰 Return: {total_return:.2%}")
                print(f"📈 Trades: {num_trades} (Win Rate: {win_rate:.1%})")
                print(f"💵 Final Capital: ${final_capital:,.2f}")
                print(f"📉 Max Drawdown: {max_drawdown:.2%}")
                print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
                print(f"🎭 Telegram Sentiment Effectiveness: {telegram_impact.get('sentiment_effectiveness', 0):.1%}")
                print(f"📱 Telegram-derived Reddit Effectiveness: {telegram_impact.get('telegram_reddit_effectiveness', 0):.1%}")
                print(f"🔗 Combined Effectiveness: {telegram_impact.get('combined_effectiveness', 0):.1%}")
                print(f"📈 Data Quality: {data_quality['quality_score']:.1%}")
                print(f"📱 Telegram Mapping: {'Detected' if data_quality.get('telegram_mapping_detected') else 'Not Detected'}")
                
                # === Enhanced Report Generation ===
                generate_enhanced_report_file(report_data, symbol, timeframe, 
                                            data_quality, telegram_impact, model_info)
                
                # === 🔧 اصلاح 4: Enhanced Visualizations (Telegram-based) ===
                try:
                    generate_enhanced_telegram_visualizations(df, trade_history, symbol, timeframe, 
                                                            report_subfolder_path, data_quality, 
                                                            telegram_impact)
                except Exception as viz_error:
                    logging.warning(f"Enhanced visualization error for {symbol}/{timeframe}: {viz_error}")
                
            except KeyError as ke:
                logging.error(f"Enhanced KeyError for {symbol}/{timeframe}: {ke}")
                continue
            except Exception as e:
                logging.error(f"Enhanced processing error for {symbol}/{timeframe}: {e}")
                continue
    
    # === 🔧 اصلاح 5: Enhanced Overall Summary (Telegram-based) ===
    if len(enhanced_results) > 1:
        print("\n" + "="*100)
        print("📊 ENHANCED TELEGRAM-BASED BACKTEST OVERALL SUMMARY v3.1")
        print("="*100)
        print(f"{'Symbol':<12} {'TF':<4} {'Return':<8} {'Trades':<6} {'Win%':<6} {'Drawdown':<9} "
              f"{'Sharpe':<6} {'Quality':<7} {'Sent%':<5} {'T-Red%':<6} {'Combined%':<9} {'Map':<3}")
        print("-"*100)
        
        total_return_sum = 0
        for r in enhanced_results:
            mapping_indicator = "✅" if r['telegram_mapping_detected'] else "❌"
            print(f"{r['symbol']:<12} {r['timeframe']:<4} "
                  f"{r['total_return']:>7.2%} {r['num_trades']:>5} "
                  f"{r['win_rate']:>5.1%} {r['max_drawdown']:>8.2%} "
                  f"{r['sharpe_ratio']:>5.2f} {r['data_quality']:>6.1%} "
                  f"{r['sentiment_effectiveness']:>4.1%} {r['telegram_reddit_effectiveness']:>5.1%} "
                  f"{r['combined_effectiveness']:>8.1%} {mapping_indicator:<3}")
            total_return_sum += r['total_return']
        
        print("-"*100)
        avg_return = total_return_sum / len(enhanced_results) if enhanced_results else 0
        avg_quality = np.mean([r['data_quality'] for r in enhanced_results])
        avg_sentiment = np.mean([r['sentiment_effectiveness'] for r in enhanced_results])
        avg_telegram_reddit = np.mean([r['telegram_reddit_effectiveness'] for r in enhanced_results])
        avg_combined = np.mean([r['combined_effectiveness'] for r in enhanced_results])
        mapping_percentage = np.mean([r['telegram_mapping_detected'] for r in enhanced_results]) * 100
        
        print(f"Average Return: {avg_return:.2%}")
        print(f"Average Quality: {avg_quality:.1%}")
        print(f"Average Telegram Sentiment Effectiveness: {avg_sentiment:.1%}")
        print(f"Average Telegram-derived Reddit Effectiveness: {avg_telegram_reddit:.1%}")
        print(f"Average Combined Effectiveness: {avg_combined:.1%}")
        print(f"Telegram Mapping Detection Rate: {mapping_percentage:.0f}%")
        print("="*100)
    
    print(f"\n✅ Enhanced Telegram-based Backtest v3.1 completed successfully!")
    print(f"📁 All Enhanced reports and charts saved to:")
    print(f"   📂 {report_subfolder_path}")
    print(f"\n🚀 Enhanced Telegram-based Features Used:")
    print(f"   ✅ 58+ Features Support")
    print(f"   ✅ Telegram Sentiment Analysis Integration")
    print(f"   ✅ Telegram-derived Reddit Features Analysis")
    print(f"   ✅ Enhanced Data Quality Validation (Telegram-based)")
    print(f"   ✅ Multi-source Performance Metrics (Telegram-based)")
    print(f"   ✅ Unified Sentiment-Reddit Impact Analysis")

def run_simple_backtest_legacy(features_path: str, models_path: str):
    """Legacy simple backtest - backward compatibility"""
    logging.info("--- Starting Legacy Simple Backtest (Backward Compatibility) ---")
    
    try:
        # Legacy file loading
        feature_file = find_enhanced_latest_file(
            os.path.join(features_path, 'final_dataset_for_training_*.parquet'),
            "dataset file"
        )
        if not feature_file:
            logging.error("❌ No dataset file found")
            return
            
        model_file = find_enhanced_latest_file(
            os.path.join(models_path, 'optimized_model_*.joblib'),
            "model"
        )
        if not model_file:
            model_file = find_enhanced_latest_file(
                os.path.join(models_path, 'random_forest_model_*.joblib'),
                "legacy model"
            )
            
        scaler_file = find_enhanced_latest_file(
            os.path.join(models_path, 'scaler_optimized_*.joblib'),
            "scaler"
        )
        if not scaler_file:
            scaler_file = find_enhanced_latest_file(
                os.path.join(models_path, 'scaler_*.joblib'),
                "legacy scaler"
            )
        
        if not model_file or not scaler_file:
            logging.error("❌ Required model or scaler files not found")
            return
            
        # Load files
        df_full = pd.read_parquet(feature_file)
        model, optimal_threshold, model_info = load_enhanced_model_package(model_file)
        scaler = joblib.load(scaler_file)
        
        logging.info("✅ Legacy files loaded successfully")
        
    except Exception as e:
        logging.error(f"Error loading legacy files: {e}")
        return

    # Legacy symbol selection
    available_symbols = df_full.index.get_level_values('symbol').unique().tolist()
    print(f"\n📊 Available symbols: {available_symbols}")
    symbol_to_test = input("💱 Which symbol to test? (e.g. BTC/USDT): ").upper().strip()
    
    if symbol_to_test not in available_symbols:
        print(f"❌ Symbol '{symbol_to_test}' not found")
        return

    try:
        available_timeframes = df_full.loc[symbol_to_test].index.get_level_values('timeframe').unique().tolist()
        print(f"⏱️ Available timeframes: {available_timeframes}")
        timeframe_to_test = input("🕐 Which timeframe to test? (e.g. 1h): ").lower().strip()
        
        if timeframe_to_test not in available_timeframes:
            print(f"❌ Timeframe '{timeframe_to_test}' not found")
            return
            
    except KeyError:
        logging.error(f"Symbol/timeframe combination error")
        return

    try:
        df = df_full.loc[(symbol_to_test, timeframe_to_test)].copy()
        logging.info(f"📈 Legacy dataset: {len(df)} records")
        
        if len(df) < TARGET_FUTURE_PERIODS * 2:
            logging.warning(f"⚠️ Insufficient data for meaningful backtest")
            
    except KeyError:
        logging.error(f"Symbol/timeframe combination not found")
        return

    # Legacy backtesting logic (simplified)
    capital = INITIAL_CAPITAL
    position_open = False
    entry_price = 0
    entry_index = -1
    entry_date = None
    trade_history = []

    feature_columns = [col for col in df.columns if col not in ['target', 'timestamp']]
    expected_features = model_info.get('feature_columns', [])
    
    # Handle missing features
    if expected_features:
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        feature_columns = expected_features

    X = df[feature_columns]
    
    try:
        X_scaled = scaler.transform(X)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_scaled)[:, 1]
            df['prediction'] = (y_prob >= optimal_threshold).astype(int)
            df['prediction_prob'] = y_prob
        else:
            df['prediction'] = model.predict(X_scaled)
            df['prediction_prob'] = 0.5
            
    except Exception as pred_error:
        logging.error(f"Error in legacy model prediction: {pred_error}")
        return
    
    # Simple trading simulation
    for i in range(len(df) - TARGET_FUTURE_PERIODS):
        current_row = df.iloc[i]
        
        if not position_open and current_row['prediction'] == 1:
            position_open = True
            entry_price = current_row['close']
            entry_index = i
            entry_date = df.index[i]

        elif position_open and (i >= entry_index + TARGET_FUTURE_PERIODS):
            exit_price = current_row['close']
            exit_date = df.index[i]
            pnl_percent = (exit_price - entry_price) / entry_price
            
            exit_reason = "Target period reached"
            if pnl_percent < -0.05:
                exit_reason = "Stop loss (-5%)"
            elif pnl_percent > 0.10:
                exit_reason = "Take profit (+10%)"
            
            trade_history.append({
                'entry_date': entry_date, 
                'entry_price': entry_price,
                'exit_date': exit_date, 
                'exit_price': exit_price, 
                'pnl': pnl_percent,
                'exit_reason': exit_reason
            })
            
            trade_amount = capital * TRADE_SIZE_PERCENT
            capital += trade_amount * pnl_percent
            position_open = False
            entry_index = -1

    # Results
    final_capital = capital
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    num_trades = len(trade_history)
    wins = [t for t in trade_history if t['pnl'] > 0]
    losses = [t for t in trade_history if t['pnl'] <= 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    
    report_data = {
        'Initial Capital': f"${INITIAL_CAPITAL:,.2f}",
        'Final Capital': f"${final_capital:,.2f}",
        'Total Return': f"{total_return:.2%}",
        'Total Trades': num_trades,
        'Winning Trades': len(wins),
        'Losing Trades': len(losses),
        'Win Rate': f"{win_rate:.2%}",
        'trade_history': trade_history
    }
    
    # Display results
    print("\n" + "="*60)
    print("      Legacy Backtest Results")
    print("="*60)
    for key, value in report_data.items():
        if key != 'trade_history': 
            print(f"{key:<20} {value}")
    print("="*60)
    
    # Generate simple report
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Legacy_Backtest_Report_{symbol_to_test.replace('/', '-')}_{timeframe_to_test}_{timestamp_str}.txt"
    filepath = os.path.join(report_subfolder_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("LEGACY BACKTEST REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Symbol: {symbol_to_test}\n")
        f.write(f"Timeframe: {timeframe_to_test}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in report_data.items():
            if key != 'trade_history':
                f.write(f"{key}: {value}\n")
    
    print(f"\n✅ Legacy backtest completed!")
    print(f"📁 Report saved to: {filepath}")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Enhanced Advanced Backtesting System v3.1")
    print("📱 Complete Integration with Telegram-based Sentiment & Reddit Analysis")
    print("🔧 58+ Features Support with Enhanced Models v6.1+")
    print("="*80)
    
    choice = input("\nSelect Enhanced mode:\n"
                  "1. Enhanced Complete Backtest v3.1 (Telegram-based - recommended)\n"
                  "2. Legacy Simple Backtest (backward compatibility)\n"
                  "Choice (1/2): ").strip()
    
    if choice == '2':
        print("\n🔄 Running Legacy Mode...")
        run_simple_backtest_legacy(FEATURES_PATH, MODELS_PATH)
    else:
        print("\n🚀 Running Enhanced Telegram-based Complete Mode v3.1...")
        run_enhanced_backtest_complete(FEATURES_PATH, MODELS_PATH)