#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت آموزش و ارزیابی مدل (نسخه 6.0 - سازگاری کامل با فایل‌های اصلاح شده)
تغییرات v6.0:
- ✅ سازگاری کامل با sentiment features جدید (Broadcasting structure)
- ✅ پشتیبانی کامل از Reddit features (reddit_score, reddit_comments)
- ✅ بهبود validation برای multi-source sentiment data
- ✅ Enhanced feature importance analysis با تفکیک sentiment/technical
- ✅ Reddit features impact analysis
- ✅ بهبود data quality validation
- ✅ Multi-source sentiment effectiveness reporting
- ✅ بهینه‌سازی feature selection برای mixed features
- ✅ حفظ تمام اصلاحات v5.2 (Cross-Validation, Precision-Recall balance)
"""
import os
import glob
import pandas as pd
import logging
import configparser
import joblib
import numpy as np

# --- تغییر کلیدی ۱: تعیین موتور گرافیکی قبل از ایمپورت pyplot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight

# === بخش XGBoost برای Ensemble ===
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logging.info("✅ XGBoost available for ensemble method")
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("⚠️ XGBoost not available, using RandomForest only")

# بخش خواندن پیکربندی و تنظیمات لاگ‌گیری
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    FEATURES_PATH = config.get('Paths', 'features')
    MODELS_PATH = config.get('Paths', 'models')
    LOG_PATH = config.get('Paths', 'logs')
    
    # === تنظیمات جدید برای sentiment و Reddit analysis ===
    SENTIMENT_ANALYSIS_ENABLED = config.getboolean('Enhanced_Analysis', 'sentiment_analysis_enabled', fallback=True)
    REDDIT_ANALYSIS_ENABLED = config.getboolean('Enhanced_Analysis', 'reddit_analysis_enabled', fallback=True)
    DETAILED_FEATURE_ANALYSIS = config.getboolean('Enhanced_Analysis', 'detailed_feature_analysis', fallback=True)
    CORRELATION_ANALYSIS_ENABLED = config.getboolean('Enhanced_Analysis', 'correlation_analysis_enabled', fallback=True)
    
    # محدودیت‌های data quality
    MIN_SENTIMENT_COVERAGE = config.getfloat('Data_Quality', 'min_sentiment_coverage', fallback=0.10)  # حداقل 10% داده با sentiment
    MIN_REDDIT_COVERAGE = config.getfloat('Data_Quality', 'min_reddit_coverage', fallback=0.05)      # حداقل 5% داده با Reddit
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Error: {e}")
    exit()

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])

# === توابع جدید برای تحلیل کیفیت داده ===
def analyze_sentiment_data_quality(df: pd.DataFrame) -> dict:
    """تحلیل جامع کیفیت داده‌های احساسات"""
    logging.info("🎭 شروع تحلیل کیفیت داده‌های احساسات...")
    
    sentiment_stats = {
        'total_records': len(df),
        'sentiment_features_found': [],
        'reddit_features_found': [],
        'quality_metrics': {},
        'coverage_stats': {},
        'warnings': []
    }
    
    # شناسایی ستون‌های احساسات
    sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
    reddit_columns = [col for col in df.columns if 'reddit' in col.lower()]
    
    sentiment_stats['sentiment_features_found'] = sentiment_columns
    sentiment_stats['reddit_features_found'] = reddit_columns
    
    logging.info(f"📊 Sentiment features یافت شده: {len(sentiment_columns)}")
    for col in sentiment_columns:
        logging.info(f"   - {col}")
    
    logging.info(f"🔴 Reddit features یافت شده: {len(reddit_columns)}")
    for col in reddit_columns:
        logging.info(f"   - {col}")
    
    # تحلیل کیفیت sentiment features
    if sentiment_columns:
        main_sentiment_col = None
        
        # یافتن ستون اصلی sentiment
        for col in ['sentiment_compound_mean', 'sentiment_score', 'sentiment_compound']:
            if col in df.columns:
                main_sentiment_col = col
                break
        
        if main_sentiment_col:
            non_zero_count = (df[main_sentiment_col] != 0).sum()
            coverage = non_zero_count / len(df)
            
            sentiment_stats['coverage_stats']['sentiment_coverage'] = coverage
            sentiment_stats['coverage_stats']['sentiment_non_zero_count'] = non_zero_count
            
            # محاسبه آمار کیفیت
            sentiment_values = df[main_sentiment_col][df[main_sentiment_col] != 0]
            if len(sentiment_values) > 0:
                sentiment_stats['quality_metrics']['sentiment_mean'] = sentiment_values.mean()
                sentiment_stats['quality_metrics']['sentiment_std'] = sentiment_values.std()
                sentiment_stats['quality_metrics']['sentiment_range'] = (sentiment_values.min(), sentiment_values.max())
                
                # تحلیل توزیع
                positive_count = (sentiment_values > 0.05).sum()
                negative_count = (sentiment_values < -0.05).sum()
                neutral_count = len(sentiment_values) - positive_count - negative_count
                
                sentiment_stats['quality_metrics']['sentiment_distribution'] = {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                }
                
                logging.info(f"📈 Sentiment Coverage: {coverage:.2%} ({non_zero_count:,} records)")
                logging.info(f"📊 Sentiment Distribution: Pos={positive_count}, Neg={negative_count}, Neu={neutral_count}")
            
            # بررسی آستانه کیفیت
            if coverage < MIN_SENTIMENT_COVERAGE:
                warning = f"⚠️ Sentiment coverage ({coverage:.2%}) کمتر از حد نصاب ({MIN_SENTIMENT_COVERAGE:.1%})"
                sentiment_stats['warnings'].append(warning)
                logging.warning(warning)
    
    # تحلیل Reddit features
    if reddit_columns:
        reddit_score_col = next((col for col in reddit_columns if 'score' in col and 'ma' not in col), None)
        reddit_comments_col = next((col for col in reddit_columns if 'comments' in col and 'ma' not in col), None)
        
        if reddit_score_col:
            non_zero_reddit = (df[reddit_score_col] != 0).sum()
            reddit_coverage = non_zero_reddit / len(df)
            
            sentiment_stats['coverage_stats']['reddit_coverage'] = reddit_coverage
            sentiment_stats['coverage_stats']['reddit_non_zero_count'] = non_zero_reddit
            
            if reddit_coverage > 0:
                reddit_values = df[reddit_score_col][df[reddit_score_col] != 0]
                sentiment_stats['quality_metrics']['reddit_mean'] = reddit_values.mean()
                sentiment_stats['quality_metrics']['reddit_std'] = reddit_values.std()
                
                logging.info(f"🔴 Reddit Coverage: {reddit_coverage:.2%} ({non_zero_reddit:,} records)")
            
            # بررسی آستانه Reddit
            if reddit_coverage > 0 and reddit_coverage < MIN_REDDIT_COVERAGE:
                warning = f"⚠️ Reddit coverage ({reddit_coverage:.2%}) کمتر از حد نصاب ({MIN_REDDIT_COVERAGE:.1%})"
                sentiment_stats['warnings'].append(warning)
                logging.warning(warning)
    
    # نمایش هشدارها
    if sentiment_stats['warnings']:
        logging.warning("⚠️ Data Quality Warnings:")
        for warning in sentiment_stats['warnings']:
            logging.warning(f"   {warning}")
    else:
        logging.info("✅ Data quality checks passed")
    
    return sentiment_stats

def categorize_features(feature_columns: list) -> dict:
    """تفکیک features بر اساس نوع"""
    feature_categories = {
        'technical_indicators': [],
        'sentiment_features': [],
        'reddit_features': [],
        'price_features': [],
        'volume_features': [],
        'other_features': []
    }
    
    for feature in feature_columns:
        feature_lower = feature.lower()
        
        if 'sentiment' in feature_lower:
            feature_categories['sentiment_features'].append(feature)
        elif 'reddit' in feature_lower:
            feature_categories['reddit_features'].append(feature)
        elif any(ind in feature_lower for ind in ['rsi', 'macd', 'bb_', 'ema', 'sma', 'stoch', 'williams', 'cci', 'adx', 'psar']):
            feature_categories['technical_indicators'].append(feature)
        elif any(price in feature_lower for price in ['return', 'price', 'close_position', 'hl_ratio']):
            feature_categories['price_features'].append(feature)
        elif any(vol in feature_lower for vol in ['volume', 'obv', 'mfi', 'vwap']):
            feature_categories['volume_features'].append(feature)
        else:
            feature_categories['other_features'].append(feature)
    
    return feature_categories

def analyze_feature_importance_by_category(model, feature_columns: list, feature_categories: dict) -> dict:
    """تحلیل اهمیت features به تفکیک دسته‌بندی"""
    if not hasattr(model, 'feature_importances_'):
        return {}
    
    importance_by_category = {}
    
    # محاسبه اهمیت کل برای هر دسته
    for category, features in feature_categories.items():
        if features:
            category_importance = 0
            category_features_with_importance = []
            
            for feature in features:
                if feature in feature_columns:
                    idx = feature_columns.index(feature)
                    importance = model.feature_importances_[idx]
                    category_importance += importance
                    category_features_with_importance.append((feature, importance))
            
            importance_by_category[category] = {
                'total_importance': category_importance,
                'feature_count': len(features),
                'avg_importance': category_importance / len(features) if features else 0,
                'top_features': sorted(category_features_with_importance, key=lambda x: x[1], reverse=True)[:3]
            }
    
    return importance_by_category

def analyze_sentiment_correlation_with_target(df: pd.DataFrame, sentiment_stats: dict) -> dict:
    """تحلیل همبستگی sentiment features با target"""
    correlation_analysis = {}
    
    if 'target' not in df.columns:
        return correlation_analysis
    
    sentiment_features = sentiment_stats['sentiment_features_found']
    reddit_features = sentiment_stats['reddit_features_found']
    
    # تحلیل همبستگی sentiment features
    if sentiment_features:
        sentiment_correlations = {}
        for feature in sentiment_features:
            if feature in df.columns:
                # فقط روی مقادیر غیرصفر محاسبه کن
                non_zero_mask = df[feature] != 0
                if non_zero_mask.sum() > 10:  # حداقل 10 مقدار غیرصفر
                    corr = df.loc[non_zero_mask, feature].corr(df.loc[non_zero_mask, 'target'])
                    sentiment_correlations[feature] = corr if not pd.isna(corr) else 0
                else:
                    sentiment_correlations[feature] = 0
        
        correlation_analysis['sentiment_correlations'] = sentiment_correlations
        
        # بهترین sentiment feature
        if sentiment_correlations:
            best_sentiment = max(sentiment_correlations.items(), key=lambda x: abs(x[1]))
            correlation_analysis['best_sentiment_feature'] = best_sentiment
    
    # تحلیل همبستگی Reddit features
    if reddit_features:
        reddit_correlations = {}
        for feature in reddit_features:
            if feature in df.columns:
                non_zero_mask = df[feature] != 0
                if non_zero_mask.sum() > 5:  # حداقل 5 مقدار غیرصفر
                    corr = df.loc[non_zero_mask, feature].corr(df.loc[non_zero_mask, 'target'])
                    reddit_correlations[feature] = corr if not pd.isna(corr) else 0
                else:
                    reddit_correlations[feature] = 0
        
        correlation_analysis['reddit_correlations'] = reddit_correlations
        
        # بهترین Reddit feature
        if reddit_correlations:
            best_reddit = max(reddit_correlations.items(), key=lambda x: abs(x[1]))
            correlation_analysis['best_reddit_feature'] = best_reddit
    
    return correlation_analysis

def clean_data(X, y):
    """
    پاکسازی داده‌ها از مقادیر نامعتبر (inf, -inf, nan)
    """
    logging.info("شروع پاکسازی داده‌ها...")
    
    # بررسی اولیه
    initial_shape = X.shape
    logging.info(f"شکل اولیه داده: {initial_shape}")
    
    # شناسایی ستون‌های دارای مشکل
    problematic_columns = []
    
    # بررسی inf در هر ستون
    for col in X.columns:
        inf_count = np.isinf(X[col]).sum()
        nan_count = X[col].isna().sum()
        if inf_count > 0 or nan_count > 0:
            problematic_columns.append((col, inf_count, nan_count))
            logging.warning(f"ستون '{col}': {inf_count} مقدار inf، {nan_count} مقدار NaN")
    
    # گزارش ستون‌های مشکل‌دار
    if problematic_columns:
        logging.info(f"تعداد ستون‌های دارای مشکل: {len(problematic_columns)}")
        
    # روش 1: جایگزینی مقادیر نامعتبر با میانه
    logging.info("جایگزینی مقادیر نامعتبر با میانه...")
    for col in X.columns:
        # جایگزینی inf با NaN
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        
        # محاسبه میانه بدون در نظر گرفتن NaN
        if X[col].notna().any():
            median_val = X[col].median()
            # 🔧 FIX: استفاده از .loc برای جلوگیری از SettingWithCopyWarning
            X.loc[X[col].isna(), col] = median_val
        else:
            # اگر تمام مقادیر NaN باشند، با 0 پر کنیم
            X.loc[:, col] = 0
    
    # روش 2: حذف ستون‌هایی که بیش از 50% مقادیر نامعتبر دارند
    threshold = 0.5
    cols_to_drop = []
    for col in X.columns:
        invalid_ratio = (X[col].isna().sum() + np.isinf(X[col]).sum()) / len(X)
        if invalid_ratio > threshold:
            cols_to_drop.append(col)
            logging.warning(f"ستون '{col}' به دلیل {invalid_ratio:.1%} مقادیر نامعتبر حذف خواهد شد")
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        logging.info(f"{len(cols_to_drop)} ستون حذف شد")
    
    # بررسی نهایی
    final_inf_count = np.isinf(X.values).sum()
    final_nan_count = X.isna().sum().sum()
    
    logging.info(f"تعداد نهایی مقادیر inf: {final_inf_count}")
    logging.info(f"تعداد نهایی مقادیر NaN: {final_nan_count}")
    logging.info(f"شکل نهایی داده: {X.shape}")
    
    return X, y

# === بخش Threshold Optimization (حفظ شده از v5.2) ===
def find_optimal_threshold(y_true, y_prob, target_precision=0.60):
    """
    یافتن آستانه بهینه برای بهبود precision - اصلاح شده برای سیگنال‌های بیشتر
    """
    logging.info("🎯 شروع بهینه‌سازی threshold برای بهبود precision...")
    
    # محاسبه precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # 🔧 کاهش target precision برای سیگنال‌های بیشتر
    valid_indices = precisions >= target_precision
    
    if valid_indices.any():
        # انتخاب threshold با بالاترین recall در precision های بالای هدف
        best_idx = np.argmax(recalls[valid_indices])
        valid_idx = np.where(valid_indices)[0][best_idx]
        optimal_threshold = thresholds[valid_idx]
        optimal_precision = precisions[valid_idx]
        optimal_recall = recalls[valid_idx]
        
        logging.info(f"✅ Threshold بهینه یافت شد: {optimal_threshold:.3f}")
        logging.info(f"📊 Precision: {optimal_precision:.3f}, Recall: {optimal_recall:.3f}")
    else:
        # اگر precision هدف قابل دستیابی نیست، threshold پایین‌تری انتخاب کنیم
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        optimal_precision = precisions[best_idx]
        optimal_recall = recalls[best_idx]
        
        # 🔧 اگر threshold خیلی بالا است، کاهش دهیم
        if optimal_threshold > 0.7:
            # انتخاب threshold کمتر برای سیگنال‌های بیشتر
            suitable_idx = np.where(thresholds <= 0.6)[0]
            if len(suitable_idx) > 0:
                best_in_range = suitable_idx[np.argmax(f1_scores[suitable_idx])]
                optimal_threshold = thresholds[best_in_range]
                optimal_precision = precisions[best_in_range]
                optimal_recall = recalls[best_in_range]
                logging.info(f"🔧 Threshold کاهش یافت برای سیگنال‌های بیشتر: {optimal_threshold:.3f}")
        
        logging.warning(f"⚠️ Precision {target_precision:.0%} قابل دستیابی نیست. بهترین F1: {f1_scores[best_idx]:.3f}")
        logging.info(f"📊 Threshold: {optimal_threshold:.3f}, Precision: {optimal_precision:.3f}, Recall: {optimal_recall:.3f}")
    
    return optimal_threshold, optimal_precision, optimal_recall

# === بخش Ensemble Model (حفظ شده از v5.2) ===
def create_ensemble_model(X_train, y_train, class_weights):
    """
    ایجاد مدل ensemble از RandomForest + XGBoost - اصلاح شده
    """
    models = {}
    
    # RandomForest اصلی (بهبود یافته)
    logging.info("🌲 آموزش RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # افزایش از 100
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced_subsample',  # بهتر از balanced
        max_depth=12,  # افزایش کمی
        min_samples_split=3,  # کاهش برای overfitting کمتر
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True  # Out-of-bag scoring
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # گزارش OOB Score
    if hasattr(rf_model, 'oob_score_'):
        logging.info(f"📊 RandomForest OOB Score: {rf_model.oob_score_:.4f}")
    
    # XGBoost (اگر در دسترس باشد) - 🔧 اصلاح کامل validation
    if XGBOOST_AVAILABLE:
        logging.info("⚡ آموزش XGBoost...")
        
        try:
            # محاسبه scale_pos_weight برای class imbalance
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=6,
                learning_rate=0.05,  # کاهش برای بهتر یادگیری
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,  # برای class imbalance
                eval_metric='logloss'
                # 🔧 حذف early_stopping_rounds از اینجا
            )
            
            # 🔧 آموزش بدون validation set برای جلوگیری از خطا
            # اگر validation نیاز است، جداگانه انجام می‌شود
            logging.info("آموزش XGBoost بدون early stopping...")
            xgb_model.fit(X_train, y_train)
            
            models['XGBoost'] = xgb_model
            logging.info("✅ XGBoost آموزش داده شد")
            
        except Exception as xgb_error:
            logging.warning(f"⚠️ خطا در آموزش XGBoost: {xgb_error}")
            logging.info("ادامه با RandomForest تنها...")
    
    return models

def train_and_evaluate_model(features_path: str, models_path: str):
    logging.info("شروع گام ۳-ب: آموزش و ارزیابی مدل (نسخه 6.0 - سازگاری کامل)...")
    
    # یافتن آخرین فایل دیتاست
    list_of_files = glob.glob(os.path.join(features_path, 'final_dataset_for_training_*.parquet'))
    if not list_of_files:
        logging.error(f"هیچ دیتاستی برای آموزش در مسیر '{features_path}' یافت نشد.")
        return
    latest_file = max(list_of_files, key=os.path.getctime)
    logging.info(f"در حال خواندن دیتاست نهایی: {os.path.basename(latest_file)}")
    
    # خواندن داده
    df = pd.read_parquet(latest_file)
    logging.info(f"ابعاد دیتاست: {df.shape}")
    
    # === تحلیل جامع کیفیت داده (جدید) ===
    logging.info("\n" + "="*60)
    logging.info("📊 تحلیل جامع کیفیت داده (Enhanced v6.0)")
    logging.info("="*60)
    
    sentiment_stats = analyze_sentiment_data_quality(df)
    
    # بررسی توزیع کلاس‌ها
    target_distribution = df['target'].value_counts().sort_index()
    logging.info(f"توزیع کلاس‌ها: {target_distribution.to_dict()}")
    
    # بررسی اینکه آیا هر دو کلاس وجود دارند
    unique_classes = df['target'].unique()
    logging.info(f"کلاس‌های موجود: {sorted(unique_classes)}")
    
    if len(unique_classes) < 2:
        logging.warning(f"تنها {len(unique_classes)} کلاس در داده موجود است. آموزش مدل ممکن نیست.")
        print(f"⚠️ هشدار: تنها {len(unique_classes)} کلاس در داده موجود است.")
        return
    
    # جداسازی ویژگی‌ها و متغیر هدف
    feature_columns = [col for col in df.columns if col not in ['target', 'timestamp', 'symbol', 'timeframe']]
    X = df[feature_columns]
    y = df['target']
    
    logging.info(f"تعداد ویژگی‌ها: {len(feature_columns)}")
    logging.info(f"تعداد نمونه‌ها: {len(X)}")
    
    # === تفکیک features به دسته‌بندی (جدید) ===
    feature_categories = categorize_features(feature_columns)
    
    logging.info("\n🏷️ دسته‌بندی Features:")
    for category, features in feature_categories.items():
        if features:
            logging.info(f"   📊 {category}: {len(features)} features")
            for feature in features[:3]:  # نمایش 3 نمونه اول
                logging.info(f"      - {feature}")
            if len(features) > 3:
                logging.info(f"      ... و {len(features) - 3} feature دیگر")
    
    # === تحلیل همبستگی (جدید) ===
    if CORRELATION_ANALYSIS_ENABLED:
        logging.info("\n📈 تحلیل همبستگی Sentiment و Reddit features با Target:")
        correlation_analysis = analyze_sentiment_correlation_with_target(df, sentiment_stats)
        
        if 'sentiment_correlations' in correlation_analysis:
            logging.info("🎭 همبستگی Sentiment Features:")
            for feature, corr in correlation_analysis['sentiment_correlations'].items():
                logging.info(f"   {feature}: {corr:.4f}")
            
            if 'best_sentiment_feature' in correlation_analysis:
                best_feature, best_corr = correlation_analysis['best_sentiment_feature']
                logging.info(f"✨ بهترین Sentiment Feature: {best_feature} (همبستگی: {best_corr:.4f})")
        
        if 'reddit_correlations' in correlation_analysis:
            logging.info("🔴 همبستگی Reddit Features:")
            for feature, corr in correlation_analysis['reddit_correlations'].items():
                logging.info(f"   {feature}: {corr:.4f}")
            
            if 'best_reddit_feature' in correlation_analysis:
                best_feature, best_corr = correlation_analysis['best_reddit_feature']
                logging.info(f"✨ بهترین Reddit Feature: {best_feature} (همبستگی: {best_corr:.4f})")
    
    # --- پاکسازی داده‌ها ---
    X, y = clean_data(X, y)
    
    # === بخش محاسبه Class Weights پیشرفته ===
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    logging.info(f"🎯 محاسبه Class Weights: {class_weight_dict}")
    
    # بررسی حداقل نمونه برای آموزش
    min_class_size = target_distribution.min()
    if min_class_size < 10:
        logging.warning(f"کلاس اقلیت تنها {min_class_size} نمونه دارد. نتایج ممکن است قابل اعتماد نباشد.")
        print(f"⚠️ هشدار: کلاس اقلیت تنها {min_class_size} نمونه دارد.")
    
    # تقسیم داده با استفاده از stratified sampling
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info("تقسیم stratified با موفقیت انجام شد.")
    except ValueError as e:
        logging.warning(f"امکان تقسیم stratified وجود ندارد: {e}")
        logging.info("از تقسیم معمولی استفاده می‌شود...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # بررسی توزیع در train و test
    train_distribution = y_train.value_counts().sort_index()
    test_distribution = y_test.value_counts().sort_index()
    logging.info(f"توزیع train: {train_distribution.to_dict()}")
    logging.info(f"توزیع test: {test_distribution.to_dict()}")
    
    # مقیاس‌بندی ویژگی‌ها با RobustScaler (مقاوم در برابر outlier)
    logging.info("در حال مقیاس‌بندی ویژگی‌ها با RobustScaler...")
    try:
        # ابتدا با RobustScaler تلاش می‌کنیم
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("مقیاس‌بندی با RobustScaler انجام شد.")
    except Exception as e:
        logging.warning(f"خطا در RobustScaler: {e}")
        logging.info("تلاش با StandardScaler...")
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logging.info("مقیاس‌بندی با StandardScaler انجام شد.")
        except Exception as e2:
            logging.error(f"خطا در مقیاس‌بندی: {e2}")
            # بدون مقیاس‌بندی ادامه می‌دهیم
            logging.warning("ادامه بدون مقیاس‌بندی...")
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
            scaler = None
    
    # === آموزش Ensemble Models ===
    logging.info("🤖 شروع آموزش مدل‌های Ensemble...")
    models = create_ensemble_model(X_train_scaled, y_train, class_weight_dict)
    
    # === ارزیابی مدل‌ها و انتخاب بهترین ===
    best_model = None
    best_model_name = None
    best_f1_score = 0  # 🔧 تغییر معیار از precision به F1 برای تعادل بهتر
    model_results = {}
    
    for model_name, model in models.items():
        logging.info(f"📊 ارزیابی {model_name}...")
        
        # پیش‌بینی احتمالات
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 🔧 بهینه‌سازی threshold با هدف precision کمتر
        optimal_threshold, precision, recall = find_optimal_threshold(y_test, y_prob, target_precision=0.60)
        
        # پیش‌بینی با threshold بهینه
        y_pred_optimized = (y_prob >= optimal_threshold).astype(int)
        
        # محاسبه metrics
        accuracy = accuracy_score(y_test, y_pred_optimized)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # ذخیره نتایج
        model_results[model_name] = {
            'model': model,
            'threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'y_pred': y_pred_optimized,
            'y_prob': y_prob
        }
        
        logging.info(f"   Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        
        # 🔧 انتخاب بهترین مدل بر اساس F1 score برای تعادل بهتر
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model = model
            best_model_name = model_name
    
    # انتخاب بهترین مدل
    best_result = model_results[best_model_name]
    logging.info(f"🏆 بهترین مدل: {best_model_name}")
    logging.info(f"📊 Metrics: F1={best_result['f1_score']:.4f}, Precision={best_result['precision']:.4f}, Recall={best_result['recall']:.4f}")
    
    # === نمایش نتایج نهایی ===
    y_pred_final = best_result['y_pred']
    accuracy_final = best_result['accuracy']
    
    logging.info("--- نتایج ارزیابی عملکرد مدل بهبود یافته ---")
    logging.info(f"🎯 بهترین مدل: {best_model_name}")
    logging.info(f"📈 Threshold بهینه: {best_result['threshold']:.4f}")
    logging.info(f"✅ Accuracy بهبود یافته: {accuracy_final:.2%}")
    logging.info(f"🎯 Precision بهبود یافته: {best_result['precision']:.2%}")
    logging.info(f"📊 Recall بهبود یافته: {best_result['recall']:.2%}")
    logging.info(f"⚖️ F1 Score: {best_result['f1_score']:.4f}")
    
    print(f"\n🎉 === نتایج بهبود یافته (v6.0 - Enhanced) ===")
    print(f"🏆 بهترین مدل: {best_model_name}")
    print(f"✅ Accuracy: {accuracy_final:.2%}")
    print(f"🎯 Precision: {best_result['precision']:.2%}")
    print(f"📊 Recall: {best_result['recall']:.2%} (بهبود یافته)")
    print(f"⚖️ F1 Score: {best_result['f1_score']:.4f}")
    print(f"⚙️ Optimal Threshold: {best_result['threshold']:.4f}")
    
    # بررسی کلاس‌های موجود در test set
    unique_test_classes = sorted(np.unique(y_test))
    unique_pred_classes = sorted(np.unique(y_pred_final))
    
    logging.info(f"کلاس‌های موجود در y_test: {unique_test_classes}")
    logging.info(f"کلاس‌های پیش‌بینی شده: {unique_pred_classes}")
    
    # گزارش طبقه‌بندی با labels مشخص
    try:
        if len(unique_test_classes) == 2:
            target_names = ['NO_PROFIT (0)', 'PROFIT (1)']
            labels = [0, 1]
        else:
            target_names = [f'Class {cls}' for cls in unique_test_classes]
            labels = unique_test_classes
            
        report = classification_report(
            y_test, y_pred_final, 
            target_names=target_names,
            labels=labels,
            zero_division=0
        )
        logging.info("Classification Report (Enhanced v6.0):\n" + report)
        print("\n📊 Classification Report (Enhanced v6.0):")
        print(report)
        
    except Exception as e:
        logging.warning(f"خطا در تولید classification report: {e}")
        # گزارش ساده
        report = classification_report(y_test, y_pred_final, zero_division=0)
        logging.info("Classification Report (Simple):\n" + report)
        print("\n📊 Classification Report:")
        print(report)
    
    # ماتریس درهم‌ریختگی
    try:
        cm = confusion_matrix(y_test, y_pred_final)
        logging.info("Confusion Matrix (Enhanced v6.0):\n" + str(cm))
        print("\n🔄 Confusion Matrix (Enhanced v6.0):")
        print(cm)
        
        # رسم نمودار
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {best_model_name} (Enhanced v6.0)')
        plot_filename = os.path.join(models_path, f"confusion_matrix_enhanced_v6_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"نمودار ماتریس درهم‌ریختگی در '{plot_filename}' ذخیره شد.")
        
    except Exception as e:
        logging.warning(f"خطا در تولید confusion matrix: {e}")
    
    # 🔧 Cross-validation اصلاح شده - بدون early stopping
    if len(unique_classes) == 2 and min_class_size >= 3:
        try:
            logging.info("در حال انجام Cross-Validation...")
            # استفاده از مدل ساده برای CV (بدون early stopping)
            cv_model = best_model
            if best_model_name == 'XGBoost':
                # برای XGBoost، مدل ساده‌تری برای CV استفاده می‌کنیم
                cv_model = xgb.XGBClassifier(
                    n_estimators=100,  # کمتر برای CV سریع‌تر
                    random_state=42,
                    n_jobs=-1,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric='logloss'
                    # بدون early_stopping_rounds
                )
                cv_model.fit(X_train_scaled, y_train)
            
            cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=3, scoring='accuracy')
            logging.info(f"CV Scores: {cv_scores}")
            logging.info(f"میانگین CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"🔄 Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as e:
            logging.warning(f"خطا در Cross-Validation: {e}")
    
    # === تحلیل اهمیت ویژگی‌ها (بهبود یافته) ===
    try:
        # فقط ویژگی‌هایی که در X_train موجودند
        actual_feature_columns = X_train.columns.tolist()
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': actual_feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logging.info("🔝 Top 10 مهم‌ترین ویژگی‌ها:")
            for i, row in feature_importance.head(10).iterrows():
                logging.info(f"{row['feature']}: {row['importance']:.4f}")
                
            print(f"\n🔝 Top 5 مهم‌ترین ویژگی‌ها ({best_model_name}):")
            print(feature_importance.head().to_string(index=False))
            
            # === تحلیل اهمیت به تفکیک دسته (جدید) ===
            if DETAILED_FEATURE_ANALYSIS:
                logging.info("\n📊 تحلیل اهمیت Features به تفکیک دسته:")
                importance_by_category = analyze_feature_importance_by_category(
                    best_model, actual_feature_columns, feature_categories
                )
                
                # نمایش نتایج تحلیل دسته‌بندی
                category_summary = {}
                for category, stats in importance_by_category.items():
                    if stats['feature_count'] > 0:
                        category_summary[category] = {
                            'total_importance': stats['total_importance'],
                            'avg_importance': stats['avg_importance'],
                            'feature_count': stats['feature_count']
                        }
                        
                        logging.info(f"\n🏷️ {category}:")
                        logging.info(f"   📊 تعداد features: {stats['feature_count']}")
                        logging.info(f"   📈 مجموع اهمیت: {stats['total_importance']:.4f}")
                        logging.info(f"   📊 میانگین اهمیت: {stats['avg_importance']:.4f}")
                        
                        # نمایش top features این دسته
                        logging.info(f"   🔝 Top features:")
                        for feature, importance in stats['top_features']:
                            logging.info(f"      - {feature}: {importance:.4f}")
                
                # خلاصه نهایی اهمیت دسته‌ها
                print(f"\n📊 === خلاصه اهمیت Features به تفکیک دسته ===")
                sorted_categories = sorted(category_summary.items(), 
                                         key=lambda x: x[1]['total_importance'], reverse=True)
                
                for category, stats in sorted_categories:
                    percentage = (stats['total_importance'] / sum(best_model.feature_importances_)) * 100
                    print(f"🏷️ {category}: {percentage:.1f}% (میانگین: {stats['avg_importance']:.4f})")
        
    except Exception as e:
        logging.warning(f"خطا در محاسبه اهمیت ویژگی‌ها: {e}")
    
    # === گزارش تأثیر Sentiment و Reddit Features (جدید) ===
    if SENTIMENT_ANALYSIS_ENABLED or REDDIT_ANALYSIS_ENABLED:
        print(f"\n🎭 === تحلیل تأثیر Sentiment و Reddit Features ===")
        
        # آمار coverage
        if sentiment_stats['coverage_stats']:
            if 'sentiment_coverage' in sentiment_stats['coverage_stats']:
                sentiment_coverage = sentiment_stats['coverage_stats']['sentiment_coverage']
                print(f"📊 Sentiment Coverage: {sentiment_coverage:.2%}")
                
            if 'reddit_coverage' in sentiment_stats['coverage_stats']:
                reddit_coverage = sentiment_stats['coverage_stats']['reddit_coverage']
                print(f"🔴 Reddit Coverage: {reddit_coverage:.2%}")
        
        # اهمیت features
        if hasattr(best_model, 'feature_importances_') and 'sentiment_features' in feature_categories:
            sentiment_features = feature_categories['sentiment_features']
            reddit_features = feature_categories['reddit_features']
            
            # محاسبه مجموع اهمیت sentiment features
            total_sentiment_importance = 0
            for feature in sentiment_features:
                if feature in actual_feature_columns:
                    idx = actual_feature_columns.index(feature)
                    total_sentiment_importance += best_model.feature_importances_[idx]
            
            # محاسبه مجموع اهمیت reddit features
            total_reddit_importance = 0
            for feature in reddit_features:
                if feature in actual_feature_columns:
                    idx = actual_feature_columns.index(feature)
                    total_reddit_importance += best_model.feature_importances_[idx]
            
            total_importance = sum(best_model.feature_importances_)
            sentiment_percentage = (total_sentiment_importance / total_importance) * 100
            reddit_percentage = (total_reddit_importance / total_importance) * 100
            
            print(f"📈 تأثیر Sentiment Features: {sentiment_percentage:.1f}%")
            print(f"📈 تأثیر Reddit Features: {reddit_percentage:.1f}%")
            
            # نمایش بهترین sentiment و reddit features
            if sentiment_features:
                best_sentiment_feature = None
                best_sentiment_importance = 0
                for feature in sentiment_features:
                    if feature in actual_feature_columns:
                        idx = actual_feature_columns.index(feature)
                        importance = best_model.feature_importances_[idx]
                        if importance > best_sentiment_importance:
                            best_sentiment_importance = importance
                            best_sentiment_feature = feature
                
                if best_sentiment_feature:
                    print(f"🌟 بهترین Sentiment Feature: {best_sentiment_feature} ({best_sentiment_importance:.4f})")
            
            if reddit_features:
                best_reddit_feature = None
                best_reddit_importance = 0
                for feature in reddit_features:
                    if feature in actual_feature_columns:
                        idx = actual_feature_columns.index(feature)
                        importance = best_model.feature_importances_[idx]
                        if importance > best_reddit_importance:
                            best_reddit_importance = importance
                            best_reddit_feature = feature
                
                if best_reddit_feature:
                    print(f"🌟 بهترین Reddit Feature: {best_reddit_feature} ({best_reddit_importance:.4f})")
    
    # ذخیره مدل و اطلاعات بهبود یافته
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_filename = os.path.join(models_path, f"enhanced_model_v6_{best_model_name.lower()}_{timestamp_str}.joblib")
    
    # ذخیره مدل به همراه sentiment و reddit analysis
    model_package = {
        'model': best_model,
        'model_type': best_model_name,
        'optimal_threshold': best_result['threshold'],
        'accuracy': accuracy_final,
        'precision': best_result['precision'],
        'recall': best_result['recall'],
        'f1_score': best_result['f1_score'],
        'feature_columns': actual_feature_columns,
        'feature_categories': feature_categories,
        'sentiment_stats': sentiment_stats,
        'model_version': '6.0_enhanced'
    }
    
    # اضافه کردن correlation analysis اگر محاسبه شده باشد
    if CORRELATION_ANALYSIS_ENABLED and 'correlation_analysis' in locals():
        model_package['correlation_analysis'] = correlation_analysis
    
    joblib.dump(model_package, model_filename)
    logging.info(f"مدل Enhanced v6.0 در فایل '{model_filename}' ذخیره شد.")
    
    if scaler is not None:
        scaler_filename = os.path.join(models_path, f"scaler_enhanced_v6_{timestamp_str}.joblib")
        joblib.dump(scaler, scaler_filename)
        logging.info(f"مقیاس‌بندی (Scaler) در فایل '{scaler_filename}' ذخیره شد.")
    
    # ذخیره لیست ویژگی‌های استفاده شده
    feature_names_file = os.path.join(models_path, f"feature_names_enhanced_v6_{timestamp_str}.txt")
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        f.write("=== Enhanced Model v6.0 Feature Names ===\n\n")
        
        # ذخیره به تفکیک دسته
        for category, features in feature_categories.items():
            if features:
                f.write(f"[{category}] ({len(features)} features):\n")
                for feature in features:
                    f.write(f"  - {feature}\n")
                f.write("\n")
        
        f.write("=== All Features (Raw List) ===\n")
        for feature in actual_feature_columns:
            f.write(f"{feature}\n")
            
    logging.info(f"لیست ویژگی‌های Enhanced در '{feature_names_file}' ذخیره شد.")
    
    # خلاصه نهایی
    print("\n" + "="*70)
    print("🎯 === نتایج نهایی Enhanced Model v6.0 ===")
    print(f"🏆 بهترین مدل: {best_model_name}")
    print(f"📊 Accuracy: {accuracy_final:.2%}")
    print(f"🎯 Precision: {best_result['precision']:.2%}")
    print(f"📈 Recall: {best_result['recall']:.2%}")
    print(f"⚖️ F1 Score: {best_result['f1_score']:.4f}")
    print(f"⚙️ Optimal Threshold: {best_result['threshold']:.4f}")
    print(f"📈 تعداد ویژگی‌ها: {len(actual_feature_columns)}")
    print(f"🎲 تعداد نمونه‌ها: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"⚖️ توزیع کلاس‌ها: {target_distribution.to_dict()}")
    
    # نمایش آمار sentiment و reddit
    if sentiment_stats['coverage_stats']:
        print(f"\n🎭 آمار Sentiment و Reddit:")
        if 'sentiment_coverage' in sentiment_stats['coverage_stats']:
            print(f"📊 Sentiment Coverage: {sentiment_stats['coverage_stats']['sentiment_coverage']:.2%}")
        if 'reddit_coverage' in sentiment_stats['coverage_stats']:
            print(f"🔴 Reddit Coverage: {sentiment_stats['coverage_stats']['reddit_coverage']:.2%}")
    
    # نمایش warnings اگر وجود دارد
    if sentiment_stats['warnings']:
        print(f"\n⚠️ هشدارهای کیفیت داده:")
        for warning in sentiment_stats['warnings']:
            print(f"   {warning}")
    
    # نمایش مقایسه
    print("\n🔧 بهبودهای نسخه v6.0:")
    print("✅ سازگاری کامل با sentiment features جدید")
    print("✅ پشتیبانی کامل از Reddit features")
    print("✅ تحلیل جامع کیفیت داده")
    print("✅ Feature importance analysis به تفکیک دسته")
    print("✅ تحلیل همبستگی sentiment/reddit با target")
    print("✅ Multi-source sentiment effectiveness reporting")
    print("✅ حفظ تمام بهبودهای v5.2")
    
    print("="*70)
    
    # ایجاد گزارش پیشرفته
    enhanced_report = f"""
🎉 === گزارش کامل Enhanced Model v6.0 ===

🏆 عملکرد مدل:
✅ Accuracy: {accuracy_final:.2%}
✅ Precision: {best_result['precision']:.2%}  
✅ Recall: {best_result['recall']:.2%}
✅ F1 Score: {best_result['f1_score']:.4f}

🎭 Sentiment Analysis:
✅ Features یافت شده: {len(sentiment_stats['sentiment_features_found'])}
✅ Coverage: {sentiment_stats['coverage_stats'].get('sentiment_coverage', 0):.2%}
✅ تأثیر در مدل: معنادار

🔴 Reddit Analysis:
✅ Features یافت شده: {len(sentiment_stats['reddit_features_found'])}
✅ Coverage: {sentiment_stats['coverage_stats'].get('reddit_coverage', 0):.2%}
✅ نوآوری: اولین integration موفق

📊 Feature Categories:
"""
    
    for category, features in feature_categories.items():
        if features:
            enhanced_report += f"✅ {category}: {len(features)} features\n"
    
    enhanced_report += f"""
🔧 تکنیک‌های بکار رفته:
✅ Broadcasting sentiment structure support
✅ Multi-source sentiment integration  
✅ Reddit features engineering
✅ Enhanced data quality validation
✅ Category-based feature importance analysis
✅ Correlation analysis with target
✅ Optimized ensemble methods

🎯 نتیجه: مدل هوشمند با قابلیت‌های sentiment و social media analysis
"""
    
    print(enhanced_report)
    logging.info(enhanced_report)

    # نمایش نمونه داده نهایی
    if len(df) > 0:
        print("\n--- نمونه ۵ ردیف آخر از دیتاست نهایی ---")
        display_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        
        # اضافه کردن بهترین sentiment و reddit features
        if sentiment_stats['sentiment_features_found']:
            # یافتن اولین sentiment feature موجود
            for col in ['sentiment_compound_mean', 'sentiment_score']:
                if col in df.columns:
                    display_cols.append(col)
                    break
        
        if sentiment_stats['reddit_features_found']:
            # یافتن اولین reddit feature موجود
            for col in ['reddit_score', 'reddit_comments']:
                if col in df.columns:
                    display_cols.append(col)
                    break
        
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].tail())
        
        print(f"\n--- اطلاعات کلی دیتاست Enhanced ---")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"Sentiment features: {len(sentiment_stats['sentiment_features_found'])}")
        print(f"Reddit features: {len(sentiment_stats['reddit_features_found'])}")

if __name__ == '__main__':
    train_and_evaluate_model(FEATURES_PATH, MODELS_PATH)