#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت آموزش و ارزیابی مدل (نسخه 5 - رفع خطای infinity و NaN)
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# بخش خواندن پیکربندی و تنظیمات لاگ‌گیری
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    FEATURES_PATH = config.get('Paths', 'features')
    MODELS_PATH = config.get('Paths', 'models')
    LOG_PATH = config.get('Paths', 'logs')
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
            X[col] = X[col].fillna(median_val)
        else:
            # اگر تمام مقادیر NaN باشند، با 0 پر کنیم
            X[col] = X[col].fillna(0)
    
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

def train_and_evaluate_model(features_path: str, models_path: str):
    logging.info("شروع گام ۳-ب: آموزش و ارزیابی مدل (نسخه 5 - با پاکسازی داده)...")
    
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
    
    # --- پاکسازی داده‌ها ---
    X, y = clean_data(X, y)
    
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
    
    # آموزش مدل
    logging.info("در حال آموزش مدل RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train_scaled, y_train)
    logging.info("آموزش مدل با موفقیت به پایان رسید.")
    
    # پیش‌بینی
    y_pred = model.predict(X_test_scaled)
    
    # ارزیابی مدل
    logging.info("--- نتایج ارزیابی عملکرد مدل ---")
    
    # دقت کلی
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"دقت کلی مدل (Accuracy): {accuracy:.2%}")
    print(f"\n✅ Accuracy: {accuracy:.2%}")
    
    # بررسی کلاس‌های موجود در test set
    unique_test_classes = sorted(np.unique(y_test))
    unique_pred_classes = sorted(np.unique(y_pred))
    
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
            y_test, y_pred, 
            target_names=target_names,
            labels=labels,
            zero_division=0
        )
        logging.info("Classification Report:\n" + report)
        print("\n📊 Classification Report:")
        print(report)
        
    except Exception as e:
        logging.warning(f"خطا در تولید classification report: {e}")
        # گزارش ساده
        report = classification_report(y_test, y_pred, zero_division=0)
        logging.info("Classification Report (Simple):\n" + report)
        print("\n📊 Classification Report:")
        print(report)
    
    # ماتریس درهم‌ریختگی
    try:
        cm = confusion_matrix(y_test, y_pred)
        logging.info("Confusion Matrix:\n" + str(cm))
        print("\n🔄 Confusion Matrix:")
        print(cm)
        
        # رسم نمودار
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plot_filename = os.path.join(models_path, f"confusion_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"نمودار ماتریس درهم‌ریختگی در '{plot_filename}' ذخیره شد.")
        
    except Exception as e:
        logging.warning(f"خطا در تولید confusion matrix: {e}")
    
    # Cross-validation برای بررسی پایداری مدل
    if len(unique_classes) == 2 and min_class_size >= 3:
        try:
            logging.info("در حال انجام Cross-Validation...")
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
            logging.info(f"CV Scores: {cv_scores}")
            logging.info(f"میانگین CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"🔄 Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as e:
            logging.warning(f"خطا در Cross-Validation: {e}")
    
    # اهمیت ویژگی‌ها
    try:
        # فقط ویژگی‌هایی که در X_train موجودند
        actual_feature_columns = X_train.columns.tolist()
        feature_importance = pd.DataFrame({
            'feature': actual_feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("🔝 Top 10 مهم‌ترین ویژگی‌ها:")
        for i, row in feature_importance.head(10).iterrows():
            logging.info(f"{row['feature']}: {row['importance']:.4f}")
            
        print("\n🔝 Top 5 مهم‌ترین ویژگی‌ها:")
        print(feature_importance.head().to_string(index=False))
        
    except Exception as e:
        logging.warning(f"خطا در محاسبه اهمیت ویژگی‌ها: {e}")
    
    # ذخیره مدل و scaler
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_filename = os.path.join(models_path, f"random_forest_model_{timestamp_str}.joblib")
    
    joblib.dump(model, model_filename)
    logging.info(f"مدل در فایل '{model_filename}' ذخیره شد.")
    
    if scaler is not None:
        scaler_filename = os.path.join(models_path, f"scaler_{timestamp_str}.joblib")
        joblib.dump(scaler, scaler_filename)
        logging.info(f"مقیاس‌بندی (Scaler) در فایل '{scaler_filename}' ذخیره شد.")
    
    # ذخیره لیست ویژگی‌های استفاده شده
    feature_names_file = os.path.join(models_path, f"feature_names_{timestamp_str}.txt")
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        for feature in actual_feature_columns:
            f.write(f"{feature}\n")
    logging.info(f"لیست ویژگی‌ها در '{feature_names_file}' ذخیره شد.")
    
    # خلاصه نهایی
    print("\n" + "="*60)
    print("🎯 خلاصه نتایج:")
    print(f"📊 دقت مدل: {accuracy:.2%}")
    print(f"📈 تعداد ویژگی‌ها: {len(actual_feature_columns)}")
    print(f"🎲 تعداد نمونه‌ها: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"⚖️ توزیع کلاس‌ها: {target_distribution.to_dict()}")
    print("="*60)

if __name__ == '__main__':
    train_and_evaluate_model(FEATURES_PATH, MODELS_PATH)
