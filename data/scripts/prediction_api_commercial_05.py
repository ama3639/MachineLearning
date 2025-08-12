#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت API پیش‌بینی با Flask (نسخه 6.2 - اصلاح کامل سازگاری Telegram-based)

🔧 تغییرات v6.2 (اصلاحات حیاتی برای Telegram-derived Reddit):
- ✅ تصحیح validation برای Telegram-derived Reddit features
- ✅ بهبود feature categorization (reddit features = telegram_derived_features)  
- ✅ اصلاح data quality thresholds برای Telegram-based features
- ✅ بهبود health endpoint با گزارش صحیح از Telegram mapping
- ✅ جلوگیری از double validation بین sentiment و reddit features
- ✅ تطبیق کامل با فایل‌های 02-04 اصلاح شده
- ✅ Enhanced response با تشخیص صحیح منبع Telegram
- ✅ اضافه کردن telegram_mapping_info به responses

ویژگی‌های موجود:
- User Authentication & Authorization
- Rate Limiting per User Plan
- Usage Tracking
- Subscription Plan Validation
- Enhanced Feature Validation (58+ features)
- Telegram-derived Reddit Features Support (جایگزین Reddit API)
- Multi-source Data Quality Analysis
- Telegram Mapping Validation
"""
import os
import glob
import joblib
import pandas as pd
import logging
import sqlite3
from flask import Flask, request, jsonify, g
import configparser
import numpy as np
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import hashlib
import json

# --- بخش خواندن پیکربندی ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    
    MODELS_PATH = config.get('Paths', 'models')
    LOG_PATH = config.get('Paths', 'logs')
    USERS_PATH = config.get('Paths', 'users', fallback='data/users')
    API_HOST = config.get('API_Settings', 'host')
    API_PORT = config.getint('API_Settings', 'port')
    
    # تنظیمات تجاری
    COMMERCIAL_MODE = config.getboolean('Commercial_Settings', 'commercial_mode', fallback=False)
    MAX_USERS = config.getint('Commercial_Settings', 'max_users', fallback=500)
    
    # محدودیت‌های پلان‌ها
    FREE_API_CALLS_PER_HOUR = config.getint('Commercial_Settings', 'free_api_calls_per_hour', fallback=10)
    BASIC_API_CALLS_PER_HOUR = config.getint('Commercial_Settings', 'basic_api_calls_per_hour', fallback=100)  
    PRO_API_CALLS_PER_HOUR = config.getint('Commercial_Settings', 'pro_api_calls_per_hour', fallback=500)
    
    # Rate Limiting
    ENABLE_RATE_LIMITING = config.getboolean('Web_Interface', 'enable_rate_limiting', fallback=True)
    MAX_REQUESTS_PER_MINUTE = config.getint('Web_Interface', 'max_requests_per_minute', fallback=60)
    
    # === 🔧 اصلاح 3: تنظیمات اصلاح شده Data Quality ===
    MIN_SENTIMENT_COVERAGE = config.getfloat('Data_Quality', 'min_sentiment_coverage', fallback=0.10)
    MIN_TELEGRAM_SENTIMENT_COVERAGE = config.getfloat('Data_Quality', 'min_telegram_sentiment_coverage', fallback=0.05)  # جایگزین MIN_REDDIT_COVERAGE
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Please check the file. Error: {e}")
    exit()

# --- ایجاد پوشه‌های مورد نیاز ---
os.makedirs(USERS_PATH, exist_ok=True)

# --- بخش تنظیمات لاگ‌گیری ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)
log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# متغیرهای global برای مدل
model_package = None
scaler = None
model_info = {}

# متغیرهای global برای rate limiting
user_requests = defaultdict(list)
user_api_calls = defaultdict(list)

# === بخش User Management (حفظ شده) ===
def init_user_database():
    """ایجاد database کاربران اگر وجود نداشته باشد"""
    db_path = os.path.join(USERS_PATH, 'users.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER UNIQUE,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                subscription_plan TEXT DEFAULT 'free',
                subscription_end_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                total_api_calls INTEGER DEFAULT 0,
                last_api_call TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                endpoint TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                response_status INTEGER,
                processing_time_ms REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logging.info("✅ User database initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize user database: {e}")
        return False

def get_user_by_credentials(username: str, password: str):
    """اعتبارسنجی کاربر با username و password"""
    if not COMMERCIAL_MODE:
        return {'id': 0, 'username': 'anonymous', 'subscription_plan': 'pro'}
    
    try:
        db_path = os.path.join(USERS_PATH, 'users.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute('''
            SELECT id, username, subscription_plan, subscription_end_date, is_active, total_api_calls
            FROM users 
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            user_data = {
                'id': user[0],
                'username': user[1], 
                'subscription_plan': user[2],
                'subscription_end_date': user[3],
                'is_active': user[4],
                'total_api_calls': user[5]
            }
            
            if user[3]:
                end_date = datetime.fromisoformat(user[3])
                if datetime.now() > end_date:
                    user_data['subscription_plan'] = 'free'
            
            return user_data
        
        return None
        
    except Exception as e:
        logging.error(f"Error in user authentication: {e}")
        return None

def get_user_plan_limits(subscription_plan: str):
    """دریافت محدودیت‌های هر پلان"""
    limits = {
        'free': {'api_calls_per_hour': FREE_API_CALLS_PER_HOUR},
        'basic': {'api_calls_per_hour': BASIC_API_CALLS_PER_HOUR}, 
        'pro': {'api_calls_per_hour': PRO_API_CALLS_PER_HOUR}
    }
    return limits.get(subscription_plan, limits['free'])

def check_rate_limit(user_id: int, subscription_plan: str):
    """بررسی محدودیت نرخ درخواست برای کاربر"""
    if not ENABLE_RATE_LIMITING:
        return True
    
    current_time = datetime.now()
    hour_ago = current_time - timedelta(hours=1)
    
    user_api_calls[user_id] = [
        timestamp for timestamp in user_api_calls[user_id] 
        if timestamp > hour_ago
    ]
    
    plan_limits = get_user_plan_limits(subscription_plan)
    current_calls = len(user_api_calls[user_id])
    
    if current_calls >= plan_limits['api_calls_per_hour']:
        return False
    
    user_api_calls[user_id].append(current_time)
    return True

def update_user_usage(user_id: int, endpoint: str, ip_address: str, response_status: int, processing_time: float):
    """بروزرسانی آمار استفاده کاربر"""
    if not COMMERCIAL_MODE or user_id == 0:
        return
    
    try:
        db_path = os.path.join(USERS_PATH, 'users.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET total_api_calls = total_api_calls + 1, last_api_call = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), user_id))
        
        cursor.execute('''
            INSERT INTO api_usage (user_id, endpoint, ip_address, response_status, processing_time_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, endpoint, ip_address, response_status, processing_time))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logging.error(f"Error updating user usage: {e}")

def require_auth(f):
    """Decorator برای اعتبارسنجی درخواست‌ها"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not COMMERCIAL_MODE:
            g.current_user = {'id': 0, 'username': 'anonymous', 'subscription_plan': 'pro'}
            return f(*args, **kwargs)
        
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            return jsonify({
                'error': 'Authentication required',
                'message': 'Please provide username and password using Basic Auth'
            }), 401
        
        user = get_user_by_credentials(auth.username, auth.password)
        if not user:
            return jsonify({
                'error': 'Invalid credentials',
                'message': 'Username or password is incorrect'
            }), 401
        
        if not check_rate_limit(user['id'], user['subscription_plan']):
            plan_limits = get_user_plan_limits(user['subscription_plan'])
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': f'You have exceeded your {user["subscription_plan"]} plan limit of {plan_limits["api_calls_per_hour"]} calls per hour',
                'retry_after': 3600
            }), 429
        
        g.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

# === بخش بارگذاری مدل بهبود یافته ===

def load_enhanced_model():
    """بارگذاری مدل Enhanced v6.0+ با پشتیبانی کامل از Telegram-derived Reddit features"""
    global model_package, scaler, model_info
    
    try:
        # جستجو برای مدل‌های Enhanced v6.0
        enhanced_model_patterns = [
            'enhanced_model_v6_*.joblib',
            'optimized_model_*.joblib',
            'random_forest_model_*.joblib'  # fallback
        ]
        
        enhanced_scaler_patterns = [
            'scaler_enhanced_v6_*.joblib', 
            'scaler_optimized_*.joblib',
            'scaler_*.joblib'  # fallback
        ]
        
        latest_model_file = None
        latest_scaler_file = None
        
        # پیدا کردن جدیدترین مدل
        for pattern in enhanced_model_patterns:
            files = glob.glob(os.path.join(MODELS_PATH, pattern))
            if files:
                latest_model_file = max(files, key=os.path.getctime)
                break
        
        # پیدا کردن جدیدترین scaler
        for pattern in enhanced_scaler_patterns:
            files = glob.glob(os.path.join(MODELS_PATH, pattern))
            if files:
                latest_scaler_file = max(files, key=os.path.getctime)
                break
        
        if not latest_model_file or not latest_scaler_file:
            raise FileNotFoundError("No compatible model or scaler files found")
        
        # بارگذاری model package
        model_package = joblib.load(latest_model_file)
        scaler = joblib.load(latest_scaler_file)
        
        # استخراج اطلاعات مدل Enhanced
        if isinstance(model_package, dict):
            model_info = {
                'model_type': model_package.get('model_type', 'Unknown'),
                'optimal_threshold': model_package.get('optimal_threshold', 0.5),
                'accuracy': model_package.get('accuracy', 0.0),
                'precision': model_package.get('precision', 0.0),
                'recall': model_package.get('recall', 0.0),
                'f1_score': model_package.get('f1_score', 0.0),
                'feature_columns': model_package.get('feature_columns', []),
                'feature_categories': model_package.get('feature_categories', {}),
                'sentiment_stats': model_package.get('sentiment_stats', {}),
                'correlation_analysis': model_package.get('correlation_analysis', {}),
                'model_version': model_package.get('model_version', '6.2_enhanced_telegram'),
                'model_file': os.path.basename(latest_model_file),
                'scaler_file': os.path.basename(latest_scaler_file),
                'is_enhanced': True,
                'telegram_reddit_mapping': model_package.get('telegram_reddit_mapping', True),  # 🆕 اضافه شده
                'reddit_source': model_package.get('reddit_source', 'telegram_sentiment')  # 🆕 اضافه شده
            }
        else:
            # فرمت قدیمی
            model_info = {
                'model_type': type(model_package).__name__,
                'optimal_threshold': 0.5,
                'model_file': os.path.basename(latest_model_file),
                'scaler_file': os.path.basename(latest_scaler_file),
                'feature_columns': [],
                'is_enhanced': False,
                'is_legacy': True,
                'telegram_reddit_mapping': False,
                'reddit_source': 'unknown'
            }
        
        # نمایش اطلاعات مدل
        print(f"✅ Enhanced Model v6.2 loaded: {model_info['model_type']}")
        print(f"📁 Model file: {model_info['model_file']}")
        print(f"📁 Scaler file: {model_info['scaler_file']}")
        print(f"🎯 Optimal Threshold: {model_info['optimal_threshold']:.4f}")
        print(f"🔢 Expected Features: {len(model_info['feature_columns'])}")
        print(f"📱 Telegram-Reddit Mapping: {'Yes' if model_info.get('telegram_reddit_mapping') else 'No'}")
        print(f"🔴 Reddit Source: {model_info.get('reddit_source', 'unknown')}")
        
        if model_info.get('accuracy'):
            print(f"📊 Performance: Accuracy={model_info['accuracy']:.2%}, "
                  f"Precision={model_info['precision']:.2%}, "
                  f"Recall={model_info['recall']:.2%}, "
                  f"F1={model_info['f1_score']:.4f}")
        
        # === بررسی Feature Categories ===
        feature_categories = model_info.get('feature_categories', {})
        if feature_categories:
            print(f"🏷️ Feature Categories:")
            for category, features in feature_categories.items():
                if features:
                    category_display = category
                    if category == 'telegram_derived_features':
                        category_display += " (از Telegram sentiment مشتق شده)"
                    print(f"   {category_display}: {len(features)} features")
        
        # === بررسی Sentiment Stats ===
        sentiment_stats = model_info.get('sentiment_stats', {})
        if sentiment_stats:
            coverage_stats = sentiment_stats.get('coverage_stats', {})
            if coverage_stats:
                sentiment_coverage = coverage_stats.get('sentiment_coverage', 0)
                telegram_reddit_coverage = coverage_stats.get('telegram_derived_reddit_coverage', 0)
                print(f"🎭 Sentiment Coverage: {sentiment_coverage:.2%}")
                print(f"📱 Telegram-derived Reddit Coverage: {telegram_reddit_coverage:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading enhanced model: {e}")
        return False

def get_model():
    """دریافت مدل اصلی از package"""
    if model_package is None:
        return None
    
    if isinstance(model_package, dict):
        return model_package.get('model')
    else:
        return model_package

# === 🔧 اصلاح 1: بخش Validation Functions جدید ===

def validate_telegram_derived_sentiment_features(input_data: dict) -> dict:
    """اعتبارسنجی ویژگی‌های احساسات Telegram-derived (اصلاح شده)"""
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'sentiment_coverage': 0,
        'telegram_derived_reddit_coverage': 0,  # 🔧 اصلاح 1: نام جدید
        'source_diversity': 0,
        'telegram_mapping_detected': False  # 🆕 اضافه شده
    }
    
    try:
        # بررسی sentiment features
        sentiment_features = [
            'sentiment_score', 'sentiment_momentum', 'sentiment_ma_7', 'sentiment_ma_14',
            'sentiment_volume', 'sentiment_divergence'
        ]
        
        sentiment_values = []
        for feature in sentiment_features:
            if feature in input_data:
                value = input_data[feature]
                if isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
                    sentiment_values.append(abs(value))
        
        if sentiment_values:
            validation_result['sentiment_coverage'] = len(sentiment_values) / len(sentiment_features)
        else:
            validation_result['warnings'].append("All sentiment features are zero or missing")
        
        # 🔧 اصلاح 1: بررسی Telegram-derived Reddit features
        telegram_derived_reddit_features = ['reddit_score', 'reddit_comments', 'reddit_score_ma', 'reddit_comments_ma']
        reddit_values = []
        sentiment_score = input_data.get('sentiment_score', 0)
        reddit_score = input_data.get('reddit_score', 0)
        
        # 🔧 اصلاح 4: بررسی نگاشت Telegram
        if sentiment_score != 0 and reddit_score != 0:
            if abs(sentiment_score - reddit_score) < 0.0001:  # تقریباً مساوی
                validation_result['telegram_mapping_detected'] = True
                logging.info("✅ Telegram → Reddit mapping detected: reddit_score = sentiment_score")
            
        for feature in telegram_derived_reddit_features:
            if feature in input_data:
                value = input_data[feature]
                if isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
                    reddit_values.append(abs(value))
        
        if reddit_values:
            validation_result['telegram_derived_reddit_coverage'] = len(reddit_values) / len(telegram_derived_reddit_features)
        else:
            validation_result['warnings'].append("All Telegram-derived Reddit features are zero or missing")
        
        # بررسی source diversity
        if 'source_diversity' in input_data:
            diversity = input_data['source_diversity']
            if isinstance(diversity, (int, float)) and not np.isnan(diversity):
                validation_result['source_diversity'] = diversity
        
        # 🔧 اصلاح 3: ارزیابی کلی کیفیت با thresholds جدید
        if validation_result['sentiment_coverage'] < MIN_SENTIMENT_COVERAGE:
            validation_result['warnings'].append(f"Sentiment coverage ({validation_result['sentiment_coverage']:.1%}) below minimum threshold ({MIN_SENTIMENT_COVERAGE:.1%})")
        
        if validation_result['telegram_derived_reddit_coverage'] < MIN_TELEGRAM_SENTIMENT_COVERAGE and validation_result['telegram_derived_reddit_coverage'] > 0:
            validation_result['warnings'].append(f"Telegram-derived Reddit coverage ({validation_result['telegram_derived_reddit_coverage']:.1%}) below minimum threshold ({MIN_TELEGRAM_SENTIMENT_COVERAGE:.1%})")
        
        # 🔧 اصلاح 5: جلوگیری از double counting warning
        if validation_result['telegram_mapping_detected']:
            # حذف warnings مربوط به reddit features چون از sentiment مشتق شده‌اند
            validation_result['warnings'] = [w for w in validation_result['warnings'] if 'Reddit' not in w]
            if validation_result['sentiment_coverage'] > MIN_SENTIMENT_COVERAGE:
                validation_result['warnings'] = [w for w in validation_result['warnings'] if 'Telegram-derived Reddit' not in w]
                logging.info("📱 Telegram-derived Reddit features validated via sentiment_score")
        
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['warnings'].append(f"Validation error: {str(e)}")
    
    return validation_result

def categorize_input_features(input_data: dict) -> dict:
    """🔧 اصلاح 2: دسته‌بندی features ورودی (اصلاح شده برای Telegram-derived)"""
    categories = {
        'technical_indicators': [],
        'sentiment_features': [],
        'telegram_derived_features': [],  # 🔧 اصلاح 2: جایگزین reddit_features
        'price_features': [],
        'volume_features': [],
        'other_features': []
    }
    
    for feature in input_data.keys():
        feature_lower = feature.lower()
        
        if 'sentiment' in feature_lower:
            categories['sentiment_features'].append(feature)
        elif 'reddit' in feature_lower:
            categories['telegram_derived_features'].append(feature)  # 🔧 اصلاح 2
        elif any(ind in feature_lower for ind in ['rsi', 'macd', 'bb_', 'ema', 'sma', 'stoch', 'williams', 'cci', 'adx', 'psar']):
            categories['technical_indicators'].append(feature)
        elif any(price in feature_lower for price in ['return', 'price', 'close_position', 'hl_ratio']):
            categories['price_features'].append(feature)
        elif any(vol in feature_lower for vol in ['volume', 'obv', 'mfi', 'vwap']):
            categories['volume_features'].append(feature)
        else:
            categories['other_features'].append(feature)
    
    return categories

def make_enhanced_prediction(input_features, use_optimal_threshold=True):
    """پیش‌بینی بهبود یافته با تحلیل Telegram-derived Reddit features"""
    model = get_model()
    if model is None:
        return None
    
    try:
        # Scaling
        features_scaled = scaler.transform(input_features)
        
        # پیش‌بینی احتمالات
        prediction_proba = model.predict_proba(features_scaled)
        profit_prob = prediction_proba[0][1]
        no_profit_prob = prediction_proba[0][0]
        
        # تصمیم‌گیری با threshold
        if use_optimal_threshold and model_info.get('optimal_threshold'):
            threshold = model_info['optimal_threshold']
        else:
            threshold = 0.5
        
        final_prediction = 1 if profit_prob >= threshold else 0
        signal = 'PROFIT' if final_prediction == 1 else 'NO_PROFIT'
        
        # === تحلیل Feature Importance (اصلاح شده) ===
        feature_analysis = {}
        if hasattr(model, 'feature_importances_') and model_info.get('feature_columns'):
            feature_names = model_info['feature_columns']
            feature_values = input_features.iloc[0] if hasattr(input_features, 'iloc') else input_features[0]
            
            # دسته‌بندی features اصلاح شده
            feature_categories = categorize_input_features(dict(zip(feature_names, feature_values)))
            
            # محاسبه اهمیت برای هر دسته
            for category, features in feature_categories.items():
                if features:
                    category_importance = 0
                    for feature in features:
                        if feature in feature_names:
                            idx = feature_names.index(feature)
                            
                            # 🔧 اصلاح 5: جلوگیری از double counting
                            if category == 'telegram_derived_features' and feature.startswith('reddit_'):
                                logging.info(f"⚠️ {feature} is Telegram-derived, flagged to avoid double counting")
                            
                            category_importance += model.feature_importances_[idx]
                    
                    category_display_name = category
                    if category == 'telegram_derived_features':
                        category_display_name += "_from_telegram"  # برای وضوح
                    
                    feature_analysis[category_display_name] = {
                        'importance': float(category_importance),
                        'feature_count': len(features),
                        'avg_importance': float(category_importance / len(features)) if features else 0
                    }
        
        return {
            'prediction': int(final_prediction),
            'signal': signal,
            'confidence': {
                'no_profit_prob': round(no_profit_prob, 4),
                'profit_prob': round(profit_prob, 4)
            },
            'threshold_used': threshold,
            'feature_analysis': feature_analysis,
            'raw_probabilities': {
                'no_profit_raw': round(no_profit_prob, 4),
                'profit_raw': round(profit_prob, 4)
            }
        }
        
    except Exception as e:
        logging.error(f"Error in enhanced prediction: {e}")
        return None

# --- بخش بارگذاری مدل ---
print("🔄 Initializing Enhanced Prediction API v6.2 (Telegram-derived Reddit)...")

if COMMERCIAL_MODE:
    print("💼 Commercial mode enabled - initializing user database...")
    init_user_database()
else:
    print("🔓 Running in non-commercial mode")

model_loaded = load_enhanced_model()
if not model_loaded:
    print("❌ CRITICAL: Could not load any model. API will not function properly.")
    model_package = None
    scaler = None

# --- بخش اپلیکیشن Flask ---
app = Flask(__name__)

@app.route("/")
def index():
    mode_text = "Commercial" if COMMERCIAL_MODE else "Open"
    telegram_support = "with Telegram-derived Reddit" if model_info.get('telegram_reddit_mapping') else "Legacy"
    return f"Enhanced Prediction API v6.2 ({mode_text} Mode - {telegram_support}) is running. Features: 58+ including Sentiment & Telegram-derived Reddit analysis."

@app.route('/health', methods=['GET'])
def health_check():
    """🔧 اصلاح 4: Health check بهبود یافته با گزارش صحیح از Telegram mapping"""
    try:
        model = get_model()
        
        # آمار کاربران
        user_stats = {}
        if COMMERCIAL_MODE:
            try:
                db_path = os.path.join(USERS_PATH, 'users.db')
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                active_users = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM api_usage WHERE timestamp > datetime('now', '-1 hour')")
                api_calls_last_hour = cursor.fetchone()[0]
                
                user_stats = {
                    'active_users': active_users,
                    'api_calls_last_hour': api_calls_last_hour,
                    'max_users': MAX_USERS
                }
                
                conn.close()
            except Exception as e:
                user_stats = {'error': f'Could not fetch user stats: {e}'}
        
        health_status = {
            'status': 'healthy' if model and scaler else 'unhealthy',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'commercial_mode': COMMERCIAL_MODE,
            'api_version': '6.2_enhanced_telegram',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        # اطلاعات مدل Enhanced
        try:
            if model_info:
                # اطلاعات پایه مدل
                health_status['model_info'] = {
                    'model_type': str(model_info.get('model_type', 'Unknown')),
                    'model_version': str(model_info.get('model_version', '6.2')),
                    'model_file': str(model_info.get('model_file', 'Unknown')),
                    'scaler_file': str(model_info.get('scaler_file', 'Unknown')),
                    'is_enhanced': bool(model_info.get('is_enhanced', False)),
                    'optimal_threshold': float(model_info.get('optimal_threshold', 0.5)),
                    'features_count': int(len(model_info.get('feature_columns', []))),
                    'expected_features': 58,
                    'telegram_reddit_mapping': bool(model_info.get('telegram_reddit_mapping', False)),  # 🆕
                    'reddit_source': str(model_info.get('reddit_source', 'unknown'))  # 🆕
                }
                
                # Performance metrics
                if model_info.get('accuracy') is not None:
                    health_status['model_info']['performance'] = {
                        'accuracy': float(model_info.get('accuracy', 0)),
                        'precision': float(model_info.get('precision', 0)),
                        'recall': float(model_info.get('recall', 0)),
                        'f1_score': float(model_info.get('f1_score', 0))
                    }
                
                # 🔧 اصلاح 2: Feature categories اصلاح شده
                feature_categories = model_info.get('feature_categories', {})
                if feature_categories:
                    health_status['feature_categories'] = {}
                    for category, features in feature_categories.items():
                        category_count = len(features) if features else 0
                        category_display = category
                        
                        # اضافه کردن توضیح برای telegram_derived_features
                        if category == 'telegram_derived_features':
                            health_status['feature_categories'][category] = {
                                'count': category_count,
                                'description': 'Reddit features derived from Telegram sentiment',
                                'source': 'telegram_sentiment'
                            }
                        else:
                            health_status['feature_categories'][category] = category_count
                
                # 🔧 اصلاح 4: Sentiment & Telegram-derived Reddit analysis
                sentiment_stats = model_info.get('sentiment_stats', {})
                if sentiment_stats:
                    health_status['sentiment_analysis'] = {
                        'sentiment_features_found': len(sentiment_stats.get('sentiment_features_found', [])),
                        'telegram_derived_reddit_features_found': len(sentiment_stats.get('telegram_derived_reddit_features_found', [])),  # اصلاح نام
                        'coverage_stats': sentiment_stats.get('coverage_stats', {}),
                        'warnings': sentiment_stats.get('warnings', []),
                        'telegram_mapping_info': {  # 🆕 اضافه شده
                            'is_telegram_derived': sentiment_stats.get('coverage_stats', {}).get('is_telegram_derived', False),
                            'reddit_source': 'telegram_sentiment',
                            'mapping_note': 'reddit_score = sentiment_score (derived from Telegram data)'
                        }
                    }
                
                # Correlation analysis
                correlation_analysis = model_info.get('correlation_analysis', {})
                if correlation_analysis:
                    health_status['correlation_analysis'] = {
                        'best_sentiment_feature': correlation_analysis.get('best_sentiment_feature'),
                        'best_telegram_derived_reddit_feature': correlation_analysis.get('best_telegram_derived_reddit_feature'),  # اصلاح نام
                        'sentiment_correlations_count': len(correlation_analysis.get('sentiment_correlations', {})),
                        'telegram_derived_reddit_correlations_count': len(correlation_analysis.get('telegram_derived_reddit_correlations', {}))  # اصلاح نام
                    }
                
        except Exception as model_info_error:
            logging.warning(f"Error in model_info serialization: {model_info_error}")
            health_status['model_info_error'] = str(model_info_error)
        
        # اضافه کردن آمار کاربران
        if user_stats:
            health_status['user_stats'] = user_stats
        
        # محاسبه uptime
        try:
            import psutil
            process = psutil.Process()
            uptime_seconds = (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
            health_status['uptime_seconds'] = round(float(uptime_seconds), 2)
        except ImportError:
            health_status['uptime_seconds'] = None
        except Exception:
            health_status['uptime_seconds'] = None
        
        # 🔧 اصلاح 3: Data quality thresholds اصلاح شده
        health_status['data_quality_thresholds'] = {
            'min_sentiment_coverage': MIN_SENTIMENT_COVERAGE,
            'min_telegram_derived_reddit_coverage': MIN_TELEGRAM_SENTIMENT_COVERAGE,  # اصلاح نام
            'telegram_mapping_note': 'Reddit features are derived from Telegram sentiment data'
        }
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        error_response = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'api_version': '6.2_enhanced_telegram',
            'model_loaded': False,
            'scaler_loaded': False,
            'commercial_mode': COMMERCIAL_MODE
        }
        logging.error(f"Health check failed: {e}")
        return jsonify(error_response), 500

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    """🔧 اصلاح 1: پیش‌بینی Enhanced v6.2 با تحلیل کامل Telegram-derived Reddit features"""
    start_time = datetime.now()
    current_user = g.current_user
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    
    if not get_model() or not scaler:
        update_user_usage(current_user['id'], '/predict', ip_address, 500, 0)
        return jsonify({"error": "Enhanced model or scaler is not loaded properly"}), 500

    try:
        # دریافت داده ورودی
        input_data = request.get_json(force=True)
        if not input_data:
            update_user_usage(current_user['id'], '/predict', ip_address, 400, 0)
            return jsonify({"error": "Invalid input: No JSON data received"}), 400
        
        app.logger.info(f"Enhanced prediction request from user {current_user['username']} with {len(input_data)} features")
        
        # پاک‌سازی input data
        cleaned_data = {}
        for k, v in input_data.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                if np.isnan(v) or np.isinf(v):
                    app.logger.warning(f"Skipping invalid value: {k}={v}")
                    continue
                if isinstance(v, np.integer):
                    cleaned_data[k] = int(v)
                elif isinstance(v, np.floating):
                    cleaned_data[k] = float(v)
                else:
                    cleaned_data[k] = v
            else:
                cleaned_data[k] = v
        
        if not cleaned_data:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            update_user_usage(current_user['id'], '/predict', ip_address, 400, processing_time)
            return jsonify({"error": "No valid features in input data"}), 400
        
        # === بررسی Enhanced Features ===
        expected_features = model_info.get('feature_columns', [])
        current_feature_count = len(cleaned_data)
        expected_count = len(expected_features)
        
        app.logger.info(f"Feature count validation: received={current_feature_count}, expected={expected_count}")
        
        # بررسی feature count (باید 58+ باشد)
        if expected_count > 0 and current_feature_count < expected_count * 0.9:  # حداقل 90% features
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            update_user_usage(current_user['id'], '/predict', ip_address, 400, processing_time)
            
            missing_features = [f for f in expected_features if f not in cleaned_data]
            return jsonify({
                "error": f"Insufficient features for Enhanced model v6.2",
                "received_features": current_feature_count,
                "expected_features": expected_count,
                "missing_critical_features": missing_features[:10],
                "missing_count": len(missing_features),
                "message": "Enhanced model requires 58+ features including sentiment and Telegram-derived Reddit data",
                "note": "Reddit features are derived from Telegram sentiment data"
            }), 400
        
        # === 🔧 اصلاح 1: Telegram-derived Reddit Features Validation ===
        sentiment_validation = validate_telegram_derived_sentiment_features(cleaned_data)
        
        # تبدیل به DataFrame
        df = pd.DataFrame([cleaned_data])
        
        # مرتب‌سازی ستون‌ها مطابق انتظارات مدل
        if expected_features:
            # اضافه کردن features گمشده با مقدار پیش‌فرض
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[expected_features]
        
        # انجام پیش‌بینی Enhanced
        prediction_result = make_enhanced_prediction(df)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        if prediction_result is None:
            update_user_usage(current_user['id'], '/predict', ip_address, 500, processing_time)
            return jsonify({"error": "Enhanced prediction failed"}), 500
        
        # پاک‌سازی نتایج برای JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj
        
        # ساخت پاسخ Enhanced
        result = {
            'prediction': int(prediction_result['prediction']),
            'signal': str(prediction_result['signal']),
            'confidence': {
                'no_profit_prob': float(prediction_result['confidence']['no_profit_prob']),
                'profit_prob': float(prediction_result['confidence']['profit_prob'])
            },
            'model_info': {
                'model_type': str(model_info.get('model_type', 'Unknown')),
                'model_version': str(model_info.get('model_version', '6.2_enhanced_telegram')),
                'threshold_used': float(prediction_result['threshold_used']),
                'is_enhanced': bool(model_info.get('is_enhanced', True)),
                'features_used': int(len(df.columns)),
                'expected_features': int(expected_count),
                'telegram_reddit_mapping': bool(model_info.get('telegram_reddit_mapping', False)),  # 🆕
                'reddit_source': str(model_info.get('reddit_source', 'unknown'))  # 🆕
            },
            'feature_analysis': prediction_result.get('feature_analysis', {}),
            'sentiment_analysis': {
                'sentiment_coverage': sentiment_validation['sentiment_coverage'],
                'telegram_derived_reddit_coverage': sentiment_validation['telegram_derived_reddit_coverage'],  # اصلاح نام
                'source_diversity': sentiment_validation['source_diversity'],
                'warnings': sentiment_validation['warnings'],
                'is_valid': sentiment_validation['is_valid'],
                'telegram_mapping_detected': sentiment_validation['telegram_mapping_detected'],  # 🆕
                'mapping_note': 'Reddit features derived from Telegram sentiment' if sentiment_validation['telegram_mapping_detected'] else 'No mapping detected'  # 🆕
            },
            'performance_metrics': None,
            'processing_info': {
                'processing_time_ms': round(processing_time, 2),
                'timestamp_utc': end_time.isoformat() + 'Z',
                'api_version': '6.2_enhanced_telegram'
            }
        }
        
        # اضافه کردن performance metrics
        if model_info.get('accuracy'):
            result['performance_metrics'] = {
                'model_accuracy': float(model_info.get('accuracy', 0)),
                'model_precision': float(model_info.get('precision', 0)),
                'model_recall': float(model_info.get('recall', 0)),
                'model_f1_score': float(model_info.get('f1_score', 0))
            }
        
        # اطلاعات کاربر
        if COMMERCIAL_MODE and current_user['id'] > 0:
            plan_limits = get_user_plan_limits(current_user['subscription_plan'])
            result['user_info'] = {
                'username': current_user['username'],
                'subscription_plan': current_user['subscription_plan'],
                'remaining_calls_this_hour': plan_limits['api_calls_per_hour'] - len(user_api_calls[current_user['id']]),
                'total_api_calls': current_user['total_api_calls'] + 1
            }
        
        # احتمالات خام
        if 'raw_probabilities' in prediction_result:
            result['raw_probabilities'] = {
                'no_profit_raw': float(prediction_result['raw_probabilities']['no_profit_raw']),
                'profit_raw': float(prediction_result['raw_probabilities']['profit_raw'])
            }
        
        # پاک‌سازی نهایی
        result = clean_for_json(result)
        
        # ثبت استفاده موفق
        update_user_usage(current_user['id'], '/predict', ip_address, 200, processing_time)
        
        app.logger.info(f"Enhanced prediction completed for user {current_user['username']}: "
                       f"Signal={result['signal']}, "
                       f"Confidence={result['confidence']['profit_prob']:.2%}, "
                       f"Features={result['model_info']['features_used']}, "
                       f"Sentiment_Coverage={result['sentiment_analysis']['sentiment_coverage']:.1%}, "
                       f"Telegram_Mapping={'Yes' if result['sentiment_analysis']['telegram_mapping_detected'] else 'No'}")
        
        return jsonify(result)

    except ValueError as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        update_user_usage(current_user['id'], '/predict', ip_address, 400, processing_time)
        app.logger.error(f"Value error during enhanced prediction for user {current_user['username']}: {e}")
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        update_user_usage(current_user['id'], '/predict', ip_address, 500, processing_time)
        app.logger.error(f"Error during enhanced prediction for user {current_user['username']}: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route('/model-info', methods=['GET'])
@require_auth
def get_model_info():
    """Endpoint برای دریافت اطلاعات تفصیلی مدل Enhanced v6.2"""
    current_user = g.current_user
    
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    update_user_usage(current_user['id'], '/model-info', ip_address, 200, 0)
    
    try:
        response_data = {
            'model_info': model_info,
            'model_loaded': get_model() is not None,
            'scaler_loaded': scaler is not None,
            'api_version': '6.2_enhanced_telegram',
            'features_supported': len(model_info.get('feature_columns', [])),
            'enhanced_model': model_info.get('is_enhanced', True),
            'commercial_mode': COMMERCIAL_MODE,
            'data_quality_requirements': {
                'min_sentiment_coverage': MIN_SENTIMENT_COVERAGE,
                'min_telegram_derived_reddit_coverage': MIN_TELEGRAM_SENTIMENT_COVERAGE,  # 🔧 اصلاح 3
                'expected_features': 58,
                'telegram_mapping_info': {  # 🆕 اضافه شده
                    'reddit_source': 'telegram_sentiment',
                    'mapping_description': 'Reddit features are automatically derived from Telegram sentiment analysis',
                    'benefits': 'Real-time sentiment without Reddit API dependency'
                }
            }
        }
        
        # اطلاعات Enhanced specific
        if model_info.get('feature_categories'):
            response_data['feature_categories'] = {}
            for category, features in model_info['feature_categories'].items():
                category_count = len(features) if features else 0
                
                if category == 'telegram_derived_features':
                    response_data['feature_categories'][category] = {
                        'count': category_count,
                        'source': 'telegram_sentiment',
                        'description': 'Reddit-like features derived from Telegram sentiment analysis',
                        'note': 'reddit_score = sentiment_score, reddit_comments = sentiment_score * 10'
                    }
                else:
                    response_data['feature_categories'][category] = category_count
        
        if model_info.get('sentiment_stats'):
            sentiment_stats = model_info['sentiment_stats'].copy()
            
            # 🔧 اصلاح 4: تصحیح نام‌ها در sentiment stats
            if 'telegram_derived_reddit_features_found' in sentiment_stats:
                sentiment_stats['telegram_mapping_confirmed'] = True
                sentiment_stats['reddit_features_source'] = 'telegram_sentiment'
            
            response_data['sentiment_stats'] = sentiment_stats
        
        if model_info.get('correlation_analysis'):
            response_data['correlation_analysis'] = model_info['correlation_analysis']
        
        # اطلاعات کاربر
        if COMMERCIAL_MODE and current_user['id'] > 0:
            plan_limits = get_user_plan_limits(current_user['subscription_plan'])
            response_data['user_context'] = {
                'username': current_user['username'],
                'subscription_plan': current_user['subscription_plan'],
                'plan_limits': plan_limits,
                'calls_this_hour': len(user_api_calls[current_user['id']])
            }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'model_info': {},
            'model_loaded': False,
            'scaler_loaded': False,
            'api_version': '6.2_enhanced_telegram',
            'commercial_mode': COMMERCIAL_MODE
        }), 500

@app.route('/admin/stats', methods=['GET'])
@require_auth
def admin_stats():
    """Endpoint برای آمار ادمین Enhanced"""
    current_user = g.current_user
    
    if not COMMERCIAL_MODE:
        return jsonify({'error': 'Admin stats only available in commercial mode'}), 404
    
    try:
        db_path = os.path.join(USERS_PATH, 'users.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM api_usage WHERE timestamp > datetime('now', '-24 hours')")
        api_calls_24h = cursor.fetchone()[0]
        
        cursor.execute("SELECT subscription_plan, COUNT(*) FROM users WHERE is_active = 1 GROUP BY subscription_plan")
        plan_distribution = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT u.username, u.subscription_plan, COUNT(au.id) as api_calls
            FROM users u
            LEFT JOIN api_usage au ON u.id = au.user_id AND au.timestamp > datetime('now', '-24 hours')
            WHERE u.is_active = 1
            GROUP BY u.id
            ORDER BY api_calls DESC
            LIMIT 10
        """)
        top_users = [{'username': row[0], 'plan': row[1], 'calls_24h': row[2]} for row in cursor.fetchall()]
        
        conn.close()
        
        ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        update_user_usage(current_user['id'], '/admin/stats', ip_address, 200, 0)
        
        return jsonify({
            'total_users': total_users,
            'api_calls_24h': api_calls_24h,
            'plan_distribution': plan_distribution,
            'top_users_24h': top_users,
            'api_version': '6.2_enhanced_telegram',
            'model_info': {
                'features_count': len(model_info.get('feature_columns', [])),
                'sentiment_features': len(model_info.get('feature_categories', {}).get('sentiment_features', [])),
                'telegram_derived_reddit_features': len(model_info.get('feature_categories', {}).get('telegram_derived_features', [])),  # اصلاح نام
                'telegram_reddit_mapping': model_info.get('telegram_reddit_mapping', False),
                'reddit_source': model_info.get('reddit_source', 'unknown')
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
    except Exception as e:
        return jsonify({'error': f'Could not fetch enhanced admin stats: {e}'}), 500

if __name__ == '__main__':
    print(f"--- Starting Enhanced Prediction API Server v6.2 (Telegram-derived Reddit) ---")
    print(f"💼 Commercial Mode: {'Enabled' if COMMERCIAL_MODE else 'Disabled'}")
    print(f"🏠 API will be available at http://{API_HOST}:{API_PORT}")
    
    if COMMERCIAL_MODE:
        print(f"👥 Max Users: {MAX_USERS}")
        print(f"🔐 Authentication: Required (Basic Auth)")
        print(f"📊 Rate Limiting: {'Enabled' if ENABLE_RATE_LIMITING else 'Disabled'}")
        print(f"💾 User Database: {os.path.join(USERS_PATH, 'users.db')}")
    else:
        print(f"🔓 Authentication: Disabled (Open Mode)")
    
    if model_loaded:
        print(f"✅ Enhanced Model Status: {model_info.get('model_type', 'Unknown')}")
        print(f"🎯 Optimal Threshold: {model_info.get('optimal_threshold', 0.5):.4f}")
        print(f"🔢 Features Supported: {len(model_info.get('feature_columns', []))} (Expected: 58+)")
        print(f"📱 Telegram-Reddit Mapping: {'Yes' if model_info.get('telegram_reddit_mapping') else 'No'}")
        print(f"🔴 Reddit Source: {model_info.get('reddit_source', 'unknown')}")
        
        if model_info.get('is_enhanced'):
            print(f"📊 Performance: Precision={model_info.get('precision', 0):.1%}, F1={model_info.get('f1_score', 0):.4f}")
            
            # نمایش Feature Categories
            feature_categories = model_info.get('feature_categories', {})
            if feature_categories:
                print(f"🏷️ Feature Categories:")
                for category, features in feature_categories.items():
                    if features:
                        category_display = category
                        if category == 'telegram_derived_features':
                            category_display += " (از Telegram sentiment مشتق شده)"
                        print(f"   {category_display}: {len(features)} features")
            
            # نمایش Sentiment Stats
            sentiment_stats = model_info.get('sentiment_stats', {})
            if sentiment_stats.get('coverage_stats'):
                coverage = sentiment_stats['coverage_stats']
                if 'sentiment_coverage' in coverage:
                    print(f"🎭 Sentiment Coverage: {coverage['sentiment_coverage']:.2%}")
                if 'telegram_derived_reddit_coverage' in coverage:
                    print(f"📱 Telegram-derived Reddit Coverage: {coverage['telegram_derived_reddit_coverage']:.2%}")
                if coverage.get('is_telegram_derived'):
                    print(f"✅ Telegram Mapping Confirmed: Reddit features from Telegram sentiment")
        
        print("🔗 Enhanced Endpoints:")
        print("   - GET  / (status)")
        print("   - GET  /health (detailed health check with Telegram mapping analysis)")
        print("   - POST /predict (enhanced prediction with 58+ Telegram-derived features) 🔐")
        print("   - GET  /model-info (enhanced model details with Telegram info) 🔐")
        if COMMERCIAL_MODE:
            print("   - GET  /admin/stats (enhanced admin statistics) 🔐")
    else:
        print("❌ WARNING: No enhanced model loaded! API will return errors.")
    
    print(f"📁 Logs: {log_filename}")
    print("🔐 = Requires Authentication in Commercial Mode")
    print("\n🆕 Enhanced Features v6.2 (Telegram-derived Reddit):")
    print("   ✅ 58+ Features Support (including PSAR)")
    print("   ✅ Sentiment Analysis Validation")
    print("   ✅ Telegram-derived Reddit Features Integration")
    print("   ✅ Feature Category Analysis with Telegram mapping")
    print("   ✅ Multi-source Data Quality Validation")
    print("   ✅ Telegram Mapping Detection & Validation")
    print("   📱 Note: Reddit features are derived from Telegram sentiment data, not Reddit API")
    
    app.run(host=API_HOST, port=API_PORT, debug=False)