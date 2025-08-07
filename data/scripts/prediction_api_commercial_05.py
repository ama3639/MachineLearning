#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت API پیش‌بینی با Flask (نسخه 6.0 - تجاری‌سازی ساده)

ویژگی‌های جدید:
- پشتیبانی از Optimized Model Package (model + optimal threshold)
- سازگاری با Ensemble Models (RandomForest + XGBoost)
- Enhanced Response با اطلاعات تفصیلی‌تر
- بهبود Health Check
- Fallback به مدل‌های قدیمی در صورت نیاز

ویژگی‌های تجاری جدید:
- User Authentication & Authorization
- Rate Limiting per User Plan
- Usage Tracking
- Subscription Plan Validation
- API Key Management (آماده برای آینده)
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
from datetime import datetime, timedelta  # 🔧 اضافه شد
from functools import wraps
from collections import defaultdict
import hashlib
import json

# --- بخش خواندن پیکربندی ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    
    # در سرور اوبونتو غیر فعال شوند
    MODELS_PATH = config.get('Paths', 'models')
    LOG_PATH = config.get('Paths', 'logs')
    USERS_PATH = config.get('Paths', 'users', fallback='data/users')
    API_HOST = config.get('API_Settings', 'host')
    API_PORT = config.getint('API_Settings', 'port') # .getint() برای خواندن عدد صحیح
    
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

# تنظیم لاگ برای Flask
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# متغیرهای global برای مدل
model_package = None
scaler = None
model_info = {}

# متغیرهای global برای rate limiting
user_requests = defaultdict(list)  # {user_id: [timestamp1, timestamp2, ...]}
user_api_calls = defaultdict(list)  # {user_id: [hour_timestamp1, hour_timestamp2, ...]}

# --- بخش User Management و Authentication ---

def init_user_database():
    """ایجاد database کاربران اگر وجود نداشته باشد"""
    db_path = os.path.join(USERS_PATH, 'users.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # جدول کاربران
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
        
        # جدول آمار API calls
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
        
        # جدول sessions (برای web interface در آینده)
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
        return {'id': 0, 'username': 'anonymous', 'subscription_plan': 'pro'}  # حالت غیرتجاری
    
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
            
            # بررسی انقضای اشتراک
            if user[3]:  # اگر تاریخ انقضا تعیین شده
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
    
    # پاک‌سازی درخواست‌های قدیمی
    user_api_calls[user_id] = [
        timestamp for timestamp in user_api_calls[user_id] 
        if timestamp > hour_ago
    ]
    
    # بررسی محدودیت
    plan_limits = get_user_plan_limits(subscription_plan)
    current_calls = len(user_api_calls[user_id])
    
    if current_calls >= plan_limits['api_calls_per_hour']:
        return False
    
    # اضافه کردن درخواست جدید
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
        
        # بروزرسانی آمار کلی کاربر
        cursor.execute('''
            UPDATE users 
            SET total_api_calls = total_api_calls + 1, last_api_call = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), user_id))
        
        # ذخیره جزئیات استفاده
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
            # حالت غیرتجاری - همه درخواست‌ها مجاز
            g.current_user = {'id': 0, 'username': 'anonymous', 'subscription_plan': 'pro'}
            return f(*args, **kwargs)
        
        # دریافت اطلاعات Authentication
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            return jsonify({
                'error': 'Authentication required',
                'message': 'Please provide username and password using Basic Auth'
            }), 401
        
        # اعتبارسنجی کاربر
        user = get_user_by_credentials(auth.username, auth.password)
        if not user:
            return jsonify({
                'error': 'Invalid credentials',
                'message': 'Username or password is incorrect'
            }), 401
        
        # بررسی محدودیت نرخ
        if not check_rate_limit(user['id'], user['subscription_plan']):
            plan_limits = get_user_plan_limits(user['subscription_plan'])
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': f'You have exceeded your {user["subscription_plan"]} plan limit of {plan_limits["api_calls_per_hour"]} calls per hour',
                'retry_after': 3600  # 1 hour
            }), 429
        
        # ذخیره اطلاعات کاربر در g
        g.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

# --- توابع موجود (بدون تغییر) ---

def load_optimized_model():
    """بارگذاری مدل بهبود یافته (Optimized Model Package)"""
    global model_package, scaler, model_info
    
    try:
        # جستجو برای مدل‌های Optimized جدید
        optimized_model_files = glob.glob(os.path.join(MODELS_PATH, 'optimized_model_*.joblib'))
        optimized_scaler_files = glob.glob(os.path.join(MODELS_PATH, 'scaler_optimized_*.joblib'))
        
        if optimized_model_files and optimized_scaler_files:
            # استفاده از مدل‌های بهبود یافته
            latest_model_file = max(optimized_model_files, key=os.path.getctime)
            latest_scaler_file = max(optimized_scaler_files, key=os.path.getctime)
            
            model_package = joblib.load(latest_model_file)
            scaler = joblib.load(latest_scaler_file)
            
            # استخراج اطلاعات مدل
            if isinstance(model_package, dict):
                model_info = {
                    'model_type': model_package.get('model_type', 'Unknown'),
                    'optimal_threshold': model_package.get('optimal_threshold', 0.5),
                    'accuracy': model_package.get('accuracy', 0.0),
                    'precision': model_package.get('precision', 0.0),
                    'recall': model_package.get('recall', 0.0),
                    'feature_columns': model_package.get('feature_columns', []),
                    'model_file': os.path.basename(latest_model_file),
                    'scaler_file': os.path.basename(latest_scaler_file),
                    'is_optimized': True
                }
            else:
                # اگر فرمت قدیمی است
                model_info = {
                    'model_type': type(model_package).__name__,
                    'optimal_threshold': 0.5,
                    'model_file': os.path.basename(latest_model_file),
                    'scaler_file': os.path.basename(latest_scaler_file),
                    'is_optimized': False
                }
            
            print(f"✅ Optimized Model loaded: {model_info['model_type']} from {model_info['model_file']}")
            print(f"✅ Optimal Threshold: {model_info['optimal_threshold']:.4f}")
            print(f"✅ Scaler loaded from: {model_info['scaler_file']}")
            
            if model_info.get('accuracy'):
                print(f"📊 Model Performance: Accuracy={model_info['accuracy']:.2%}, "
                      f"Precision={model_info['precision']:.2%}, Recall={model_info['recall']:.2%}")
            
            return True
            
        else:
            print("⚠️ No optimized models found, trying legacy models...")
            return load_legacy_model()
            
    except Exception as e:
        print(f"❌ Error loading optimized model: {e}")
        print("🔄 Falling back to legacy model...")
        return load_legacy_model()

def load_legacy_model():
    """بارگذاری مدل‌های قدیمی (fallback)"""
    global model_package, scaler, model_info
    
    try:
        # جستجو برای مدل‌های قدیمی
        legacy_model_files = glob.glob(os.path.join(MODELS_PATH, 'random_forest_model_*.joblib'))
        legacy_scaler_files = glob.glob(os.path.join(MODELS_PATH, 'scaler_*.joblib'))
        
        if not legacy_model_files or not legacy_scaler_files:
            raise FileNotFoundError("No legacy models found either")
        
        latest_model_file = max(legacy_model_files, key=os.path.getctime)
        latest_scaler_file = max(legacy_scaler_files, key=os.path.getctime)
        
        # بارگذاری مدل قدیمی
        model_package = {'model': joblib.load(latest_model_file)}
        scaler = joblib.load(latest_scaler_file)
        
        model_info = {
            'model_type': 'RandomForestClassifier (Legacy)',
            'optimal_threshold': 0.5,  # threshold پیش‌فرض
            'model_file': os.path.basename(latest_model_file),
            'scaler_file': os.path.basename(latest_scaler_file),
            'is_optimized': False,
            'is_legacy': True
        }
        
        print(f"⚠️ Legacy Model loaded: {model_info['model_file']}")
        print(f"⚠️ Using default threshold: {model_info['optimal_threshold']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading legacy model: {e}")
        return False

def get_model():
    """دریافت مدل اصلی از package"""
    if model_package is None:
        return None
    
    if isinstance(model_package, dict):
        return model_package.get('model')
    else:
        return model_package  # legacy format

def make_prediction(input_features, use_optimal_threshold=True):
    """پیش‌بینی با استفاده از مدل و threshold بهینه"""
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
            threshold = 0.5  # threshold پیش‌فرض
        
        # تصمیم‌گیری نهایی
        final_prediction = 1 if profit_prob >= threshold else 0
        signal = 'PROFIT' if final_prediction == 1 else 'NO_PROFIT'
        
        return {
            'prediction': int(final_prediction),
            'signal': signal,
            'confidence': {
                'no_profit_prob': round(no_profit_prob, 4),
                'profit_prob': round(profit_prob, 4)
            },
            'threshold_used': threshold,
            'raw_probabilities': {
                'no_profit_raw': round(no_profit_prob, 4),
                'profit_raw': round(profit_prob, 4)
            }
        }
        
    except Exception as e:
        logging.error(f"Error in make_prediction: {e}")
        return None

# --- بخش بارگذاری مدل ---
print("🔄 Initializing Enhanced Prediction API v6.0...")

# Initialize user database
if COMMERCIAL_MODE:
    print("💼 Commercial mode enabled - initializing user database...")
    init_user_database()
else:
    print("🔓 Running in non-commercial mode")

model_loaded = load_optimized_model()
if not model_loaded:
    print("❌ CRITICAL: Could not load any model. API will not function properly.")
    model_package = None
    scaler = None

# --- بخش اپلیکیشن Flask ---
app = Flask(__name__)

@app.route("/")
def index():
    mode_text = "Commercial" if COMMERCIAL_MODE else "Open"
    return f"Enhanced Prediction API v6.0 ({mode_text} Mode) is running. Use the /predict endpoint for predictions."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check با اطلاعات تفصیلی مدل و حالت تجاری (اصلاح شده)"""
    try:
        model = get_model()
        
        # آمار کاربران (در حالت تجاری)
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
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        # 🔧 اضافه کردن اطلاعات مدل با error handling
        try:
            if model_info:
                health_status.update({
                    'model_info': {
                        'model_type': str(model_info.get('model_type', 'Unknown')),
                        'model_file': str(model_info.get('model_file', 'Unknown')),
                        'scaler_file': str(model_info.get('scaler_file', 'Unknown')),
                        'is_optimized': bool(model_info.get('is_optimized', False)),
                        'optimal_threshold': float(model_info.get('optimal_threshold', 0.5)),
                        'features_count': int(len(model_info.get('feature_columns', []))),
                        'performance': {
                            'accuracy': float(model_info.get('accuracy')) if model_info.get('accuracy') is not None else None,
                            'precision': float(model_info.get('precision')) if model_info.get('precision') is not None else None,
                            'recall': float(model_info.get('recall')) if model_info.get('recall') is not None else None
                        } if model_info.get('accuracy') is not None else None
                    }
                })
        except Exception as model_info_error:
            logging.warning(f"Error in model_info serialization: {model_info_error}")
            health_status['model_info_error'] = str(model_info_error)
        
        # اضافه کردن آمار کاربران
        if user_stats:
            health_status['user_stats'] = user_stats
        
        # محاسبه uptime (ساده)
        try:
            import psutil
            process = psutil.Process()
            uptime_seconds = (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
            health_status['uptime_seconds'] = round(float(uptime_seconds), 2)
        except ImportError:
            health_status['uptime_seconds'] = None
        except Exception as uptime_error:
            logging.warning(f"Uptime calculation error: {uptime_error}")
            health_status['uptime_seconds'] = None
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        # مدیریت کامل خطا
        error_response = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model_loaded': False,
            'scaler_loaded': False,
            'commercial_mode': COMMERCIAL_MODE
        }
        logging.error(f"Health check failed: {e}")
        return jsonify(error_response), 500

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    """پیش‌بینی با اعتبارسنجی و ردیابی استفاده (اصلاح شده)"""
    start_time = datetime.now()
    current_user = g.current_user
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    
    if not get_model() or not scaler:
        update_user_usage(current_user['id'], '/predict', ip_address, 500, 0)
        return jsonify({"error": "Model or scaler is not loaded properly"}), 500

    try:
        # دریافت داده ورودی
        input_data = request.get_json(force=True)
        if not input_data:
            update_user_usage(current_user['id'], '/predict', ip_address, 400, 0)
            return jsonify({"error": "Invalid input: No JSON data received"}), 400
        
        # لاگ کردن درخواست ورودی (فقط تعداد فیلدها)
        app.logger.info(f"Received prediction request from user {current_user['username']} with {len(input_data)} features")
        
        # 🔧 پاک‌سازی input data (همان اصلاحات از فایل 05)
        cleaned_data = {}
        for k, v in input_data.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                if np.isnan(v) or np.isinf(v):
                    app.logger.warning(f"Skipping invalid value: {k}={v}")
                    continue
                # تبدیل numpy types به Python native
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
        
        # تبدیل به DataFrame
        df = pd.DataFrame([cleaned_data])
        
        # بررسی ویژگی‌های مورد نیاز (اگر موجود باشد)
        expected_features = model_info.get('feature_columns', [])
        if expected_features:
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                update_user_usage(current_user['id'], '/predict', ip_address, 400, processing_time)
                return jsonify({
                    "error": f"Missing required features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}",
                    "missing_count": len(missing_features),
                    "total_expected": len(expected_features),
                    "received_features": len(df.columns)
                }), 400
            
            # مرتب‌سازی ستون‌ها مطابق انتظارات مدل
            df = df[expected_features]
        
        # انجام پیش‌بینی
        prediction_result = make_prediction(df)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000  # میلی‌ثانیه
        
        if prediction_result is None:
            update_user_usage(current_user['id'], '/predict', ip_address, 500, processing_time)
            return jsonify({"error": "Prediction failed"}), 500
        
        # 🔧 پاک‌سازی نتایج برای JSON serialization (مثل فایل 05)
        def clean_for_json(obj):
            """تبدیل numpy types به Python native types"""
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
        
        # ساخت پاسخ کامل با اطلاعات تجاری
        result = {
            'prediction': int(prediction_result['prediction']),
            'signal': str(prediction_result['signal']),
            'confidence': {
                'no_profit_prob': float(prediction_result['confidence']['no_profit_prob']),
                'profit_prob': float(prediction_result['confidence']['profit_prob'])
            },
            'model_info': {
                'model_type': str(model_info.get('model_type', 'Unknown')),
                'threshold_used': float(prediction_result['threshold_used']),
                'is_optimized': bool(model_info.get('is_optimized', False)),
                'features_used': int(len(df.columns))
            },
            'performance_metrics': None,
            'processing_info': {
                'processing_time_ms': round(processing_time, 2),
                'timestamp_utc': end_time.isoformat() + 'Z'
            }
        }
        
        # اضافه کردن performance metrics (با clean کردن)
        if model_info.get('accuracy'):
            result['performance_metrics'] = {
                'model_accuracy': float(model_info.get('accuracy', 0)),
                'model_precision': float(model_info.get('precision', 0)),
                'model_recall': float(model_info.get('recall', 0))
            }
        
        # اطلاعات کاربر و استفاده (در حالت تجاری)
        if COMMERCIAL_MODE and current_user['id'] > 0:
            plan_limits = get_user_plan_limits(current_user['subscription_plan'])
            result['user_info'] = {
                'username': current_user['username'],
                'subscription_plan': current_user['subscription_plan'],
                'remaining_calls_this_hour': plan_limits['api_calls_per_hour'] - len(user_api_calls[current_user['id']]),
                'total_api_calls': current_user['total_api_calls'] + 1
            }
        
        # اضافه کردن احتمالات خام (برای debugging)
        if 'raw_probabilities' in prediction_result:
            result['raw_probabilities'] = {
                'no_profit_raw': float(prediction_result['raw_probabilities']['no_profit_raw']),
                'profit_raw': float(prediction_result['raw_probabilities']['profit_raw'])
            }
        
        # پاک‌سازی نهایی
        result = clean_for_json(result)
        
        # ثبت استفاده موفق
        update_user_usage(current_user['id'], '/predict', ip_address, 200, processing_time)
        
        app.logger.info(f"Prediction completed for user {current_user['username']}: Signal={result['signal']}, "
                       f"Confidence={result['confidence']['profit_prob']:.2%}, "
                       f"Threshold={prediction_result['threshold_used']:.3f}")
        
        return jsonify(result)

    except ValueError as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        update_user_usage(current_user['id'], '/predict', ip_address, 400, processing_time)
        app.logger.error(f"Value error during prediction for user {current_user['username']}: {e}")
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        update_user_usage(current_user['id'], '/predict', ip_address, 500, processing_time)
        app.logger.error(f"Error during prediction for user {current_user['username']}: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route('/model-info', methods=['GET'])
@require_auth
def get_model_info():
    """Endpoint برای دریافت اطلاعات تفصیلی مدل با احراز هویت"""
    current_user = g.current_user
    
    # ثبت استفاده
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    update_user_usage(current_user['id'], '/model-info', ip_address, 200, 0)
    
    try:
        response_data = {
            'model_info': model_info,
            'model_loaded': get_model() is not None,
            'scaler_loaded': scaler is not None,
            'api_version': '6.0',
            'features_supported': len(model_info.get('feature_columns', [])),
            'optimized_model': model_info.get('is_optimized', False),
            'commercial_mode': COMMERCIAL_MODE
        }
        
        # اطلاعات کاربر (در حالت تجاری)
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
            'api_version': '6.0',
            'commercial_mode': COMMERCIAL_MODE
        }), 500

# --- Endpoint جدید برای آمار Admin ---
@app.route('/admin/stats', methods=['GET'])
@require_auth
def admin_stats():
    """Endpoint برای آمار ادمین (فقط برای آینده)"""
    current_user = g.current_user
    
    # برای آینده: بررسی اینکه کاربر ادمین است
    # فعلاً همه کاربران دسترسی دارند
    
    if not COMMERCIAL_MODE:
        return jsonify({'error': 'Admin stats only available in commercial mode'}), 404
    
    try:
        db_path = os.path.join(USERS_PATH, 'users.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # آمار کلی
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM api_usage WHERE timestamp > datetime('now', '-24 hours')")
        api_calls_24h = cursor.fetchone()[0]
        
        cursor.execute("SELECT subscription_plan, COUNT(*) FROM users WHERE is_active = 1 GROUP BY subscription_plan")
        plan_distribution = dict(cursor.fetchall())
        
        # Top users
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
        
        # ثبت استفاده
        ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        update_user_usage(current_user['id'], '/admin/stats', ip_address, 200, 0)
        
        return jsonify({
            'total_users': total_users,
            'api_calls_24h': api_calls_24h,
            'plan_distribution': plan_distribution,
            'top_users_24h': top_users,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
    except Exception as e:
        return jsonify({'error': f'Could not fetch admin stats: {e}'}), 500

if __name__ == '__main__':
    print(f"--- Starting Enhanced Prediction API Server v6.0 ---")
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
        print(f"✅ Model Status: {model_info.get('model_type', 'Unknown')}")
        print(f"🎯 Optimal Threshold: {model_info.get('optimal_threshold', 0.5):.4f}")
        if model_info.get('is_optimized'):
            print(f"📊 Performance: Precision={model_info.get('precision', 0):.1%}")
        print("🔗 Endpoints:")
        print("   - GET  / (status)")
        print("   - GET  /health (detailed health check)")
        print("   - POST /predict (main prediction) 🔐")
        print("   - GET  /model-info (model details) 🔐")
        if COMMERCIAL_MODE:
            print("   - GET  /admin/stats (admin statistics) 🔐")
    else:
        print("❌ WARNING: No model loaded! API will return errors.")
    
    print(f"📁 Logs: {log_filename}")
    print("🔐 = Requires Authentication in Commercial Mode")
    
    app.run(host=API_HOST, port=API_PORT, debug=False)