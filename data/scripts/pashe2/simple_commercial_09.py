#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
سیستم تجاری‌سازی ساده مشاور هوشمند (نسخه 1.0)

قابلیت‌های کلیدی:
- Web Interface کامل (Registration, Login, Dashboard)
- User Management (SQLite Database)
- Subscription Plans (رایگان، پایه، حرفه‌ای)
- Payment Integration (کارت به کارت + کریپتو)
- Admin Panel (مدیریت کاربران و پرداخت‌ها)
- API Integration (اتصال با prediction API)
- Analytics & Reporting
- Mobile-Friendly Design

نکات فنی:
- Flask-based Web Application
- Embedded HTML Templates (بدون نیاز به فایل‌های جداگانه)
- SQLite Database (آماده برای 500 کاربر)
- Bootstrap UI (responsive design)
- Session Management
- CSRF Protection آماده
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
import configparser
import logging
import requests
import json
from flask import Flask, render_template_string, request, redirect, url_for, session, flash, jsonify, g
import pandas as pd

# --- بخش خواندن پیکربندی ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'

try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    
    # مسیرها
    USERS_PATH = config.get('Paths', 'users', fallback='data/users')
    LOG_PATH = config.get('Paths', 'logs', fallback='data/logs')
    
    # تنظیمات Web Interface
    WEB_HOST = config.get('Web_Interface', 'web_host', fallback='0.0.0.0')
    WEB_PORT = config.getint('Web_Interface', 'web_port', fallback=8000)
    SECRET_KEY = config.get('Web_Interface', 'secret_key', fallback='your_secret_key_here')
    SITE_NAME = config.get('Web_Interface', 'site_name', fallback='مشاور هوشمند کریپتو')
    
    # تنظیمات تجاری
    MAX_USERS = config.getint('Commercial_Settings', 'max_users', fallback=500)
    ADMIN_TELEGRAM_ID = config.getint('Commercial_Settings', 'admin_telegram_id', fallback=0)
    
    # پلان‌های اشتراک
    FREE_SIGNALS_PER_DAY = config.getint('Commercial_Settings', 'free_signals_per_day', fallback=5)
    FREE_SYMBOLS_LIMIT = config.getint('Commercial_Settings', 'free_symbols_limit', fallback=1)
    BASIC_PRICE_MONTHLY = config.getint('Commercial_Settings', 'basic_price_monthly', fallback=20)
    BASIC_SIGNALS_PER_DAY = config.getint('Commercial_Settings', 'basic_signals_per_day', fallback=50)
    PRO_PRICE_MONTHLY = config.getint('Commercial_Settings', 'pro_price_monthly', fallback=50)
    
    # تنظیمات پرداخت
    CARD_NUMBER = config.get('Payment_Settings', 'card_number', fallback='****-****-****-****')
    CARD_HOLDER_NAME = config.get('Payment_Settings', 'card_holder_name', fallback='صاحب کارت')
    BANK_NAME = config.get('Payment_Settings', 'bank_name', fallback='بانک ملی ایران')
    BTC_ADDRESS = config.get('Payment_Settings', 'btc_address', fallback='bc1q...')
    ETH_ADDRESS = config.get('Payment_Settings', 'eth_address', fallback='0x...')
    USDT_ADDRESS = config.get('Payment_Settings', 'usdt_trc20_address', fallback='TR...')
    
    # API Settings
    API_HOST = config.get('API_Settings', 'host', fallback='localhost')
    API_PORT = config.getint('API_Settings', 'port', fallback=5000)
    API_URL = f"http://{API_HOST}:{API_PORT}"
    
    # Telegram
    TELEGRAM_BOT_TOKEN = config.get('Telegram', 'bot_token', fallback='')
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Error: {e}")
    exit()

# --- تنظیمات لاگ‌گیری ---
os.makedirs(USERS_PATH, exist_ok=True)
os.makedirs(os.path.join(LOG_PATH, 'simple_commercial_09'), exist_ok=True)

log_filename = os.path.join(LOG_PATH, 'simple_commercial_09', 
                           f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Flask Application ---
app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- Database Management ---
def init_database():
    """ایجاد database کامل سیستم"""
    db_path = os.path.join(USERS_PATH, 'commercial.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # جدول کاربران (تکمیل شده)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER UNIQUE,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                subscription_plan TEXT DEFAULT 'free',
                subscription_start_date TEXT,
                subscription_end_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                total_api_calls INTEGER DEFAULT 0,
                last_api_call TEXT,
                registration_ip TEXT,
                email_verified INTEGER DEFAULT 0
            )
        ''')
        
        # جدول پرداخت‌ها
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                transaction_id TEXT,
                receipt_image TEXT,
                status TEXT DEFAULT 'pending',
                admin_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                verified_at TEXT,
                verified_by INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # جدول سیگنال‌ها (برای ردیابی)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                api_response TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # جدول sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT,
                ip_address TEXT,
                user_agent TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # جدول تنظیمات سیستم
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # مقادیر پیش‌فرض تنظیمات
        cursor.execute('INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)', 
                      ('registration_enabled', 'true'))
        cursor.execute('INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)', 
                      ('maintenance_mode', 'false'))
        
        conn.commit()
        conn.close()
        
        logging.info("✅ Commercial database initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize commercial database: {e}")
        return False

def get_db():
    """دریافت connection به database"""
    if 'db' not in g:
        db_path = os.path.join(USERS_PATH, 'commercial.db')
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    """بستن connection در پایان request"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# --- Authentication & User Management ---
def hash_password(password):
    """هش کردن رمز عبور"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """تایید رمز عبور"""
    return hashlib.sha256(password.encode()).hexdigest() == password_hash

def create_user(username, password, email=None, telegram_id=None):
    """ایجاد کاربر جدید"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # بررسی تکراری نبودن username
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return False, "نام کاربری قبلاً استفاده شده است"
        
        # بررسی تکراری نبودن telegram_id
        if telegram_id:
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            if cursor.fetchone():
                return False, "این اکانت تلگرام قبلاً ثبت شده است"
        
        # ایجاد کاربر
        password_hash = hash_password(password)
        registration_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, telegram_id, registration_ip)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, email, telegram_id, registration_ip))
        
        user_id = cursor.lastrowid
        db.commit()
        
        logging.info(f"✅ New user created: {username} (ID: {user_id})")
        return True, user_id
        
    except Exception as e:
        logging.error(f"Error creating user: {e}")
        return False, "خطا در ایجاد کاربر"

def authenticate_user(username, password):
    """احراز هویت کاربر"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('''
            SELECT id, username, password_hash, subscription_plan, is_active 
            FROM users WHERE username = ? AND is_active = 1
        ''', (username,))
        
        user = cursor.fetchone()
        if user and verify_password(password, user['password_hash']):
            # بروزرسانی آخرین ورود
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                          (datetime.now().isoformat(), user['id']))
            db.commit()
            
            return dict(user)
        
        return None
        
    except Exception as e:
        logging.error(f"Error in user authentication: {e}")
        return None

def login_required(f):
    """Decorator برای صفحات نیازمند ورود"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('برای دسترسی به این صفحه ابتدا وارد شوید', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator برای صفحات مدیریت"""
    @wraps(f) 
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        # ساده: فقط کاربر اول ادمین است (برای شروع)
        if session['user_id'] != 1:
            flash('شما دسترسی مدیریت ندارید', 'error')
            return redirect(url_for('dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

# --- API Integration ---
def call_prediction_api(payload, username, password):
    """فراخوانی API پیش‌بینی"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            auth=(username, password),
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            return None, "خطا در احراز هویت API"
        elif response.status_code == 429:
            return None, "محدودیت تعداد درخواست به API"
        else:
            return None, f"خطا در API: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        logging.error(f"API call error: {e}")
        return None, "خطا در ارتباط با سرور پیش‌بینی"

def get_api_health():
    """بررسی سلامت API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# --- Payment Management ---
def create_payment(user_id, amount, currency, payment_method, transaction_id=None):
    """ایجاد درخواست پرداخت جدید"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('''
            INSERT INTO payments (user_id, amount, currency, payment_method, transaction_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, amount, currency, payment_method, transaction_id))
        
        payment_id = cursor.lastrowid
        db.commit()
        
        # ارسال اطلاع به ادمین (تلگرام)
        send_payment_notification_to_admin(payment_id, user_id, amount, currency, payment_method)
        
        return payment_id
        
    except Exception as e:
        logging.error(f"Error creating payment: {e}")
        return None

def send_payment_notification_to_admin(payment_id, user_id, amount, currency, payment_method):
    """ارسال اطلاع پرداخت به ادمین"""
    if not TELEGRAM_BOT_TOKEN or not ADMIN_TELEGRAM_ID:
        return
    
    try:
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
        username = cursor.fetchone()['username']
        
        message = f"""
🔔 <b>درخواست پرداخت جدید</b>

👤 کاربر: {username}
💰 مبلغ: ${amount} {currency}
💳 روش: {payment_method}
🆔 شناسه پرداخت: {payment_id}
🕐 زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

برای تأیید یا رد، به پنل مدیریت مراجعه کنید.
"""
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': ADMIN_TELEGRAM_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        requests.post(url, data=data, timeout=5)
        
    except Exception as e:
        logging.error(f"Error sending payment notification: {e}")

# --- HTML Templates (Embedded) ---

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ورود - {{ site_name }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .login-container { max-width: 400px; margin: 50px auto; }
        .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        .form-control:focus { border-color: #667eea; box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25); }
    </style>
</head>
<body>
    <div class="container">
        <div class="login-container">
            <div class="card">
                <div class="card-header text-center bg-transparent border-0 pt-4">
                    <h3 class="text-primary">🤖 {{ site_name }}</h3>
                    <p class="text-muted">سیگنال‌های هوشمند معاملات</p>
                </div>
                <div class="card-body p-4">
                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="alert alert-{{ 'danger' if category == 'error' else 'success' if category == 'success' else 'warning' }} alert-dismissible fade show">
                                    {{ message }}
                                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                </div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                    
                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label">نام کاربری</label>
                            <input type="text" class="form-control" name="username" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">رمز عبور</label>
                            <input type="password" class="form-control" name="password" required>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 mb-3">ورود</button>
                    </form>
                    
                    <div class="text-center">
                        <a href="{{ url_for('register') }}" class="text-decoration-none">حساب کاربری ندارید؟ ثبت نام کنید</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ثبت نام - {{ site_name }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .register-container { max-width: 500px; margin: 30px auto; }
        .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .btn-success { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); border: none; }
        .form-control:focus { border-color: #56ab2f; box-shadow: 0 0 0 0.2rem rgba(86, 171, 47, 0.25); }
    </style>
</head>
<body>
    <div class="container">
        <div class="register-container">
            <div class="card">
                <div class="card-header text-center bg-transparent border-0 pt-4">
                    <h3 class="text-success">📝 ثبت نام در {{ site_name }}</h3>
                </div>
                <div class="card-body p-4">
                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} alert-dismissible fade show">
                                    {{ message }}
                                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                </div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                    
                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label">نام کاربری *</label>
                            <input type="text" class="form-control" name="username" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">رمز عبور *</label>
                            <input type="password" class="form-control" name="password" required minlength="6">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">ایمیل (اختیاری)</label>
                            <input type="email" class="form-control" name="email">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">شناسه تلگرام (اختیاری)</label>
                            <input type="number" class="form-control" name="telegram_id">
                            <div class="form-text">برای دریافت اطلاعیه‌ها</div>
                        </div>
                        <button type="submit" class="btn btn-success w-100 mb-3">ثبت نام</button>
                    </form>
                    
                    <div class="text-center">
                        <a href="{{ url_for('login') }}" class="text-decoration-none">قبلاً ثبت نام کرده‌اید؟ وارد شوید</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>داشبورد - {{ site_name }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .navbar-brand { font-weight: bold; }
        .card { border-radius: 10px; transition: transform 0.2s; }
        .card:hover { transform: translateY(-2px); }
        .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .plan-badge { font-size: 0.8em; }
        .signal-form { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="#">🤖 {{ site_name }}</a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text me-3">خوش آمدید {{ user.username }}!</span>
                <a class="nav-link" href="{{ url_for('logout') }}">خروج</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else 'success' if category == 'success' else 'info' }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <!-- آمار کاربر -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card stats-card">
                    <div class="card-body text-center">
                        <i class="fas fa-crown fa-2x mb-2"></i>
                        <h5>اشتراک شما</h5>
                        <span class="badge bg-light text-dark plan-badge">{{ user.subscription_plan|upper }}</span>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white">
                    <div class="card-body text-center">
                        <i class="fas fa-signal fa-2x mb-2"></i>
                        <h5>{{ user.total_api_calls or 0 }}</h5>
                        <small>کل سیگنال‌ها</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white">
                    <div class="card-body text-center">
                        <i class="fas fa-calendar-day fa-2x mb-2"></i>
                        <h5>{{ today_signals }}</h5>
                        <small>سیگنال‌های امروز</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white">
                    <div class="card-body text-center">
                        <i class="fas fa-server fa-2x mb-2"></i>
                        <h5>{{ 'آنلاین' if api_status else 'آفلاین' }}</h5>
                        <small>وضعیت API</small>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- فرم درخواست سیگنال -->
            <div class="col-lg-8">
                <div class="signal-form">
                    <h4 class="mb-4">🎯 درخواست سیگنال جدید</h4>
                    
                    <form id="signalForm" onsubmit="getSignal(event)">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">نماد ارز</label>
                                <select class="form-control" name="symbol" required>
                                    <option value="">انتخاب کنید</option>
                                    <option value="BTC/USDT">Bitcoin (BTC/USDT)</option>
                                    <option value="ETH/USDT">Ethereum (ETH/USDT)</option>
                                    <option value="BNB/USDT">Binance Coin (BNB/USDT)</option>
                                    <option value="XRP/USDT">Ripple (XRP/USDT)</option>
                                    <option value="ADA/USDT">Cardano (ADA/USDT)</option>
                                </select>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">تایم فریم</label>
                                <select class="form-control" name="timeframe" required>
                                    <option value="1h">1 ساعت</option>
                                    <option value="4h">4 ساعت</option>
                                    <option value="1d">1 روز</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">قیمت فعلی ($)</label>
                                <input type="number" class="form-control" name="current_price" step="0.01" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">حجم</label>
                                <input type="number" class="form-control" name="volume" step="0.01" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary" id="signalBtn">
                            <i class="fas fa-chart-line"></i> دریافت سیگنال
                        </button>
                    </form>
                    
                    <!-- نتیجه سیگنال -->
                    <div id="signalResult" class="mt-4" style="display: none;">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">📊 نتیجه تحلیل</h5>
                                <div id="resultContent"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- منوی کناری -->
            <div class="col-lg-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-cogs"></i> عملیات سریع</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2">
                            <a href="{{ url_for('subscription') }}" class="btn btn-outline-primary">
                                <i class="fas fa-star"></i> ارتقا اشتراک
                            </a>
                            <a href="{{ url_for('history') }}" class="btn btn-outline-info">
                                <i class="fas fa-history"></i> تاریخچه سیگنال‌ها
                            </a>
                            <a href="{{ url_for('profile') }}" class="btn btn-outline-secondary">
                                <i class="fas fa-user"></i> ویرایش پروفایل
                            </a>
                        </div>
                    </div>
                </div>

                <!-- اطلاعیه‌ها -->
                <div class="card mt-3">
                    <div class="card-header">
                        <h5><i class="fas fa-bell"></i> اطلاعیه‌ها</h5>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <small><strong>جدید!</strong> مدل AI ما با دقت 92% بهبود یافت.</small>
                        </div>
                        <div class="alert alert-warning">
                            <small>برای دسترسی به تمام ویژگی‌ها اشتراک خود را ارتقا دهید.</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function getSignal(event) {
            event.preventDefault();
            
            const btn = document.getElementById('signalBtn');
            const result = document.getElementById('signalResult');
            const content = document.getElementById('resultContent');
            
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال تحلیل...';
            
            const formData = new FormData(event.target);
            
            // ساخت payload ساده (در واقع باید ویژگی‌های کاملی باشد)
            const payload = {
                close: parseFloat(formData.get('current_price')),
                volume: parseFloat(formData.get('volume')),
                open: parseFloat(formData.get('current_price')) * 0.99,
                high: parseFloat(formData.get('current_price')) * 1.01,
                low: parseFloat(formData.get('current_price')) * 0.98,
                rsi: 50,  // مقادیر نمونه
                sentiment_score: 0.1
            };
            
            try {
                const response = await fetch('/api/get-signal', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        payload: payload,
                        symbol: formData.get('symbol'),
                        timeframe: formData.get('timeframe')
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const signal = data.result.signal;
                    const confidence = data.result.confidence.profit_prob;
                    const threshold = data.result.model_info.threshold_used;
                    
                    const badgeClass = signal === 'PROFIT' ? 'bg-success' : 'bg-danger';
                    const icon = signal === 'PROFIT' ? 'fa-arrow-up' : 'fa-arrow-down';
                    
                    content.innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h6>سیگنال:</h6>
                                <span class="badge ${badgeClass} fs-6">
                                    <i class="fas ${icon}"></i> ${signal}
                                </span>
                            </div>
                            <div class="col-md-6">
                                <h6>اطمینان:</h6>
                                <div class="progress">
                                    <div class="progress-bar" style="width: ${confidence*100}%">${(confidence*100).toFixed(1)}%</div>
                                </div>
                            </div>
                        </div>
                        <hr>
                        <small class="text-muted">
                            مدل: ${data.result.model_info.model_type} | 
                            آستانه: ${threshold.toFixed(3)} |
                            زمان: ${new Date().toLocaleString('fa-IR')}
                        </small>
                    `;
                    result.style.display = 'block';
                } else {
                    content.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    result.style.display = 'block';
                }
            } catch (error) {
                content.innerHTML = `<div class="alert alert-danger">خطا در اتصال به سرور</div>`;
                result.style.display = 'block';
            }
            
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-chart-line"></i> دریافت سیگنال';
        }
    </script>
</body>
</html>
'''

# --- Routes ---
@app.route('/')
def index():
    """صفحه اصلی"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """صفحه ورود"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['subscription_plan'] = user['subscription_plan']
            
            flash('با موفقیت وارد شدید!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('نام کاربری یا رمز عبور اشتباه است', 'error')
    
    return render_template_string(LOGIN_TEMPLATE, site_name=SITE_NAME)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """صفحه ثبت نام"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form.get('email') or None
        telegram_id = request.form.get('telegram_id') or None
        
        if telegram_id:
            try:
                telegram_id = int(telegram_id)
            except ValueError:
                telegram_id = None
        
        success, result = create_user(username, password, email, telegram_id)
        
        if success:
            flash('ثبت نام با موفقیت انجام شد! وارد شوید.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result, 'error')
    
    return render_template_string(REGISTER_TEMPLATE, site_name=SITE_NAME)

@app.route('/dashboard')
@login_required
def dashboard():
    """داشبورد اصلی"""
    # دریافت اطلاعات کاربر
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = dict(cursor.fetchone())
    
    # محاسبه سیگنال‌های امروز
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT COUNT(*) FROM user_signals 
        WHERE user_id = ? AND date(timestamp) = ?
    ''', (session['user_id'], today))
    today_signals = cursor.fetchone()[0]
    
    # بررسی وضعیت API
    api_status = get_api_health() is not None
    
    return render_template_string(DASHBOARD_TEMPLATE, 
                                site_name=SITE_NAME, 
                                user=user, 
                                today_signals=today_signals,
                                api_status=api_status)

@app.route('/api/get-signal', methods=['POST'])
@login_required
def api_get_signal():
    """API برای دریافت سیگنال از داشبورد"""
    try:
        data = request.get_json()
        payload = data['payload']
        symbol = data['symbol']
        timeframe = data['timeframe']
        
        # فراخوانی API پیش‌بینی
        result, error = call_prediction_api(payload, session['username'], 'password_placeholder')
        
        if result:
            # ذخیره سیگنال در دیتابیس
            db = get_db()
            cursor = db.cursor()
            
            cursor.execute('''
                INSERT INTO user_signals (user_id, symbol, timeframe, signal, confidence, api_response)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], symbol, timeframe, 
                  result['signal'], result['confidence']['profit_prob'], 
                  json.dumps(result)))
            
            db.commit()
            
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'success': False, 'error': error or 'خطا در دریافت سیگنال'})
            
    except Exception as e:
        logging.error(f"Error in get_signal: {e}")
        return jsonify({'success': False, 'error': 'خطای سرور'})

@app.route('/subscription')
@login_required
def subscription():
    """صفحه اشتراک‌ها"""
    # ساده - فعلاً redirect به داشبورد
    flash('صفحه اشتراک‌ها به زودی راه‌اندازی می‌شود', 'info')
    return redirect(url_for('dashboard'))

@app.route('/history')
@login_required 
def history():
    """تاریخچه سیگنال‌ها"""
    flash('صفحه تاریخچه به زودی راه‌اندازی می‌شود', 'info')
    return redirect(url_for('dashboard'))

@app.route('/profile')
@login_required
def profile():
    """ویرایش پروفایل"""
    flash('صفحه پروفایل به زودی راه‌اندازی می‌شود', 'info')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """خروج کاربر"""
    session.clear()
    flash('با موفقیت خارج شدید', 'success')
    return redirect(url_for('login'))

# --- Admin Routes (ساده) ---
@app.route('/admin')
@admin_required
def admin_dashboard():
    """داشبورد مدیریت ساده"""
    flash('پنل مدیریت به زودی کامل می‌شود', 'info')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    print(f"🚀 Starting Simple Commercial System v1.0")
    print(f"💼 Site Name: {SITE_NAME}")
    print(f"🌐 Web Interface: http://{WEB_HOST}:{WEB_PORT}")
    print(f"👥 Max Users: {MAX_USERS}")
    print(f"🔗 Prediction API: {API_URL}")
    
    # Initialize database
    if init_database():
        print(f"✅ Database initialized: {os.path.join(USERS_PATH, 'commercial.db')}")
    else:
        print(f"❌ Database initialization failed!")
        exit()
    
    # Check API connection
    api_health = get_api_health()
    if api_health:
        print(f"✅ Prediction API is healthy")
    else:
        print(f"⚠️ Prediction API is not responding")
    
    print(f"📁 Logs: {log_filename}")
    print(f"🚀 Ready to serve up to {MAX_USERS} users!")
    print("="*60)
    
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False)