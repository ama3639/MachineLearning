#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت API پیش‌بینی با Flask (نسخه نهایی و قابل پیکربندی)

ویژگی‌ها:
- خواندن تنظیمات host و port از فایل config.ini.
- بارگذاری آخرین مدل و scaler ذخیره شده.
- ایجاد یک endpoint به نام /predict برای دریافت داده‌های جدید.
"""
import os
import glob
import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify
import configparser

# --- بخش خواندن پیکربندی ---
config = configparser.ConfigParser()
CONFIG_FILE_PATH = 'config.ini'
try:
    config.read(CONFIG_FILE_PATH, encoding='utf-8')
    
    # در سرور اوبونتو غیر فعال شوند
    MODELS_PATH = config.get('Paths', 'models')
    LOG_PATH = config.get('Paths', 'logs')
    API_HOST = config.get('API_Settings', 'host')
    API_PORT = config.getint('API_Settings', 'port') # .getint() برای خواندن عدد صحیح
except Exception as e:
    print(f"CRITICAL ERROR: Could not read 'config.ini'. Please check the file. Error: {e}")
    exit()

# --- بخش تنظیمات لاگ‌گیری ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_subfolder_path = os.path.join(LOG_PATH, script_name)
os.makedirs(log_subfolder_path, exist_ok=True)
log_filename = os.path.join(log_subfolder_path, f"log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
# تنظیم لاگ برای Flask
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- windowsبخش بارگذاری مدل ---
try:
    latest_model_file = max(glob.glob(os.path.join(MODELS_PATH, 'random_forest_model_*.joblib')), key=os.path.getctime)
    latest_scaler_file = max(glob.glob(os.path.join(MODELS_PATH, 'scaler_*.joblib')), key=os.path.getctime)


# # ---در صورت قدر مطلق config.ini درست نشود با ubuntuبخش بارگذاری مدل ---
# try:
#     latest_model_file = max(glob.glob('../models/random_forest_model_*.joblib'), key=os.path.getctime)
#     latest_scaler_file = max(glob.glob('../models/scaler_*.joblib'), key=os.path.getctime)
    
    
    model = joblib.load(latest_model_file)
    scaler = joblib.load(latest_scaler_file)
    
    print(f"INFO: Model loaded from -> {os.path.basename(latest_model_file)}")
    print(f"INFO: Scaler loaded from -> {os.path.basename(latest_scaler_file)}")

except (ValueError, FileNotFoundError) as e:
    print(f"ERROR: Could not load model or scaler. {e}")
    model = None
    scaler = None

# --- بخش اپلیکیشن Flask ---
app = Flask(__name__)

@app.route("/")
def index():
    return "Prediction API is running. Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or scaler is not loaded properly"}), 500

    try:
        input_data = request.get_json(force=True)
        if not input_data:
            return jsonify({"error": "Invalid input: No JSON data received"}), 400
        
        # لاگ کردن درخواست ورودی
        app.logger.info(f"Received request: {input_data}")
            
        df = pd.DataFrame([input_data])
        
        # اطمینان از ترتیب ستون‌ها مطابق با آموزش مدل
        # (در اینجا فرض می‌کنیم ترتیب ستون‌ها در درخواست صحیح است)
        
        data_scaled = scaler.transform(df)
        
        prediction = model.predict(data_scaled)
        prediction_proba = model.predict_proba(data_scaled)
        
        result = {
            'prediction': int(prediction[0]),
            'signal': 'PROFIT' if int(prediction[0]) == 1 else 'NO_PROFIT',
            'confidence': {
                'no_profit_prob': round(prediction_proba[0][0], 4),
                'profit_prob': round(prediction_proba[0][1], 4)
            },
            'timestamp_utc': pd.Timestamp.utcnow().isoformat()
        }
        
        app.logger.info(f"Sending response: {result}")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    print(f"--- Starting Prediction API Server ---")
    print(f"INFO: The API will be available at http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False) # debug=False برای محیط عملیاتی بهتر است