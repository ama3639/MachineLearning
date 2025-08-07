#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت API پیش‌بینی با Flask (نسخه 5.1 - پشتیبانی Optimized Models)

ویژگی‌های جدید:
- پشتیبانی از Optimized Model Package (model + optimal threshold)
- سازگاری با Ensemble Models (RandomForest + XGBoost)
- Enhanced Response با اطلاعات تفصیلی‌تر
- بهبود Health Check
- Fallback به مدل‌های قدیمی در صورت نیاز
"""
import os
import glob
import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify
import configparser
import numpy as np
from datetime import datetime

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

# متغیرهای global برای مدل
model_package = None
scaler = None
model_info = {}

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
model_loaded = load_optimized_model()
if not model_loaded:
    print("❌ CRITICAL: Could not load any model. API will not function properly.")
    model_package = None
    scaler = None

# --- بخش اپلیکیشن Flask ---
app = Flask(__name__)

@app.route("/")
def index():
    return "Prediction API v5.1 is running. Use the /predict endpoint for predictions."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check با اطلاعات تفصیلی مدل"""
    model = get_model()
    health_status = {
        'status': 'healthy' if model and scaler else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # اضافه کردن اطلاعات مدل
    if model_info:
        health_status.update({
            'model_info': {
                'model_type': model_info.get('model_type', 'Unknown'),
                'model_file': model_info.get('model_file', 'Unknown'),
                'scaler_file': model_info.get('scaler_file', 'Unknown'),
                'is_optimized': model_info.get('is_optimized', False),
                'optimal_threshold': model_info.get('optimal_threshold', 0.5),
                'features_count': len(model_info.get('feature_columns', [])),
                'performance': {
                    'accuracy': model_info.get('accuracy'),
                    'precision': model_info.get('precision'),
                    'recall': model_info.get('recall')
                } if model_info.get('accuracy') else None
            }
        })
    
    # محاسبه uptime (ساده)
    try:
        import psutil
        process = psutil.Process()
        uptime_seconds = (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
        health_status['uptime_seconds'] = round(uptime_seconds, 2)
    except:
        health_status['uptime_seconds'] = None
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code

@app.route('/predict', methods=['POST'])
def predict():
    if not get_model() or not scaler:
        return jsonify({"error": "Model or scaler is not loaded properly"}), 500

    try:
        # دریافت داده ورودی
        input_data = request.get_json(force=True)
        if not input_data:
            return jsonify({"error": "Invalid input: No JSON data received"}), 400
        
        # لاگ کردن درخواست ورودی (فقط تعداد فیلدها)
        app.logger.info(f"Received prediction request with {len(input_data)} features")
        
        # تبدیل به DataFrame
        df = pd.DataFrame([input_data])
        
        # بررسی ویژگی‌های مورد نیاز (اگر موجود باشد)
        expected_features = model_info.get('feature_columns', [])
        if expected_features:
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                return jsonify({
                    "error": f"Missing required features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}",
                    "missing_count": len(missing_features),
                    "total_expected": len(expected_features)
                }), 400
            
            # مرتب‌سازی ستون‌ها مطابق انتظارات مدل
            df = df[expected_features]
        
        # انجام پیش‌بینی
        start_time = datetime.now()
        prediction_result = make_prediction(df)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000  # میلی‌ثانیه
        
        if prediction_result is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        # ساخت پاسخ کامل
        result = {
            'prediction': prediction_result['prediction'],
            'signal': prediction_result['signal'],
            'confidence': prediction_result['confidence'],
            'model_info': {
                'model_type': model_info.get('model_type', 'Unknown'),
                'threshold_used': prediction_result['threshold_used'],
                'is_optimized': model_info.get('is_optimized', False),
                'features_used': len(df.columns)
            },
            'performance_metrics': {
                'model_accuracy': model_info.get('accuracy'),
                'model_precision': model_info.get('precision'),
                'model_recall': model_info.get('recall')
            } if model_info.get('accuracy') else None,
            'processing_info': {
                'processing_time_ms': round(processing_time, 2),
                'timestamp_utc': end_time.isoformat() + 'Z'
            }
        }
        
        # اضافه کردن احتمالات خام (برای debugging)
        if 'raw_probabilities' in prediction_result:
            result['raw_probabilities'] = prediction_result['raw_probabilities']
        
        app.logger.info(f"Prediction completed: Signal={result['signal']}, "
                       f"Confidence={result['confidence']['profit_prob']:.2%}, "
                       f"Threshold={prediction_result['threshold_used']:.3f}")
        
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Endpoint برای دریافت اطلاعات تفصیلی مدل"""
    return jsonify({
        'model_info': model_info,
        'model_loaded': get_model() is not None,
        'scaler_loaded': scaler is not None,
        'api_version': '5.1',
        'features_supported': len(model_info.get('feature_columns', [])),
        'optimized_model': model_info.get('is_optimized', False)
    })

if __name__ == '__main__':
    print(f"--- Starting Enhanced Prediction API Server v5.1 ---")
    print(f"🏠 API will be available at http://{API_HOST}:{API_PORT}")
    
    if model_loaded:
        print(f"✅ Model Status: {model_info.get('model_type', 'Unknown')}")
        print(f"🎯 Optimal Threshold: {model_info.get('optimal_threshold', 0.5):.4f}")
        if model_info.get('is_optimized'):
            print(f"📊 Performance: Precision={model_info.get('precision', 0):.1%}")
        print("🔗 Endpoints:")
        print("   - GET  / (status)")
        print("   - GET  /health (detailed health check)")
        print("   - POST /predict (main prediction)")
        print("   - GET  /model-info (model details)")
    else:
        print("❌ WARNING: No model loaded! API will return errors.")
    
    print(f"📁 Logs: {log_filename}")
    
    app.run(host=API_HOST, port=API_PORT, debug=False)