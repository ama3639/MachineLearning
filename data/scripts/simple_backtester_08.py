#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت بک تست پیشرفته با معیارهای جامع و پشتیبانی از چند دارایی
نسخه 2.1 - رفع مشکل "max() arg is an empty sequence"

🔧 اصلاحات v2.1:
- ✅ رفع خطای "max() arg is an empty sequence"  
- ✅ بهبود error handling برای فایل‌های موجود نبودن
- ✅ اضافه کردن fallback mechanism
- ✅ بررسی دقیق‌تر مسیرها و فایل‌ها
- ✅ گزارش‌های بهتر برای debugging

ویژگی‌ها:
- بک تست چند نمادی و چند بازه زمانی
- ردیابی دلیل خروج
- معیارهای عملکرد پیشرفته (شارپ، حداکثر افت سرمایه و غیره)
- تجسم تعاملی با برچسب‌های انگلیسی
- گزارش‌های معاملاتی دقیق
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

# --- Configuration and Logging Setup ---
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

# 🔧 تابع کمکی برای یافتن فایل‌ها با error handling بهتر
def find_latest_file(pattern: str, description: str) -> str:
    """
    یافتن آخرین فایل با pattern مشخص - اصلاح شده برای رفع خطای empty sequence
    """
    try:
        # جستجو در مسیر اصلی
        files = glob.glob(pattern)
        logging.info(f"🔍 جستجو برای {description} در: {pattern}")
        logging.info(f"📁 فایل‌های یافت شده: {len(files)}")
        
        if files:
            # فیلتر کردن فایل‌های موجود
            existing_files = [f for f in files if os.path.exists(f) and os.path.getsize(f) > 0]
            logging.info(f"📄 فایل‌های معتبر: {len(existing_files)}")
            
            if existing_files:
                latest_file = max(existing_files, key=os.path.getctime)
                logging.info(f"✅ آخرین فایل {description}: {os.path.basename(latest_file)}")
                return latest_file
        
        # اگر در مسیر اصلی فایل نیافت، جستجو در زیرپوشه‌ها
        parent_dir = os.path.dirname(pattern)
        file_pattern = os.path.basename(pattern)
        
        logging.info(f"🔍 جستجو در زیرپوشه‌های {parent_dir}...")
        alternative_patterns = [
            os.path.join(parent_dir, "**", file_pattern),  # جستجو recursive
            os.path.join(parent_dir, "run_*", file_pattern),  # پوشه‌های run_*
        ]
        
        for alt_pattern in alternative_patterns:
            alt_files = glob.glob(alt_pattern, recursive=True)
            logging.info(f"📁 در {alt_pattern}: {len(alt_files)} فایل")
            
            if alt_files:
                existing_alt_files = [f for f in alt_files if os.path.exists(f) and os.path.getsize(f) > 0]
                if existing_alt_files:
                    latest_file = max(existing_alt_files, key=os.path.getctime)
                    logging.info(f"✅ فایل جایگزین {description}: {os.path.basename(latest_file)}")
                    return latest_file
        
        # اگر هیچ فایلی پیدا نشد
        logging.error(f"❌ هیچ فایل معتبری برای {description} یافت نشد")
        logging.error(f"💡 Pattern جستجو شده: {pattern}")
        logging.error(f"💡 لطفاً مطمئن شوید که فایل‌های مورد نیاز در مسیر صحیح موجود هستند")
        
        # نمایش فایل‌های موجود در پوشه برای debugging
        try:
            parent_directory = os.path.dirname(pattern) if os.path.dirname(pattern) else "."
            if os.path.exists(parent_directory):
                all_files = os.listdir(parent_directory)
                logging.info(f"📋 فایل‌های موجود در {parent_directory}:")
                for file in all_files[:10]:  # نمایش اول 10 فایل
                    logging.info(f"   - {file}")
                if len(all_files) > 10:
                    logging.info(f"   ... و {len(all_files) - 10} فایل دیگر")
        except Exception as list_error:
            logging.warning(f"نمی‌توان فایل‌های پوشه را لیست کرد: {list_error}")
        
        return None
        
    except Exception as e:
        logging.error(f"❌ خطا در جستجوی فایل {description}: {e}")
        return None

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    if equity_curve.empty:
        return 0
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate/252  # Assuming 252 trading days
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def generate_report_file(report_data, symbol, timeframe):
    """Generate simple text report (backward compatibility)"""
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Backtest-Report_{symbol.replace('/', '-')}_{timeframe}_{timestamp_str}.txt"
    filepath = os.path.join(report_subfolder_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("================================================\n")
        f.write("      BACKTEST STRATEGY PERFORMANCE REPORT\n")
        f.write("================================================\n\n")
        f.write(f"Symbol tested:         {symbol}\n")
        f.write(f"Timeframe tested:      {timeframe}\n")
        f.write(f"Report date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("------------------------------------------------\n")
        f.write("      Summary of Results\n")
        f.write("------------------------------------------------\n")
        for key, value in report_data.items():
            if key != 'trade_history':
                f.write(f"{key:<20} {value}\n")
        f.write("\n\n------------------------------------------------\n")
        f.write("      Trade History\n")
        f.write("------------------------------------------------\n")
        if report_data['trade_history']:
            f.write(f"{'#':<3} {'Entry Time':<16} {'Exit Time':<16} {'Entry $':<10} {'Exit $':<10} {'P/L':<8} {'Exit Reason':<20}\n")
            f.write("-"*90 + "\n")
            for i, trade in enumerate(report_data['trade_history'], 1):
                entry_ts = pd.to_datetime(str(trade['entry_date'])).strftime('%Y-%m-%d %H:%M')
                exit_ts = pd.to_datetime(str(trade['exit_date'])).strftime('%Y-%m-%d %H:%M')
                exit_reason = trade.get('exit_reason', 'Target reached')
                f.write(f"{i:<3} {entry_ts:<16} {exit_ts:<16} "
                       f"${trade['entry_price']:<9.4f} ${trade['exit_price']:<9.4f} "
                       f"{trade['pnl']:<7.2%} {exit_reason:<20}\n")
        else:
            f.write("No trades executed in this period.\n")
    
    logging.info(f"Report file saved to '{filepath}'")

def generate_visualizations(df, trade_history, symbol, timeframe, report_path):
    """Generate analytical charts"""
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Price and signals chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Price and entry/exit points
    ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)
    
    # Display trades with reasons
    if trade_history:
        for trade in trade_history:
            try:
                entry_idx = df.index.get_loc(trade['entry_date'])
                exit_idx = df.index.get_loc(trade['exit_date'])
                color = 'green' if trade['pnl'] > 0 else 'red'
                
                # Plot trade line
                ax1.plot([trade['entry_date'], trade['exit_date']], 
                        [trade['entry_price'], trade['exit_price']], 
                        'o-', color=color, markersize=8, linewidth=2)
                
                # Add reason annotation
                mid_date = df.index[entry_idx + (exit_idx - entry_idx) // 2]
                mid_price = (trade['entry_price'] + trade['exit_price']) / 2
                reason = trade.get('exit_reason', 'Target reached')
                ax1.annotate(f"{trade['pnl']:.1%}\n{reason}", 
                            xy=(mid_date, mid_price),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=8, ha='left',
                            bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.3))
            except (KeyError, ValueError) as e:
                logging.warning(f"Error plotting trade: {e}")
                continue
    
    ax1.set_ylabel('Price')
    ax1.set_title(f'Price Chart and Trades - {symbol} ({timeframe})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Model prediction chart
    if 'prediction' in df.columns:
        ax2.plot(df.index, df['prediction'], label='Model Prediction', color='orange', alpha=0.5)
        ax2.fill_between(df.index, 0, df['prediction'], alpha=0.3, color='orange')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Signal')
        ax2.set_title('Model Signals')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    chart_filename = os.path.join(report_path, f"backtest_chart_{symbol.replace('/', '-')}_{timeframe}_{timestamp_str}.png")
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Chart saved to '{chart_filename}'")
    
    # P&L distribution chart
    if trade_history:
        pnl_values = [t['pnl'] * 100 for t in trade_history]
        
        plt.figure(figsize=(10, 6))
        plt.hist(pnl_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Profit/Loss (%)')
        plt.ylabel('Number of Trades')
        plt.title(f'P&L Distribution - {symbol} ({timeframe})')
        plt.grid(True, alpha=0.3)
        
        hist_filename = os.path.join(report_path, f"pnl_distribution_{symbol.replace('/', '-')}_{timeframe}_{timestamp_str}.png")
        plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Distribution chart saved to '{hist_filename}'")

def generate_enhanced_report(report_data, symbol, timeframe, df, equity_curve):
    """Generate enhanced report with additional metrics"""
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Enhanced_Backtest_Report_{symbol.replace('/', '-')}_{timeframe}_{timestamp_str}.txt"
    filepath = os.path.join(report_subfolder_path, filename)
    
    # Calculate additional metrics
    if report_data['trade_history']:
        returns = pd.Series([t['pnl'] for t in report_data['trade_history']])
        avg_win = np.mean([t['pnl'] for t in report_data['trade_history'] if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in report_data['trade_history']) else 0
        avg_loss = np.mean([t['pnl'] for t in report_data['trade_history'] if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in report_data['trade_history']) else 0
        profit_factor = abs(sum([t['pnl'] for t in report_data['trade_history'] if t['pnl'] > 0]) / 
                           sum([t['pnl'] for t in report_data['trade_history'] if t['pnl'] < 0])) if any(t['pnl'] < 0 for t in report_data['trade_history']) else np.inf
    else:
        returns = pd.Series()
        avg_win = avg_loss = profit_factor = 0
    
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("      COMPREHENSIVE BACKTEST REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("📊 General Information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Symbol:                {symbol}\n")
        f.write(f"Timeframe:             {timeframe}\n")
        f.write(f"Report Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Period:           {df.index[0]} to {df.index[-1]}\n")
        f.write(f"Total Candles:         {len(df)}\n\n")
        
        f.write("💰 Financial Results:\n")
        f.write("-"*50 + "\n")
        f.write(f"Initial Capital:       ${float(report_data.get('Initial Capital', report_data.get('سرمایه اولیه', '0')).replace(',', '')):,.2f}\n")
        f.write(f"Final Capital:         ${float(report_data.get('Final Capital', report_data.get('سرمایه نهایی', '0')).replace(',', '')):,.2f}\n")
        f.write(f"Total Return:          {report_data.get('Total Return', report_data.get('بازده کل', 'N/A'))}\n")
        f.write(f"Max Drawdown:          {max_drawdown:.2%}\n")
        f.write(f"Sharpe Ratio:          {sharpe_ratio:.2f}\n\n")
        
        f.write("📈 Trade Statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total Trades:          {report_data.get('Total Trades', report_data.get('تعداد کل معاملات', 0))}\n")
        f.write(f"Winning Trades:        {report_data.get('Winning Trades', report_data.get('تعداد معاملات موفق', 0))}\n")
        f.write(f"Losing Trades:         {report_data.get('Losing Trades', report_data.get('تعداد معاملات ناموفق', 0))}\n")
        f.write(f"Win Rate:              {report_data.get('Win Rate', report_data.get('درصد موفقیت', 'N/A'))}\n")
        f.write(f"Average Win:           {avg_win:.2%}\n")
        f.write(f"Average Loss:          {avg_loss:.2%}\n")
        f.write(f"Profit Factor:         {profit_factor:.2f}\n\n")
        
        f.write("📋 Trade Details:\n")
        f.write("-"*90 + "\n")
        if report_data['trade_history']:
            f.write(f"{'#':<3} {'Entry':<16} {'Exit':<16} {'Entry $':<10} {'Exit $':<10} {'P&L':<8} {'Reason':<20}\n")
            f.write("-"*90 + "\n")
            for i, trade in enumerate(report_data['trade_history'], 1):
                entry_ts = pd.to_datetime(str(trade['entry_date'])).strftime('%Y-%m-%d %H:%M')
                exit_ts = pd.to_datetime(str(trade['exit_date'])).strftime('%Y-%m-%d %H:%M')
                reason = trade.get('exit_reason', 'Target reached')
                f.write(f"{i:<3} {entry_ts:<16} {exit_ts:<16} "
                       f"${trade['entry_price']:<9.4f} ${trade['exit_price']:<9.4f} "
                       f"{trade['pnl']:<7.2%} {reason:<20}\n")
        else:
            f.write("No trades executed during this period.\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("End of Report\n")
        
    logging.info(f"Enhanced report saved to '{filepath}'")

def select_symbols_and_timeframes(df_full):
    """Interactive selection of symbols and timeframes with multi-select option"""
    # Symbol selection
    available_symbols = df_full.index.get_level_values('symbol').unique().tolist()
    print("\n📊 Available Symbols:")
    print("-" * 50)
    for i, sym in enumerate(available_symbols, 1):
        print(f"{i:3d}. {sym}")
    print(f"{len(available_symbols)+1:3d}. ALL SYMBOLS")
    print("-" * 50)
    
    symbol_choice = input("\n💱 Enter symbol number(s) or symbol name (e.g., 1 or BTC/USDT or 1,3,5): ").strip()
    
    selected_symbols = []
    if symbol_choice == str(len(available_symbols)+1) or symbol_choice.upper() == 'ALL':
        selected_symbols = available_symbols
        print(f"✅ Selected ALL {len(selected_symbols)} symbols")
    elif ',' in symbol_choice:
        # Multiple selection
        try:
            indices = [int(x.strip()) - 1 for x in symbol_choice.split(',') if x.strip().isdigit()]
            selected_symbols = [available_symbols[i] for i in indices if 0 <= i < len(available_symbols)]
        except (ValueError, IndexError):
            print("❌ Invalid selection format")
            return None, None
    elif symbol_choice.isdigit():
        # Single number selection
        try:
            idx = int(symbol_choice) - 1
            if 0 <= idx < len(available_symbols):
                selected_symbols = [available_symbols[idx]]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return None, None
    else:
        # Direct symbol name
        symbol_choice = symbol_choice.upper()
        if symbol_choice in available_symbols:
            selected_symbols = [symbol_choice]
    
    if not selected_symbols:
        print("❌ Invalid selection")
        return None, None
    
    # Timeframe selection for first symbol (to get available timeframes)
    try:
        first_symbol_tf = df_full.loc[selected_symbols[0]].index.get_level_values('timeframe').unique().tolist()
    except KeyError:
        print(f"❌ Symbol {selected_symbols[0]} not found in dataset")
        return None, None
        
    print(f"\n⏱️ Available Timeframes:")
    print("-" * 30)
    for i, tf in enumerate(first_symbol_tf, 1):
        print(f"{i:2d}. {tf}")
    print(f"{len(first_symbol_tf)+1:2d}. ALL TIMEFRAMES")
    print("-" * 30)
    
    tf_choice = input("\n🕐 Enter timeframe number(s) or name (e.g., 2 or 1h or 1,2,3): ").strip()
    
    selected_timeframes = []
    if tf_choice == str(len(first_symbol_tf)+1) or tf_choice.upper() == 'ALL':
        selected_timeframes = first_symbol_tf
        print(f"✅ Selected ALL {len(selected_timeframes)} timeframes")
    elif ',' in tf_choice:
        # Multiple selection
        try:
            indices = [int(x.strip()) - 1 for x in tf_choice.split(',') if x.strip().isdigit()]
            selected_timeframes = [first_symbol_tf[i] for i in indices if 0 <= i < len(first_symbol_tf)]
        except (ValueError, IndexError):
            print("❌ Invalid selection format")
            return None, None
    elif tf_choice.isdigit():
        # Single number selection
        try:
            idx = int(tf_choice) - 1
            if 0 <= idx < len(first_symbol_tf):
                selected_timeframes = [first_symbol_tf[idx]]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return None, None
    else:
        # Direct timeframe name
        tf_choice = tf_choice.lower()
        if tf_choice in first_symbol_tf:
            selected_timeframes = [tf_choice]
    
    if not selected_timeframes:
        print("❌ Invalid selection")
        return None, None
    
    return selected_symbols, selected_timeframes

def run_simple_backtest(features_path: str, models_path: str):
    """Run simple backtest (backward compatibility) - اصلاح شده"""
    logging.info("--- Starting Strategy Backtest Process ---")
    
    try:
        # 🔧 یافتن فایل‌ها با error handling بهتر
        feature_file = find_latest_file(
            os.path.join(features_path, 'final_dataset_for_training_*.parquet'),
            "dataset file"
        )
        if not feature_file:
            logging.error("❌ No dataset file found. Cannot proceed with backtest.")
            print("❌ خطا: هیچ فایل دیتاست یافت نشد")
            print("💡 لطفاً مطمئن شوید که prepare_features_03.py اجرا شده است")
            return
            
        model_file = find_latest_file(
            os.path.join(models_path, 'optimized_model_*.joblib'),
            "optimized model"
        )
        if not model_file:
            # fallback به مدل‌های قدیمی
            model_file = find_latest_file(
                os.path.join(models_path, 'random_forest_model_*.joblib'),
                "random forest model"
            )
            
        scaler_file = find_latest_file(
            os.path.join(models_path, 'scaler_optimized_*.joblib'),
            "optimized scaler"
        )
        if not scaler_file:
            # fallback به scaler قدیمی
            scaler_file = find_latest_file(
                os.path.join(models_path, 'scaler_*.joblib'),
                "scaler"
            )
        
        if not model_file or not scaler_file:
            logging.error("❌ Required model or scaler files not found")
            print("❌ خطا: فایل‌های مدل یا scaler یافت نشد")
            print("💡 لطفاً مطمئن شوید که train_model_04.py اجرا شده است")
            return
            
        # بارگذاری فایل‌ها
        logging.info(f"📁 Loading dataset: {os.path.basename(feature_file)}")
        df_full = pd.read_parquet(feature_file)
        
        logging.info(f"🤖 Loading model: {os.path.basename(model_file)}")
        model_data = joblib.load(model_file)
        
        # بررسی نوع مدل (optimized یا قدیمی)
        if isinstance(model_data, dict) and 'model' in model_data:
            # مدل بهینه‌شده
            model = model_data['model']
            optimal_threshold = model_data.get('optimal_threshold', 0.5)
            logging.info(f"✅ Optimized model loaded with threshold: {optimal_threshold:.4f}")
        else:
            # مدل قدیمی
            model = model_data
            optimal_threshold = 0.5
            logging.info("⚠️ Legacy model loaded, using default threshold: 0.5")
        
        logging.info(f"📏 Loading scaler: {os.path.basename(scaler_file)}")
        scaler = joblib.load(scaler_file)
        
        logging.info("✅ Data files, model and scaler loaded successfully.")
        
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        print(f"❌ خطا در بارگذاری فایل‌ها: {e}")
        return

    # بررسی نمادهای موجود
    if df_full.index.nlevels > 1:
        available_symbols = df_full.index.get_level_values('symbol').unique().tolist()
    else:
        logging.error("❌ Dataset format is not compatible (missing multi-index)")
        print("❌ فرمت دیتاست سازگار نیست")
        return
        
    print(f"\n📊 Available symbols in dataset: {available_symbols}")
    symbol_to_test = input("💱 Which symbol to test? (e.g. BTC/USDT): ").upper().strip()
    
    if symbol_to_test not in available_symbols:
        print(f"❌ Symbol '{symbol_to_test}' not found in dataset")
        print(f"✅ Available symbols: {', '.join(available_symbols)}")
        return

    try:
        available_timeframes = df_full.loc[symbol_to_test].index.get_level_values('timeframe').unique().tolist()
        print(f"⏱️ Available timeframes for {symbol_to_test}: {available_timeframes}")
        timeframe_to_test = input("🕐 Which timeframe to test? (e.g. 1h): ").lower().strip()
        
        if timeframe_to_test not in available_timeframes:
            print(f"❌ Timeframe '{timeframe_to_test}' not found")
            print(f"✅ Available timeframes: {', '.join(available_timeframes)}")
            return
            
    except KeyError as e:
        logging.error(f"Symbol/timeframe combination error: {e}")
        print(f"❌ خطا در دسترسی به داده‌های نماد")
        return

    try:
        df = df_full.loc[(symbol_to_test, timeframe_to_test)].copy()
        logging.info(f"📈 Dataset for {symbol_to_test} {timeframe_to_test}: {len(df)} records")
        
        if len(df) < TARGET_FUTURE_PERIODS * 2:
            logging.warning(f"⚠️ Insufficient data for meaningful backtest")
            print(f"⚠️ داده‌های ناکافی برای بک‌تست معنادار")
            
    except KeyError:
        logging.error(f"Symbol/timeframe combination not found.")
        print(f"❌ ترکیب نماد/تایم‌فریم یافت نشد")
        return
        
    logging.info(f"🚀 Starting backtest for {symbol_to_test} on {timeframe_to_test}...")

    # شروع بک‌تست
    capital = INITIAL_CAPITAL
    position_open = False
    entry_price = 0
    entry_index = -1
    entry_date = None
    trade_history = []

    # استخراج ویژگی‌ها و پیش‌بینی
    feature_columns = [col for col in df.columns if col not in ['target', 'timestamp']]
    if not feature_columns:
        logging.error("❌ No feature columns found in dataset")
        print("❌ هیچ ویژگی در دیتاست یافت نشد")
        return
        
    X = df[feature_columns]
    
    try:
        X_scaled = scaler.transform(X)
        if hasattr(model, 'predict_proba'):
            # استفاده از threshold بهینه برای مدل‌های جدید
            y_prob = model.predict_proba(X_scaled)[:, 1]
            df['prediction'] = (y_prob >= optimal_threshold).astype(int)
            df['prediction_prob'] = y_prob
            logging.info(f"✅ Using optimized threshold: {optimal_threshold:.4f}")
        else:
            df['prediction'] = model.predict(X_scaled)
            df['prediction_prob'] = 0.5  # پیش‌فرض
            
    except Exception as pred_error:
        logging.error(f"Error in model prediction: {pred_error}")
        print(f"❌ خطا در پیش‌بینی مدل: {pred_error}")
        return
    
    # شبیه‌سازی معاملات
    for i in range(len(df) - TARGET_FUTURE_PERIODS):
        current_row = df.iloc[i]
        
        if not position_open and current_row['prediction'] == 1:
            position_open = True
            entry_price = current_row['close']
            entry_index = i
            entry_date = df.index[i]
            logging.info(f"🟢 Entry at {entry_date}: price ${entry_price:.4f}")

        elif position_open and (i >= entry_index + TARGET_FUTURE_PERIODS):
            exit_price = current_row['close']
            exit_date = df.index[i]
            pnl_percent = (exit_price - entry_price) / entry_price
            
            # تعیین دلیل خروج
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
            logging.info(f"🔴 Exit at {exit_date}: price ${exit_price:.4f}, P/L: {pnl_percent:.2%}, Reason: {exit_reason}")

    # محاسبه نتایج نهایی
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
    
    # نمایش نتایج
    print("\n" + "="*50 + "\n      Strategy Performance Report\n" + "="*50)
    for key, value in report_data.items():
        if key != 'trade_history': 
            print(f"{key:<20} {value}")
    print("="*50)
    
    # تولید گزارش
    generate_report_file(report_data, symbol_to_test, timeframe_to_test)
    
    # تولید نمودارها
    try:
        generate_visualizations(df, trade_history, symbol_to_test, timeframe_to_test, report_subfolder_path)
    except Exception as viz_error:
        logging.warning(f"Error generating visualizations: {viz_error}")
    
    print(f"\n✅ Backtest completed successfully!")
    print(f"📁 Reports saved to: {report_subfolder_path}")

def run_enhanced_backtest(features_path: str, models_path: str):
    """Run enhanced backtest with multi-symbol and multi-timeframe support - اصلاح شده"""
    logging.info("="*70)
    logging.info("Starting Enhanced Backtest Strategy")
    logging.info("="*70)
    
    try:
        # بارگذاری داده‌ها با error handling بهتر
        feature_file = find_latest_file(
            os.path.join(features_path, 'final_dataset_for_training_*.parquet'),
            "dataset file"
        )
        if not feature_file:
            logging.error("❌ No dataset file found")
            print("❌ فایل دیتاست یافت نشد")
            return
            
        logging.info(f"Loading dataset: {os.path.basename(feature_file)}")
        df_full = pd.read_parquet(feature_file)
        
        # بارگذاری مدل
        model_file = find_latest_file(
            os.path.join(models_path, 'optimized_model_*.joblib'),
            "optimized model"
        )
        if not model_file:
            model_file = find_latest_file(
                os.path.join(models_path, 'random_forest_model_*.joblib'),
                "random forest model"
            )
            
        scaler_file = find_latest_file(
            os.path.join(models_path, 'scaler_optimized_*.joblib'),
            "optimized scaler"
        )
        if not scaler_file:
            scaler_file = find_latest_file(
                os.path.join(models_path, 'scaler_*.joblib'),
                "scaler"
            )
        
        if not model_file or not scaler_file:
            logging.error("❌ Required model or scaler files not found")
            print("❌ فایل‌های مدل یا scaler یافت نشد")
            return
        
        # بارگذاری مدل و scaler
        model_data = joblib.load(model_file)
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            optimal_threshold = model_data.get('optimal_threshold', 0.5)
        else:
            model = model_data
            optimal_threshold = 0.5
            
        scaler = joblib.load(scaler_file)
        logging.info("✅ Model and scaler loaded successfully.")
        
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        print(f"❌ خطا در بارگذاری فایل‌ها: {e}")
        return

    # انتخاب نمادها و تایم‌فریم‌ها
    selected_symbols, selected_timeframes = select_symbols_and_timeframes(df_full)
    if not selected_symbols or not selected_timeframes:
        return
    
    # ذخیره نتایج کلی
    all_results = []
    
    # اجرای بک‌تست برای هر ترکیب
    for symbol in selected_symbols:
        for timeframe in selected_timeframes:
            try:
                df = df_full.loc[(symbol, timeframe)].copy()
                if len(df) < TARGET_FUTURE_PERIODS * 2:
                    logging.warning(f"Insufficient data for {symbol}/{timeframe}. Skipping...")
                    continue
                
                logging.info(f"\n{'='*50}")
                logging.info(f"Backtesting {symbol} on {timeframe}")
                logging.info(f"Records: {len(df)}")
                
                # مقداردهی اولیه متغیرهای بک‌تست
                capital = INITIAL_CAPITAL
                equity_curve = [capital]
                position_open = False
                entry_price = 0
                entry_index = -1
                entry_date = None
                trade_history = []
                
                # پیش‌بینی‌های مدل
                feature_columns = [col for col in df.columns if col not in ['target', 'timestamp']]
                X = df[feature_columns]
                X_scaled = scaler.transform(X)
                
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_scaled)[:, 1]
                    df['prediction'] = (y_prob >= optimal_threshold).astype(int)
                    df['prediction_proba'] = y_prob
                else:
                    df['prediction'] = model.predict(X_scaled)
                    df['prediction_proba'] = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # شبیه‌سازی معاملات
                for i in range(len(df) - TARGET_FUTURE_PERIODS):
                    current_row = df.iloc[i]
                    
                    # منطق ورود
                    if not position_open and current_row['prediction'] == 1:
                        position_open = True
                        entry_price = current_row['close']
                        entry_index = i
                        entry_date = df.index[i]
                        logging.info(f"🟢 Entry at {entry_date}: ${entry_price:.4f}")
                    
                    # منطق خروج
                    elif position_open:
                        exit_condition = False
                        exit_reason = ""
                        
                        # بررسی شرایط مختلف خروج
                        if i >= entry_index + TARGET_FUTURE_PERIODS:
                            exit_condition = True
                            exit_reason = "Target period reached"
                        elif (current_row['close'] - entry_price) / entry_price < -0.05:  # 5% stop loss
                            exit_condition = True
                            exit_reason = "Stop loss (-5%)"
                        elif (current_row['close'] - entry_price) / entry_price > 0.10:  # 10% take profit
                            exit_condition = True
                            exit_reason = "Take profit (+10%)"
                        elif current_row['prediction'] == 0 and i > entry_index + 5:  # تغییر سیگنال
                            exit_condition = True
                            exit_reason = "Signal reversed"
                        
                        if exit_condition:
                            exit_price = current_row['close']
                            exit_date = df.index[i]
                            pnl_percent = (exit_price - entry_price) / entry_price
                            
                            trade_history.append({
                                'entry_date': entry_date,
                                'entry_price': entry_price,
                                'exit_date': exit_date,
                                'exit_price': exit_price,
                                'pnl': pnl_percent,
                                'exit_reason': exit_reason
                            })
                            
                            # بروزرسانی سرمایه
                            trade_amount = capital * TRADE_SIZE_PERCENT
                            profit_loss = trade_amount * pnl_percent
                            capital += profit_loss
                            equity_curve.append(capital)
                            
                            position_open = False
                            entry_index = -1
                            
                            emoji = "✅" if pnl_percent > 0 else "❌"
                            logging.info(f"{emoji} Exit at {exit_date}: ${exit_price:.4f}, "
                                       f"P&L: {pnl_percent:.2%}, Reason: {exit_reason}")
                
                # محاسبه نتایج نهایی
                final_capital = capital
                total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
                num_trades = len(trade_history)
                wins = [t for t in trade_history if t['pnl'] > 0]
                losses = [t for t in trade_history if t['pnl'] <= 0]
                win_rate = len(wins) / num_trades if num_trades > 0 else 0
                
                # ذخیره نتایج
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'total_return': total_return,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'final_capital': final_capital
                }
                all_results.append(result)
                
                # تولید داده‌های گزارش
                report_data = {
                    'Initial Capital': f"{INITIAL_CAPITAL:,.2f}",
                    'Final Capital': f"{final_capital:,.2f}",
                    'Total Return': f"{total_return:.2%}",
                    'Total Trades': num_trades,
                    'Winning Trades': len(wins),
                    'Losing Trades': len(losses),
                    'Win Rate': f"{win_rate:.2%}",
                    'trade_history': trade_history
                }
                
                # نمایش خلاصه
                print(f"\n{'='*40}")
                print(f"📊 {symbol} ({timeframe}) Results:")
                print(f"💰 Return: {total_return:.2%}")
                print(f"📈 Trades: {num_trades} (Win Rate: {win_rate:.1%})")
                print(f"💵 Final Capital: ${final_capital:,.2f}")
                
                # تولید تجسم‌ها و گزارش‌ها
                equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
                generate_enhanced_report(report_data, symbol, timeframe, df, equity_series)
                
                try:
                    generate_visualizations(df, trade_history, symbol, timeframe, report_subfolder_path)
                except Exception as viz_error:
                    logging.warning(f"Error generating visualizations for {symbol}/{timeframe}: {viz_error}")
                
            except KeyError:
                logging.error(f"Combination {symbol}/{timeframe} not found.")
                continue
            except Exception as e:
                logging.error(f"Error processing {symbol}/{timeframe}: {e}")
                continue
    
    # نمایش خلاصه کلی در صورت وجود چند تست
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("📊 OVERALL BACKTEST SUMMARY")
        print("="*60)
        print(f"{'Symbol':<15} {'TF':<5} {'Return':<10} {'Trades':<8} {'Win%':<8} {'Final $':<12}")
        print("-"*60)
        
        total_return_sum = 0
        for r in all_results:
            print(f"{r['symbol']:<15} {r['timeframe']:<5} "
                  f"{r['total_return']:>9.2%} {r['num_trades']:>7} "
                  f"{r['win_rate']:>7.1%} ${r['final_capital']:>11,.2f}")
            total_return_sum += r['total_return']
        
        print("-"*60)
        avg_return = total_return_sum / len(all_results) if all_results else 0
        print(f"Average Return: {avg_return:.2%}")
        print("="*60)
    
    print(f"\n✅ All reports and charts saved to:")
    print(f"   📁 {report_subfolder_path}")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Advanced Backtesting System v2.1")
    print("📊 Multi-Symbol & Multi-Timeframe Support")
    print("🔧 Fixed 'max() arg is an empty sequence' Error")
    print("="*70)
    
    choice = input("\nSelect mode:\n1. Enhanced Backtest (recommended)\n2. Simple Backtest\nChoice (1/2): ").strip()
    
    if choice == '2':
        run_simple_backtest(FEATURES_PATH, MODELS_PATH)
    else:
        run_enhanced_backtest(FEATURES_PATH, MODELS_PATH)