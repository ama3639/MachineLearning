#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت بک تست پیشرفته با معیارهای جامع و پشتیبانی از چند دارایی
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

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
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
    
    ax1.set_ylabel('Price')
    ax1.set_title(f'Price Chart and Trades - {symbol} ({timeframe})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Model prediction chart
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
        indices = [int(x.strip()) - 1 for x in symbol_choice.split(',') if x.strip().isdigit()]
        selected_symbols = [available_symbols[i] for i in indices if 0 <= i < len(available_symbols)]
    elif symbol_choice.isdigit():
        # Single number selection
        idx = int(symbol_choice) - 1
        if 0 <= idx < len(available_symbols):
            selected_symbols = [available_symbols[idx]]
    else:
        # Direct symbol name
        symbol_choice = symbol_choice.upper()
        if symbol_choice in available_symbols:
            selected_symbols = [symbol_choice]
    
    if not selected_symbols:
        print("❌ Invalid selection")
        return None, None
    
    # Timeframe selection for first symbol (to get available timeframes)
    first_symbol_tf = df_full.loc[selected_symbols[0]].index.get_level_values('timeframe').unique().tolist()
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
        indices = [int(x.strip()) - 1 for x in tf_choice.split(',') if x.strip().isdigit()]
        selected_timeframes = [first_symbol_tf[i] for i in indices if 0 <= i < len(first_symbol_tf)]
    elif tf_choice.isdigit():
        # Single number selection
        idx = int(tf_choice) - 1
        if 0 <= idx < len(first_symbol_tf):
            selected_timeframes = [first_symbol_tf[idx]]
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
    """Run simple backtest (backward compatibility)"""
    logging.info("--- Starting Strategy Backtest Process ---")
    
    try:
        list_of_feature_files = glob.glob(os.path.join(features_path, 'final_dataset_for_training_*.parquet'))
        latest_feature_file = max(list_of_feature_files, key=os.path.getctime)
        df_full = pd.read_parquet(latest_feature_file)
        
        latest_model_file = max(glob.glob(os.path.join(models_path, 'random_forest_model_*.joblib')), key=os.path.getctime)
        latest_scaler_file = max(glob.glob(os.path.join(models_path, 'scaler_*.joblib')), key=os.path.getctime)
        model = joblib.load(latest_model_file)
        scaler = joblib.load(latest_scaler_file)
        logging.info("Data files, model and scaler loaded successfully.")
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Error loading files: {e}")
        return

    available_symbols = df_full.index.get_level_values('symbol').unique().tolist()
    print("Available symbols in dataset:", available_symbols)
    symbol_to_test = input("Which symbol to test? (e.g. BTC/USDT): ").upper()
    
    available_timeframes = df_full.loc[symbol_to_test].index.get_level_values('timeframe').unique().tolist()
    print(f"Available timeframes for {symbol_to_test}:", available_timeframes)
    timeframe_to_test = input("Which timeframe to test? (e.g. 1h): ").lower()

    try:
        df = df_full.loc[(symbol_to_test, timeframe_to_test)].copy()
    except KeyError:
        logging.error(f"Symbol/timeframe combination not found.")
        return
        
    logging.info(f"Starting backtest for {symbol_to_test} on {timeframe_to_test}...")

    capital = INITIAL_CAPITAL
    position_open = False
    entry_price = 0
    entry_index = -1
    entry_date = None
    trade_history = []

    X = df.drop('target', axis=1)
    X_scaled = scaler.transform(X)
    df['prediction'] = model.predict(X_scaled)
    
    for i in range(len(df) - TARGET_FUTURE_PERIODS):
        current_row = df.iloc[i]
        
        if not position_open and current_row['prediction'] == 1:
            position_open = True
            entry_price = current_row['close']
            entry_index = i
            entry_date = df.index[i]
            logging.info(f"Entry at {entry_date}: price ${entry_price:.4f}")

        elif position_open and (i >= entry_index + TARGET_FUTURE_PERIODS):
            exit_price = current_row['close']
            exit_date = df.index[i]
            pnl_percent = (exit_price - entry_price) / entry_price
            
            # Determine exit reason
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
            logging.info(f"Exit at {exit_date}: price ${exit_price:.4f}, P/L: {pnl_percent:.2%}, Reason: {exit_reason}")

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
    
    print("\n" + "="*50 + "\n      Strategy Performance Report\n" + "="*50)
    for key, value in report_data.items():
        if key != 'trade_history': 
            print(f"{key:<20} {value}")
    print("="*50)
    
    generate_report_file(report_data, symbol_to_test, timeframe_to_test)

def run_enhanced_backtest(features_path: str, models_path: str):
    """Run enhanced backtest with multi-symbol and multi-timeframe support"""
    logging.info("="*70)
    logging.info("Starting Enhanced Backtest Strategy")
    logging.info("="*70)
    
    try:
        # Load data
        list_of_feature_files = glob.glob(os.path.join(features_path, 'final_dataset_for_training_*.parquet'))
        if not list_of_feature_files:
            raise FileNotFoundError("No dataset files found.")
        
        latest_feature_file = max(list_of_feature_files, key=os.path.getctime)
        logging.info(f"Loading dataset: {os.path.basename(latest_feature_file)}")
        df_full = pd.read_parquet(latest_feature_file)
        
        # Load model
        latest_model_file = max(glob.glob(os.path.join(models_path, 'random_forest_model_*.joblib')), 
                               key=os.path.getctime)
        latest_scaler_file = max(glob.glob(os.path.join(models_path, 'scaler_*.joblib')), 
                                key=os.path.getctime)
        
        model = joblib.load(latest_model_file)
        scaler = joblib.load(latest_scaler_file)
        logging.info("Model and scaler loaded successfully.")
        
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Error loading files: {e}")
        return

    # Select symbols and timeframes
    selected_symbols, selected_timeframes = select_symbols_and_timeframes(df_full)
    if not selected_symbols or not selected_timeframes:
        return
    
    # Overall results storage
    all_results = []
    
    # Run backtest for each combination
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
                
                # Initialize backtest variables
                capital = INITIAL_CAPITAL
                equity_curve = [capital]
                position_open = False
                entry_price = 0
                entry_index = -1
                entry_date = None
                trade_history = []
                
                # Model predictions
                feature_columns = [col for col in df.columns if col != 'target']
                X = df[feature_columns]
                X_scaled = scaler.transform(X)
                df['prediction'] = model.predict(X_scaled)
                df['prediction_proba'] = model.predict_proba(X_scaled)[:, 1]
                
                # Trading simulation
                for i in range(len(df) - TARGET_FUTURE_PERIODS):
                    current_row = df.iloc[i]
                    
                    # Entry logic
                    if not position_open and current_row['prediction'] == 1:
                        position_open = True
                        entry_price = current_row['close']
                        entry_index = i
                        entry_date = df.index[i]
                        logging.info(f"🟢 Entry at {entry_date}: ${entry_price:.4f}")
                    
                    # Exit logic
                    elif position_open:
                        exit_condition = False
                        exit_reason = ""
                        
                        # Check various exit conditions
                        if i >= entry_index + TARGET_FUTURE_PERIODS:
                            exit_condition = True
                            exit_reason = "Target period reached"
                        elif (current_row['close'] - entry_price) / entry_price < -0.05:  # 5% stop loss
                            exit_condition = True
                            exit_reason = "Stop loss (-5%)"
                        elif (current_row['close'] - entry_price) / entry_price > 0.10:  # 10% take profit
                            exit_condition = True
                            exit_reason = "Take profit (+10%)"
                        elif current_row['prediction'] == 0 and i > entry_index + 5:  # Signal changed
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
                            
                            # Update capital
                            trade_amount = capital * TRADE_SIZE_PERCENT
                            profit_loss = trade_amount * pnl_percent
                            capital += profit_loss
                            equity_curve.append(capital)
                            
                            position_open = False
                            entry_index = -1
                            
                            emoji = "✅" if pnl_percent > 0 else "❌"
                            logging.info(f"{emoji} Exit at {exit_date}: ${exit_price:.4f}, "
                                       f"P&L: {pnl_percent:.2%}, Reason: {exit_reason}")
                
                # Calculate final results
                final_capital = capital
                total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
                num_trades = len(trade_history)
                wins = [t for t in trade_history if t['pnl'] > 0]
                losses = [t for t in trade_history if t['pnl'] <= 0]
                win_rate = len(wins) / num_trades if num_trades > 0 else 0
                
                # Store results
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'total_return': total_return,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'final_capital': final_capital
                }
                all_results.append(result)
                
                # Generate report data
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
                
                # Display summary
                print(f"\n{'='*40}")
                print(f"📊 {symbol} ({timeframe}) Results:")
                print(f"💰 Return: {total_return:.2%}")
                print(f"📈 Trades: {num_trades} (Win Rate: {win_rate:.1%})")
                print(f"💵 Final Capital: ${final_capital:,.2f}")
                
                # Generate visualizations and reports
                equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
                generate_enhanced_report(report_data, symbol, timeframe, df, equity_series)
                generate_visualizations(df, trade_history, symbol, timeframe, report_subfolder_path)
                
            except KeyError:
                logging.error(f"Combination {symbol}/{timeframe} not found.")
                continue
            except Exception as e:
                logging.error(f"Error processing {symbol}/{timeframe}: {e}")
                continue
    
    # Display overall summary if multiple tests
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
        avg_return = total_return_sum / len(all_results)
        print(f"Average Return: {avg_return:.2%}")
        print("="*60)
    
    print(f"\n✅ All reports and charts saved to:")
    print(f"   📁 {report_subfolder_path}")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Advanced Backtesting System v2.0")
    print("📊 Multi-Symbol & Multi-Timeframe Support")
    print("="*70)
    
    choice = input("\nSelect mode:\n1. Enhanced Backtest (recommended)\n2. Simple Backtest\nChoice (1/2): ").strip()
    
    if choice == '2':
        run_simple_backtest(FEATURES_PATH, MODELS_PATH)
    else:
        run_enhanced_backtest(FEATURES_PATH, MODELS_PATH)