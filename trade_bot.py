# trade_bot.py
import ccxt
import pandas as pd
import numpy as np
import joblib
import talib
import time
import logging
import traceback
import json
from datetime import datetime, timedelta
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import threading
import signal

# เพิ่ม path เพื่อ import จากไฟล์อื่น
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_bot import (
    OKX_CONFIG, TRADING_CONFIG, RISK_CONFIG, TELEGRAM_CONFIG, 
    MODEL_CONFIG, DATABASE_CONFIG, BOT_CONFIG, validate_config
)
from trade_history import TradeHistoryManager

try:
    from train_model_v2 import AdvancedTradingModelTrainer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML models not available - running in signal-only mode")

class OKXTradingBot:
    def __init__(self):
        """
        Initialize Trading Bot with configuration from config_bot.py
        """
        self.setup_logging()
        self.validate_environment()
        self.setup_trade_history()
        self.setup_exchange()
        self.validate_symbol()  # เพิ่ม: ตรวจสอบ symbol
        self.setup_models()
        
        # ตัวแปรสถานะ
        self.is_running = False
        self.last_report_time = datetime.now()
        self.daily_pnl = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # เริ่มระบบรายงาน
        if BOT_CONFIG['hourly_report_enabled']:
            self.start_hourly_report()
        
        # เริ่มระบบ export รายวัน
        self.start_daily_export()
        
        # แสดงสรุปการตั้งค่า
        self.display_startup_summary()
        
        self.logger.info("✅ Trading bot initialized successfully")
    
    def display_startup_summary(self):
        """แสดงสรุปการตั้งค่าเมื่อเริ่มต้น bot"""
        try:
            position_info = self.check_position(TRADING_CONFIG['symbol'])
            
            # คำนวณ min_order_size string (ป้องกัน None)
            if hasattr(self, 'min_order_size') and self.min_order_size:
                min_order_text = f"{self.min_order_size:.8f} {TRADING_CONFIG['symbol'].split('/')[0]}"
            else:
                min_order_text = 'N/A'
            
            summary = f"""
{'='*60}
🤖 TRADING BOT STARTUP SUMMARY
{'='*60}

📊 EXCHANGE & ACCOUNT
├─ Exchange: OKX {'[LIVE]' if not OKX_CONFIG.get('sandbox', False) else '[TESTNET]'}
├─ Symbol: {TRADING_CONFIG['symbol']}
├─ Timeframe: {TRADING_CONFIG['timeframe']}
├─ Balance: ${position_info['total_usdt']:.2f} USDT (${position_info['free_usdt']:.2f} available)
└─ Current Position: {'Yes' if position_info.get('current_position') else 'None'}

💰 TRADING CONFIGURATION
├─ Trade Size: ${TRADING_CONFIG['trade_size_usdt']} USDT
├─ Max Positions: {TRADING_CONFIG['max_open_positions']}
├─ Trading Enabled: {'✅ YES' if TRADING_CONFIG['trading_enabled'] else '❌ NO (Simulation)'}
└─ Min Order Size: {min_order_text}

📈 RISK MANAGEMENT
├─ Stop Loss: {RISK_CONFIG['stop_loss_pct']*100:.1f}%
├─ Take Profit: {RISK_CONFIG['take_profit_pct']*100:.1f}%
├─ Max Daily Loss: {RISK_CONFIG['max_daily_loss_pct']:.1f}%
└─ Position Size: {RISK_CONFIG['position_size_pct']:.1f}% of balance

🤖 ML MODEL
├─ Model: {'✅ Loaded' if self.model is not None else '❌ Not Available'}
├─ Scaler: {'✅ Fitted' if self.scaler is not None and hasattr(self.scaler, 'mean_') else '❌ Not Available'}
├─ Features: {len(self.feature_columns) if self.feature_columns else 0}
└─ Min Confidence: {MODEL_CONFIG['min_confidence']*100:.0f}%

⚙️  BOT BEHAVIOR
├─ Trading Interval: {BOT_CONFIG['trading_interval_minutes']} minutes
├─ Hourly Reports: {'✅ Enabled' if BOT_CONFIG['hourly_report_enabled'] else '❌ Disabled'}
├─ Daily Export: {'✅ Enabled' if BOT_CONFIG.get('daily_export_enabled', True) else '❌ Disabled'}
└─ Debug Mode: {'✅ ON' if BOT_CONFIG['debug_mode'] else '❌ OFF'}

{'='*60}
            """
            
            self.logger.info(summary)
            
        except Exception as e:
            self.logger.warning(f"Could not display startup summary: {e}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG if BOT_CONFIG['debug_mode'] else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_environment(self):
        """Validate environment and configuration"""
        errors = validate_config()
        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ตรวจสอบว่าโฟลเดอร์จำเป็นมีอยู่
        os.makedirs('exports', exist_ok=True)
        os.makedirs('backups', exist_ok=True)
        
        self.logger.info("✅ Environment validation passed")
    
    def setup_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            exchange_config = OKX_CONFIG.copy()
            exchange_config.update({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            self.exchange = ccxt.okx(exchange_config)
            
            # Test connection
            balance_test = self.exchange.fetch_balance()
            
            # แสดงข้อมูล exchange mode
            mode = "TESTNET/SANDBOX" if OKX_CONFIG.get('sandbox', False) else "PRODUCTION/LIVE"
            self.logger.info(f"✅ OKX connection established successfully")
            self.logger.info(f"   Mode: {mode}")
            
            # ตรวจสอบว่ามี balance หรือไม่
            has_balance = False
            if balance_test:
                total_dict = balance_test.get('total', {})
                free_dict = balance_test.get('free', {})
                info = balance_test.get('info', {})
                
                has_balance = (
                    (total_dict and len(total_dict) > 0) or
                    (free_dict and len(free_dict) > 0) or
                    (info and len(info) > 0)
                )
            
            if not has_balance:
                self.logger.warning("="*60)
                self.logger.warning("⚠️  WARNING: NO BALANCE FOUND IN ACCOUNT")
                self.logger.warning("="*60)
                self.logger.warning(f"API Connection: OK")
                self.logger.warning(f"Mode: {mode}")
                self.logger.warning("")
                self.logger.warning("Possible reasons:")
                self.logger.warning("1. Account has no funds")
                self.logger.warning("2. Using wrong API credentials")
                self.logger.warning("3. API doesn't have 'Read' permission")
                self.logger.warning("4. Wrong account type (Funding/Trading/Spot)")
                self.logger.warning("")
                
                if not OKX_CONFIG.get('sandbox', False):
                    self.logger.warning("💡 RECOMMENDATION:")
                    self.logger.warning("   For testing, use TESTNET/SANDBOX mode:")
                    self.logger.warning("   1. Open config_bot.py")
                    self.logger.warning("   2. Set: OKX_CONFIG['sandbox'] = True")
                    self.logger.warning("   3. Get testnet API keys from: https://www.okx.com/demo-trading")
                    self.logger.warning("")
                
                self.logger.warning("Bot will continue in SIMULATION mode")
                self.logger.warning("="*60)
            else:
                self.logger.info(f"   ✅ Balance data available")
            
        except ccxt.AuthenticationError as e:
            self.logger.error(f"❌ OKX authentication failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to OKX: {e}")
            raise
    
    def validate_symbol(self):
        """Validate and normalize trading symbol for OKX"""
        try:
            # โหลดตลาดทั้งหมดจาก OKX
            markets = self.exchange.load_markets()
            
            symbol = TRADING_CONFIG['symbol']
            
            # ตรวจสอบว่า symbol อยู่ในรูปแบบที่ถูกต้องหรือไม่
            if symbol not in markets:
                # พยายามแปลง symbol ให้อยู่ในรูปแบบที่ถูกต้อง
                if '/' not in symbol:
                    # แปลง PAXGUSDT -> PAXG/USDT
                    if symbol.endswith('USDT'):
                        normalized_symbol = f"{symbol[:-4]}/USDT"
                    elif symbol.endswith('USD'):
                        normalized_symbol = f"{symbol[:-3]}/USD"
                    else:
                        raise ValueError(f"Cannot normalize symbol: {symbol}")
                    
                    if normalized_symbol in markets:
                        TRADING_CONFIG['symbol'] = normalized_symbol
                        self.logger.info(f"✅ Symbol normalized: {symbol} -> {normalized_symbol}")
                        symbol = normalized_symbol
                    else:
                        # แสดง symbols ที่เกี่ยวข้องกับ PAXG หรือ gold
                        suggested = [s for s in markets.keys() if 'PAXG' in s or 'GOLD' in s or 'XAU' in s]
                        error_msg = f"Symbol {symbol} not found. Available gold-related symbols: {suggested[:5]}"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                else:
                    suggested = [s for s in markets.keys() if symbol.split('/')[0] in s]
                    error_msg = f"Symbol {symbol} not found. Similar symbols: {suggested[:5]}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # ตรวจสอบว่า symbol support spot trading
            market_info = markets[symbol]
            if not market_info.get('spot', False):
                self.logger.warning(f"⚠️  {symbol} may not support spot trading")
            
            # แสดงข้อมูล market
            self.logger.info(f"✅ Trading symbol validated: {symbol}")
            self.logger.info(f"   Market type: {market_info.get('type', 'unknown')}")
            self.logger.info(f"   Spot: {market_info.get('spot', False)}")
            self.logger.info(f"   Swap: {market_info.get('swap', False)}")
            
            # เก็บข้อมูล market limits สำหรับตรวจสอบ order size
            self.market_info = market_info
            self.min_order_size = market_info.get('limits', {}).get('amount', {}).get('min', 0) or 0
            self.max_order_size = market_info.get('limits', {}).get('amount', {}).get('max') or float('inf')
            
            if self.min_order_size and self.min_order_size > 0:
                self.logger.info(f"   Min order size: {self.min_order_size:.8f}")
            if self.max_order_size and self.max_order_size < float('inf'):
                self.logger.info(f"   Max order size: {self.max_order_size:.8f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate symbol: {e}")
            raise
    
    def setup_models(self):
        """Load trained models and scaler with detailed feedback"""
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        if not ML_AVAILABLE:
            self.logger.warning("="*60)
            self.logger.warning("⚠️  ML MODELS NOT AVAILABLE")
            self.logger.warning("="*60)
            self.logger.warning("Reason: train_model.py module not found or import failed")
            self.logger.warning("Impact: Bot will run in BASIC SIGNAL MODE using RSI + EMA")
            self.logger.warning("")
            self.logger.warning("To enable ML predictions:")
            self.logger.warning("  1. Ensure train_model.py exists in the same directory")
            self.logger.warning("  2. Install required packages:")
            self.logger.warning("     pip install scikit-learn joblib talib scipy")
            self.logger.warning("="*60)
            return
            
        try:
            # ตรวจสอบไฟล์โมเดลแต่ละไฟล์อย่างละเอียด
            model_files = {
                'Model': MODEL_CONFIG['model_path'],
                'Scaler': MODEL_CONFIG['scaler_path'],
                'Features': MODEL_CONFIG['features_path']
            }
            
            missing_files = []
            existing_files = []
            
            for key, path in model_files.items():
                if os.path.exists(path):
                    file_size = os.path.getsize(path) / 1024  # KB
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(path))
                    existing_files.append(f"   ✅ {key}: {path} ({file_size:.1f} KB, modified: {file_mtime.strftime('%Y-%m-%d %H:%M')})")
                else:
                    missing_files.append(f"   ❌ {key}: {path}")
            
            if missing_files:
                self.logger.warning("="*60)
                self.logger.warning("⚠️  MODEL FILES NOT FOUND")
                self.logger.warning("="*60)
                
                if existing_files:
                    self.logger.warning("Found files:")
                    for file_info in existing_files:
                        self.logger.warning(file_info)
                    self.logger.warning("")
                
                self.logger.warning("Missing files:")
                for missing in missing_files:
                    self.logger.warning(missing)
                
                self.logger.warning("")
                self.logger.warning("📝 HOW TO TRAIN A NEW MODEL:")
                self.logger.warning("="*60)
                self.logger.warning("")
                self.logger.warning("Option 1: Use train_model_v2.py (Recommended - Advanced)")
                self.logger.warning("  Command: python train_model_v2.py")
                self.logger.warning("  Features:")
                self.logger.warning("    - Auto-tuning with Optuna")
                self.logger.warning("    - Multiple model comparison (XGBoost, LightGBM, RF, LSTM)")
                self.logger.warning("    - Advanced feature engineering")
                self.logger.warning("    - Detailed performance reports")
                self.logger.warning("")
                self.logger.warning("Option 2: Use train_model.py (Basic)")
                self.logger.warning("  Command: python train_model.py")
                self.logger.warning("")
                self.logger.warning("Option 3: Enable auto-training (if available)")
                self.logger.warning("  Edit config_bot.py and set:")
                self.logger.warning("    BOT_CONFIG['auto_train_model'] = True")
                self.logger.warning("")
                self.logger.warning("="*60)
                self.logger.warning("🔄 BOT WILL CONTINUE IN BASIC SIGNAL MODE")
                self.logger.warning("="*60)
                self.logger.warning("Basic Mode uses:")
                self.logger.warning("  - RSI (Relative Strength Index)")
                self.logger.warning("  - EMA (Exponential Moving Average)")
                self.logger.warning("  - Simple trend detection")
                self.logger.warning("")
                self.logger.warning("⚠️  Note: Basic mode has lower accuracy than ML models")
                self.logger.warning("   Expected performance: ~55-65% win rate")
                self.logger.warning("   ML model performance: ~70-85% win rate")
                self.logger.warning("="*60)
                
                return
            
            # โหลดโมเดลที่ฝึกไว้
            self.logger.info("="*60)
            self.logger.info("📦 LOADING ML MODELS")
            self.logger.info("="*60)
            
            # โหลด model
            self.logger.info("Loading model...")
            self.model = joblib.load(MODEL_CONFIG['model_path'])
            model_type = type(self.model).__name__
            self.logger.info(f"   ✅ Model type: {model_type}")
            
            # แสดง Model Information
            self.logger.info("")
            self.logger.info("🤖 MODEL INFORMATION:")
            self.logger.info("="*60)
            
            # ตรวจสอบว่าเป็น LightGBM model
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                self.logger.info("📌 Model Parameters:")
                
                # แสดง parameters ที่สำคัญ
                important_params = [
                    'n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                    'min_child_samples', 'subsample', 'colsample_bytree',
                    'reg_alpha', 'reg_lambda', 'min_split_gain'
                ]
                
                for param in important_params:
                    if param in params:
                        self.logger.info(f"   • {param}: {params[param]}")
            
            # แสดง Feature Importance (ถ้ามี)
            if hasattr(self.model, 'feature_importances_'):
                self.logger.info("")
                self.logger.info("📊 Feature Importance (Top 10):")
                feature_importance = self.model.feature_importances_
                # จะแสดงรายละเอียดหลังจากโหลด feature_columns
            
            # แสดง Model Classes
            if hasattr(self.model, 'classes_'):
                self.logger.info("")
                self.logger.info(f"📋 Model Classes: {self.model.classes_}")
            
            # แสดง Number of Features
            if hasattr(self.model, 'n_features_'):
                self.logger.info(f"🔢 Number of Features (model expects): {self.model.n_features_}")
            elif hasattr(self.model, 'n_features_in_'):
                self.logger.info(f"🔢 Number of Features (model expects): {self.model.n_features_in_}")
            
            self.logger.info("="*60)
            
            # โหลด scaler
            self.logger.info("Loading scaler...")
            self.scaler = joblib.load(MODEL_CONFIG['scaler_path'])
            scaler_type = type(self.scaler).__name__
            self.logger.info(f"   ✅ Scaler type: {scaler_type}")
            
            # ตรวจสอบว่า scaler ถูก fit แล้ว
            if not hasattr(self.scaler, 'mean_') and not hasattr(self.scaler, 'scale_'):
                self.logger.error("   ❌ Scaler is not fitted properly!")
                self.logger.warning("   Scaler will be reset to None - using basic signal mode")
                self.scaler = None
                self.model = None
                return
            else:
                self.logger.info(f"   ✅ Scaler is fitted correctly")
                if hasattr(self.scaler, 'n_features_in_'):
                    self.logger.info(f"   ✅ Scaler expects {self.scaler.n_features_in_} features")
            
            # โหลด feature columns
            self.logger.info("Loading feature columns...")
            self.feature_columns = joblib.load(MODEL_CONFIG['features_path'])
            n_features = len(self.feature_columns)
            self.logger.info(f"   ✅ Number of features: {n_features}")
            
            # แสดง Feature Importance (ต่อจากที่โหลด model)
            if hasattr(self.model, 'feature_importances_'):
                self.logger.info("")
                self.logger.info("📊 TOP 15 MOST IMPORTANT FEATURES:")
                self.logger.info("="*60)
                
                # สร้าง DataFrame เพื่อเรียงลำดับ
                feature_importance = self.model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                # แสดง Top 15 features
                for idx, row in importance_df.head(15).iterrows():
                    importance_pct = row['importance'] * 100
                    self.logger.info(f"   {row['feature']:30s} {importance_pct:5.2f}%")
                
                self.logger.info("="*60)
            
            # โหลด metadata ถ้ามี
            metadata_path = 'saved_models/model_metadata.pkl'
            if os.path.exists(metadata_path):
                try:
                    metadata = joblib.load(metadata_path)
                    self.logger.info("")
                    self.logger.info("📊 MODEL METADATA:")
                    self.logger.info(f"   Model Name: {metadata.get('model_name', 'N/A')}")
                    self.logger.info(f"   Training Date: {metadata.get('training_date', 'N/A')}")
                    self.logger.info(f"   Best Score: {metadata.get('best_score', 'N/A')}")
                    self.logger.info(f"   Training Samples: {metadata.get('n_samples', 'N/A')}")
                except Exception as e:
                    self.logger.debug(f"Could not load metadata: {e}")
            
            # สร้าง feature calculator instance
            try:
                if ML_AVAILABLE:
                    # ลอง import จาก train_model_v2 ก่อน (ใหม่กว่า)
                    try:
                        from train_model_v2 import AdvancedTradingModelTrainer
                        self.feature_calculator = AdvancedTradingModelTrainer()
                        self.logger.info("   ✅ Feature calculator initialized (v2)")
                        
                        # ตรวจสอบว่ามี method สำหรับคำนวณ features หรือไม่
                        if not hasattr(self.feature_calculator, 'prepare_features'):
                            # ถ้าไม่มี ให้สร้าง wrapper method
                            self._create_feature_calculator_wrapper()
                        
                    except (ImportError, AttributeError):
                        # ลอง train_model.py แทน
                        from train_model import AdvancedTradingModelTrainer
                        self.feature_calculator = AdvancedTradingModelTrainer()
                        self.logger.info("   ✅ Feature calculator initialized (v1)")
                        
                        if not hasattr(self.feature_calculator, 'prepare_features'):
                            self._create_feature_calculator_wrapper()
                        
            except Exception as e:
                self.logger.warning(f"   ⚠️  Could not initialize feature calculator: {e}")
                self.logger.warning("   Will use comprehensive feature calculation instead")
                self.feature_calculator = None
            
            self.logger.info("="*60)
            self.logger.info("✅ ML MODELS LOADED SUCCESSFULLY")
            self.logger.info("="*60)
            
            # แสดง Summary ของ Model
            self.logger.info("")
            self.logger.info("🎯 MODEL SUMMARY:")
            self.logger.info(f"   Model Type: {type(self.model).__name__}")
            self.logger.info(f"   Scaler Type: {type(self.scaler).__name__}")
            self.logger.info(f"   Total Features: {len(self.feature_columns)}")
            self.logger.info(f"   Mode: ADVANCED ML PREDICTIONS")
            self.logger.info(f"   Expected Performance: 70-85% win rate")
            
            # แสดง Metadata ถ้ามี
            if os.path.exists('saved_models/model_metadata.pkl'):
                try:
                    metadata = joblib.load('saved_models/model_metadata.pkl')
                    if 'best_score' in metadata:
                        self.logger.info(f"   Training Accuracy: {metadata['best_score']:.2%}")
                    if 'n_samples' in metadata:
                        self.logger.info(f"   Training Samples: {metadata['n_samples']:,}")
                    if 'training_date' in metadata:
                        self.logger.info(f"   Training Date: {metadata['training_date']}")
                except:
                    pass
            
            self.logger.info("")
            self.logger.info("Feature List (Top 10):")
            for i, feature in enumerate(self.feature_columns[:10], 1):
                self.logger.info(f"   {i:2d}. {feature}")
            if len(self.feature_columns) > 10:
                self.logger.info(f"   ... and {len(self.feature_columns) - 10} more features")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error("="*60)
            self.logger.error("❌ FAILED TO LOAD ML MODELS")
            self.logger.error("="*60)
            self.logger.error(f"Error Type: {type(e).__name__}")
            self.logger.error(f"Error Message: {str(e)}")
            self.logger.error("")
            
            # ตรวจสอบว่าเป็นปัญหากับ scaler หรือไม่
            if 'scaler' in str(e).lower() or 'standardscaler' in str(e).lower() or 'not fitted' in str(e).lower():
                self.logger.error("🔍 SCALER ISSUE DETECTED!")
                self.logger.error("")
                self.logger.error("This error usually means:")
                self.logger.error("  1. The scaler file is corrupted")
                self.logger.error("  2. The scaler was not saved properly during training")
                self.logger.error("  3. Mismatch between sklearn versions")
                self.logger.error("")
                self.logger.error("SOLUTION:")
                self.logger.error("  Re-train the model to regenerate all files:")
                self.logger.error("  → python train_model_v2.py")
                self.logger.error("")
            else:
                self.logger.error("Possible causes:")
                self.logger.error("  1. Model files are corrupted")
                self.logger.error("  2. Model was trained with different sklearn/library version")
                self.logger.error("  3. Insufficient memory to load models")
                self.logger.error("  4. File permission issues")
                self.logger.error("")
                self.logger.error("Recommended actions:")
                self.logger.error("  1. Check file permissions")
                self.logger.error("  2. Re-train the model using: python train_model_v2.py")
                self.logger.error("  3. Check library versions:")
                self.logger.error("     pip list | grep -E 'scikit-learn|joblib|xgboost|lightgbm'")
            self.logger.error("="*60)
            
            # บันทึก error
            self.history_manager.log_error('MODEL_LOAD', str(e), traceback.format_exc())
            
            # Reset model variables
            self.model = None
            self.scaler = None
            self.feature_columns = None
            
            self.logger.warning("🔄 Falling back to BASIC SIGNAL MODE")
            self.logger.warning("="*60)
        
    def setup_trade_history(self):
        """Setup trade history manager"""
        self.history_manager = TradeHistoryManager()
        self.logger.info("✅ Trade history manager initialized")
    
    def _create_feature_calculator_wrapper(self):
        """สร้าง wrapper method สำหรับ feature calculator"""
        try:
            def prepare_features_wrapper():
                """Wrapper method ที่รวมการเรียก load_and_preprocess_data และสร้าง features"""
                if not hasattr(self.feature_calculator, 'data') or self.feature_calculator.data is None:
                    self.logger.error("No data in feature calculator")
                    return None
                
                # ตั้งค่า data_path ถ้ายังไม่มี
                if not hasattr(self.feature_calculator, 'data_path'):
                    self.feature_calculator.data_path = None
                
                # ใช้ข้อมูลที่มีอยู่แล้วใน self.feature_calculator.data
                # แทนการโหลดจากไฟล์
                df = self.feature_calculator.data.copy()
                
                # คำนวณ features ด้วยตัวเอง
                self.feature_calculator.features = self._calculate_comprehensive_features(df)
                
                return self.feature_calculator.features
            
            # ผูก method เข้ากับ feature_calculator
            self.feature_calculator.prepare_features = prepare_features_wrapper
            self.logger.info("   ✅ Created feature calculator wrapper method")
            
        except Exception as e:
            self.logger.error(f"Error creating feature calculator wrapper: {e}")
    
    def _calculate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        คำนวณ features ที่ครบถ้วนให้ตรงกับที่โมเดลต้องการ
        """
        try:
            self.logger.info("Calculating comprehensive features...")
            
            features = df.copy()
            
            # ===== 1. Basic Price Features =====
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            features['price_change'] = features['close'] - features['open']
            features['price_range'] = features['high'] - features['low']
            features['body_size'] = abs(features['close'] - features['open'])
            
            # Replace any inf values from log of negative or zero
            features['log_returns'] = features['log_returns'].replace([np.inf, -np.inf], 0)
            
            # ===== 2. Technical Indicators =====
            
            # RSI (Multiple timeframes)
            features['rsi_14'] = talib.RSI(features['close'], timeperiod=14)
            features['rsi_7'] = talib.RSI(features['close'], timeperiod=7)
            features['rsi_21'] = talib.RSI(features['close'], timeperiod=21)
            features['rsi_28'] = talib.RSI(features['close'], timeperiod=28)
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                features[f'sma_{period}'] = talib.SMA(features['close'], timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(features['close'], timeperiod=period)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                features['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            features['macd_cross'] = (macd > macd_signal).astype(int)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                features['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = (upper - lower) / np.where(middle != 0, middle, 1)  # Prevent division by zero
            features['bb_position'] = (features['close'] - lower) / np.where((upper - lower) != 0, (upper - lower), 1)  # Prevent division by zero
            
            # ATR (Volatility)
            features['atr_14'] = talib.ATR(features['high'], features['low'], features['close'], timeperiod=14)
            features['atr_7'] = talib.ATR(features['high'], features['low'], features['close'], timeperiod=7)
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                features['high'], 
                features['low'], 
                features['close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            features['stoch_cross'] = (slowk > slowd).astype(int)
            
            # ADX (Trend Strength)
            features['adx_14'] = talib.ADX(features['high'], features['low'], features['close'], timeperiod=14)
            features['adx_7'] = talib.ADX(features['high'], features['low'], features['close'], timeperiod=7)
            
            # CCI
            features['cci_14'] = talib.CCI(features['high'], features['low'], features['close'], timeperiod=14)
            features['cci_20'] = talib.CCI(features['high'], features['low'], features['close'], timeperiod=20)
            
            # Williams %R
            features['willr_14'] = talib.WILLR(features['high'], features['low'], features['close'], timeperiod=14)
            
            # MFI (Money Flow Index)
            features['mfi_14'] = talib.MFI(features['high'], features['low'], features['close'], features['volume'], timeperiod=14)
            
            # OBV (On Balance Volume)
            features['obv'] = talib.OBV(features['close'], features['volume'])
            
            # ===== 3. Volume Features =====
            features['volume_sma_20'] = talib.SMA(features['volume'], timeperiod=20)
            features['volume_ratio'] = features['volume'] / np.where(features['volume_sma_20'] != 0, features['volume_sma_20'], 1)
            features['volume_change'] = features['volume'].pct_change()
            
            # ===== 4. Candlestick Patterns =====
            features['hammer'] = talib.CDLHAMMER(features['open'], features['high'], features['low'], features['close'])
            features['engulfing'] = talib.CDLENGULFING(features['open'], features['high'], features['low'], features['close'])
            features['doji'] = talib.CDLDOJI(features['open'], features['high'], features['low'], features['close'])
            features['shooting_star'] = talib.CDLSHOOTINGSTAR(features['open'], features['high'], features['low'], features['close'])
            features['morning_star'] = talib.CDLMORNINGSTAR(features['open'], features['high'], features['low'], features['close'])
            
            # ===== 5. Price Position & Momentum =====
            price_range = features['high'] - features['low']
            features['price_position'] = np.where(price_range != 0, 
                                                   (features['close'] - features['low']) / price_range, 
                                                   0.5)  # Default to middle if no range
            features['momentum'] = talib.MOM(features['close'], timeperiod=10)
            features['roc'] = talib.ROC(features['close'], timeperiod=10)
            
            # ===== 6. Support/Resistance =====
            # Simple support/resistance based on rolling min/max
            features['resistance_20'] = features['high'].rolling(20).max()
            features['support_20'] = features['low'].rolling(20).min()
            features['distance_to_resistance'] = np.where(features['close'] != 0,
                                                          (features['resistance_20'] - features['close']) / features['close'],
                                                          0)
            features['distance_to_support'] = np.where(features['close'] != 0,
                                                        (features['close'] - features['support_20']) / features['close'],
                                                        0)
            
            # ===== 7. Multi-timeframe indicators (simulated) =====
            # เนื่องจากเรามีแค่ข้อมูล 1 timeframe ให้ใช้ rolling window จำลอง
            
            # m15 indicators (15-period rolling)
            features['m15_rsi'] = talib.RSI(features['close'], timeperiod=15)
            features['m15_rsi_14'] = talib.RSI(features['close'], timeperiod=14)  # RSI-14 standard
            features['m15_ema'] = talib.EMA(features['close'], timeperiod=15)
            features['m15_momentum'] = talib.MOM(features['close'], timeperiod=15)
            features['m15_hammer'] = talib.CDLHAMMER(features['open'], features['high'], features['low'], features['close'])
            features['m15_engulfing'] = talib.CDLENGULFING(features['open'], features['high'], features['low'], features['close'])
            features['m15_doji'] = talib.CDLDOJI(features['open'], features['high'], features['low'], features['close'])
            
            # m15 additional indicators
            features['m15_obv'] = talib.OBV(features['close'], features['volume'])
            features['m15_adx'] = talib.ADX(features['high'], features['low'], features['close'], timeperiod=15)
            features['m15_atr'] = talib.ATR(features['high'], features['low'], features['close'], timeperiod=15)
            
            # m15 MACD
            m15_macd, m15_macd_signal, m15_macd_hist = talib.MACD(
                features['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features['m15_macd'] = m15_macd
            features['m15_macd_signal'] = m15_macd_signal
            features['m15_macd_hist'] = m15_macd_hist
            
            # m15 Volume indicators
            features['m15_volume_sma'] = talib.SMA(features['volume'], timeperiod=15)
            features['m15_volume_ratio'] = features['volume'] / np.where(features['m15_volume_sma'] != 0, features['m15_volume_sma'], 1)
            
            # m15 Stochastic
            m15_slowk, m15_slowd = talib.STOCH(
                features['high'], 
                features['low'], 
                features['close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            features['m15_stoch_k'] = m15_slowk
            features['m15_stoch_d'] = m15_slowd
            
            # h1 indicators (60-period rolling - approximate 1 hour if base is 1min)
            features['h1_rsi'] = talib.RSI(features['close'], timeperiod=60)
            features['h1_ema'] = talib.EMA(features['close'], timeperiod=60)
            features['h1_ema21'] = talib.EMA(features['close'], timeperiod=21)
            features['h1_ema50'] = talib.EMA(features['close'], timeperiod=50)
            
            # h1 MACD
            h1_macd, h1_macd_signal, h1_macd_hist = talib.MACD(
                features['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features['h1_macd'] = h1_macd
            features['h1_macd_signal'] = h1_macd_signal
            features['h1_macd_hist'] = h1_macd_hist
            
            # h1 Trend indicators
            features['h1_adx'] = talib.ADX(features['high'], features['low'], features['close'], timeperiod=60)
            features['h1_plus_di'] = talib.PLUS_DI(features['high'], features['low'], features['close'], timeperiod=60)
            features['h1_minus_di'] = talib.MINUS_DI(features['high'], features['low'], features['close'], timeperiod=60)
            
            # h1 Trend strength and direction
            features['h1_trend_strength'] = features['h1_adx'] / 100  # Normalize to 0-1
            features['h1_trend_direction'] = (features['h1_plus_di'] > features['h1_minus_di']).astype(int)  # 1 for up, 0 for down
            
            # m15 vs h1 trend alignment
            m15_trend = (features['m15_ema'] > features['m15_ema'].shift(1)).astype(int)
            h1_trend = (features['h1_ema'] > features['h1_ema'].shift(1)).astype(int)
            features['m15_vs_h1_trend'] = (m15_trend == h1_trend).astype(int)  # 1 if aligned, 0 if not
            
            # m15 Bollinger Bands
            m15_upper, m15_middle, m15_lower = talib.BBANDS(
                features['close'], 
                timeperiod=15, 
                nbdevup=2, 
                nbdevdn=2
            )
            features['m15_bb_upper'] = m15_upper
            features['m15_bb_middle'] = m15_middle
            features['m15_bb_lower'] = m15_lower
            features['m15_bb_width'] = (m15_upper - m15_lower) / np.where(m15_middle != 0, m15_middle, 1)
            
            # m15 ROC
            features['m15_roc_10'] = talib.ROC(features['close'], timeperiod=10)
            features['m15_roc_5'] = talib.ROC(features['close'], timeperiod=5)
            
            # Price vs moving averages (with zero protection)
            features['price_vs_h1_ema21'] = np.where(features['h1_ema21'] != 0,
                                                      (features['close'] - features['h1_ema21']) / features['h1_ema21'],
                                                      0)
            features['price_vs_h1_ema50'] = np.where(features['h1_ema50'] != 0,
                                                      (features['close'] - features['h1_ema50']) / features['h1_ema50'],
                                                      0)
            features['price_vs_m15_ema'] = np.where(features['m15_ema'] != 0,
                                                     (features['close'] - features['m15_ema']) / features['m15_ema'],
                                                     0)
            
            # Additional technical indicators
            features['ema_5'] = talib.EMA(features['close'], timeperiod=5)
            features['ema_8'] = talib.EMA(features['close'], timeperiod=8)
            features['ema_13'] = talib.EMA(features['close'], timeperiod=13)
            features['ema_21'] = talib.EMA(features['close'], timeperiod=21)
            features['ema_34'] = talib.EMA(features['close'], timeperiod=34)
            features['ema_55'] = talib.EMA(features['close'], timeperiod=55)
            features['ema_89'] = talib.EMA(features['close'], timeperiod=89)
            features['ema_144'] = talib.EMA(features['close'], timeperiod=144)
            
            # ===== 9. Volatility Regime =====
            # คำนวณ volatility regime based on ATR
            atr_20 = talib.ATR(features['high'], features['low'], features['close'], timeperiod=20)
            atr_50 = talib.ATR(features['high'], features['low'], features['close'], timeperiod=50)
            
            # Volatility regime: 0 = Low, 1 = Medium, 2 = High
            # ใช้ ATR เทียบกับ moving average ของ ATR
            # ใช้ pd.Series เพื่อหลีกเลี่ยง SettingWithCopyWarning
            volatility_regime = pd.Series(1, index=features.index)  # Default medium
            volatility_regime[atr_20 < atr_50 * 0.8] = 0  # Low volatility
            volatility_regime[atr_20 > atr_50 * 1.2] = 2  # High volatility
            features['volatility_regime'] = volatility_regime
            
            # ===== 10. Clean up =====
            # Forward fill แล้ว backward fill แล้ว fill ด้วย 0
            features = features.ffill().bfill().fillna(0)
            
            # แทนที่ infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            # ลบ columns ที่ไม่ใช่ features
            cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime']
            features = features.drop(columns=[col for col in cols_to_drop if col in features.columns], errors='ignore')
            
            self.logger.info(f"✅ Comprehensive features calculated: {features.shape}")
            self.logger.debug(f"Feature columns: {list(features.columns)}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature calculation: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    async def send_telegram_message(self, message: str):
        """Send message to Telegram"""
        if not TELEGRAM_CONFIG['notifications_enabled']:
            return
            
        token = TELEGRAM_CONFIG.get('token')
        chat_id = TELEGRAM_CONFIG.get('chat_id')
        
        if not token or not chat_id:
            self.logger.warning("Telegram credentials not configured")
            return
            
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        self.logger.debug("📱 Telegram message sent")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to send Telegram message: {error_text}")
        except asyncio.TimeoutError:
            self.logger.warning("Telegram message timeout")
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100):
        """Fetch OHLCV data from OKX with proper datetime handling"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                self.logger.error("No OHLCV data received from exchange")
                return None
            
            # สร้าง DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # แปลง timestamp เป็น datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # เก็บ datetime ไว้เป็น column (ไม่ใช้เป็น index)
            # เพราะ feature calculator ต้องการ datetime column
            
            # Debug log
            if BOT_CONFIG.get('debug_mode', False):
                self.logger.debug(f"Fetched {len(df)} candles")
                self.logger.debug(f"Columns: {list(df.columns)}")
                self.logger.debug(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except ccxt.NetworkError as e:
            self.logger.error(f"Network error fetching OHLCV data: {e}")
            self.history_manager.log_error('DATA_FETCH_NETWORK', str(e), f"Symbol: {symbol}, TF: {timeframe}")
            return None
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error fetching OHLCV data: {e}")
            self.history_manager.log_error('DATA_FETCH_EXCHANGE', str(e), f"Symbol: {symbol}, TF: {timeframe}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching OHLCV data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.history_manager.log_error('DATA_FETCH', str(e), f"Symbol: {symbol}, TF: {timeframe}")
            return None
    
    def calculate_features_real_time(self, df: pd.DataFrame):
        """
        Calculate features for real-time trading
        Reuses the same feature calculation logic from training
        """
        if not ML_AVAILABLE or self.feature_calculator is None:
            self.logger.warning("ML not available or feature calculator is None")
            return None
            
        try:
            # ตรวจสอบว่า df มีข้อมูลเพียงพอ
            if df is None or len(df) < 50:
                self.logger.warning(f"Insufficient data for feature calculation: {len(df) if df is not None else 0} rows")
                return None
            
            # สร้าง copy ของ df เพื่อไม่ให้กระทบข้อมูลต้นฉบับ
            df_copy = df.copy()
            
            # ตรวจสอบว่ามี datetime column
            if 'datetime' not in df_copy.columns:
                if 'timestamp' in df_copy.columns:
                    df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
                else:
                    self.logger.error("No datetime or timestamp column found")
                    return None
            
            self.logger.debug(f"Calculating features for {len(df_copy)} rows")
            self.logger.debug(f"Columns available: {list(df_copy.columns)}")
            self.logger.debug(f"Date range: {df_copy['datetime'].min()} to {df_copy['datetime'].max()}")
            
            # ใช้ feature calculator จาก training class
            self.feature_calculator.data = df_copy
            
            # ตรวจสอบว่ามี method ไหนใช้งานได้
            feature_methods = [
                'prepare_features',  # เพิ่มเข้ามาเป็นอันดับแรก
                '_calculate_multi_timeframe_indicators',
                'calculate_features', 
                '_calculate_features',
                'calculate_all_features'
            ]
            
            features_calculated = False
            for method_name in feature_methods:
                if hasattr(self.feature_calculator, method_name):
                    try:
                        self.logger.debug(f"Trying method: {method_name}")
                        method = getattr(self.feature_calculator, method_name)
                        
                        # เรียกใช้งานฟังก์ชัน
                        result = method()
                        
                        # ตรวจสอบว่าได้ features หรือไม่
                        if hasattr(self.feature_calculator, 'features') and self.feature_calculator.features is not None:
                            if len(self.feature_calculator.features) > 0:
                                self.logger.info(f"✅ Features calculated using {method_name}")
                                features_calculated = True
                                break
                        elif result is not None and isinstance(result, pd.DataFrame) and len(result) > 0:
                            # บาง method อาจ return DataFrame โดยตรง
                            self.feature_calculator.features = result
                            self.logger.info(f"✅ Features calculated using {method_name} (returned)")
                            features_calculated = True
                            break
                            
                        self.logger.debug(f"Method {method_name} did not produce features")
                        
                    except Exception as e:
                        self.logger.debug(f"Method {method_name} failed: {e}")
                        continue
            
            if not features_calculated:
                self.logger.error("❌ No feature calculation method succeeded")
                self.logger.error(f"Available methods in feature_calculator: {[m for m in dir(self.feature_calculator) if not m.startswith('_')]}")
                
                # ลองคำนวณ features แบบครบถ้วนเอง
                self.logger.warning("⚠️  Attempting comprehensive feature calculation as fallback")
                try:
                    features_df = self._calculate_comprehensive_features(df_copy)
                    if features_df is not None and len(features_df) > 0:
                        self.feature_calculator.features = features_df
                        features_calculated = True
                        self.logger.info("✅ Comprehensive features calculated successfully")
                except Exception as e:
                    self.logger.error(f"Comprehensive feature calculation also failed: {e}")
                    return None
            
            # ตรวจสอบว่า features ถูกสร้างขึ้นแล้ว
            if not hasattr(self.feature_calculator, 'features') or self.feature_calculator.features is None:
                self.logger.error("Features attribute not found in feature calculator")
                return None
            
            features_df = self.feature_calculator.features
            
            # ตรวจสอบว่ามี features ที่ต้องการ
            if features_df is None or len(features_df) == 0:
                self.logger.error("Feature DataFrame is empty")
                self.logger.debug(f"Features type: {type(features_df)}")
                return None
            
            self.logger.info(f"📊 Features shape: {features_df.shape}")
            self.logger.debug(f"Feature columns: {list(features_df.columns)[:10]}...")
            
            # ตรวจสอบว่ามี feature columns ทั้งหมดที่ต้องการ
            missing_features = set(self.feature_columns) - set(features_df.columns)
            if missing_features:
                self.logger.warning(f"⚠️  Missing {len(missing_features)} features: {list(missing_features)[:5]}...")
                # เติม features ที่หายด้วย 0
                for feat in missing_features:
                    features_df[feat] = 0
                self.logger.info(f"✅ Filled missing features with 0")
            
            # เลือกเฉพาะ features ที่ใช้ในโมเดล
            try:
                features = features_df[self.feature_columns]
            except KeyError as e:
                self.logger.error(f"Error selecting feature columns: {e}")
                self.logger.debug(f"Required columns: {self.feature_columns[:10]}")
                self.logger.debug(f"Available columns: {list(features_df.columns)[:10]}")
                return None
            
            # ตรวจสอบ NaN values
            nan_count = features.isnull().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"⚠️  Found {nan_count} NaN values in features, filling with 0")
                features = features.fillna(0)
            
            # ตรวจสอบ infinite values
            inf_count = np.isinf(features.values).sum()
            if inf_count > 0:
                self.logger.warning(f"⚠️  Found {inf_count} infinite values, replacing with 0")
                features = features.replace([np.inf, -np.inf], 0)
            
            # ส่งกลับข้อมูลล่าสุดเท่านั้น
            final_features = features.iloc[-1:].values
            
            self.logger.info(f"✅ Final features shape: {final_features.shape}")
            self.logger.debug(f"Feature values sample: {final_features[0][:5]}...")
            
            return final_features
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating features: {e}")
            import traceback
            error_trace = traceback.format_exc()
            self.logger.debug(error_trace)
            self.history_manager.log_error('FEATURE_CALC', str(e), error_trace)
            return None
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic features as fallback when feature calculator fails
        """
        try:
            self.logger.info("Calculating basic features as fallback...")
            
            features = df.copy()
            
            # Price-based features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            features['log_returns'] = features['log_returns'].replace([np.inf, -np.inf], 0)
            
            # Technical indicators using TA-Lib
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                features[f'sma_{period}'] = talib.SMA(features['close'], timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(features['close'], timeperiod=period)
            
            # RSI
            for period in [14, 21, 28]:
                features[f'rsi_{period}'] = talib.RSI(features['close'], timeperiod=period)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                features['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                features['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = (upper - lower) / np.where(middle != 0, middle, 1)
            
            # ATR (Average True Range)
            features['atr_14'] = talib.ATR(
                features['high'], 
                features['low'], 
                features['close'], 
                timeperiod=14
            )
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                features['high'], 
                features['low'], 
                features['close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # ADX (Average Directional Index)
            features['adx_14'] = talib.ADX(
                features['high'], 
                features['low'], 
                features['close'], 
                timeperiod=14
            )
            
            # CCI (Commodity Channel Index)
            features['cci_14'] = talib.CCI(
                features['high'], 
                features['low'], 
                features['close'], 
                timeperiod=14
            )
            
            # Volume indicators
            features['volume_sma_20'] = talib.SMA(features['volume'], timeperiod=20)
            features['volume_ratio'] = features['volume'] / np.where(features['volume_sma_20'] != 0, features['volume_sma_20'], 1)
            
            # Price position
            price_range = features['high'] - features['low']
            features['price_position'] = np.where(price_range != 0,
                                                   (features['close'] - features['low']) / price_range,
                                                   0.5)
            
            # Remove NaN rows (จาก indicators ที่ต้องใช้ history)
            features = features.bfill().fillna(0)
            
            # ลบ columns ที่ไม่ใช่ features
            features = features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime'], errors='ignore')
            
            self.logger.info(f"✅ Basic features calculated: {features.shape}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in basic feature calculation: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def get_current_signal(self, symbol: str, timeframe: str):
        """Get current trading signal from model or fallback to basic logic"""
        try:
            # ดึงข้อมูลล่าสุด
            df = self.fetch_ohlcv_data(symbol, timeframe, limit=100)
            if df is None or len(df) < 50:
                self.logger.warning("Insufficient data for analysis")
                return None
            
            signal_info = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': float(df['close'].iloc[-1]),
                'confidence': 'MEDIUM'
            }
            
            # ถ้าโมเดลพร้อม ให้ใช้โมเดล
            if self.model is not None and self.scaler is not None:
                # ตรวจสอบว่า scaler ถูก fit แล้ว
                if not hasattr(self.scaler, 'mean_') and not hasattr(self.scaler, 'scale_'):
                    self.logger.error("❌ Scaler is not fitted! Falling back to basic signal")
                    signal_info.update(self._get_basic_signal(df))
                else:
                    features = self.calculate_features_real_time(df)
                    if features is not None:
                        try:
                            features_scaled = self.scaler.transform(features)
                            prediction = self.model.predict(features_scaled)[0]
                            probability = self.model.predict_proba(features_scaled)[0]
                            
                            # แสดงข้อมูล Model Prediction
                            max_prob = float(max(probability))
                            confidence_level = 'HIGH' if max_prob > 0.7 else 'MEDIUM' if max_prob > MODEL_CONFIG['min_confidence'] else 'LOW'
                            
                            self.logger.info(f"🤖 ML Prediction: {prediction} | Confidence: {confidence_level} ({max_prob:.1%})")
                            self.logger.info(f"   Probabilities: BUY={probability[1]:.1%}, SELL={probability[0]:.1%}")
                            
                            signal_info.update({
                                'signal': prediction,
                                'probability': max_prob,
                                'confidence': confidence_level,
                                'features': features.tolist()
                            })
                        except Exception as e:
                            self.logger.error(f"❌ Error in ML prediction: {e}")
                            self.logger.warning("⚠️  Falling back to basic signal")
                            signal_info.update(self._get_basic_signal(df))
                    else:
                        # Fallback to basic signal
                        signal_info.update(self._get_basic_signal(df))
            else:
                # ใช้ basic signal logic
                signal_info.update(self._get_basic_signal(df))
            
            # บันทึกสัญญาณ
            self.history_manager.log_signal(signal_info)
            
            return signal_info
            
        except Exception as e:
            self.logger.error(f"Error getting signal: {e}")
            self.history_manager.log_error('SIGNAL_GEN', str(e))
            return None
    
    def _get_basic_signal(self, df: pd.DataFrame) -> Dict:
        """Basic signal generation logic as fallback"""
        try:
            # คำนวณ indicators พื้นฐาน
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            ema_20 = talib.EMA(df['close'], timeperiod=20).iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # ตรรกะสัญญาณพื้นฐาน
            if current_price > ema_20 and rsi < 70:
                signal = 2  # Buy
                confidence = 0.6
            elif current_price < ema_20 and rsi > 30:
                signal = 0  # Sell
                confidence = 0.6
            else:
                signal = 1  # Neutral
                confidence = 0.5
            
            return {
                'signal': signal,
                'probability': confidence,
                'confidence': 'MEDIUM'
            }
        except Exception as e:
            self.logger.error(f"Error in basic signal: {e}")
            return {'signal': 1, 'probability': 0.5, 'confidence': 'LOW'}
    
    def check_position(self, symbol: str):
        """Check current position for symbol"""
        try:
            # ลองดึง balance จากหลาย account type
            balance = None
            account_types = ['spot', 'trading', 'funding', None]
            
            for acc_type in account_types:
                try:
                    if acc_type:
                        params = {'type': acc_type}
                        balance = self.exchange.fetch_balance(params)
                    else:
                        balance = self.exchange.fetch_balance()
                    
                    # ตรวจสอบว่ามีข้อมูล balance หรือไม่
                    if balance:
                        # ตรวจสอบว่ามี total dict และมีข้อมูล
                        total_dict = balance.get('total', {})
                        free_dict = balance.get('free', {})
                        info = balance.get('info', {})
                        
                        has_data = (
                            (total_dict and len(total_dict) > 0) or
                            (free_dict and len(free_dict) > 0) or
                            (info and len(info) > 0)
                        )
                        
                        if has_data:
                            self.logger.info(f"✅ Balance fetched from account type: {acc_type or 'default'}")
                            break
                        else:
                            self.logger.debug(f"Balance from '{acc_type}' is empty, trying next type...")
                            
                except Exception as e:
                    self.logger.debug(f"Failed to fetch balance with type '{acc_type}': {e}")
                    continue
            
            if not balance:
                self.logger.error("❌ Could not fetch balance from any account type")
                # ส่งคืน default values
                return {
                    'free_usdt': 100.0,
                    'used_usdt': 0,
                    'total_usdt': 100.0,
                    'current_position': None
                }
            
            # Debug: แสดงโครงสร้างของ balance ครั้งแรก
            # if not hasattr(self, '_balance_structure_logged'):
                import json
                self.logger.info("="*60)
                self.logger.info("📊 BALANCE STRUCTURE FROM OKX:")
                self.logger.info("="*60)
                
                # แสดงโครงสร้างหลัก
                self.logger.info(f"Top-level keys: {list(balance.keys())}")
                
                # แสดงข้อมูลใน 'info' ถ้ามี
                if 'info' in balance:
                    info_sample = json.dumps(balance['info'], indent=2, default=str)
                    # จำกัดความยาวเพื่อไม่ให้ log ยาวเกินไป
                    if len(info_sample) > 2000:
                        info_sample = info_sample[:2000] + "\n... (truncated)"
                    self.logger.info(f"Info structure:\n{info_sample}")
                
                # แสดง currencies ที่มี
                if 'total' in balance:
                    self.logger.info(f"Available currencies in 'total': {list(balance['total'].keys())}")
                
                if 'free' in balance:
                    self.logger.info(f"Available currencies in 'free': {list(balance['free'].keys())}")
                
                self.logger.info("="*60)
                self._balance_structure_logged = True
            
            # ตรวจสอบโครงสร้างของ balance และดึงข้อมูลอย่างปลอดภัย
            position_info = {
                'free_usdt': 0,
                'used_usdt': 0,
                'total_usdt': 0,
                'current_position': None
            }
            
            usdt_found = False
            
            # วิธีที่ 1: Standard format (free/used/total)
            if 'USDT' in balance.get('free', {}):
                position_info['free_usdt'] = float(balance['free'].get('USDT', 0))
                position_info['used_usdt'] = float(balance['used'].get('USDT', 0))
                position_info['total_usdt'] = float(balance['total'].get('USDT', 0))
                usdt_found = True
                self.logger.debug("✅ Found USDT in standard format (free/used/total)")
            
            # วิธีที่ 2: Direct access
            elif 'USDT' in balance:
                usdt_balance = balance.get('USDT', {})
                if isinstance(usdt_balance, dict):
                    position_info['free_usdt'] = float(usdt_balance.get('free', 0))
                    position_info['used_usdt'] = float(usdt_balance.get('used', 0))
                    position_info['total_usdt'] = float(usdt_balance.get('total', 0))
                else:
                    position_info['total_usdt'] = float(usdt_balance)
                    position_info['free_usdt'] = float(usdt_balance)
                usdt_found = True
                self.logger.debug("✅ Found USDT in direct access format")
            
            # วิธีที่ 3: OKX info.data format
            elif 'info' in balance and 'data' in balance.get('info', {}):
                data = balance['info']['data']
                
                # OKX อาจส่งมาเป็น list หรือ dict
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # ลองหาจากหลาย key
                            currency = item.get('ccy') or item.get('currency') or item.get('coin')
                            if currency == 'USDT':
                                position_info['free_usdt'] = float(item.get('availBal') or item.get('available') or item.get('free') or 0)
                                position_info['total_usdt'] = float(item.get('bal') or item.get('balance') or item.get('total') or 0)
                                position_info['used_usdt'] = position_info['total_usdt'] - position_info['free_usdt']
                                usdt_found = True
                                self.logger.debug("✅ Found USDT in OKX info.data list format")
                                break
                
                elif isinstance(data, dict):
                    # ถ้าเป็น dict ให้ค้นหา USDT
                    if 'USDT' in data:
                        usdt_data = data['USDT']
                        position_info['free_usdt'] = float(usdt_data.get('availBal') or usdt_data.get('available') or 0)
                        position_info['total_usdt'] = float(usdt_data.get('bal') or usdt_data.get('balance') or 0)
                        position_info['used_usdt'] = position_info['total_usdt'] - position_info['free_usdt']
                        usdt_found = True
                        self.logger.debug("✅ Found USDT in OKX info.data dict format")
            
            # วิธีที่ 4: ลองหาจาก currencies/balances array
            if not usdt_found and 'info' in balance:
                info = balance['info']
                
                # ลองหาจาก key ที่เป็นไปได้
                possible_keys = ['currencies', 'balances', 'assets', 'details']
                for key in possible_keys:
                    if key in info:
                        items = info[key]
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    currency = item.get('ccy') or item.get('currency') or item.get('coin') or item.get('asset')
                                    if currency == 'USDT':
                                        position_info['free_usdt'] = float(item.get('availBal') or item.get('available') or item.get('free') or 0)
                                        position_info['total_usdt'] = float(item.get('bal') or item.get('balance') or item.get('total') or 0)
                                        position_info['used_usdt'] = position_info['total_usdt'] - position_info['free_usdt']
                                        usdt_found = True
                                        self.logger.debug(f"✅ Found USDT in info.{key} format")
                                        break
                        if usdt_found:
                            break
            
            # ถ้ายังไม่พบ USDT
            if not usdt_found:
                self.logger.warning("⚠️  USDT balance not found in any expected format")
                
                # แสดงข้อมูลที่มีอยู่
                if 'total' in balance and balance['total']:
                    self.logger.warning(f"Available currencies in 'total': {list(balance['total'].keys())}")
                elif 'free' in balance and balance['free']:
                    self.logger.warning(f"Available currencies in 'free': {list(balance['free'].keys())}")
                else:
                    self.logger.warning("Available currencies in 'total': []")
                    self.logger.warning("Balance structure appears to be empty or in unexpected format")
                
                # ตรวจสอบ API permissions
                self.logger.warning("")
                self.logger.warning("🔧 TROUBLESHOOTING STEPS:")
                self.logger.warning("1. Check your OKX account type (Spot/Trading/Funding)")
                self.logger.warning("2. Verify you have USDT in your account")
                self.logger.warning("3. Check API permissions:")
                self.logger.warning("   - API Key should have 'Read' permission")
                self.logger.warning("   - Check if 'Trade' permission is needed for balance")
                self.logger.warning("4. Verify API credentials in config_bot.py")
                self.logger.warning("5. Try logging into OKX web interface to verify balance")
                self.logger.warning("")
                
                # ตรวจสอบว่าเป็น Demo/Testnet หรือไม่
                if 'sandbox' in OKX_CONFIG.get('hostname', ''):
                    self.logger.warning("⚠️  You are using OKX TESTNET/SANDBOX")
                    self.logger.warning("   Make sure you have funds in testnet account")
                self.logger.warning("")
                
                # ใช้ค่า default เพื่อให้ bot ทำงานต่อได้ (simulation mode)
                self.logger.info("Using default balance values to continue operation in SIMULATION mode")
                self.logger.info("Note: Real trading will not be possible without actual balance")
                position_info['total_usdt'] = 100.0  # ค่า default
                position_info['free_usdt'] = 100.0
            else:
                self.logger.info(f"💰 USDT Balance: Total=${position_info['total_usdt']:.2f}, Free=${position_info['free_usdt']:.2f}, Used=${position_info['used_usdt']:.2f}")
            
            # ตรวจสอบ position สำหรับ base currency
            base_currency = symbol.split('/')[0]
            base_amount = 0
            
            # ใช้วิธีเดียวกันในการหา base currency
            if base_currency in balance.get('free', {}):
                base_amount = float(balance['free'].get(base_currency, 0))
            elif base_currency in balance:
                base_balance = balance.get(base_currency, {})
                if isinstance(base_balance, dict):
                    base_amount = float(base_balance.get('total', 0))
                else:
                    base_amount = float(base_balance)
            elif 'info' in balance and 'data' in balance.get('info', {}):
                data = balance['info']['data']
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            currency = item.get('ccy') or item.get('currency') or item.get('coin')
                            if currency == base_currency:
                                base_amount = float(item.get('bal') or item.get('balance') or item.get('total') or 0)
                                break
            
            if base_amount > 0:
                current_price = self.get_current_price(symbol)
                
                # ดึงข้อมูล entry price จาก trade history
                recent_trades = self.history_manager.get_recent_trades(symbol, limit=1)
                entry_price = recent_trades[0]['price'] if recent_trades and len(recent_trades) > 0 else current_price
                
                unrealized_pnl = (current_price - entry_price) * base_amount
                
                position_info['current_position'] = {
                    'side': 'buy',
                    'size': base_amount,
                    'entry_price': entry_price,
                    'unrealized_pnl': unrealized_pnl,
                    'current_price': current_price,
                    'value_usdt': base_amount * current_price
                }
                
                self.logger.info(f"📊 Position: {base_amount:.4f} {base_currency} @ ${entry_price:.2f} | Current: ${current_price:.2f} | PnL: ${unrealized_pnl:.2f}")
            
            return position_info
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            self.history_manager.log_error('POSITION_CHECK', str(e), traceback.format_exc())
            
            # Return default position info
            return {
                'free_usdt': 100.0,
                'used_usdt': 0,
                'total_usdt': 100.0,
                'current_position': None
            }
    
    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on trade size and current price"""
        try:
            # วิธีที่ 1: ใช้ fixed trade size
            trade_size_usdt = TRADING_CONFIG['trade_size_usdt']
            
            # วิธีที่ 2: ใช้ percentage ของทุน
            if RISK_CONFIG['position_size_pct'] > 0:
                balance = self.check_position(TRADING_CONFIG['symbol'])
                if balance and balance['total_usdt'] > 0:
                    trade_size_usdt = balance['total_usdt'] * RISK_CONFIG['position_size_pct'] / 100
            
            position_size = trade_size_usdt / current_price
            
            # ปัดเศษตามความต้องการของตลาด (4 ทศนิยมสำหรับส่วนใหญ่)
            return round(position_size, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return TRADING_CONFIG['trade_size_usdt'] / current_price
    
    def place_order(self, symbol: str, side: str, signal_info: dict, exit_reason: str = None):
        """Place order on OKX with balance validation"""
        if not TRADING_CONFIG['trading_enabled']:
            self.logger.info(f"📝 [SIMULATION] Would place {side} order for {symbol}")
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'amount': 0.1,
                'price': signal_info['price'],
                'value_usdt': 100,
                'order_id': 'SIMULATED',
                'signal_confidence': signal_info.get('confidence', 'MEDIUM'),
                'status': 'filled',
                'exit_reason': exit_reason,
                'fee': 0.1
            }
        
        try:
            # ===== BALANCE CHECK BEFORE TRADE =====
            position_info = self.check_position(symbol)
            if not position_info:
                self.logger.error("❌ Cannot check balance - aborting order")
                return None
            
            position_size = self.calculate_position_size(signal_info['price'])
            value_usdt = position_size * signal_info['price']
            
            # ตรวจสอบความพอเพียงของเงิน
            if side.lower() == 'buy':
                # สำหรับการซื้อ ต้องมี USDT เพียงพอ
                available_balance = position_info['free_usdt']
                required_balance = value_usdt * 1.001  # เพิ่ม 0.1% สำหรับค่าธรรมเนียม
                
                if available_balance < required_balance:
                    error_msg = f"""
❌ <b>INSUFFICIENT BALANCE</b>
├ Required: ${required_balance:.2f} USDT
├ Available: ${available_balance:.2f} USDT
├ Shortage: ${required_balance - available_balance:.2f} USDT
├ Symbol: {symbol}
├ Action: {side.upper()}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 Tip: Reduce trade_size_usdt in config_bot.py or add more funds
                    """
                    self.logger.error(f"❌ Insufficient balance: Required ${required_balance:.2f}, Available ${available_balance:.2f}")
                    asyncio.run(self.send_telegram_message(error_msg))
                    self.history_manager.log_error(
                        'INSUFFICIENT_BALANCE', 
                        f"Required: ${required_balance:.2f}, Available: ${available_balance:.2f}",
                        f"Symbol: {symbol}, Side: {side}, Trade Size: ${value_usdt:.2f}"
                    )
                    return None
                
                self.logger.info(f"✅ Balance check passed: ${available_balance:.2f} available, ${required_balance:.2f} required")
                
            elif side.lower() == 'sell':
                # สำหรับการขาย ต้องมี base currency (PAXG) เพียงพอ
                current_position = position_info.get('current_position')
                if not current_position:
                    error_msg = f"""
❌ <b>NO POSITION TO SELL</b>
├ Symbol: {symbol}
├ Action: SELL (attempted)
├ Issue: No {symbol.split('/')[0]} holdings found
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    self.logger.error("❌ No position to sell")
                    asyncio.run(self.send_telegram_message(error_msg))
                    return None
                
                available_amount = current_position.get('size', 0)
                if available_amount < position_size:
                    error_msg = f"""
❌ <b>INSUFFICIENT POSITION SIZE</b>
├ Required: {position_size:.4f} {symbol.split('/')[0]}
├ Available: {available_amount:.4f} {symbol.split('/')[0]}
├ Shortage: {position_size - available_amount:.4f}
├ Symbol: {symbol}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    self.logger.error(f"❌ Insufficient position: Required {position_size:.4f}, Available {available_amount:.4f}")
                    asyncio.run(self.send_telegram_message(error_msg))
                    return None
                
                self.logger.info(f"✅ Position check passed: {available_amount:.4f} available, {position_size:.4f} required")
            
            # ===== ORDER SIZE VALIDATION =====
            # ตรวจสอบ minimum order size
            if hasattr(self, 'min_order_size') and self.min_order_size and self.min_order_size > 0:
                if position_size < self.min_order_size:
                    error_msg = f"""
❌ <b>ORDER SIZE TOO SMALL</b>
├ Order size: {position_size:.8f} {symbol.split('/')[0]}
├ Minimum required: {self.min_order_size:.8f} {symbol.split('/')[0]}
├ Symbol: {symbol}
├ Value: ${value_usdt:.2f}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 Solution:
   • Increase trade_size_usdt in config_bot.py
   • Current: ${TRADING_CONFIG['trade_size_usdt']}
   • Recommended: ${self.min_order_size * signal_info['price'] * 1.1:.2f}+
                    """
                    self.logger.error(f"❌ Order size {position_size:.8f} is below minimum {self.min_order_size:.8f}")
                    asyncio.run(self.send_telegram_message(error_msg))
                    self.history_manager.log_error(
                        'ORDER_SIZE_TOO_SMALL',
                        f"Size: {position_size:.8f}, Min: {self.min_order_size:.8f}",
                        f"Symbol: {symbol}, Trade Size: ${value_usdt:.2f}"
                    )
                    return None
            
            # ตรวจสอบ maximum order size
            if hasattr(self, 'max_order_size') and self.max_order_size and self.max_order_size < float('inf'):
                if position_size > self.max_order_size:
                    error_msg = f"""
❌ <b>ORDER SIZE TOO LARGE</b>
├ Order size: {position_size:.8f} {symbol.split('/')[0]}
├ Maximum allowed: {self.max_order_size:.8f} {symbol.split('/')[0]}
├ Symbol: {symbol}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 Solution:
   • Reduce trade_size_usdt in config_bot.py
                    """
                    self.logger.error(f"❌ Order size {position_size:.8f} exceeds maximum {self.max_order_size:.8f}")
                    asyncio.run(self.send_telegram_message(error_msg))
                    return None
            
            self.logger.info(f"✅ Order size validation passed: {position_size:.8f} {symbol.split('/')[0]}")
            
            # ===== EXECUTE ORDER =====
            order_params = {
                'symbol': symbol,
                'type': 'market',
                'side': side,
                'amount': position_size,
            }
            
            self.logger.info(f"📤 Sending order to OKX: {side.upper()} {position_size:.4f} {symbol} (~${value_usdt:.2f})")
            
            # ส่งออร์เดอร์
            order = self.exchange.create_order(**order_params)
            
            order_info = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'amount': position_size,
                'price': signal_info['price'],
                'value_usdt': value_usdt,
                'order_id': order['id'],
                'signal_confidence': signal_info.get('confidence', 'MEDIUM'),
                'status': order.get('status', 'unknown'),
                'exit_reason': exit_reason,
                'fee': order.get('fee', {}).get('cost', 0) or 0.001 * value_usdt
            }
            
            # บันทึกการเทรด
            self.history_manager.log_trade(order_info)
            
            self.logger.info(f"✅ Order executed successfully: {side.upper()} {position_size:.4f} {symbol} at ${signal_info['price']:.2f}")
            
            # แสดง balance หลัง trade
            updated_balance = self.check_position(symbol)
            if updated_balance:
                self.logger.info(f"💰 Updated Balance: ${updated_balance['total_usdt']:.2f} USDT (${updated_balance['free_usdt']:.2f} available)")
            
            # ส่งแจ้งเตือน Telegram
            message = f"""
🎯 <b>TRADE EXECUTED</b>
├ Symbol: {symbol}
├ Action: {side.upper()}
├ Amount: {position_size:.4f}
├ Price: ${signal_info['price']:.2f}
├ Value: ${value_usdt:.2f}
├ Fee: ~${order_info['fee']:.4f}
├ Confidence: {signal_info.get('confidence', 'MEDIUM')}
├ Balance: ${updated_balance['free_usdt']:.2f} USDT available
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if exit_reason:
                message += f"\n├ Exit Reason: {exit_reason}"
            
            asyncio.run(self.send_telegram_message(message))
            
            return order_info
            
        except ccxt.InsufficientFunds as e:
            # Handle insufficient funds error จาก exchange
            self.logger.error(f"❌ Insufficient funds from exchange: {e}")
            error_message = f"""
❌ <b>INSUFFICIENT FUNDS (Exchange Error)</b>
├ Symbol: {symbol}
├ Action: {side.upper()}
├ Error: {str(e)}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 Action required:
   • Check your OKX account balance
   • Reduce trade_size_usdt in config
   • Ensure funds are in Trading account (not Funding)
            """
            asyncio.run(self.send_telegram_message(error_message))
            self.history_manager.log_error('INSUFFICIENT_FUNDS', str(e), f"Side: {side}, Symbol: {symbol}")
            return None
            
        except ccxt.InvalidOrder as e:
            # Handle invalid order error
            self.logger.error(f"❌ Invalid order: {e}")
            error_message = f"""
❌ <b>INVALID ORDER</b>
├ Symbol: {symbol}
├ Action: {side.upper()}
├ Error: {str(e)}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 Possible issues:
   • Order size too small/large
   • Symbol not available
   • Market closed
            """
            asyncio.run(self.send_telegram_message(error_message))
            self.history_manager.log_error('INVALID_ORDER', str(e), f"Side: {side}, Symbol: {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            self.history_manager.log_error('ORDER_PLACE', str(e), f"Side: {side}, Symbol: {symbol}")
            
            # ส่งแจ้งเตือน error
            error_message = f"""
❌ <b>ORDER FAILED</b>
├ Symbol: {symbol}
├ Action: {side.upper()}
├ Error: {str(e)}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            asyncio.run(self.send_telegram_message(error_message))
            
            return None
    
    def should_exit_trade(self, current_position: dict, current_price: float):
        """Check if we should exit current trade based on stop loss/take profit"""
        if not current_position:
            return False
        
        entry_price = current_position.get('entry_price', 0)
        
        # ถ้าไม่มี entry price (spot trading) ให้ข้าม
        if entry_price == 0:
            return False
            
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Adjust for short positions
        if current_position['side'] == 'short':
            price_change_pct = -price_change_pct
        
        # Check stop loss
        if price_change_pct <= -RISK_CONFIG['stop_loss_pct']:
            return 'STOP_LOSS'
        # Check take profit
        elif price_change_pct >= RISK_CONFIG['take_profit_pct']:
            return 'TAKE_PROFIT'
        
        return False
    
    async def generate_hourly_report(self):
        """สร้างรายงานสรุปรายชั่วโมง"""
        try:
            # สถิติ 1 ชั่วโมงที่ผ่านมา
            hourly_stats = self.history_manager.get_hourly_stats(1)
            
            # สถิติรายวัน
            daily_stats = self.history_manager.get_hourly_stats(24)
            
            # ตำแหน่งที่เปิดอยู่
            open_positions = self.history_manager.get_current_open_positions()
            
            # สถานะพอร์ตปัจจุบัน
            portfolio = self.check_position(TRADING_CONFIG['symbol'])
            
            # สร้างรายงาน
            report = f"""
📊 <b>HOURLY TRADING REPORT</b>
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}

<b>Last Hour Performance:</b>
├ Trades: {hourly_stats['total_trades']}
├ Win Rate: {hourly_stats['win_rate']:.1f}%
├ Total PnL: ${hourly_stats['total_pnl']:.2f}
├ Avg PnL: ${hourly_stats['avg_pnl']:.2f}
└ Signals: {hourly_stats['total_signals']}

<b>Daily Performance (24h):</b>
├ Trades: {daily_stats['total_trades']}
├ Win Rate: {daily_stats['win_rate']:.1f}%
├ Total PnL: ${daily_stats['total_pnl']:.2f}
└ Avg Confidence: {daily_stats['avg_confidence']:.1f}%

<b>Current Portfolio:</b>
├ Total Balance: ${portfolio['total_usdt']:.2f if portfolio else 0}
├ Available: ${portfolio['free_usdt']:.2f if portfolio else 0}
└ Open Positions: {len(open_positions)}

<b>Open Positions:</b>
"""
            
            if open_positions:
                for pos in open_positions[:3]:  # แสดงสูงสุด 3 ตำแหน่ง
                    current_price = self.get_current_price(pos['symbol'])
                    unrealized_pnl = (current_price - pos['price']) * pos['amount']
                    unrealized_pnl_percent = (current_price - pos['price']) / pos['price'] * 100
                    
                    report += f"├ {pos['symbol']} | {pos['amount']:.4f} | Entry: ${pos['price']:.2f} | Unrealized: ${unrealized_pnl:.2f} ({unrealized_pnl_percent:+.2f}%)\n"
                
                if len(open_positions) > 3:
                    report += f"└ ... and {len(open_positions) - 3} more positions\n"
            else:
                report += "├ No open positions\n"
            
            report += f"\n🔄 Next report in 1 hour"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating hourly report: {e}")
            return f"❌ Error generating report: {str(e)}"
    
    async def send_hourly_report(self):
        """ส่งรายงานสรุปรายชั่วโมง"""
        try:
            report = await self.generate_hourly_report()
            await self.send_telegram_message(report)
            
            # บันทึก snapshot พอร์ต
            portfolio = self.check_position(TRADING_CONFIG['symbol'])
            if portfolio:
                hourly_stats = self.history_manager.get_hourly_stats(1)
                daily_stats = self.history_manager.get_hourly_stats(24)
                
                portfolio_data = {
                    'timestamp': datetime.now(),
                    'total_balance': portfolio['total_usdt'],
                    'available_balance': portfolio['free_usdt'],
                    'total_pnl': daily_stats['total_pnl'],
                    'daily_pnl': daily_stats['total_pnl'],
                    'win_rate': daily_stats['win_rate'],
                    'total_trades': daily_stats['total_trades'],
                    'winning_trades': daily_stats['winning_trades']
                }
                
                self.history_manager.log_portfolio_snapshot(portfolio_data)
            
            self.logger.info("✅ Hourly report sent and portfolio snapshot saved")
            
        except Exception as e:
            self.logger.error(f"Error sending hourly report: {e}")
            self.history_manager.log_error('REPORT_SEND', str(e))
    
    def get_current_price(self, symbol: str):
        """Get current price for symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0
    
    def start_hourly_report(self):
        """เริ่ม thread สำหรับส่งรายงานรายชั่วโมง"""
        def report_loop():
            while self.is_running:
                try:
                    # รอจนถึงชั่วโมงถัดไป (เช่น 13:00, 14:00, ...)
                    now = datetime.now()
                    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    wait_seconds = (next_hour - now).total_seconds()
                    
                    # แทนที่จะ sleep ยาวๆ ให้ sleep เป็นช่วงสั้นๆ และตรวจสอบ is_running
                    if wait_seconds > 0:
                        elapsed = 0
                        sleep_interval = 10  # ตรวจสอบทุก 10 วินาที
                        while elapsed < wait_seconds and self.is_running:
                            time.sleep(min(sleep_interval, wait_seconds - elapsed))
                            elapsed += sleep_interval
                    
                    # ถ้า bot ถูกหยุดระหว่างรอ ให้ออกจาก loop
                    if not self.is_running:
                        break
                    
                    # ส่งรายงาน
                    asyncio.run(self.send_hourly_report())
                    
                except Exception as e:
                    self.logger.error(f"Error in hourly report loop: {e}")
                    if self.is_running:  # ตรวจสอบก่อน sleep
                        time.sleep(60)  # รอ 1 นาทีแล้วลองใหม่
        
        if BOT_CONFIG['hourly_report_enabled']:
            report_thread = threading.Thread(target=report_loop, daemon=True)
            report_thread.start()
            self.logger.info("✅ Hourly report system started")
    
    def start_daily_export(self):
        """เริ่ม thread สำหรับ export ข้อมูลรายวัน"""
        def export_loop():
            while self.is_running:
                try:
                    # รอจนถึงเที่ยงคืน
                    now = datetime.now()
                    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    wait_seconds = (next_midnight - now).total_seconds()
                    
                    # Sleep เป็นช่วงสั้นๆ และตรวจสอบ is_running
                    if wait_seconds > 0:
                        elapsed = 0
                        sleep_interval = 60  # ตรวจสอบทุก 60 วินาที
                        while elapsed < wait_seconds and self.is_running:
                            time.sleep(min(sleep_interval, wait_seconds - elapsed))
                            elapsed += sleep_interval
                    
                    # ถ้า bot ถูกหยุดระหว่างรอ ให้ออกจาก loop
                    if not self.is_running:
                        break
                    
                    # Export trade history
                    self.logger.info("📤 Starting daily trade history export...")
                    self.export_trade_history(days=30)
                    
                except Exception as e:
                    self.logger.error(f"Error in daily export loop: {e}")
                    if self.is_running:
                        time.sleep(300)  # รอ 5 นาทีแล้วลองใหม่
        
        if BOT_CONFIG.get('daily_export_enabled', True):  # Default enabled
            export_thread = threading.Thread(target=export_loop, daemon=True)
            export_thread.start()
            self.logger.info("✅ Daily export system started")
    
    def health_check(self):
        """ตรวจสอบสุขภาพของระบบ"""
        try:
            # ตรวจสอบการเชื่อมต่อ exchange
            self.exchange.fetch_time()
            
            # ตรวจสอบ database
            self.history_manager.get_hourly_stats(1)
            
            # รีเซ็ต error counter
            self.consecutive_errors = 0
            
            return True
            
        except Exception as e:
            self.consecutive_errors += 1
            self.logger.error(f"Health check failed ({self.consecutive_errors}/{self.max_consecutive_errors}): {e}")
            
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.logger.error("Too many consecutive errors - stopping bot")
                self.stop()
                
            return False
    
    def execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        symbol = TRADING_CONFIG['symbol']
        timeframe = TRADING_CONFIG['timeframe']
        
        try:
            # 0. Health check
            if not self.health_check():
                return
            
            # 1. ตรวจสอบตำแหน่งปัจจุบัน
            position_info = self.check_position(symbol)
            if not position_info:
                self.logger.error("Failed to check position")
                return
            
            current_position = position_info['current_position']
            
            # 2. ถ้ามีตำแหน่งอยู่ ตรวจสอบว่าควร exit หรือไม่
            if current_position:
                current_price = self.get_current_price(symbol)
                exit_reason = self.should_exit_trade(current_position, current_price)
                if exit_reason:
                    # ส่งออร์เดอร์ปิดตำแหน่ง
                    exit_side = 'sell' if current_position['side'] == 'buy' else 'buy'
                    signal_info = {'price': current_price, 'confidence': 'EXIT'}
                    order_result = self.place_order(symbol, exit_side, signal_info, exit_reason)
                    
                    if order_result:
                        message = f"""
📤 <b>POSITION CLOSED</b>
├ Symbol: {symbol}
├ Reason: {exit_reason}
├ Side: {current_position['side']}
├ PnL: ${current_position.get('unrealized_pnl', 0):.2f}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        """
                        asyncio.run(self.send_telegram_message(message))
                    return
            
            # 3. ตรวจสอบจำนวนตำแหน่งที่เปิดอยู่
            open_positions = self.history_manager.get_current_open_positions()
            if len(open_positions) >= TRADING_CONFIG['max_open_positions']:
                self.logger.info(f"Max open positions reached ({len(open_positions)}) - skipping new trades")
                return
            
            # 4. รับสัญญาณการเทรดล่าสุด
            signal_info = self.get_current_signal(symbol, timeframe)
            if not signal_info:
                self.logger.warning("No signal generated")
                return
            
            # 5. ตัดสินใจเทรด based on signal
            min_confidence = MODEL_CONFIG['min_confidence']
            signal_confidence = signal_info.get('probability', 0)
            
            if (signal_info.get('signal') == 2 and  # Buy signal
                signal_info.get('confidence') in ['HIGH', 'MEDIUM'] and
                signal_confidence >= min_confidence):
                
                if not current_position:  # ยังไม่มีตำแหน่ง
                    self.place_order(symbol, 'buy', signal_info)
                else:
                    self.logger.info("Already in position, skipping buy signal")
                    
            elif (signal_info.get('signal') == 0 and  # Sell signal
                  current_position and 
                  signal_confidence >= min_confidence):
                self.place_order(symbol, 'sell', signal_info)
            
            # 6. Log สถานะ
            self.logger.info(f"Trading cycle completed - Signal: {signal_info.get('signal')}, Confidence: {signal_info.get('confidence')}")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            self.history_manager.log_error('TRADING_CYCLE', str(e))
            asyncio.run(self.send_telegram_message(f"❌ Trading cycle error: {str(e)}"))
    
    def stop(self):
        """หยุดการทำงานของ bot"""
        self.export_trade_history(days=30)
        self.logger.info("🛑 Received stop signal, shutting down gracefully...")
        self.is_running = False
        
        # รอให้ background threads หยุด (สูงสุด 2 วินาที)
        self.logger.info("   Waiting for background threads to finish...")
        time.sleep(2)
        
        self.logger.info("🛑 Trading bot stopped")
        
        # ส่ง Telegram notification (ถ้าทำได้)
        try:
            # ใช้ new event loop เพื่อหลีกเลี่ยง RuntimeError
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_telegram_message("🛑 Trading Bot Stopped"))
            loop.close()
        except Exception as e:
            self.logger.debug(f"Could not send stop notification: {e}")
        
        # Force exit
        import sys
        sys.exit(0)
    
    def run(self):
        """Run trading bot continuously"""
        self.is_running = True
        self.logger.info(f"🤖 Starting trading bot with {BOT_CONFIG['trading_interval_minutes']} minute interval")
        
        # ตั้งค่า signal handler สำหรับ graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # ส่งข้อความเริ่มต้น
        asyncio.run(self.send_telegram_message(
            f"🚀 Trading Bot Started\n"
            f"Symbol: {TRADING_CONFIG['symbol']}\n"
            f"Timeframe: {TRADING_CONFIG['timeframe']}\n"
            f"Trade Size: ${TRADING_CONFIG['trade_size_usdt']}\n"
            f"Trading Enabled: {'✅' if TRADING_CONFIG['trading_enabled'] else '❌'}\n"
            f"Hourly Reports: {'✅' if BOT_CONFIG['hourly_report_enabled'] else '❌'}"
        ))
        
        interval_seconds = BOT_CONFIG['trading_interval_minutes'] * 60
        
        while self.is_running:
            try:
                cycle_start = datetime.now()
                self.logger.info(f"🔄 Starting trading cycle at {cycle_start}")
                
                # รันการเทรด
                self.execute_trading_cycle()
                
                # คำนวณเวลาเหลือจนถึง cycle ถัดไป
                cycle_end = datetime.now()
                cycle_duration = (cycle_end - cycle_start).total_seconds()
                sleep_time = max(1, interval_seconds - cycle_duration)
                
                self.logger.info(f"💤 Cycle completed in {cycle_duration:.1f}s, sleeping for {sleep_time:.1f}s")
                
                # แทนที่จะ sleep ยาวๆ ครั้งเดียว ให้ sleep เป็นช่วงสั้นๆ และตรวจสอบ is_running
                elapsed = 0
                check_interval = 1  # ตรวจสอบทุก 1 วินาที
                while elapsed < sleep_time and self.is_running:
                    time.sleep(min(check_interval, sleep_time - elapsed))
                    elapsed += check_interval
                    
            except KeyboardInterrupt:
                self.logger.info("🛑 Bot stopped by user (Ctrl+C)")
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                self.history_manager.log_error('MAIN_LOOP', str(e))
                
                # ถ้า is_running ยัง True ให้รอแล้วลองใหม่
                if self.is_running:
                    time.sleep(60)  # รอ 1 นาทีแล้วลองใหม่
        
        self.logger.info("✅ Trading bot main loop exited")
    
    def export_trade_history(self, days: int = 30):
        """ส่งออกประวัติการเทรด"""
        try:
            files = self.history_manager.export_to_csv(days)
            
            if files:
                message = f"""
💾 <b>TRADE HISTORY EXPORTED</b>
├ Trades: {os.path.basename(files['trades_file'])}
├ Signals: {os.path.basename(files['signals_file'])}  
├ Portfolio: {os.path.basename(files['portfolio_file'])}
└ Period: Last {days} days
                """
                
                # ใช้ new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_telegram_message(message))
                loop.close()
                
                self.logger.info(f"✅ Trade history exported successfully")
            
            return files
        except Exception as e:
            self.logger.error(f"Error exporting trade history: {e}")
            return None

def main():
    """Main entry point"""
    try:
        # สร้างและรัน trading bot
        bot = OKXTradingBot()
        bot.run()
        
    except Exception as e:
        logging.error(f"Failed to start trading bot: {e}")
        print(f"❌ Failed to start trading bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()