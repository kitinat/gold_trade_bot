# config_bot.py
# Configuration for OKX Trading Bot

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== EXCHANGE CONFIG ====================
OKX_CONFIG = {
    'api_key': os.getenv('OKX_API_KEY', ''),
    'secret': os.getenv('OKX_SECRET_KEY', ''),
    'password': os.getenv('OKX_PASSPHRASE', ''),
    'sandbox': False,  # ใช้ True สำหรับ testnet, False สำหรับ real trading
}

# ==================== TRADING CONFIG ====================
TRADING_CONFIG = {
    'symbol': os.getenv('TRADING_SYMBOL', 'PAXG/USDT'),
    'timeframe': os.getenv('TIMEFRAME', '15m'),
    'trade_size_usdt': float(os.getenv('TRADE_SIZE_USDT', 100)),
    'max_open_positions': 1,
    'trading_enabled': True,  # ตั้งเป็น False เพื่อทดสอบโดยไม่ส่งออร์เดอร์จริง
}

# ==================== RISK MANAGEMENT ====================
RISK_CONFIG = {
    'stop_loss_pct': 0.02,  # 2% stop loss
    'take_profit_pct': 0.04,  # 4% take profit
    'max_daily_loss_pct': 5.0,  # หยุดเทรดถ้าขาดทุนเกิน 5% ต่อวัน
    'position_size_pct': 1.0,  # ขนาดออร์เดอร์ 1% ของทุน
}

# ==================== TELEGRAM CONFIG ====================
TELEGRAM_CONFIG = {
    'token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'notifications_enabled': True,
}

# ==================== MODEL CONFIG ====================
MODEL_CONFIG = {
    'model_path': 'best_trading_model.pkl',
    'scaler_path': 'feature_scaler.pkl',
    'features_path': 'feature_columns.pkl',
    'min_confidence': 0.6,  # ความมั่นใจขั้นต่ำที่ยอมรับ
}

# ==================== DATABASE CONFIG ====================
DATABASE_CONFIG = {
    'db_path': 'trading_history.db',
    'backup_interval_hours': 24,
}

# ==================== BOT BEHAVIOR ====================
BOT_CONFIG = {
    'trading_interval_minutes': 15,
    'health_check_interval_minutes': 5,
    'hourly_report_enabled': True,
    'debug_mode': False,
}

# ==================== VALIDATION ====================
def validate_config():
    """Validate configuration and return errors if any"""
    errors = []
    
    # Check required API keys
    if not OKX_CONFIG['api_key']:
        errors.append("OKX_API_KEY is required")
    if not OKX_CONFIG['secret']:
        errors.append("OKX_SECRET_KEY is required")
    if not OKX_CONFIG['password']:
        errors.append("OKX_PASSPHRASE is required")
    
    # Check trading parameters
    if TRADING_CONFIG['trade_size_usdt'] <= 0:
        errors.append("TRADE_SIZE_USDT must be positive")
    
    # Check risk parameters
    if RISK_CONFIG['stop_loss_pct'] <= 0:
        errors.append("Stop loss percentage must be positive")
    if RISK_CONFIG['take_profit_pct'] <= RISK_CONFIG['stop_loss_pct']:
        errors.append("Take profit must be greater than stop loss")
    
    return errors

# ==================== EXPORT CONFIG ====================
def get_full_config():
    """Get complete configuration dictionary"""
    return {
        'okx': OKX_CONFIG,
        'trading': TRADING_CONFIG,
        'risk': RISK_CONFIG,
        'telegram': TELEGRAM_CONFIG,
        'model': MODEL_CONFIG,
        'database': DATABASE_CONFIG,
        'bot': BOT_CONFIG,
    }