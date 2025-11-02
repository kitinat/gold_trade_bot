# trade_bot.py
import ccxt
import pandas as pd
import numpy as np
import joblib
import talib
import time
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional
import sys
import os
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
    from train_model import AdvancedTradingModelTrainer
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
        self.setup_trade_history()  # ต้องสร้างก่อน setup_models เพราะ setup_models ใช้ history_manager
        self.setup_exchange()
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
        
        self.logger.info("✅ Trading bot initialized successfully")
    
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
            self.exchange.fetch_balance()
            self.logger.info("✅ OKX connection established successfully")
            
        except ccxt.AuthenticationError as e:
            self.logger.error(f"❌ OKX authentication failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to OKX: {e}")
            raise
    
    def setup_models(self):
        """Load trained models and scaler"""
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        if not ML_AVAILABLE:
            self.logger.warning("⚠️  ML models not available - running in basic mode")
            return
            
        try:
            # ตรวจสอบว่าไฟล์โมเดลมีอยู่
            if not all(os.path.exists(MODEL_CONFIG[path]) for path in ['model_path', 'scaler_path', 'features_path']):
                self.logger.warning("⚠️  Model files not found. Running without ML predictions.")
                return
            
            # โหลดโมเดลที่ฝึกไว้
            self.model = joblib.load(MODEL_CONFIG['model_path'])
            self.scaler = joblib.load(MODEL_CONFIG['scaler_path'])
            self.feature_columns = joblib.load(MODEL_CONFIG['features_path'])
            
            # สร้าง feature calculator instance
            self.feature_calculator = AdvancedTradingModelTrainer()
            
            self.logger.info("✅ ML models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            self.history_manager.log_error('MODEL_LOAD', str(e))
    
    def setup_trade_history(self):
        """Setup trade history manager"""
        self.history_manager = TradeHistoryManager()
        self.logger.info("✅ Trade history manager initialized")
    
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
        """Fetch OHLCV data from OKX"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")
            self.history_manager.log_error('DATA_FETCH', str(e), f"Symbol: {symbol}, TF: {timeframe}")
            return None
    
    def calculate_features_real_time(self, df: pd.DataFrame):
        """
        Calculate features for real-time trading
        Reuses the same feature calculation logic from training
        """
        if not ML_AVAILABLE or self.feature_calculator is None:
            return None
            
        try:
            # ใช้ feature calculator จาก training class
            self.feature_calculator.data = df.copy()
            self.feature_calculator._calculate_multi_timeframe_indicators()
            
            # เลือกเฉพาะ features ที่ใช้ในโมเดล
            if hasattr(self.feature_calculator, 'features'):
                features = self.feature_calculator.features[self.feature_columns]
                return features.iloc[-1:].values  # ข้อมูลล่าสุดเท่านั้น
            else:
                self.logger.error("Features not calculated properly")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            self.history_manager.log_error('FEATURE_CALC', str(e))
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
                features = self.calculate_features_real_time(df)
                if features is not None:
                    features_scaled = self.scaler.transform(features)
                    prediction = self.model.predict(features_scaled)[0]
                    probability = self.model.predict_proba(features_scaled)[0]
                    
                    signal_info.update({
                        'signal': prediction,
                        'probability': float(max(probability)),
                        'confidence': 'HIGH' if max(probability) > 0.7 else 'MEDIUM' if max(probability) > MODEL_CONFIG['min_confidence'] else 'LOW',
                        'features': features.tolist()
                    })
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
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions([symbol.replace('/', '')])
            
            position_info = {
                'free_usdt': float(balance['USDT']['free']),
                'used_usdt': float(balance['USDT']['used']),
                'total_usdt': float(balance['USDT']['total']),
                'current_position': None
            }
            
            for pos in positions:
                if pos['symbol'] == symbol.replace('/', '') and float(pos['contracts']) > 0:
                    position_info['current_position'] = {
                        'side': pos['side'],
                        'size': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unrealizedPnl'])
                    }
                    break
            
            return position_info
            
        except Exception as e:
            self.logger.error(f"Error checking position: {e}")
            self.history_manager.log_error('POSITION_CHECK', str(e))
            return None
    
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
        """Place order on OKX"""
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
            position_size = self.calculate_position_size(signal_info['price'])
            value_usdt = position_size * signal_info['price']
            
            order_params = {
                'symbol': symbol,
                'type': 'market',
                'side': side,
                'amount': position_size,
                'params': {
                    'tdMode': 'cash'  # spot trading
                }
            }
            
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
            
            self.logger.info(f"✅ Order placed: {side} {position_size} {symbol} at {signal_info['price']}")
            
            # ส่งแจ้งเตือน Telegram
            message = f"""
🎯 <b>TRADE EXECUTED</b>
├ Symbol: {symbol}
├ Action: {side.upper()}
├ Amount: {position_size:.4f}
├ Price: ${signal_info['price']:.2f}
├ Value: ${value_usdt:.2f}
├ Confidence: {signal_info.get('confidence', 'MEDIUM')}
└ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if exit_reason:
                message += f"\n├ Exit Reason: {exit_reason}"
            
            asyncio.run(self.send_telegram_message(message))
            
            return order_info
            
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
            
        entry_price = current_position['entry_price']
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
                    
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    
                    # ส่งรายงาน
                    asyncio.run(self.send_hourly_report())
                    
                except Exception as e:
                    self.logger.error(f"Error in hourly report loop: {e}")
                    time.sleep(60)  # รอ 1 นาทีแล้วลองใหม่
        
        if BOT_CONFIG['hourly_report_enabled']:
            report_thread = threading.Thread(target=report_loop, daemon=True)
            report_thread.start()
            self.logger.info("✅ Hourly report system started")
    
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
├ PnL: ${current_position['unrealized_pnl']:.2f}
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
        self.is_running = False
        self.logger.info("🛑 Trading bot stopping...")
        asyncio.run(self.send_telegram_message("🛑 Trading Bot Stopped"))
    
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
                time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user (KeyboardInterrupt)")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                self.history_manager.log_error('MAIN_LOOP', str(e))
                time.sleep(60)  # รอ 1 นาทีแล้วลองใหม่
        
        self.logger.info("Trading bot stopped")
    
    def export_trade_history(self, days: int = 30):
        """ส่งออกประวัติการเทรด"""
        files = self.history_manager.export_to_csv(days)
        
        if files:
            message = f"""
💾 <b>TRADE HISTORY EXPORTED</b>
├ Trades: {os.path.basename(files['trades_file'])}
├ Signals: {os.path.basename(files['signals_file'])}  
├ Portfolio: {os.path.basename(files['portfolio_file'])}
└ Period: Last {days} days
            """
            
            asyncio.run(self.send_telegram_message(message))
        
        return files

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