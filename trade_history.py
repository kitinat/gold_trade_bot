# trade_history.py
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
from config_bot import DATABASE_CONFIG

class TradeHistoryManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_CONFIG['db_path']
        self.setup_logging()      # ✅ แก้: ย้ายมาก่อน setup_database()
        self.setup_database()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Setup SQLite database for trade history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ตารางบันทึกการเทรด
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    side TEXT,
                    amount REAL,
                    price REAL,
                    value_usdt REAL,
                    order_id TEXT,
                    signal_confidence TEXT,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    fee REAL DEFAULT 0,
                    status TEXT,
                    exit_reason TEXT,
                    entry_order_id TEXT,
                    exit_timestamp DATETIME,
                    holding_period_minutes INTEGER
                )
            ''')
            
            # ตารางบันทึกสัญญาณ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    signal INTEGER,
                    probability REAL,
                    confidence TEXT,
                    price REAL,
                    features_json TEXT,
                    executed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # ตารางบันทึกพอร์ตโฟลิโอ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    total_balance REAL,
                    available_balance REAL,
                    total_pnl REAL,
                    daily_pnl REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER
                )
            ''')
            
            # ตารางบันทึก errors
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    error_type TEXT,
                    error_message TEXT,
                    context TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("✅ Trade history database setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Database setup failed: {e}")
            raise
    
    def log_trade(self, trade_data: Dict):
        """บันทึกการเทรด"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # เตรียมข้อมูล
            trade_data.setdefault('pnl', 0)
            trade_data.setdefault('pnl_percent', 0)
            trade_data.setdefault('fee', 0)
            trade_data.setdefault('exit_reason', '')
            trade_data.setdefault('entry_order_id', '')
            trade_data.setdefault('exit_timestamp', None)
            trade_data.setdefault('holding_period_minutes', 0)
            
            columns = ['timestamp', 'symbol', 'side', 'amount', 'price', 'value_usdt', 
                      'order_id', 'signal_confidence', 'pnl', 'pnl_percent', 'fee', 
                      'status', 'exit_reason', 'entry_order_id', 'exit_timestamp', 
                      'holding_period_minutes']
            
            placeholders = ', '.join(['?'] * len(columns))
            columns_str = ', '.join(columns)
            
            values = [trade_data.get(col, None) for col in columns]
            
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO trades ({columns_str})
                VALUES ({placeholders})
            ''', values)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"📝 Trade logged: {trade_data['side']} {trade_data['symbol']} at {trade_data['price']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log trade: {e}")
    
    def log_signal(self, signal_data: Dict):
        """บันทึกสัญญาณการเทรด"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (timestamp, symbol, signal, probability, confidence, price, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['timestamp'],
                signal_data['symbol'],
                signal_data.get('signal', 0),
                signal_data.get('probability', 0),
                signal_data.get('confidence', 'LOW'),
                signal_data.get('price', 0),
                json.dumps(signal_data.get('features', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log signal: {e}")
    
    def log_portfolio_snapshot(self, portfolio_data: Dict):
        """บันทึกสถานะพอร์ตโฟลิโอ"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio (timestamp, total_balance, available_balance, total_pnl, daily_pnl, win_rate, total_trades, winning_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                portfolio_data.get('timestamp', datetime.now()),
                portfolio_data.get('total_balance', 0),
                portfolio_data.get('available_balance', 0),
                portfolio_data.get('total_pnl', 0),
                portfolio_data.get('daily_pnl', 0),
                portfolio_data.get('win_rate', 0),
                portfolio_data.get('total_trades', 0),
                portfolio_data.get('winning_trades', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log portfolio snapshot: {e}")
    
    def log_error(self, error_type: str, error_message: str, context: str = ""):
        """บันทึก error"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO errors (timestamp, error_type, error_message, context)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now(), error_type, error_message, context))
            
            conn.commit()
            conn.close()
            
            self.logger.error(f"📝 Error logged: {error_type} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log error: {e}")
    
    def get_entry_trade_for_exit(self, symbol: str) -> Optional[Dict]:
        """หาออร์เดอร์ซื้อล่าสุดสำหรับสัญลักษณ์นี้"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM trades 
                WHERE symbol = ? AND side = 'buy' AND status = 'filled'
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                # ✅ แก้: เก็บ description ก่อน close connection
                columns = [description[0] for description in cursor.description]
                result = dict(zip(columns, row))
                conn.close()
                return result
            
            conn.close()  # ✅ แก้: ปิด connection แม้ไม่มีข้อมูล
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get entry trade: {e}")
            return None
    
    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict]:
        """ดึงข้อมูลการเทรดล่าสุดสำหรับ symbol นี้"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM trades 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, limit))
            
            rows = cursor.fetchall()
            
            if rows:
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in rows]
                conn.close()
                return results
            
            conn.close()
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get recent trades: {e}")
            return []
    
    def get_hourly_stats(self, hours: int = 1) -> Dict:
        """คำนวณสถิติในช่วงเวลาที่กำหนด"""
        try:
            conn = sqlite3.connect(self.db_path)
            start_time = datetime.now() - timedelta(hours=hours)
            
            # สถิติการเทรด
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buy_trades,
                    SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sell_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    AVG(pnl_percent) as avg_pnl_percent
                FROM trades 
                WHERE timestamp >= ? AND side = 'sell'
            ''', (start_time,))
            
            trade_stats = cursor.fetchone()
            
            # สถิติสัญญาณ
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_signals,
                    AVG(probability) as avg_confidence,
                    SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_signals
                FROM signals 
                WHERE timestamp >= ?
            ''', (start_time,))
            
            signal_stats = cursor.fetchone()
            
            conn.close()
            
            # ✅ แก้: ปรับการคำนวณ win_rate ให้ชัดเจนขึ้น
            if trade_stats and trade_stats[0] and trade_stats[0] > 0:
                win_rate = (trade_stats[3] / trade_stats[2]) * 100 if trade_stats[2] > 0 else 0
            else:
                win_rate = 0
            
            return {
                'period_hours': hours,
                'start_time': start_time,
                'end_time': datetime.now(),
                'total_trades': trade_stats[0] if trade_stats and trade_stats[0] else 0,
                'buy_trades': trade_stats[1] if trade_stats and trade_stats[1] else 0,
                'sell_trades': trade_stats[2] if trade_stats and trade_stats[2] else 0,
                'winning_trades': trade_stats[3] if trade_stats and trade_stats[3] else 0,
                'losing_trades': trade_stats[4] if trade_stats and trade_stats[4] else 0,
                'win_rate': win_rate,
                'avg_pnl': trade_stats[5] if trade_stats and trade_stats[5] else 0,
                'total_pnl': trade_stats[6] if trade_stats and trade_stats[6] else 0,
                'avg_pnl_percent': trade_stats[7] if trade_stats and trade_stats[7] else 0,
                'total_signals': signal_stats[0] if signal_stats and signal_stats[0] else 0,
                'avg_confidence': signal_stats[1] if signal_stats and signal_stats[1] else 0,
                'executed_signals': signal_stats[2] if signal_stats and signal_stats[2] else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get hourly stats: {e}")
            return {
                'period_hours': hours,
                'start_time': datetime.now() - timedelta(hours=hours),
                'end_time': datetime.now(),
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'avg_pnl_percent': 0,
                'total_signals': 0,
                'avg_confidence': 0,
                'executed_signals': 0
            }
    
    def get_daily_stats(self) -> Dict:
        """คำนวณสถิติรายวัน"""
        return self.get_hourly_stats(24)
    
    def get_current_open_positions(self) -> List[Dict]:
        """หาตำแหน่งที่ยังเปิดอยู่"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT t1.* 
                FROM trades t1
                LEFT JOIN trades t2 ON t1.order_id = t2.entry_order_id
                WHERE t1.side = 'buy' AND t2.id IS NULL AND t1.status = 'filled'
            ''')
            
            # ✅ แก้: เก็บ description ก่อน iterate
            columns = [description[0] for description in cursor.description]
            
            open_positions = []
            for row in cursor.fetchall():
                open_positions.append(dict(zip(columns, row)))
            
            conn.close()
            return open_positions
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get open positions: {e}")
            return []
    
    def export_to_csv(self, days: int = 30):
        """ส่งออกข้อมูลเป็น CSV"""
        try:
            conn = sqlite3.connect(self.db_path)
            start_time = datetime.now() - timedelta(days=days)
            
            # ส่งออกการเทรด
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades WHERE timestamp >= ?
            ''', conn, params=(start_time,))
            
            # ส่งออกสัญญาณ
            signals_df = pd.read_sql_query('''
                SELECT * FROM signals WHERE timestamp >= ?
            ''', conn, params=(start_time,))
            
            # ส่งออกพอร์ตโฟลิโอ
            portfolio_df = pd.read_sql_query('''
                SELECT * FROM portfolio WHERE timestamp >= ?
            ''', conn, params=(start_time,))
            
            conn.close()
            
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs('exports', exist_ok=True)
            
            # บันทึกไฟล์
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_file = f'exports/trade_history_{timestamp}.csv'
            signals_file = f'exports/signal_history_{timestamp}.csv'
            portfolio_file = f'exports/portfolio_history_{timestamp}.csv'
            
            trades_df.to_csv(trades_file, index=False)
            signals_df.to_csv(signals_file, index=False)
            portfolio_df.to_csv(portfolio_file, index=False)
            
            self.logger.info(f"💾 Exported history to CSV files")
            
            return {
                'trades_file': trades_file,
                'signals_file': signals_file,
                'portfolio_file': portfolio_file
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export to CSV: {e}")
            return {}