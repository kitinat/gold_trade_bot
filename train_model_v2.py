import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

import talib
from scipy import stats
from datetime import datetime
import json

class PurgedGroupTimeSeriesSplit:
    """Time Series Cross-Validator with purging and embargo"""
    def __init__(self, n_splits=5, group_gap=1):
        self.n_splits = n_splits
        self.group_gap = group_gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)
        gaps = self.group_gap

        for i in range(self.n_splits):
            # Train ใช้ข้อมูลตั้งแต่ต้นจนถึง fold ปัจจุบัน
            train_start = 0
            train_end = (i + 1) * fold_size
            
            # Test อยู่หลังจาก train พร้อม gap
            test_start = train_end + gaps
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples:
                continue

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices

class AdvancedTradingModelTrainer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0

    def save_models(self, models, best_model_name):
        """
        Save trained models and preprocessing objects
        """
        import joblib
        
        try:
            # ตรวจสอบว่ามี model อยู่จริง
            if best_model_name not in models:
                raise ValueError(f"Model '{best_model_name}' not found in models dictionary")
            
            # สร้าง directory สำหรับบันทึก
            os.makedirs('saved_models', exist_ok=True)
            
            # บันทึก best model
            best_model = models[best_model_name]
            model_path = f'saved_models/best_trading_model_{best_model_name}.pkl'
            joblib.dump(best_model, model_path)
            print(f"✅ Saved model: {model_path}")
            
            # บันทึก scaler
            scaler_path = 'saved_models/feature_scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Saved scaler: {scaler_path}")
            
            # บันทึก feature columns
            features_path = 'saved_models/feature_columns.pkl'
            joblib.dump(list(self.features.columns), features_path)
            print(f"✅ Saved features: {features_path}")
            
            # บันทึก metadata
            metadata = {
                'model_name': best_model_name,
                'best_score': self.best_score,
                'n_features': len(self.features.columns),
                'n_samples': len(self.features),
                'feature_names': list(self.features.columns),
                'training_date': pd.Timestamp.now().isoformat()
            }
            metadata_path = 'saved_models/model_metadata.pkl'
            joblib.dump(metadata, metadata_path)
            print(f"✅ Saved metadata: {metadata_path}")
            
            print(f"\n🎉 All models and preprocessing objects saved successfully!")
            print(f"📁 Location: saved_models/")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            raise

    def save_training_report(self, results, best_params, studies):
        """บันทึกรายงานผลการ training"""
        import json
        from datetime import datetime
        
        os.makedirs('reports', exist_ok=True)
        
        # สร้างรายงาน
        report = {
            'training_date': datetime.now().isoformat(),
            'data_info': {
                'n_samples': len(self.features),
                'n_features': len(self.features.columns),
                'feature_names': list(self.features.columns),
                'target_distribution': pd.Series(self.target).value_counts().to_dict()
            },
            'model_results': results,
            'best_parameters': best_params,
            'optimization_summary': {}
        }
        
        # เพิ่มข้อมูลจาก Optuna studies
        for model_name, study in studies.items():
            report['optimization_summary'][model_name] = {
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'best_params': study.best_params
            }
        
        # บันทึกเป็น JSON
        report_path = f'reports/training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Saved training report: {report_path}")
        
        # บันทึกเป็น text file ที่อ่านง่าย
        txt_path = report_path.replace('.json', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("GOLD TRADING MODEL - TRAINING REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training Date: {report['training_date']}\n\n")
            
            f.write("DATA INFORMATION:\n")
            f.write(f"Samples: {report['data_info']['n_samples']}\n")
            f.write(f"Features: {report['data_info']['n_features']}\n\n")
            
            f.write("MODEL RESULTS:\n")
            for model, metrics in results.items():
                f.write(f"\n{model.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"📄 Saved text report: {txt_path}")

    def load_and_preprocess_data(self):
        """
        โหลดและเตรียมข้อมูลการเทรดด้วย Multi-Timeframe Approach
        """
        if self.data_path and os.path.exists(self.data_path):
            try:
                print(f"📁 กำลังโหลดข้อมูลจาก: {self.data_path}")
                # โหลดข้อมูลและจัดการ datetime
                self.data = pd.read_csv(self.data_path)
                
                # ตรวจสอบชื่อ column และแปลงเป็น datetime
                if 'open_time' in self.data.columns:
                    self.data['datetime'] = pd.to_datetime(self.data['open_time'], utc=True)
                    self.data = self.data.drop('open_time', axis=1)
                elif 'datetime' in self.data.columns:
                    self.data['datetime'] = pd.to_datetime(self.data['datetime'], utc=True)
                else:
                    raise ValueError("ไม่พบ column 'open_time' หรือ 'datetime' ในไฟล์")
                
                # ลบ timezone เพื่อให้ใช้งานง่ายขึ้น
                self.data['datetime'] = self.data['datetime'].dt.tz_localize(None)
                
                # เรียงลำดับตาม datetime
                self.data = self.data.sort_values('datetime').reset_index(drop=True)
                
                # ตรวจสอบว่ามี columns ที่จำเป็น
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in self.data.columns]
                if missing_cols:
                    raise ValueError(f"ขาด columns ที่จำเป็น: {missing_cols}")
                
                # ลบแถวที่มี volume = 0 (ไม่มีการซื้อขาย)
                initial_len = len(self.data)
                self.data = self.data[self.data['volume'] > 0].reset_index(drop=True)
                removed = initial_len - len(self.data)
                if removed > 0:
                    print(f"⚠️  ลบข้อมูลที่ volume = 0 จำนวน {removed} แถว")
                
                print(f"✅ โหลดข้อมูลสำเร็จ: {len(self.data)} แถว")
                print(f"📅 ช่วงเวลา: {self.data['datetime'].min()} ถึง {self.data['datetime'].max()}")
                
            except Exception as e:
                print(f"❌ ไม่สามารถโหลดไฟล์ {self.data_path} ได้: {e}")
                print("🔁 ใช้ข้อมูลตัวอย่างแทน...")
                self._generate_sample_data()
        else:
            if self.data_path:
                print(f"⚠️  ไม่พบไฟล์ {self.data_path}")
            print("📊 ใช้ข้อมูลตัวอย่าง...")
            self._generate_sample_data()
        
        # ตรวจสอบว่าข้อมูลถูกโหลดสำเร็จ
        if self.data is None or self.data.empty:
            print("⚠️  ไม่พบข้อมูล ใช้ข้อมูลตัวอย่างแทน")
            self._generate_sample_data()
        
        # ตรวจสอบขนาดข้อมูล
        if len(self.data) < 500:
            print(f"⚠️  ข้อมูลมีเพียง {len(self.data)} แถว ซึ่งอาจไม่เพียงพอสำหรับการ train")
            print("   แนะนำให้มีอย่างน้อย 1,000-10,000 แถว")
        
        # คำนวณ indicators ทั้งหมด
        self._calculate_multi_timeframe_indicators()
        
        # สร้าง features และ target ที่เหมาะสมกับกลยุทธี
        self._create_strategy_features_and_target()
        
        print(f"✅ เตรียมข้อมูลสำเร็จ: {len(self.data)} แถว, {len(self.features.columns)} features")
        
        return self.features, self.target
    
    def _generate_sample_data(self):
        """สร้างข้อมูลตัวอย่างที่มีความเหมือนจริงมากขึ้น"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='15T')
        np.random.seed(42)
        
        # สร้างราคาที่มีความสัมพันธ์กับทองคำจริง
        base_trend = np.cumsum(np.random.normal(0.00005, 0.002, len(dates)))
        seasonal = 0.001 * np.sin(2 * np.pi * np.arange(len(dates)) / (6.5 * 4 * 5))  # ฤดูกาลรายวัน
        noise = np.random.normal(0, 0.0015, len(dates))
        
        price = 1800 * np.exp(base_trend + seasonal + noise)
        
        self.data = pd.DataFrame({
            'datetime': dates,
            'open': price,
            'high': price * (1 + np.abs(np.random.normal(0, 0.0015, len(dates)))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.0015, len(dates)))),
            'close': price,
            'volume': np.random.lognormal(9, 1.2, len(dates))  # Volume แบบ log-normal
        })
        
        # ทำให้ high/low สอดคล้องกับ open/close
        self.data['high'] = np.maximum(self.data['high'], self.data[['open', 'close']].max(axis=1))
        self.data['low'] = np.minimum(self.data['low'], self.data[['open', 'close']].min(axis=1))
    
    def _calculate_multi_timeframe_indicators(self):
        """
        คำนวณ indicators สำหรับ Multi-Timeframe Trend Confirmation Strategy
        """
        df = self.data.copy()
        
        # === H1 TREND INDICATORS (Resample จาก 15min) ===
        df_h1 = df.set_index('datetime').resample('1H').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # คำนวณ indicators บน H1
        df_h1['h1_ema_21'] = talib.EMA(df_h1['close'], timeperiod=21)
        df_h1['h1_ema_50'] = talib.EMA(df_h1['close'], timeperiod=50)
        df_h1['h1_rsi'] = talib.RSI(df_h1['close'], timeperiod=14)
        df_h1['h1_macd'], df_h1['h1_macd_signal'], _ = talib.MACD(df_h1['close'])
        
        # Trend Strength บน H1
        df_h1['h1_trend_strength'] = (df_h1['h1_ema_21'] - df_h1['h1_ema_50']) / df_h1['h1_ema_50']
        df_h1['h1_trend_direction'] = np.where(df_h1['h1_ema_21'] > df_h1['h1_ema_50'], 1, -1)
        
        # Merge H1 data กลับไปที่ M15
        df_h1['datetime_h1'] = df_h1['datetime']
        df = pd.merge_asof(df.sort_values('datetime'), 
                          df_h1[['datetime', 'h1_ema_21', 'h1_ema_50', 'h1_rsi', 
                                'h1_macd', 'h1_macd_signal', 'h1_trend_strength', 
                                'h1_trend_direction']].sort_values('datetime'),
                          on='datetime', direction='backward')
        
        # === M15 MOMENTUM INDICATORS ===
        # Price-based
        df['m15_ema_9'] = talib.EMA(df['close'], timeperiod=9)
        df['m15_ema_21'] = talib.EMA(df['close'], timeperiod=21)
        df['m15_sma_50'] = talib.SMA(df['close'], timeperiod=50)
        
        # Momentum
        df['m15_rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['m15_macd'], df['m15_macd_signal'], df['m15_macd_hist'] = talib.MACD(df['close'])
        df['m15_stoch_k'], df['m15_stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Volatility
        df['m15_atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['m15_bb_upper'], df['m15_bb_middle'], df['m15_bb_lower'] = talib.BBANDS(df['close'], timeperiod=20)
        df['m15_bb_width'] = (df['m15_bb_upper'] - df['m15_bb_lower']) / df['m15_bb_middle']
        
        # Volume
        df['m15_volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['m15_volume_ratio'] = df['volume'] / df['m15_volume_sma']
        df['m15_obv'] = talib.OBV(df['close'], df['volume'])
        
        # Support/Resistance Dynamics
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']
        
        # Price Relationships (Multi-timeframe)
        df['price_vs_h1_ema21'] = (df['close'] - df['h1_ema_21']) / df['h1_ema_21']
        df['price_vs_h1_ema50'] = (df['close'] - df['h1_ema_50']) / df['h1_ema_50']
        df['m15_vs_h1_trend'] = (df['m15_ema_21'] - df['h1_ema_21']) / df['h1_ema_21']
        
        # Rate of Change และ Momentum
        df['m15_roc_5'] = talib.ROC(df['close'], timeperiod=5)
        df['m15_roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['m15_momentum'] = talib.MOM(df['close'], timeperiod=10)
        
        # Candlestick Patterns (สำหรับ confirmation)
        df['m15_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['m15_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['m15_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        
        # Volatility Regime
        df['volatility_regime'] = df['m15_atr'] / df['m15_atr'].rolling(50).mean()
        
        self.data = df.dropna()
    
    def _create_strategy_features_and_target(self, lookahead_periods=8, risk_reward_ratio=2.0):
        """
        สร้าง features และ target ที่สอดคล้องกับกลยุทธี Multi-Timeframe Trend Confirmation
        
        Args:
            lookahead_periods: จำนวนช่วงที่มองไปข้างหน้า (8*15min = 2 ชั่วโมง)
            risk_reward_ratio: อัตราส่วน risk:reward ที่ต้องการ
        """
        df = self.data.copy()
        
        # ===== STRATEGY-BASED TARGET CREATION =====
        
        # 1. คำนวณ stop loss และ take profit ตามกลยุทธี
        stop_loss_pct = 0.002  # 0.2% stop loss
        take_profit_pct = stop_loss_pct * risk_reward_ratio  # 0.4% take profit
        
        # 2. สร้าง target ตามเงื่อนไขกลยุทธี
        buy_signals = []
        
        for i in range(len(df)):
            if i < 50:  # ต้องการข้อมูลพอสำหรับ indicators
                buy_signals.append(0)
                continue
                
            # เงื่อนไขการเข้าซื้อตามกลยุทธี
            current = df.iloc[i]
            
            # Condition 1: H1 Trend Up
            h1_trend_up = (current['h1_trend_direction'] > 0 and 
                          current['h1_trend_strength'] > 0.001)
            
            # Condition 2: Price Pullback to Support
            price_pullback = (current['distance_to_support'] < 0.005 and  # ใกล้แนวรับ
                            current['m15_rsi_14'] < 60 and  # ไม่ overbought
                            current['price_vs_h1_ema21'] < 0.01)  # อยู่ใต้ H1 EMA21
            
            # Condition 3: Momentum Confirmation
            momentum_confirm = (current['m15_macd_hist'] > 0 and  # MACD histogram positive
                              current['m15_stoch_k'] > current['m15_stoch_d'] and  # Stochastic bullish
                              current['m15_volume_ratio'] > 1.0)  # Volume สูงกว่าค่าเฉลี่ย
            
            # Condition 4: Price Action Confirmation
            price_action_confirm = (current['m15_hammer'] > 0 or  # Hammer pattern
                                  current['m15_engulfing'] > 0 or  # Bullish engulfing
                                  current['m15_roc_5'] > 0)  # Momentum positive
            
            # รวมเงื่อนไขทั้งหมด
            if (h1_trend_up and price_pullback and 
                (momentum_confirm or price_action_confirm)):
                buy_signals.append(1)
            else:
                buy_signals.append(0)
        
        df['buy_signal'] = buy_signals
        
        # 3. สร้าง target ตามผลการเทรดในอนาคต
        future_returns = []
        
        for i in range(len(df)):
            if df.iloc[i]['buy_signal'] == 0:
                future_returns.append(0)
                continue
                
            current_price = df.iloc[i]['close']
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            
            # ตรวจสอบผลการเทรดใน lookahead_periods
            future_prices = df['close'].iloc[i+1:i+lookahead_periods+1]
            
            hit_take_profit = False
            hit_stop_loss = False
            
            for future_price in future_prices:
                if future_price >= take_profit_price:
                    hit_take_profit = True
                    break
                if future_price <= stop_loss_price:
                    hit_stop_loss = True
                    break
            
            # กำหนด target
            if hit_take_profit:
                future_returns.append(1)  # กำไร
            elif hit_stop_loss:
                future_returns.append(-1)  # ขาดทุน
            else:
                future_returns.append(0)  # ไม่เกิดอะไร
        
        df['strategy_target'] = future_returns
        
        # ===== FEATURE SELECTION =====
        
        # เลือก features ที่สำคัญสำหรับกลยุทธี
        feature_columns = [
            # H1 Trend Features
            'h1_trend_strength', 'h1_trend_direction', 'h1_rsi', 'h1_macd',
            
            # M15 Momentum Features
            'm15_rsi_14', 'm15_macd', 'm15_macd_hist', 'm15_stoch_k', 'm15_stoch_d',
            'm15_roc_5', 'm15_roc_10', 'm15_momentum',
            
            # Price Position Features
            'price_vs_h1_ema21', 'price_vs_h1_ema50', 'distance_to_support', 
            'distance_to_resistance', 'm15_vs_h1_trend',
            
            # Volume and Volatility
            'm15_volume_ratio', 'm15_obv', 'm15_atr', 'm15_bb_width', 'volatility_regime',
            
            # Candlestick Patterns
            'm15_hammer', 'm15_engulfing', 'm15_doji'
        ]
        
        self.features = df[feature_columns]
        
        # สร้าง multi-class target สำหรับการเรียนรู้ที่ละเอียดยิ่งขึ้น
        conditions = [
            df['strategy_target'] == 1,   # กำไร
            df['strategy_target'] == -1,  # ขาดทุน
            df['strategy_target'] == 0    # ไม่เกิดอะไร
        ]
        choices = [2, 0, 1]  # 2=กำไร, 1=กลาง, 0=ขาดทุน
        
        self.target = np.select(conditions, choices, default=1)
        
        print("Target Distribution:")
        target_counts = pd.Series(self.target).value_counts().sort_index()
        for val, count in target_counts.items():
            label = {2: 'Profit', 1: 'Neutral', 0: 'Loss'}[val]
            print(f"  {label}: {count} ({count/len(self.target)*100:.1f}%)")
    
    def prepare_advanced_lstm_data(self, time_steps=30, sequence_stride=2):
        """
        เตรียมข้อมูลสำหรับ LSTM ที่ซับซ้อนขึ้น
        """
        X = self.features.values
        y = self.target
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences with stride
        X_seq, y_seq = [], []
        for i in range(time_steps * sequence_stride, len(X_scaled), sequence_stride):
            X_seq.append(X_scaled[i-time_steps*sequence_stride:i:sequence_stride])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _create_time_groups(self, period='D'):
        """สร้าง groups ตามเวลาเพื่อป้องกัน data leakage"""
        if hasattr(self, 'data') and self.data is not None:
            dates = self.data['datetime'].dt.to_period(period)
            return dates.astype('category').cat.codes.values
        return np.arange(len(self.features))
    
    def _cross_validate_model(self, model, X, y, n_splits=3):
        """ฟังก์ชันร่วมสำหรับ cross-validation"""
        
        tscv = PurgedGroupTimeSeriesSplit(
            n_splits=n_splits,
            group_gap=5
        )
        
        scores = []
        groups = self._create_time_groups()
        
        for train_idx, val_idx in tscv.split(X, groups=groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # ใช้ multiple metrics
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # รวมคะแนน (weighted combination)
            combined_score = 0.7 * f1 + 0.3 * accuracy
            scores.append(combined_score)
        
        return np.mean(scores)
    
    def objective_advanced_xgboost(self, trial):
        """Advanced XGBoost optimization สำหรับ trading strategy"""
        
        # สร้าง parameter space ที่ครอบคลุม
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_leaves': trial.suggest_int('max_leaves', 0, 64),
            'tree_method': trial.suggest_categorical('tree_method', ['hist', 'auto']),
            'eval_metric': 'mlogloss'
        }
        
        model = XGBClassifier(**params, random_state=42, n_jobs=-1)
        
        return self._cross_validate_model(model, self.features, self.target)
    
    def objective_random_forest(self, trial):
        """Random Forest optimization สำหรับ trading"""
        
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, 40, None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': bootstrap,
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
        }
        
        # เพิ่ม max_samples เฉพาะเมื่อ bootstrap=True
        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', 0.6, 1.0)
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        return self._cross_validate_model(model, self.features, self.target)
    
    def objective_lightgbm(self, trial):
        """LightGBM optimization"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'is_unbalance': True,  # เพิ่มเพื่อจัดการ class imbalance
            'objective': 'multiclass',
            'metric': 'multi_logloss'
        }
        
        model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        
        return self._cross_validate_model(model, self.features, self.target)
    
    def objective_advanced_lstm(self, trial):
        """Advanced LSTM optimization"""
        time_steps = trial.suggest_categorical('time_steps', [20, 30, 40])
        sequence_stride = trial.suggest_int('sequence_stride', 1, 3)
        
        X_seq, y_seq = self.prepare_advanced_lstm_data(time_steps, sequence_stride)
        
        # Split data
        split_idx = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Model architecture
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        model = Sequential()
        model.add(Conv1D(
            filters=trial.suggest_int('filters_1', 32, 128),
            kernel_size=trial.suggest_int('kernel_size', 2, 5),
            activation='relu',
            input_shape=(time_steps, X_train.shape[2])
        ))
        model.add(MaxPooling1D(pool_size=2))
        
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)
            model.add(LSTM(
                units=trial.suggest_int(f'units_{i+1}', 32, 128),
                return_sequences=return_sequences,
                kernel_regularizer=l2(trial.suggest_float(f'l2_{i+1}', 1e-5, 1e-2))
            ))
            model.add(Dropout(trial.suggest_float(f'dropout_{i+1}', 0.1, 0.5)))
            if i < n_layers - 1:
                model.add(BatchNormalization())
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(trial.suggest_float('dropout_final', 0.1, 0.5)))
        model.add(Dense(3, activation='softmax'))
        
        # Compile
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
            epochs=100,
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7)
            ]
        )
        
        # Pruning
        trial.report(history.history['val_accuracy'][-1], step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return max(history.history['val_accuracy'])
    
    def create_advanced_sampler(self):
        """สร้าง advanced sampler สำหรับ Optuna"""
        
        # ใช้ TPESampler ด้วยการตั้งค่าขั้นสูง
        sampler = TPESampler(
            seed=42,
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            n_startup_trials=10,
            n_ei_candidates=24
        )
        
        return sampler
    
    def create_advanced_pruner(self):
        """สร้าง advanced pruner"""
        
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=100,
            reduction_factor=3
        )
        
        return pruner
    
    def advanced_auto_tune(self, n_trials=100, models_to_tune=None):
        """Advanced auto-tuning สำหรับ multiple models"""
        
        if models_to_tune is None:
            models_to_tune = ['xgboost', 'lightgbm', 'random_forest', 'lstm']
        
        print("🚀 Starting Advanced Auto Tuning...")
        print(f"📊 Models to tune: {models_to_tune}")
        print(f"🔬 Number of trials: {n_trials}")
        
        studies = {}
        best_params = {}
        
        # สร้าง study สำหรับแต่ละโมเดล
        for model_name in models_to_tune:
            print(f"\n🎯 Tuning {model_name.upper()}...")
            
            # ตั้งค่า study
            study = optuna.create_study(
                direction='maximize',
                sampler=self.create_advanced_sampler(),
                pruner=self.create_advanced_pruner(),
                study_name=f"{model_name}_tuning"
            )
            
            # กำหนดฟังก์ชัน objective ตามโมเดล
            if model_name == 'xgboost':
                objective_func = self.objective_advanced_xgboost
                n_trials_model = n_trials
            elif model_name == 'lightgbm':
                objective_func = self.objective_lightgbm
                n_trials_model = n_trials
            elif model_name == 'random_forest':
                objective_func = self.objective_random_forest
                n_trials_model = n_trials
            elif model_name == 'lstm':
                objective_func = self.objective_advanced_lstm
                n_trials_model = min(50, n_trials)  # LSTM ใช้เวลามาก
            else:
                continue
            
            # เรียก optimization
            study.optimize(
                objective_func, 
                n_trials=n_trials_model,
                show_progress_bar=True,
                gc_after_trial=True
            )
            
            studies[model_name] = study
            best_params[model_name] = study.best_params
            
            # แสดงผลลัพธ์
            print(f"✅ Best {model_name} score: {study.best_value:.4f}")
            print(f"🔧 Best parameters: {study.best_params}")
            
            # บันทึก visualization
            self._save_optuna_visualizations(study, model_name)
        
        return studies, best_params
    
    def _save_optuna_visualizations(self, study, model_name):
        """บันทึก visualization ของ Optuna"""
        
        try:
            # สร้าง directory สำหรับบันทึก
            os.makedirs('optuna_plots', exist_ok=True)
            
            # Optimization history plot
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(f'optuna_plots/{model_name}_optimization_history.html')
            
            # Parameter importance plot
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(f'optuna_plots/{model_name}_param_importance.html')
            
            # Parallel coordinate plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(f'optuna_plots/{model_name}_parallel_coordinate.html')
            
            print(f"📈 Saved visualizations for {model_name}")
            
        except Exception as e:
            print(f"⚠️ Could not save visualizations: {e}")
    
    def analyze_optuna_results(self, studies):
        """วิเคราะห์ผลลัพธ์จากการ tuning"""
        
        print("\n" + "="*60)
        print("OPTUNA TUNING ANALYSIS")
        print("="*60)
        
        for model_name, study in studies.items():
            print(f"\n📊 {model_name.upper()} Analysis:")
            print(f"Best Trial: #{study.best_trial.number}")
            print(f"Best Value: {study.best_value:.4f}")
            print(f"Completed Trials: {len(study.trials)}")
            
            # วิเคราะห์ parameter importance
            try:
                importance_df = self._get_parameter_importance(study)
                print("Parameter Importance:")
                print(importance_df.head(10))
            except:
                print("Could not calculate parameter importance")
            
            # วิเคราะห์ convergence
            self._analyze_convergence(study, model_name)
    
    def _get_parameter_importance(self, study):
        """คำนวณ parameter importance"""
        param_importance = optuna.importance.get_param_importances(study)
        return pd.DataFrame(
            list(param_importance.items()), 
            columns=['Parameter', 'Importance']
        ).sort_values('Importance', ascending=False)
    
    def _analyze_convergence(self, study, model_name):
        """วิเคราะห์การลู่เข้าของการ optimization"""
        values = [trial.value for trial in study.trials if trial.value is not None]
        
        if len(values) > 10:
            initial_avg = np.mean(values[:10])
            final_avg = np.mean(values[-10:])
            improvement = final_avg - initial_avg
            
            print(f"Convergence Analysis:")
            print(f"Initial 10 trials avg: {initial_avg:.4f}")
            print(f"Final 10 trials avg: {final_avg:.4f}")
            print(f"Improvement: {improvement:.4f}")
            
            if improvement < 0.01:
                print("⚠️  Convergence may be stagnating")
    
    def train_models_with_best_params(self, best_params):
        """Train models ด้วย best parameters ที่ได้จากการ tuning"""
        print("\n🏋️ Training Models with Best Parameters...")
        
        best_models = {}
        
        # Train XGBoost
        if 'xgboost' in best_params:
            print("Training XGBoost...")
            xgb_params = best_params['xgboost'].copy()
            # Remove parameters that might cause issues
            xgb_params.pop('eval_metric', None)
            best_xgb = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1)
            best_xgb.fit(self.features, self.target)
            best_models['xgboost'] = best_xgb
        
        # Train LightGBM
        if 'lightgbm' in best_params:
            print("Training LightGBM...")
            lgb_params = best_params['lightgbm'].copy()
            # Remove parameters that might cause issues
            lgb_params.pop('objective', None)
            lgb_params.pop('metric', None)
            lgb_params.pop('is_unbalance', None)
            # เพิ่ม num_class สำหรับ multi-class classification
            best_lgb = LGBMClassifier(
                **lgb_params, 
                objective='multiclass',
                num_class=3,
                is_unbalance=True,
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
            )
            best_lgb.fit(self.features, self.target)
            best_models['lightgbm'] = best_lgb
        
        # Train Random Forest
        if 'random_forest' in best_params:
            print("Training Random Forest...")
            rf_params = best_params['random_forest'].copy()
            best_rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
            best_rf.fit(self.features, self.target)
            best_models['random_forest'] = best_rf
        
        # Train LSTM
        if 'lstm' in best_params:
            print("Training LSTM...")
            lstm_params = best_params['lstm'].copy()
            time_steps = lstm_params['time_steps']
            sequence_stride = lstm_params['sequence_stride']
            
            X_seq, y_seq = self.prepare_advanced_lstm_data(time_steps, sequence_stride)
            
            best_lstm = Sequential()
            best_lstm.add(Conv1D(
                filters=lstm_params['filters_1'],
                kernel_size=lstm_params['kernel_size'],
                activation='relu',
                input_shape=(time_steps, X_seq.shape[2])
            ))
            best_lstm.add(MaxPooling1D(pool_size=2))
            
            for i in range(lstm_params['n_layers']):
                return_sequences = (i < lstm_params['n_layers'] - 1)
                best_lstm.add(LSTM(
                    units=lstm_params[f'units_{i+1}'],
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(lstm_params[f'l2_{i+1}'])
                ))
                best_lstm.add(Dropout(lstm_params[f'dropout_{i+1}']))
                if i < lstm_params['n_layers'] - 1:
                    best_lstm.add(BatchNormalization())
            
            best_lstm.add(Dense(64, activation='relu'))
            best_lstm.add(Dropout(lstm_params['dropout_final']))
            best_lstm.add(Dense(3, activation='softmax'))
            
            best_lstm.compile(
                optimizer=Adam(learning_rate=lstm_params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_seq), 
                y=y_seq
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            best_lstm.fit(
                X_seq, y_seq,
                batch_size=lstm_params['batch_size'],
                epochs=150,
                validation_split=0.2,
                class_weight=class_weight_dict,
                verbose=1,
                callbacks=[
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
                ]
            )
            
            best_models['lstm'] = best_lstm
        
        return best_models
    
    def evaluate_strategy_performance(self, models):
        """ประเมินผลลัพธ์ของกลยุทธีอย่างละเอียด"""
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in models.items():
            if name == 'lstm':
                # LSTM evaluation
                time_steps = 30
                X_seq, y_seq = self.prepare_advanced_lstm_data(time_steps)
                split_idx = int(0.8 * len(X_seq))
                X_test_seq = X_seq[split_idx:]
                y_test_seq = y_seq[split_idx:]
                
                y_pred_proba = model.predict(X_test_seq)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
            else:
                # Tree-based models evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    self.features, self.target, test_size=0.2, 
                    random_state=42, stratify=self.target
                )
                
                y_pred = model.predict(X_test)
                y_test_seq = y_test
            
            # Detailed metrics
            accuracy = accuracy_score(y_test_seq, y_pred)
            f1 = f1_score(y_test_seq, y_pred, average='weighted')
            
            # Strategy-specific metrics
            profit_signals = np.sum((y_pred == 2) & (y_test_seq == 2))
            total_signals = np.sum(y_pred == 2)
            success_rate = profit_signals / total_signals if total_signals > 0 else 0
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'success_rate': success_rate,
                'total_signals': total_signals,
                'profit_signals': profit_signals
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Success Rate: {success_rate:.4f}")
            print(f"Total Buy Signals: {total_signals}")
            print(f"Profitable Signals: {profit_signals}")
            print(classification_report(y_test_seq, y_pred, 
                                      target_names=['Loss', 'Neutral', 'Profit']))
        
        # Find best model
        if results:
            best_model_name = max(results, key=lambda x: results[x]['success_rate'])
            self.best_model = models[best_model_name]
            self.best_score = results[best_model_name]['success_rate']
            
            print(f"\n🏆 BEST STRATEGY MODEL: {best_model_name}")
            print(f"🎯 Success Rate: {self.best_score:.4f}")
            print(f"📊 Accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        return results

def main():
    """ฟังก์ชันหลักสำหรับรันการฝึกโมเดลขั้นสูง"""
    # สร้าง trainer
    trainer = AdvancedTradingModelTrainer(data_path='historical_data/PAXG-USDT_15min_20230101-20251018.csv')
    
    # โหลดและเตรียมข้อมูล
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    features, target = trainer.load_and_preprocess_data()
    
    # Auto tuning
    print("\n" + "="*60)
    print("STEP 2: AUTO-TUNING MODELS")
    print("="*60)
    studies, best_params = trainer.advanced_auto_tune(n_trials=50)
    
    # วิเคราะห์ผลลัพธ์ tuning
    print("\n" + "="*60)
    print("STEP 3: ANALYZING TUNING RESULTS")
    print("="*60)
    trainer.analyze_optuna_results(studies)
    
    # Training ด้วย best parameters
    print("\n" + "="*60)
    print("STEP 4: TRAINING FINAL MODELS")
    print("="*60)
    best_models = trainer.train_models_with_best_params(best_params)
    
    # Evaluation
    print("\n" + "="*60)
    print("STEP 5: EVALUATING PERFORMANCE")
    print("="*60)
    results = trainer.evaluate_strategy_performance(best_models)
    
    # บันทึกผลลัพธ์
    print("\n" + "="*60)
    print("STEP 6: SAVING RESULTS")
    print("="*60)
    
    if results and best_models:
        # หา best model
        best_model_name = max(results, key=lambda x: results[x]['success_rate'])
        
        # บันทึก models
        print(f"\n💾 Saving best model ({best_model_name})...")
        print(f"🎯 Best success rate: {results[best_model_name]['success_rate']:.4f}")
        trainer.save_models(best_models, best_model_name)
        
        # บันทึกรายงาน
        print("\n📊 Generating training report...")
        trainer.save_training_report(results, best_params, studies)
        
        # บันทึกข้อมูลเพิ่มเติม
        import joblib
        os.makedirs('saved_models', exist_ok=True)
        
        joblib.dump(results, 'saved_models/model_results.pkl')
        joblib.dump(best_params, 'saved_models/best_params.pkl')
        joblib.dump(studies, 'saved_models/optuna_studies.pkl')
        
        print("✅ Saved all results and parameters")
        
        # สรุปผล
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"✅ Training completed successfully!")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"🎯 Success Rate: {results[best_model_name]['success_rate']:.4f}")
        print(f"📊 Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"📈 F1-Score: {results[best_model_name]['f1_score']:.4f}")
        print(f"\n📁 Saved files:")
        print(f"   - saved_models/best_trading_model_{best_model_name}.pkl")
        print(f"   - saved_models/feature_scaler.pkl")
        print(f"   - saved_models/feature_columns.pkl")
        print(f"   - saved_models/model_metadata.pkl")
        print(f"   - reports/training_report_*.json")
        print(f"   - reports/training_report_*.txt")
        print("="*60)
        
    else:
        print("⚠️ WARNING: No models trained or evaluated!")
        print("Check if data was loaded correctly and models were created.")
    
    return trainer, best_models, results, studies

if __name__ == "__main__":
    trainer, models, results, studies = main()