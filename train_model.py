import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import optuna
from optuna.samplers import TPESampler
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

import talib
from scipy import stats

class AdvancedTradingModelTrainer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
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
    
    def objective_advanced_xgboost(self, trial):
        """Advanced XGBoost optimization สำหรับ trading strategy"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, 
            random_state=42, stratify=self.target
        )
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0)
        }
        
        model = XGBClassifier(**params, random_state=42, n_jobs=-1)
        
        # ใช้ TimeSeriesSplit สำหรับ validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred, average='weighted'))
        
        return np.mean(scores)
    
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
        
        return max(history.history['val_accuracy'])
    
    def advanced_auto_tune(self, n_trials=100):
        """Advanced auto-tuning สำหรับ multiple models"""
        print("Starting Advanced Auto Tuning...")
        
        studies = {}
        
        # XGBoost Tuning
        print("Tuning XGBoost...")
        study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study_xgb.optimize(self.objective_advanced_xgboost, n_trials=n_trials)
        studies['xgboost'] = study_xgb
        
        # LSTM Tuning
        print("Tuning LSTM...")
        study_lstm = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study_lstm.optimize(self.objective_advanced_lstm, n_trials=min(50, n_trials))
        studies['lstm'] = study_lstm
        
        return studies
    
    def train_ensemble_model(self, studies):
        """Train ensemble model ด้วย best parameters"""
        print("\nTraining Ensemble Models...")
        
        best_models = {}
        
        # Best XGBoost
        xgb_params = studies['xgboost'].best_params
        best_xgb = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1)
        best_xgb.fit(self.features, self.target)
        best_models['xgboost'] = best_xgb
        
        # Best LSTM
        lstm_params = studies['lstm'].best_params
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
    print("Loading and preprocessing multi-timeframe data...")
    features, target = trainer.load_and_preprocess_data()
    
    # Auto tuning
    print("Starting advanced auto-tuning...")
    studies = trainer.advanced_auto_tune(n_trials=80)
    
    # Training
    print("Training ensemble models...")
    best_models = trainer.train_ensemble_model(studies)
    
    # Evaluation
    print("Evaluating strategy performance...")
    results = trainer.evaluate_strategy_performance(best_models)
    
    print(f"\n✅ Training completed! Best model success rate: {trainer.best_score:.4f}")
    
    return trainer, best_models, results

if __name__ == "__main__":
    trainer, models, results = main()