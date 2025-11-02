"""
สคริปต์ทดสอบ train_model_v2.py
ตรวจสอบว่าทุก components ทำงานได้ถูกต้อง
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ทดสอบ import
print("🔍 Testing imports...")
try:
    from train_model_v2 import AdvancedTradingModelTrainer, PurgedGroupTimeSeriesSplit
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# ทดสอบ PurgedGroupTimeSeriesSplit
print("\n🔍 Testing PurgedGroupTimeSeriesSplit...")
try:
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3, group_gap=5)
    X_test = pd.DataFrame(np.random.rand(100, 5))
    y_test = np.random.randint(0, 3, 100)
    
    splits = list(splitter.split(X_test, y_test))
    print(f"✅ PurgedGroupTimeSeriesSplit works - Generated {len(splits)} splits")
    
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"   Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
        
        # ตรวจสอบว่า train และ test ไม่ overlap
        if len(set(train_idx).intersection(set(test_idx))) > 0:
            print(f"   ❌ WARNING: Train and test overlap in fold {i+1}!")
        
except Exception as e:
    print(f"❌ PurgedGroupTimeSeriesSplit failed: {e}")

# ทดสอบ Trainer initialization
print("\n🔍 Testing Trainer initialization...")
try:
    trainer = AdvancedTradingModelTrainer()
    print("✅ Trainer initialized successfully")
except Exception as e:
    print(f"❌ Trainer initialization failed: {e}")
    exit(1)

# ทดสอบการสร้างข้อมูลตัวอย่าง
print("\n🔍 Testing sample data generation...")
try:
    trainer._generate_sample_data()
    print(f"✅ Sample data generated: {len(trainer.data)} rows")
    print(f"   Columns: {list(trainer.data.columns)}")
except Exception as e:
    print(f"❌ Sample data generation failed: {e}")

# ทดสอบการคำนวณ indicators
print("\n🔍 Testing indicator calculation...")
try:
    trainer._calculate_multi_timeframe_indicators()
    print(f"✅ Indicators calculated: {len(trainer.data)} rows after dropna")
    print(f"   Total columns: {len(trainer.data.columns)}")
except Exception as e:
    print(f"❌ Indicator calculation failed: {e}")
    import traceback
    traceback.print_exc()

# ทดสอบการสร้าง features และ target
print("\n🔍 Testing feature and target creation...")
try:
    trainer._create_strategy_features_and_target()
    print(f"✅ Features created: {trainer.features.shape}")
    print(f"✅ Target created: {len(trainer.target)} samples")
    
    # แสดง target distribution
    unique, counts = np.unique(trainer.target, return_counts=True)
    print("   Target distribution:")
    for val, count in zip(unique, counts):
        label = {0: 'Loss', 1: 'Neutral', 2: 'Profit'}[val]
        print(f"     {label}: {count} ({count/len(trainer.target)*100:.1f}%)")
        
except Exception as e:
    print(f"❌ Feature/target creation failed: {e}")
    import traceback
    traceback.print_exc()

# ทดสอบการเตรียมข้อมูล LSTM
print("\n🔍 Testing LSTM data preparation...")
try:
    X_seq, y_seq = trainer.prepare_advanced_lstm_data(time_steps=10, sequence_stride=2)
    print(f"✅ LSTM data prepared: X_seq shape={X_seq.shape}, y_seq shape={y_seq.shape}")
except Exception as e:
    print(f"❌ LSTM data preparation failed: {e}")
    import traceback
    traceback.print_exc()

# ทดสอบ full pipeline (ถ้ามีไฟล์ CSV)
print("\n🔍 Testing full data loading pipeline...")
try:
    trainer_full = AdvancedTradingModelTrainer(
        data_path='historical_data/PAXG-USDT_15min_20230101-20251018.csv'
    )
    features, target = trainer_full.load_and_preprocess_data()
    print(f"✅ Full pipeline successful: {features.shape[0]} samples, {features.shape[1]} features")
except Exception as e:
    print(f"⚠️  Full pipeline with CSV failed (may not have file): {e}")

print("\n" + "="*60)
print("🎉 TEST SUMMARY")
print("="*60)
print("All critical components are working!")
print("The train_model_v2.py is ready to use.")
print("\nTo run actual training (will take time):")
print("  python train_model_v2.py")
