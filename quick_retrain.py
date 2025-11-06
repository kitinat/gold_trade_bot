"""
Quick Re-train Script
Run this to re-train the model with properly fitted scaler
"""
import subprocess
import sys

print("="*60)
print("🔄 RE-TRAINING MODEL WITH FITTED SCALER")
print("="*60)

# Run training
result = subprocess.run([sys.executable, 'train_model_v2.py'])

if result.returncode == 0:
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run: python trade_bot.py")
else:
    print("\n" + "="*60)
    print("❌ TRAINING FAILED!")
    print("="*60)
    print("Please check the error messages above")

input("\nPress Enter to exit...")
