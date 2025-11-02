# run_bot.py
#!/usr/bin/env python3
"""
Main script to run the OKX Trading Bot
"""

import sys
import os
import logging
from trade_bot import OKXTradingBot

def setup_environment():
    """Setup environment and check prerequisites"""
    # ตรวจสอบว่าไฟล์ .env มีอยู่
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please create .env file with your configuration.")
        print("   Copy from .env.example and fill in your API keys.")
        return False
    
    # ตรวจสอบว่าโฟลเดอร์จำเป็นมีอยู่
    os.makedirs('exports', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    return True

def main():
    """Main function"""
    print("🚀 Starting OKX Trading Bot...")
    
    if not setup_environment():
        sys.exit(1)
    
    try:
        # สร้างและรันบอท
        bot = OKXTradingBot()
        
        print("✅ Bot initialized successfully")
        print("📊 Configuration:")
        print(f"   - Symbol: PAXG/USDT")
        print(f"   - Timeframe: 15m") 
        print(f"   - Trade Size: $100")
        print(f"   - Trading Enabled: Yes")
        print(f"   - Hourly Reports: Yes")
        print("\n🔄 Starting main loop... (Press Ctrl+C to stop)")
        
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Bot crashed")
        sys.exit(1)

if __name__ == "__main__":
    main()