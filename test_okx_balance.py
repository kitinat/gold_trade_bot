# test_okx_balance.py
# ทดสอบการดึง balance จาก OKX และแสดงโครงสร้างข้อมูล

import ccxt
import json
from config_bot import OKX_CONFIG

def test_okx_balance():
    """ทดสอบการดึง balance และแสดงโครงสร้างข้อมูล"""
    
    print("="*60)
    print("OKX BALANCE STRUCTURE TEST")
    print("="*60)
    
    # สร้าง exchange instance
    exchange = ccxt.okx(OKX_CONFIG)
    
    # แสดงข้อมูล configuration
    mode = "TESTNET/SANDBOX" if OKX_CONFIG.get('sandbox', False) else "PRODUCTION/LIVE"
    print(f"\n📡 Connection Mode: {mode}")
    print(f"API Key: {OKX_CONFIG['api_key'][:10]}...{OKX_CONFIG['api_key'][-4:]}")
    
    # ทดสอบ account types ต่างๆ
    account_types = [None, 'spot', 'trading', 'funding']
    
    for acc_type in account_types:
        print(f"\n{'='*60}")
        print(f"Testing Account Type: {acc_type or 'default'}")
        print("="*60)
        
        try:
            if acc_type:
                params = {'type': acc_type}
                balance = exchange.fetch_balance(params)
            else:
                balance = exchange.fetch_balance()
            
            # แสดงโครงสร้างหลัก
            print(f"\n✅ Successfully fetched balance")
            print(f"Top-level keys: {list(balance.keys())}")
            
            # แสดง 'total' dict
            if 'total' in balance:
                print(f"\n📊 'total' dict:")
                total_dict = balance['total']
                if total_dict:
                    print(f"  Currencies: {list(total_dict.keys())}")
                    for currency, amount in list(total_dict.items())[:5]:
                        print(f"    {currency}: {amount}")
                    if len(total_dict) > 5:
                        print(f"    ... and {len(total_dict) - 5} more")
                else:
                    print("  (empty)")
            
            # แสดง 'free' dict
            if 'free' in balance:
                print(f"\n💵 'free' dict:")
                free_dict = balance['free']
                if free_dict:
                    print(f"  Currencies: {list(free_dict.keys())}")
                    for currency, amount in list(free_dict.items())[:5]:
                        print(f"    {currency}: {amount}")
                    if len(free_dict) > 5:
                        print(f"    ... and {len(free_dict) - 5} more")
                else:
                    print("  (empty)")
            
            # แสดง 'info' structure
            if 'info' in balance:
                print(f"\n📋 'info' structure:")
                info = balance['info']
                print(f"  Keys: {list(info.keys())}")
                
                # ถ้ามี 'data' ให้แสดงตัวอย่าง
                if 'data' in info:
                    data = info['data']
                    print(f"\n  'data' type: {type(data).__name__}")
                    
                    if isinstance(data, list) and len(data) > 0:
                        print(f"  'data' length: {len(data)}")
                        print(f"  First item keys: {list(data[0].keys()) if data[0] else 'N/A'}")
                        print(f"\n  First 3 items:")
                        for i, item in enumerate(data[:3], 1):
                            if isinstance(item, dict):
                                currency = item.get('ccy') or item.get('currency') or item.get('coin')
                                balance_val = item.get('bal') or item.get('balance') or item.get('total')
                                avail = item.get('availBal') or item.get('available') or item.get('free')
                                print(f"    {i}. Currency: {currency}, Balance: {balance_val}, Available: {avail}")
                    elif isinstance(data, dict):
                        print(f"  'data' keys: {list(data.keys())[:10]}")
                
                # แสดง info raw (จำกัด 1000 characters)
                info_str = json.dumps(info, indent=2, default=str)
                if len(info_str) > 1000:
                    info_str = info_str[:1000] + "\n... (truncated)"
                print(f"\n  Raw info:\n{info_str}")
            
            # สรุป USDT
            print(f"\n{'='*60}")
            print("USDT SUMMARY:")
            
            usdt_found = False
            
            # Method 1: Standard
            if 'USDT' in balance.get('total', {}):
                print(f"✅ Method 1 (Standard): USDT in 'total'")
                print(f"   Total: {balance['total']['USDT']}")
                print(f"   Free: {balance['free'].get('USDT', 0)}")
                usdt_found = True
            
            # Method 2: Direct
            if 'USDT' in balance:
                print(f"✅ Method 2 (Direct): USDT found")
                print(f"   Value: {balance['USDT']}")
                usdt_found = True
            
            # Method 3: info.data
            if 'info' in balance and 'data' in balance['info']:
                data = balance['info']['data']
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            currency = item.get('ccy') or item.get('currency') or item.get('coin')
                            if currency == 'USDT':
                                print(f"✅ Method 3 (info.data list): USDT found")
                                print(f"   Balance: {item.get('bal') or item.get('balance')}")
                                print(f"   Available: {item.get('availBal') or item.get('available')}")
                                usdt_found = True
                                break
            
            if not usdt_found:
                print("❌ USDT NOT FOUND in this account type")
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # คำแนะนำ
    print("\n💡 RECOMMENDATIONS:")
    print("1. If USDT not found in any account type:")
    print("   - Verify you have USDT in your OKX account")
    print("   - Check you're using correct API credentials")
    print("   - Verify API has 'Read' permission")
    print("")
    print("2. If using PRODUCTION mode without funds:")
    print("   - Switch to TESTNET/SANDBOX in config_bot.py")
    print("   - Set: OKX_CONFIG['sandbox'] = True")
    print("   - Get testnet keys from: https://www.okx.com/demo-trading")
    print("")
    print("3. If balance structure is different:")
    print("   - Review the 'info' raw data above")
    print("   - Update check_position() function in trade_bot.py")
    print("="*60)

if __name__ == "__main__":
    try:
        test_okx_balance()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
