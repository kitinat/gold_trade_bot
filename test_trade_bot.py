"""
สคริปต์ทดสอบ trade_bot.py 
ตรวจสอบ syntax และ logical errors
"""
import sys
import ast

print("="*60)
print("🔍 ANALYZING trade_bot.py FOR BUGS")
print("="*60)

bugs_found = []

# Test 1: Syntax check
print("\n[1/4] Checking Python syntax...")
try:
    with open('trade_bot.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    ast.parse(code)
    print("✅ No syntax errors")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: Line {e.lineno}: {e.msg}")
    bugs_found.append(f"Syntax error at line {e.lineno}")
    sys.exit(1)

# Test 2: Check for setup_trade_history called before usage
print("\n[2/4] Checking initialization order...")
with open('trade_bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

setup_trade_history_line = None
first_history_usage_line = None

for i, line in enumerate(lines, 1):
    if 'def setup_trade_history' in line:
        setup_trade_history_line = i
    if 'self.history_manager' in line and first_history_usage_line is None:
        if 'def setup_trade_history' not in line and '=' not in line:
            first_history_usage_line = i

# ตรวจสอบว่า setup_models ใช้ history_manager ก่อนที่จะถูกสร้าง
setup_models_end = None
for i, line in enumerate(lines, 1):
    if i > 106 and i < 136:  # ช่วง setup_models
        if 'self.history_manager.log_error' in line:
            bugs_found.append({
                'type': 'CRITICAL',
                'line': i,
                'issue': 'self.history_manager used before initialization',
                'code': line.strip(),
                'fix': 'Move setup_trade_history() before setup_models() in __init__'
            })
            print(f"❌ BUG at line {i}: history_manager used before initialization!")

if not bugs_found or 'history_manager' not in str(bugs_found):
    print("✅ Initialization order appears correct")

# Test 3: Check for f-string syntax errors
print("\n[3/4] Checking f-string formatting...")
fstring_bugs = []
for i, line in enumerate(lines, 1):
    if 'f"""' in line or "f'''" in line or 'f"' in line or "f'" in line:
        # Check for common f-string issues
        if '{' in line and '}' in line:
            # ตรวจสอบ conditional expression ใน f-string
            if ' if ' in line and ' else ' in line and ':.2f' in line:
                # Pattern: ${value:.2f if condition else 0:.2f}
                if line.count(':.2f') > 1 and 'if' in line and 'else' in line:
                    fstring_bugs.append({
                        'line': i,
                        'code': line.strip()[:100],
                        'issue': 'Invalid f-string format - double formatting spec'
                    })

if fstring_bugs:
    print(f"❌ Found {len(fstring_bugs)} f-string formatting issues:")
    for bug in fstring_bugs:
        print(f"  Line {bug['line']}: {bug['code']}")
        bugs_found.append(bug)
else:
    print("✅ F-string formatting looks good")

# Test 4: Check specific known bugs
print("\n[4/4] Checking specific known issues...")

known_issues = [
    {
        'line_range': (483, 484),
        'pattern': 'portfolio[\'total_usdt\']:.2f if portfolio else 0:.2f',
        'issue': 'Invalid f-string conditional with format spec',
        'severity': 'CRITICAL'
    }
]

for issue in known_issues:
    start, end = issue['line_range']
    found = False
    for i in range(start-1, min(end, len(lines))):
        if issue['pattern'] in lines[i]:
            found = True
            bugs_found.append({
                'type': issue['severity'],
                'line': i+1,
                'issue': issue['issue'],
                'code': lines[i].strip()[:80]
            })
            print(f"❌ {issue['severity']}: Line {i+1}")
            print(f"   {issue['issue']}")
            break
    
    if not found:
        print(f"✅ No issue found in lines {start}-{end}")

# Summary
print("\n" + "="*60)
if bugs_found:
    print(f"❌ FOUND {len(bugs_found)} BUG(S)!")
    print("="*60)
    
    critical = [b for b in bugs_found if b.get('type') == 'CRITICAL' or b.get('severity') == 'CRITICAL']
    medium = [b for b in bugs_found if b.get('type') not in ['CRITICAL', None] and b.get('severity') not in ['CRITICAL', None]]
    
    if critical:
        print(f"\n🔴 CRITICAL BUGS ({len(critical)}):")
        for i, bug in enumerate(critical, 1):
            print(f"\n  Bug #{i}:")
            print(f"  Line: {bug.get('line', 'N/A')}")
            print(f"  Issue: {bug.get('issue', 'N/A')}")
            if 'code' in bug:
                print(f"  Code: {bug['code']}")
            if 'fix' in bug:
                print(f"  Fix: {bug['fix']}")
    
    print("\n📋 Action Required:")
    print("  1. Review and fix the bugs listed above")
    print("  2. Re-run this test after fixes")
    print("  3. Test the bot in simulation mode before live trading")
    
else:
    print("✅ NO BUGS FOUND!")
    print("="*60)
    print("\n💡 trade_bot.py passed all checks")
    print("\nRecommendations:")
    print("  1. Test with simulation mode first")
    print("  2. Verify all config files are present")
    print("  3. Check trade_history.py is available")
