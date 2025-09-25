#!/usr/bin/env python3
"""
Verify that the fixes work by running the problematic scripts
"""

import subprocess
import sys
from pathlib import Path

def test_script(script_name, timeout=60):
    """Test a single script"""
    print(f"\n🧪 Testing {script_name}...")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd='.',
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[-500:]}")  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out")
        return False
    except Exception as e:
        print(f"💥 Error running {script_name}: {e}")
        return False

def check_outputs():
    """Check if expected outputs were created"""
    print(f"\n📋 Checking outputs...")
    
    expected_files = [
        'output/model_features_train.csv',
        'output/model_features_validation.csv'
    ]
    
    all_exist = True
    for filepath in expected_files:
        path = Path(filepath)
        if path.exists():
            print(f"✓ {filepath}")
        else:
            print(f"❌ {filepath} missing")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("🔧 Verifying fixes for run_validation.py errors...")
    
    # Test the problematic scripts
    scripts_to_test = [
        '03_feature_engineering.py',
        '04_train_credit_model.py'
    ]
    
    all_passed = True
    
    for script in scripts_to_test:
        success = test_script(script)
        if not success:
            all_passed = False
    
    # Check if outputs were created
    if all_passed:
        outputs_ok = check_outputs()
        all_passed = outputs_ok
    
    if all_passed:
        print(f"\n🎉 All tests passed! The fixes should resolve the run_validation.py errors.")
    else:
        print(f"\n💥 Some tests failed. Additional fixes may be needed.")
