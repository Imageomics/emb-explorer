#!/usr/bin/env python3
"""
Development helper script to test installation and basic functionality.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"\n{'='*50}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=60
        )
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout[:500])  # Limit output length
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout[:500]}")
        if e.stderr:
            print(f"Stderr: {e.stderr[:500]}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return False


def main():
    """Run development tests."""
    print("🔧 Development Test Suite for emb-explorer")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Run from project root.")
        sys.exit(1)
    
    tests = [
        ("python -c 'import utils.models; print(\"Import test passed\")'", 
         "Import utils.models"),
        
        ("python list_models.py --format json | head -10", 
         "List models (JSON format)"),
        
        ("python list_models.py --format table | head -5", 
         "List models (table format)"),
        
        ("python -c 'from utils.models import list_available_models; models = list_available_models(); print(f\"Found {len(models)} models\")'", 
         "Function call test"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {desc}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The installation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
