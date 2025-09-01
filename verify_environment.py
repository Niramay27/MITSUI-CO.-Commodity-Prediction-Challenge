#!/usr/bin/env python3
"""
Environment Verification Script for MITSUI CO. Commodity Prediction Challenge
"""

import sys
import importlib

def check_package(name, min_version=None):
    """Check if a package is installed and optionally check version."""
    try:
        module = importlib.import_module(name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠️  {name}: {version} (requires >= {min_version})")
                return False
                
        print(f"✅ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: Not installed")
        return False

def main():
    print("🔍 Environment Verification")
    print("="*50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        print(f"✅ Python: {python_version}")
    else:
        print(f"❌ Python: {python_version} (requires >= 3.8)")
        return False
    
    print("\n📦 Core Libraries:")
    core_packages = [
        ('pandas', '1.5.0'),
        ('numpy', '1.21.0'),
        ('scipy', '1.9.0'),
        ('sklearn', '1.1.0'),
    ]
    
    all_good = True
    for package, min_ver in core_packages:
        if not check_package(package, min_ver):
            all_good = False
    
    print("\n🤖 Machine Learning:")
    ml_packages = [
        ('xgboost', '1.6.0'),
        ('lightgbm', '3.3.0'),
        ('catboost', '1.1.0'),
    ]
    
    for package, min_ver in ml_packages:
        if not check_package(package, min_ver):
            all_good = False
    
    print("\n📊 Visualization:")
    viz_packages = [
        ('matplotlib', '3.5.0'),
        ('seaborn', '0.11.0'),
        ('plotly', '5.9.0'),
    ]
    
    for package, min_ver in viz_packages:
        check_package(package, min_ver)
    
    print("\n🔧 Development Tools:")
    dev_packages = [
        'jupyter',
        'tqdm',
        'joblib',
    ]
    
    for package in dev_packages:
        check_package(package)
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 Environment setup is complete and ready!")
        return True
    else:
        print("⚠️  Some critical packages are missing or outdated.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
