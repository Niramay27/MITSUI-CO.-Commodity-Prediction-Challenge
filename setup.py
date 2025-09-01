#!/usr/bin/env python3
"""
MITSUI CO. Commodity Prediction Challenge - Environment Setup Script
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required. Current version:", f"{version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def verify_installations():
    """Verify that key packages are installed correctly."""
    key_packages = [
        "pandas", "numpy", "scikit-learn", "matplotlib", 
        "seaborn", "xgboost", "lightgbm", "jupyter"
    ]
    
    failed_packages = []
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)
    
    return len(failed_packages) == 0

def create_jupyter_kernel():
    """Create a Jupyter kernel for this project."""
    try:
        kernel_name = "mitsui-commodity"
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", 
            "--user", "--name", kernel_name, 
            "--display-name", "MITSUI Commodity Prediction"
        ])
        print(f"✅ Jupyter kernel '{kernel_name}' created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not create Jupyter kernel: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 MITSUI CO. Commodity Prediction Challenge - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    print("\n🔍 Verifying installations...")
    if not verify_installations():
        print("\n⚠️  Some packages failed to install. Please check the errors above.")
        sys.exit(1)
    
    # Create Jupyter kernel
    create_jupyter_kernel()
    
    print("\n" + "=" * 60)
    print("🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: jupyter notebook")
    print("2. Open eda.ipynb")
    print("3. Select kernel: 'MITSUI Commodity Prediction'")
    print("4. Start exploring the data!")

if __name__ == "__main__":
    main()
