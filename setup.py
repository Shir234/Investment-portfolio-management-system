"""
This script will:

- Check your Python version to ensure it's compatible
- Create a virtual environment to isolate your project dependencies
- Generate a requirements.txt file with all necessary libraries
- Install all the required packages
- Create basic folders for your project structure

Run the script with: python setup.py

After running the setup:
- The script will create a virtual environment named "venv" in your project directory
- Install all necessary packages including pandas, yfinance, pandas-ta, scikit-learn, etc.
- Create basic folders like "data", "models", and "results" for organizing your project


Activating the environment:
- On Windows: venv\Scripts\activate
- Activate this environment each time before working on your project
"""

#!/usr/bin/env python3
"""
Setup script for Stock Analysis Project
This script sets up a Python environment and installs required dependencies.
Run this script before executing your stock analysis code in VS Code.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def check_python_version():
    """Check if Python version is compatible."""
    print_section("Checking Python Version")
    
    python_version = sys.version_info
    required_version = (3, 7)
    
    if python_version < required_version:
        print(f"ERROR: Python {required_version[0]}.{required_version[1]} or higher required.")
        print(f"Current version: {python_version[0]}.{python_version[1]}")
        sys.exit(1)
    else:
        print(f"✓ Python version {python_version[0]}.{python_version[1]}.{python_version[2]} detected. Compatible!")

def create_virtual_env():
    """Create a virtual environment if one doesn't exist."""
    print_section("Setting Up Virtual Environment")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists at './venv/'")
        return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created successfully at './venv/'")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to create virtual environment: {e}")
        sys.exit(1)

def get_pip_path():
    """Get the pip executable path based on the OS."""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip")
    else:
        return os.path.join("venv", "bin", "pip")

def create_requirements_file():
    """Create requirements.txt with all needed dependencies."""
    print_section("Creating Requirements File")
    
    requirements_path = Path("requirements.txt")
    
    if requirements_path.exists():
        print("✓ requirements.txt already exists. Using existing file.")
        return
    
    dependencies = [
        "yfinance",
        "pandas",
        "pandas-ta",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "scikeras",
        "xgboost",
        "lightgbm",
        "ipykernel",  # For Jupyter notebook support in VS Code
        "pandas-datareader",  # Added for FRED data access
    ]
    
    with open(requirements_path, "w") as f:
        f.write("\n".join(dependencies))
    
    print("✓ Created requirements.txt with the following dependencies:")
    for dep in dependencies:
        print(f"  - {dep}")

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print_section("Installing Dependencies")
    
    pip_path = get_pip_path()
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install dependencies
        print("Installing required packages...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✓ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print("\nTry activating the virtual environment first with:")
        if platform.system() == "Windows":
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        sys.exit(1)

def setup_project_structure():
    """Create a basic project structure for the stock analysis code."""
    print_section("Setting Up Project Structure")
    
    # Define directories to create
    directories = ["data", "models", "results"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def print_next_steps():
    """Print instructions for next steps."""
    print_section("Next Steps")
    
    print("To run your stock analysis code:")
    print("\n1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Run your main Python script:")
    print("   python main.py")
    
    print("\n3. To deactivate the virtual environment when finished:")
    print("   deactivate")

def main():
    """Main function to run the setup process."""
    print_section("Stock Analysis Project Setup")
    
    check_python_version()
    create_virtual_env()
    create_requirements_file()
    
    # Ask user if they want to install dependencies now
    install_now = input("\nDo you want to install dependencies now? (y/n): ").lower()
    
    if install_now == 'y':
        # Check if virtual environment is activated
        if not os.environ.get("VIRTUAL_ENV"):
            print("\nVirtual environment is not activated.")
            print("For best results, activate the virtual environment first.")
            
            proceed = input("Do you want to proceed anyway? (y/n): ").lower()
            if proceed != 'y':
                print("\nPlease activate the virtual environment before installing dependencies:")
                if platform.system() == "Windows":
                    print("  venv\\Scripts\\activate")
                else:
                    print("  source venv/bin/activate")
                print("\nThen run this script again.")
                sys.exit(0)
        
        install_dependencies()
    
    setup_project_structure()
    print_next_steps()
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()