"""
This script will:

- Auto-detect the best Python installation to use
- Check the Python version to ensure it's compatible
- Create a virtual environment to isolate the project dependencies
- Generate a requirements.txt file with all necessary libraries
- Install all the required packages
- Create basic folders for the project structure

Run the script with: python setup.py

After running the setup:
- The script will create a virtual environment named "venv" in the project directory
- Install all necessary packages including pandas, yfinance, pandas-ta, scikit-learn, etc.
- Create basic folders like "data", "models", and "results" for organizing your project

Activating the environment:
- On Windows: venv\Scripts\activate
- Activate this environment each time before working on the project

"""

#!/usr/bin/env python3
"""
Setup script for Stock Analysis Project
This script sets up a Python environment and installs required dependencies.
Automatically detects the best Python installation to use.

"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def print_section(title):
    """
    Print a formatted section title.
    """

    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")


def find_best_python():
    """
    Find the best Python installation to use.
    """

    print_section("Detecting Python Installation")
    
    # Common Windows Python locations
    common_windows_paths = [
        # Python.org installations
        f"C:\\Users\\{os.getenv('USERNAME', 'user')}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        f"C:\\Users\\{os.getenv('USERNAME', 'user')}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        f"C:\\Users\\{os.getenv('USERNAME', 'user')}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
        f"C:\\Users\\{os.getenv('USERNAME', 'user')}\\AppData\\Local\\Programs\\Python\\Python39\\python.exe",
        # System-wide installations
        "C:\\Python311\\python.exe",
        "C:\\Python312\\python.exe",
        "C:\\Python310\\python.exe",
        "C:\\Python39\\python.exe",
    ]
    
    python_candidates = []
    
    # Always add the current Python as a candidate
    current_python = sys.executable
    python_candidates.append(("Current Python", current_python))
    
    # Add Windows Python installations if on Windows
    if platform.system() == "Windows":
        for path in common_windows_paths:
            if os.path.exists(path):
                python_candidates.append(("Windows Python", path))
    
    # Test each candidate
    best_python = None
    best_info = None
    
    print("Found Python installations:")
    for name, path in python_candidates:
        try:
            # Test if Python works and get version
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, text=True, check=True, timeout=10)
            version_str = result.stdout.strip()
            
            # Test if it can create virtual environments
            test_result = subprocess.run([path, "-m", "venv", "--help"], 
                                       capture_output=True, text=True, timeout=10)
            
            if test_result.returncode == 0:
                # Extract version for comparison
                version_parts = version_str.split()[1].split('.')
                major, minor = int(version_parts[0]), int(version_parts[1])
                
                print(f"  ✓ {name}: {version_str} at {path}")
                
                # Prefer Windows Python installations over MSYS2/others
                if best_python is None or ("Windows Python" in name and "Windows Python" not in best_info[0]):
                    best_python = path
                    best_info = (name, version_str, major, minor)
            else:
                print(f"  ✗ {name}: {version_str} (can't create venv)")
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            print(f"  ✗ {name}: Not working or invalid")
    
    if best_python is None:
        print("ERROR: No working Python installation found!")
        sys.exit(1)
    
    name, version_str, major, minor = best_info
    
    # Check version compatibility
    if (major, minor) < (3, 7):
        print(f"ERROR: Python 3.7 or higher required. Best found: {major}.{minor}")
        sys.exit(1)
    
    print(f"\n✓ Selected: {name} - {version_str}")
    print(f"  Path: {best_python}")
    
    return best_python


def configure_tensorflow_settings():
    """
    Create a script to configure TensorFlow environment settings.
    """

    print_section("Creating TensorFlow Configuration Script")
    
    # Create a directory for scripts if it doesn't exist
    scripts_dir = Path("scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Create batch file for Windows
    tf_config_bat = scripts_dir / "configure_tensorflow.bat"
    with open(tf_config_bat, "w") as f:
        f.write("@echo off\n")
        f.write("echo Setting TensorFlow options...\n")
        f.write("set TF_ENABLE_ONEDNN_OPTS=0\n")
        f.write("echo TensorFlow environment variables have been set for this session.\n")
    
    # Create PowerShell script for Windows
    tf_config_ps1 = scripts_dir / "configure_tensorflow.ps1"
    with open(tf_config_ps1, "w") as f:
        f.write("Write-Host \"Setting TensorFlow options...\"\n")
        f.write("$env:TF_ENABLE_ONEDNN_OPTS=0\n")
        f.write("Write-Host \"TensorFlow environment variables have been set for this session.\"\n")
    
    # Create shell script for Unix/Linux/MacOS
    tf_config_sh = scripts_dir / "configure_tensorflow.sh"
    with open(tf_config_sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo Setting TensorFlow options...\n")
        f.write("export TF_ENABLE_ONEDNN_OPTS=0\n")
        f.write("echo TensorFlow environment variables have been set for this session.\n")
    
    # Make the shell script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(tf_config_sh, 0o755)
    
    print("✓ TensorFlow configuration scripts created")
    print("  - scripts/configure_tensorflow.bat (Windows Command Prompt)")
    print("  - scripts/configure_tensorflow.ps1 (Windows PowerShell)")
    print("  - scripts/configure_tensorflow.sh (Unix/Linux/MacOS)")
    
    # Create a wrapper file to run from Python
    create_tf_config_module()


def create_tf_config_module():
    """
    Create a Python module to configure TensorFlow in code.
    """

    tf_config_py = Path("tf_config.py")
    with open(tf_config_py, "w") as f:
        f.write("""# tf_config.py
    # Configure TensorFlow environment variables
    import os

    def configure():
        \"\"\"Configure TensorFlow environment variables.\"\"\"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        print("TensorFlow environment variables configured programmatically.")

    # Configure when this module is imported
    configure()
    """)
    
    print("✓ Created tf_config.py module")
    print("  - Import this module before TensorFlow to set environment variables programmatically")


# def check_python_version():
#     """Check if Python version is compatible."""
#     print_section("Checking Python Version")
    
#     python_version = sys.version_info
#     required_version = (3, 7)
    
#     if python_version < required_version:
#         print(f"ERROR: Python {required_version[0]}.{required_version[1]} or higher required.")
#         print(f"Current version: {python_version[0]}.{python_version[1]}")
#         sys.exit(1)
#     else:
#         print(f"✓ Python version {python_version[0]}.{python_version[1]}.{python_version[2]} detected. Compatible!")



def create_virtual_env(python_executable):
    """
    Create a virtual environment if one doesn't exist.
    """

    print_section("Setting Up Virtual Environment")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Existing virtual environment found. Removing it to create a fresh one...")
        shutil.rmtree(venv_path)
    
    try:
        print(f"Creating virtual environment using: {python_executable}")
        subprocess.run([python_executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created successfully at './venv/'")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to create virtual environment: {e}")
        sys.exit(1)


def get_venv_python():
    """
    Get the Python executable path in the virtual environment.
    """

    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")


def create_requirements_file():
    """
    Create requirements.txt with all needed dependencies.
    """

    print_section("Creating Requirements File")
    
    requirements_path = Path("requirements.txt")
    
    if requirements_path.exists():
        print("✓ requirements.txt already exists. Using existing file.")
        return
    
    dependencies = [
        # Core data science libraries
        "pandas",
        "numpy",
        "scipy",
        
        # Financial data libraries
        "yfinance",
        "pandas-ta",
        "pandas-datareader",  # For FRED data access
        
        # Machine learning libraries
        "scikit-learn",
        "tensorflow",
        "scikeras",
        "xgboost",
        "lightgbm",
        "optuna",  # For hyperparameter optimization
        "joblib",  # For model persistence
        
        # Visualization libraries
        "matplotlib",
        "seaborn",
        "plotly",  # Interactive plotting
        
        # GUI framework
        "PyQt5",
        
        # Development tools
        "ipykernel",  # For Jupyter notebook support in VS Code
    ]
    
    with open(requirements_path, "w") as f:
        f.write("\n".join(dependencies))
    
    print("✓ Created requirements.txt with the following dependencies:")
    for dep in dependencies:
        print(f"  - {dep}")


# def get_pip_path():
#     """Get the pip executable path based on the OS."""
#     if platform.system() == "Windows":
#         return os.path.join("venv", "Scripts", "pip")
#     else:
#         return os.path.join("venv", "bin", "pip")


def install_dependencies():
    """
    Install dependencies using the virtual environment Python.
    """
    
    print_section("Installing Dependencies")
    
    venv_python = get_venv_python()
    
    if not os.path.exists(venv_python):
        print(f"ERROR: Virtual environment Python not found at {venv_python}")
        sys.exit(1)
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install dependencies in groups for better error handling
        dependency_groups = [
            ("Core Data Science", ["pandas", "numpy", "scipy"]),
            ("Financial Data", ["yfinance", "pandas-datareader", "pandas-ta"]),
            ("Machine Learning", ["scikit-learn", "xgboost", "lightgbm", "optuna", "joblib"]),
            ("Deep Learning", ["tensorflow", "scikeras"]),
            ("Visualization", ["matplotlib", "seaborn", "plotly"]),
            ("GUI Framework", ["PyQt5"]),
            ("Development Tools", ["ipykernel"])
        ]
        
        for group_name, packages in dependency_groups:
            print(f"\nInstalling {group_name}...")
            subprocess.run([venv_python, "-m", "pip", "install"] + packages, check=True)
            print(f"✓ {group_name} installed successfully!")
        
        print("\n✓ All dependencies installed successfully!")
        
        # Test key installations
        print("\nTesting key installations...")
        test_imports = [
            ("pandas", "import pandas; print(f'✓ Pandas {pandas.__version__} works!')"),
            ("numpy", "import numpy; print(f'✓ NumPy {numpy.__version__} works!')"),
            ("scikit-learn", "import sklearn; print(f'✓ Scikit-learn {sklearn.__version__} works!')"),
            ("matplotlib", "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__} works!')"),
            ("PyQt5", "from PyQt5.QtWidgets import QApplication; print('✓ PyQt5 GUI framework works!')"),
            ("tensorflow", "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} works!')"),
            ("xgboost", "import xgboost as xgb; print(f'✓ XGBoost {xgb.__version__} works!')")
        ]
        
        for name, test_code in test_imports:
            try:
                subprocess.run([venv_python, "-c", test_code], check=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(f"⚠ Warning: {name} installation test failed")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print("\nYou can try installing manually after activating the virtual environment:")
        if platform.system() == "Windows":
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        print("  python -m pip install -r requirements.txt")
        sys.exit(1)


def setup_project_structure():
    """
    Create a basic project structure for the stock analysis code.
    """

    print_section("Setting Up Project Structure")
    
    # Define directories to create
    directories = ["data", "models", "results", "scripts"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")


def print_next_steps():
    """
    Print instructions for next steps.
    """

    print_section("Next Steps")
    
    print("Development environment is ready! To start working:")
    print("\n1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Configure TensorFlow environment (optional, to suppress warnings):")
    if platform.system() == "Windows":
        print("   PowerShell: . .\\scripts\\configure_tensorflow.ps1")
        print("   Command Prompt: .\\scripts\\configure_tensorflow.bat")
    else:
        print("   source ./scripts/configure_tensorflow.sh")
    
    print("\n   Alternatively, use the Python module by adding this to your code:")
    print("   import tf_config  # Import before TensorFlow")
    
    print("\n3. Run main Python scripts:")
    print("   python ticker_combiner.py")
    print("   python tests_on_one_ticker.py")
    print("   # or any other script in your project")
    
    print("\n4. To deactivate the virtual environment when finished:")
    print("   deactivate")
    
    print("\n5. Remember to activate the virtual environment each time working on this project!")


def main():
    """
    Main function to run the setup process.
    """

    print_section("Stock Analysis Project Setup")
    
    # Find the best Python to use
    python_executable = find_best_python()
    
    # Create virtual environment with the selected Python
    create_virtual_env(python_executable)
    
    # Create requirements file
    create_requirements_file()
    
    # Ask user if they want to install dependencies now
    install_now = input("\nDo you want to install dependencies now? (y/n): ").lower()
    
    if install_now == 'y':
        install_dependencies()
    else:
        print("\nSkipping dependency installation.")
        print("You can install them later by running:")
        print("1. Activate the virtual environment")
        print("2. Run: python -m pip install -r requirements.txt")
    
    setup_project_structure()
    configure_tensorflow_settings()
    print_next_steps()
    
    print("\n" + "="*80)
    print("✓ Setup completed successfully!")
    print("✓ The project is ready for development across different machines!")
    print("="*80)


if __name__ == "__main__":
    main()