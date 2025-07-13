"""
This script will:

- Auto-detect the best Python installation to use
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


def configure_tensorflow_settings():
    """Create a script to configure TensorFlow environment settings."""
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
    
    print(" TensorFlow configuration scripts created")
    print("  - scripts/configure_tensorflow.bat (Windows Command Prompt)")
    print("  - scripts/configure_tensorflow.ps1 (Windows PowerShell)")
    print("  - scripts/configure_tensorflow.sh (Unix/Linux/MacOS)")
    
    # Create a wrapper file to run from Python
    create_tf_config_module()

def create_tf_config_module():
    """Create a Python module to configure TensorFlow in code."""
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
        "optuna",  # Added for hyperparameter optimization
        "optuna-integration[tfkeras]",  # Added for TensorFlow/Keras integration with Optuna
        "ipykernel",  # For Jupyter notebook support in VS Code
        "pandas-datareader",  # Added for FRED data access
        "pydrive",  # Added for Google Drive integration
        "pytest",  # Added for testing framework
        "pytest-qt",  # Added for Qt testing with pytest
        "pyinstaller",  # Added for creating executable files
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
    
    print("\n2. Configure TensorFlow environment (optional, to suppress warnings):")
    if platform.system() == "Windows":
        print("   PowerShell: . .\\scripts\\configure_tensorflow.ps1")
        print("   Command Prompt: .\\scripts\\configure_tensorflow.bat")
    else:
        print("   source ./scripts/configure_tensorflow.sh")
    
    print("\n   Alternatively, use the Python module by adding this to your code:")
    print("   import tf_config  # Import before TensorFlow")
    
    print("\n3. Run your main Python script:")
    print("   python main.py")
    
    print("\n4. To deactivate the virtual environment when finished:")
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
    configure_tensorflow_settings()
    print_next_steps()
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()