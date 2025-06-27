# tests/test_imports.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"sys.path: {sys.path}")

# Verify file paths
logging_config_path = os.path.join(project_root, 'frontend', 'logging_config.py')
data_manager_path = os.path.join(project_root, 'frontend', 'data', 'data_manager.py')
main_window_path = os.path.join(project_root, 'frontend', 'gui', 'main_window.py')
analysis_dashboard_path = os.path.join(project_root, 'frontend', 'gui', 'analysis_dashboard.py')

print(f"logging_config.py exists: {os.path.exists(logging_config_path)}")
print(f"data_manager.py exists: {os.path.exists(data_manager_path)}")
print(f"main_window.py exists: {os.path.exists(main_window_path)}")
print(f"analysis_dashboard.py exists: {os.path.exists(analysis_dashboard_path)}")

try:
    import frontend
    print(f"frontend module: {frontend.__file__}")
except ImportError as e:
    print(f"Failed to import frontend: {e}")

try:
    from frontend.data.data_manager import DataManager
    print("Imported DataManager successfully")
except ImportError as e:
    print(f"Failed to import DataManager: {e}")

try:
    from frontend.gui.main_window import MainWindow
    print("Imported MainWindow successfully")
except ImportError as e:
    print(f"Failed to import MainWindow: {e}")

try:
    from frontend.logging_config import get_logger
    print("Imported get_logger successfully")
except ImportError as e:
    print(f"Failed to import get_logger: {e}")

try:
    from frontend.gui.analysis_dashboard import AnalysisDashboard
    print("Imported AnalysisDashboard successfully")
except ImportError as e:
    print(f"Failed to import AnalysisDashboard: {e}")