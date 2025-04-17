import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
from gui.main_window import MainWindow
from gui.splash_screen import SplashScreen
from data.data_manager import DataManager

def set_dark_mode(app):
    """Apply dark mode styling to the application"""
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)
    
    # Additional dark mode styles for all components
    app.setStyleSheet("""
        QWidget {
            background-color: #353535;
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background-color: #353535;
        }
        QTabBar::tab {
            background-color: #353535;
            padding: 6px 12px;
            margin: 0px 2px;
            color: white;
            border: 1px solid #444;
            border-bottom: 0px;
        }
        QTabBar::tab:selected {
            background-color: #444;
            border-bottom-color: #444;
        }
        QPushButton {
            background-color: #555;
            border: 1px solid #888;
            padding: 5px 15px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #666;
        }
        QPushButton:pressed {
            background-color: #444;
        }
        QMessageBox {
            background-color: #353535;
            color: white;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #3c3f41;
            color: white;
            border: 1px solid #555;
            padding: 3px;
        }
        QTableWidget {
            gridline-color: #555;
            color: white;
        }
        QHeaderView::section {
            background-color: #444;
            color: white;
            padding: 4px;
            border: 1px solid #666;
        }
        QStatusBar {
            background-color: #333;
            color: #ddd;
        }
    """)

def main():
    # Initialize the application
    app = QApplication(sys.argv)
    
    # Construct the absolute path to the logo file
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.JPG')
    app_icon = QIcon(logo_path)
    app.setWindowIcon(app_icon)
    
    # Apply dark mode
    set_dark_mode(app)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Construct the absolute path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '20250415_all_tickers_results.csv')
    
    # Initialize data manager with the absolute path
    data_manager = DataManager(csv_path)
    
    # Create main window but don't show it yet
    window = MainWindow(data_manager)
    window.setWindowIcon(app_icon)
    
    # Start fade transition after 4 seconds
    QTimer.singleShot(3500, lambda: splash.fade_out(window))
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()