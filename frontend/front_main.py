import sys
import os
import signal
import logging
import pandas as pd
from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect, QFileDialog, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer

# Import the logging configuration FIRST
from logging_config import setup_logging, cleanup_logging, force_log_flush, get_logger

# Suppress matplotlib and NumExpr logs
os.environ["NUMEXPR_MAX_THREADS"] = "8"

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui.main_window import MainWindow
from gui.splash_screen import SplashScreen
from data.data_manager import DataManager

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    force_log_flush()
    cleanup_logging()
    QApplication.instance().quit()

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
    """Main application entry point with proper logging setup."""
    
    # Setup centralized logging FIRST
    setup_logging(logging.INFO)
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("STARTING INVESTMENT PORTFOLIO MANAGEMENT SYSTEM")
    logger.info("="*60)
    
    # Log file locations for user reference
    from datetime import datetime
    today = datetime.now().strftime('%Y%m%d')
    print(f"\n Log files being created:")
    print(f"   Main App Log: logs/app_{today}.log")
    print(f"   Trading Only Log: logs/trading_only_{today}.log") 
    print(f"   Pipeline Log: logs/pipeline_{today}.log")
    print(f"   All logs will be flushed when you close the app\n")
    
    # Initialize the application
    app = QApplication(sys.argv)
    app.setApplicationName("Investment Portfolio Management System")
    app.setApplicationVersion("1.0")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Construct the absolute path to the logo file
        logo_path = os.path.join(os.path.dirname(__file__), 'logo.JPG')
        
        if os.path.exists(logo_path):
            app_icon = QIcon(logo_path)
            app.setWindowIcon(app_icon)
            logger.info(f"Logo loaded from: {logo_path}")
        else:
            logger.warning(f"Logo file not found at: {logo_path}")
        
        # Apply dark mode
        set_dark_mode(app)
        logger.info("Dark mode styling applied")
        
        # Prompt user to select the CSV file
        logger.info("Prompting user for CSV file selection...")
        csv_path, _ = QFileDialog.getOpenFileName(
            None, 
            "Select CSV File with Trading Data", 
            "", 
            "CSV Files (*.csv)"
        )
        
        if not csv_path:
            logger.error("No file selected by user")
            QMessageBox.critical(None, "Error", "No file selected. Application will exit.")
            return 1
        
        logger.info(f"User selected CSV file: {csv_path}")
        
        # Validate file exists
        if not os.path.exists(csv_path):
            logger.error(f"Selected file does not exist: {csv_path}")
            QMessageBox.critical(None, "Error", f"Selected file does not exist: {csv_path}")
            return 1
        
        # Load data with the selected file
        logger.info("Initializing data manager...")
        try:
            data_manager = DataManager(csv_path=csv_path)
            
            if data_manager.data is None or data_manager.data.empty:
                logger.error("Failed to load data from selected file")
                QMessageBox.critical(None, "Error", "Failed to load data from selected file")
                return 1
            
            logger.info(f"Data loaded successfully: {len(data_manager.data)} rows, {len(data_manager.data.columns)} columns")
            logger.info(f"Date range: {data_manager.dataset_start_date} to {data_manager.dataset_end_date}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            QMessageBox.critical(None, "Error", f"Error loading data: {e}")
            return 1
        
        # Show splash screen
        logger.info("Displaying splash screen...")
        splash = SplashScreen()
        splash.show()
        
        # Process events to show splash screen
        app.processEvents()
        
        # Create main window
        logger.info("Creating main window...")
        window = MainWindow(data_manager)
        
        # Setup timer to close splash screen and show main window
        def show_main_window():
            logger.info("Transitioning from splash to main window")
            try:
                splash.finish(window)
                window.show()
                logger.info("Main window displayed successfully")
            except Exception as e:
                logger.error(f"Error showing main window: {e}", exc_info=True)
        
        # Wait 3 seconds then show main window
        QTimer.singleShot(3000, show_main_window)
        
        logger.info("Starting Qt event loop...")
        
        # Execute the application
        result = app.exec_()
        
        logger.info(f"Qt event loop finished with code: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Critical error in main application: {e}", exc_info=True)
        QMessageBox.critical(None, "Critical Error", f"Application failed to start: {e}")
        return 1
    
    finally:
        logger.info("Application shutdown - performing cleanup...")
        try:
            # Force flush all logs before cleanup
            force_log_flush()
            logger.info("Final log flush completed")
        except Exception as e:
            print(f"Error during log flush: {e}")
        
        try:
            # Clean up logging
            cleanup_logging()
        except Exception as e:
            print(f"Error during logging cleanup: {e}")
        
        print("Application closed successfully.")

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)