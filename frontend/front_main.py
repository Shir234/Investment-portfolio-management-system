import sys
import os
import logging
import pandas as pd
from logging_config import setup_logging, get_logger
# Configure logging with FileHandler
log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'logs')
os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
log_file = os.path.join(log_dir, 'app.log')
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Logging initialized")

# Suppress matplotlib and NumExpr logs
os.environ["NUMEXPR_MAX_THREADS"] = "8"
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Add parent and backend directories to sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'backend'))

from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QToolButton, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow
from gui.splash_screen import SplashScreen
from data.data_manager import DataManager

def resource_path(relative_path):
    """Get absolute path to resources, works for dev and PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

def set_dark_mode(app):
    """Apply dark mode styling to the application."""
    logger.info("Setting dark mode")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.black)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    app.setStyleSheet("""
        QWidget { background-color: #353535; color: #ffffff; }
        QTabWidget::pane { border: 1px solid #444; background-color: #353535; }
        QTabBar::tab { background-color: #353535; padding: 8px 12px; margin: 0px 2px; color: white; border: 1px solid #444; border-bottom: none; }
        QTabBar::tab:selected { background-color: #444; border-bottom-color: #444; }
        QPushButton { background-color: #555; border: 1px solid #888; padding: 5px 15px; border-radius: 4px; }
        QPushButton:hover { background-color: #666; }
        QPushButton:pressed { background-color: #444; }
        QToolButton { background-color: transparent; color: #ffffff; font-size: 16px; }
        QToolButton:hover { background-color: #444444; }
        QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #3c3f41; color: white; border: 1px solid #555; padding: 3px; }
        QTableWidget { gridline-color: #555; color: white; }
        QHeaderView::section { background-color: #444; color: white; padding: 4px; border: 1px solid #666; }
        QStatusBar { background-color: #333; color: #ddd; }
        QDialog { background-color: #353535; color: #ffffff; }
        QToolTip { background-color: #353535; color: #ffffff; border: 1px solid #555; }
    """)

class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV File Help")
        self.setFixedSize(450, 300)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("CSV File Requirements")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        instructions = QLabel(
            "Please select the <b>all_tickers_results.csv</b> file to begin.\n\n"
            "<b>Required Columns</b>:\n"
            "Date, Ticker, Close, Buy, Sell, Actual_Sharpe, Best_Prediction\n\n"
            "<b>Example Row</b>:\n"
            "2021-10-14,WBD,25.26,-1.0,-1.0,-2.789610324596968,-1.8321122673517407\n\n"
            "A sample file is available in the 'data/' folder."
        )
        instructions.setStyleSheet("color: #ffffff; font-size: 12px;")
        instructions.setAlignment(Qt.AlignLeft)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        close_button = QPushButton("Close")
        close_button.setStyleSheet("background-color: #2a82da; color: #ffffff; padding: 8px;")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

class StarterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to SharpSight")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setup_ui()

    def setup_ui(self):
        logger.info("Setting up StarterDialog")
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Welcome to SharpSight Investment System")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        prompt = QLabel("Please select 'all_tickers_results.csv' to begin.")
        prompt.setStyleSheet("color: #ffffff; font-size: 12px;")
        prompt.setAlignment(Qt.AlignCenter)
        layout.addWidget(prompt)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setFixedSize(24, 24)
        help_button.setStyleSheet("background-color: #2a82da; color: #ffffff; border-radius: 12px;")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        select_button = QPushButton("Select CSV File")
        select_button.setStyleSheet("background-color: #2a82da; color: #ffffff; padding: 8px;")
        select_button.clicked.connect(self.accept)
        button_layout.addWidget(select_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

    def show_help(self):
        logger.info("Help button clicked")
        help_dialog = HelpDialog()
        help_dialog.exec_()

def main():
    logger.info("Starting main function")
    app = QApplication(sys.argv)
    logger.info("QApplication initialized")
    logo_path = resource_path('logo.ico')
    logger.info(f"Logo path: {logo_path}")
    if os.path.exists(logo_path):
        app_icon = QIcon(logo_path)
        app.setWindowIcon(app_icon)
        logger.info("Logo set successfully")
    else:
        logger.error(f"Logo file not found at: {logo_path}")

    set_dark_mode(app)
    logger.info("Dark mode set")
    starter = StarterDialog()
    logger.info("StarterDialog created")
    if starter.exec_() != QDialog.Accepted:
        logger.info("User cancelled file selection")
        sys.exit(1)
    logger.info("StarterDialog accepted")
    csv_path, _ = QFileDialog.getOpenFileName(
        None, 
        "Select all_tickers_results.csv", 
        "", 
        "CSV Files (*.csv)", 
        options=QFileDialog.DontUseNativeDialog
    )
    logger.info(f"Selected CSV: {csv_path}")
    if not csv_path:
        logger.error("No file selected")
        sys.exit(1)

    data_manager = DataManager(csv_path=csv_path)
    logger.info("DataManager initialized")
    success, error_msg = data_manager.load_data(csv_path)
    logger.info(f"Data loading result: success={success}, error={error_msg}")
    if not success:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(f"Failed to load CSV: {error_msg}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox { background-color: #353535; color: white; }
            QMessageBox QLabel { color: white; }
            QMessageBox QPushButton { background-color: #444444; color: white; }
        """)
        msg.exec_()
        logger.error(f"CSV loading failed: {error_msg}")
        sys.exit(1)

    splash = SplashScreen()
    logger.info("SplashScreen created")
    splash.show()
    window = MainWindow(data_manager)
    logger.info("MainWindow created")
    splash.finish(window)
    logger.info("Application fully loaded")
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()