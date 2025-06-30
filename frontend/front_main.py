import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import logging
from .logging_config import setup_logging, get_logger
from PyQt6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QToolButton, QMessageBox
from PyQt6.QtGui import QPalette, QColor, QIcon, QFont
from PyQt6.QtCore import Qt
from .gui.main_window import MainWindow
from .gui.splash_screen import SplashScreen
from .data.data_manager import DataManager
from .utils import resource_path

# Configure logging
log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')
setup_logging()
logger = get_logger(__name__)
logger.info("Logging initialized")

# Suppress matplotlib and NumExpr logs
os.environ["NUMEXPR_MAX_THREADS"] = "8"

# Add parent and backend directories to sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'backend'))

# Centralized theme colors
THEME_COLORS = {
    'dark': {
        'background': '#2D2D2D',
        'text': '#FFFFFF',
        'border': '#555555',
        'card': '#3C3F41',
        'highlight': '#0078D4',
        'hover': '#005BA1',
        'pressed': '#003E7E',
        'alternate': '#2A2A2A'
    },
    'light': {
        'background': '#F5F5F5',
        'text': '#2C2C2C',
        'border': '#CCCCCC',
        'card': '#FFFFFF',
        'highlight': '#0078D4',
        'hover': '#005BA1',
        'pressed': '#003E7E',
        'alternate': '#F0F0F0'
    }
}

def set_global_style(app, is_dark_mode=True):
    """Apply modern global styling to the application."""
    theme = THEME_COLORS['dark' if is_dark_mode else 'light']
    logger.info(f"Setting {'dark' if is_dark_mode else 'light'} mode")
    palette = QPalette()
    if is_dark_mode:
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 212))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    else:
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Button, QColor(230, 230, 230))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 212))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {theme['background']};
            color: {theme['text']};
            font-family: 'Segoe UI', sans-serif;
        }}
        QTabWidget::pane {{
            border: 1px solid {theme['border']};
            background-color: {theme['background']};
            border-radius: 8px;
        }}
        QTabBar::tab {{
            background-color: {theme['card']};
            color: {theme['text']};
            padding: 12px 24px;
            margin: 2px;
            border: 1px solid {theme['border']};
            border-radius: 6px;
        }}
        QTabBar::tab:selected {{
            background-color: {theme['highlight']};
            color: white;
            border-bottom: none;
        }}
        QPushButton {{
            background-color: {theme['highlight']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'Segoe UI';
        }}
        QPushButton:hover {{
            background-color: {theme['hover']};
        }}
        QPushButton:pressed {{
            background-color: {theme['pressed']};
        }}
        QLineEdit, QDoubleSpinBox, QDateEdit {{
            background-color: {theme['card']};
            color: {theme['text']};
            border: 1px solid {theme['border']};
            padding: 6px;
            border-radius: 4px;
        }}
        QComboBox {{
            background-color: {theme['card']};
            color: {theme['text']};
            border: 1px solid {theme['border']};
            padding: 6px;
            border-radius: 4px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QTableWidget {{
            background-color: {theme['card']};
            color: {theme['text']};
            gridline-color: {theme['border']};
            border-radius: 4px;
        }}
        QHeaderView::section {{
            background-color: {theme['card']};
            color: {theme['text']};
            padding: 6px;
            border: 1px solid {theme['border']};
        }}
        QDialog {{
            background-color: {theme['background']};
            color: {theme['text']};
        }}
        QToolTip {{
            background-color: {theme['background']};
            color: {theme['text']};
            border: 1px solid {theme['border']};
        }}
        QToolButton {{
            background-color: transparent;
            color: {theme['text']};
            font-size: 16px;
            border: 1px solid {theme['border']};
            border-radius: 20px;
            padding: 6px;
        }}
        QToolButton:hover {{
            background-color: {theme['border']};
        }}
    """)
    app.setProperty("isDarkMode", is_dark_mode)

def get_dialog_style(is_dark_mode):
    """Return stylesheet for dialogs based on the current theme."""
    theme = THEME_COLORS['dark' if is_dark_mode else 'light']
    return f"""
        QDialog {{
            background-color: {theme['background']};
            color: {theme['text']};
            font-family: 'Segoe UI';
            border: 1px solid {theme['border']};
            border-radius: 8px;
        }}
        QDialog QLabel {{
            color: {theme['text']};
            font-family: 'Segoe UI';
            font-size: 14px;
        }}
        QDialog QPushButton {{
            background-color: {theme['highlight']};
            color: #FFFFFF;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-family: 'Segoe UI';
            font-weight: bold;
        }}
        QDialog QPushButton:hover {{
            background-color: {theme['hover']};
        }}
        QDialog QPushButton:pressed {{
            background-color: {theme['pressed']};
        }}
        QDialog QTableWidget {{
            background-color: {theme['card']};
            color: {theme['text']};
            border: 1px solid {theme['border']};
            border-radius: 4px;
            gridline-color: {theme['border']};
            alternate-background-color: {theme['alternate']};
        }}
        QDialog QTableWidget::item {{
            border: none;
            padding: 8px;
        }}
        QDialog QTableWidget::item:selected {{
            background-color: {theme['highlight']};
            color: #FFFFFF;
        }}
        QDialog QHeaderView::section {{
            background-color: {theme['card']};
            color: {theme['text']};
            padding: 8px;
            border: 1px solid {theme['border']};
        }}
        QDialog QCheckBox {{
            color: {theme['text']};
        }}
    """

class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV File Help")
        self.setFixedSize(500, 350)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setup_ui()

    def setup_ui(self):
        """Configure the HelpDialog UI with modern styling."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("CSV File Requirements")
        title.setStyleSheet("font-size: 18px; font-weight: bold; font-family: 'Segoe UI'; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        instructions = QLabel(
            "To begin, select the <b>all_tickers_results.csv</b> file.\n\n"
            "<b>Required Columns:</b>\n"
            "• Date\n• Ticker\n• Close\n• Buy\n• Sell\n• Actual_Sharpe\n• Best_Prediction\n\n"
            "<b>Example Row:</b>\n"
            "2021-10-14, WBD, 25.26, -1.0, -1.0, -2.789610324596968, -1.8321122673517407\n\n"
            "A sample file is available in the 'data/' folder."
        )
        instructions.setStyleSheet("font-size: 14px; font-family: 'Segoe UI'; line-height: 22px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        close_button = QPushButton("Close")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005BA1;
            }
            QPushButton:pressed {
                background-color: #003E7E;
            }
        """)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

class StarterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to SharpSight")
        self.setFixedSize(450, 250)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setup_ui()

    def setup_ui(self):
        """Configure the StarterDialog UI with modern styling."""
        logger.info("Setting up StarterDialog")
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Welcome to SharpSight")
        title.setStyleSheet("font-size: 24px; font-weight: bold; font-family: 'Segoe UI';")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        prompt = QLabel("Select 'all_tickers_results.csv' to start your investment journey.")
        prompt.setStyleSheet("font-size: 16px; font-family: 'Segoe UI'; margin-bottom: 20px;")
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setFixedSize(40, 40)
        help_button.setToolTip("View CSV file requirements")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        select_button = QPushButton("Select CSV File")
        select_button.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 14px;
                font-family: 'Segoe UI';
                font-weight: bold;
                background-color: #0078D4;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005BA1;
            }
            QPushButton:pressed {
                background-color: #003E7E;
            }
        """)
        select_button.clicked.connect(self.accept)
        button_layout.addWidget(select_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

    def show_help(self):
        """Display the HelpDialog."""
        logger.info("Help button clicked")
        help_dialog = HelpDialog()
        help_dialog.exec()

def main():
    """Main entry point for the application."""
    logger.info("Starting main function")
    app = QApplication(sys.argv)
    logger.info("QApplication initialized")
    logo_path = resource_path('frontend/gui/icons/logo.ico')
    logger.info(f"Logo path: {logo_path}")
    if os.path.exists(logo_path):
        app_icon = QIcon(logo_path)
        app.setWindowIcon(app_icon)
        logger.info("Logo set successfully")
    else:
        logger.error(f"Logo file not found at: {logo_path}")

    set_global_style(app)
    logger.info("Global style set")
    starter = StarterDialog()
    logger.info("StarterDialog created")
    if starter.exec() != QDialog.DialogCode.Accepted:
        logger.info("User cancelled file selection")
        sys.exit(1)
    logger.info("StarterDialog accepted")
    csv_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select all_tickers_results.csv",
        "",
        "CSV Files (*.csv)",
        options=QFileDialog.Option.DontUseNativeDialog
    )
    logger.info(f"Selected CSV: {csv_path}")
    if not csv_path:
        logger.error("No file selected")
        sys.exit(1)

    data_manager = DataManager(csv_path=csv_path)
    logger.info("DataManager initialized")
    if data_manager.data is None or data_manager.data.empty:
        theme = THEME_COLORS['dark' if app.property("isDarkMode") else 'light']
        error_msg = f"Failed to load CSV: {data_manager.csv_path}"
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Error")
        msg.setText(error_msg)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {theme['background']};
                color: {theme['text']};
                font-family: 'Segoe UI';
                border: 1px solid {theme['border']};
                border-radius: 4px;
            }}
            QMessageBox QLabel {{
                color: {theme['text']};
            }}
            QMessageBox QPushButton {{
                background-color: {theme['highlight']};
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {theme['hover']};
            }}
            QMessageBox QPushButton:pressed {{
                background-color: {theme['pressed']};
            }}
        """)
        msg.exec()
        logger.error(error_msg)
        sys.exit(1)

    splash = SplashScreen()
    logger.info("SplashScreen created")
    splash.show()
    window = MainWindow(data_manager)
    logger.info("MainWindow created")
    splash.loading_completed.connect(lambda: splash.finish(window))
    logger.info("Application fully loaded")
    sys.exit(app.exec())

if __name__ == '__main__':
    main()