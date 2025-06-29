import sys
import os
import logging
from logging_config import setup_logging, get_logger

# Configure logging with FileHandler
log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'logs')
os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
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

from PyQt6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QToolButton, QMessageBox
from PyQt6.QtGui import QPalette, QColor, QIcon
from PyQt6.QtCore import Qt

# Import our modules
try:
    from frontend.gui.main_window import MainWindow
    from frontend.gui.splash_screen import SplashScreen
    from frontend.data.data_manager import DataManager
    from frontend.gui.styles import ModernStyles
except ImportError as e:
    logger.error(f"Import error: {e}")
    print(f"Import error: {e}")
    sys.exit(1)

def resource_path(relative_path):
    """Get absolute path to resources, works for dev and PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

def set_modern_theme(app):
    """Apply modern theme styling to the application."""
    logger.info("Setting modern theme")
    
    # Set the complete modern style (default to dark mode)
    style = ModernStyles.get_complete_style(is_dark=True)
    app.setStyleSheet(style)
    
    # Set modern palette for better theme consistency
    colors = ModernStyles.COLORS['dark']
    palette = QPalette()
    
    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(colors['primary']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors['text_primary']))
    
    # Base colors (for input fields, etc.)
    palette.setColor(QPalette.ColorRole.Base, QColor(colors['surface']))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors['secondary']))
    
    # Text colors
    palette.setColor(QPalette.ColorRole.Text, QColor(colors['text_primary']))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(colors['text_primary']))
    
    # Button colors
    palette.setColor(QPalette.ColorRole.Button, QColor(colors['surface']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors['text_primary']))
    
    # Tooltip colors
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors['surface']))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors['text_primary']))
    
    # Selection and highlight colors
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors['accent']))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#FFFFFF'))
    
    # Link colors
    palette.setColor(QPalette.ColorRole.Link, QColor(colors['accent']))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(colors['accent_hover']))
    
    # Special colors
    palette.setColor(QPalette.ColorRole.Light, QColor(colors['border_light']))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(colors['border']))
    palette.setColor(QPalette.ColorRole.Dark, QColor(colors['text_muted']))
    palette.setColor(QPalette.ColorRole.Mid, QColor(colors['text_secondary']))
    palette.setColor(QPalette.ColorRole.Shadow, QColor('#000000'))
    
    app.setPalette(palette)

class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV File Help")
        self.setFixedSize(500, 350)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("CSV File Requirements")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        instructions = QLabel(
            "Please select the <b>all_tickers_results.csv</b> file to begin.\n\n"
            "<b>Required Columns</b>:\n"
            "Date, Ticker, Close, Buy, Sell, Actual_Sharpe, Best_Prediction\n\n"
            "<b>Example Row</b>:\n"
            "2021-10-14,WBD,25.26,-1.0,-1.0,-2.789610324596968,-1.8321122673517407\n\n"
            "A sample file is available in the 'data/' folder."
        )
        instructions.setProperty("class", "subtitle")
        instructions.setAlignment(Qt.AlignmentFlag.AlignLeft)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        # Apply modern styling
        try:
            style = ModernStyles.get_complete_style(is_dark=True)
            self.setStyleSheet(style)
        except Exception as e:
            logger.warning(f"Could not apply help dialog styling: {e}")
            pass  # Fallback to default styling

        # Center the dialog
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

class StarterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to SharpSight")
        self.setFixedSize(520, 300)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setup_ui()

    def setup_ui(self):
        logger.info("Setting up StarterDialog")
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Welcome to SharpSight")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Investment Portfolio\nManagement System")
        subtitle.setProperty("class", "subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        prompt = QLabel("Please select 'all_tickers_results.csv' to begin.")
        prompt.setProperty("class", "caption")
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setFixedSize(32, 32)
        help_button.setProperty("class", "secondary")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        select_button = QPushButton("Select CSV File")
        select_button.clicked.connect(self.accept)
        button_layout.addWidget(select_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Apply modern styling
        try:
            colors = ModernStyles.COLORS['dark']
            style = ModernStyles.get_complete_style(is_dark=True)
            
            # Add specific styling for this dialog
            dialog_style = f"""
                QDialog {{
                    background-color: {colors['primary']};
                    border: 1px solid {colors['border']};
                    border-radius: 12px;
                }}
                QLabel[class="title"] {{
                    color: {colors['text_primary']};
                    font-size: 24px;
                    font-weight: 700;
                    margin: 8px 0;
                }}
                QLabel[class="subtitle"] {{
                    color: {colors['text_secondary']};
                    font-size: 14px;
                    font-weight: 500;
                    margin: 5px 0;
                    line-height: 1.4;
                }}
                QLabel[class="caption"] {{
                    color: {colors['text_secondary']};
                    font-size: 12px;
                    font-weight: 400;
                    margin: 10px 0;
                }}
            """
            
            self.setStyleSheet(style + dialog_style)
        except Exception as e:
            logger.warning(f"Could not apply styling: {e}")
            pass  # Fallback to default styling

        # Center the dialog
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

    def show_help(self):
        logger.info("Help button clicked")
        help_dialog = HelpDialog()
        help_dialog.exec()

def show_modern_message_box(icon, title, text, buttons=QMessageBox.StandardButton.Ok):
    """Show a modern styled message box."""
    msg = QMessageBox()
    msg.setIcon(icon)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    
    # Apply modern styling if available
    try:
        colors = ModernStyles.COLORS['dark']  # Default to dark for startup dialogs
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                font-size: 14px;
                padding: 20px;
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QMessageBox QLabel {{
                color: {colors['text_primary']};
                padding: 16px;
                font-size: 14px;
                background-color: transparent;
            }}
            QMessageBox QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                min-width: 100px;
                margin: 4px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {colors['accent_hover']};
            }}
            QMessageBox QPushButton:pressed {{
                background-color: {colors['accent_pressed']};
            }}
        """)
    except Exception as e:
        logger.warning(f"Could not apply message box styling: {e}")
        pass  # Use default styling if ModernStyles not available
    
    return msg.exec()

def main():
    logger.info("Starting main function")
    app = QApplication(sys.argv)
    logger.info("QApplication initialized")
    
    # Set application properties
    app.setApplicationName("SharpSight Investment System")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("SharpSight")
    
    # Set up logo
    logo_path = resource_path('logo.ico')
    logger.info(f"Logo path: {logo_path}")
    if os.path.exists(logo_path):
        app_icon = QIcon(logo_path)
        app.setWindowIcon(app_icon)
        logger.info("Logo set successfully")
    else:
        logger.warning(f"Logo file not found at: {logo_path}")

    # Apply modern theme
    try:
        set_modern_theme(app)
        logger.info("Modern theme applied")
    except Exception as e:
        logger.warning(f"Could not apply modern theme: {e}")
    
    # Show starter dialog
    starter = StarterDialog()
    logger.info("StarterDialog created")
    if starter.exec() != QDialog.DialogCode.Accepted:
        logger.info("User cancelled file selection")
        sys.exit(1)
    logger.info("StarterDialog accepted")
    
    # File selection dialog
    csv_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select all_tickers_results.csv",
        "",
        "CSV Files (*.csv);;All Files (*)",
        options=QFileDialog.Option.DontUseNativeDialog
    )
    logger.info(f"Selected CSV: {csv_path}")
    
    if not csv_path:
        logger.error("No file selected")
        sys.exit(1)

    # Initialize data manager
    try:
        data_manager = DataManager(csv_path=csv_path)
        logger.info("DataManager initialized")
    except Exception as e:
        error_msg = f"Failed to initialize DataManager: {e}"
        logger.error(error_msg)
        show_modern_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            error_msg,
            QMessageBox.StandardButton.Ok
        )
        sys.exit(1)
    
    if data_manager.data is None or data_manager.data.empty:
        error_msg = f"Failed to load CSV: {data_manager.csv_path}"
        show_modern_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            error_msg,
            QMessageBox.StandardButton.Ok
        )
        logger.error(error_msg)
        sys.exit(1)

    # Show splash screen
    try:
        splash = SplashScreen()
        logger.info("SplashScreen created")
        splash.show()
    except Exception as e:
        logger.warning(f"Could not create splash screen: {e}")
        splash = None
    
    # Create main window
    try:
        window = MainWindow(data_manager)
        logger.info("MainWindow created")
        
        # Finish splash and show main window
        if splash:
            splash.finish(window)
        else:
            window.show()
            
        logger.info("Application fully loaded")
        
        sys.exit(app.exec())
        
    except Exception as e:
        error_msg = f"Failed to create main window: {e}"
        logger.error(error_msg, exc_info=True)
        show_modern_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            error_msg,
            QMessageBox.StandardButton.Ok
        )
        sys.exit(1)

if __name__ == '__main__':
    main()