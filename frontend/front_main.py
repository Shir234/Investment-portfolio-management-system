# front_main.py
"""
Main entry point for the SharpSight Investment Portfolio Management System.
Handles application initialization, file selection, theme setup, and window management.
"""
import sys
import os
import logging
from logging_config import setup_logging, get_logger

# Initialize logging
log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'logs')
os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
log_file = os.path.join(log_dir, 'app.log')
setup_logging()
logger = get_logger(__name__)
logger.info("Logging initialized")

# Environment configuration
os.environ["NUMEXPR_MAX_THREADS"] = "8"

# Path setup for imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'backend'))

from PyQt6.QtWidgets import (QApplication, QFileDialog, QDialog, QVBoxLayout, QLabel, 
                             QPushButton, QHBoxLayout, QToolButton, QMessageBox, QTableWidget, 
                             QTableWidgetItem, QHeaderView)
from PyQt6.QtGui import QPalette, QColor, QIcon
from PyQt6.QtCore import Qt

# Import application modules
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
    """Configure the application with modern dark theme styling and color palette."""
    logger.info("Setting modern theme")
    
    # Set the complete modern style (default to dark mode)
    style = ModernStyles.get_complete_style(is_dark=True)
    app.setStyleSheet(style)
    
    # Set modern palette for theme consistency
    colors = ModernStyles.COLORS['dark']
    palette = QPalette()
    
    # Window and base colors
    palette.setColor(QPalette.ColorRole.Window, QColor(colors['primary']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors['text_primary']))   
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
    """Dialog showing CSV file format requirements with example data table."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV File Help")
        self.setFixedSize(517, 350)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setup_ui()
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowCloseButtonHint)
        self.raise_()
        self.activateWindow()

    def setup_ui(self):
        """Set up the UI for the HelpDialog with a table for CSV requirements."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QLabel("CSV File Requirements")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Table for required columns and example row
        self._setup_table(layout)

        # Sample file note
        sample_note = QLabel("File located in 'data\\all_tickers_results.csv'.")
        sample_note.setProperty("class", "subtitle")
        sample_note.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sample_note.setWordWrap(True)
        layout.addWidget(sample_note)

        # Close button
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

    def _setup_table(self, layout):
        """Set up the table displaying CSV required columns and example row, optimized to fit dialog width."""
        table = QTableWidget(1, 7)  # 1 row, 7 columns
        table.setHorizontalHeaderLabels([
            "Date", "Ticker", "Close", "Buy", "Sell", 
            "Actual_Sharpe", "Best_Prediction"
        ])
        
        # Example data row
        example_data = [
            "2021-10-14", "WBD", "25.26", "-1.0", "-1.0", 
            "-2.7896", "-1.8321"  # Truncated for display
        ]
        for col, value in enumerate(example_data):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Read-only
            table.setItem(0, col, item)

        # Table properties
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFixedHeight(80)  # 2 rows (header + 1 data row) at ~40px each
        table.setShowGrid(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setMaximumSectionSize(100)  # Cap column width
        table.verticalHeader().setVisible(False)

        # Table styling
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {ModernStyles.COLORS['dark']['surface']};
                color: {ModernStyles.COLORS['dark']['text_primary']};
                border: 1px solid {ModernStyles.COLORS['dark']['border']};
                border-radius: 6px;
                font-size: 9px;
            }}
            QTableWidget::item {{
                background-color: {ModernStyles.COLORS['dark']['surface']};
                border: none;
                padding: 5px;
            }}
            QHeaderView::section {{
                background-color: {ModernStyles.COLORS['dark']['secondary']};
                color: {ModernStyles.COLORS['dark']['text_primary']};
                font-size: 10px;
                font-weight: bold;
                border: none;
                padding: 5px;
            }}
        """)
        
        layout.addWidget(table)


class StarterDialog(QDialog):
    """Welcome dialog that prompts user to select the CSV file to begin analysis."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to SharpSight")
        self.setFixedSize(520, 300)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setup_ui()
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowCloseButtonHint)
        self.raise_()
        self.activateWindow()

    def setup_ui(self):
        """Set up the welcome dialog UI with title, subtitle, and action buttons."""
        logger.info("Setting up StarterDialog")
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title and subtitle
        title = QLabel("Welcome to SharpSight")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Investment Portfolio Management System")
        subtitle.setProperty("class", "subtitle-large")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # File selection prompt
        prompt = QLabel("Please select 'all_tickers_results.csv' to begin.")
        prompt.setProperty("class", "prompt")
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Help button
        help_button = QToolButton()
        help_button.setText("?")
        help_button.setFixedSize(32, 32)
        help_button.setProperty("class", "help-button")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        # File selection button
        select_button = QPushButton("Select CSV File")
        select_button.clicked.connect(self.accept)
        button_layout.addWidget(select_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Apply modern styling
        try:
            colors = ModernStyles.COLORS['dark']
            style = ModernStyles.get_complete_style(is_dark=True)
            
            # Dialog-specific styling
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
                QLabel[class="subtitle-large"] {{
                    color: {colors['text_secondary']};
                    font-size: 18px;
                    font-weight: 600;
                    margin: 8px 0;
                    line-height: 1.4;
                }}
                QLabel[class="prompt"] {{
                    color: {colors['text_secondary']};
                    font-size: 14px;
                    font-weight: 500;
                    margin: 12px 0;
                }}
                QToolButton[class="help-button"] {{
                    background-color: {colors['surface']};
                    color: {colors['text_primary']};
                    border: 2px solid {colors['border']};
                    border-radius: 16px;
                    font-size: 16px;
                    font-weight: bold;
                    margin: 2px;
                }}
                QToolButton[class="help-button"]:hover {{
                    background-color: {colors['secondary']};
                    border-color: {colors['accent']};
                }}
                QToolButton[class="help-button"]:pressed {{
                    background-color: {colors['accent']};
                    color: white;
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
        """Display the CSV file requirements help dialog."""
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
    
    # Force message box to appear on top of splash screen
    msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
    
    # Apply modern styling
    try:
        colors = ModernStyles.COLORS['dark']
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
        pass
    
    return msg.exec()


def force_window_to_front(window):
    """Force a window to appear in front and be activated."""
    window.show()
    window.raise_()
    window.activateWindow()
    
    # For Windows OS - additional activation
    if sys.platform.startswith('win'):
        try:
            import ctypes
            from ctypes import wintypes
            
            hwnd = int(window.winId())
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
        except Exception:
            pass  # Fallback if ctypes fails


def main():
    """Main function to initialize and run the application."""
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
    
    # Show welcome dialog
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

    # Show splash screen AFTER file selection
    splash = None
    try:
        splash = SplashScreen()
        logger.info("SplashScreen created")
        splash.show()
        splash.raise_()
        splash.activateWindow()
    except Exception as e:
        logger.warning(f"Could not create splash screen: {e}")

    # Initialize data manager with selected CSV
    try:
        data_manager = DataManager(csv_path=csv_path)
        logger.info("DataManager initialized")
    except Exception as e:
        error_msg = f"Failed to initialize DataManager: {e}"
        logger.error(error_msg)
        if splash:
            splash.hide()
        show_modern_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            error_msg,
            QMessageBox.StandardButton.Ok
        )
        sys.exit(1)
    
    # Validate loaded data
    if data_manager.data is None or data_manager.data.empty:
        error_msg = f"Failed to load CSV: {data_manager.csv_path}"
        if splash:
            splash.hide()
        show_modern_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            error_msg,
            QMessageBox.StandardButton.Ok
        )
        logger.error(error_msg)
        sys.exit(1)
    
    # Create main window
    try:
        window = MainWindow(data_manager)
        logger.info("MainWindow created")
        
        # Finish splash and show main window
        if splash:
            splash.finish(window)
        else:
            window.show()
            window.raise_()
            window.activateWindow()
            
        logger.info("Application fully loaded")
        
        sys.exit(app.exec())
        
    except Exception as e:
        error_msg = f"Failed to create main window: {e}"
        logger.error(error_msg, exc_info=True)
        # Hide splash before showing error
        if splash:
            splash.hide()
        show_modern_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            error_msg,
            QMessageBox.StandardButton.Ok
        )
        sys.exit(1)


if __name__ == '__main__':
    main()