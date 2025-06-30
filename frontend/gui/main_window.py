from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QColor
from .input_panel import InputPanel
from .analysis_dashboard import AnalysisDashboard
from .recommendation_panel import RecommendationPanel
from ..logging_config import get_logger
from ..utils import resource_path

# Configure logging
logger = get_logger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        """Initialize the main window with a data manager and optional parent."""
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True
        self.setWindowTitle("SharpSight Investment System")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()
        logger.info("MainWindow initialized")

    def setup_ui(self):
        """Set up the UI with a modern, professional design."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Header with title and theme toggle
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        self.title_label = QLabel("SharpSight")
        self.title_label.setStyleSheet(self.get_title_style())
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(3, 3)
        self.title_label.setGraphicsEffect(shadow)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.theme_button = QPushButton()
        self.theme_button.setFixedSize(36, 36)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.theme_button.setToolTip("Switch to Light Mode")
        header_layout.addWidget(self.theme_button)
        main_layout.addLayout(header_layout)

        # Tab widget for panels
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        main_layout.addWidget(self.tabs)

        # Initialize panels
        self.input_panel = InputPanel(self.data_manager, self)
        self.dashboard_panel = AnalysisDashboard(self.data_manager, self)
        self.recommendation_panel = RecommendationPanel(self.data_manager, self)

        # Add tabs with icons
        icon_paths = [
            "frontend/gui/icons/portfolio.png",
            "frontend/gui/icons/analytics.png",
            "frontend/gui/icons/history.png"
        ]
        tab_names = ["Portfolio Setup", "Analytics Dashboard", "Trading History"]
        for i, (icon_path, tab_name) in enumerate(zip(icon_paths, tab_names)):
            icon = QIcon(resource_path(icon_path))
            if icon.isNull():
                logger.warning(f"Tab icon not found: {icon_path}")
            else:
                logger.info(f"Tab icon loaded: {icon_path}")
            self.tabs.addTab([self.input_panel, self.dashboard_panel, self.recommendation_panel][i], icon, tab_name)

        # Apply initial theme
        self.set_theme(self.is_dark_mode)

    def get_title_style(self):
        """Return stylesheet for the title label."""
        return f"""
            font-size: 32px;
            font-weight: bold;
            font-family: 'Segoe UI';
            color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
            padding: 8px;
        """

    def get_button_style(self):
        """Return stylesheet for buttons."""
        return f"""
            QPushButton {{
                background-color: #0078D4;
                color: #FFFFFF;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Segoe UI';
                padding: 8px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #005BA1;
            }}
            QPushButton:pressed {{
                background-color: #003E7E;
            }}
        """

    def get_tab_style(self):
        """Return stylesheet for tabs."""
        return f"""
            QTabWidget::pane {{
                background-color: {'#2D2D2D' if self.is_dark_mode else '#F5F5F5'};
                border: 1px solid {'#444444' if self.is_dark_mode else '#CCCCCC'};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background-color: {'#3C3F41' if self.is_dark_mode else '#E0E0E0'};
                color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
                padding: 12px 24px;
                margin: 4px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-size: 14px;
            }}
            QTabBar::tab:selected {{
                background-color: #0078D4;
                color: #FFFFFF;
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                background-color: {'#555555' if self.is_dark_mode else '#D0D0D0'};
            }}
        """

    def get_main_style(self):
        """Return stylesheet for the main window."""
        return f"""
            QMainWindow {{
                background-color: {'#2D2D2D' if self.is_dark_mode else '#F5F5F5'};
            }}
            QWidget {{
                background-color: {'#2D2D2D' if self.is_dark_mode else '#F5F5F5'};
                color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
            }}
            QLabel {{
                color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
                font-family: 'Segoe UI';
                font-size: 14px;
            }}
        """

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the main window and its panels."""
        self.is_dark_mode = is_dark_mode
        theme_icon_path = "frontend/gui/icons/yellow_sun.png" if is_dark_mode else "frontend/gui/icons/black_sun.png"
        theme_icon = QIcon(resource_path(theme_icon_path))
        if theme_icon.isNull():
            logger.warning(f"Theme icon not found: {theme_icon_path}")
        else:
            logger.info(f"Theme icon loaded: {theme_icon_path}")
        self.theme_button.setIcon(theme_icon)
        self.theme_button.setToolTip("Switch to Light Mode" if is_dark_mode else "Switch to Dark Mode")
        self.theme_button.setStyleSheet(self.get_button_style())
        self.setStyleSheet(self.get_main_style())
        self.tabs.setStyleSheet(self.get_tab_style())
        self.title_label.setStyleSheet(self.get_title_style())
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(3, 3)
        self.title_label.setGraphicsEffect(shadow)
        self.input_panel.set_theme(is_dark_mode)
        self.dashboard_panel.set_theme(is_dark_mode)
        self.recommendation_panel.set_theme(is_dark_mode)
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self.set_theme(not self.is_dark_mode)

    def update_dashboard(self):
        """Refresh the dashboard panel."""
        self.dashboard_panel.update_dashboard()
        logger.debug("Dashboard updated")

    def update_recommendations(self):
        """Refresh the recommendations panel."""
        self.recommendation_panel.update_recommendations()
        logger.debug("Recommendations updated")