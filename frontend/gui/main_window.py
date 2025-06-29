import os
import sys
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QApplication
from PyQt6.QtCore import Qt
from frontend.gui.input_panel import InputPanel
from frontend.gui.analysis_dashboard import AnalysisDashboard
from frontend.gui.recommendation_panel import RecommendationPanel

class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        """Initialize the main window with a data manager and optional parent."""
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True  # Default to dark mode
        self.setWindowTitle("Investment Portfolio Management System")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { background-color: #2b2b2b; } "
            "QTabBar::tab { background-color: #3c3f41; color: #ffffff; padding: 8px; } "
            "QTabBar::tab:selected { background-color: #2a82da; }"
            if self.is_dark_mode else
            "QTabWidget::pane { background-color: #e8e8e8; } "
            "QTabBar::tab { background-color: #d0d0d0; color: #2c2c2c; padding: 8px; } "
            "QTabBar::tab:selected { background-color: #2a82da; color: #ffffff; }"
        )

        # Theme toggle button
        self.theme_button = QPushButton("🌙 Dark")
        self.theme_button.setFixedSize(90, 28)  # Fixed size - keep this consistent
        self.theme_button.clicked.connect(self.toggle_theme)
        
        # Apply initial styling using the same method as toggle
        self.apply_button_style()

        # Set the button as a corner widget of the tab widget
        self.tabs.setCornerWidget(self.theme_button, Qt.Corner.TopRightCorner)

        
        # Add tabs directly to main layout
        main_layout.addWidget(self.tabs)

        # Initialize panels
        self.input_panel = InputPanel(self.data_manager, self)
        self.dashboard_panel = AnalysisDashboard(self.data_manager, self)
        self.recommendation_panel = RecommendationPanel(self.data_manager, self)

        # Add panels to tabs
        self.tabs.addTab(self.input_panel, "Portfolio Setup")
        self.tabs.addTab(self.dashboard_panel, "Dashboard")
        self.tabs.addTab(self.recommendation_panel, "Trading History")
    
    def get_button_style(self, text_color="#ffffff"):
        """Get consistent button styling - SINGLE SOURCE OF TRUTH"""
        return f"""
            QPushButton {{
                background-color: #2a82da;
                color: {text_color};
                border-radius: 14px;
                font-size: 11px;
                font-weight: bold;
                border: none;
                padding: 4px 8px;
                margin-right: 18px;
                margin-top: 6px;
                margin-bottom: 2px;
            }}
            QPushButton:hover {{
                background-color: #3a92ea;
            }}
            QPushButton:pressed {{
                background-color: #1a72ca;
            }}
        """

    def apply_button_style(self):
        """Apply button styling consistently"""
        text_color = "#ffffff" if self.is_dark_mode else "#000000"
        self.theme_button.setStyleSheet(self.get_button_style(text_color))

    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self.is_dark_mode = not self.is_dark_mode
        
        # Update button text
        if self.is_dark_mode:
            self.theme_button.setText("🌙 Dark")
        else:
            self.theme_button.setText("☀ Light")

        # Apply consistent button styling
        self.apply_button_style()
        
        # Update main window style
        self.setStyleSheet(
            "background-color: #353535; color: #ffffff;"
            if self.is_dark_mode else
            "background-color: #e0e0e0; color: #2c2c2c;"  # Darker light gray - more comfortable
        )

        # Update tabs style
        self.tabs.setStyleSheet(
            "QTabWidget::pane { background-color: #2b2b2b; } "
            "QTabBar::tab { background-color: #3c3f41; color: #ffffff; padding: 8px; } "
            "QTabBar::tab:selected { background-color: #2a82da; }"
            if self.is_dark_mode else
            "QTabWidget::pane { background-color: #e8e8e8; } "
            "QTabBar::tab { background-color: #d0d0d0; color: #2c2c2c; padding: 8px; } "
            "QTabBar::tab:selected { background-color: #2a82da; color: #ffffff; }"
        )

        # Update panel themes
        self.input_panel.set_theme(self.is_dark_mode)
        self.dashboard_panel.set_theme(self.is_dark_mode)
        self.recommendation_panel.set_theme(self.is_dark_mode)
        self.update_style_recursive(self)
        QApplication.instance().processEvents()

    def update_style_recursive(self, widget):
        """Recursively update the style of all child widgets."""
        for child in widget.findChildren(QWidget):
            if hasattr(child, 'setStyleSheet'):
                child.setStyleSheet(child.styleSheet())  # Reapply to force update
            self.update_style_recursive(child)

    def update_dashboard(self):
        """Refresh the dashboard and recommendations."""
        self.dashboard_panel.update_dashboard()
        self.recommendation_panel.update_recommendations()

    def update_recommendations(self):
        """Refresh the recommendations."""
        self.recommendation_panel.update_recommendations()