import os
import sys
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QApplication
from PyQt5.QtCore import Qt
from gui.input_panel import InputPanel
from gui.analysis_dashboard import AnalysisDashboard
from gui.recommendation_panel import RecommendationPanel

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

        # Theme toggle button layout
        theme_layout = QHBoxLayout()
        theme_layout.addStretch()
        self.theme_button = QPushButton("🌙")
        self.theme_button.setFixedSize(24, 24)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.theme_button.setStyleSheet("background-color: #2a82da; color: #ffffff; border-radius: 12px;" if self.is_dark_mode else "background-color: #2a82da; color: black; border-radius: 12px;")
        theme_layout.addWidget(self.theme_button)
        theme_layout.addStretch()
        main_layout.addLayout(theme_layout)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { background-color: #2b2b2b; } QTabBar::tab { background-color: #3c3f41; color: #ffffff; } QTabBar::tab:selected { background-color: #2a82da; }" 
            if self.is_dark_mode else 
            "QTabWidget::pane { background-color: #ffffff; } QTabBar::tab { background-color: #e0e0e0; color: black; } QTabBar::tab:selected { background-color: #2a82da; }"
        )
        main_layout.addWidget(self.tabs)

        # Initialize panels
        self.input_panel = InputPanel(self.data_manager, self)
        self.dashboard_panel = AnalysisDashboard(self.data_manager, self)
        self.recommendation_panel = RecommendationPanel(self.data_manager, self)

        # Add panels to tabs
        self.tabs.addTab(self.input_panel, "Portfolio Setup")
        self.tabs.addTab(self.dashboard_panel, "Dashboard")
        self.tabs.addTab(self.recommendation_panel, "Trading History")

    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self.is_dark_mode = not self.is_dark_mode

        # Update theme button
        self.theme_button.setText("🌙" if self.is_dark_mode else "☀")
        self.theme_button.setStyleSheet(
            "background-color: #2a82da; color: #ffffff; border-radius: 12px;" 
            if self.is_dark_mode else 
            "background-color: #2a82da; color: black; border-radius: 12px;"
        )

        # Update main window and tabs
        self.setStyleSheet(
            "background-color: #353535; color: #ffffff;" 
            if self.is_dark_mode else 
            "background-color: #f0f0f0; color: black;"
        )
        self.tabs.setStyleSheet(
            "QTabWidget::pane { background-color: #2b2b2b; } QTabBar::tab { background-color: #3c3f41; color: #ffffff; } QTabBar::tab:selected { background-color: #2a82da; }" 
            if self.is_dark_mode else 
            "QTabWidget::pane { background-color: #ffffff; } QTabBar::tab { background-color: #e0e0e0; color: black; } QTabBar::tab:selected { background-color: #2a82da; }"
        )

        # Propagate theme to panels
        self.input_panel.set_theme(self.is_dark_mode)
        self.dashboard_panel.set_theme(self.is_dark_mode)
        self.recommendation_panel.set_theme(self.is_dark_mode)

        # Force a full repaint and update all widgets
        self.update_style_recursive(self)
        QApplication.instance().processEvents()  # Ensure all widgets refresh

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