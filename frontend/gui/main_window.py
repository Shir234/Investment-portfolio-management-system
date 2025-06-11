
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend')))

from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QPoint
from gui.input_panel import InputPanel
from gui.analysis_dashboard import AnalysisDashboard
from gui.recommendation_panel import RecommendationPanel

class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True  # Default to dark mode
        self.setWindowTitle("Investment Portfolio Management System")
        self.setGeometry(100, 100, 800, 600)

        # Remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Custom title bar
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(30)
        self.title_bar.setStyleSheet("background-color: #3c3f41;" if self.is_dark_mode else "background-color: #e0e0e0;")
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(10, 0, 10, 0)

        # Title label
        self.title_label = QLabel("Investment Portfolio Management System")
        self.title_label.setStyleSheet("color: #ffffff;" if self.is_dark_mode else "color: black;")
        title_bar_layout.addWidget(self.title_label)

        # Spacer
        title_bar_layout.addStretch()

        # Theme toggle button
        self.theme_button = QPushButton("🌙")
        self.theme_button.setFixedSize(24, 24)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.theme_button.setStyleSheet("background-color: #2a82da; color: #ffffff; border-radius: 12px;" if self.is_dark_mode else "background-color: #2a82da; color: black; border-radius: 12px;")
        title_bar_layout.addWidget(self.theme_button)

        # Window control buttons
        minimize_button = QPushButton("−")
        minimize_button.setFixedSize(24, 24)
        minimize_button.clicked.connect(self.showMinimized)
        minimize_button.setStyleSheet("background-color: #3c3f41; color: #ffffff;" if self.is_dark_mode else "background-color: #e0e0e0; color: black;")
        title_bar_layout.addWidget(minimize_button)

        maximize_button = QPushButton("□")
        maximize_button.setFixedSize(24, 24)
        maximize_button.clicked.connect(self.toggle_maximize)
        maximize_button.setStyleSheet("background-color: #3c3f41; color: #ffffff;" if self.is_dark_mode else "background-color: #e0e0e0; color: black;")
        title_bar_layout.addWidget(maximize_button)

        close_button = QPushButton("✕")
        close_button.setFixedSize(24, 24)
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet("background-color: #ff4444; color: #ffffff;")
        title_bar_layout.addWidget(close_button)

        main_layout.addWidget(self.title_bar)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { background-color: #2b2b2b; } QTabBar::tab { background-color: #3c3f41; color: #ffffff; } QTabBar::tab:selected { background-color: #2a82da; }" if self.is_dark_mode else "QTabWidget::pane { background-color: #ffffff; } QTabBar::tab { background-color: #e0e0e0; color: black; } QTabBar::tab:selected { background-color: #2a82da; }")
        main_layout.addWidget(self.tabs)

        # Initialize panels
        self.input_panel = InputPanel(data_manager, self)
        self.dashboard_panel = AnalysisDashboard(data_manager, self)
        self.recommendation_panel = RecommendationPanel(data_manager, self)

        # Add panels to tabs
        self.tabs.addTab(self.input_panel, "Portfolio Setup")
        self.tabs.addTab(self.dashboard_panel, "Dashboard")
        self.tabs.addTab(self.recommendation_panel, "Trading History")

        # Variables for dragging
        self.dragging = False
        self.drag_position = QPoint()

    def mousePressEvent(self, event):
        """Handle mouse press to start dragging."""
        if event.button() == Qt.LeftButton and self.title_bar.underMouse():
            self.dragging = True
            self.drag_position = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move to drag the window."""
        if self.dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging."""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def toggle_theme(self):
        """Toggle between light and dark mode."""
        self.is_dark_mode = not self.is_dark_mode
        
        # Update title bar and theme button
        self.title_bar.setStyleSheet("background-color: #3c3f41;" if self.is_dark_mode else "background-color: #e0e0e0;")
        self.title_label.setStyleSheet("color: #ffffff;" if self.is_dark_mode else "color: black;")
        self.theme_button.setText("🌙" if self.is_dark_mode else "☀")
        self.theme_button.setStyleSheet("background-color: #2a82da; color: #ffffff; border-radius: 12px;" if self.is_dark_mode else "background-color: #2a82da; color: black; border-radius: 12px;")
        
        # Update window control buttons
        for button in self.findChildren(QPushButton):
            if button.text() in ["−", "□"]:
                button.setStyleSheet("background-color: #3c3f41; color: #ffffff;" if self.is_dark_mode else "background-color: #e0e0e0; color: black;")
            elif button.text() == "✕":
                button.setStyleSheet("background-color: #ff4444; color: #ffffff;")

        # Update main window and tabs
        self.setStyleSheet("background-color: #353535; color: #ffffff;" if self.is_dark_mode else "background-color: #f0f0f0; color: black;")
        self.tabs.setStyleSheet("QTabWidget::pane { background-color: #2b2b2b; } QTabBar::tab { background-color: #3c3f41; color: #ffffff; } QTabBar::tab:selected { background-color: #2a82da; }" if self.is_dark_mode else "QTabWidget::pane { background-color: #ffffff; } QTabBar::tab { background-color: #e0e0e0; color: black; } QTabBar::tab:selected { background-color: #2a82da; }")
        
        # Propagate theme to panels
        self.input_panel.set_theme(self.is_dark_mode)
        self.dashboard_panel.set_theme(self.is_dark_mode)
        self.recommendation_panel.set_theme(self.is_dark_mode)

    def update_dashboard(self):
        """Method to refresh the dashboard, called by InputPanel"""
        self.dashboard_panel.update_dashboard()
        self.recommendation_panel.update_recommendations()

    def update_recommendations(self):
        """Method to refresh the recommendations, called by InputPanel"""
        self.recommendation_panel.update_recommendations()