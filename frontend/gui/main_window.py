from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from .input_panel import InputPanel
from .analysis_dashboard import AnalysisDashboard
from .recommendation_panel import RecommendationPanel

class MainWindow(QMainWindow):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.setWindowTitle("SharpSight - Portfolio Management System")
        self.setMinimumSize(1200, 800)
        
        # Set window icon
        self.setWindowIcon(QIcon('logo.JPG'))
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget with dark styling
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create and add tabs
        self.setup_tabs()
        
        # Create status bar with dark styling
        self.statusBar().showMessage("Ready")
        
    def setup_tabs(self):
        # Portfolio Overview Tab
        portfolio_tab = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_tab)
        
        # Add input panel
        self.input_panel = InputPanel(self.data_manager)
        portfolio_layout.addWidget(self.input_panel)
        
        # Add recommendation panel
        self.recommendation_panel = RecommendationPanel(self.data_manager)
        portfolio_layout.addWidget(self.recommendation_panel)
        
        # Analysis Dashboard Tab
        self.analysis_dashboard = AnalysisDashboard(self.data_manager)
        
        # Add tabs to widget
        self.tab_widget.addTab(portfolio_tab, "Portfolio Overview")
        self.tab_widget.addTab(self.analysis_dashboard, "Analysis Dashboard")
        
    def update_dashboard(self):
        """Update all dashboard components with new data"""
        self.analysis_dashboard.update_visualizations()
        self.recommendation_panel.update_recommendations()

