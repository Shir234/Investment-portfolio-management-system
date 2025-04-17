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
        self.setWindowTitle("Investment Portfolio Management System")
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.input_panel = InputPanel(parent=self, data_manager=self.data_manager)
        self.tabs.addTab(self.input_panel, "Portfolio Overview")
        
        self.analysis_dashboard = AnalysisDashboard(self.data_manager)
        self.tabs.addTab(self.analysis_dashboard, "Analysis Dashboard")
        
        self.recommendation_panel = RecommendationPanel(self.data_manager, parent=self)  # Pass self as parent
        self.tabs.addTab(self.recommendation_panel, "Recommendations")
    
    def update_dashboard(self):
        self.analysis_dashboard.update_visualizations()
        self.recommendation_panel.update_recommendations()