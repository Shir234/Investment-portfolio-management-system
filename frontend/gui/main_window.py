from PyQt5.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from gui.input_panel import InputPanel
from gui.analysis_dashboard import AnalysisDashboard
from gui.recommendation_panel import RecommendationPanel

class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setWindowTitle("Investment Portfolio Management System")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Initialize panels
        self.input_panel = InputPanel(data_manager, self)
        self.dashboard_panel = AnalysisDashboard(data_manager, self)
        self.recommendation_panel = RecommendationPanel(data_manager, self)

        # Add panels to tabs
        self.tabs.addTab(self.input_panel, "Portfolio Setup")
        self.tabs.addTab(self.dashboard_panel, "Dashboard")
        self.tabs.addTab(self.recommendation_panel, "Recommendations")

    def update_dashboard(self):
        """Method to refresh the dashboard, called by InputPanel"""
        self.dashboard_panel.update_dashboard()
        self.recommendation_panel.update_recommendations()
