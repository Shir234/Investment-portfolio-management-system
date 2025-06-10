from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QPushButton, QToolBar, QAction, QStackedLayout, QLabel
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt

from gui.input_panel import InputPanel
from gui.analysis_dashboard import AnalysisDashboard
from gui.recommendation_panel import RecommendationPanel

class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True
        self.setWindowTitle("Investment Portfolio Management System")
        self.setMinimumSize(800, 600)
        self.resize(1280, 800)

        # Apply initial theme
        self.apply_theme()

        # Setup UI components
        self._create_toolbar()
        self._create_tabs()

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(Qt.QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Theme toggle action
        self.theme_action = QAction(QIcon("icons/theme.png"), "Toggle Theme", self)
        self.theme_action.setToolTip("Switch between Dark and Light mode")
        self.theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_action)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(spacer.Expanding, spacer.Preferred)
        toolbar.addWidget(spacer)

        # Logo or title label
        logo = QLabel("<b>PortfolioPro</b>")
        logo.setFont(QFont("Segoe UI", 14))
        toolbar.addWidget(logo)

    def _create_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.setStyleSheet(self._tab_style())

        self.input_panel = InputPanel(self.data_manager, self)
        self.dashboard_panel = AnalysisDashboard(self.data_manager, self)
        self.recommendation_panel = RecommendationPanel(self.data_manager, self)

        self.tabs.addTab(self.input_panel, QIcon("icons/portfolio.png"), "Portfolio Setup")
        self.tabs.addTab(self.dashboard_panel, QIcon("icons/dashboard.png"), "Dashboard")
        self.tabs.addTab(self.recommendation_panel, QIcon("icons/history.png"), "History")

        central = QWidget()
        layout = QStackedLayout(central)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()
        self.tabs.setStyleSheet(self._tab_style())
        self.input_panel.set_theme(self.is_dark_mode)
        self.dashboard_panel.set_theme(self.is_dark_mode)
        self.recommendation_panel.set_theme(self.is_dark_mode)

    def apply_theme(self):
        palette = QPalette()
        if self.is_dark_mode:
            palette.setColor(QPalette.Window, QColor(45, 45, 45))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(30, 30, 30))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
            self.theme_action.setIcon(QIcon("icons/sun.png"))
        else:
            palette = QPalette()
            self.theme_action.setIcon(QIcon("icons/moon.png"))
        self.setPalette(palette)

    def _tab_style(self):
        if self.is_dark_mode:
            return """
                QTabWidget::pane { border: none; background: #353535; }
                QTabBar::tab { background: #3c3f41; color: #cfd2d7; padding: 10px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
                QTabBar::tab:selected { background: #2a82da; color: white; }
                QTabBar::tab:hover { background: #4c5053; }
            """
        else:
            return """
                QTabWidget::pane { border: none; background: #f0f0f0; }
                QTabBar::tab { background: #e0e0e0; color: #333; padding: 10px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
                QTabBar::tab:selected { background: #2a82da; color: white; }
                QTabBar::tab:hover { background: #d4d4d4; }
            """
