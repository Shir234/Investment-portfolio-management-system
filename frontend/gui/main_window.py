import os
import sys
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QApplication, QFrame, QLabel
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QThread, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon
from frontend.utils import resource_path
from frontend.logging_config import get_logger
from frontend.gui.input_panel import InputPanel
from frontend.gui.analysis_dashboard import AnalysisDashboard
from frontend.gui.recommendation_panel import RecommendationPanel
from frontend.gui.styles import ModernStyles

logger = get_logger(__name__)

class ThemeWorker(QObject):
    """Worker class to handle theme changes in background."""
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    
    def __init__(self, main_window, is_dark_mode):
        super().__init__()
        self.main_window = main_window
        self.is_dark_mode = is_dark_mode
    
    def run(self):
        """Apply theme changes in background."""
        try:
            self.progress.emit("Applying theme...")
            
            # Get the complete style
            complete_style = ModernStyles.get_complete_style(self.is_dark_mode)
            
            # Apply to main window on main thread via signal
            self.main_window.apply_style_signal.emit(complete_style)
            
            self.progress.emit("Updating panels...")
            
            # Update panel themes
            panels = [
                self.main_window.input_panel,
                self.main_window.dashboard_panel,
                self.main_window.recommendation_panel
            ]
            
            for panel in panels:
                if hasattr(panel, 'set_theme'):
                    # Use signal to update on main thread
                    self.main_window.update_panel_signal.emit(panel, self.is_dark_mode)
            
            self.progress.emit("Finalizing...")
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"Error in theme worker: {e}")
            self.finished.emit()

class MainWindow(QMainWindow):
    # Signals for thread-safe UI updates
    apply_style_signal = pyqtSignal(str)
    update_panel_signal = pyqtSignal(object, bool)
    
    def __init__(self, data_manager, parent=None):
        """Initialize the modern main window with enhanced styling."""
        super().__init__(parent)
        self.data_manager = data_manager
        self.is_dark_mode = True  # Default to dark mode
        self.theme_changing = False  # Flag to prevent multiple theme changes
        
        # Connect signals for thread-safe updates
        self.apply_style_signal.connect(self.apply_style_safely)
        self.update_panel_signal.connect(self.update_panel_safely)
        
        # Window setup
        self.setWindowTitle("SharpSight Investment System")
        self.setGeometry(100, 100, 1200, 800)  # Larger default size
        self.setMinimumSize(1000, 700)  # Minimum size for better UX

       # Set window icon
        try:
            window_icon = QIcon(resource_path("frontend/gui/icons/portfolio.png"))
            if not window_icon.isNull():
                self.setWindowIcon(window_icon)
                logger.info("Window icon loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load window icon: {e}")
        
        # Create the UI
        self.setup_ui()
        self.apply_modern_theme()
        
    def setup_ui(self):
        """Setup the modern UI with better layout and spacing."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)  # Reduced from (20, 20, 20, 20)
        main_layout.setSpacing(12)  # Reduced from 16
        
        # Create header with title and theme toggle
        self.create_header(main_layout)
        
        # Create modern tab widget
        self.create_tab_widget(main_layout)
        
        # Initialize panels with modern styling
        self.initialize_panels()

        
    def create_header(self, main_layout):
        """Create a modern header with title and controls."""
        header_frame = QFrame()
        header_frame.setFixedHeight(60)  # Reduced from 80
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # App title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)  # Reduced from 4
        
        self.title_label = QLabel("SharpSight")
        self.title_label.setProperty("class", "title")
        
        self.subtitle_label = QLabel("Investment Portfolio Management System")
        self.subtitle_label.setProperty("class", "subtitle")
        
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)
        title_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Theme toggle button (no emoji)
        self.theme_button = QPushButton("Light Mode")  # Start with "Light Mode" since we default to dark
        self.theme_button.setProperty("class", "theme-toggle")
        self.theme_button.setFixedSize(100, 32)  # Reduced from (120, 40)
        self.theme_button.clicked.connect(self.toggle_theme)

        # Try to load theme toggle icon
        try:
            theme_icon_path = "frontend/gui/icons/yellow_sun.png" if self.is_dark_mode else "frontend/gui/icons/black_sun.png"
            theme_icon = QIcon(resource_path(theme_icon_path))
            if not theme_icon.isNull():
                self.theme_button.setIcon(theme_icon)
                logger.info(f"Theme icon loaded: {theme_icon_path}")
        except Exception as e:
            logger.warning(f"Could not load theme icon: {e}")
        
        header_layout.addWidget(self.theme_button)
        main_layout.addWidget(header_frame)
        
    def create_tab_widget(self, main_layout):
        """Create modern tab widget with better styling."""
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)
        self.tabs.setTabsClosable(False)
        
        # Add some padding around tab content
        tab_frame = QFrame()
        tab_layout = QVBoxLayout(tab_frame)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(self.tabs)
        
        main_layout.addWidget(tab_frame)
        
    def initialize_panels(self):
        """Initialize all panels with modern styling and icons."""
        # Create panels with date constraints
        self.input_panel = InputPanel(self.data_manager, self)
        self.dashboard_panel = AnalysisDashboard(self.data_manager, self)
        self.recommendation_panel = RecommendationPanel(self.data_manager, self)
        
        # Set date constraints based on data
        self.set_date_constraints()
        
        # Icon paths for tabs
        icon_paths = [
            "frontend/gui/icons/portfolio.png",
            "frontend/gui/icons/analytics.png", 
            "frontend/gui/icons/history.png"
        ]
        
        # Tab names
        tab_names = ["Portfolio Setup", "Analytics Dashboard", "Trading History"]
        panels = [self.input_panel, self.dashboard_panel, self.recommendation_panel]
        
        # Add panels to tabs with icons
        for i, (icon_path, tab_name, panel) in enumerate(zip(icon_paths, tab_names, panels)):
            try:
                icon = QIcon(resource_path(icon_path))
                if not icon.isNull():
                    self.tabs.addTab(panel, icon, tab_name)
                    logger.info(f"Tab icon loaded: {icon_path}")
                else:
                    self.tabs.addTab(panel, tab_name)
                    logger.warning(f"Tab icon not found: {icon_path}")
            except Exception as e:
                self.tabs.addTab(panel, tab_name)
                logger.warning(f"Error loading tab icon {icon_path}: {e}")
        
        # Set initial tab
        self.tabs.setCurrentIndex(0)
        
    def set_date_constraints(self):
        """Set date constraints based on available data in the CSV file."""
        if self.data_manager and self.data_manager.data is not None:
            try:
                # Get date range from data
                dates = self.data_manager.data['Date'].dropna()
                if not dates.empty:
                    # Convert to datetime if not already
                    import pandas as pd
                    dates = pd.to_datetime(dates)
                    
                    min_date = dates.min().date()
                    max_date = dates.max().date()
                    
                    # Set constraints in input panel if it has date fields
                    if hasattr(self.input_panel, 'set_date_constraints'):
                        self.input_panel.set_date_constraints()
                    elif hasattr(self.input_panel, 'start_date_edit') and hasattr(self.input_panel, 'end_date_edit'):
                        # Direct access to date widgets
                        from PyQt6.QtCore import QDate
                        
                        # Convert Python dates to QDate
                        q_min_date = QDate(min_date.year, min_date.month, min_date.day)
                        q_max_date = QDate(max_date.year, max_date.month, max_date.day)
                        
                        # Set date ranges
                        self.input_panel.start_date_edit.setDateRange(q_min_date, q_max_date)
                        self.input_panel.end_date_edit.setDateRange(q_min_date, q_max_date)
                        
                        # Set default values
                        self.input_panel.start_date_edit.setDate(q_min_date)
                        self.input_panel.end_date_edit.setDate(q_max_date)
                        
            except Exception as e:
                print(f"Error setting date constraints: {e}")
        
    def apply_modern_theme(self):
        """Apply the complete modern theme."""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        self.setStyleSheet(style)
        
        # Update colors for specific elements
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        # Header styling
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['text_primary']};
                    font-size: 28px;
                    font-weight: 700;
                    margin: 0;
                }}
            """)
            
        if hasattr(self, 'subtitle_label'):
            self.subtitle_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['text_secondary']};
                    font-size: 14px;
                    font-weight: 400;
                    margin: 0;
                }}
            """)
        
    def toggle_theme(self):
        """Toggle between light and dark mode using worker thread."""
        # Prevent multiple theme changes at once
        if self.theme_changing:
            return
            
        self.theme_changing = True
        self.theme_button.setEnabled(False)
        self.theme_button.setText("Changing...")
        
        # Toggle the mode
        self.is_dark_mode = not self.is_dark_mode
        
        # Create and start worker thread
        self.theme_thread = QThread()
        self.theme_worker = ThemeWorker(self, self.is_dark_mode)
        self.theme_worker.moveToThread(self.theme_thread)
        
        # Connect signals
        self.theme_thread.started.connect(self.theme_worker.run)
        self.theme_worker.finished.connect(self.on_theme_change_finished)
        self.theme_worker.finished.connect(self.theme_thread.quit)
        self.theme_worker.finished.connect(self.theme_worker.deleteLater)
        self.theme_thread.finished.connect(self.theme_thread.deleteLater)
        
        # Start the thread
        self.theme_thread.start()
        
    def apply_style_safely(self, style):
        """Apply style safely on main thread."""
        try:
            self.setStyleSheet(style)
            # Update header elements
            self.update_header_styling()
        except Exception as e:
            logger.error(f"Error applying style: {e}")
    
    def update_panel_safely(self, panel, is_dark_mode):
        """Update panel theme safely on main thread."""
        try:
            if hasattr(panel, 'set_theme'):
                panel.set_theme(is_dark_mode)
        except Exception as e:
            logger.error(f"Error updating panel theme: {e}")
    
    def update_header_styling(self):
        """Update header styling for current theme."""
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['text_primary']};
                    font-size: 28px;
                    font-weight: 700;
                    margin: 0;
                }}
            """)
            
        if hasattr(self, 'subtitle_label'):
            self.subtitle_label.setStyleSheet(f"""
                QLabel {{
                    color: {colors['text_secondary']};
                    font-size: 14px;
                    font-weight: 400;
                    margin: 0;
                }}
            """)
    
    def on_theme_change_finished(self):
        """Handle theme change completion."""
        try:
            # Update button text and icon
            if self.is_dark_mode:
                self.theme_button.setText("Light Mode")
                try:
                    theme_icon = QIcon(resource_path("frontend/gui/icons/yellow_sun.png"))
                    if not theme_icon.isNull():
                        self.theme_button.setIcon(theme_icon)
                except:
                    pass
            else:
                self.theme_button.setText("Dark Mode")
                try:
                    theme_icon = QIcon(resource_path("frontend/gui/icons/black_sun.png"))
                    if not theme_icon.isNull():
                        self.theme_button.setIcon(theme_icon)
                except:
                    pass
            
            # Re-enable button and reset flag
            self.theme_button.setEnabled(True)
            self.theme_changing = False
            
            # Force a gentle refresh of the UI
            QTimer.singleShot(100, self.gentle_ui_refresh)
            
            logger.info(f"Theme toggled to {'dark' if self.is_dark_mode else 'light'} mode")
            
        except Exception as e:
            logger.error(f"Error finishing theme change: {e}")
            self.theme_button.setEnabled(True)
            self.theme_changing = False
    
    def gentle_ui_refresh(self):
        """Gently refresh the UI without blocking."""
        try:
            # Process any pending events
            QApplication.processEvents()
            
            # Update the main window
            self.update()
            
            # Update tabs if they exist
            if hasattr(self, 'tabs'):
                self.tabs.update()
                
        except Exception as e:
            logger.error(f"Error in gentle UI refresh: {e}")
        
    def update_message_box_styles(self):
        """Update message box styling for current theme."""
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        # Store the current theme colors for message boxes
        self._current_theme_colors = colors
            
    def update_dashboard(self):
        """Refresh the dashboard and recommendations."""
        if hasattr(self.dashboard_panel, 'update_dashboard'):
            self.dashboard_panel.update_dashboard()
        if hasattr(self.recommendation_panel, 'update_recommendations'):
            self.recommendation_panel.update_recommendations()
        
    def update_recommendations(self):
        """Refresh the recommendations."""
        if hasattr(self.recommendation_panel, 'update_recommendations'):
            self.recommendation_panel.update_recommendations()
            
    def show_success_message(self, title, message):
        """Show a success message with modern styling."""
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Apply modern styling to message box
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                font-size: 14px;
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QMessageBox QLabel {{
                color: {colors['text_primary']};
                padding: 16px;
                background-color: transparent;
            }}
            QMessageBox QPushButton {{
                background-color: {colors['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #059669;
            }}
        """)
        msg.exec()
        
    def show_error_message(self, title, message):
        """Show an error message with modern styling."""
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Apply modern styling to message box
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                font-size: 14px;
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QMessageBox QLabel {{
                color: {colors['text_primary']};
                padding: 16px;
                background-color: transparent;
            }}
            QMessageBox QPushButton {{
                background-color: {colors['danger']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #DC2626;
            }}
        """)
        msg.exec()
        
    def show_warning_message(self, title, message):
        """Show a warning message with modern styling."""
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        # Apply modern styling to message box
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                font-size: 14px;
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QMessageBox QLabel {{
                color: {colors['text_primary']};
                padding: 16px;
                background-color: transparent;
            }}
            QMessageBox QPushButton {{
                background-color: {colors['warning']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
                margin: 4px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #D97706;
            }}
        """)
        return msg.exec()