# input_panel.py
import os
import pandas as pd
from datetime import datetime, date
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
                             QDateEdit, QPushButton, QComboBox, QMessageBox,
                             QDoubleSpinBox, QSpinBox, QDialog, QTableWidget, QTableWidgetItem,
                             QCheckBox, QFrame, QGridLayout, QSpacerItem, QSizePolicy,
                             QGroupBox, QProgressBar, QToolButton, QScrollArea)
from PyQt6.QtCore import QDate, Qt, QThread, QObject, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
from frontend.logging_config import get_logger
from frontend.data.trading_connector import execute_trading_strategy, get_order_history_df, log_trading_orders
from backend.trading_logic_new import get_orders, get_portfolio_history
from frontend.gui.styles import ModernStyles
from frontend.utils import resource_path
from frontend.gui.wheel_disabled_widgets import WheelDisabledSpinBox, WheelDisabledDateEdit, WheelDisabledComboBox
from frontend.gui.semi_automated_manager import SemiAutomatedManager
from frontend.gui.worker import Worker


# Set up logging
logger = get_logger(__name__)

class TradeConfirmationDialog(QDialog):
    """Modern dialog for confirming trades in semi-automatic mode."""
    def __init__(self, orders, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Trades")
        self.setFixedSize(800, 600)
        self.orders = orders
        self.selected_orders = []
        self.is_dark_mode = getattr(parent, 'is_dark_mode', True)
        self.setup_ui()
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.raise_()
        self.activateWindow()

    def setup_ui(self):
        """Configure the modern dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Header
        header_label = QLabel("Review and Confirm Trades")
        header_label.setProperty("class", "title")
        layout.addWidget(header_label)

        subtitle_label = QLabel(f"Found {len(self.orders)} potential trades. Select which ones to execute:")
        subtitle_label.setProperty("class", "subtitle")
        layout.addWidget(subtitle_label)

        # Table
        self.table = QTableWidget()
        self.table.setRowCount(len(self.orders))
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Select", "Date", "Ticker", "Action", "Shares", "Price"])
        
        # Set specific column widths - fix for checkbox column
        self.table.setColumnWidth(0, 80)  # Select column - wider for checkboxes
        self.table.setColumnWidth(1, 140)  # Date column
        self.table.setColumnWidth(2, 80)   # Ticker column
        self.table.setColumnWidth(3, 80)   # Action column
        self.table.setColumnWidth(4, 80)   # Shares column
        self.table.setColumnWidth(5, 100)  # Price column
        
        # Set row height to accommodate larger checkboxes
        self.table.verticalHeader().setDefaultSectionSize(50)
        
        # Make only the last column stretch to fill remaining space
        self.table.horizontalHeader().setStretchLastSection(True)

        for row, order in enumerate(self.orders):
            # Create checkbox widget
            checkbox_widget = self.create_custom_checkbox()
            self.table.setCellWidget(row, 0, checkbox_widget)

            self.table.setItem(row, 1, QTableWidgetItem(str(order.get('date', ''))))
            self.table.setItem(row, 2, QTableWidgetItem(order.get('ticker', '')))
            self.table.setItem(row, 3, QTableWidgetItem(order.get('action', '')))
            self.table.setItem(row, 4, QTableWidgetItem(str(order.get('shares_amount', 0))))
            self.table.setItem(row, 5, QTableWidgetItem(f"${order.get('price', 0):,.2f}"))

            for col in range(1, 6):
                if self.table.item(row, col):
                    self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setProperty("class", "secondary")
        cancel_button.clicked.connect(self.reject)
        
        accept_button = QPushButton("Execute Selected Trades")
        accept_button.setProperty("class", "success")
        accept_button.clicked.connect(self.accept_selected)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(accept_button)
        layout.addLayout(button_layout)

        # Apply modern styling
        self.apply_styles()

    def create_custom_checkbox(self):
        """Create a custom checkbox widget that shows ✓ when checked."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create a clickable label that acts as checkbox
        check_label = QLabel()
        check_label.setFixedSize(20, 20)
        check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Track checked state
        check_label.checked = True
        
        def update_appearance():
            if check_label.checked:
                check_label.setText("✓")
                check_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #4CAF50;
                        border-radius: 6px;
                        background-color: #4CAF50;
                        font-size: 16px;
                        font-weight: bold;
                        color: white;
                        padding: 1px;
                    }
                    QLabel:hover {
                        background-color: #66bb6a;
                        border: 3px solid #66bb6a;
                    }
                """)
            else:
                check_label.setText("")
                check_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #ccc;
                        border-radius: 6px;
                        background-color: white;
                        font-size: 16px;
                        font-weight: bold;
                        color: white;
                        padding: 1px;
                    }
                    QLabel:hover {
                        background-color: #f5f5f5;
                        border: 3px solid #999;
                    }
                """)
        
        def toggle_check(event):
            check_label.checked = not check_label.checked
            update_appearance()
        
        # Make it clickable
        check_label.mousePressEvent = toggle_check
        
        # Set initial appearance
        update_appearance()
        
        layout.addWidget(check_label)
        
        # Store reference to check state
        widget.is_checked = lambda: check_label.checked
        
        return widget

    def apply_styles(self):
        """Apply modern styling to the panel with darker label backgrounds."""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        
        # Add custom styles for the dark label frames
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        # Additional styles for label frames and input frames with reduced spacing
        additional_styles = f"""
            /* Label Frame Styling - Darker background for labels with reduced height */
            QFrame[class="label-frame"] {{
                background-color: {'#252538' if self.is_dark_mode else '#9CA3AF'};
                border: 1px solid {colors['border']};
                border-bottom: none;
                border-top-left-radius: 6px;  /* Reduced from 8px */
                border-top-right-radius: 6px;  /* Reduced from 8px */
            }}
            
            /* Input Frame Styling - Matches the surface color with reduced spacing */
            QFrame[class="input-frame"] {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-top: none;
                border-bottom-left-radius: 6px;  /* Reduced from 8px */
                border-bottom-right-radius: 6px;  /* Reduced from 8px */
            }}
            
            /* Label separator line */
            QFrame[class="label-separator"] {{
                color: {colors['border_light']};
                background-color: {colors['border_light']};
                max-width: 1px;
                margin: 2px 0px;  /* Reduced from 4px 0px */
            }}
            
            /* Dark label text styling with smaller font */
            QLabel[class="label-dark"] {{
                color: {'#FFFFFF' if self.is_dark_mode else '#FFFFFF'};
                font-size: 13px;  /* Reduced from 14px */
                font-weight: 600;
                font-family: 'Segoe UI';
                background-color: transparent;
                border: none;
            }}
            
            /* Input field styling with visible borders and reduced padding */
            QLineEdit[class="input-field"], 
            QSpinBox[class="input-field"], 
            QDateEdit[class="input-field"], 
            QComboBox[class="input-field"] {{
                border: 2px solid {colors['border_light']};
                border-radius: 6px;  /* Reduced from 6px to match other updates */
                background-color: {'#3A3A54' if self.is_dark_mode else '#F8FAFC'};
                padding: 8px 10px;  /* Reduced from 10px 12px */
                font-size: 13px;  /* Reduced from 14px */
                color: {colors['text_primary']};
                margin: 1px;  /* Reduced from 2px */
            }}
            
            QLineEdit[class="input-field"]:focus, 
            QSpinBox[class="input-field"]:focus, 
            QDateEdit[class="input-field"]:focus, 
            QComboBox[class="input-field"]:focus {{
                border: 2px solid {colors['accent']};
                background-color: {'#404060' if self.is_dark_mode else '#FFFFFF'};
            }}
            
            QLineEdit[class="input-field"]:hover, 
            QSpinBox[class="input-field"]:hover, 
            QDateEdit[class="input-field"]:hover, 
            QComboBox[class="input-field"]:hover {{
                border: 2px solid {colors['accent_hover']};
            }}
            
            /* Dropdown styling for combo box and date edit with reduced size */
            QComboBox[class="input-field"]::drop-down {{
                border: none;
                border-left: 1px solid {colors['border']};
                border-radius: 0px 4px 4px 0px;  /* Reduced radius */
                background-color: {colors['secondary']};
                width: 18px;  /* Reduced from 20px */
            }}
            
            QComboBox[class="input-field"]::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            
            QComboBox[class="input-field"]::down-arrow {{
                border-left: 3px solid transparent;  /* Smaller arrow */
                border-right: 3px solid transparent;
                border-top: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            QDateEdit[class="input-field"]::drop-down {{
                border: none;
                border-left: 1px solid {colors['border']};
                border-radius: 0px 4px 4px 0px;  /* Reduced radius */
                background-color: {colors['secondary']};
                width: 18px;  /* Reduced from 20px */
            }}
            
            QDateEdit[class="input-field"]::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            
            QDateEdit[class="input-field"]::down-arrow {{
                border-left: 3px solid transparent;  /* Smaller arrow */
                border-right: 3px solid transparent;
                border-top: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            /* SpinBox buttons styling with reduced size */
            QSpinBox[class="input-field"]::up-button, QSpinBox[class="input-field"]::down-button {{
                border: none;
                background-color: {colors['secondary']};
                width: 18px;  /* Reduced from 20px */
            }}
            
            QSpinBox[class="input-field"]::up-button:hover, QSpinBox[class="input-field"]::down-button:hover {{
                background-color: {colors['accent']};
            }}
            
            QSpinBox[class="input-field"]::up-arrow {{
                border-left: 3px solid transparent;  /* Smaller arrow */
                border-right: 3px solid transparent;
                border-bottom: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            QSpinBox[class="input-field"]::down-arrow {{
                border-left: 3px solid transparent;  /* Smaller arrow */
                border-right: 3px solid transparent;
                border-top: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            /* Enhanced button styling with visible borders and reduced padding */
            QPushButton {{
                border: 2px solid transparent;
                border-radius: 6px;  /* Reduced from 8px */
                padding: 8px 18px;  /* Reduced from 12px 24px */
                font-weight: 600;
                font-size: 13px;  /* Reduced from 14px */
                min-height: 16px;  /* Reduced from 20px */
            }}
            
            QPushButton[class="primary"] {{
                background-color: {colors['accent']};
                color: white;
                border: 2px solid {colors['accent']};
            }}
            
            QPushButton[class="primary"]:hover {{
                background-color: {colors['accent_hover']};
                border: 2px solid {colors['accent_hover']};
            }}
            
            QPushButton[class="primary"]:pressed {{
                background-color: {colors['accent_pressed']};
                border: 2px solid {colors['accent_pressed']};
            }}
            
            QPushButton[class="danger"] {{
                background-color: {colors['danger']};
                color: white;
                border: 2px solid {colors['danger']};
            }}
            
            QPushButton[class="danger"]:hover {{
                background-color: #DC2626;
                border: 2px solid #DC2626;
            }}
            
            QPushButton[class="danger"]:pressed {{
                background-color: #B91C1C;
                border: 2px solid #B91C1C;
            }}
            
            QPushButton[class="secondary"] {{
                background-color: {colors['secondary']};
                color: {colors['text_primary']};
                border: 2px solid {colors['border']};
            }}
            
            QPushButton[class="secondary"]:hover {{
                background-color: {colors['hover']};
                border: 2px solid {colors['accent']};
            }}
            
            QPushButton[class="success"] {{
                background-color: {colors['success']};
                color: white;
                border: 2px solid {colors['success']};
            }}
            
            QPushButton[class="success"]:hover {{
                background-color: #059669;
                border: 2px solid #059669;
            }}
            
            /* Enhanced Metrics Styling with reduced padding */
            QLabel[class="metric"] {{
                font-size: 15px;  /* Reduced from 16px */
                font-weight: 600;
                padding: 12px 16px;  /* Reduced from 16px 20px */
                background-color: {colors['surface']};
                border: 2px solid {colors['border_light']};
                border-radius: 8px;  /* Reduced from 10px */
                color: {colors['text_primary']};
                text-align: center;
                margin: 2px;  /* Reduced from 4px */
            }}
            
            /* Base metric style with border */
            QLabel[class~="metric"] {{
                font-size: 15px;  /* Reduced from 16px */
                font-weight: 600;
                padding: 12px 16px;  /* Reduced from 16px 20px */
                background-color: {colors['surface']};
                border: 2px solid {colors['border_light']};
                border-radius: 8px;  /* Reduced from 10px */
                color: {colors['text_primary']};
                text-align: center;
                margin: 2px;  /* Reduced from 4px */
            }}
            
            /* Success metric styling */
            QLabel[class~="metric-success"] {{
                color: {colors['success']};
                border: 2px solid {colors['success']};
                background-color: {'rgba(16, 185, 129, 0.1)' if self.is_dark_mode else 'rgba(16, 185, 129, 0.05)'};
            }}
            
            /* Warning metric styling */
            QLabel[class~="metric-warning"] {{
                color: {colors['warning']};
                border: 2px solid {colors['warning']};
                background-color: {'rgba(245, 158, 11, 0.1)' if self.is_dark_mode else 'rgba(245, 158, 11, 0.05)'};
            }}
            
            /* Danger metric styling */
            QLabel[class~="metric-danger"] {{
                color: {colors['danger']};
                border: 2px solid {colors['danger']};
                background-color: {'rgba(239, 68, 68, 0.1)' if self.is_dark_mode else 'rgba(239, 68, 68, 0.05)'};
            }}
        """
        
        # Combine all styles
        complete_style = style + additional_styles
        self.setStyleSheet(complete_style)

    def accept_selected(self):
        """Collect selected orders and accept the dialog."""
        self.selected_orders = []
        for row in range(self.table.rowCount()):
            # Get the custom checkbox widget
            checkbox_widget = self.table.cellWidget(row, 0)
            if checkbox_widget and checkbox_widget.is_checked():
                self.selected_orders.append(self.orders[row])
        self.accept()

class InputPanel(QWidget):
    """Modern panel for user inputs and financial metrics display."""
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.portfolio_state_file = os.path.join(self.project_root, 'data', 'portfolio_state.json')
        self.is_dark_mode = True
        self.init_ui()
        self.set_default_values()
        self.set_date_constraints()
        self.update_date_tooltips()
        logger.info("InputPanel initialized")

    def init_ui(self):
        """Initialize the modern UI components with tighter spacing."""
        # Create scroll area for better responsiveness
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)  # Reduced from 16

        # Create main sections
        self.create_configuration_section(layout)
        self.create_action_buttons(layout)
        self.create_status_section(layout)
        self.create_metrics_section(layout)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Set content widget to scroll area
        scroll.setWidget(content_widget)
        
        # Main layout for this panel
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self.apply_styles()
        self.update_financial_metrics(0, 0)
        
        # Check if we should lock the investment field on startup
        self.lock_investment_field_after_first_trade()

    def create_configuration_section(self, main_layout):
        """Create the configuration input section with darker label backgrounds."""
        config_group = QGroupBox("Strategy Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(16, 20, 16, 20)  # Reduced from (20, 30, 20, 30)
        config_layout.setSpacing(10)  # Reduced from 15

        # Investment Amount with Add Funds button
        investment_container = QFrame()
        investment_layout = QVBoxLayout(investment_container)
        investment_layout.setContentsMargins(0, 0, 0, 0)
        investment_layout.setSpacing(0)
        
        # Dark label background frame
        investment_label_frame = QFrame()
        investment_label_frame.setProperty("class", "label-frame")
        investment_label_frame.setFixedHeight(28)
        investment_label_layout = QHBoxLayout(investment_label_frame)
        investment_label_layout.setContentsMargins(10, 6, 10, 6)
        
        self.investment_label = QLabel("Investment Amount ($):")  # Make it instance variable
        self.investment_label.setProperty("class", "label-dark")
        investment_label_layout.addWidget(self.investment_label)
        investment_label_layout.addStretch()
        
        # Input field with Add Funds button
        investment_input_frame = QFrame()
        investment_input_frame.setProperty("class", "input-frame")
        investment_input_layout = QHBoxLayout(investment_input_frame)
        investment_input_layout.setContentsMargins(10, 8, 10, 8)
        
        self.investment_input = QLineEdit("10000")
        self.investment_input.setPlaceholderText("Enter amount (e.g., 10000)")
        self.investment_input.setToolTip("Enter the initial investment amount in USD")
        self.investment_input.setProperty("class", "input-field")
        
        # Add Funds button (initially hidden)
        self.add_funds_button = QPushButton("Add Funds")
        self.add_funds_button.setProperty("class", "primary")
        self.add_funds_button.setVisible(False)
        self.add_funds_button.clicked.connect(self.add_funds_to_portfolio)
        self.add_funds_button.setMinimumHeight(32)
        self.add_funds_button.setMaximumHeight(32)
        
        investment_input_layout.addWidget(self.investment_input)
        investment_input_layout.addWidget(self.add_funds_button)
        
        investment_layout.addWidget(investment_label_frame)
        investment_layout.addWidget(investment_input_frame)
        config_layout.addWidget(investment_container)

        # Risk Level - with darker label background
        risk_container = QFrame()
        risk_layout = QVBoxLayout(risk_container)
        risk_layout.setContentsMargins(0, 0, 0, 0)
        risk_layout.setSpacing(0)
        
        # Dark label background frame
        risk_label_frame = QFrame()
        risk_label_frame.setProperty("class", "label-frame")
        risk_label_frame.setFixedHeight(28)  # Reduced from 35
        risk_label_layout = QHBoxLayout(risk_label_frame)
        risk_label_layout.setContentsMargins(10, 6, 10, 6)  # Reduced from (12, 8, 12, 8)
        
        risk_label = QLabel("Risk Level (0-10):")
        risk_label.setProperty("class", "label-dark")
        risk_label_layout.addWidget(risk_label)
        risk_label_layout.addStretch()
        
        # Input field
        risk_input_frame = QFrame()
        risk_input_frame.setProperty("class", "input-frame")
        risk_input_layout = QHBoxLayout(risk_input_frame)
        risk_input_layout.setContentsMargins(10, 8, 10, 8)  # Reduced from (12, 12, 12, 12)
        
        #self.risk_input = QSpinBox()
        self.risk_input = WheelDisabledSpinBox()
        self.risk_input.setRange(0, 10)
        self.risk_input.setValue(0)
        self.risk_input.setToolTip("Set risk level (0 = low risk, 10 = high risk)")
        self.risk_input.setProperty("class", "input-field")
        risk_input_layout.addWidget(self.risk_input)
        
        risk_layout.addWidget(risk_label_frame)
        risk_layout.addWidget(risk_input_frame)
        config_layout.addWidget(risk_container)

        # Date Range - with darker label backgrounds
        date_container = QFrame()
        date_layout = QVBoxLayout(date_container)
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(0)
        
        # Dark labels background frame
        date_label_frame = QFrame()
        date_label_frame.setProperty("class", "label-frame")
        date_label_frame.setFixedHeight(28)  # Reduced from 35
        date_label_layout = QHBoxLayout(date_label_frame)
        date_label_layout.setContentsMargins(10, 6, 10, 6)  # Reduced from (12, 8, 12, 8)
        
        start_label = QLabel("Start Date:")
        start_label.setProperty("class", "label-dark")
        end_label = QLabel("End Date:")
        end_label.setProperty("class", "label-dark")
        
        date_label_layout.addWidget(start_label)
        date_label_layout.addWidget(end_label)
        
        # Date inputs frame
        date_input_frame = QFrame()
        date_input_frame.setProperty("class", "input-frame")
        date_input_layout = QHBoxLayout(date_input_frame)
        date_input_layout.setContentsMargins(10, 8, 10, 8)  # Reduced from (12, 12, 12, 12)
        date_input_layout.setSpacing(8)  # Reduced from 10
        
        #self.start_date_input = QDateEdit()
        self.start_date_input = WheelDisabledDateEdit()
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDisplayFormat("yyyy-MM-dd")
        self.start_date_input.dateChanged.connect(self.update_end_date_minimum)
        self.start_date_input.setProperty("class", "input-field")
        
        #self.end_date_input = QDateEdit()
        self.end_date_input = WheelDisabledDateEdit()
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDisplayFormat("yyyy-MM-dd")
        self.end_date_input.setProperty("class", "input-field")
        
        date_input_layout.addWidget(self.start_date_input)
        date_input_layout.addWidget(self.end_date_input)
        
        date_layout.addWidget(date_label_frame)
        date_layout.addWidget(date_input_frame)
        config_layout.addWidget(date_container)

        # Trading Mode - with darker label background
        mode_container = QFrame()
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(0)
        
        # Dark label background frame
        mode_label_frame = QFrame()
        mode_label_frame.setProperty("class", "label-frame")
        mode_label_frame.setFixedHeight(28)  # Reduced from 35
        mode_label_layout = QHBoxLayout(mode_label_frame)
        mode_label_layout.setContentsMargins(10, 6, 10, 6)  # Reduced from (12, 8, 12, 8)
        
        mode_label = QLabel("Trading Mode:")
        mode_label.setProperty("class", "label-dark")
        mode_label_layout.addWidget(mode_label)
        mode_label_layout.addStretch()
        
        # Input field
        mode_input_frame = QFrame()
        mode_input_frame.setProperty("class", "input-frame")
        mode_input_layout = QHBoxLayout(mode_input_frame)
        mode_input_layout.setContentsMargins(10, 8, 10, 8)  # Reduced from (12, 12, 12, 12)
        
        #self.mode_combo = QComboBox()
        self.mode_combo = WheelDisabledComboBox()
        self.mode_combo.addItems(["Automatic", "Semi-Automatic"])
        self.mode_combo.setCurrentText("Automatic")
        self.mode_combo.setToolTip("Automatic: Execute trades automatically\nSemi-Automatic: Confirm trades manually")
        self.mode_combo.setProperty("class", "input-field")
        mode_input_layout.addWidget(self.mode_combo)
        
        mode_layout.addWidget(mode_label_frame)
        mode_layout.addWidget(mode_input_frame)
        config_layout.addWidget(mode_container)

        main_layout.addWidget(config_group)

        # Window Size (only for semi-automatic)
        window_container = QFrame()
        window_layout = QVBoxLayout(window_container)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(0)

        window_label_frame = QFrame()
        window_label_frame.setProperty("class", "label-frame")
        window_label_frame.setFixedHeight(28)
        window_label_layout = QHBoxLayout(window_label_frame)
        window_label_layout.setContentsMargins(10, 6, 10, 6)

        window_label = QLabel("Window Size (days):")
        window_label.setProperty("class", "label-dark")
        window_label_layout.addWidget(window_label)
        window_label_layout.addStretch()

        window_input_frame = QFrame()
        window_input_frame.setProperty("class", "input-frame")
        window_input_layout = QHBoxLayout(window_input_frame)
        window_input_layout.setContentsMargins(10, 8, 10, 8)

        self.window_size_input = WheelDisabledSpinBox()
        self.window_size_input.setRange(3, 7)
        self.window_size_input.setValue(5)
        self.window_size_input.setToolTip("Number of days per trading window (3-7 days recommended)")
        self.window_size_input.setProperty("class", "input-field")
        window_input_layout.addWidget(self.window_size_input)

        window_layout.addWidget(window_label_frame)
        window_layout.addWidget(window_input_frame)
        config_layout.addWidget(window_container)

        # Hide/show based on mode selection
        def on_mode_change():
            is_semi_auto = self.mode_combo.currentText() == "Semi-Automatic"
            window_container.setVisible(is_semi_auto)

        self.mode_combo.currentTextChanged.connect(on_mode_change)
        on_mode_change()  # Set initial visibility

    def set_default_values(self):
        """Set default values based on data availability."""
        try:
            # Set date range based on data
            if self.data_manager.data is not None and not self.data_manager.data.empty:
                # Try different possible column names for dates
                date_column = None
                for col in ['Date', 'date', 'DATE']:
                    if col in self.data_manager.data.columns:
                        date_column = col
                        break
                
                if date_column:
                    dates = pd.to_datetime(self.data_manager.data[date_column])
                    min_date = dates.min().date()
                    max_date = dates.max().date()
                    
                    # Set start date to minimum date from file
                    self.start_date_input.setDate(QDate(min_date.year, min_date.month, min_date.day))
                    # Set end date to maximum date from file
                    self.end_date_input.setDate(QDate(max_date.year, max_date.month, max_date.day))
                    
                    logger.info(f"Set default date range: {min_date} to {max_date}")
                else:
                    logger.warning("No date column found in data")
                    self.set_fallback_dates()
            else:
                logger.warning("No data available, using fallback dates")
                self.set_fallback_dates()
                
        except Exception as e:
            logger.error(f"Error setting default values: {e}")
            self.set_fallback_dates()

    def set_fallback_dates(self):
        """Set fallback dates when data is not available."""
        self.start_date_input.setDate(QDate(2021, 10, 18))
        self.end_date_input.setDate(QDate(2023, 12, 22))

    def set_date_constraints(self):
        """Set date constraints based on available data in the CSV file AND existing trades."""
        if self.data_manager and self.data_manager.data is not None:
            try:
                # Try different possible column names for dates
                date_column = None
                for col in ['Date', 'date', 'DATE']:
                    if col in self.data_manager.data.columns:
                        date_column = col
                        break
                
                if date_column:
                    dates = pd.to_datetime(self.data_manager.data[date_column])
                    dataset_min_date = dates.min().date()
                    dataset_max_date = dates.max().date()
                    
                    # Get the minimum allowable date based on existing trades
                    min_allowable_date = self.get_minimum_allowable_date(dataset_min_date)
                    
                    # Convert Python dates to QDate
                    q_min_date = QDate(min_allowable_date.year, min_allowable_date.month, min_allowable_date.day)
                    q_max_date = QDate(dataset_max_date.year, dataset_max_date.month, dataset_max_date.day)
                    
                    # Set date ranges - start date cannot go before last traded date
                    self.start_date_input.setDateRange(q_min_date, q_max_date)
                    self.end_date_input.setDateRange(q_min_date, q_max_date)
                    
                    # Update tooltips to explain the constraints
                    self.update_date_constraint_tooltips(min_allowable_date, dataset_max_date)
                    
                    logger.info(f"Set date constraints: minimum allowable={min_allowable_date}, max={dataset_max_date}")
                else:
                    logger.warning("No date column found for setting constraints")
                    
            except Exception as e:
                logger.error(f"Error setting date constraints: {e}")

    def get_minimum_allowable_date(self, dataset_min_date):
        """
        Get the minimum allowable trading date based on existing trade history.
        If there are existing trades, can only trade AFTER the last traded date.
        If no trades exist, can start from dataset minimum.
        """
        try:
            # Try to get existing orders
            orders = get_orders()
            
            if orders:
                # Find the latest trade date
                order_dates = []
                for order in orders:
                    if 'date' in order:
                        order_date = order['date']
                        if isinstance(order_date, str):
                            order_date = pd.to_datetime(order_date, utc=True)
                        elif isinstance(order_date, pd.Timestamp):
                            order_date = order_date if order_date.tz else order_date.tz_localize('UTC')
                        order_dates.append(order_date)
                
                if order_dates:
                    latest_trade_date = max(order_dates)
                    # Can only trade AFTER the last traded date (add 1 day)
                    min_allowable_date = (latest_trade_date + pd.Timedelta(days=1)).date()
                    
                    # Ensure it's not before dataset minimum
                    min_allowable_date = max(min_allowable_date, dataset_min_date)
                    
                    logger.info(f"Found existing trades. Latest trade: {latest_trade_date.date()}, min allowable: {min_allowable_date}")
                    return min_allowable_date
            
            # No existing trades, can start from dataset minimum
            logger.info(f"No existing trades found. Using dataset minimum: {dataset_min_date}")
            return dataset_min_date
            
        except Exception as e:
            logger.error(f"Error getting minimum allowable date: {e}")
            return dataset_min_date

    def update_date_constraint_tooltips(self, min_allowable_date, max_date):
        """Update tooltips to explain date constraints clearly."""
        try:
            # Check if there are existing orders to determine message
            orders = get_orders()
            
            if orders:
                # Find latest trade date for tooltip
                latest_trade = None
                for order in orders:
                    if 'date' in order:
                        order_date = order['date']
                        if isinstance(order_date, str):
                            order_date = pd.to_datetime(order_date, utc=True)
                        elif isinstance(order_date, pd.Timestamp):
                            order_date = order_date if order_date.tz else order_date.tz_localize('UTC')
                        if latest_trade is None or order_date > latest_trade:
                            latest_trade = order_date
                
                if latest_trade:
                    start_tooltip = (f"Select strategy start date\n"
                                f"Cannot trade on dates before {min_allowable_date}\n"
                                f"Last trade was executed on: {latest_trade.date()}\n"
                                f"Available range: {min_allowable_date} to {max_date}")
                    
                    end_tooltip = (f"Select strategy end date\n"
                                f"Must be after start date\n"
                                f"Available range: {min_allowable_date} to {max_date}")
                else:
                    start_tooltip = f"Select strategy start date\nAvailable range: {min_allowable_date} to {max_date}"
                    end_tooltip = f"Select strategy end date\nAvailable range: {min_allowable_date} to {max_date}"
            else:
                start_tooltip = f"Select strategy start date\nAvailable range: {min_allowable_date} to {max_date}"
                end_tooltip = f"Select strategy end date\nAvailable range: {min_allowable_date} to {max_date}"

            self.start_date_input.setToolTip(start_tooltip)
            self.end_date_input.setToolTip(end_tooltip)
            
        except Exception as e:
            logger.error(f"Error updating date constraint tooltips: {e}")

    def refresh_date_constraints_after_trade(self):
        """Refresh date constraints after a trade has been executed."""
        try:
            # Recalculate and apply date constraints
            self.set_date_constraints()
            
            # Get the new minimum allowable date
            if self.data_manager and self.data_manager.data is not None:
                date_column = None
                for col in ['Date', 'date', 'DATE']:
                    if col in self.data_manager.data.columns:
                        date_column = col
                        break
                
                if date_column:
                    dates = pd.to_datetime(self.data_manager.data[date_column])
                    dataset_min_date = dates.min().date()
                    min_allowable_date = self.get_minimum_allowable_date(dataset_min_date)
                    
                    # Update start date to the minimum allowable date
                    q_min_date = QDate(min_allowable_date.year, min_allowable_date.month, min_allowable_date.day)
                    self.start_date_input.setDate(q_min_date)
                    
                    # If end date is before minimum allowable, update it too
                    if self.end_date_input.date().toPyDate() < min_allowable_date:
                        # Set end date to a reasonable default (e.g., 30 days after min allowable)
                        default_end = min_allowable_date + pd.Timedelta(days=30).to_pytimedelta()
                        
                        # Make sure it doesn't exceed dataset maximum
                        dataset_max_date = dates.max().date()
                        if default_end > dataset_max_date:
                            default_end = dataset_max_date
                        
                        q_end_date = QDate(default_end.year, default_end.month, default_end.day)
                        self.end_date_input.setDate(q_end_date)
                    
                    logger.info(f"Refreshed date constraints after trade. New minimum: {min_allowable_date}")
            
        except Exception as e:
            logger.error(f"Error refreshing date constraints after trade: {e}")

    def create_action_buttons(self, main_layout):
        """Create the action buttons section with smaller, more compact buttons."""
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)  # Reduced from 10

        # Execute button with smaller size
        self.execute_button = QPushButton("Execute Strategy")
        self.execute_button.setProperty("class", "primary")
        self.execute_button.clicked.connect(self.execute_strategy)
        self.execute_button.setMinimumHeight(32)  # Reduced from 40
        self.execute_button.setMaximumHeight(32)  # Force consistent height
        
        # Try to load execute icon
        try:
            execute_icon = QIcon(resource_path("frontend/gui/icons/portfolio.png"))
            if not execute_icon.isNull():
                self.execute_button.setIcon(execute_icon)
        except:
            pass

        # Reset button with smaller size
        self.reset_button = QPushButton("Reset Portfolio")
        self.reset_button.setProperty("class", "danger")
        self.reset_button.clicked.connect(self.reset_portfolio)
        self.reset_button.setMinimumHeight(32)  # Reduced from 40
        self.reset_button.setMaximumHeight(32)  # Force consistent height
        
        # Try to load reset icon
        try:
            reset_icon = QIcon(resource_path("frontend/gui/icons/history.png"))
            if not reset_icon.isNull():
                self.reset_button.setIcon(reset_icon)
        except:
            pass

        button_layout.addWidget(self.execute_button, 2)
        button_layout.addWidget(self.reset_button, 1)

        main_layout.addWidget(button_frame)

    def create_status_section(self, main_layout):
        """Create a more compact status section."""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(16, 16, 16, 12)  # Reduced from (16, 20, 16, 16)
        status_layout.setSpacing(6)  # Reduced from 8

        self.status_label = QLabel("Ready to execute strategy")
        self.status_label.setProperty("class", "status-text")  # New smaller class
        status_layout.addWidget(self.status_label)

        # Progress bar (hidden by default) - make it smaller
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumHeight(16)  # Smaller progress bar
        status_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_group)

    def create_metrics_section(self, main_layout):
        """Create a more compact financial metrics section."""
        metrics_group = QGroupBox("Portfolio Overview")
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setContentsMargins(16, 16, 16, 12)  # Reduced from (16, 20, 16, 16)
        metrics_layout.setSpacing(8)  # Reduced from 12

        # Set column stretch for responsive layout
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnStretch(1, 1)
        metrics_layout.setColumnStretch(2, 1)

        # Create smaller metric cards
        self.cash_label = QLabel("Liquid Cash: N/A")
        self.cash_label.setProperty("class", "metric-compact")  # New compact class
        self.cash_label.setWordWrap(True)
        self.cash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.portfolio_label = QLabel("Portfolio Value: N/A")
        self.portfolio_label.setProperty("class", "metric-compact")  # New compact class
        self.portfolio_label.setWordWrap(True)
        self.portfolio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.total_label = QLabel("Total Value: N/A")
        self.total_label.setProperty("class", "metric-compact")  # New compact class
        self.total_label.setWordWrap(True)
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add widgets with proper spacing
        metrics_layout.addWidget(self.cash_label, 0, 0)
        metrics_layout.addWidget(self.portfolio_label, 0, 1)
        metrics_layout.addWidget(self.total_label, 0, 2)

        main_layout.addWidget(metrics_group)

    def apply_styles(self):
        """Apply modern styling to the panel with compact elements."""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        
        # Add custom styles for the dark label frames and compact elements
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        # Additional styles for compact layout
        additional_styles = f"""
            /* Label Frame Styling - Darker background for labels */
            QFrame[class="label-frame"] {{
                background-color: {'#252538' if self.is_dark_mode else '#9CA3AF'};
                border: 1px solid {colors['border']};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            
            /* Input Frame Styling - Matches the surface color */
            QFrame[class="input-frame"] {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-top: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            
            /* Dark label text styling */
            QLabel[class="label-dark"] {{
                color: {'#FFFFFF' if self.is_dark_mode else '#FFFFFF'};
                font-size: 13px;
                font-weight: 600;
                font-family: 'Segoe UI';
                background-color: transparent;
                border: none;
            }}
            
            /* Input field styling with visible borders */
            QLineEdit[class="input-field"], 
            QSpinBox[class="input-field"], 
            QDateEdit[class="input-field"], 
            QComboBox[class="input-field"] {{
                border: 2px solid {colors['border_light']};
                border-radius: 6px;
                background-color: {'#3A3A54' if self.is_dark_mode else '#F8FAFC'};
                padding: 8px 10px;
                font-size: 13px;
                color: {colors['text_primary']};
                margin: 1px;
            }}
            
            QLineEdit[class="input-field"]:focus, 
            QSpinBox[class="input-field"]:focus, 
            QDateEdit[class="input-field"]:focus, 
            QComboBox[class="input-field"]:focus {{
                border: 2px solid {colors['accent']};
                background-color: {'#404060' if self.is_dark_mode else '#FFFFFF'};
            }}
            
            QLineEdit[class="input-field"]:hover, 
            QSpinBox[class="input-field"]:hover, 
            QDateEdit[class="input-field"]:hover, 
            QComboBox[class="input-field"]:hover {{
                border: 2px solid {colors['accent_hover']};
            }}
            
            /* Compact Button Styling */
            QPushButton {{
                border: 2px solid transparent;
                border-radius: 6px;
                padding: 6px 14px;  /* Reduced padding for smaller buttons */
                font-weight: 600;
                font-size: 12px;  /* Smaller font for compact buttons */
                min-height: 14px;  /* Reduced min-height */
            }}
            
            QPushButton[class="primary"] {{
                background-color: {colors['accent']};
                color: white;
                border: 2px solid {colors['accent']};
            }}
            
            QPushButton[class="primary"]:hover {{
                background-color: {colors['accent_hover']};
                border: 2px solid {colors['accent_hover']};
            }}
            
            QPushButton[class="primary"]:pressed {{
                background-color: {colors['accent_pressed']};
                border: 2px solid {colors['accent_pressed']};
            }}
            
            QPushButton[class="danger"] {{
                background-color: {colors['danger']};
                color: white;
                border: 2px solid {colors['danger']};
            }}
            
            QPushButton[class="danger"]:hover {{
                background-color: #DC2626;
                border: 2px solid #DC2626;
            }}
            
            QPushButton[class="danger"]:pressed {{
                background-color: #B91C1C;
                border: 2px solid #B91C1C;
            }}
            
            /* Compact Status Text Styling */
            QLabel[class="status-text"] {{
                font-size: 13px;  /* Smaller status text */
                font-weight: 500;
                padding: 8px 12px;  /* Reduced padding */
                background-color: {colors['surface']};
                border: 1px solid {colors['border_light']};
                border-radius: 6px;
                color: {colors['text_primary']};
                text-align: center;
            }}
            
            /* Compact Metrics Styling */
            QLabel[class="metric-compact"] {{
                font-size: 13px;  /* Smaller metric font */
                font-weight: 600;
                padding: 8px 12px;  /* Reduced padding for compact metrics */
                background-color: {colors['surface']};
                border: 1px solid {colors['border_light']};  /* Thinner border */
                border-radius: 6px;  /* Smaller radius */
                color: {colors['text_primary']};
                text-align: center;
                margin: 1px;  /* Minimal margin */
            }}
            
            /* Compact metric success styling */
            QLabel[class~="metric-compact"][class~="metric-success"] {{
                color: {colors['success']};
                border: 1px solid {colors['success']};
                background-color: {'rgba(16, 185, 129, 0.1)' if self.is_dark_mode else 'rgba(16, 185, 129, 0.05)'};
            }}
            
            /* Compact metric warning styling */
            QLabel[class~="metric-compact"][class~="metric-warning"] {{
                color: {colors['warning']};
                border: 1px solid {colors['warning']};
                background-color: {'rgba(245, 158, 11, 0.1)' if self.is_dark_mode else 'rgba(245, 158, 11, 0.05)'};
            }}
            
            /* Compact metric danger styling */
            QLabel[class~="metric-compact"][class~="metric-danger"] {{
                color: {colors['danger']};
                border: 1px solid {colors['danger']};
                background-color: {'rgba(239, 68, 68, 0.1)' if self.is_dark_mode else 'rgba(239, 68, 68, 0.05)'};
            }}
            
            /* Smaller Progress Bar */
            QProgressBar {{
                background-color: {colors['secondary']};
                border: 1px solid {colors['border_light']};
                border-radius: 6px;
                text-align: center;
                font-weight: 500;
                font-size: 11px;  /* Smaller progress bar font */
                color: {colors['text_primary']};
                min-height: 16px;  /* Smaller progress bar height */
            }}
            
            QProgressBar::chunk {{
                background-color: {colors['accent']};
                border-radius: 4px;
                margin: 1px;
            }}
            
            /* Dropdown styling for combo box and date edit */
            QComboBox[class="input-field"]::drop-down {{
                border: none;
                border-left: 1px solid {colors['border']};
                border-radius: 0px 4px 4px 0px;
                background-color: {colors['secondary']};
                width: 18px;
            }}
            
            QComboBox[class="input-field"]::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            
            QComboBox[class="input-field"]::down-arrow {{
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            QDateEdit[class="input-field"]::drop-down {{
                border: none;
                border-left: 1px solid {colors['border']};
                border-radius: 0px 4px 4px 0px;
                background-color: {colors['secondary']};
                width: 18px;
            }}
            
            QDateEdit[class="input-field"]::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            
            QDateEdit[class="input-field"]::down-arrow {{
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            /* SpinBox buttons styling */
            QSpinBox[class="input-field"]::up-button, QSpinBox[class="input-field"]::down-button {{
                border: none;
                background-color: {colors['secondary']};
                width: 18px;
            }}
            
            QSpinBox[class="input-field"]::up-button:hover, QSpinBox[class="input-field"]::down-button:hover {{
                background-color: {colors['accent']};
            }}
            
            QSpinBox[class="input-field"]::up-arrow {{
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-bottom: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
            
            QSpinBox[class="input-field"]::down-arrow {{
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
            }}
        """
        
        # Combine all styles
        complete_style = style + additional_styles
        self.setStyleSheet(complete_style)

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        self.apply_styles()

    def update_date_tooltips(self):
        """Update date input tooltips based on data and order history."""
        try:
            min_date = None
            max_date = None
            latest_order_date = None

            if self.data_manager.data is not None and not self.data_manager.data.empty:
                # Try different possible column names for dates
                date_column = None
                for col in ['Date', 'date', 'DATE']:
                    if col in self.data_manager.data.columns:
                        date_column = col
                        break
                
                if date_column:
                    dates = pd.to_datetime(self.data_manager.data[date_column])
                    min_date = dates.min().date()
                    max_date = dates.max().date()
            else:
                logger.warning("No market data available for date tooltips")

            try:
                orders_df = get_order_history_df()
                if not orders_df.empty:
                    latest_order_date = pd.to_datetime(orders_df['date']).max().date()
            except:
                pass  # No orders yet

            start_tooltip = "Select strategy start date"
            end_tooltip = "Select strategy end date"

            if min_date:
                start_tooltip += f"\nEarliest available data: {min_date}"
            if max_date:
                end_tooltip += f"\nLatest available data: {max_date}"
            if latest_order_date:
                start_tooltip += f"\nLast trade executed: {latest_order_date}"

            self.start_date_input.setToolTip(start_tooltip)
            self.end_date_input.setToolTip(end_tooltip)
            logger.debug("Updated date tooltips")
        except Exception as e:
            logger.error(f"Error updating date tooltips: {e}")

    def update_end_date_minimum(self):
        """Ensure end_date is after start_date."""
        self.end_date_input.setMinimumDate(self.start_date_input.date())

    def show_message_box(self, icon, title, text, buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel):
        """Show a modern styled message box."""
        msg = QMessageBox(self)  # Set parent to ensure proper stacking
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        
        # Set window flags to keep on top
        msg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        
        # Apply modern styling
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                font-size: 14px;
                padding: 16px;
            }}
            QMessageBox QLabel {{
                color: {colors['text_primary']};
                padding: 16px;
            }}
            QMessageBox QPushButton {{
                background-color: {colors['accent']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                min-width: 80px;
                margin: 4px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {colors['accent_hover']};
            }}
        """)
        return msg.exec()

    def validate_inputs(self):
        """Validate user inputs with timeline protection checks."""
        try:
            # Check if investment field is locked (add funds mode)
            if not self.investment_input.isEnabled():
                # In add funds mode, allow empty field or validate additional amount
                if self.investment_input.text().strip():
                    investment_amount = float(self.investment_input.text().replace(',', ''))
                    if investment_amount <= 0:
                        raise ValueError("Additional amount must be greater than zero")
                else:
                    investment_amount = 0  # No additional funds
            else:
                # Normal investment mode
                investment_amount = float(self.investment_input.text().replace(',', ''))
                if investment_amount <= 0:
                    raise ValueError("Investment amount must be greater than zero")
                if investment_amount > 1000000000:  # 1 billion limit
                    raise ValueError("Investment amount seems too large")
                    
        except ValueError as e:
            self.show_message_box(
                QMessageBox.Icon.Warning,
                "Invalid Investment Amount",
                f"Please enter a valid investment amount.\n\nError: {e}",
                QMessageBox.StandardButton.Ok
            )
            return None

        risk_level = self.risk_input.value()
        start_date = self.start_date_input.date().toPyDate()
        end_date = self.end_date_input.date().toPyDate()

        if self.data_manager.data is None or self.data_manager.data.empty:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "No Data Available",
                "Failed to load the dataset. Please ensure the data file exists and is properly formatted.",
                QMessageBox.StandardButton.Ok
            )
            return None

        if (end_date - start_date).days < 1:
            self.show_message_box(
                QMessageBox.Icon.Warning,
                "Invalid Date Range",
                "End date must be after start date.",
                QMessageBox.StandardButton.Ok
            )
            return None

        # Timeline protection validation
        timeline_validation = self.validate_timeline_constraints(start_date, end_date)
        if not timeline_validation[0]:
            self.show_message_box(
                QMessageBox.Icon.Warning,
                "Timeline Constraint Violation",
                timeline_validation[1],
                QMessageBox.StandardButton.Ok
            )
            return None

        return investment_amount, risk_level, start_date, end_date

    def validate_timeline_constraints(self, start_date, end_date):
        """
        Validate that the selected dates don't violate timeline constraints.
        Returns (is_valid, error_message)
        """
        try:
            # Get existing orders
            orders = get_orders()
            
            if not orders:
                return True, "No timeline constraints - no previous trades"
            
            # Find all traded dates
            traded_dates = set()
            latest_trade_date = None
            
            for order in orders:
                if 'date' in order:
                    order_date = order['date']
                    if isinstance(order_date, str):
                        order_date = pd.to_datetime(order_date, utc=True)
                    elif isinstance(order_date, pd.Timestamp):
                        order_date = order_date if order_date.tz else order_date.tz_localize('UTC')
                    
                    trade_date = order_date.date()
                    traded_dates.add(trade_date)
                    
                    if latest_trade_date is None or trade_date > latest_trade_date:
                        latest_trade_date = trade_date
            
            if latest_trade_date:
                # Check if trying to trade before the last traded date
                if start_date <= latest_trade_date:
                    return False, (f"Cannot trade on or before {latest_trade_date}.\n"
                                f"You have already executed trades up to this date.\n"
                                f"Please select a start date after {latest_trade_date}.")
                
                # Check if the date range includes any previously traded dates
                current_date = start_date
                while current_date <= end_date:
                    if current_date in traded_dates:
                        return False, (f"Cannot trade on {current_date} - already traded on this date.\n"
                                    f"Please select a date range that doesn't include previously traded dates.")
                    current_date += pd.Timedelta(days=1).to_pytimedelta()
            
            return True, "Timeline constraints validated successfully"
            
        except Exception as e:
            logger.error(f"Error validating timeline constraints: {e}")
            return False, f"Error validating timeline constraints: {e}"

    def update_financial_metrics(self, cash=0, portfolio_value=0):
        """Update financial metrics display with color coding."""
        self.cash_label.setText(f"Liquid Cash: ${cash:,.2f}")
        portfolio_value_only = portfolio_value- cash
        self.portfolio_label.setText(f"Portfolio Value: ${portfolio_value_only:,.2f}")
        total_value = portfolio_value
        self.total_label.setText(f"Total Value: ${total_value:,.2f}")
        
        # Reset all labels to base metric class first
        self.cash_label.setProperty("class", "metric")
        self.portfolio_label.setProperty("class", "metric")
        self.total_label.setProperty("class", "metric")
        
        # Apply color coding based on performance
        if total_value > 10000:  # Assuming 10k initial
            self.total_label.setProperty("class", "metric metric-success")
        elif total_value < 9500:
            self.total_label.setProperty("class", "metric metric-danger")
        else:
            self.total_label.setProperty("class", "metric metric-warning")
        
        # Force style refresh by reapplying stylesheet
        current_style = self.styleSheet()
        self.setStyleSheet("")
        self.setStyleSheet(current_style)
        
        logger.debug(f"Updated financial metrics: Cash=${cash:,.2f}, Portfolio=${portfolio_value:,.2f}")

    def show_progress(self, message):
        """Show progress during strategy execution."""
        self.status_label.setText(message)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

    def hide_progress(self):
        """Hide progress indicators."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready to execute strategy")

    def execute_strategy(self):
        """Execute the trading strategy with enhanced semi-automated mode."""
        inputs = self.validate_inputs()
        if inputs is None:
            return

        investment_amount, risk_level, start_date, end_date = inputs
        mode = self.mode_combo.currentText().lower()

        # Check for short date range
        start_date = pd.Timestamp(start_date, tz='UTC')
        end_date = pd.Timestamp(end_date, tz='UTC')
        date_diff = (end_date - start_date).days
        
        if date_diff < 7:
            result = self.show_message_box(
                QMessageBox.Icon.Warning,
                "Short Date Range Warning",
                f"The selected date range is only {date_diff} days. For better trading results, "
                "a minimum one-week period is recommended.\n\nWould you like to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result == QMessageBox.StandardButton.No:
                logger.info("User cancelled execution due to short date range")
                return

        # Handle semi-automatic mode differently
        if mode == "semi-automatic":
            self._execute_windowed_semi_automatic(investment_amount, risk_level, start_date, end_date)
        else:
            self._execute_automatic_mode(investment_amount, risk_level, start_date, end_date)
    
    def _execute_windowed_semi_automatic(self, investment_amount, risk_level, start_date, end_date):
        """Execute windowed semi-automatic trading with proper portfolio state management."""
        # Check if this is continuing or new portfolio
        portfolio_history = get_portfolio_history()
        if portfolio_history:
            # Use current cash from portfolio
            current_cash = portfolio_history[-1].get('cash', investment_amount)
            reset_state = False
            logger.info(f"Continuing windowed trading with existing portfolio: ${current_cash:,.2f} cash")
        else:
            # New portfolio
            current_cash = investment_amount
            reset_state = True
            logger.info(f"Starting new windowed trading with: ${investment_amount:,.2f}")
        
        # Check if date range is too long for semi-automatic
        date_diff = (end_date - start_date).days
        
        if date_diff > 30:  # More than 30 days
            result = self.show_message_box(
                QMessageBox.Icon.Warning,
                "Long Date Range for Semi-Automatic",
                f"The selected date range is {date_diff} days long.\n"
                f"Semi-automatic mode works best with shorter periods (≤30 days).\n\n"
                f"This will create approximately {date_diff // 5} trading windows.\n"
                f"Would you like to proceed or reduce the date range?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result == QMessageBox.StandardButton.No:
                logger.info("User cancelled semi-automatic execution due to long date range")
                return
        
        # Only reset portfolio state if starting fresh
        if reset_state:
            try:
                if os.path.exists(self.portfolio_state_file):
                    os.remove(self.portfolio_state_file)
                    logger.info("Reset portfolio state for fresh semi-automated trading")
            except Exception as e:
                logger.warning(f"Could not reset portfolio state: {e}")
        
        # Create and start semi-automated manager
        self.semi_auto_manager = SemiAutomatedManager(self, self.main_window)
        self.semi_auto_manager.start_windowed_trading(
            current_cash, risk_level, start_date, end_date  # Use current_cash instead of investment_amount
        )

    def _execute_automatic_mode(self, investment_amount, risk_level, start_date, end_date):
        """Execute automatic mode using current portfolio cash or new investment."""
        
        # Check if this is the first trade or additional trade
        portfolio_history = get_portfolio_history()
        if portfolio_history:
            # Use current cash from portfolio, not new investment amount
            current_cash = portfolio_history[-1].get('cash', investment_amount)
            current_holdings = portfolio_history[-1].get('holdings', {})
            reset_state = False
            actual_investment = current_cash  # Use available cash
            logger.info(f"Continuing with existing portfolio: ${current_cash:,.2f} cash available")
        else:
            # First trade - use investment amount
            current_cash = investment_amount
            current_holdings = {}
            reset_state = True
            actual_investment = investment_amount
            logger.info(f"Starting new portfolio with: ${investment_amount:,.2f}")
        
        self.execute_button.setEnabled(False)
        self.show_progress("Initializing trading strategy...")

        self.thread = QThread()
        self.worker = Worker(
            investment_amount=actual_investment,  # Use actual available cash
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            data_manager=self.data_manager,
            mode="automatic",
            reset_state=reset_state,
            selected_orders=None,
            current_cash=current_cash,
            current_holdings=current_holdings
        )
        
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.show_progress)
        self.worker.finished.connect(self.handle_strategy_result)
        self.worker.error.connect(self.handle_strategy_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def handle_strategy_result(self, success, result):
        """Handle strategy execution results with modern UI feedback."""
        self.execute_button.setEnabled(True)
        self.hide_progress()
        
        mode = self.mode_combo.currentText().lower()
        
        if not success:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Strategy Execution Failed",
                f"The trading strategy failed to execute:\n\n{result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
            logger.error(f"Strategy failed: {result.get('warning_message')}")
            return

        orders = result.get('orders', [])
        portfolio_value = result.get('portfolio_value', 0)
        cash = result.get('cash', 0)
        warning_message = result.get('warning_message', '')

        if warning_message:
            self.show_message_box(
                QMessageBox.Icon.Information,
                "Strategy Completed with Warnings",
                warning_message,
                QMessageBox.StandardButton.Ok
            )

        if mode == "semi-automatic" and orders:
            dialog = TradeConfirmationDialog(orders, self)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_orders:
                self.execute_selected_trades(dialog.selected_orders)
            else:
                logger.info("User cancelled trade execution or no trades selected")
                self.update_financial_metrics(cash, portfolio_value)
        else:
            self.update_financial_metrics(cash, portfolio_value)
            log_trading_orders()
            if hasattr(self.main_window, 'update_dashboard'):
                self.main_window.update_dashboard()
            
            # Lock investment field after first successful trade
            if orders:  # Only if trades were actually executed
                self.lock_investment_field_after_first_trade()
                self.refresh_date_constraints_after_trade()
            
            # Show success message
            self.show_message_box(
                QMessageBox.Icon.Information,
                "Strategy Executed Successfully",
                f"Trading strategy completed successfully!\n\n"
                f"Orders executed: {len(orders)}\n"
                f"Final portfolio value: ${portfolio_value:,.2f}",
                QMessageBox.StandardButton.Ok
            )
            logger.info("Strategy execution completed successfully")

            if orders:  # Only if trades were actually executed
                self.update_ui_after_trading()

    def handle_semi_auto_result(self, success, result):
        """Handle semi-automatic trade execution results."""
        self.execute_button.setEnabled(True)
        self.hide_progress()
        
        if not success:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Trade Execution Failed",
                f"Failed to execute selected trades:\n\n{result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
            logger.error(f"Trade execution failed: {result.get('warning_message')}")
            return

        portfolio_value = result.get('portfolio_value', 0)
        cash = result.get('cash', 0)
        orders = result.get('orders', [])

        self.update_financial_metrics(cash, portfolio_value)
        log_trading_orders()
        
        if hasattr(self.main_window, 'update_dashboard'):
            self.main_window.update_dashboard()
        
        # NEW: Refresh date constraints after successful trade
        if orders:  # Only if trades were actually executed
            self.refresh_date_constraints_after_trade()
            
        # Show success message
        self.show_message_box(
            QMessageBox.Icon.Information,
            "Trades Executed Successfully",
            f"Selected trades executed successfully!\n\n"
            f"Orders completed: {len(orders)}\n"
            f"Current portfolio value: ${(cash + portfolio_value):,.2f}",
            QMessageBox.StandardButton.Ok
        )
        logger.info("Semi-automatic strategy execution completed successfully")

    def handle_window_semi_auto_result(self, success, result):
        """Handle semi-automatic trade execution results for windowed trading."""
        self.execute_button.setEnabled(True)
        self.hide_progress()
        
        if not success:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Window Trade Execution Failed",
                f"Failed to execute selected trades for this window:\n\n{result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
            logger.error(f"Window trade execution failed: {result.get('warning_message')}")
            return

        portfolio_value = result.get('portfolio_value', 0)
        cash = result.get('cash', 0)
        orders = result.get('orders', [])

        # Update financial metrics to show current state
        self.update_financial_metrics(cash, portfolio_value)
        
        # Log the orders but don't call log_trading_orders() as it might interfere
        logger.info(f"Window completed: {len(orders)} orders executed, Cash=${cash:,.2f}, Portfolio=${portfolio_value:,.2f}")
        
        # Update dashboard if available
        if hasattr(self.main_window, 'update_dashboard'):
            self.main_window.update_dashboard()
        
        # The SemiAutomatedManager will handle moving to the next window
        return success, result

    def execute_selected_trades(self, selected_orders):
        """Execute selected trades from semi-automatic mode."""
        # Re-run strategy with selected orders
        investment_amount = float(self.investment_input.text().replace(',', ''))
        risk_level = self.risk_input.value()
        start_date = pd.Timestamp(self.start_date_input.date().toPyDate(), tz='UTC')
        end_date = pd.Timestamp(self.end_date_input.date().toPyDate(), tz='UTC')

        self.execute_button.setEnabled(False)
        self.show_progress("Executing selected trades...")

        self.thread = QThread()
        self.worker = Worker(
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            data_manager=self.data_manager,
            mode="semi-automatic",
            reset_state=False,
            selected_orders=selected_orders
        )
        
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.show_progress)
        self.worker.finished.connect(self.handle_semi_auto_result)
        self.worker.error.connect(self.handle_strategy_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def handle_semi_auto_result(self, success, result):
        """Handle semi-automatic trade execution results."""
        self.execute_button.setEnabled(True)
        self.hide_progress()
        
        if not success:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Trade Execution Failed",
                f"Failed to execute selected trades:\n\n{result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
            logger.error(f"Trade execution failed: {result.get('warning_message')}")
            return

        portfolio_value = result.get('portfolio_value', 0)
        cash = result.get('cash', 0)
        orders = result.get('orders', [])

        self.update_financial_metrics(cash, portfolio_value)
        log_trading_orders()
        
        if hasattr(self.main_window, 'update_dashboard'):
            self.main_window.update_dashboard()
            
        # Show success message
        self.show_message_box(
            QMessageBox.Icon.Information,
            "Trades Executed Successfully",
            f"Selected trades executed successfully!\n\n"
            f"Orders completed: {len(orders)}\n"
            f"Current portfolio value: ${(cash + portfolio_value):,.2f}",
            QMessageBox.StandardButton.Ok
        )
        logger.info("Semi-automatic strategy execution completed successfully")

    def handle_strategy_error(self, error_message):
        """Handle errors from the worker thread."""
        self.execute_button.setEnabled(True)
        self.hide_progress()
        
        self.show_message_box(
            QMessageBox.Icon.Critical,
            "Unexpected Error",
            f"An unexpected error occurred during strategy execution:\n\n{error_message}",
            QMessageBox.StandardButton.Ok
        )
        logger.error(f"Unexpected error in worker thread: {error_message}")

    def reset_portfolio(self):
        """Reset portfolio state with confirmation."""
        result = self.show_message_box(
            QMessageBox.Icon.Question,
            "Reset Portfolio",
            "Are you sure you want to reset the portfolio?\n\n"
            "This will clear all trading history and return to initial state.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            try:
                # Reset the in-memory orders list FIRST
                from backend.trading_logic_new import reset_portfolio_for_semi_auto
                reset_portfolio_for_semi_auto()
                
                if os.path.exists(self.portfolio_state_file):
                    os.remove(self.portfolio_state_file)
                    logger.debug(f"Deleted {self.portfolio_state_file}")
                
                # Unlock investment field after reset
                self.unlock_investment_field_after_reset()
                
                # Reset financial metrics to default values
                investment_amount = float(self.investment_input.text().replace(',', '')) if self.investment_input.text() else 10000
                self.update_financial_metrics(investment_amount, investment_amount)
                
                # Refresh date constraints to allow trading from dataset beginning
                self.set_date_constraints()
                
                # Force update the start date to dataset minimum after constraints are reset
                if self.data_manager.data is not None and not self.data_manager.data.empty:
                    try:
                        # Get dataset start date
                        date_column = None
                        for col in ['Date', 'date', 'DATE']:
                            if col in self.data_manager.data.columns:
                                date_column = col
                                break
                        
                        if date_column:
                            dates = pd.to_datetime(self.data_manager.data[date_column])
                            dataset_min_date = dates.min().date()
                            dataset_max_date = dates.max().date()
                            
                            # Explicitly set the start date to dataset minimum
                            q_min_date = QDate(dataset_min_date.year, dataset_min_date.month, dataset_min_date.day)
                            q_max_date = QDate(dataset_max_date.year, dataset_max_date.month, dataset_max_date.day)
                            
                            # Force set the start date to the beginning of dataset
                            self.start_date_input.setDate(q_min_date)
                            self.end_date_input.setDate(q_max_date)
                            
                            logger.info(f"Reset start date to dataset minimum: {dataset_min_date}")
                        else:
                            # Fallback if no date column found
                            self.set_default_values()
                    except Exception as e:
                        logger.error(f"Error setting reset dates: {e}")
                        self.set_default_values()
                else:
                    # No data available, use fallback
                    self.set_default_values()
                
                # Update date tooltips to reflect the reset state
                self.update_date_tooltips()
                
                # Update dashboard and recommendations
                if hasattr(self.main_window, 'update_dashboard'):
                    self.main_window.update_dashboard()
                
                self.show_message_box(
                    QMessageBox.Icon.Information,
                    "Portfolio Reset",
                    "Portfolio has been reset successfully.\n\n"
                    "You can now start fresh with a new investment amount.",
                    QMessageBox.StandardButton.Ok
                )
                
                logger.info("Portfolio reset completed - investment field unlocked")
                
            except Exception as e:
                logger.error(f"Error resetting portfolio: {e}")
                self.show_message_box(
                    QMessageBox.Icon.Critical,
                    "Reset Failed",
                    f"Failed to reset portfolio:\n\n{e}",
                    QMessageBox.StandardButton.Ok
                )

    def get_current_settings(self):
        """Get current input panel settings as a dictionary."""
        return {
            'investment_amount': float(self.investment_input.text().replace(',', '')),
            'risk_level': self.risk_input.value(),
            'start_date': self.start_date_input.date().toPyDate(),
            'end_date': self.end_date_input.date().toPyDate(),
            'trading_mode': self.mode_combo.currentText()
        }

    def set_settings(self, settings):
        """Set input panel settings from a dictionary."""
        try:
            if 'investment_amount' in settings:
                self.investment_input.setText(str(settings['investment_amount']))
            if 'risk_level' in settings:
                self.risk_input.setValue(settings['risk_level'])
            if 'start_date' in settings:
                date_obj = settings['start_date']
                if isinstance(date_obj, str):
                    date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
                self.start_date_input.setDate(QDate(date_obj.year, date_obj.month, date_obj.day))
            if 'end_date' in settings:
                date_obj = settings['end_date']
                if isinstance(date_obj, str):
                    date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
                self.end_date_input.setDate(QDate(date_obj.year, date_obj.month, date_obj.day))
            if 'trading_mode' in settings:
                self.mode_combo.setCurrentText(settings['trading_mode'])
        except Exception as e:
            logger.error(f"Error setting input panel settings: {e}")

    def refresh_ui(self):
        """Refresh the UI elements and update constraints."""
        try:
            self.update_date_constraints()
            self.update_date_tooltips()
            self.apply_styles()
        except Exception as e:
            logger.error(f"Error refreshing UI: {e}")

    def is_valid_configuration(self):
        """Check if current configuration is valid."""
        try:
            investment_amount = float(self.investment_input.text().replace(',', ''))
            if investment_amount <= 0:
                return False, "Investment amount must be greater than zero"
            
            start_date = self.start_date_input.date().toPyDate()
            end_date = self.end_date_input.date().toPyDate()
            
            if end_date <= start_date:
                return False, "End date must be after start date"
            
            if (end_date - start_date).days < 1:
                return False, "Date range too short"
                
            return True, "Configuration is valid"
        except ValueError:
            return False, "Invalid investment amount"
        except Exception as e:
            return False, f"Configuration error: {e}"
        
    def add_funds_to_portfolio(self):
        """Show dialog to add funds to portfolio."""
        try:
            from PyQt6.QtWidgets import QInputDialog
            
            amount, ok = QInputDialog.getDouble(
                self, 
                "Add Funds", 
                "Enter amount to add to portfolio:", 
                value=0.0, 
                min=0.01, 
                max=1000000000.0, 
                decimals=2
            )
            
            if ok and amount > 0:
                # Get current portfolio state
                portfolio_history = get_portfolio_history()
                if portfolio_history:
                    current_cash = portfolio_history[-1].get('cash', 0)
                    current_value = portfolio_history[-1].get('value', 0)
                    
                    # Add cash using backend function
                    from backend.trading_logic_new import add_cash_to_portfolio
                    add_cash_to_portfolio(amount)
                    
                    new_cash = current_cash + amount
                    new_total_value = current_value + amount
                    
                    self.update_financial_metrics(new_cash, new_total_value)
                    
                    self.show_message_box(
                        QMessageBox.Icon.Information,
                        "Funds Added",
                        f"${amount:,.2f} added successfully!\n"
                        f"New cash balance: ${new_cash:,.2f}",
                        QMessageBox.StandardButton.Ok
                    )
                    logger.info(f"Added ${amount:,.2f} to portfolio")
                else:
                    self.show_message_box(
                        QMessageBox.Icon.Warning,
                        "No Portfolio Found",
                        "No existing portfolio found.",
                        QMessageBox.StandardButton.Ok
                    )
        except Exception as e:
            logger.error(f"Error adding funds: {e}")
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Failed to add funds: {e}",
                QMessageBox.StandardButton.Ok
            )

    def update_ui_after_trading(self):
        """Update UI state after any trading activity."""
        # Lock investment field and show add funds if there are orders
        orders = get_orders()
        if orders:
            self.lock_investment_field_after_first_trade()
            self.refresh_date_constraints_after_trade()

    def lock_investment_field_after_first_trade(self):
        """Lock investment field and show add funds functionality after first trade."""
        orders = get_orders()
        if orders:
            self.investment_input.setEnabled(False)
            self.investment_input.clear()
            self.investment_input.setPlaceholderText("Investment locked")
            self.add_funds_button.setVisible(True)
            self.investment_label.setText("Portfolio Funding:")
            logger.info("Investment field locked - showing add funds button")
            return True
        return False

    def unlock_investment_field_after_reset(self):
        """Unlock investment field after portfolio reset."""
        self.investment_input.setEnabled(True)
        self.investment_input.setPlaceholderText("Enter amount (e.g., 10000)")
        self.investment_input.setText("10000")
        self.add_funds_button.setVisible(False)
        self.investment_label.setText("Investment Amount ($):")
        logger.info("Investment field unlocked after reset")