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
from .wheel_disabled_widgets import WheelDisabledSpinBox, WheelDisabledDateEdit, WheelDisabledComboBox


# Set up logging
logger = get_logger(__name__)

class Worker(QObject):
    """Worker class to run execute_trading_strategy in a background thread."""
    finished = pyqtSignal(bool, dict)  # success, result
    error = pyqtSignal(str)  # error message
    progress = pyqtSignal(str)  # progress message

    def __init__(self, investment_amount, risk_level, start_date, end_date, data_manager, mode, reset_state, selected_orders=None):
        super().__init__()
        self.investment_amount = investment_amount
        self.risk_level = risk_level
        self.start_date = start_date
        self.end_date = end_date
        self.data_manager = data_manager
        self.mode = mode
        self.reset_state = reset_state
        self.selected_orders = selected_orders

    def run(self):
        """Execute the trading strategy in the background."""
        try:
            self.progress.emit("Initializing strategy...")
            success, result = execute_trading_strategy(
                investment_amount=self.investment_amount,
                risk_level=self.risk_level,
                start_date=self.start_date,
                end_date=self.end_date,
                data_manager=self.data_manager,
                mode=self.mode,
                reset_state=self.reset_state,
                selected_orders=self.selected_orders
            )
            self.finished.emit(success, result)
        except Exception as e:
            logger.error(f"Error in Worker.run: {e}", exc_info=True)
            self.error.emit(str(e))

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

    def create_configuration_section(self, main_layout):
        """Create the configuration input section with darker label backgrounds."""
        config_group = QGroupBox("Strategy Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(16, 20, 16, 20)  # Reduced from (20, 30, 20, 30)
        config_layout.setSpacing(10)  # Reduced from 15

        # Investment Amount - with darker label background
        investment_container = QFrame()
        investment_layout = QVBoxLayout(investment_container)
        investment_layout.setContentsMargins(0, 0, 0, 0)
        investment_layout.setSpacing(0)
        
        # Dark label background frame
        investment_label_frame = QFrame()
        investment_label_frame.setProperty("class", "label-frame")
        investment_label_frame.setFixedHeight(28)  # Reduced from 35
        investment_label_layout = QHBoxLayout(investment_label_frame)
        investment_label_layout.setContentsMargins(10, 6, 10, 6)  # Reduced from (12, 8, 12, 8)
        
        investment_label = QLabel("Investment Amount ($):")
        investment_label.setProperty("class", "label-dark")
        investment_label_layout.addWidget(investment_label)
        investment_label_layout.addStretch()
        
        # Input field
        investment_input_frame = QFrame()
        investment_input_frame.setProperty("class", "input-frame")
        investment_input_layout = QHBoxLayout(investment_input_frame)
        investment_input_layout.setContentsMargins(10, 8, 10, 8)  # Reduced from (12, 12, 12, 12)
        
        self.investment_input = QLineEdit("10000")
        self.investment_input.setPlaceholderText("Enter amount (e.g., 10000)")
        self.investment_input.setToolTip("Enter the initial investment amount in USD")
        self.investment_input.setProperty("class", "input-field")
        investment_input_layout.addWidget(self.investment_input)
        
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
        """Set date constraints based on available data in the CSV file."""
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
                    min_date = dates.min().date()
                    max_date = dates.max().date()
                    
                    # Convert Python dates to QDate
                    q_min_date = QDate(min_date.year, min_date.month, min_date.day)
                    q_max_date = QDate(max_date.year, max_date.month, max_date.day)
                    
                    # Set date ranges
                    self.start_date_input.setDateRange(q_min_date, q_max_date)
                    self.end_date_input.setDateRange(q_min_date, q_max_date)
                    
                    logger.info(f"Set date constraints: {min_date} to {max_date}")
                else:
                    logger.warning("No date column found for setting constraints")
                    
            except Exception as e:
                logger.error(f"Error setting date constraints: {e}")

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

    # Updated apply_styles method for input_panel.py

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
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        
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
        """Validate user inputs with better error messages."""
        try:
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

        return investment_amount, risk_level, start_date, end_date


    def update_financial_metrics(self, cash=0, portfolio_value=0):
        """Update financial metrics display with color coding."""
        self.cash_label.setText(f"Liquid Cash: ${cash:,.2f}")
        self.portfolio_label.setText(f"Portfolio Value: ${portfolio_value:,.2f}")
        total_value = cash + portfolio_value
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
        """Execute the trading strategy with modern UI feedback."""
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

        # Show progress and disable controls
        self.execute_button.setEnabled(False)
        self.show_progress("Initializing trading strategy...")

        # Set up worker and thread
        self.thread = QThread()
        self.worker = Worker(
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            data_manager=self.data_manager,
            mode=mode,
            reset_state=True,
            selected_orders=None
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
        investment_amount = float(self.investment_input.text().replace(',', ''))
        
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
        portfolio_value = result.get('portfolio_value', investment_amount)
        cash = result.get('cash', investment_amount)
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
            
            # Show success message
            self.show_message_box(
                QMessageBox.Icon.Information,
                "Strategy Executed Successfully",
                f"Trading strategy completed successfully!\n\n"
                f"Orders executed: {len(orders)}\n"
                f"Final portfolio value: ${(cash + portfolio_value):,.2f}",
                QMessageBox.StandardButton.Ok
            )
            logger.info("Strategy execution completed successfully")

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
                if os.path.exists(self.portfolio_state_file):
                    os.remove(self.portfolio_state_file)
                    logger.debug(f"Deleted {self.portfolio_state_file}")
                
                self.update_financial_metrics()
                
                if hasattr(self.main_window, 'update_dashboard'):
                    self.main_window.update_dashboard()
                
                self.show_message_box(
                    QMessageBox.Icon.Information,
                    "Portfolio Reset",
                    "Portfolio has been reset successfully.",
                    QMessageBox.StandardButton.Ok
                )
                
            except Exception as e:
                logger.error(f"Error resetting portfolio: {e}")
                self.show_message_box(
                    QMessageBox.Icon.Critical,
                    "Reset Failed",
                    f"Failed to reset portfolio:\n\n{e}",
                    QMessageBox.StandardButton.Ok
                )

    # Additional utility methods for compatibility and functionality
    def update_date_constraints(self):
        """Set minimum dates based on existing trades and dataset."""
        try:
            orders = get_orders()
            
            # Get dataset dates
            if self.data_manager.data is not None and not self.data_manager.data.empty:
                # Try different possible column names for dates
                date_column = None
                for col in ['Date', 'date', 'DATE']:
                    if col in self.data_manager.data.columns:
                        date_column = col
                        break
                
                if date_column:
                    dates = pd.to_datetime(self.data_manager.data[date_column])
                    dataset_start = dates.min()
                    dataset_end = dates.max()
                else:
                    dataset_start = pd.Timestamp(datetime(2021, 1, 1), tz='UTC')
                    dataset_end = pd.Timestamp(datetime(2023, 12, 31), tz='UTC')
            else:
                dataset_start = pd.Timestamp(datetime(2021, 1, 1), tz='UTC')
                dataset_end = pd.Timestamp(datetime(2023, 12, 31), tz='UTC')

            if orders:
                order_dates = pd.to_datetime([order['date'] for order in orders], utc=True)
                latest_trade_date = order_dates.max()
                min_date = latest_trade_date + pd.Timedelta(days=1)
            else:
                min_date = dataset_start

            max_date = dataset_end

            self.start_date_input.setMinimumDate(QDate(min_date.year, min_date.month, min_date.day))
            self.start_date_input.setMaximumDate(QDate(max_date.year, max_date.month, max_date.day))
            self.end_date_input.setMinimumDate(QDate(min_date.year, min_date.month, min_date.day))
            self.end_date_input.setMaximumDate(QDate(max_date.year, max_date.month, max_date.day))

            self.start_date_input.setToolTip(
                f"Select a date between {min_date.date()} and {max_date.date()}"
            )
            self.end_date_input.setToolTip(
                f"Select a date after start date and before {max_date.date()}"
            )
        except Exception as e:
            logger.error(f"Error updating date constraints: {e}")

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