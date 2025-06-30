import os
import pandas as pd
from datetime import datetime, date
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
                             QDateEdit, QPushButton, QComboBox, QMessageBox,
                             QDoubleSpinBox, QDialog, QTableWidget, QTableWidgetItem,
                             QCheckBox, QFrame, QGridLayout, QSpacerItem, QSizePolicy,
                             QGroupBox, QProgressBar, QToolButton, QScrollArea)
from PyQt6.QtCore import QDate, Qt, QThread, QObject, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor
from frontend.logging_config import get_logger
from frontend.data.trading_connector import execute_trading_strategy, get_order_history_df, log_trading_orders
from backend.trading_logic_new import get_orders, get_portfolio_history
from frontend.gui.styles import ModernStyles

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
            # Create checkbox with larger size
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            # Make checkbox much larger with intuitive checkmark behavior
            checkbox.setStyleSheet("""
                QCheckBox {
                    font-size: 18px;
                    padding: 2px;
                    spacing: 2px;
                }
                QCheckBox::indicator {
                    width: 30px;
                    height: 30px;
                    border: 3px solid #ccc;
                    border-radius: 4px;
                    background-color: white;
                }
                QCheckBox::indicator:unchecked {
                    background-color: white;
                    border: 3px solid #ccc;
                }
                QCheckBox::indicator:unchecked:hover {
                    background-color: #f5f5f5;
                    border: 3px solid #999;
                }
                QCheckBox::indicator:checked {
                    background-color: #4CAF50;
                    border: 3px solid #4CAF50;
                    color: white;
                    font-weight: bold;
                    font-size: 20px;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #66bb6a;
                    border: 3px solid #66bb6a;
                }
            """)
            
            # Create a custom checkbox that shows ✓ when checked
            def create_custom_checkbox():
                widget = QWidget()
                layout = QHBoxLayout(widget)
                layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
                
                # Create a clickable label that acts as checkbox
                check_label = QLabel()
                check_label.setFixedSize(20, 20)  # Keep original size
                check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                check_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #ccc;
                        border-radius: 6px;
                        background-color: white;
                        font-size: 10px;
                        font-weight: bold;
                        color: white;
                        padding: 1px;
                    }
                """)
                
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
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Store reference to check state
                widget.is_checked = lambda: check_label.checked
                
                return widget
            
            checkbox_widget = create_custom_checkbox()
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

    # def apply_styles(self):
    #     """Apply modern styling to the dialog."""
    #     style = ModernStyles.get_complete_style(self.is_dark_mode)
    #     self.setStyleSheet(style)
    def apply_styles(self):
        """Apply modern styling to the panel with risk button styles."""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        
        # Add custom styles for risk buttons and input fixes
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        custom_styles = f"""
            /* Risk buttons */
            QToolButton[class="risk-button"] {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                color: {colors['text_primary']};
                font-size: 16px;
                font-weight: 600;
            }}
            QToolButton[class="risk-button"]:hover {{
                background-color: {colors['accent']};
                border-color: {colors['accent']};
                color: white;
            }}
            QToolButton[class="risk-button"]:pressed {{
                background-color: {colors['accent_hover']};
            }}
            
            /* Fix input field backgrounds to match overall background */
            QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox {{
                background-color: {colors['primary']};  /* Same as main background */
                color: {colors['text_primary']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 15px;  /* Slightly bigger font */
                font-weight: 600;  /* Bold font */
                min-width: 200px;
                min-height: 20px;
            }}
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QComboBox:focus {{
                border: 2px solid {colors['accent']};
                outline: none;
            }}
            
            /* Fix dropdown arrows to be proper arrows */
            QDateEdit::drop-down, QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left: 2px solid {colors['border']};
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background-color: {colors['secondary']};
            }}
            
            QDateEdit::down-arrow, QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid {colors['text_primary']};
                width: 0px;
                height: 0px;
                margin: 0px;
            }}
            
            QDateEdit::drop-down:hover, QComboBox::drop-down:hover {{
                background-color: {colors['accent']};
            }}
            
            QDateEdit::down-arrow:hover, QComboBox::down-arrow:hover {{
                border-top: 8px solid white;
            }}
            
            /* Fix ComboBox dropdown menu */
            QComboBox QAbstractItemView {{
                background-color: {colors['primary']};
                color: {colors['text_primary']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                selection-background-color: {colors['accent']};
                outline: none;
                font-weight: 600;
            }}
            
            /* Make labels bold and slightly bigger */
            QLabel[class="label"] {{
                color: {colors['text_primary']};
                font-size: 15px;  /* Slightly bigger */
                font-weight: 700;  /* Bold */
                margin-bottom: 8px;
            }}
            
            /* Fix Portfolio Overview metrics positioning */
            QLabel[class="metric"] {{
                font-size: 16px;
                font-weight: 600;
                padding: 12px 16px;
                background-color: {colors['surface']};
                border: 1px solid {colors['border_light']};
                border-radius: 8px;
                color: {colors['text_primary']};
                text-align: center;  /* Center the text */
                margin: 0px;  /* Remove any margins */
            }}
            
            QLabel[class="metric-success"] {{
                color: {colors['success']};
                border-left: 4px solid {colors['success']};
                text-align: center;
            }}
            
            QLabel[class="metric-warning"] {{
                color: {colors['warning']};
                border-left: 4px solid {colors['warning']};
                text-align: center;
            }}
            
            QLabel[class="metric-danger"] {{
                color: {colors['danger']};
                border-left: 4px solid {colors['danger']};
                text-align: center;
            }}
        """
        
        complete_style = style + custom_styles
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
        """Initialize the modern UI components with scroll area for responsiveness."""
        # Create scroll area for better responsiveness
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

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
        """Create the configuration input section with better spacing and responsiveness."""
        config_group = QGroupBox("Strategy Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setContentsMargins(20, 30, 20, 30)  # Added bottom margin
        config_layout.setVerticalSpacing(25)  # Increased vertical spacing even more
        config_layout.setHorizontalSpacing(20)  # Increased horizontal spacing

        # Set column stretch factors for responsive layout
        config_layout.setColumnStretch(0, 0)  # Labels don't stretch
        config_layout.setColumnStretch(1, 1)  # Inputs stretch

        # Investment Amount
        investment_label = QLabel("Investment Amount ($):")
        investment_label.setProperty("class", "label")
        self.investment_input = QLineEdit("10000")
        self.investment_input.setPlaceholderText("Enter amount in USD")
        self.investment_input.setMinimumWidth(200)
        self.investment_input.setMaximumWidth(45)
        
       # config_layout.addWidget(investment_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        config_layout.addWidget(investment_label, 0, 0, Qt.AlignmentFlag.AlignVCenter)  # Changed to AlignVCenter
        config_layout.addWidget(self.investment_input, 0, 1)

        # Risk Level with horizontal buttons and integer steps
        risk_label = QLabel("Risk Level (0-10):")
        risk_label.setProperty("class", "label")
        
        # Create risk level container with horizontal layout
        risk_container = QWidget()
        risk_layout = QHBoxLayout(risk_container)
        risk_layout.setContentsMargins(0, 0, 0, 0)
        risk_layout.setSpacing(10)
        
        self.risk_input = QDoubleSpinBox()
        self.risk_input.setRange(0, 10)
        self.risk_input.setValue(0)  # Default to 0
        self.risk_input.setSingleStep(1)  # Changed to integer steps
        self.risk_input.setDecimals(0)  # No decimal places for integers
        self.risk_input.setSuffix(" / 10")
        self.risk_input.setMinimumWidth(120)
        self.risk_input.setMaximumWidth(200)
        self.risk_input.setFixedHeight(45)  # Make input field smaller
        self.risk_input.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)  # Hide default buttons
        
        # Create custom horizontal buttons with better styling
        risk_down_btn = QToolButton()
        risk_down_btn.setText("−")
        risk_down_btn.setFixedSize(45, 45)  # Match input field height
        risk_down_btn.setProperty("class", "risk-button")
        risk_down_btn.clicked.connect(lambda: self.risk_input.stepDown())
        
        risk_up_btn = QToolButton()
        risk_up_btn.setText("+")
        risk_up_btn.setFixedSize(45, 45)  # Match input field height
        risk_up_btn.setProperty("class", "risk-button")
        risk_up_btn.clicked.connect(lambda: self.risk_input.stepUp())
        
        # Add widgets to risk layout
        risk_layout.addWidget(risk_down_btn)
        risk_layout.addWidget(self.risk_input)
        risk_layout.addWidget(risk_up_btn)
        risk_layout.addStretch()
        
        config_layout.addWidget(risk_label, 1, 0, Qt.AlignmentFlag.AlignVCenter)  # Changed to AlignVCenter
        config_layout.addWidget(risk_container, 1, 1)

        # Date Range with better spacing
        start_label = QLabel("Start Date:")
        start_label.setProperty("class", "label")
        self.start_date_input = QDateEdit()
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDisplayFormat("yyyy-MM-dd")
        self.start_date_input.setMinimumWidth(200)
        self.start_date_input.setMaximumWidth(400)
        self.start_date_input.setFixedHeight(45)  # Make input field smaller
        self.start_date_input.dateChanged.connect(self.update_end_date_minimum)
        
        config_layout.addWidget(start_label, 2, 0, Qt.AlignmentFlag.AlignVCenter)  # Changed to AlignVCenter
        config_layout.addWidget(self.start_date_input, 2, 1)

        end_label = QLabel("End Date:")
        end_label.setProperty("class", "label")
        self.end_date_input = QDateEdit()
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDisplayFormat("yyyy-MM-dd")
        self.end_date_input.setMinimumWidth(200)
        self.end_date_input.setMaximumWidth(400)
        self.end_date_input.setFixedHeight(45)  # Make input field smaller
        
        config_layout.addWidget(end_label, 3, 0, Qt.AlignmentFlag.AlignVCenter)  # Changed to AlignVCenter
        config_layout.addWidget(self.end_date_input, 3, 1)

        # Trading Mode
        mode_label = QLabel("Trading Mode:")
        mode_label.setProperty("class", "label")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Semi-Automatic"])
        self.mode_combo.setCurrentText("Automatic")
        self.mode_combo.setMinimumWidth(200)
        self.mode_combo.setMaximumWidth(400)
        self.mode_combo.setFixedHeight(45)  # Make input field smaller
        
        config_layout.addWidget(mode_label, 4, 0, Qt.AlignmentFlag.AlignVCenter)  # Changed to AlignVCenter
        config_layout.addWidget(self.mode_combo, 4, 1)

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
        """Create the action buttons section without emojis."""
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)

        # Execute button (no emoji)
        self.execute_button = QPushButton("Execute Trading Strategy")
        self.execute_button.setProperty("class", "primary")
        self.execute_button.clicked.connect(self.execute_strategy)
        self.execute_button.setMinimumHeight(50)

        # Reset button (no emoji)
        self.reset_button = QPushButton("Reset Portfolio")
        self.reset_button.setProperty("class", "danger")
        self.reset_button.clicked.connect(self.reset_portfolio)
        self.reset_button.setMinimumHeight(50)

        button_layout.addWidget(self.execute_button, 2)
        button_layout.addWidget(self.reset_button, 1)

        main_layout.addWidget(button_frame)

    def create_status_section(self, main_layout):
        """Create the status and progress section."""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(20, 30, 20, 20)
        status_layout.setSpacing(12)

        self.status_label = QLabel("Ready to execute strategy")
        self.status_label.setProperty("class", "metric")
        status_layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_group)

    def create_metrics_section(self, main_layout):
        """Create the financial metrics section without emojis."""
        metrics_group = QGroupBox("Portfolio Overview")
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setContentsMargins(20, 30, 20, 20)
        metrics_layout.setSpacing(16)

        # Set column stretch for responsive layout
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnStretch(1, 1)
        metrics_layout.setColumnStretch(2, 1)

        # Create metric cards without emojis - with proper alignment
        self.cash_label = QLabel("Liquid Cash: N/A")
        self.cash_label.setProperty("class", "metric")
        self.cash_label.setWordWrap(True)
        self.cash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align text
        
        self.portfolio_label = QLabel("Portfolio Value: N/A")
        self.portfolio_label.setProperty("class", "metric")
        self.portfolio_label.setWordWrap(True)
        self.portfolio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align text
        
        self.total_label = QLabel("Total Value: N/A")
        self.total_label.setProperty("class", "metric")
        self.total_label.setWordWrap(True)
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align text

        # Add widgets with proper spacing
        metrics_layout.addWidget(self.cash_label, 0, 0)
        metrics_layout.addWidget(self.portfolio_label, 0, 1)
        metrics_layout.addWidget(self.total_label, 0, 2)

        main_layout.addWidget(metrics_group)

    def apply_styles(self):
        """Apply modern styling to the dialog."""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        self.setStyleSheet(style)

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
        """Update financial metrics display with color coding (no emojis)."""
        self.cash_label.setText(f"Liquid Cash: ${cash:,.2f}")
        self.portfolio_label.setText(f"Portfolio Value: ${portfolio_value:,.2f}")
        total_value = cash + portfolio_value
        self.total_label.setText(f"Total Value: ${total_value:,.2f}")
        
        # Color coding based on performance
        if total_value > 10000:  # Assuming 10k initial
            self.total_label.setProperty("class", "metric-success")
        elif total_value < 9500:
            self.total_label.setProperty("class", "metric-danger")
        else:
            self.total_label.setProperty("class", "metric-warning")
            
        self.apply_styles()  # Refresh styles
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

    def update_portfolio(self):
        """Execute trading strategy and update portfolio metrics."""
        try:
            investment_amount = float(self.investment_input.text())
            risk_level = self.risk_input.value()
            start_date = pd.Timestamp(self.start_date_input.date().toPyDate(), tz='UTC')
            end_date = pd.Timestamp(self.end_date_input.date().toPyDate(), tz='UTC')

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
                if start_date <= latest_trade_date:
                    self.show_message_box(
                        QMessageBox.Icon.Critical,
                        "Invalid Date Range",
                        f"Start date must be after {latest_trade_date.date()} due to existing trades.\n"
                        f"Valid range: {latest_trade_date.date() + pd.Timedelta(days=1)} to {dataset_end.date() if dataset_end else 'today'}.",
                        QMessageBox.StandardButton.Ok
                    )
                    return

            if dataset_start and start_date < dataset_start:
                self.show_message_box(
                    QMessageBox.Icon.Critical,
                    "Invalid Date Range",
                    f"Start date cannot be before dataset start ({dataset_start.date()}).",
                    QMessageBox.StandardButton.Ok
                )
                return

            if dataset_end and end_date > dataset_end:
                self.show_message_box(
                    QMessageBox.Icon.Critical,
                    "Invalid Date Range",
                    f"End date cannot be after dataset end ({dataset_end.date()}).",
                    QMessageBox.StandardButton.Ok
                )
                return

            success, message = self.data_manager.set_date_range(start_date, end_date)
            if not success:
                self.show_message_box(
                    QMessageBox.Icon.Critical,
                    "Invalid Date Range",
                    message,
                    QMessageBox.StandardButton.Ok
                )
                return
            if message:
                self.show_message_box(
                    QMessageBox.Icon.Information,
                    "Date Range Adjusted",
                    message,
                    QMessageBox.StandardButton.Ok
                )

            logger.debug(f"Executing with risk_level={risk_level}, mode={self.mode_combo.currentText().lower()}")
            success, result = execute_trading_strategy(
                investment_amount=investment_amount,
                risk_level=risk_level,
                start_date=start_date,
                end_date=end_date,
                data_manager=self.data_manager,
                mode=self.mode_combo.currentText().lower(),
                reset_state=True
            )
            if success:
                portfolio_history = result.get('portfolio_history', [])
                portfolio_value = result.get('portfolio_value', investment_amount)
                cash = result.get('cash', investment_amount)
                orders = result.get('orders', [])
                warning_message = result.get('warning_message', '')
                correlation = result.get('signal_correlation', 0.0)
                buy_hit_rate = result.get('buy_hit_rate', 0.0)
                sell_hit_rate = result.get('sell_hit_rate', 0.0)

                signal_quality_message = (
                    f"Signal Quality Metrics:\n"
                    f"Correlation: {correlation:.3f}\n"
                    f"Buy Hit Rate: {buy_hit_rate:.1%}\n"
                    f"Sell Hit Rate: {sell_hit_rate:.1%}"
                )
                self.show_message_box(
                    QMessageBox.Icon.Information,
                    "Signal Quality",
                    signal_quality_message,
                    QMessageBox.StandardButton.Ok
                )

                if correlation < 0.1:
                    self.show_message_box(
                        QMessageBox.Icon.Warning,
                        "Low Signal Quality",
                        "Signal-return correlation is low. Strategy may be unreliable.",
                        QMessageBox.StandardButton.Ok
                    )

                if self.mode_combo.currentText().lower() == "semi-automatic" and orders:
                    dialog = TradeConfirmationDialog(orders, self)
                    if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_orders:
                        success, result = execute_trading_strategy(
                            investment_amount=investment_amount,
                            risk_level=risk_level,
                            start_date=start_date,
                            end_date=end_date,
                            data_manager=self.data_manager,
                            mode="semi-automatic",
                            reset_state=False,
                            selected_orders=dialog.selected_orders
                        )
                        if not success:
                            self.show_message_box(
                                QMessageBox.Icon.Critical,
                                "Error",
                                f"Failed to execute trades: {result.get('warning_message', 'Unknown error')}",
                                QMessageBox.StandardButton.Ok
                            )
                            return
                        portfolio_history = result.get('portfolio_history', [])
                        portfolio_value = result.get('portfolio_value', investment_amount)
                        cash = result.get('cash', investment_amount)
                        orders = result.get('orders', [])
                        warning_message = result.get('warning_message', '')

                if not orders and warning_message:
                    self.show_message_box(
                        QMessageBox.Icon.Warning,
                        "No Signals Detected",
                        warning_message,
                        QMessageBox.StandardButton.Ok
                    )

                self.update_financial_metrics(cash, portfolio_value)
                if hasattr(self, 'main_window'):
                    self.main_window.update_dashboard()
            else:
                error_message = result.get('warning_message', 'Unknown error')
                self.show_message_box(
                    QMessageBox.Icon.Critical,
                    "Error",
                    f"Failed to execute strategy: {error_message}",
                    QMessageBox.StandardButton.Ok
                )
                self.update_financial_metrics()
        except Exception as e:
            logger.error(f"Error in update_portfolio: {e}", exc_info=True)
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Failed to run strategy: {e}",
                QMessageBox.StandardButton.Ok
            )
            self.update_financial_metrics()

    def update_date_constraints_based_on_orders(self):
        """Update date constraints based on existing orders."""
        try:
            orders = get_orders()
            if orders:
                # Get the latest order date
                order_dates = pd.to_datetime([order['date'] for order in orders], utc=True)
                latest_order_date = order_dates.max()
                
                # Set minimum start date to day after latest order
                min_start_date = latest_order_date + pd.Timedelta(days=1)
                q_min_date = QDate(min_start_date.year, min_start_date.month, min_start_date.day)
                
                self.start_date_input.setMinimumDate(q_min_date)
                self.end_date_input.setMinimumDate(q_min_date)
                
                logger.info(f"Updated date constraints based on latest order: {latest_order_date.date()}")
        except Exception as e:
            logger.error(f"Error updating date constraints based on orders: {e}")

    def refresh_ui(self):
        """Refresh the UI elements and update constraints."""
        try:
            self.update_date_constraints()
            self.update_date_tooltips()
            self.apply_styles()
        except Exception as e:
            logger.error(f"Error refreshing UI: {e}")

    def get_current_settings(self):
        """Get current input panel settings as a dictionary."""
        return {
            'investment_amount': float(self.investment_input.text().replace(',', '')),
            'risk_level': int(self.risk_input.value()),
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

    def reset_to_defaults(self):
        """Reset all inputs to their default values."""
        try:
            self.investment_input.setText("10000")
            self.risk_input.setValue(0)
            self.mode_combo.setCurrentText("Automatic")
            self.set_default_values()  # Reset dates
            self.update_financial_metrics(0, 0)
            logger.info("Input panel reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")

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

    def enable_inputs(self, enabled=True):
        """Enable or disable all input fields."""
        self.investment_input.setEnabled(enabled)
        self.risk_input.setEnabled(enabled)
        self.start_date_input.setEnabled(enabled)
        self.end_date_input.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)

    def get_date_range_info(self):
        """Get information about the current date range."""
        start_date = self.start_date_input.date().toPyDate()
        end_date = self.end_date_input.date().toPyDate()
        days_diff = (end_date - start_date).days
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'days_difference': days_diff,
            'is_short_range': days_diff < 7,
            'is_valid_range': days_diff >= 1
        }