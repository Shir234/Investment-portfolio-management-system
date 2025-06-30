import os
import pandas as pd
from datetime import datetime, date
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QDoubleSpinBox, QDateEdit, QPushButton, QComboBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
                             QMessageBox, QDialog, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import QDate, Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QColor
from ..data.trading_connector import execute_trading_strategy, get_order_history_df, log_trading_orders
from ..logging_config import get_logger
from backend.trading_logic_new import get_orders, get_portfolio_history

# Configure logging
logger = get_logger(__name__)

# Centralized theme colors (matching front_main.py)
THEME_COLORS = {
    'dark': {
        'background': '#2D2D2D',
        'text': '#FFFFFF',
        'border': '#555555',
        'card': '#3C3F41',
        'highlight': '#0078D4',
        'hover': '#005BA1',
        'pressed': '#003E7E',
        'alternate': '#2A2A2A'
    },
    'light': {
        'background': '#F5F5F5',
        'text': '#2C2C2C',
        'border': '#CCCCCC',
        'card': '#FFFFFF',
        'highlight': '#0078D4',
        'hover': '#005BA1',
        'pressed': '#003E7E',
        'alternate': '#F0F0F0'
    }
}

class Worker(QObject):
    """Worker class to run execute_trading_strategy in a background thread."""
    finished = pyqtSignal(bool, dict)  # success, result
    error = pyqtSignal(str)  # error message

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
    """Dialog for confirming trades in semi-automatic mode."""
    def __init__(self, orders, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Trades")
        self.orders = orders
        self.selected_orders = []
        self.setFixedSize(600, 400)
        self.is_dark_mode = True  # Default to dark mode
        self.setup_ui()

    def setup_ui(self):
        """Configure the dialog UI with a table and buttons."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title with shadow effect
        self.title = QLabel("Confirm Trades")
        self.title.setStyleSheet(f"""
            font-size: 18px; 
            font-weight: bold; 
            font-family: 'Segoe UI'; 
            color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 128))
        shadow.setOffset(2, 2)
        self.title.setGraphicsEffect(shadow)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        # Table
        self.table = QTableWidget()
        self.table.setRowCount(len(self.orders))
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Select", "Date", "Ticker", "Action", "Shares", "Price"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setMinimumSectionSize(100)

        for row, order in enumerate(self.orders):
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.table.setCellWidget(row, 0, checkbox)
            self.table.setItem(row, 1, QTableWidgetItem(str(order.get('date', ''))))
            self.table.setItem(row, 2, QTableWidgetItem(order.get('ticker', '')))
            self.table.setItem(row, 3, QTableWidgetItem(order.get('action', '').upper()))
            shares = order.get('shares_amount', 0)
            self.table.setItem(row, 4, QTableWidgetItem(str(int(shares)) if shares == int(shares) else str(shares)))
            self.table.setItem(row, 5, QTableWidgetItem(f"${order.get('price', 0):,.2f}"))

            for col in range(1, 6):
                if self.table.item(row, col):
                    self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        accept_button = QPushButton("Confirm Trades")
        accept_button.setToolTip("Execute selected trades")
        accept_button.clicked.connect(self.accept_selected)
        cancel_button = QPushButton("Cancel")
        cancel_button.setToolTip("Cancel trade execution")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(accept_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Apply initial theme
        self.set_theme(self.is_dark_mode)

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the dialog."""
        self.is_dark_mode = is_dark_mode
        theme = THEME_COLORS['dark' if is_dark_mode else 'light']
        self.setStyleSheet(f"background-color: {theme['background']};")
        self.title.setStyleSheet(f"""
            font-size: 18px; 
            font-weight: bold; 
            font-family: 'Segoe UI'; 
            color: {theme['text']};
        """)
        table_style = f"""
            QTableWidget {{
                background-color: {theme['card']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
                border-radius: 4px;
                gridline-color: {theme['border']};
                alternate-background-color: {theme['alternate']};
            }}
            QTableWidget::item {{
                border: none;
                padding: 8px;
            }}
            QTableWidget::item:selected {{
                background-color: {theme['highlight']};
                color: #FFFFFF;
            }}
            QHeaderView::section {{
                background-color: {theme['card']};
                color: {theme['text']};
                padding: 8px;
                border: 1px solid {theme['border']};
            }}
        """
        button_style = f"""
            QPushButton {{
                background-color: {theme['highlight']};
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme['hover']};
            }}
            QPushButton:pressed {{
                background-color: {theme['pressed']};
            }}
        """
        self.table.setStyleSheet(table_style)
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        logger.debug(f"Applied theme to TradeConfirmationDialog: {'dark' if is_dark_mode else 'light'}")

    def accept_selected(self):
        """Collect selected orders and accept the dialog."""
        self.selected_orders = [self.orders[row] for row in range(self.table.rowCount()) if self.table.cellWidget(row, 0).isChecked()]
        self.accept()
        logger.debug(f"Selected {len(self.selected_orders)} orders for confirmation")

class InputPanel(QWidget):
    """Panel for user inputs and financial metrics display."""
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.portfolio_state_file = os.path.join(self.project_root, 'data', 'portfolio_state.json')
        self.is_dark_mode = True
        self.init_ui()
        self.update_date_tooltips()
        logger.info("InputPanel initialized")

    def init_ui(self):
        """Initialize the UI components with modern design."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title with shadow effect
        title = QLabel("Portfolio Setup")
        title.setStyleSheet(f"""
            font-size: 18px; 
            font-weight: bold; 
            font-family: 'Segoe UI'; 
            color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 128))
        shadow.setOffset(2, 2)
        title.setGraphicsEffect(shadow)
        layout.addWidget(title)

        # Investment Parameters Group
        investment_group = QGroupBox("Investment Parameters")
        investment_group_layout = QVBoxLayout(investment_group)
        investment_group_layout.setSpacing(10)

        # Investment Amount
        investment_layout = QVBoxLayout()
        self.investment_label = QLabel("Investment Amount ($)")
        self.investment_input = QLineEdit("10000")
        self.investment_input.setPlaceholderText("Enter amount (e.g., 10000)")
        self.investment_input.setToolTip("Enter the initial investment amount in USD")
        investment_layout.addWidget(self.investment_label)
        investment_layout.addWidget(self.investment_input)
        investment_group_layout.addLayout(investment_layout)

        # Risk Level
        risk_layout = QVBoxLayout()
        self.risk_label = QLabel("Risk Level (0-10)")
        self.risk_input = QDoubleSpinBox()
        self.risk_input.setRange(0, 10)
        self.risk_input.setValue(5)
        self.risk_input.setSingleStep(0.1)
        self.risk_input.setToolTip("Set risk level (0 = low risk, 10 = high risk)")
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_input)
        investment_group_layout.addLayout(risk_layout)
        layout.addWidget(investment_group)

        # Date Range Group
        date_group = QGroupBox("Date Range")
        date_group_layout = QHBoxLayout(date_group)
        date_group_layout.setSpacing(20)

        start_layout = QVBoxLayout()
        self.start_label = QLabel("Start Date")
        self.start_date_input = QDateEdit()
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDate(QDate(2021, 10, 18))
        self.start_date_input.dateChanged.connect(self.update_end_date_minimum)
        start_layout.addWidget(self.start_label)
        start_layout.addWidget(self.start_date_input)
        date_group_layout.addLayout(start_layout)

        end_layout = QVBoxLayout()
        self.end_label = QLabel("End Date")
        self.end_date_input = QDateEdit()
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDate(QDate(2023, 12, 22))
        end_layout.addWidget(self.end_label)
        end_layout.addWidget(self.end_date_input)
        date_group_layout.addLayout(end_layout)
        layout.addWidget(date_group)

        # Trading Mode Group
        mode_group = QGroupBox("Trading Mode")
        mode_group_layout = QVBoxLayout(mode_group)
        self.mode_label = QLabel("Trading Mode")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Semi-Automatic"])
        self.mode_combo.setToolTip("Automatic: Execute trades automatically\nSemi-Automatic: Confirm trades manually")
        mode_group_layout.addWidget(self.mode_label)
        mode_group_layout.addWidget(self.mode_combo)
        layout.addWidget(mode_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.execute_button = QPushButton("Execute Strategy")
        self.execute_button.setToolTip("Run the trading strategy with the specified parameters")
        self.execute_button.clicked.connect(self.execute_strategy)
        self.reset_button = QPushButton("Reset Portfolio")
        self.reset_button.setToolTip("Clear portfolio state and reset metrics")
        self.reset_button.clicked.connect(self.reset_portfolio)
        button_layout.addWidget(self.execute_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Metrics Group
        metrics_group = QGroupBox("Financial Metrics")
        metrics_group_layout = QVBoxLayout(metrics_group)
        metrics_group_layout.setSpacing(10)
        self.status_label = QLabel("Ready")
        self.cash_label = QLabel("Liquid Cash: N/A")
        self.portfolio_label = QLabel("Portfolio Value: N/A")
        self.total_label = QLabel("Total Value: N/A")
        metrics_group_layout.addWidget(self.status_label)
        metrics_group_layout.addWidget(self.cash_label)
        metrics_group_layout.addWidget(self.portfolio_label)
        metrics_group_layout.addWidget(self.total_label)
        layout.addWidget(metrics_group)

        layout.addStretch()
        self.set_theme(self.is_dark_mode)
        self.update_financial_metrics(0, 0)

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        theme = THEME_COLORS['dark' if is_dark_mode else 'light']
        group_style = f"""
            QGroupBox {{
                background-color: {theme['card']};
                border: 1px solid {theme['border']};
                border-radius: 8px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI';
                color: {theme['text']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: {theme['text']};
            }}
        """
        label_style = f"color: {theme['text']}; font-family: 'Segoe UI'; font-size: 14px;"
        bold_label_style = f"color: {theme['text']}; font-family: 'Segoe UI'; font-size: 14px; font-weight: bold;"
        input_style = f"""
            background-color: {theme['card']};
            color: {theme['text']};
            border: 1px solid {theme['border']};
            border-radius: 4px;
            padding: 6px;
        """
        button_style_execute = f"""
            QPushButton {{
                background-color: {theme['highlight']};
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme['hover']};
            }}
            QPushButton:pressed {{
                background-color: {theme['pressed']};
            }}
            QPushButton:disabled {{
                background-color: {'#555555' if is_dark_mode else '#CCCCCC'};
                color: {'#AAAAAA' if is_dark_mode else '#666666'};
            }}
        """
        button_style_reset = f"""
            QPushButton {{
                background-color: #D32F2F;
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #B71C1C;
            }}
            QPushButton:pressed {{
                background-color: #7F0000;
            }}
        """
        self.setStyleSheet(f"background-color: {theme['background']};")
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(group_style)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(8)
            shadow.setColor(QColor(0, 0, 0, 128))
            shadow.setOffset(2, 2)
            group.setGraphicsEffect(shadow)
        self.investment_label.setStyleSheet(bold_label_style)
        self.risk_label.setStyleSheet(bold_label_style)
        self.start_label.setStyleSheet(bold_label_style)
        self.end_label.setStyleSheet(bold_label_style)
        self.mode_label.setStyleSheet(bold_label_style)
        self.status_label.setStyleSheet(bold_label_style)
        self.cash_label.setStyleSheet(label_style)
        self.portfolio_label.setStyleSheet(label_style)
        # Handle non-numeric total_label text
        total_value_text = self.total_label.text().replace('Total Value: $', '').replace(',', '')
        try:
            total_value = float(total_value_text)
            total_color = '#4CAF50' if total_value > 0 else theme['text']
        except ValueError:
            total_color = theme['text']
        self.total_label.setStyleSheet(f"""
            color: {total_color}; 
            font-family: 'Segoe UI'; 
            font-size: 15px; 
            font-weight: bold;
        """)
        self.investment_input.setStyleSheet(input_style)
        self.risk_input.setStyleSheet(input_style)
        self.start_date_input.setStyleSheet(input_style)
        self.end_date_input.setStyleSheet(input_style)
        self.mode_combo.setStyleSheet(f"{input_style} QComboBox::drop-down {{ border: none; width: 20px; }}")
        self.execute_button.setStyleSheet(button_style_execute)
        self.reset_button.setStyleSheet(button_style_reset)
        logger.debug(f"Applied theme to InputPanel: {'dark' if is_dark_mode else 'light'}")

    def get_message_box_style(self):
        """Return stylesheet for QMessageBox based on the current theme."""
        theme = THEME_COLORS['dark' if self.is_dark_mode else 'light']
        return f"""
            QMessageBox {{
                background-color: {theme['background']};
                color: {theme['text']};
                font-family: 'Segoe UI';
                border: 1px solid {theme['border']};
                border-radius: 4px;
            }}
            QMessageBox QLabel {{
                color: {theme['text']};
            }}
            QMessageBox QPushButton {{
                background-color: {theme['highlight']};
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
            }}
            QMessageBox QPushButton:hover {{
                background-color: {theme['hover']};
            }}
            QMessageBox QPushButton:pressed {{
                background-color: {theme['pressed']};
            }}
        """

    def update_date_tooltips(self):
        """Update date input tooltips based on data and order history."""
        try:
            min_date = None
            max_date = None
            latest_order_date = None

            if self.data_manager.data is not None and not self.data_manager.data.empty:
                dates = pd.to_datetime(self.data_manager.data['date'])
                min_date = dates.min().date()
                max_date = dates.max().date()
            else:
                logger.warning("No market data available for date tooltips")

            orders_df = get_order_history_df()
            if not orders_df.empty:
                latest_order_date = pd.to_datetime(orders_df['date']).max().date()

            start_tooltip = "Select start date for trading period"
            end_tooltip = "Select end date for trading period"

            if min_date:
                start_tooltip += f"\nEarliest data: {min_date}"
                self.start_date_input.setMinimumDate(QDate(min_date.year, min_date.month, min_date.day))
            if max_date:
                end_tooltip += f"\nLatest data: {max_date}"
                self.end_date_input.setMaximumDate(QDate(max_date.year, max_date.month, max_date.day))
            if latest_order_date:
                start_tooltip += f"\nLatest order: {latest_order_date}"
                self.start_date_input.setMinimumDate(
                    QDate(latest_order_date.year, latest_order_date.month, latest_order_date.day)
                )

            self.start_date_input.setToolTip(start_tooltip)
            self.end_date_input.setToolTip(end_tooltip)
            logger.debug("Updated date tooltips")
        except Exception as e:
            logger.error(f"Error updating date tooltips: {e}")

    def update_date_constraints(self):
        """Set minimum dates based on existing trades and dataset."""
        orders = get_orders()
        dataset_start = self.data_manager.dataset_start_date
        dataset_end = self.data_manager.dataset_end_date

        if orders:
            order_dates = pd.to_datetime([order['date'] for order in orders], utc=True)
            latest_trade_date = order_dates.max()
            min_date = latest_trade_date + pd.Timedelta(days=1)
        elif dataset_start:
            min_date = dataset_start
        else:
            min_date = pd.Timestamp(datetime(2000, 1, 1), tz='UTC')

        max_date = dataset_end if dataset_end else pd.Timestamp(datetime.now(), tz='UTC')

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

    def update_end_date_minimum(self):
        """Ensure end_date is after start_date."""
        self.end_date_input.setMinimumDate(self.start_date_input.date())
        logger.debug("Updated end date minimum")

    def show_message_box(self, icon, title, text, buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel):
        """Show a message box with the specified icon, title, text, and buttons."""
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        msg.setStyleSheet(self.get_message_box_style())
        return msg.exec()

    def validate_inputs(self):
        """Validate user inputs."""
        try:
            investment_amount = float(self.investment_input.text())
            if investment_amount <= 0:
                raise ValueError("Investment amount must be positive")
        except ValueError as e:
            self.show_message_box(
                QMessageBox.Icon.Warning,
                "Invalid Input",
                f"Invalid investment amount: {e}",
                QMessageBox.StandardButton.Ok
            )
            return None

        try:
            risk_level = self.risk_input.value()
            if not 0 <= risk_level <= 10:
                raise ValueError("Risk level must be between 0 and 10")
        except ValueError as e:
            self.show_message_box(
                QMessageBox.Icon.Warning,
                "Invalid Input",
                f"Invalid risk level: {e}",
                QMessageBox.StandardButton.Ok
            )
            return None

        start_date = self.start_date_input.date().toPyDate()
        end_date = self.end_date_input.date().toPyDate()

        if self.data_manager.data is None or self.data_manager.data.empty:
            self.show_message_box(
                QMessageBox.Icon.Warning,
                "No Data",
                "Failed to load the dataset. Please ensure the data file exists.",
                QMessageBox.StandardButton.Ok
            )
            return None

        return investment_amount, risk_level, start_date, end_date

    def update_financial_metrics(self, cash=0, portfolio_value=0):
        """Update financial metrics display with color-coded total value."""
        self.cash_label.setText(f"Liquid Cash: ${cash:,.2f}")
        self.portfolio_label.setText(f"Portfolio Value: ${portfolio_value:,.2f}")
        self.total_label.setText(f"Total Value: ${(cash + portfolio_value):,.2f}")
        self.set_theme(self.is_dark_mode)  # Reapply theme to update total_label color
        logger.debug(f"Updated financial metrics: Cash=${cash:,.2f}, Portfolio=${portfolio_value:,.2f}")

    def execute_strategy(self):
        """Execute the trading strategy in a background thread with warning for short date ranges."""
        inputs = self.validate_inputs()
        if inputs is None:
            return

        investment_amount, risk_level, start_date, end_date = inputs
        mode = self.mode_combo.currentText().lower()

        # Check for date range less than 7 days
        start_date = pd.Timestamp(start_date, tz='UTC')
        end_date = pd.Timestamp(end_date, tz='UTC')
        date_diff = (end_date - start_date).days
        if date_diff < 7:
            result = self.show_message_box(
                QMessageBox.Icon.Warning,
                "Short Date Range",
                "The selected date range is less than 7 days. For better trading results, a minimum one-week period is recommended. Proceed anyway?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            if result == QMessageBox.StandardButton.Cancel:
                logger.info("User cancelled execution due to short date range")
                return

        # Disable execute button and update status
        self.execute_button.setEnabled(False)
        self.status_label.setText("Executing strategy... Please wait.")
        logger.debug("Starting strategy execution")

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
        self.worker.finished.connect(self.handle_strategy_result)
        self.worker.error.connect(self.handle_strategy_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def handle_strategy_result(self, success, result):
        """Handle the result of the trading strategy."""
        self.execute_button.setEnabled(True)
        self.status_label.setText("Ready")
        mode = self.mode_combo.currentText().lower()
        investment_amount = float(self.investment_input.text())
        risk_level = self.risk_input.value()
        start_date = pd.Timestamp(self.start_date_input.date().toPyDate(), tz='UTC')
        end_date = pd.Timestamp(self.end_date_input.date().toPyDate(), tz='UTC')

        if not success:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Strategy failed: {result.get('warning_message', 'Unknown error')}",
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
                QMessageBox.Icon.Warning,
                "Warning",
                warning_message,
                QMessageBox.StandardButton.Ok
            )

        if mode == "semi-automatic" and orders:
            dialog = TradeConfirmationDialog(orders, self)
            dialog.set_theme(self.is_dark_mode)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_orders:
                # Run semi-automatic execution in a new thread
                self.thread = QThread()
                self.worker = Worker(
                    investment_amount=investment_amount,
                    risk_level=risk_level,
                    start_date=start_date,
                    end_date=end_date,
                    data_manager=self.data_manager,
                    mode="semi-automatic",
                    reset_state=False,
                    selected_orders=dialog.selected_orders
                )
                self.worker.moveToThread(self.thread)
                self.thread.started.connect(self.worker.run)
                self.worker.finished.connect(self.handle_semi_auto_result)
                self.worker.error.connect(self.handle_strategy_error)
                self.worker.finished.connect(self.thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
                self.thread.finished.connect(self.thread.deleteLater)
                self.execute_button.setEnabled(False)
                self.status_label.setText("Executing semi-automatic trades... Please wait.")
                self.thread.start()
            else:
                logger.info("User cancelled trade execution or no trades selected")
                self.update_financial_metrics(cash, portfolio_value)
                return
        else:
            self.update_financial_metrics(cash, portfolio_value)
            log_trading_orders()
            if hasattr(self, 'main_window'):
                self.main_window.update_dashboard()
            logger.info("Strategy execution completed successfully")

    def handle_semi_auto_result(self, success, result):
        """Handle the result of semi-automatic trading."""
        self.execute_button.setEnabled(True)
        self.status_label.setText("Ready")
        investment_amount = float(self.investment_input.text())
        if not success:
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Failed to execute trades: {result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
            logger.error(f"Trade execution failed: {result.get('warning_message')}")
            return

        portfolio_value = result.get('portfolio_value', investment_amount)
        cash = result.get('cash', investment_amount)
        orders = result.get('orders', [])
        warning_message = result.get('warning_message', '')

        self.update_financial_metrics(cash, portfolio_value)
        log_trading_orders()
        if hasattr(self, 'main_window'):
            self.main_window.update_dashboard()
        logger.info("Semi-automatic strategy execution completed successfully")

    def handle_strategy_error(self, error_message):
        """Handle errors from the worker thread."""
        self.execute_button.setEnabled(True)
        self.status_label.setText("Ready")
        self.show_message_box(
            QMessageBox.Icon.Critical,
            "Error",
            f"Unexpected error: {error_message}",
            QMessageBox.StandardButton.Ok
        )
        logger.error(f"Unexpected error in worker thread: {error_message}")

    def reset_portfolio(self):
        """Delete portfolio_state.json to reset portfolio state."""
        try:
            if os.path.exists(self.portfolio_state_file):
                os.remove(self.portfolio_state_file)
                logger.debug(f"Deleted {self.portfolio_state_file}")
                self.show_message_box(
                    QMessageBox.Icon.Information,
                    "Success",
                    "Portfolio state reset successfully.",
                    QMessageBox.StandardButton.Ok
                )
            else:
                self.show_message_box(
                    QMessageBox.Icon.Information,
                    "Info",
                    "No portfolio state file to reset.",
                    QMessageBox.StandardButton.Ok
                )
            self.update_financial_metrics()
            if hasattr(self, 'main_window'):
                self.main_window.update_dashboard()
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")
            self.show_message_box(
                QMessageBox.Icon.Critical,
                "Error",
                f"Failed to reset portfolio: {str(e)}",
                QMessageBox.StandardButton.Ok
            )

    def update_portfolio(self):
        """Execute trading strategy and update portfolio metrics."""
        try:
            investment_amount = float(self.investment_input.text())
            risk_level = self.risk_input.value()
            start_date = pd.Timestamp(self.start_date_input.date().toPyDate(), tz='UTC')
            end_date = pd.Timestamp(self.end_date_input.date().toPyDate(), tz='UTC')

            orders = get_orders()
            dataset_start = self.data_manager.dataset_start_date
            dataset_end = self.data_manager.dataset_end_date

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
                    dialog.set_theme(self.is_dark_mode)
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
                f"Failed to run strategy: {str(e)}",
                QMessageBox.StandardButton.Ok
            )
            self.update_financial_metrics()