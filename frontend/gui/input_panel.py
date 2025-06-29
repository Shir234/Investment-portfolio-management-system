import os
import pandas as pd
from datetime import datetime, date
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
                             QDateEdit, QPushButton, QComboBox, QMessageBox,
                             QDoubleSpinBox, QDialog, QTableWidget, QTableWidgetItem,
                             QCheckBox)
from PyQt6.QtCore import QDate, Qt, QThread, QObject, pyqtSignal
from frontend.logging_config import get_logger
from frontend.data.trading_connector import execute_trading_strategy, get_order_history_df, log_trading_orders
from backend.trading_logic_new import get_orders, get_portfolio_history

# Set up logging
logger = get_logger(__name__)

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
        self.setup_ui()

    def setup_ui(self):
        """Configure the dialog UI with a table and buttons."""
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setRowCount(len(self.orders))
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Select", "Date", "Ticker", "Action", "Shares", "Price"])
        self.table.horizontalHeader().setStretchLastSection(True)

        for row, order in enumerate(self.orders):
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.table.setCellWidget(row, 0, checkbox)

            self.table.setItem(row, 1, QTableWidgetItem(str(order.get('date', ''))))
            self.table.setItem(row, 2, QTableWidgetItem(order.get('ticker', '')))
            self.table.setItem(row, 3, QTableWidgetItem(order.get('action', '')))
            self.table.setItem(row, 4, QTableWidgetItem(str(order.get('shares_amount', 0))))
            self.table.setItem(row, 5, QTableWidgetItem(f"${order.get('price', 0):,.2f}"))

            for col in range(1, 6):
                if self.table.item(row, col):
                    self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        accept_button = QPushButton("Accept Selected")
        accept_button.clicked.connect(self.accept_selected)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(accept_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setStyleSheet("""
            QTableWidget {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QTableWidget::item {
                border: 1px solid #555555;
            }
            QTableWidget::item:selected {
                background-color: #2a82da;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #ffffff;
                border: 1px solid #555555;
            }
            QPushButton {
                background-color: #2a82da;
                color: #ffffff;
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3a92ea;
            }
        """)

    def accept_selected(self):
        """Collect selected orders and accept the dialog."""
        self.selected_orders = []
        for row in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(row, 0)
            if checkbox.isChecked():
                self.selected_orders.append(self.orders[row])
        self.accept()

class InputPanel(QWidget):
    """Panel for user inputs and financial metrics display."""
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.portfolio_state_file = os.path.join(self.project_root, 'data', 'portfolio_state.json')
        self.is_dark_mode = True
        self.init_ui()
        self.update_date_tooltips()
        logger.info("InputPanel initialized")

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)

        # Investment Amount
        investment_layout = QHBoxLayout()
        self.investment_label = QLabel("Investment Amount ($):")
        self.investment_label.setObjectName("investment_label")
        self.investment_input = QLineEdit("10000")
        self.investment_input.setPlaceholderText("Enter amount")
        investment_layout.addWidget(self.investment_label)
        investment_layout.addWidget(self.investment_input)
        layout.addLayout(investment_layout)

        # Risk Level
        risk_layout = QHBoxLayout()
        self.risk_label = QLabel("Risk Level (0-10):")
        self.risk_label.setObjectName("risk_label")
        self.risk_input = QDoubleSpinBox()
        self.risk_input.setRange(0, 10)
        self.risk_input.setValue(5)
        self.risk_input.setSingleStep(0.1)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_input)
        layout.addLayout(risk_layout)

        # Date Range
        date_layout = QHBoxLayout()
        self.start_label = QLabel("Start Date:")
        self.start_label.setObjectName("start_label")
        self.start_date_input = QDateEdit()
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDate(QDate(2021, 10, 18))
        self.start_date_input.dateChanged.connect(self.update_end_date_minimum)
        self.end_label = QLabel("End Date:")
        self.end_label.setObjectName("end_label")
        self.end_date_input = QDateEdit()
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setDate(QDate(2023, 12, 22))
        date_layout.addWidget(self.start_label)
        date_layout.addWidget(self.start_date_input)
        date_layout.addWidget(self.end_label)
        date_layout.addWidget(self.end_date_input)
        layout.addLayout(date_layout)

        # Mode Selection
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Trading Mode:")
        self.mode_label.setObjectName("mode_label")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Semi-Automatic"])
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute Trading Strategy")
        self.execute_button.setObjectName("execute_button")
        self.execute_button.clicked.connect(self.execute_strategy)
        self.reset_button = QPushButton("Reset Portfolio")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self.reset_portfolio)
        button_layout.addWidget(self.execute_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)

        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status_label")
        layout.addWidget(self.status_label)

        # Financial Metrics
        metrics_layout = QVBoxLayout()
        self.cash_label = QLabel("Liquid Cash: N/A")
        self.cash_label.setObjectName("cash_label")
        self.portfolio_label = QLabel("Portfolio Value: N/A")
        self.portfolio_label.setObjectName("portfolio_value_label")
        self.total_label = QLabel("Total Value: N/A")
        self.total_label.setObjectName("total_value_label")
        metrics_layout.addWidget(self.cash_label)
        metrics_layout.addWidget(self.portfolio_label)
        metrics_layout.addWidget(self.total_label)
        layout.addLayout(metrics_layout)

        self.setLayout(layout)
        self.set_theme(self.is_dark_mode)
        self.update_financial_metrics(0, 0)

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

            start_tooltip = "Select start date"
            end_tooltip = "Select end date"

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

    def show_message_box(self, icon, title, text, buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel):
        """Show a message box with the specified icon, title, text, and buttons."""
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        msg.setStyleSheet(self.get_message_box_style())
        return msg.exec_()

    def get_message_box_style(self):
        """Return stylesheet for QMessageBox based on the current theme."""
        if self.is_dark_mode:
            return (
                f"QMessageBox {{ background-color: #353535; color: #ffffff; }}"
                f"QMessageBox QLabel {{ color: #ffffff; }}"
                f"QPushButton {{ background-color: #2a82da; color: #ffffff; border: none; padding: 5px; border-radius: 3px; }}"
                f"QPushButton:hover {{ background-color: #3a92ea; }}"
            )
        else:
            return (
                f"QMessageBox {{ background-color: #f0f0f0; color: black; }}"
                f"QMessageBox QLabel {{ color: black; }}"
                f"QPushButton {{ background-color: #2a82da; color: #ffffff; border: none; padding: 5px; border-radius: 3px; }}"
                f"QPushButton:hover {{ background-color: #3a92ea; }}"
            )

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        if is_dark_mode:
            label_style = "color: #ffffff;"
            input_style = "background-color: #3c3f41; color: #ffffff; border: 1px solid #555555; border-radius: 3px;"
            combo_style = "background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da; border: 1px solid #555555; border-radius: 3px;"
            button_style_execute = "QPushButton {background-color: #2a82da; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #3a92ea;}"
            button_style_reset = "QPushButton {background-color: #ff4444; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #ff6666;}"
        else:
            label_style = "color: black;"
            input_style = "background-color: #ffffff; color: black; border: 1px solid #cccccc; border-radius: 3px;"
            combo_style = "background-color: #ffffff; color: black; selection-background-color: #2a82da; border: 1px solid #cccccc; border-radius: 3px;"
            button_style_execute = "QPushButton {background-color: #2a82da; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #3a92ea;}"
            button_style_reset = "QPushButton {background-color: #ff4444; color: #ffffff; border: none; padding: 5px; border-radius: 3px;} QPushButton:hover {background-color: #ff6666;}"

        self.investment_label.setStyleSheet(label_style)
        self.risk_label.setStyleSheet(label_style)
        self.start_label.setStyleSheet(label_style)
        self.end_label.setStyleSheet(label_style)
        self.mode_label.setStyleSheet(label_style)
        self.cash_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.portfolio_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.total_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.status_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.investment_input.setStyleSheet(input_style)
        self.risk_input.setStyleSheet(input_style)
        self.start_date_input.setStyleSheet(input_style)
        self.end_date_input.setStyleSheet(input_style)
        self.mode_combo.setStyleSheet(combo_style)
        self.execute_button.setStyleSheet(button_style_execute)
        self.reset_button.setStyleSheet(button_style_reset)

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
        """Update financial metrics display."""
        self.cash_label.setText(f"Liquid Cash: ${cash:,.2f}")
        self.portfolio_label.setText(f"Portfolio Value: ${portfolio_value:,.2f}")
        self.total_label.setText(f"Total Value: ${(cash + portfolio_value):,.2f}")
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
                "The selected date range is less than 7 days. For better trading results, a minimum one-week period is recommended. Proceed anyway?"
            )
            if result == QMessageBox.StandardButton.Cancel:
                logger.info("User cancelled execution due to short date range")
                return

        # Disable execute button and update status
        self.execute_button.setEnabled(False)
        self.status_label.setText("Executing strategy... Please wait.")

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
        portfolio_history = result.get('portfolio_history', [])
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

        portfolio_history = result.get('portfolio_history', [])
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
                f"Failed to reset portfolio: {e}",
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
                    if dialog.exec_() == QDialog.Accepted and dialog.selected_orders:
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