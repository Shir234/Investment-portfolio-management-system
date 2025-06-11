import os
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, 
                             QDateEdit, QPushButton, QComboBox, QMessageBox, 
                             QVBoxLayout, QDoubleSpinBox)
from PyQt5.QtCore import QDate
from backend.trading_logic_new import get_orders
from datetime import datetime
import pandas as pd
import logging
from logging_config import get_logger
from data.trading_connector import execute_trading_strategy
from backend.trading_logic_new import get_portfolio_history
# Set up logging
logger = get_logger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class InputPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.is_dark_mode = True
        self.portfolio_state_file = 'data/portfolio_state.json'
        self.setup_ui()
        self.update_date_constraints()  # Initialize date constraints
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Investment input
        investment_layout = QHBoxLayout()
        self.investment_label = QLabel("Investment Amount ($):")
        self.investment_label.setObjectName("investment_label")
        self.investment_label.setStyleSheet("color: #ffffff;")
        self.investment_input = QLineEdit("10000")
        self.investment_input.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        investment_layout.addWidget(self.investment_label)
        investment_layout.addWidget(self.investment_input)
        layout.addLayout(investment_layout)
        
        # Risk level input
        risk_layout = QHBoxLayout()
        self.risk_label = QLabel("Risk Level (0-10):")
        self.risk_label.setObjectName("risk_label")
        self.risk_label.setStyleSheet("color: #ffffff;")
        self.risk_input = QDoubleSpinBox()
        self.risk_input.setRange(0, 10)
        self.risk_input.setValue(10)
        self.risk_input.setSingleStep(0.1)
        self.risk_input.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_input)
        layout.addLayout(risk_layout)
        
        # Date input
        date_layout = QHBoxLayout()
        self.start_label = QLabel("Start Date:")
        self.start_label.setObjectName("start_label")
        self.start_label.setStyleSheet("color: #ffffff;")
        self.start_date = QDateEdit()
        self.start_date.setDate(datetime(2021, 10, 18))
        self.start_date.setCalendarPopup(True)
        self.start_date.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        self.start_date.setToolTip("Select a date after the latest trade or dataset start.")
        self.end_label = QLabel("End Date:")
        self.end_label.setObjectName("end_label")
        self.end_label.setStyleSheet("color: #ffffff;")
        self.end_date = QDateEdit()
        self.end_date.setDate(datetime(2023, 12, 22))
        self.end_date.setCalendarPopup(True)
        self.end_date.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        self.end_date.setToolTip("Select a date after the start date and within dataset range.")
        date_layout.addWidget(self.start_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(self.end_label)
        date_layout.addWidget(self.end_date)
        layout.addLayout(date_layout)
        
        # Trading mode
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Trading Mode:")
        self.mode_label.setObjectName("mode_label")
        self.mode_label.setStyleSheet("color: #ffffff;")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Semi-automatic"])
        self.mode_combo.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute Trading Strategy")
        self.execute_button.setObjectName("execute_button")
        self.execute_button.clicked.connect(self.update_portfolio)
        self.execute_button.setStyleSheet("background-color: #2a82da; color: #ffffff;")
        self.reset_button = QPushButton("Reset Portfolio")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self.reset_portfolio)
        self.reset_button.setStyleSheet("background-color: #ff4444; color: #ffffff;")
        button_layout.addWidget(self.execute_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        # Financial metrics
        metrics_layout = QVBoxLayout()
        self.cash_label = QLabel("Liquid Cash: N/A")
        self.cash_label.setObjectName("cash_label")
        self.cash_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.portfolio_value_label = QLabel("Portfolio Value: N/A")
        self.portfolio_value_label.setObjectName("portfolio_value_label")
        self.portfolio_value_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.total_value_label = QLabel("Total Value: N/A")
        self.total_value_label.setObjectName("total_value_label")
        self.total_value_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        metrics_layout.addWidget(self.cash_label)
        metrics_layout.addWidget(self.portfolio_value_label)
        metrics_layout.addWidget(self.total_value_label)
        layout.addLayout(metrics_layout)
        
        self.update_financial_metrics()
    
    def update_financial_metrics(self):
        """Update financial metrics labels based on portfolio history."""
        portfolio_history = get_portfolio_history()
        if portfolio_history:
            latest_entry = portfolio_history[-1]
            cash = latest_entry.get('cash', 0.0)
            portfolio_value = latest_entry.get('value', 0.0)
            total_value = cash + portfolio_value
            self.cash_label.setText(f"Liquid Cash: ${cash:,.2f}")
            self.portfolio_value_label.setText(f"Portfolio Value: ${portfolio_value:,.2f}")
            self.total_value_label.setText(f"Total Value: ${total_value:,.2f}")
        else:
            self.cash_label.setText("Liquid Cash: N/A")
            self.portfolio_value_label.setText("Portfolio Value: N/A")
            self.total_value_label.setText("Total Value: N/A")
    
    def update_date_constraints(self):
        """Set minimum and maximum dates based on existing trades and dataset."""
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
            min_date = pd.Timestamp(datetime(2000, 1, 1), tz='UTC')  # Fallback
        
        max_date = dataset_end if dataset_end else pd.Timestamp(datetime.now(), tz='UTC')
        
        self.start_date.setMinimumDate(min_date.date())
        self.start_date.setMaximumDate(max_date.date())
        self.end_date.setMinimumDate(min_date.date())
        self.end_date.setMaximumDate(max_date.date())
        self.start_date.dateChanged.connect(self.update_end_date_minimum)
        
        # Update tooltips with valid range
        self.start_date.setToolTip(
            f"Select a date between {min_date.date()} and {max_date.date()}"
        )
        self.end_date.setToolTip(
            f"Select a date after start date and before {max_date.date()}"
        )
        
        
    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        if is_dark_mode:
            label_style = "color: #ffffff;"
            input_style = "background-color: #3c3f41; color: #ffffff;"
            combo_style = "background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;"
            button_style_execute = "background-color: #2a82da; color: #ffffff;"
            button_style_reset = "background-color: #ff4444; color: #ffffff;"
        else:
            label_style = "color: black;"
            input_style = "background-color: #ffffff; color: black;"
            combo_style = "background-color: #ffffff; color: black; selection-background-color: #2a82da;"
            button_style_execute = "background-color: #2a82da; color: black;"
            button_style_reset = "background-color: #ff4444; color: black;"
        
        self.investment_label.setStyleSheet(label_style)
        self.risk_label.setStyleSheet(label_style)
        self.start_label.setStyleSheet(label_style)
        self.end_label.setStyleSheet(label_style)
        self.mode_label.setStyleSheet(label_style)
        self.cash_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.portfolio_value_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.total_value_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.investment_input.setStyleSheet(input_style)
        self.risk_input.setStyleSheet(input_style)
        self.start_date.setStyleSheet(input_style)
        self.end_date.setStyleSheet(input_style)
        self.mode_combo.setStyleSheet(combo_style)
        self.execute_button.setStyleSheet(button_style_execute)
        self.reset_button.setStyleSheet(button_style_reset)
        
    def get_message_box_style(self):
        """Return stylesheet for QMessageBox based on the current theme."""
        if self.is_dark_mode:
            return """
                QMessageBox { background-color: #353535; color: #ffffff; }
                QMessageBox QLabel { color: #ffffff; }
                QMessageBox QPushButton { background-color: #444444; color: #ffffff; border: 1px solid #666666; }
                QMessageBox QPushButton:hover { background-color: #555555; }
                QMessageBox::item:selected { background-color: #2a82da; color: #ffffff; }
            """
        else:
            return """
                QMessageBox { background-color: #f0f0f0; color: black; }
                QMessageBox QLabel { color: black; }
                QMessageBox QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #cccccc; }
                QMessageBox QPushButton:hover { background-color: #d0d0d0; }
                QMessageBox::item:selected { background-color: #2a82da; color: black; }
            """
        
    def show_message_box(self, icon, title, text, buttons):
        """Helper method to create and show a QMessageBox with current theme."""
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        msg.setStyleSheet(self.get_message_box_style())
        return msg.exec_()
        
    def reset_portfolio(self):
        """Delete portfolio_state.json to reset portfolio state."""
        try:
            if os.path.exists(self.portfolio_state_file):
                os.remove(self.portfolio_state_file)
                logger.debug(f"Deleted {self.portfolio_state_file}")
                self.show_message_box(
                    QMessageBox.Information,
                    "Success",
                    "Portfolio state reset successfully.",
                    QMessageBox.Ok
                )
            else:
                self.show_message_box(
                    QMessageBox.Information,
                    "Info",
                    "No portfolio state file to reset.",
                    QMessageBox.Ok
                )
            self.update_financial_metrics()
            self.main_window.update_dashboard()
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")
            self.show_message_box(
                QMessageBox.Critical,
                "Error",
                f"Failed to reset portfolio: {e}",
                QMessageBox.Ok
            )
      
    def update_date_constraints(self):
        """Set minimum dates based on existing trades."""
        orders = get_orders()
        if orders:
            order_dates = pd.to_datetime([order['date'] for order in orders], utc=True)
            latest_trade_date = order_dates.max()
            # Set minimum date to the day after the latest trade
            min_date = latest_trade_date + pd.Timedelta(days=1)
            self.start_date.setMinimumDate(min_date.date())
            self.end_date.setMinimumDate(min_date.date())
            # Ensure end_date is after start_date
            self.start_date.dateChanged.connect(self.update_end_date_minimum)
        else:
            # No trades; use dataset start date if available
            if self.data_manager.dataset_start_date:
                min_date = self.data_manager.dataset_start_date.date()
                self.start_date.setMinimumDate(min_date)
                self.end_date.setMinimumDate(min_date)
    
    def update_end_date_minimum(self):
        """Ensure end_date is after start_date."""
        self.end_date.setMinimumDate(self.start_date.date())
            
    def update_portfolio(self):
        try:
            investment_amount = float(self.investment_input.text())
            risk_level = self.risk_input.value() * 10
            start_date = pd.Timestamp(self.start_date.date().toPyDate(), tz='UTC')
            end_date = pd.Timestamp(self.end_date.date().toPyDate(), tz='UTC')
            
            # Validate date range against existing trades and dataset
            orders = get_orders()
            dataset_start = self.data_manager.dataset_start_date
            dataset_end = self.data_manager.dataset_end_date
            
            if orders:
                order_dates = pd.to_datetime([order['date'] for order in orders], utc=True)
                latest_trade_date = order_dates.max()
                if start_date <= latest_trade_date:
                    self.show_message_box(
                        QMessageBox.Critical,
                        "Invalid Date Range",
                        f"Start date must be after {latest_trade_date.date()} due to existing trades.\n"
                        f"Valid range: {latest_trade_date.date() + pd.Timedelta(days=1)} to {dataset_end.date() if dataset_end else 'today'}.",
                        QMessageBox.Ok
                    )
                    return
            
            if dataset_start and start_date < dataset_start:
                self.show_message_box(
                    QMessageBox.Critical,
                    "Invalid Date Range",
                    f"Start date cannot be before dataset start ({dataset_start.date()}).",
                    QMessageBox.Ok
                )
                return
            
            if dataset_end and end_date > dataset_end:
                self.show_message_box(
                    QMessageBox.Critical,
                    "Invalid Date Range",
                    f"End date cannot be after dataset end ({dataset_end.date()}).",
                    QMessageBox.Ok
                )
                return
            
            success, message = self.data_manager.set_date_range(start_date, end_date)
            if not success:
                self.show_message_box(
                    QMessageBox.Critical,
                    "Invalid Date Range",
                    message,
                    QMessageBox.Ok
                )
                return
            if message:
                self.show_message_box(
                    QMessageBox.Information,
                    "Date Range Adjusted",
                    message,
                    QMessageBox.Ok
                )
            
            logger.debug(f"Executing with risk_level={risk_level}, mode={self.mode_combo.currentText().lower()}")
            success, result = execute_trading_strategy(
                investment_amount,
                risk_level,
                start_date,
                end_date,
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
                    QMessageBox.Information,
                    "Signal Quality",
                    signal_quality_message,
                    QMessageBox.Ok
                )
                
                if correlation < 0.1:
                    self.show_message_box(
                        QMessageBox.Warning,
                        "Low Signal Quality",
                        "Signal-return correlation is low. Strategy may be unreliable.",
                        QMessageBox.Ok
                    )
                
                if self.mode_combo.currentText().lower() == "semi-automatic" and orders:
                    self.show_message_box(
                        QMessageBox.Information,
                        "Confirm Trades",
                        f"{len(orders)} trade(s) suggested. Execute them?\n" +
                        "\n".join([f"{order['action'].capitalize()} {order['shares_amount']} shares of {order['ticker']} at ${order['price']:.2f}" for order in orders]),
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if msg.exec_() == QMessageBox.Yes:
                        success, result = execute_trading_strategy(
                            investment_amount,
                            risk_level,
                            start_date,
                            end_date,
                            data_manager=self.data_manager,
                            mode="automatic",
                            reset_state=False
                        )
                        if not success:
                            self.show_message_box(
                                QMessageBox.Critical,
                                "Error",
                                f"Failed to execute trades: {result.get('warning_message', 'Unknown error')}",
                                QMessageBox.Ok
                            )
                            return
                        portfolio_history = result.get('portfolio_history', [])
                        portfolio_value = result.get('portfolio_value', investment_amount)
                        cash = result.get('cash', investment_amount)
                        orders = result.get('orders', [])
                        warning_message = result.get('warning_message', '')
                
                if not orders and warning_message:
                    self.show_message_box(
                        QMessageBox.Warning,
                        "No Signals Detected",
                        warning_message,
                        QMessageBox.Ok
                    )
                
                self.update_financial_metrics()
                self.main_window.update_dashboard()
            else:
                error_message = result.get('warning_message', 'Unknown error')
                self.show_message_box(
                    QMessageBox.Critical,
                    "Error",
                    f"Failed to execute strategy: {error_message}",
                    QMessageBox.Ok
                )
                self.update_financial_metrics()
        except Exception as e:
            logger.error(f"Error in update_portfolio: {e}", exc_info=True)
            self.show_message_box(
                QMessageBox.Critical,
                "Error",
                f"Failed to run strategy: {e}",
                QMessageBox.Ok
            )
            self.update_financial_metrics()
            
            