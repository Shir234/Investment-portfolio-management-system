from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QDateEdit, QPushButton, QComboBox, QMessageBox)
from datetime import datetime
import logging
from data.trading_connector import execute_trading_strategy
from backend.trading_logic import run_integrated_trading_strategy
import pandas as pd
import os

# Suppress matplotlib logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class InputPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.portfolio_state_file = 'data/portfolio_state.json'
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Investment amount
        investment_layout = QHBoxLayout()
        investment_label = QLabel("Investment Amount ($):")
        investment_label.setStyleSheet("color: #ffffff;")
        self.investment_input = QLineEdit("10000")
        self.investment_input.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        investment_layout.addWidget(investment_label)
        investment_layout.addWidget(self.investment_input)
        layout.addLayout(investment_layout)
        
        # Risk level
        risk_layout = QHBoxLayout()
        risk_label = QLabel("Risk Level (0-10):")
        risk_label.setStyleSheet("color: #ffffff;")
        self.risk_input = QLineEdit("10")
        self.risk_input.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        risk_layout.addWidget(risk_label)
        risk_layout.addWidget(self.risk_input)
        layout.addLayout(risk_layout)
        
        # Date range
        date_layout = QHBoxLayout()
        start_label = QLabel("Start Date:")
        start_label.setStyleSheet("color: #ffffff;")
        self.start_date = QDateEdit()
        self.start_date.setDate(datetime(2021, 10, 18))
        self.start_date.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        end_label = QLabel("End Date:")
        end_label.setStyleSheet("color: #ffffff;")
        self.end_date = QDateEdit()
        self.end_date.setDate(datetime(2023, 12, 22))
        self.end_date.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        date_layout.addWidget(start_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(end_label)
        date_layout.addWidget(self.end_date)
        layout.addLayout(date_layout)
        
        # Mode selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Trading Mode:")
        mode_label.setStyleSheet("color: #ffffff;")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Semi-automatic"])
        self.mode_combo.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        execute_button = QPushButton("Execute Trading Strategy")
        execute_button.clicked.connect(self.update_portfolio)
        execute_button.setStyleSheet("background-color: #2a82da; color: #ffffff;")
        reset_button = QPushButton("Reset Portfolio")
        reset_button.clicked.connect(self.reset_portfolio)
        reset_button.setStyleSheet("background-color: #ff4444; color: #ffffff;")
        button_layout.addWidget(execute_button)
        button_layout.addWidget(reset_button)
        layout.addLayout(button_layout)
        
        # Portfolio value display
        self.portfolio_value_label = QLabel("Current Portfolio Value: N/A")
        self.portfolio_value_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        layout.addWidget(self.portfolio_value_label)
        
    def reset_portfolio(self):
        """Delete portfolio_state.json to reset portfolio state."""
        try:
            if os.path.exists(self.portfolio_state_file):
                os.remove(self.portfolio_state_file)
                logging.debug(f"Deleted {self.portfolio_state_file}")
                QMessageBox.information(self, "Success", "Portfolio state reset successfully.", QMessageBox.Ok)
            else:
                QMessageBox.information(self, "Info", "No portfolio state file to reset.", QMessageBox.Ok)
            self.portfolio_value_label.setText("Current Portfolio Value: N/A")
            self.main_window.update_dashboard()
        except Exception as e:
            logging.error(f"Error resetting portfolio: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset portfolio: {e}", QMessageBox.Ok)
        
    def update_portfolio(self):
        try:
            investment_amount = float(self.investment_input.text())
            risk_level = float(self.risk_input.text()) * 10  # Scale 0-10 to 0-100
            # Convert dates to pd.Timestamp with UTC timezone
            start_date = pd.Timestamp(self.start_date.date().toPyDate(), tz='UTC')
            end_date = pd.Timestamp(self.end_date.date().toPyDate(), tz='UTC')

            # Set date range in data_manager
            success, message = self.data_manager.set_date_range(start_date, end_date)
            if not success:
                QMessageBox.critical(
                    self,
                    "Invalid Date Range",
                    message,
                    QMessageBox.Ok
                )
                return
            if message:  # Display adjustment message if date range was modified
                QMessageBox.information(
                    self,
                    "Date Range Adjusted",
                    message,
                    QMessageBox.Ok
                )

            logging.debug(f"Executing with risk_level={risk_level}, mode={self.mode_combo.currentText().lower()}")
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
                orders = result.get('orders', [])
                warning_message = result.get('warning_message', '')
                
                if self.mode_combo.currentText().lower() == "semi-automatic" and orders:
                    # Show confirmation dialog
                    msg = QMessageBox()
                    msg.setWindowTitle("Confirm Trades")
                    msg.setText(f"{len(orders)} trade(s) suggested. Execute them?\n" +
                                "\n".join([f"{order['action'].capitalize()} {order['shares_amount']} shares of {order['ticker']} at ${order['price']:.2f}" for order in orders]))
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    if msg.exec_() == QMessageBox.Yes:
                        # Re-run in automatic mode to execute trades
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
                            QMessageBox.critical(self, "Error", f"Failed to execute trades: {result.get('error', 'Unknown error')}")
                            return
                        portfolio_history = result.get('portfolio_history', [])
                        portfolio_value = result.get('portfolio_value', investment_amount)
                        orders = result.get('orders', [])
                        warning_message = result.get('warning_message', '')
                
                if not orders and warning_message:
                    QMessageBox.warning(
                        self,
                        "No Signals Detected",
                        warning_message,
                        QMessageBox.Ok
                    )
                
                logging.debug(f"Portfolio history: {portfolio_history[-10:]}, Portfolio value: {portfolio_value}")
                self.portfolio_value_label.setText(f"Current Portfolio Value: ${portfolio_value:,.2f}")
                self.main_window.update_dashboard()
            else:
                error_message = result.get('error', 'Unknown error')
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to execute strategy: {error_message}",
                    QMessageBox.Ok
                )
                self.portfolio_value_label.setText("Current Portfolio Value: N/A")
        except Exception as e:
            logging.error(f"Error in update_portfolio: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to run strategy: {e}",
                QMessageBox.Ok
            )
            self.portfolio_value_label.setText("Current Portfolio Value: N/A")