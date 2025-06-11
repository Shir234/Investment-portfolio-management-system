from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, 
                             QDateEdit, QPushButton, QComboBox, QMessageBox, QVBoxLayout)
from datetime import datetime
import logging
from data.trading_connector import execute_trading_strategy
import pandas as pd
import os

# Ensure logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class InputPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.is_dark_mode = True  # Default to dark mode
        self.portfolio_state_file = 'data/portfolio_state.json'
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        investment_layout = QHBoxLayout()
        self.investment_label = QLabel("Investment Amount ($):")
        self.investment_label.setObjectName("investment_label")
        self.investment_label.setStyleSheet("color: #ffffff;")
        self.investment_input = QLineEdit("10000")
        self.investment_input.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        investment_layout.addWidget(self.investment_label)
        investment_layout.addWidget(self.investment_input)
        layout.addLayout(investment_layout)
        
        risk_layout = QHBoxLayout()
        self.risk_label = QLabel("Risk Level (0-10):")
        self.risk_label.setObjectName("risk_label")
        self.risk_label.setStyleSheet("color: #ffffff;")
        self.risk_input = QLineEdit("10")
        self.risk_input.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_input)
        layout.addLayout(risk_layout)
        
        date_layout = QHBoxLayout()
        self.start_label = QLabel("Start Date:")
        self.start_label.setObjectName("start_label")
        self.start_label.setStyleSheet("color: #ffffff;")
        self.start_date = QDateEdit()
        self.start_date.setDate(datetime(2021, 10, 18))
        self.start_date.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        self.end_label = QLabel("End Date:")
        self.end_label.setObjectName("end_label")
        self.end_label.setStyleSheet("color: #ffffff;")
        self.end_date = QDateEdit()
        self.end_date.setDate(datetime(2023, 12, 22))
        self.end_date.setStyleSheet("background-color: #3c3f41; color: #ffffff;")
        date_layout.addWidget(self.start_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(self.end_label)
        date_layout.addWidget(self.end_date)
        layout.addLayout(date_layout)
        
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
        
        self.portfolio_value_label = QLabel("Current Portfolio Value: N/A")
        self.portfolio_value_label.setObjectName("portfolio_value_label")
        self.portfolio_value_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        layout.addWidget(self.portfolio_value_label)
        
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
        self.portfolio_value_label.setStyleSheet(f"font-weight: bold; {label_style}")
        self.investment_input.setStyleSheet(input_style)
        self.risk_input.setStyleSheet(input_style)
        self.start_date.setStyleSheet(input_style)
        self.end_date.setStyleSheet(input_style)
        self.mode_combo.setStyleSheet(combo_style)
        self.execute_button.setStyleSheet(button_style_execute)
        self.reset_button.setStyleSheet(button_style_reset)
        
    def get_message_box_style(self):
        """Return QMessageBox stylesheet based on theme."""
        return f"""
            QMessageBox {{ 
                background-color: {'#353535' if self.is_dark_mode else '#f0f0f0'}; 
                color: {'#ffffff' if self.is_dark_mode else 'black'}; 
            }}
            QMessageBox QLabel {{ 
                color: {'#ffffff' if self.is_dark_mode else 'black'}; 
            }}
            QMessageBox QPushButton {{ 
                background-color: {'#444444' if self.is_dark_mode else '#e0e0e0'}; 
                color: {'#ffffff' if self.is_dark_mode else 'black'}; 
                border: 1px solid {'#666666' if self.is_dark_mode else '#cccccc'}; 
                padding: 5px 15px; 
                border-radius: 3px; 
            }}
            QMessageBox QPushButton:hover {{ 
                background-color: {'#555555' if self.is_dark_mode else '#d0d0d0'}; 
            }}
        """
        
    def reset_portfolio(self):
        """Delete portfolio_state.json to reset portfolio state."""
        try:
            if os.path.exists(self.portfolio_state_file):
                os.remove(self.portfolio_state_file)
                logger.debug(f"Deleted {self.portfolio_state_file}")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Success")
                msg.setText("Portfolio state reset successfully.")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Info")
                msg.setText("No portfolio state file to reset.")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
            self.portfolio_value_label.setText("Current Portfolio Value: N/A")
            self.main_window.update_dashboard()
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to reset portfolio: {e}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
        
    def update_portfolio(self):
        try:
            investment_amount = float(self.investment_input.text())
            risk_level = float(self.risk_input.text()) * 10
            start_date = pd.Timestamp(self.start_date.date().toPyDate(), tz='UTC')
            end_date = pd.Timestamp(self.end_date.date().toPyDate(), tz='UTC')

            success, message = self.data_manager.set_date_range(start_date, end_date)
            if not success:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Invalid Date Range")
                msg.setText(message)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
                return
            if message:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Date Range Adjusted")
                msg.setText(message)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()

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
                orders = result.get('orders', [])
                
                
                
                if self.mode_combo.currentText().lower() == "semi-automatic" and orders:
                    msg = QMessageBox()
                    msg.setWindowTitle("Confirm Trades")
                    msg.setText(f"{len(orders)} trade(s) suggested. Execute them?\n" +
                                "\n".join([f"{order['action'].capitalize()} {order['shares_amount']} shares of {order['ticker']} at ${order['price']:.2f}" for order in orders]))
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg.setStyleSheet(self.get_message_box_style())
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
                            msg = QMessageBox()
                            msg.setIcon(QMessageBox.Critical)
                            msg.setWindowTitle("Error")
                            msg.setText(f"Failed to execute trades: {result.get('warning_message', 'Unknown error')}")
                            msg.setStyleSheet(self.get_message_box_style())
                            msg.exec_()
                            return
                        portfolio_history = result.get('portfolio_history', [])
                        portfolio_value = result.get('portfolio_value', investment_amount)
                        orders = result.get('orders', [])
                        warning_message = result.get('warning_message', '')
                
                if not orders and warning_message:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("No Signals Detected")
                    msg.setText(warning_message)
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.setStyleSheet(self.get_message_box_style())
                    msg.exec_()
                
                logger.debug(f"Portfolio history: {portfolio_history[-10:]}, Portfolio value: {portfolio_value}")
                self.portfolio_value_label.setText(f"Current Portfolio Value: ${portfolio_value:,.2f}")
                self.main_window.update_dashboard()
            else:
                error_message = result.get('warning_message', 'Unknown error')
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Error")
                msg.setText(f"Failed to execute strategy: {error_message}")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec_()
                self.portfolio_value_label.setText("Current Portfolio Value: N/A")
        except Exception as e:
            logger.error(f"Error in update_portfolio: {e}", exc_info=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to run strategy: {e}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec_()
            self.portfolio_value_label.setText("Current Portfolio Value: N/A")