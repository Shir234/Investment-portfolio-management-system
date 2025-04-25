from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QDateEdit, QPushButton, QComboBox)
from datetime import datetime
import logging
from data.trading_connector import execute_trading_strategy

# Suppress matplotlib logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class InputPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
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
        self.end_date.setDate(datetime(2023, 12, 29))
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
        
        # Execute button
        execute_button = QPushButton("Execute Trading Strategy")
        execute_button.clicked.connect(self.update_portfolio)
        execute_button.setStyleSheet("background-color: #2a82da; color: #ffffff;")
        layout.addWidget(execute_button)
        
        # Portfolio value display
        self.portfolio_value_label = QLabel("Current Portfolio Value: N/A")
        self.portfolio_value_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        layout.addWidget(self.portfolio_value_label)
        
    def update_portfolio(self):
        try:
            investment_amount = float(self.investment_input.text())
            risk_level = float(self.risk_input.text()) * 10  # Scale 0-10 to 0-100
            start_date = self.start_date.date().toPython()
            end_date = self.end_date.date().toPython()
            mode = self.mode_combo.currentText().lower()
            logging.debug(f"Executing with risk_level={risk_level}, mode={mode}")
            success, result = execute_trading_strategy(
                investment_amount,
                risk_level,
                start_date,
                end_date,
                data_manager=self.data_manager,
                mode=mode,
                reset_state=True
            )
            if success:
                portfolio_history = result['portfolio_history']
                portfolio_value = result['portfolio_value']
                logging.debug(f"Portfolio history: {portfolio_history[-10:]}, Portfolio value: {portfolio_value}")
                self.portfolio_value_label.setText(f"Current Portfolio Value: ${portfolio_value:,.2f}")
                self.main_window.update_dashboard()
            else:
                self.portfolio_value_label.setText("Current Portfolio Value: N/A")
        except Exception as e:
            logging.error(f"Error in update_portfolio: {e}", exc_info=True)
            self.portfolio_value_label.setText("Current Portfolio Value: N/A")
