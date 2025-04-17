from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QMessageBox)
from PyQt5.QtCore import Qt
from trading_logic import get_orders, get_portfolio_history
import pandas as pd

class RecommendationPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Trading Recommendations")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Recommendations table
        self.table = QTableWidget()
        self.table.setColumnCount(11)  # Added column for Portfolio Value After
        self.table.setHorizontalHeaderLabels([
            "Date", "Ticker", "Action", "Shares", "Price", 
            "Investment Amount", "Previous Shares", "New Shares", "Sharpe", 
            "Cash Balance After", "Portfolio Value After"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("QTableWidget { background-color: #2b2b2b; color: #ffffff; gridline-color: #555555; } QHeaderView::section { background-color: #3c3f41; color: #ffffff; }")
        layout.addWidget(self.table)
        
        # Alert message
        self.alert_label = QLabel()
        self.alert_label.setStyleSheet("color: red;")
        self.alert_label.setVisible(False)
        layout.addWidget(self.alert_label)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Recommendations")
        refresh_button.clicked.connect(self.update_recommendations)
        layout.addWidget(refresh_button)
        
    def update_recommendations(self):
        order_history = get_orders()
        portfolio_history = get_portfolio_history()
        if not order_history:
            self.show_no_recommendations_alert()
            return

        # Create a mapping of dates to portfolio values
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date']).dt.date
            portfolio_map = portfolio_df.set_index('date')['portfolio_value'].to_dict()
        else:
            portfolio_map = {}

        self.table.setRowCount(0)
        for order in order_history:
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            order_date = pd.to_datetime(order['date']).date()
            portfolio_value = portfolio_map.get(order_date, order['capital_after'])
            self.table.setItem(row_position, 0, QTableWidgetItem(str(order['date'])))
            self.table.setItem(row_position, 1, QTableWidgetItem(order['ticker']))
            self.table.setItem(row_position, 2, QTableWidgetItem(order['action']))
            self.table.setItem(row_position, 3, QTableWidgetItem(str(order['shares_amount'])))
            self.table.setItem(row_position, 4, QTableWidgetItem(f"${order['price']:,.2f}"))
            self.table.setItem(row_position, 5, QTableWidgetItem(f"${order['investment_amount']:,.2f}"))
            self.table.setItem(row_position, 6, QTableWidgetItem(str(order['previous_shares'])))
            self.table.setItem(row_position, 7, QTableWidgetItem(str(order['new_total_shares'])))
            self.table.setItem(row_position, 8, QTableWidgetItem(f"{order['sharpe']:.2f}"))
            self.table.setItem(row_position, 9, QTableWidgetItem(f"${order['capital_after']:,.2f}"))
            self.table.setItem(row_position, 10, QTableWidgetItem(f"${portfolio_value:,.2f}"))

        self.table.resizeColumnsToContents()
        
    def show_no_recommendations_alert(self):
        self.alert_label.setText("No order history generated for the selected parameters.")
        self.alert_label.setVisible(True)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("No Orders Generated")
        msg.setText("No trading orders were generated for the selected date range and parameters.\n"
                    "Adjust the risk level, investment amount, or date range.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox { background-color: #353535; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #444; color: white; border: 1px solid #666; padding: 5px 15px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; }
        """)
        msg.exec_()