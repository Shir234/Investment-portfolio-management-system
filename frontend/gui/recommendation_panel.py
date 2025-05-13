from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton, 
                             QMessageBox, QComboBox)
from PyQt5.QtCore import Qt
from trading_logic import get_orders, get_portfolio_history
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Filter selector
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter Orders:")
        filter_label.setStyleSheet("color: #ffffff;")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Orders", "Buy Orders", "Sell Orders", "Cover Orders", "Short Orders"])
        self.filter_combo.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        self.filter_combo.currentIndexChanged.connect(self.update_recommendations)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)
        
        # Trade count label
        self.trade_count_label = QLabel("Number of Recommendations: 0")
        self.trade_count_label.setStyleSheet("font-size: 12px; color: #ffffff;")
        layout.addWidget(self.trade_count_label)
        
        # Recommendations table
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Date", "Ticker", "Action", "Shares", "Price", 
            "Investment Amount", "Previous Shares", "Total Shares", 
            "Pred. Sharpe", "Actual Sharpe", "Cash After", "Portfolio Value"
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
        # Get filter selection
        filter_type = self.filter_combo.currentText()
        all_orders = get_orders()
        
        # Apply filter based on selection
        if filter_type == "Buy Orders":
            order_history = [order for order in all_orders if order['action'] == 'buy']
        elif filter_type == "Sell Orders":
            order_history = [order for order in all_orders if order['action'] == 'sell']
        elif filter_type == "Cover Orders":
            order_history = [order for order in all_orders if order['action'] == 'cover']
        elif filter_type == "Short Orders":
            order_history = [order for order in all_orders if order['action'] == 'short']
        else:  # All Orders
            order_history = all_orders
        
        portfolio_history = get_portfolio_history()
        
        # Update trade count label
        self.trade_count_label.setText(f"Number of {filter_type}: {len(order_history)}")
        
        if not order_history:
            self.show_no_recommendations_alert()
            self.table.setRowCount(0)
            return

        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date']).dt.date
            logger.debug(f"Portfolio DataFrame columns: {portfolio_df.columns}")
            portfolio_map = portfolio_df.set_index('date')['value'].to_dict()
        else:
            portfolio_map = {}
            logger.debug("Portfolio history is empty")

        # Calculate cash balance incrementally
        try:
            initial_cash = float(self.main_window.input_panel.investment_input.text())
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not retrieve investment amount, defaulting to 10000: {e}")
            initial_cash = 10000.0
        cash = initial_cash
        cash_balances = []
        for order in order_history:
            if order['action'] in ['buy', 'short']:
                cash -= order['investment_amount']
            else:  # sell or cover
                cash += order['investment_amount']
            cash_balances.append(cash)

        # Verify alignment between orders and portfolio history
        if order_history and portfolio_history:
            order_dates = set(pd.to_datetime(order['date']).date() for order in order_history)
            portfolio_dates = set(pd.to_datetime(entry['date']).date() for entry in portfolio_history)
            if not order_dates.issubset(portfolio_dates):
                logger.warning("Some order dates are not present in portfolio history")

        self.table.setRowCount(0)
        for i, order in enumerate(order_history):
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            order_date = pd.to_datetime(order['date']).date()
            portfolio_value = portfolio_map.get(order_date, cash_balances[i])
            self.table.setItem(row_position, 0, QTableWidgetItem(str(order['date'].date())))
            self.table.setItem(row_position, 1, QTableWidgetItem(order['ticker']))
            self.table.setItem(row_position, 2, QTableWidgetItem(order['action'].capitalize()))
            self.table.setItem(row_position, 3, QTableWidgetItem(str(order['shares_amount'])))
            self.table.setItem(row_position, 4, QTableWidgetItem(f"${order['price']:,.2f}"))
            self.table.setItem(row_position, 5, QTableWidgetItem(f"${order['investment_amount']:,.2f}"))
            self.table.setItem(row_position, 6, QTableWidgetItem(str(order['previous_shares'])))
            self.table.setItem(row_position, 7, QTableWidgetItem(str(order['new_total_shares'])))
            self.table.setItem(row_position, 8, QTableWidgetItem(f"{order['sharpe']:.2f}"))
            self.table.setItem(row_position, 9, QTableWidgetItem(f"{order.get('actual_sharpe', 0.0):.2f}"))
            self.table.setItem(row_position, 10, QTableWidgetItem(f"${cash_balances[i]:,.2f}"))
            self.table.setItem(row_position, 11, QTableWidgetItem(f"${portfolio_value:,.2f}"))

        self.table.resizeColumnsToContents()
        logger.debug(f"Table updated with {len(order_history)} orders")
        
    def show_no_recommendations_alert(self):
        filter_type = self.filter_combo.currentText()
        self.alert_label.setText(f"No {filter_type.lower()} generated for the selected parameters.")
        self.alert_label.setVisible(True)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("No Recommendations Generated")
        msg.setText(f"No {filter_type.lower()} were generated for the selected date range and parameters.\n"
                    "Adjust the risk level, investment amount, or date range.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox { background-color: #353535; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #444; color: white; border: 1px solid #666; padding: 5px 15px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; }
        """)
        msg.exec_()