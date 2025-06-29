from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton, 
                             QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
from backend.trading_logic_new import get_orders, get_portfolio_history
import pandas as pd
import logging
from frontend.logging_config import get_logger

import os

# Configure logging
logger = get_logger(__name__)


class RecommendationPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.is_dark_mode = True  # Default to dark mode
        self.portfolio_state_file = 'data/portfolio_state.json'
        self.setup_ui()
        logger.info("RecommendationPanel initialized")
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title label
        self.title_label = QLabel("Trading History")
        self.title_label.setObjectName("title_label")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        layout.addWidget(self.title_label)
        
        # Filter layout
        filter_layout = QHBoxLayout()
        self.filter_label = QLabel("Filter Orders:")
        self.filter_label.setObjectName("filter_label")
        self.filter_label.setStyleSheet("color: #ffffff;")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Orders", "Buy Orders", "Sell Orders"])
        self.filter_combo.setStyleSheet("background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;")
        self.filter_combo.currentIndexChanged.connect(self.update_recommendations)
        filter_layout.addWidget(self.filter_label)
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)
        
        # Trade count label
        self.trade_count_label = QLabel("Number of Orders: 0")
        self.trade_count_label.setObjectName("trade_count_label")
        self.trade_count_label.setStyleSheet("font-size: 12px; color: #ffffff;")
        layout.addWidget(self.trade_count_label)
        
        # Table for trade history
        self.table = QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
                "Date",           # When the trade happened
                "Ticker",         # Stock symbol  
                "Action",         # Buy/Sell
                "Price",          # Price per share
                "Shares",         # Shares in this transaction
                "Trade Value",    # shares * price (gross amount)
                "Total Cost",     # Trade value + transaction costs
                "Total Shares",   # Cumulative shares after this trade
                "Pred. Sharpe",   # Prediction that triggered the trade
                "Actual Sharpe",  # Actual Sharpe to show in the test
                "Signal Strength" # How strong the signal was
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #3c3f41; color: #ffffff; }"
            "QTableWidget::item { border: 1px solid #555555; }"
            "QTableWidget::item:selected { background-color: #2a82da; }"
            "QHeaderView::section { background-color: #353535; color: #ffffff; border: 1px solid #555555; }"
        )
        layout.addWidget(self.table)
        
        # Clear history button
        self.clear_button = QPushButton("Clear Trade History")
        self.clear_button.setObjectName("clear_button")
        self.clear_button.clicked.connect(self.clear_trade_history)
        self.clear_button.setStyleSheet("background-color: #ff4444; color: #ffffff;")
        layout.addWidget(self.clear_button)
        
        self.update_recommendations()
        
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
        
    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        if is_dark_mode:
            label_style = "color: #ffffff;"
            combo_style = "background-color: #3c3f41; color: #ffffff; selection-background-color: #2a82da;"
            button_style = "background-color: #ff4444; color: #ffffff;"
            table_style = (
                "QTableWidget { background-color: #3c3f41; color: #ffffff; }"
                "QTableWidget::item { border: 1px solid #555555; }"
                "QTableWidget::item:selected { background-color: #2a82da; }"
                "QHeaderView::section { background-color: #353535; color: #ffffff; border: 1px solid #555555; }"
            )
        else:
            label_style = "color: black;"
            combo_style = "background-color: #ffffff; color: black; selection-background-color: #2a82da;"
            button_style = "background-color: #ff4444; color: black;"
            table_style = (
                "QTableWidget { background-color: #ffffff; color: black; }"
                "QTableWidget::item { border: 1px solid #cccccc; }"
                "QTableWidget::item:selected { background-color: #2a82da; }"
                "QHeaderView::section { background-color: #e0e0e0; color: black; border: 1px solid #cccccc; }"
            )
        
        self.title_label.setStyleSheet(f"font-size: 14px; font-weight: bold; {label_style}")
        self.filter_label.setStyleSheet(label_style)
        self.trade_count_label.setStyleSheet(f"font-size: 12px; {label_style}")
        self.filter_combo.setStyleSheet(combo_style)
        self.clear_button.setStyleSheet(button_style)
        self.table.setStyleSheet(table_style)
        self.update_recommendations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")
        
    def update_recommendations(self):
        """Update the table with trade history."""
        try:
            orders = get_orders()
            if not orders:
                self.table.setRowCount(0)
                self.trade_count_label.setText("Number of Orders: 0")
                logger.info("No orders found")
                return
            
            orders_df = pd.DataFrame(orders)
            filter_type = self.filter_combo.currentText()
            
            if filter_type == "Buy Orders":
                orders_df = orders_df[orders_df['action'] == 'buy']
            elif filter_type == "Sell Orders":
                orders_df = orders_df[orders_df['action'] == 'sell']
            
            self.table.setRowCount(len(orders_df))
            self.trade_count_label.setText(f"Number of Orders: {len(orders_df)}")
            
            for row, order in orders_df.iterrows():
                # Column 0: Date
                self.table.setItem(row, 0, QTableWidgetItem(str(order.get('date', ''))))
                # Column 1: Ticker
                self.table.setItem(row, 1, QTableWidgetItem(order.get('ticker', '')))
                # Column 2: Action
                action = order.get('action', '').upper()
                self.table.setItem(row, 2, QTableWidgetItem(action))
                ###self.table.setItem(row, 2, QTableWidgetItem(order.get('action', '')))

                # Column 3: Price
                price = order.get('price', 0)
                self.table.setItem(row, 3, QTableWidgetItem(f"${price:,.2f}"))
                ###self.table.setItem(row, 3, QTableWidgetItem(f"${order.get('price', 0):,.2f}"))
                
                # Column 4: Shares
                shares = order.get('shares_amount', 0)
                self.table.setItem(row, 4, QTableWidgetItem(str(shares)))
                ###self.table.setItem(row, 4, QTableWidgetItem(str(order.get('shares_amount', 0))))
                
                # Column 5: Trade Value (shares * price)
                trade_value = price * shares
                self.table.setItem(row, 5, QTableWidgetItem(f"${trade_value:,.2f}"))
                # trade_value = order.get('price', 0) * order.get('shares_amount', 0)
                # self.table.setItem(row, 5, QTableWidgetItem(f"${trade_value:,.2f}"))
                
                # Column 6: Total Cost (different calculation for buy vs sell)
                if action.lower() == 'buy':
                    # For buys: investment_amount + transaction_cost OR total_cost
                    total_cost = order.get('total_cost', 
                        order.get('investment_amount', 0) + order.get('transaction_cost', 0))
                else:  # sell
                    # For sells: use total_proceeds
                    total_cost = order.get('total_proceeds', 
                        trade_value - order.get('transaction_cost', 0))
                self.table.setItem(row, 6, QTableWidgetItem(f"${total_cost:,.2f}"))
                # total_cost = order.get('investment_amount', 0) + order.get('transaction_cost', 0)
                # self.table.setItem(row, 6, QTableWidgetItem(f"${total_cost:,.2f}"))

                # Column 7: Total Shares (different logic for buy vs sell)
                if action.lower() == 'buy':
                    total_shares = order.get('new_total_shares', order.get('total_shares', 0))
                else:  # sell
                    # For sells, show remaining shares (could be 0 if position closed)
                    total_shares = order.get('new_total_shares', 0)
                if pd.isna(total_shares) or total_shares is None:
                    total_shares = 0
                elif isinstance(total_shares, (int, float)) and total_shares == int(total_shares):
                    total_shares = int(total_shares)
                self.table.setItem(row, 7, QTableWidgetItem(str(total_shares)))
                ###self.table.setItem(row, 7, QTableWidgetItem(str(order.get('total_shares', 0))))

                # Column 8: Pred. Sharpe (show "SELL" for sell orders)
                if action.lower() == 'sell':
                    sharpe_display = "SELL"
                else:
                    sharpe = order.get('sharpe', order.get('Best_Prediction', 0))
                    if sharpe == -1 or pd.isna(sharpe):
                        sharpe_display = "N/A"
                    else:
                        sharpe_display = f"{sharpe:.3f}"
                self.table.setItem(row, 8, QTableWidgetItem(sharpe_display))
                # sharpe = order.get('sharpe', 0)
                # self.table.setItem(row, 10, QTableWidgetItem(f"{sharpe:.2f}" if sharpe != -1 else "N/A"))

                # Column 9: Actual. Sharpe (show "SELL" for sell orders)
                if action.lower() == 'sell':
                    actual_sharpe_display = "SELL"
                else:
                    actual_sharpe = self.get_actual_sharpe(order.get('ticker', ''), order.get('date', ''))
                    if isinstance(actual_sharpe, (int, float)):
                        if actual_sharpe == -1 or pd.isna(actual_sharpe):
                            actual_sharpe_display = "N/A"
                        else:
                            actual_sharpe_display = f"{actual_sharpe:.3f}"
                    else:
                        actual_sharpe_display = str(actual_sharpe)  # In case it already returns a string
                self.table.setItem(row, 9, QTableWidgetItem(actual_sharpe_display))

                # Column 10: Signal Strength (show "N/A" for sell orders or if missing)
                if action.lower() == 'sell':
                    signal_display = "SELL"
                else:
                    signal_strength = order.get('signal_strength', 0)
                    if pd.isna(signal_strength) or signal_strength == 0:
                        signal_display = "N/A"
                    else:
                        signal_display = f"{signal_strength:.3f}"
                self.table.setItem(row, 10, QTableWidgetItem(signal_display))

                for col in range(14):
                    if self.table.item(row, col):
                        self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.table.resizeColumnsToContents()
            logger.info(f"Updated recommendations with {len(orders_df)} orders")
        except Exception as e:
            logger.error(f"Error updating recommendations: {e}", exc_info=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to update trade history: {e}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()
            
    def get_actual_sharpe(self, ticker, date):
        """Retrieve actual Sharpe ratio for a given ticker and date."""
        try:
            if self.data_manager.data is None or self.data_manager.data.empty:
                logger.warning("Data manager is empty")
                return -1
            date = pd.to_datetime(date, utc=True)
            data = self.data_manager.data
            data['date'] = pd.to_datetime(data['date'], utc=True)
            mask = (data['Ticker'] == ticker) & (data['date'].dt.date == date.date())
            if mask.any():
                actual_sharpe = data.loc[mask, 'Actual_Sharpe'].iloc[0]
                return actual_sharpe if actual_sharpe != -1 else -1
            logger.debug(f"No actual Sharpe found for {ticker} on {date.date()}")
            return -1
        except Exception as e:
            logger.error(f"Error retrieving actual Sharpe for {ticker} on {date}: {e}")
            return -1
            
    def clear_trade_history(self):
        """Clear the trade history."""
        try:
            msg = QMessageBox()
            msg.setWindowTitle("Confirm Clear")
            msg.setText("Are you sure you want to clear the trade history?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setStyleSheet(self.get_message_box_style())
            if msg.exec() == QMessageBox.StandardButton.Yes:
                if os.path.exists(self.portfolio_state_file):
                    with open(self.portfolio_state_file, 'w') as f:
                        f.write('{"orders": [], "portfolio_history": []}')
                    logger.info(f"Cleared trade history in {self.portfolio_state_file}")
                else:
                    logger.info(f"No portfolio state file found at {self.portfolio_state_file}")
                self.update_recommendations()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Success")
                msg.setText("Trade history cleared successfully.")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)                
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec()
        except Exception as e:
            logger.error(f"Error clearing trade history: {e}", exc_info=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to clear trade history: {e}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)            
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()