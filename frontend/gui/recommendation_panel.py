from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QMessageBox)
from PyQt5.QtCore import Qt
from data.trading_connector import get_order_history_df

class RecommendationPanel(QWidget):
    def __init__(self, data_manager, parent=None):  # Add parent parameter
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent  # Store reference to MainWindow
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Trading Recommendations")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Recommendations table
        self.table = QTableWidget()
        self.table.setColumnCount(9)  # Adjusted for all columns
        self.table.setHorizontalHeaderLabels([
            "Date", "Ticker", "Action", "Shares", "Price", 
            "Investment Amount", "Previous Shares", "New Shares", "Sharpe"
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
        """
        Update the recommendations table with the latest order history.
        
        This method fetches the order history from trading_connector and displays it in the table.
        The table shows a log of past trading actions with details like date, ticker, action, etc.
        """
        order_history = get_order_history_df()
        if order_history.empty:
            self.show_no_recommendations_alert()
            return

        self.table.setRowCount(0)
        for idx, row in order_history.iterrows():
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(row['date'])))
            self.table.setItem(row_position, 1, QTableWidgetItem(row['ticker']))
            self.table.setItem(row_position, 2, QTableWidgetItem(row['action']))
            self.table.setItem(row_position, 3, QTableWidgetItem(str(row['shares_amount'])))
            self.table.setItem(row_position, 4, QTableWidgetItem(f"${row['price']:,.2f}"))
            self.table.setItem(row_position, 5, QTableWidgetItem(f"${row['investment_amount']:,.2f}"))
            self.table.setItem(row_position, 6, QTableWidgetItem(str(row['previous_shares'])))
            self.table.setItem(row_position, 7, QTableWidgetItem(str(row['new_total_shares'])))
            self.table.setItem(row_position, 8, QTableWidgetItem(f"{row['sharpe']:.2f}"))

        self.table.resizeColumnsToContents()
        
    def show_no_recommendations_alert(self):
        """
        Display an alert when there is no order history to show.
        """
        self.alert_label.setText("No order history generated for the selected parameters.")
        self.alert_label.setVisible(True)

        # Optional: Add a message box to inform the user
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("No Orders Generated")
        msg.setText("No trading orders were generated for the selected date range and parameters. This could be due to:\n"
                    "- No stocks meeting the risk criteria.\n"
                    "- Insufficient investment amount.\n"
                    "- Data unavailable or filtered out for the selected dates.\n\n"
                    "Try adjusting the risk level, investment amount, or date range.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox { background-color: #353535; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #444; color: white; border: 1px solid #666; padding: 5px 15px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; }
        """)
        msg.exec_()