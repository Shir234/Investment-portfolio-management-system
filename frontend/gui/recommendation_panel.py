from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
                             QTableWidgetItem, QPushButton, QMessageBox, QComboBox, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from backend.trading_logic_new import get_orders, get_portfolio_history
import pandas as pd
import os
from ..logging_config import get_logger

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

class RecommendationPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.main_window = parent
        self.is_dark_mode = True  # Default to dark mode
        self.portfolio_state_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'data', 'portfolio_state.json')
        self.setup_ui()
        logger.info("RecommendationPanel initialized")

    def setup_ui(self):
        """Set up the UI for the recommendation panel with modern design."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title with shadow effect
        self.title_label = QLabel("Trading History")
        self.title_label.setStyleSheet(f"""
            font-size: 18px; 
            font-weight: bold; 
            font-family: 'Segoe UI'; 
            color: {'#FFFFFF' if self.is_dark_mode else '#2C2C2C'};
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 128))
        shadow.setOffset(2, 2)
        self.title_label.setGraphicsEffect(shadow)
        layout.addWidget(self.title_label)

        # Filter layout
        filter_layout = QHBoxLayout()
        self.filter_label = QLabel("Filter Orders:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Orders", "Buy Orders", "Sell Orders"])
        self.filter_combo.setToolTip("Filter trade history by order type")
        self.filter_combo.currentIndexChanged.connect(self.update_recommendations)
        filter_layout.addWidget(self.filter_label)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Trade count
        self.trade_count_label = QLabel("Number of Orders: 0")
        layout.addWidget(self.trade_count_label)

        # Table for trade history
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Date", "Ticker", "Action", "Price", "Shares", "Trade Value",
            "Total Cost", "Total Shares", "Pred. Sharpe", "Actual Sharpe", "Signal Strength"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setMinimumSectionSize(100)  # Ensure readable column widths
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Clear history button
        self.clear_button = QPushButton("Clear Trade History")
        self.clear_button.setToolTip("Clear all trade history and reset portfolio state")
        self.clear_button.clicked.connect(self.clear_trade_history)
        layout.addWidget(self.clear_button)

        # Apply initial theme
        self.set_theme(self.is_dark_mode)
        self.update_recommendations()

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        theme = THEME_COLORS['dark' if is_dark_mode else 'light']
        self.setStyleSheet(f"background-color: {theme['background']};")
        self.title_label.setStyleSheet(f"""
            font-size: 18px; 
            font-weight: bold; 
            font-family: 'Segoe UI'; 
            color: {theme['text']};
        """)
        label_style = f"color: {theme['text']}; font-family: 'Segoe UI'; font-size: 14px;"
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
        combo_style = f"""
            QComboBox {{
                background-color: {theme['card']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
                border-radius: 4px;
                padding: 6px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """
        self.filter_label.setStyleSheet(label_style)
        self.trade_count_label.setStyleSheet(label_style)
        self.filter_combo.setStyleSheet(combo_style)
        self.table.setStyleSheet(table_style)
        self.clear_button.setStyleSheet(button_style)
        logger.debug(f"Applied theme to RecommendationPanel: {'dark' if is_dark_mode else 'light'}")

    def get_message_box_style(self):
        """Return QMessageBox stylesheet based on theme."""
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
            QPushButton:pressed {{
                background-color: {theme['pressed']};
            }}
        """

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
                # Column 3: Price
                price = order.get('price', 0)
                self.table.setItem(row, 3, QTableWidgetItem(f"${price:,.2f}"))
                # Column 4: Shares
                shares = order.get('shares_amount', 0)
                self.table.setItem(row, 4, QTableWidgetItem(str(int(shares)) if shares == int(shares) else str(shares)))
                # Column 5: Trade Value
                trade_value = price * shares
                self.table.setItem(row, 5, QTableWidgetItem(f"${trade_value:,.2f}"))
                # Column 6: Total Cost
                if action.lower() == 'buy':
                    total_cost = order.get('total_cost', order.get('investment_amount', 0) + order.get('transaction_cost', 0))
                else:
                    total_cost = order.get('total_proceeds', trade_value - order.get('transaction_cost', 0))
                self.table.setItem(row, 6, QTableWidgetItem(f"${total_cost:,.2f}"))
                # Column 7: Total Shares
                if action.lower() == 'buy':
                    total_shares = order.get('new_total_shares', order.get('total_shares', 0))
                else:
                    total_shares = order.get('new_total_shares', 0)
                if pd.isna(total_shares) or total_shares is None:
                    total_shares = 0
                elif isinstance(total_shares, (int, float)) and total_shares == int(total_shares):
                    total_shares = int(total_shares)
                self.table.setItem(row, 7, QTableWidgetItem(str(total_shares)))
                # Column 8: Pred. Sharpe
                if action.lower() == 'sell':
                    sharpe_display = "SELL"
                else:
                    sharpe = order.get('sharpe', order.get('Best_Prediction', 0))
                    sharpe_display = "N/A" if sharpe == -1 or pd.isna(sharpe) else f"{sharpe:.3f}"
                self.table.setItem(row, 8, QTableWidgetItem(sharpe_display))
                # Column 9: Actual Sharpe
                if action.lower() == 'sell':
                    actual_sharpe_display = "SELL"
                else:
                    actual_sharpe = self.get_actual_sharpe(order.get('ticker', ''), order.get('date', ''))
                    actual_sharpe_display = "N/A" if isinstance(actual_sharpe, (int, float)) and (actual_sharpe == -1 or pd.isna(actual_sharpe)) else f"{actual_sharpe:.3f}"
                self.table.setItem(row, 9, QTableWidgetItem(actual_sharpe_display))
                # Column 10: Signal Strength
                if action.lower() == 'sell':
                    signal_display = "SELL"
                else:
                    signal_strength = order.get('signal_strength', 0)
                    signal_display = "N/A" if pd.isna(signal_strength) or signal_strength == 0 else f"{signal_strength:.3f}"
                self.table.setItem(row, 10, QTableWidgetItem(signal_display))

                # Center-align all cells
                for col in range(11):
                    if self.table.item(row, col):
                        self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table.resizeColumnsToContents()
            logger.info(f"Updated recommendations with {len(orders_df)} orders")
        except Exception as e:
            logger.error(f"Error updating recommendations: {e}", exc_info=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to update trade history: {str(e)}")
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
            msg = QMessageBox(self)
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
            msg.setText(f"Failed to clear trade history: {str(e)}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()