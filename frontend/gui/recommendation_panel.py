from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton, 
                             QMessageBox, QComboBox, QGroupBox, QFrame)
from PyQt6.QtCore import Qt
from backend.trading_logic_new import get_orders, get_portfolio_history
import pandas as pd
import logging
from frontend.logging_config import get_logger
from frontend.gui.styles import ModernStyles
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
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Header section
        self.create_header_section(layout)
        
        # Controls section
        self.create_controls_section(layout)
        
        # Table section
        self.create_table_section(layout)
        
        # Actions section
        self.create_actions_section(layout)
        
        self.apply_styles()
        self.update_recommendations()
        
    def create_header_section(self, main_layout):
        """Create modern header section"""
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(8)
        
        self.title_label = QLabel("Trading History & Analysis")
        self.title_label.setProperty("class", "title")
        header_layout.addWidget(self.title_label)
        
        self.subtitle_label = QLabel("Review your executed trades and performance metrics")
        self.subtitle_label.setProperty("class", "subtitle")
        header_layout.addWidget(self.subtitle_label)
        
        main_layout.addWidget(header_frame)
        
    def create_controls_section(self, main_layout):
        """Create modern controls section"""
        controls_group = QGroupBox("Filter & Display Options")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(20, 30, 20, 20)
        controls_layout.setSpacing(20)
        
        # Filter section
        filter_frame = QFrame()
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setSpacing(12)
        
        filter_label = QLabel("Show:")
        filter_label.setProperty("class", "label")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Orders", "Buy Orders", "Sell Orders"])
        self.filter_combo.setMinimumWidth(150)
        self.filter_combo.currentIndexChanged.connect(self.update_recommendations)
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        
        # Trade count section
        count_frame = QFrame()
        count_layout = QHBoxLayout(count_frame)
        
        self.trade_count_label = QLabel("📊 Total Orders: 0")
        self.trade_count_label.setProperty("class", "metric")
        count_layout.addWidget(self.trade_count_label)
        
        controls_layout.addWidget(filter_frame, 1)
        controls_layout.addWidget(count_frame, 1)
        
        main_layout.addWidget(controls_group)
        
    def create_table_section(self, main_layout):
        """Create modern table section"""
        table_group = QGroupBox("Trade Details")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(20, 30, 20, 20)
        
        # Table for trade history
        self.table = QTableWidget()
        self.table.setColumnCount(11)  # Reduced columns for better readability
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
        
        # Modern table styling
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(400)
        
        table_layout.addWidget(self.table)
        main_layout.addWidget(table_group, stretch=2)
        
    def create_actions_section(self, main_layout):
        """Create modern actions section"""
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(20, 30, 20, 20)
        actions_layout.setSpacing(12)
        
        # Export button (future feature)
        export_button = QPushButton("📊 Export to CSV")
        export_button.setProperty("class", "secondary")
        export_button.setEnabled(False)  # Disabled for now
        export_button.setToolTip("Export trading history to CSV file (Coming Soon)")
        
        # Refresh button
        refresh_button = QPushButton("🔄 Refresh")
        refresh_button.clicked.connect(self.update_recommendations)
        
        # Clear history button
        self.clear_button = QPushButton("🗑️ Clear History")
        self.clear_button.setProperty("class", "danger")
        self.clear_button.clicked.connect(self.clear_trade_history)
        
        actions_layout.addWidget(export_button)
        actions_layout.addStretch()
        actions_layout.addWidget(refresh_button)
        actions_layout.addWidget(self.clear_button)
        
        main_layout.addWidget(actions_group)
        
    def apply_styles(self):
        """Apply modern styling to the panel"""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        self.setStyleSheet(style)
        
    def get_message_box_style(self):
        """Return QMessageBox stylesheet based on theme."""
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        return f"""
            QMessageBox {{ 
                background-color: {colors['primary']}; 
                color: {colors['text_primary']}; 
                font-size: 14px;
                border-radius: 12px;
            }}
            QMessageBox QLabel {{ 
                color: {colors['text_primary']}; 
                padding: 16px;
            }}
            QMessageBox QPushButton {{ 
                background-color: {colors['accent']}; 
                color: white; 
                border: none;
                border-radius: 6px;
                padding: 8px 16px; 
                font-weight: 600;
                min-width: 80px;
                margin: 4px;
            }}
            QMessageBox QPushButton:hover {{ 
                background-color: {colors['accent_hover']}; 
            }}
        """
        
    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the panel."""
        self.is_dark_mode = is_dark_mode
        self.apply_styles()
        self.update_recommendations()
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")
        
    def update_recommendations(self):
        """Update the table with trade history."""
        try:
            orders = get_orders()
            if not orders:
                self.table.setRowCount(0)
                self.trade_count_label.setText("📊 Total Orders: 0")
                logger.info("No orders found")
                return
            
            orders_df = pd.DataFrame(orders)
            filter_type = self.filter_combo.currentText()
            
            # Apply filter
            if filter_type == "Buy Orders":
                orders_df = orders_df[orders_df['action'] == 'buy']
            elif filter_type == "Sell Orders":
                orders_df = orders_df[orders_df['action'] == 'sell']
            
            self.table.setRowCount(len(orders_df))
            self.trade_count_label.setText(f"📊 Total Orders: {len(orders_df)}")
            
            # Define colors for different action types
            colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
            
            for row, (_, order) in enumerate(orders_df.iterrows()):
                # Column 0: Date
                date_item = QTableWidgetItem(str(order.get('date', '')))
                self.table.setItem(row, 0, date_item)
                
                # Column 1: Ticker
                ticker_item = QTableWidgetItem(order.get('ticker', ''))
                self.table.setItem(row, 1, ticker_item)
                
                # Column 2: Action (with color coding)
                action = order.get('action', '').upper()
                action_item = QTableWidgetItem(action)
                if action == 'BUY':
                    action_item.setForeground(self.get_color_from_hex(colors['success']))
                elif action == 'SELL':
                    action_item.setForeground(self.get_color_from_hex(colors['danger']))
                self.table.setItem(row, 2, action_item)

                # Column 3: Price
                price = order.get('price', 0)
                price_item = QTableWidgetItem(f"${price:,.2f}")
                self.table.setItem(row, 3, price_item)
                
                # Column 4: Shares
                shares = order.get('shares_amount', 0)
                shares_item = QTableWidgetItem(str(shares))
                self.table.setItem(row, 4, shares_item)
                
                # Column 5: Trade Value (shares * price)
                trade_value = price * shares
                trade_value_item = QTableWidgetItem(f"${trade_value:,.2f}")
                self.table.setItem(row, 5, trade_value_item)
                
                # Column 6: Total Cost
                if action.lower() == 'buy':
                    total_cost = order.get('total_cost', 
                        order.get('investment_amount', 0) + order.get('transaction_cost', 0))
                else:  # sell
                    total_cost = order.get('total_proceeds', 
                        trade_value - order.get('transaction_cost', 0))
                total_cost_item = QTableWidgetItem(f"${total_cost:,.2f}")
                self.table.setItem(row, 6, total_cost_item)

                # Column 7: Total Shares
                if action.lower() == 'buy':
                    total_shares = order.get('new_total_shares', order.get('total_shares', 0))
                else:  # sell
                    total_shares = order.get('new_total_shares', 0)
                if pd.isna(total_shares) or total_shares is None:
                    total_shares = 0
                elif isinstance(total_shares, (int, float)) and total_shares == int(total_shares):
                    total_shares = int(total_shares)
                total_shares_item = QTableWidgetItem(str(total_shares))
                self.table.setItem(row, 7, total_shares_item)

                # Column 8: Predicted Sharpe
                if action.lower() == 'sell':
                    sharpe_display = "SELL"
                    sharpe_item = QTableWidgetItem(sharpe_display)
                    sharpe_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                else:
                    sharpe = order.get('sharpe', order.get('Best_Prediction', 0))
                    if sharpe == -1 or pd.isna(sharpe):
                        sharpe_display = "N/A"
                        sharpe_item = QTableWidgetItem(sharpe_display)
                        sharpe_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                    else:
                        sharpe_display = f"{sharpe:.3f}"
                        sharpe_item = QTableWidgetItem(sharpe_display)
                        # Color code based on Sharpe value
                        if sharpe > 1:
                            sharpe_item.setForeground(self.get_color_from_hex(colors['success']))
                        elif sharpe > 0:
                            sharpe_item.setForeground(self.get_color_from_hex(colors['warning']))
                        else:
                            sharpe_item.setForeground(self.get_color_from_hex(colors['danger']))
                self.table.setItem(row, 8, sharpe_item)

                # Column 9: Actual Sharpe
                if action.lower() == 'sell':
                    actual_sharpe_display = "SELL"
                    actual_item = QTableWidgetItem(actual_sharpe_display)
                    actual_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                else:
                    actual_sharpe = self.get_actual_sharpe(order.get('ticker', ''), order.get('date', ''))
                    if isinstance(actual_sharpe, (int, float)):
                        if actual_sharpe == -1 or pd.isna(actual_sharpe):
                            actual_sharpe_display = "N/A"
                            actual_item = QTableWidgetItem(actual_sharpe_display)
                            actual_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                        else:
                            actual_sharpe_display = f"{actual_sharpe:.3f}"
                            actual_item = QTableWidgetItem(actual_sharpe_display)
                            # Color code based on actual Sharpe value
                            if actual_sharpe > 1:
                                actual_item.setForeground(self.get_color_from_hex(colors['success']))
                            elif actual_sharpe > 0:
                                actual_item.setForeground(self.get_color_from_hex(colors['warning']))
                            else:
                                actual_item.setForeground(self.get_color_from_hex(colors['danger']))
                    else:
                        actual_sharpe_display = str(actual_sharpe)
                        actual_item = QTableWidgetItem(actual_sharpe_display)
                        actual_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                self.table.setItem(row, 9, actual_item)

                # Column 10: Signal Strength
                if action.lower() == 'sell':
                    signal_display = "SELL"
                    signal_item = QTableWidgetItem(signal_display)
                    signal_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                else:
                    signal_strength = order.get('signal_strength', 0)
                    if pd.isna(signal_strength) or signal_strength == 0:
                        signal_display = "N/A"
                        signal_item = QTableWidgetItem(signal_display)
                        signal_item.setForeground(self.get_color_from_hex(colors['text_muted']))
                    else:
                        signal_display = f"{signal_strength:.3f}"
                        signal_item = QTableWidgetItem(signal_display)
                        # Color code based on signal strength
                        if abs(signal_strength) > 2:
                            signal_item.setForeground(self.get_color_from_hex(colors['success']))
                        elif abs(signal_strength) > 1:
                            signal_item.setForeground(self.get_color_from_hex(colors['warning']))
                        else:
                            signal_item.setForeground(self.get_color_from_hex(colors['text_secondary']))
                self.table.setItem(row, 10, signal_item)

                # Center align all items
                for col in range(11):
                    if self.table.item(row, col):
                        self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Auto-resize columns for better display
            self.table.resizeColumnsToContents()
            logger.info(f"Updated recommendations with {len(orders_df)} orders")
            
        except Exception as e:
            logger.error(f"Error updating recommendations: {e}", exc_info=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Failed to update trade history:\n\n{e}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()
            
    def get_color_from_hex(self, hex_color):
        """Convert hex color string to QColor"""
        from PyQt6.QtGui import QColor
        return QColor(hex_color)
            
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
        """Clear the trade history with confirmation."""
        try:
            msg = QMessageBox()
            msg.setWindowTitle("Confirm Clear History")
            msg.setText("Are you sure you want to clear all trade history?\n\nThis action cannot be undone.")
            msg.setIcon(QMessageBox.Icon.Warning)
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
                
                # Show success message
                success_msg = QMessageBox()
                success_msg.setIcon(QMessageBox.Icon.Information)
                success_msg.setWindowTitle("Success")
                success_msg.setText("Trade history cleared successfully.")
                success_msg.setStandardButtons(QMessageBox.StandardButton.Ok)                
                success_msg.setStyleSheet(self.get_message_box_style())
                success_msg.exec()
                
        except Exception as e:
            logger.error(f"Error clearing trade history: {e}", exc_info=True)
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.setWindowTitle("Error")
            error_msg.setText(f"Failed to clear trade history:\n\n{e}")
            error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)            
            error_msg.setStyleSheet(self.get_message_box_style())
            error_msg.exec()