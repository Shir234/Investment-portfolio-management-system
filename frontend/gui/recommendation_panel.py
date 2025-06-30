from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton, 
                             QMessageBox, QComboBox, QGroupBox, QFrame, QFileDialog)
from PyQt6.QtCore import Qt
from backend.trading_logic_new import get_orders, get_portfolio_history
import pandas as pd
import logging
from frontend.logging_config import get_logger
from frontend.gui.styles import ModernStyles
import os
import csv

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
        
        # Controls section - smaller
        self.create_controls_section(layout)
        
        # Table section - bigger (given more space)
        self.create_table_section(layout)
        
        # Actions section - single row at bottom
        self.create_actions_section(layout)
        
        self.apply_styles()
        self.update_recommendations()
        
    def create_controls_section(self, main_layout):
        """Create compact controls section"""
        controls_group = QGroupBox("Filter & Display Options")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(15, 20, 15, 15)  # Reduced padding
        controls_layout.setSpacing(15)
        
        # Filter section - more compact
        filter_label = QLabel("Show:")
        filter_label.setProperty("class", "label")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Orders", "Buy Orders", "Sell Orders"])
        self.filter_combo.setMinimumWidth(120)  # Smaller width
        self.filter_combo.currentIndexChanged.connect(self.update_recommendations)
        
        # Trade count section - more compact
        self.trade_count_label = QLabel("📊 Total Orders: 0")
        self.trade_count_label.setProperty("class", "metric")
        
        controls_layout.addWidget(filter_label)
        controls_layout.addWidget(self.filter_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.trade_count_label)
        
        # Give minimal stretch to controls section
        main_layout.addWidget(controls_group, stretch=0)
        
    def create_table_section(self, main_layout):
        """Create enlarged table section with maximum space allocation"""
        table_group = QGroupBox("Trade Details")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(20, 30, 20, 20)
        
        # Table for trade history
        self.table = QTableWidget()
        self.table.setColumnCount(11)  # Same columns
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
        self.table.setMinimumHeight(600)  # Even larger minimum height for more table space
        
        # Configure row headers (index column) to show row numbers properly
        self.table.verticalHeader().setVisible(False)
        
        # Set proper width for the vertical header (index column) so numbers show
        self.table.verticalHeader().setFixedWidth(600)  # Wider fixed width for row numbers
        self.table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        
        table_layout.addWidget(self.table)
        # Give the table section maximum stretch factor to occupy most space
        main_layout.addWidget(table_group, stretch=4)  # Increased stretch
        
    def create_actions_section(self, main_layout):
        """Create compact single-row actions section with only export"""
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(20, 20, 20, 15)  # Reduced vertical padding
        actions_layout.setSpacing(12)
        
        # Export button - now functional
        export_button = QPushButton("📊 Export to CSV")
        export_button.setProperty("class", "primary")
        export_button.clicked.connect(self.export_to_csv)
        export_button.setToolTip("Export trading history to CSV file")
        
        actions_layout.addStretch()  # Push button to center/right
        actions_layout.addWidget(export_button)
        actions_layout.addStretch()
        
        # Give minimal stretch to actions section
        main_layout.addWidget(actions_group, stretch=0)
        
    def export_to_csv(self):
        """Export trading history to CSV file"""
        try:
            orders = get_orders()
            if not orders:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("No Data")
                msg.setText("No trading history to export.")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec()
                return
            
            # Create DataFrame from orders
            orders_df = pd.DataFrame(orders)
            
            # Apply current filter
            filter_type = self.filter_combo.currentText()
            if filter_type == "Buy Orders":
                orders_df = orders_df[orders_df['action'] == 'buy']
            elif filter_type == "Sell Orders":
                orders_df = orders_df[orders_df['action'] == 'sell']
            
            if orders_df.empty:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("No Data")
                msg.setText(f"No {filter_type.lower()} to export.")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.setStyleSheet(self.get_message_box_style())
                msg.exec()
                return
            
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Trading History",
                f"trading_history_{filter_type.lower().replace(' ', '_')}.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Prepare data for export (same structure as table)
            export_data = []
            for _, order in orders_df.iterrows():
                # Calculate values same as in table
                price = order.get('price', 0)
                shares = order.get('shares_amount', 0)
                trade_value = price * shares
                action = order.get('action', '').upper()
                
                if action.lower() == 'buy':
                    total_cost = order.get('total_cost', 
                        order.get('investment_amount', 0) + order.get('transaction_cost', 0))
                    total_shares = order.get('new_total_shares', order.get('total_shares', 0))
                else:  # sell
                    total_cost = order.get('total_proceeds', 
                        trade_value - order.get('transaction_cost', 0))
                    total_shares = order.get('new_total_shares', 0)
                
                # Get Sharpe values
                if action.lower() == 'sell':
                    pred_sharpe = "SELL"
                    actual_sharpe = "SELL"
                    signal_strength = "SELL"
                else:
                    sharpe = order.get('sharpe', order.get('Best_Prediction', 0))
                    pred_sharpe = "N/A" if sharpe == -1 or pd.isna(sharpe) else f"{sharpe:.3f}"
                    
                    actual_sharpe_val = self.get_actual_sharpe(order.get('ticker', ''), order.get('date', ''))
                    if isinstance(actual_sharpe_val, (int, float)) and actual_sharpe_val != -1 and not pd.isna(actual_sharpe_val):
                        actual_sharpe = f"{actual_sharpe_val:.3f}"
                    else:
                        actual_sharpe = "N/A"
                    
                    signal_strength_val = order.get('signal_strength', 0)
                    if pd.isna(signal_strength_val) or signal_strength_val == 0:
                        signal_strength = "N/A"
                    else:
                        signal_strength = f"{signal_strength_val:.3f}"
                
                export_row = {
                    'Date': str(order.get('date', '')),
                    'Ticker': order.get('ticker', ''),
                    'Action': action,
                    'Price': f"{price:.2f}",
                    'Shares': str(shares),
                    'Trade Value': f"{trade_value:.2f}",
                    'Total Cost': f"{total_cost:.2f}",
                    'Total Shares': str(total_shares if not pd.isna(total_shares) else 0),
                    'Predicted Sharpe': pred_sharpe,
                    'Actual Sharpe': actual_sharpe,
                    'Signal Strength': signal_strength
                }
                export_data.append(export_row)
            
            # Write to CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
            
            # Show success message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Export Successful")
            msg.setText(f"Trading history exported successfully!\n\nFile: {file_path}\nRecords: {len(export_data)}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()
            
            logger.info(f"Exported {len(export_data)} records to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}", exc_info=True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Export Failed")
            msg.setText(f"Failed to export trading history:\n\n{e}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(self.get_message_box_style())
            msg.exec()
        
    def apply_styles(self):
        """Apply modern styling to the panel with improved table row colors for light mode"""
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        
        # Add enhanced table styling for better light mode contrast
        colors = ModernStyles.COLORS['dark'] if self.is_dark_mode else ModernStyles.COLORS['light']
        
        additional_table_styles = f"""
            /* Enhanced table styling with better alternating row colors */
            QTableWidget {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                border: 1px solid {colors['border_light']};
                border-radius: 12px;
                gridline-color: {colors['border_light']};
                font-size: 13px;
                selection-background-color: {colors['selected']};
                alternate-background-color: {'#2A2A3E' if self.is_dark_mode else '#E5E7EB'};
            }}
            
            QTableWidget::item {{
                border: none;
                padding: 12px 8px;
                border-bottom: 1px solid {colors['border_light']};
            }}
            
            QTableWidget::item:selected {{
                background-color: {colors['selected']};
                color: {'#FFFFFF' if self.is_dark_mode else colors['text_primary']};
            }}
            
            QTableWidget::item:hover {{
                background-color: {colors['hover']};
            }}
            
            QHeaderView::section {{
                background-color: {colors['secondary']};
                color: {colors['text_primary']};
                border: none;
                border-bottom: 2px solid {colors['border']};
                padding: 16px 8px;
                font-weight: 600;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            QHeaderView::section:hover {{
                background-color: {colors['hover']};
            }}
        """
        
        complete_style = style + additional_table_styles
        self.setStyleSheet(complete_style)
        
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