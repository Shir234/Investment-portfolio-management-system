# semi_automated_manager.py
import pandas as pd
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar, QTextEdit, QScrollArea, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from frontend.logging_config import get_logger
from frontend.gui.worker import Worker
from backend.trading_logic_new import get_orders, get_portfolio_history

logger = get_logger(__name__)

class TradeValidationResult:
    """Holds validation results for trade dependencies."""
    def __init__(self):
        self.valid_orders = []
        self.invalid_orders = []
        self.warnings = []
        self.dependencies = {}  # order_id -> [dependent_order_ids]

class TradeDependencyValidator:
    """Validates trade dependencies and portfolio state."""
    
    def __init__(self, current_cash, current_holdings):
        self.current_cash = current_cash
        self.current_holdings = current_holdings.copy()
        
    def validate_order_sequence(self, orders):
        """Validate orders individually, not cumulatively."""
        result = TradeValidationResult()
        
        for order in orders:
            validation = self._validate_single_order(order, self.current_cash, self.current_holdings)
            
            if validation['valid']:
                result.valid_orders.append(order)
            else:
                result.invalid_orders.append(order)
                order_date = order['date']
                if isinstance(order_date, str):
                    order_date = pd.to_datetime(order_date, utc=True)
                date_str = order_date.date() if order_date else ''
                result.warnings.append(
                    f"{date_str} - {order['ticker']} {order['action']}: {validation['reason']}"
                )
        
        return result
    
    def _validate_single_order(self, order, cash, holdings):
        """Validate a single order against current portfolio state."""
        ticker = order['ticker']
        action = order['action']
        shares = order['shares_amount']
        total_cost = order.get('total_cost', order.get('investment_amount', 0))
        
        if action == 'buy':
            # Check if we have enough cash for THIS order alone
            if total_cost > cash:
                return {
                    'valid': False,
                    'reason': f"Insufficient cash: need ${total_cost:,.2f}, have ${cash:,.2f}"
                }
            return {'valid': True, 'reason': 'Valid buy order'}
            
        elif action == 'sell':
            # Check if we have enough shares for THIS order alone
            current_shares = holdings.get(ticker, {}).get('shares', 0)
            if shares > current_shares:
                return {
                    'valid': False,
                    'reason': f"Insufficient shares: trying to sell {shares}, have {current_shares}"
                }
            return {'valid': True, 'reason': 'Valid sell order'}
        
        return {'valid': False, 'reason': 'Unknown order type'}
    
    def _apply_order_to_state(self, order, cash, holdings):
        """Apply an order to portfolio state simulation."""
        ticker = order['ticker']
        action = order['action']
        shares = order['shares_amount']
        price = order['price']
        
        if action == 'buy':
            total_cost = order.get('total_cost', shares * price)
            cash -= total_cost
            
            if ticker in holdings:
                # Average cost calculation
                old_shares = holdings[ticker]['shares']
                old_price = holdings[ticker]['purchase_price']
                new_shares = old_shares + shares
                new_avg_price = ((old_shares * old_price) + (shares * price)) / new_shares
                
                holdings[ticker]['shares'] = new_shares
                holdings[ticker]['purchase_price'] = new_avg_price
            else:
                holdings[ticker] = {
                    'shares': shares,
                    'purchase_price': price,
                    'purchase_date': order['date']
                }
                
        elif action == 'sell':
            proceeds = order.get('total_proceeds', shares * price)
            cash += proceeds
            
            if ticker in holdings:
                holdings[ticker]['shares'] -= shares
                if holdings[ticker]['shares'] <= 0:
                    del holdings[ticker]
        
        return cash, holdings

class WindowedTradeConfirmationDialog(QDialog):
    """Enhanced dialog for windowed semi-automated trading."""
    
    def __init__(self, window_start, window_end, orders, current_cash, current_holdings, parent=None):
        super().__init__(parent)
        self.window_start = window_start
        self.window_end = window_end
        self.orders = orders
        
        # Initialize cash FIRST
        portfolio_history = get_portfolio_history()
        if portfolio_history:
            self.current_cash = portfolio_history[-1].get('cash', current_cash)
        else:
            self.current_cash = current_cash
        
        # Initialize tracking variables
        self.selected_value = 0.0  # Track selected orders value
        self.current_holdings = current_holdings
        self.selected_orders = []
        self.is_dark_mode = getattr(parent, 'is_dark_mode', True)
        
        self.setWindowTitle(f"Trading Window: {window_start.date()} to {window_end.date()}")
        self.setFixedSize(1000, 700)
        self.setup_ui()
        self.validate_and_display_orders()
             
    def setup_ui(self):
        """Setup the enhanced UI with validation information."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header_label = QLabel(f"Trading Window: {self.window_start.date()} to {self.window_end.date()}")
        header_label.setProperty("class", "title")
        layout.addWidget(header_label)

        # Portfolio status with dynamic cash display
        status_layout = QHBoxLayout()
        self.cash_label = QLabel(f"Available Cash: ${self.current_cash:,.2f}")
        self.cash_label.setProperty("class", "subtitle")
        
        self.selected_label = QLabel(f"Selected Value: $0.00")
        self.selected_label.setProperty("class", "subtitle")
        
        self.remaining_label = QLabel(f"Remaining: ${self.current_cash:,.2f}")
        self.remaining_label.setProperty("class", "subtitle")
        
        status_layout.addWidget(self.cash_label)
        status_layout.addStretch()
        status_layout.addWidget(self.selected_label)
        status_layout.addWidget(self.remaining_label)
        layout.addLayout(status_layout)

        holdings_label = QLabel(f"Holdings: {len(self.current_holdings)} positions")
        holdings_label.setProperty("class", "subtitle")
        layout.addWidget(holdings_label)

        # Validation warnings area
        self.warnings_text = QTextEdit()
        self.warnings_text.setMaximumHeight(100)
        self.warnings_text.setPlaceholderText("Trade validation warnings will appear here...")
        layout.addWidget(self.warnings_text)

        # Orders table
        self.setup_orders_table(layout)

        # Action buttons
        self.setup_action_buttons(layout)

        # Apply styling
        self.apply_styles()

    def setup_orders_table(self, layout):
        """Setup the orders table with validation indicators."""
        self.table = QTableWidget()
        self.table.setRowCount(len(self.orders))
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Select", "Date", "Ticker", "Action", "Shares", "Price", "Value", "Status"
        ])
        
        # Set column widths
        column_widths = [80, 140, 80, 80, 80, 100, 120, 120]
        for i, width in enumerate(column_widths):
            self.table.setColumnWidth(i, width)
        
        self.table.verticalHeader().setDefaultSectionSize(40)  # Increase row height
        layout.addWidget(self.table)

    def setup_action_buttons(self, layout):
        """Setup action buttons with additional options."""
        button_layout = QHBoxLayout()
        
        # Select all valid button
        self.select_all_valid_button = QPushButton("Select All Valid")
        self.select_all_valid_button.setProperty("class", "secondary")
        self.select_all_valid_button.clicked.connect(self.select_all_valid_trades)
        
        # Clear selection button
        self.clear_selection_button = QPushButton("Clear Selection")
        self.clear_selection_button.setProperty("class", "secondary")
        self.clear_selection_button.clicked.connect(self.clear_all_selections)
        
        button_layout.addWidget(self.select_all_valid_button)
        button_layout.addWidget(self.clear_selection_button)
        button_layout.addStretch()
        
        # Main action buttons
        cancel_button = QPushButton("Skip Window")
        cancel_button.setProperty("class", "secondary")
        cancel_button.clicked.connect(self.reject)
        
        self.execute_button = QPushButton("Execute Selected")
        self.execute_button.setProperty("class", "success")
        self.execute_button.clicked.connect(self.accept_selected)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.execute_button)
        layout.addLayout(button_layout)

    def validate_and_display_orders(self):
        """Validate orders and display them with status indicators."""
        validator = TradeDependencyValidator(self.current_cash, self.current_holdings)
        validation_result = validator.validate_order_sequence(self.orders)
        
        # Display warnings
        if validation_result.warnings:
            warning_text = "Trade Validation Warnings:\n" + "\n".join(validation_result.warnings)
            self.warnings_text.setPlainText(warning_text)
        else:
            self.warnings_text.setPlainText("All trades passed validation checks")

        # Populate table
        all_orders = validation_result.valid_orders + validation_result.invalid_orders
        for row, order in enumerate(all_orders):
            is_valid = order in validation_result.valid_orders
            
            # Checkbox
            checkbox_widget = self.create_custom_checkbox(enabled=is_valid)
            self.table.setCellWidget(row, 0, checkbox_widget)

            # Order details
            order_date = order.get('date', '')
            # Convert date string to pd.Timestamp if necessary
            if isinstance(order_date, str):
                order_date = pd.to_datetime(order_date, utc=True)
            date_str = str(order_date.date()) if order_date else ''

            self.table.setItem(row, 1, QTableWidgetItem(date_str))
            self.table.setItem(row, 2, QTableWidgetItem(order.get('ticker', '')))
            self.table.setItem(row, 3, QTableWidgetItem(order.get('action', '').upper()))
            self.table.setItem(row, 4, QTableWidgetItem(str(order.get('shares_amount', 0))))
            self.table.setItem(row, 5, QTableWidgetItem(f"${order.get('price', 0):,.2f}"))
            
            # Calculate value
            value = order.get('shares_amount', 0) * order.get('price', 0)
            self.table.setItem(row, 6, QTableWidgetItem(f"${value:,.2f}"))
            
            # Status
            status = " Valid" if is_valid else " Invalid"
            status_item = QTableWidgetItem(status)
            if not is_valid:
                status_item.setBackground(Qt.GlobalColor.red)
            self.table.setItem(row, 7, status_item)

            # Center align all items
            for col in range(1, 8):
                if self.table.item(row, col):
                    self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def create_custom_checkbox(self, enabled=True):
        """Create checkbox with enabled/disabled state."""        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        check_label = QLabel()
        check_label.setFixedSize(24, 24)
        check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        check_label.checked = False
        check_label.enabled = enabled
        
        def update_appearance():
            if not check_label.enabled:
                check_label.setText("🚫")
                check_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #ccc;
                        border-radius: 6px;
                        background-color: #f5f5f5;
                        color: #999;
                    }
                """)
            elif check_label.checked:
                check_label.setText("✓")
                check_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #4CAF50;
                        border-radius: 6px;
                        background-color: #4CAF50;
                        font-size: 16px;
                        font-weight: bold;
                        color: white;
                    }
                    QLabel:hover {
                        background-color: #66bb6a;
                        border: 3px solid #66bb6a;
                    }
                """)
            else:
                check_label.setText("")
                check_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #ccc;
                        border-radius: 6px;
                        background-color: white;
                    }
                    QLabel:hover {
                        background-color: #f5f5f5;
                        border: 3px solid #999;
                    }
                """)
        
        def toggle_check(event):
            if check_label.enabled:
                new_state = not check_label.checked
                # Check if we can afford this selection
                if new_state:
                    if not self.can_afford_selection(widget):
                        return  # Don't allow selection if can't afford
                
                check_label.checked = new_state
                update_appearance()
                self.update_selected_totals()  # Update totals when selection changes
        
        check_label.mousePressEvent = toggle_check
        update_appearance()
        
        layout.addWidget(check_label)
        
        widget.is_checked = lambda: check_label.checked if check_label.enabled else False
        widget.set_checked = lambda checked: setattr(check_label, 'checked', checked) or update_appearance()
        
        return widget

    def select_all_valid_trades(self):
        """Select all valid trades that we can afford."""
        for row in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(row, 0)
            status_item = self.table.item(row, 7)
            if status_item and "Valid" in status_item.text():
                order = self.orders[row]
                if order.get('action') == 'sell' or self.can_afford_selection(checkbox_widget):
                    checkbox_widget.set_checked(True)
        self.update_selected_totals()

    def clear_all_selections(self):
        """Clear all selections."""
        for row in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(row, 0)
            checkbox_widget.set_checked(False)
        self.update_selected_totals()

    def accept_selected(self):
        """Collect selected valid orders."""
        self.selected_orders = []
        for row in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(row, 0)
            if checkbox_widget and checkbox_widget.is_checked():
                self.selected_orders.append(self.orders[row])
        
        if not self.selected_orders:
            QMessageBox.warning(self, "No Selection", "Please select at least one trade to execute.")
            return
            
        self.accept()

    def apply_styles(self):
        """Apply modern styling."""
        from frontend.gui.styles import ModernStyles
        style = ModernStyles.get_complete_style(self.is_dark_mode)
        self.setStyleSheet(style)
    
    def can_afford_selection(self, checkbox_widget):
        """Check if we can afford to select this order."""
        # Find which row this checkbox belongs to
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 0) == checkbox_widget:
                order = self.orders[row]
                order_value = order.get('total_cost', order.get('shares_amount', 0) * order.get('price', 0))
                
                # Only check buy orders against cash
                if order.get('action') == 'buy':
                    return self.selected_value + order_value <= self.current_cash
                return True  # Sell orders are always allowed
        return False

    def update_selected_totals(self):
        """Update the selected value and remaining cash displays."""
        total_buy_value = 0.0
        total_sell_value = 0.0
        
        for row in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(row, 0)
            if checkbox_widget and checkbox_widget.is_checked():
                order = self.orders[row]
                if order.get('action') == 'buy':
                    total_buy_value += order.get('total_cost', order.get('shares_amount', 0) * order.get('price', 0))
                elif order.get('action') == 'sell':
                    total_sell_value += order.get('total_proceeds', order.get('shares_amount', 0) * order.get('price', 0))
        
        self.selected_value = total_buy_value
        net_cash_change = total_sell_value - total_buy_value
        remaining_cash = self.current_cash + net_cash_change
        
        # Update labels
        self.selected_label.setText(f"Selected Buys: ${total_buy_value:,.2f}")
        if total_sell_value > 0:
            self.selected_label.setText(f"Buys: ${total_buy_value:,.2f} | Sells: ${total_sell_value:,.2f}")
        
        self.remaining_label.setText(f"Remaining: ${remaining_cash:,.2f}")
        
        # Update remaining cash color based on availability
        if remaining_cash < 0:
            self.remaining_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.remaining_label.setStyleSheet("color: green; font-weight: bold;")
        
        # Disable checkboxes for buy orders we can't afford
        self.update_checkbox_availability()

    def update_checkbox_availability(self):
        """Enable/disable checkboxes based on available cash."""
        for row in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(row, 0)
            if checkbox_widget and not checkbox_widget.is_checked():
                order = self.orders[row]
                if order.get('action') == 'buy':
                    order_value = order.get('total_cost', order.get('shares_amount', 0) * order.get('price', 0))
                    can_afford = self.selected_value + order_value <= self.current_cash
                    
                    # Get the checkbox label
                    check_label = checkbox_widget.findChild(QLabel)
                    if check_label:
                        check_label.enabled = can_afford
                        # Update appearance based on affordability
                        if not can_afford:
                            check_label.setText("💰")
                            check_label.setStyleSheet("""
                                QLabel {
                                    border: 2px solid #ff9800;
                                    border-radius: 6px;
                                    background-color: #fff3e0;
                                    color: #ff9800;
                                }
                            """)
                        else:
                            # FIXED: Restore normal unchecked appearance
                            check_label.setText("")
                            check_label.setStyleSheet("""
                                QLabel {
                                    border: 3px solid #ccc;
                                    border-radius: 6px;
                                    background-color: white;
                                }
                                QLabel:hover {
                                    background-color: #f5f5f5;
                                    border: 3px solid #999;
                                }
                            """)

class SemiAutomatedManager:
    """Manages the windowed semi-automated trading process using existing Worker class."""
    
    def __init__(self, input_panel, main_window):
        self.input_panel = input_panel
        self.main_window = main_window
        self.data_manager = input_panel.data_manager
        
    def start_windowed_trading(self, investment_amount, risk_level, start_date, end_date):
        """Start the windowed semi-automated trading process with USER parameters."""
        logger.info(f"Starting windowed semi-automated trading from {start_date} to {end_date}")
        logger.info(f"User settings: Investment=${investment_amount}, Risk={risk_level}")
        
        # Get window size from user input (if available)
        if hasattr(self.input_panel, 'window_size_input'):
            window_size_days = self.input_panel.window_size_input.value()
        else:
            window_size_days = 5  # Default fallback
        
        # Calculate trading windows
        windows = self._create_trading_windows(start_date, end_date, window_size_days)
        
        if not windows:
            self.input_panel.show_message_box(
                QMessageBox.Icon.Warning,
                "No Trading Windows",
                "No valid trading windows found for the selected date range.",
                QMessageBox.StandardButton.Ok
            )
            return
        
        # Store user parameters for the entire session
        self.user_investment_amount = investment_amount
        self.user_risk_level = risk_level
        self.user_start_date = start_date
        self.user_end_date = end_date
        self.window_size_days = window_size_days
        
        # Start processing windows sequentially
        self.current_window_index = 0
        self.trading_windows = windows
        
        # Show overview dialog
        self._show_trading_overview()
        
    def _create_trading_windows(self, start_date, end_date, window_size_days):
        """Create trading windows of specified size."""
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            window_end = min(
                current_start + pd.Timedelta(days=window_size_days),
                end_date
            )
            
            # Only create window if it has at least 1 day
            if (window_end - current_start).days >= 1:
                windows.append((current_start, window_end))
            
            current_start = window_end
            
        return windows
    
    def _show_trading_overview(self):
        """Show overview of trading windows before starting."""
        overview_text = f"Semi-Automated Trading Plan:\n\n"
        overview_text += f"Investment Amount: ${self.user_investment_amount:,.2f}\n"
        overview_text += f"Risk Level: {self.user_risk_level}/10\n"
        overview_text += f"Total Windows: {len(self.trading_windows)}\n"
        overview_text += f"Window Size: {self.window_size_days} days\n\n"
        overview_text += "Windows:\n"
        
        for i, (start, end) in enumerate(self.trading_windows, 1):
            days = (end - start).days
            overview_text += f"  {i}. {start.date()} to {end.date()} ({days} days)\n"
        
        overview_text += f"\nYou'll review and approve trades for each window separately."
        
        # Create a custom dialog with fixed size and scrollable text
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton
        
        dialog = QDialog(self.input_panel)
        dialog.setWindowTitle("Trading Overview")
        dialog.setFixedSize(500, 400)  # Fixed size
        
        layout = QVBoxLayout(dialog)
        
        # Scrollable text area
        text_edit = QTextEdit()
        text_edit.setPlainText(overview_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        # Buttons at bottom
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        ok_button = QPushButton("OK")
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)
        
        # Connect buttons
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._process_next_window()
        else:
            logger.info("User cancelled windowed trading")
    
    def _process_next_window(self):
        """Process the next trading window using REAL user parameters."""
        if self.current_window_index >= len(self.trading_windows):
            self._finish_windowed_trading()
            return
        
        window_start, window_end = self.trading_windows[self.current_window_index]
        logger.info(f"Processing window {self.current_window_index + 1}/{len(self.trading_windows)}: {window_start.date()} to {window_end.date()}")
        
        # Show progress
        self.input_panel.show_progress(f"Analyzing window {self.current_window_index + 1}/{len(self.trading_windows)}...")
        
        self.thread = QThread()
        # Get current portfolio state for this window
        portfolio_history = get_portfolio_history()
        if portfolio_history:
            latest_state = portfolio_history[-1]
            current_cash = latest_state.get('cash', self.user_investment_amount)
            current_holdings = latest_state.get('holdings', {})
        else:
            current_cash = self.user_investment_amount
            current_holdings = {}

        self.worker = Worker(
            investment_amount=current_cash,                 # Use current cash, not initial
            risk_level=self.user_risk_level,
            start_date=window_start,
            end_date=window_end,
            data_manager=self.data_manager,
            mode="semi-automatic",                          # Changed to semi-automatic
            reset_state=False,
            selected_orders=None,
            current_cash=current_cash,
            current_holdings=current_holdings
        )
        
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_window_result)
        self.worker.error.connect(self._handle_window_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
    
    def _handle_window_result(self, success, result):
        """Handle the result of a trading window analysis."""
        self.input_panel.hide_progress()
        
        if not success:
            self.input_panel.show_message_box(
                QMessageBox.Icon.Critical,
                "Window Analysis Failed",
                f"Failed to analyze window {self.current_window_index + 1}:\n\n{result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
            self._skip_to_next_window()
            return
        
        orders = result.get('orders', [])
        
        if not orders:
            # No trades in this window, move to next
            self.input_panel.show_message_box(
                QMessageBox.Icon.Information,
                "No Trades Found",
                f"No trading opportunities found for window {self.current_window_index + 1}.",
                QMessageBox.StandardButton.Ok
            )
            self._skip_to_next_window()
            return
        
        # Get current portfolio state for validation
        portfolio_history = get_portfolio_history()
        if portfolio_history:
            latest_state = portfolio_history[-1]
            current_cash = latest_state.get('cash', self.user_investment_amount)
            current_holdings = latest_state.get('holdings', {})
        else:
            current_cash = self.user_investment_amount
            current_holdings = {}
        
        # Show windowed confirmation dialog with REAL data
        window_start, window_end = self.trading_windows[self.current_window_index]
        dialog = WindowedTradeConfirmationDialog(
            window_start, window_end, orders, current_cash, current_holdings, self.input_panel
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_orders:
            self._execute_window_trades(dialog.selected_orders)
        else:
            self._skip_to_next_window()
    
    def _execute_window_trades(self, selected_orders):
        """Execute the selected trades for the current window using EXISTING logic."""
        self.input_panel.show_progress("Executing selected trades...")
        
        # Use your EXISTING Worker class to execute selected trades
        self.execution_thread = QThread()
        self.execution_worker = Worker(
            investment_amount=self.user_investment_amount,
            risk_level=self.user_risk_level,
            start_date=self.user_start_date,
            end_date=self.user_end_date,
            data_manager=self.data_manager,
            mode="semi-automatic",
            reset_state=False,
            selected_orders=selected_orders,
            current_cash=None,  # Will use current state
            current_holdings=None  # Will use current state
        )
        
        self.execution_worker.moveToThread(self.execution_thread)
        self.execution_thread.started.connect(self.execution_worker.run)
        self.execution_worker.finished.connect(self._handle_execution_result)
        self.execution_worker.error.connect(self._handle_window_error)
        self.execution_worker.finished.connect(self.execution_thread.quit)
        self.execution_worker.finished.connect(self.execution_worker.deleteLater)
        self.execution_thread.finished.connect(self.execution_thread.deleteLater)
        
        self.execution_thread.start()
    
    def _handle_execution_result(self, success, result):
        """Handle execution results."""
        self.input_panel.hide_progress()
        
        if success:
            orders = result.get('orders', [])
            portfolio_value = result.get('portfolio_value', 0)
            cash = result.get('cash', 0)
            
            logger.info(f"Successfully executed {len(orders)} trades for window {self.current_window_index + 1}")
            
            # Update the UI metrics
            self.input_panel.update_financial_metrics(cash, portfolio_value)
            
            # Show execution summary
            self.input_panel.show_message_box(
                QMessageBox.Icon.Information,
                "Window Trades Executed",
                f"Window {self.current_window_index + 1}/{len(self.trading_windows)} completed!\n\n"
                f"Executed: {len(orders)} trades\n"
                f"Current Cash: ${cash:,.2f}\n"
                f"Portfolio Value: ${portfolio_value:,.2f}",
                QMessageBox.StandardButton.Ok
            )
            self.input_panel.update_ui_after_trading()
        else:
            logger.error(f"Failed to execute trades for window {self.current_window_index + 1}: {result.get('warning_message', 'Unknown error')}")
            self.input_panel.show_message_box(
                QMessageBox.Icon.Critical,
                "Execution Failed",
                f"Failed to execute trades for window {self.current_window_index + 1}:\n\n{result.get('warning_message', 'Unknown error')}",
                QMessageBox.StandardButton.Ok
            )
        
        # Move to next window regardless of success/failure
        self.current_window_index += 1
        self._process_next_window()
    
    def _skip_to_next_window(self):
        """Skip the current window and move to the next."""
        self.current_window_index += 1
        self._process_next_window()
    
    def _handle_window_error(self, error_message):
        """Handle errors in window processing."""
        self.input_panel.hide_progress()
        self.input_panel.show_message_box(
            QMessageBox.Icon.Critical,
            "Window Processing Error",
            f"Error processing trading window:\n\n{error_message}",
            QMessageBox.StandardButton.Ok
        )
        self._skip_to_next_window()
    
    def _finish_windowed_trading(self):
        """Finish the windowed trading process."""
        logger.info("Windowed semi-automated trading completed")
        
        # Update UI state after completion
        self.input_panel.update_ui_after_trading()
        
        self.input_panel.show_message_box(
            QMessageBox.Icon.Information,
            "Trading Complete",
            "Semi-automated trading session completed!\n\nAll trading windows have been processed.",
            QMessageBox.StandardButton.Ok
        )
        
        # Refresh the UI using existing methods
        if hasattr(self.main_window, 'update_dashboard'):
            self.main_window.update_dashboard()
        
        # Refresh date constraints using existing method
        if hasattr(self.input_panel, 'refresh_date_constraints_after_trade'):
            self.input_panel.refresh_date_constraints_after_trade()