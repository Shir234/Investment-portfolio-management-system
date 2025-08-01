# worker.py
from PyQt6.QtCore import QObject, pyqtSignal
from frontend.logging_config import get_logger
from frontend.data.trading_connector import execute_trading_strategy

logger = get_logger(__name__)


class Worker(QObject):
    """Worker class to run execute_trading_strategy in a background thread."""
    finished = pyqtSignal(bool, dict)   # success, result
    error = pyqtSignal(str)             # error message
    progress = pyqtSignal(str)          # progress message

    def __init__(self, investment_amount, risk_level, start_date, end_date, data_manager, mode, reset_state, selected_orders=None, current_cash=None, current_holdings=None):
        """
        Initialize worker with complete trading strategy parameters for background execution.
        Supports both automatic and semi-automatic modes with portfolio state management.
        """
        super().__init__()
        self.investment_amount = investment_amount
        self.risk_level = risk_level
        self.start_date = start_date
        self.end_date = end_date
        self.data_manager = data_manager
        self.mode = mode
        self.reset_state = reset_state
        self.selected_orders = selected_orders
        self.current_cash = current_cash
        self.current_holdings = current_holdings

    def run(self):
        """
        Execute trading strategy in background thread.
        Emits signals for UI updates while maintaining thread safety and exception handling.
        """
        try:
            self.progress.emit("Initializing strategy...")
            success, result = execute_trading_strategy(
                investment_amount=self.investment_amount,
                risk_level=self.risk_level,
                start_date=self.start_date,
                end_date=self.end_date,
                data_manager=self.data_manager,
                mode=self.mode,
                reset_state=self.reset_state,
                selected_orders=self.selected_orders,
                current_cash=self.current_cash,
                current_holdings=self.current_holdings
            )
            self.finished.emit(success, result)
        except Exception as e:
            logger.error(f"Error in Worker.run: {e}", exc_info=True)
            self.error.emit(str(e))