# import sys
# import os
# import pandas as pd
# import unittest
# import pytest
# from datetime import datetime
# import logging

# # Add project root to sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# from frontend.data.data_manager import DataManager
# from frontend.logging_config import get_isolated_logger
# from frontend.data.trading_connector import execute_trading_strategy

# # Initialize logger
# logger = get_isolated_logger("test_frontend", "trading_only", logging.INFO)

# # Path to the real CSV file
# CSV_PATH = r"E:\Afeka\FinalProject\Project\Investment-portfolio-management-system\resources\all_tickers_results.csv"

# @pytest.fixture
# def data_manager():
#     """Fixture to create a DataManager with the real CSV file."""
#     if not os.path.exists(CSV_PATH):
#         pytest.fail(f"CSV file not found at {CSV_PATH}")
#     start_time = datetime.now()
#     dm = DataManager(CSV_PATH)
#     if dm.data is None or dm.data.empty:
#         pytest.fail(f"Failed to load data from {CSV_PATH}")
#     # Filter using DataManager's set_date_range and get_filtered_data
#     success, msg = dm.set_date_range(pd.to_datetime('2022-01-01', utc=True), pd.to_datetime('2023-01-01', utc=True))
#     if not success:
#         pytest.fail(f"Failed to set date range: {msg}")
#     dm.data = dm.get_filtered_data()
#     duration = (datetime.now() - start_time).total_seconds()
#     logger.info(f"DataManager loading took {duration:.3f} seconds")
#     if dm.data is None or dm.data.empty:
#         logger.warning("No data in date range 2022-01-01 to 2023-01-01")
#         dm.data = pd.DataFrame(columns=['date', 'Ticker', 'Close', 'Best_Prediction', 'Actual_Sharpe', 'Buy', 'Sell', 'Prediction_Uncertainty'])
#     logger.info(f"DataManager loaded: {dm.data.shape} rows, columns: {list(dm.data.columns)}")
#     logger.info(f"Test date range: {dm.dataset_start_date} to {dm.dataset_end_date}")
#     # Log data for key tickers
#     tickers_to_check = ['ABBV', 'CTSH', 'FICO', 'ATO', 'LLY', 'TPL', 'MCK', 'MPC', 'CBOE', 'KEYS', 'LW', 'UAL', 'AXON', 'LDOS', 'PSX', 'ERIE', 'APD', 'PG']
#     for ticker in tickers_to_check:
#         ticker_data = dm.data[dm.data['Ticker'] == ticker]
#         if not ticker_data.empty:
#             logger.info(f"Data for {ticker} (first 5 rows): {ticker_data[['date', 'Close', 'Best_Prediction', 'Actual_Sharpe']].head().to_dict(orient='records')}")
#         else:
#             logger.info(f"No data for {ticker} in filtered dataset")
#     return dm

# class TestFrontend(unittest.TestCase):
#     @pytest.fixture(autouse=True)
#     def setup(self, data_manager):
#         """Setup for each test."""
#         self.data_manager = data_manager
#         # Reset portfolio state
#         portfolio_file = os.path.join(project_root, 'data', 'portfolio_state.json')
#         if os.path.exists(portfolio_file):
#             os.remove(portfolio_file)
#             logger.info(f"Cleared portfolio state file: {portfolio_file}")

#     def test_execute_trading_strategy_timing(self):
#         """Test that execute_trading_strategy completes within 45 seconds and generates trades at risk level 0.0."""
#         start_time = datetime.now()
#         logger.info("Testing execute_trading_strategy with risk level 0.0")
        
#         # Validate merged_data
#         merged_data = self.data_manager.data
#         if merged_data is None or merged_data.empty:
#             pytest.fail("Merged data is None or empty")
#         logger.info(f"Merged data shape: {merged_data.shape}")
#         logger.info(f"Merged data date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
#         logger.info(f"Sample data (first 5 rows): {merged_data.head().to_dict(orient='records')}")

#         # Log expected vs actual columns
#         expected_columns = ['date', 'Ticker', 'Close', 'Best_Prediction', 'Actual_Sharpe', 'Buy', 'Sell', 'Prediction_Uncertainty']
#         actual_columns = list(merged_data.columns)
#         logger.info(f"Expected columns: {expected_columns}")
#         logger.info(f"Actual columns: {actual_columns}")
#         if set(expected_columns) != set(actual_columns):
#             logger.warning(f"Column mismatch: Missing {set(expected_columns) - set(actual_columns)}")
#             # Add missing columns with defaults
#             for col in expected_columns:
#                 if col not in actual_columns:
#                     merged_data[col] = 0 if col in ['Buy', 'Sell'] else 1.0
#                     logger.info(f"Added default column {col}")

#         # Check for weights file
#         weights_file = os.path.join(project_root, 'resources', 'final_tickers_score.csv')
#         if not os.path.exists(weights_file):
#             logger.warning(f"Weights file {weights_file} not found. Creating default weights.")
#             tickers = merged_data['Ticker'].unique()
#             weights = pd.DataFrame({'Ticker': tickers, 'Weight': [0.02] * len(tickers)})
#             weights.to_csv(weights_file, index=False)
#             logger.info(f"Created default weights file at {weights_file}")

#         # Call execute_trading_strategy
#         start_date = pd.to_datetime('2022-01-01', utc=True)
#         end_date = pd.to_datetime('2023-01-01', utc=True)
#         success, result = execute_trading_strategy(
#             investment_amount=10000,
#             risk_level=0.0,
#             start_date=start_date,
#             end_date=end_date,
#             data_manager=self.data_manager,
#             mode="automatic",
#             reset_state=True,
#             selected_orders=None
#         )

#         duration = (datetime.now() - start_time).total_seconds()
#         logger.info(f"execute_trading_strategy execution took {duration:.3f} seconds")
#         logger.info(f"Result: success={success}, orders={len(result['orders'])}, "
#                     f"portfolio_value={result['portfolio_value']}, warning={result['warning_message']}")
#         if not result['orders']:
#             logger.error("No orders generated when trades were expected")
#         else:
#             logger.info(f"Orders executed: {len(result['orders'])}")
#             for order in result['orders'][:5]:  # Log first 5 orders
#                 logger.info(f"Order: {order}")

#         # Assertions
#         self.assertLessEqual(duration, 45, f"Trading strategy execution took {duration:.3f} seconds, exceeded 45 seconds")
#         self.assertTrue(success, "execute_trading_strategy failed")
#         self.assertGreater(len(result['orders']), 20, f"Expected at least 20 orders, got {len(result['orders'])}")
#         self.assertGreater(result['portfolio_value'], 10000, f"Portfolio value {result['portfolio_value']} should be greater than initial $10,000")
#         self.assertLess(result['portfolio_value'], 15000, f"Portfolio value {result['portfolio_value']} unexpectedly high")
#         self.assertEqual(result['warning_message'], "", "No warning message expected when trades are executed")
#         self.assertGreater(len(result['portfolio_history']), 300, f"Expected at least 300 portfolio history entries, got {len(result['portfolio_history'])}")
        
#     def test_execute_trading_strategy_no_trades(self):
#         """Test that execute_trading_strategy returns a warning and no trades when thresholds are misaligned."""
#         start_time = datetime.now()
#         logger.info("Testing execute_trading_strategy with high risk level to prevent trades")
        
#         # Validate merged_data
#         merged_data = self.data_manager.data
#         if merged_data is None or merged_data.empty:
#             pytest.fail("Merged data is None or empty")
#         logger.info(f"Merged data shape: {merged_data.shape}")
#         logger.info(f"Merged data date range: {merged_data['date'].min()} to {merged_data['date'].max()}")

#         # Check for weights file
#         weights_file = os.path.join(project_root, 'resources', 'final_tickers_score.csv')
#         if not os.path.exists(weights_file):
#             logger.warning(f"Weights file {weights_file} not found. Creating default weights.")
#             tickers = merged_data['Ticker'].unique()
#             weights = pd.DataFrame({'Ticker': tickers, 'Weight': [0.02] * len(tickers)})
#             weights.to_csv(weights_file, index=False)
#             logger.info(f"Created default weights file at {weights_file}")

#         # Test with high risk level (e.g., 1.0) to set unreachable thresholds
#         success, result = execute_trading_strategy(
#             investment_amount=10000,
#             risk_level=1.0,  # High risk to make thresholds unreachable
#             start_date=pd.to_datetime('2022-01-01', utc=True),
#             end_date=pd.to_datetime('2023-01-01', utc=True),
#             data_manager=self.data_manager,
#             mode="automatic",
#             reset_state=True,
#             selected_orders=None
#         )

#         duration = (datetime.now() - start_time).total_seconds()
#         logger.info(f"execute_trading_strategy execution took {duration:.3f} seconds")
#         logger.info(f"Result: success={success}, orders={len(result['orders'])}, "
#                     f"portfolio_value={result['portfolio_value']}, warning={result['warning_message']}")

#         # Assertions
#         self.assertLessEqual(duration, 30, f"Execution took {duration:.3f} seconds, exceeded 30 seconds")
#         self.assertTrue(success, "execute_trading_strategy failed")
#         self.assertEqual(len(result['orders']), 0, f"Expected no orders, got {len(result['orders'])}")
#         self.assertEqual(result['portfolio_value'], 10000, f"Expected portfolio value $10,000, got {result['portfolio_value']}")
#         self.assertNotEqual(result['warning_message'], "", "Expected a non-empty warning message")
#         self.assertTrue("no trades" in result['warning_message'].lower() or "threshold" in result['warning_message'].lower(),
#                         f"Warning message '{result['warning_message']}' does not indicate no trades or threshold issue")

#         # Optional: Test with invalid date range
#         logger.info("Testing execute_trading_strategy with invalid date range")
#         start_time = datetime.now()
#         success, result = execute_trading_strategy(
#             investment_amount=10000,
#             risk_level=0.0,  # Normal risk level, but invalid dates
#             start_date=pd.to_datetime('2020-01-01', utc=True),  # Outside dataset range
#             end_date=pd.to_datetime('2020-12-31', utc=True),
#             data_manager=self.data_manager,
#             mode="automatic",
#             reset_state=True,
#             selected_orders=None
#         )

#         duration = (datetime.now() - start_time).total_seconds()
#         logger.info(f"execute_trading_strategy execution took {duration:.3f} seconds")
#         logger.info(f"Result: success={success}, orders={len(result['orders'])}, "
#                     f"portfolio_value={result['portfolio_value']}, warning={result['warning_message']}")

#         # Assertions for invalid date range
#         self.assertLessEqual(duration, 30, f"Execution took {duration:.3f} seconds, exceeded 30 seconds")
#         self.assertTrue(success, "execute_trading_strategy failed")
#         self.assertEqual(len(result['orders']), 0, f"Expected no orders, got {len(result['orders'])}")
#         self.assertEqual(result['portfolio_value'], 10000, f"Expected portfolio value $10,000, got {result['portfolio_value']}")
#         self.assertNotEqual(result['warning_message'], "", "Expected a non-empty warning message")
#         self.assertTrue("no data" in result['warning_message'].lower() or "date range" in result['warning_message'].lower(),
#                         f"Warning message '{result['warning_message']}' does not indicate no data or date range issue")

# if __name__ == '__main__':
#     pytest.main(['-v', '--tb=short'])

import sys
import os
import pandas as pd
import unittest
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QDate
import pytest
from datetime import datetime
import builtins

# Store original os.path.exists and builtins.open
original_exists = os.path.exists
original_open = builtins.open

print(f"Initial sys.path: {sys.path}")  # Debug: Print sys.path before modifications

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"Updated sys.path: {sys.path}")  # Debug: Print sys.path after modifications

# Verify frontend module exists
frontend_path = os.path.join(project_root, 'frontend')
if not original_exists(frontend_path):
    raise RuntimeError(f"Frontend directory not found at {frontend_path}")
if not original_exists(os.path.join(frontend_path, '__init__.py')):
    raise RuntimeError("Frontend directory is missing __init__.py")
if not original_exists(os.path.join(frontend_path, 'data', '__init__.py')):
    raise RuntimeError("Frontend/data directory is missing __init__.py")
if not original_exists(os.path.join(frontend_path, 'gui', '__init__.py')):
    raise RuntimeError("Frontend/gui directory is missing __init__.py")
if not original_exists(os.path.join(frontend_path, 'logging_config.py')):
    raise RuntimeError("logging_config.py not found at frontend/logging_config.py")
print(f"Frontend directory verified: {frontend_path}")

# Imports
try:
    from frontend.data.data_manager import DataManager
    from frontend.gui.main_window import MainWindow
    from frontend.data.trading_connector import get_order_history_df
    from frontend.logging_config import get_logger
    print("Frontend imports successful")
except ImportError as e:
    raise RuntimeError(f"Failed to import frontend modules: {e}")

# Initialize logger
logger = get_logger(__name__)

# Path to the real CSV file
CSV_PATH = r"backend\resources\all_tickers_results.csv"

@pytest.fixture(scope="session")
def app():
    """Fixture to initialize QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

@pytest.fixture
def data_manager():
    """Fixture to create a DataManager with the real CSV file."""
    abs_csv_path = os.path.join(project_root, CSV_PATH)
    if not original_exists(abs_csv_path):
        pytest.fail(f"CSV file not found at {abs_csv_path}")
    dm = DataManager(abs_csv_path)
    if dm.data is None or dm.data.empty:
        pytest.fail(f"Failed to load data from {abs_csv_path}")
    # Log dataset date range for debugging
    logger.info(f"Dataset date range: {dm.dataset_start_date} to {dm.dataset_end_date}")
    return dm

@pytest.fixture
def main_window(data_manager, tmp_path):
    """Fixture to create MainWindow with real functions."""
    window = MainWindow(data_manager)
    # Set date range and trading mode
    window.input_panel.start_date_input.setDate(QDate(2021, 10, 15))
    window.input_panel.end_date_input.setDate(QDate(2021, 10, 20))
    window.input_panel.mode_combo.setCurrentText("Automatic")  # Ensure automatic mode
    start_date = pd.to_datetime(window.input_panel.start_date_input.date().toPyDate(), utc=True)
    end_date = pd.to_datetime(window.input_panel.end_date_input.date().toPyDate(), utc=True)
    success, msg = data_manager.set_date_range(start_date, end_date)
    if not success:
        pytest.fail(f"Failed to set date range: {msg}")
    yield window
    window.close()

class TestFrontend(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, app, main_window, data_manager):
        """Setup for each test."""
        self.app = app
        self.main_window = main_window
        self.data_manager = data_manager
        self.input_panel = self.main_window.input_panel

    def test_execute_trading_strategy_timing(self):
        """Test 1: Ensure execute trading strategy completes within 45 seconds."""
        start_time = datetime.now()
        self.input_panel.execute_button.click()
        self.app.processEvents()  # Process GUI events
        QTimer.singleShot(100, lambda: self.app.processEvents())  # Additional event processing
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Trading strategy execution took {duration} seconds")
        self.assertLessEqual(duration, 45, "Trading strategy execution exceeded 45 seconds")

    def test_no_trades_message_timing(self):
        """Test 2: Ensure no-trades message appears within 30 seconds when thresholds not met."""
        # Simulate no trades by setting an extreme risk value
        self.input_panel.risk_input.setValue(1000)  # QDoubleSpinBox
        start_time = datetime.now()
        self.input_panel.execute_button.click()
        self.app.processEvents()  # Process GUI events
        QTimer.singleShot(100, lambda: self.app.processEvents())  # Additional event processing
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"No-trades message appeared in {duration} seconds")
        self.assertLessEqual(duration, 30, "No-trades message took longer than 30 seconds")

    def test_profit_percentage(self):
        """Test 3: Ensure at least 70% of transactions made a profit."""
        self.input_panel.execute_button.click()
        self.app.processEvents()  # Process GUI events
        orders_df = get_order_history_df()  # Use real function to get orders
        logger.info(f"Orders retrieved: {len(orders_df)} rows")
        if orders_df.empty:
            pytest.skip("No orders available to test profitability")
        orders = orders_df.to_dict('records')

        profitable_trades = 0
        total_trades = 0
        buy_price = None
        buy_cost = None

        for order in orders:
            if order['action'] == 'buy':
                buy_price = order['price']
                buy_cost = order['investment_amount'] + order.get('transaction_cost', 0)
            elif order['action'] == 'sell' and buy_price is not None:
                sell_proceeds = order['investment_amount'] - order.get('transaction_cost', 0)
                if sell_proceeds > buy_cost:
                    profitable_trades += 1
                total_trades += 1
                buy_price = None
                buy_cost = None

        profit_percentage = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        logger.info(f"Profitable trades: {profitable_trades}/{total_trades} ({profit_percentage:.2f}%)")
        self.assertGreaterEqual(profit_percentage, 70, f"Only {profit_percentage:.2f}% of trades were profitable, expected at least 70%")

    def test_sharpe_accuracy(self):
        """Test 4: Ensure Predicted Sharpe is within 15% of Actual Sharpe in Trading History."""
        self.input_panel.execute_button.click()
        self.app.processEvents()
        self.main_window.tabs.setCurrentIndex(2)  # Switch to trading history tab
        self.main_window.recommendation_panel.update_recommendations()
        self.app.processEvents()
        recommendation_panel = self.main_window.recommendation_panel
        table = recommendation_panel.table
        total_rows = table.rowCount()
        logger.info(f"Trading history table rows: {total_rows}")

        if total_rows == 0:
            pytest.skip("No rows in trading history table to test Sharpe accuracy")

        accurate_rows = 0
        for row in range(total_rows):
            pred_sharpe_item = table.item(row, 10)  # Predicted Sharpe column
            actual_sharpe_item = table.item(row, 11)  # Actual Sharpe column

            if pred_sharpe_item and actual_sharpe_item and pred_sharpe_item.text() != "N/A" and actual_sharpe_item.text() != "N/A":
                try:
                    pred_sharpe = float(pred_sharpe_item.text())
                    actual_sharpe = float(actual_sharpe_item.text())
                    if actual_sharpe != 0:
                        error_percentage = abs(pred_sharpe - actual_sharpe) / abs(actual_sharpe) * 100
                        if error_percentage <= 15:
                            accurate_rows += 1
                        logger.info(f"Row {row}: Pred Sharpe={pred_sharpe:.2f}, Actual Sharpe={actual_sharpe:.2f}, Error={error_percentage:.2f}%")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Row {row}: Could not parse Sharpe values - {e}")
                    continue

        accuracy_percentage = (accurate_rows / total_rows) * 100 if total_rows > 0 else 0
        logger.info(f"Sharpe accuracy: {accurate_rows}/{total_rows} rows within 15% ({accuracy_percentage:.2f}%)")
        self.assertGreaterEqual(accuracy_percentage, 100, f"Only {accuracy_percentage:.2f}% of Predicted Sharpes were within 15% of Actual Sharpes")

if __name__ == '__main__':
    pytest.main(['-v', '--tb=long'])