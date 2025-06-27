import sys
import os
import pandas as pd
import unittest
import pytest
from datetime import datetime
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from frontend.data.data_manager import DataManager
from frontend.logging_config import get_isolated_logger
from frontend.data.trading_connector import execute_trading_strategy

# Initialize logger
logger = get_isolated_logger("test_frontend", "trading_only", logging.INFO)

# Path to the real CSV file
CSV_PATH = r"E:\Afeka\FinalProject\Project\Investment-portfolio-management-system\resources\all_tickers_results.csv"

@pytest.fixture
def data_manager():
    """Fixture to create a DataManager with the real CSV file."""
    if not os.path.exists(CSV_PATH):
        pytest.fail(f"CSV file not found at {CSV_PATH}")
    start_time = datetime.now()
    dm = DataManager(CSV_PATH)
    if dm.data is None or dm.data.empty:
        pytest.fail(f"Failed to load data from {CSV_PATH}")
    # Filter using DataManager's set_date_range and get_filtered_data
    success, msg = dm.set_date_range(pd.to_datetime('2022-01-01', utc=True), pd.to_datetime('2023-01-01', utc=True))
    if not success:
        pytest.fail(f"Failed to set date range: {msg}")
    dm.data = dm.get_filtered_data()
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"DataManager loading took {duration:.3f} seconds")
    if dm.data is None or dm.data.empty:
        logger.warning("No data in date range 2022-01-01 to 2023-01-01")
        dm.data = pd.DataFrame(columns=['date', 'Ticker', 'Close', 'Best_Prediction', 'Actual_Sharpe', 'Buy', 'Sell', 'Prediction_Uncertainty'])
    logger.info(f"DataManager loaded: {dm.data.shape} rows, columns: {list(dm.data.columns)}")
    logger.info(f"Test date range: {dm.dataset_start_date} to {dm.dataset_end_date}")
    # Log data for key tickers
    tickers_to_check = ['ABBV', 'CTSH', 'FICO', 'ATO', 'LLY', 'TPL', 'MCK', 'MPC', 'CBOE', 'KEYS', 'LW', 'UAL', 'AXON', 'LDOS', 'PSX', 'ERIE', 'APD', 'PG']
    for ticker in tickers_to_check:
        ticker_data = dm.data[dm.data['Ticker'] == ticker]
        if not ticker_data.empty:
            logger.info(f"Data for {ticker} (first 5 rows): {ticker_data[['date', 'Close', 'Best_Prediction', 'Actual_Sharpe']].head().to_dict(orient='records')}")
        else:
            logger.info(f"No data for {ticker} in filtered dataset")
    return dm

class TestFrontend(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, data_manager):
        """Setup for each test."""
        self.data_manager = data_manager
        # Reset portfolio state
        portfolio_file = os.path.join(project_root, 'data', 'portfolio_state.json')
        if os.path.exists(portfolio_file):
            os.remove(portfolio_file)
            logger.info(f"Cleared portfolio state file: {portfolio_file}")

    def test_execute_trading_strategy_timing(self):
        """Test that execute_trading_strategy completes within 45 seconds and generates trades at risk level 0.0."""
        start_time = datetime.now()
        logger.info("Testing execute_trading_strategy with risk level 0.0")
        
        # Validate merged_data
        merged_data = self.data_manager.data
        if merged_data is None or merged_data.empty:
            pytest.fail("Merged data is None or empty")
        logger.info(f"Merged data shape: {merged_data.shape}")
        logger.info(f"Merged data date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
        logger.info(f"Sample data (first 5 rows): {merged_data.head().to_dict(orient='records')}")

        # Log expected vs actual columns
        expected_columns = ['date', 'Ticker', 'Close', 'Best_Prediction', 'Actual_Sharpe', 'Buy', 'Sell', 'Prediction_Uncertainty']
        actual_columns = list(merged_data.columns)
        logger.info(f"Expected columns: {expected_columns}")
        logger.info(f"Actual columns: {actual_columns}")
        if set(expected_columns) != set(actual_columns):
            logger.warning(f"Column mismatch: Missing {set(expected_columns) - set(actual_columns)}")
            # Add missing columns with defaults
            for col in expected_columns:
                if col not in actual_columns:
                    merged_data[col] = 0 if col in ['Buy', 'Sell'] else 1.0
                    logger.info(f"Added default column {col}")

        # Check for weights file
        weights_file = os.path.join(project_root, 'resources', 'final_tickers_score.csv')
        if not os.path.exists(weights_file):
            logger.warning(f"Weights file {weights_file} not found. Creating default weights.")
            tickers = merged_data['Ticker'].unique()
            weights = pd.DataFrame({'Ticker': tickers, 'Weight': [0.02] * len(tickers)})
            weights.to_csv(weights_file, index=False)
            logger.info(f"Created default weights file at {weights_file}")

        # Call execute_trading_strategy
        start_date = pd.to_datetime('2022-01-01', utc=True)
        end_date = pd.to_datetime('2023-01-01', utc=True)
        success, result = execute_trading_strategy(
            investment_amount=10000,
            risk_level=0.0,
            start_date=start_date,
            end_date=end_date,
            data_manager=self.data_manager,
            mode="automatic",
            reset_state=True,
            selected_orders=None
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"execute_trading_strategy execution took {duration:.3f} seconds")
        logger.info(f"Result: success={success}, orders={len(result['orders'])}, "
                    f"portfolio_value={result['portfolio_value']}, warning={result['warning_message']}")
        if not result['orders']:
            logger.error("No orders generated when trades were expected")
        else:
            logger.info(f"Orders executed: {len(result['orders'])}")
            for order in result['orders'][:5]:  # Log first 5 orders
                logger.info(f"Order: {order}")

        # Assertions
        self.assertLessEqual(duration, 45, f"Trading strategy execution took {duration:.3f} seconds, exceeded 45 seconds")
        self.assertTrue(success, "execute_trading_strategy failed")
        self.assertGreater(len(result['orders']), 20, f"Expected at least 20 orders, got {len(result['orders'])}")
        self.assertGreater(result['portfolio_value'], 10000, f"Portfolio value {result['portfolio_value']} should be greater than initial $10,000")
        self.assertLess(result['portfolio_value'], 15000, f"Portfolio value {result['portfolio_value']} unexpectedly high")
        self.assertEqual(result['warning_message'], "", "No warning message expected when trades are executed")
        self.assertGreater(len(result['portfolio_history']), 300, f"Expected at least 300 portfolio history entries, got {len(result['portfolio_history'])}")
        
    def test_execute_trading_strategy_no_trades(self):
        """Test that execute_trading_strategy returns a warning and no trades when thresholds are misaligned."""
        start_time = datetime.now()
        logger.info("Testing execute_trading_strategy with high risk level to prevent trades")
        
        # Validate merged_data
        merged_data = self.data_manager.data
        if merged_data is None or merged_data.empty:
            pytest.fail("Merged data is None or empty")
        logger.info(f"Merged data shape: {merged_data.shape}")
        logger.info(f"Merged data date range: {merged_data['date'].min()} to {merged_data['date'].max()}")

        # Check for weights file
        weights_file = os.path.join(project_root, 'resources', 'final_tickers_score.csv')
        if not os.path.exists(weights_file):
            logger.warning(f"Weights file {weights_file} not found. Creating default weights.")
            tickers = merged_data['Ticker'].unique()
            weights = pd.DataFrame({'Ticker': tickers, 'Weight': [0.02] * len(tickers)})
            weights.to_csv(weights_file, index=False)
            logger.info(f"Created default weights file at {weights_file}")

        # Test with high risk level (e.g., 1.0) to set unreachable thresholds
        success, result = execute_trading_strategy(
            investment_amount=10000,
            risk_level=1.0,  # High risk to make thresholds unreachable
            start_date=pd.to_datetime('2022-01-01', utc=True),
            end_date=pd.to_datetime('2023-01-01', utc=True),
            data_manager=self.data_manager,
            mode="automatic",
            reset_state=True,
            selected_orders=None
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"execute_trading_strategy execution took {duration:.3f} seconds")
        logger.info(f"Result: success={success}, orders={len(result['orders'])}, "
                    f"portfolio_value={result['portfolio_value']}, warning={result['warning_message']}")

        # Assertions
        self.assertLessEqual(duration, 30, f"Execution took {duration:.3f} seconds, exceeded 30 seconds")
        self.assertTrue(success, "execute_trading_strategy failed")
        self.assertEqual(len(result['orders']), 0, f"Expected no orders, got {len(result['orders'])}")
        self.assertEqual(result['portfolio_value'], 10000, f"Expected portfolio value $10,000, got {result['portfolio_value']}")
        self.assertNotEqual(result['warning_message'], "", "Expected a non-empty warning message")
        self.assertTrue("no trades" in result['warning_message'].lower() or "threshold" in result['warning_message'].lower(),
                        f"Warning message '{result['warning_message']}' does not indicate no trades or threshold issue")

        # Optional: Test with invalid date range
        logger.info("Testing execute_trading_strategy with invalid date range")
        start_time = datetime.now()
        success, result = execute_trading_strategy(
            investment_amount=10000,
            risk_level=0.0,  # Normal risk level, but invalid dates
            start_date=pd.to_datetime('2020-01-01', utc=True),  # Outside dataset range
            end_date=pd.to_datetime('2020-12-31', utc=True),
            data_manager=self.data_manager,
            mode="automatic",
            reset_state=True,
            selected_orders=None
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"execute_trading_strategy execution took {duration:.3f} seconds")
        logger.info(f"Result: success={success}, orders={len(result['orders'])}, "
                    f"portfolio_value={result['portfolio_value']}, warning={result['warning_message']}")

        # Assertions for invalid date range
        self.assertLessEqual(duration, 30, f"Execution took {duration:.3f} seconds, exceeded 30 seconds")
        self.assertTrue(success, "execute_trading_strategy failed")
        self.assertEqual(len(result['orders']), 0, f"Expected no orders, got {len(result['orders'])}")
        self.assertEqual(result['portfolio_value'], 10000, f"Expected portfolio value $10,000, got {result['portfolio_value']}")
        self.assertNotEqual(result['warning_message'], "", "Expected a non-empty warning message")
        self.assertTrue("no data" in result['warning_message'].lower() or "date range" in result['warning_message'].lower(),
                        f"Warning message '{result['warning_message']}' does not indicate no data or date range issue")

if __name__ == '__main__':
    pytest.main(['-v', '--tb=short'])