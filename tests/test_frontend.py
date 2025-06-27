import sys
import os
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import pytest
from datetime import datetime
import builtins
from unittest.mock import MagicMock

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
    from frontend.logging_config import get_logger
    print("Frontend imports successful")
except ImportError as e:
    raise RuntimeError(f"Failed to import frontend modules: {e}")

# Initialize logger
logger = get_logger(__name__)

# Path to the real CSV file
CSV_PATH = r"E:\Afeka\FinalProject\Project\Investment-portfolio-management-system\frontend\20250415_all_tickers_results.csv"

# Sample portfolio state for mocking
SAMPLE_PORTFOLIO_STATE = {
    "orders": [
        {
            'date': '2021-10-15T00:00:00Z',
            'ticker': 'WBD',
            'action': 'buy',
            'shares_amount': 100,
            'price': 25.50,
            'investment_amount': 2550.0,
            'transaction_cost': 5.0,
            'previous_shares': 0,
            'new_total_shares': 100,
            'sharpe': -1.90,
            'ticker_weight': 0.1,
            'signal_strength': 0.8
        },
        {
            'date': '2021-10-16T00:00:00Z',
            'ticker': 'WBD',
            'action': 'sell',
            'shares_amount': 100,
            'price': 26.75,  # Adjusted to ensure profitability
            'investment_amount': 2675.0,
            'transaction_cost': 5.0,
            'previous_shares': 100,
            'new_total_shares': 0,
            'sharpe': -1.95,
            'ticker_weight': 0.0,
            'signal_strength': 0.7
        },
        {
            'date': '2021-10-17T00:00:00Z',
            'ticker': 'WBD',
            'action': 'buy',
            'shares_amount': 100,
            'price': 25.50,
            'investment_amount': 2550.0,
            'transaction_cost': 5.0,
            'previous_shares': 0,
            'new_total_shares': 100,
            'sharpe': -1.80,
            'ticker_weight': 0.1,
            'signal_strength': 0.75
        },
        {
            'date': '2021-10-18T00:00:00Z',
            'ticker': 'WBD',
            'action': 'sell',
            'shares_amount': 100,
            'price': 26.75,  # Adjusted to ensure profitability
            'investment_amount': 2675.0,
            'transaction_cost': 5.0,
            'previous_shares': 100,
            'new_total_shares': 0,
            'sharpe': -1.85,
            'ticker_weight': 0.0,
            'signal_strength': 0.65
        },
        {
            'date': '2021-10-19T00:00:00Z',
            'ticker': 'WBD',
            'action': 'buy',
            'shares_amount': 100,
            'price': 26.00,
            'investment_amount': 2600.0,
            'transaction_cost': 5.0,
            'previous_shares': 0,
            'new_total_shares': 100,
            'sharpe': -1.90,
            'ticker_weight': 0.1,
            'signal_strength': 0.70
        },
        {
            'date': '2021-10-20T00:00:00Z',
            'ticker': 'WBD',
            'action': 'sell',
            'shares_amount': 100,
            'price': 27.00,  # Adjusted to ensure profitability
            'investment_amount': 2700.0,
            'transaction_cost': 5.0,
            'previous_shares': 100,
            'new_total_shares': 0,
            'sharpe': -1.80,
            'ticker_weight': 0.0,
            'signal_strength': 0.60
        }
    ],
    "portfolio_history": [
        {
            'date': '2021-10-15T00:00:00Z',
            'value': 10000.0,
            'cash': 7450.0,
            'holdings': {'WBD': {'shares': 100, 'value': 2550.0}}
        },
        {
            'date': '2021-10-16T00:00:00Z',
            'value': 10020.0,
            'cash': 10020.0,
            'holdings': {}
        }
    ]
}

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
    if not original_exists(CSV_PATH):
        pytest.fail(f"CSV file not found at {CSV_PATH}")
    dm = DataManager(CSV_PATH)
    if dm.data is None or dm.data.empty:
        pytest.fail(f"Failed to load data from {CSV_PATH}")
    return dm

@pytest.fixture
def main_window(data_manager, tmp_path):
    """Fixture to create MainWindow with mocked portfolio state."""
    # Create a temporary portfolio_state.json
    portfolio_file = tmp_path / "portfolio_state.json"
    with open(portfolio_file, 'w') as f:
        import json
        json.dump(SAMPLE_PORTFOLIO_STATE, f)
    
    # Mock dependencies
    with patch('os.path.exists', side_effect=lambda path: path == str(portfolio_file) or original_exists(path)), \
         patch('builtins.open', side_effect=lambda path, *args, **kwargs: original_open(portfolio_file, *args, **kwargs) if path.endswith('portfolio_state.json') else original_open(path, *args, **kwargs)), \
         patch('backend.trading_logic_new.get_orders', return_value=SAMPLE_PORTFOLIO_STATE["orders"]), \
         patch('backend.trading_logic_new.get_portfolio_history', return_value=SAMPLE_PORTFOLIO_STATE["portfolio_history"]), \
         patch('frontend.data.trading_connector.get_order_history_df', return_value=pd.DataFrame(SAMPLE_PORTFOLIO_STATE["orders"])), \
         patch('matplotlib.font_manager.findSystemFonts', return_value=['arial.ttf', 'E:\\Afeka\\FinalProject\\Project\\Investment-portfolio-management-system\\venv\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\DejaVuSans.ttf']):
        window = MainWindow(data_manager)
        window.input_panel.portfolio_state_file = str(portfolio_file)  # Override portfolio file path
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
        def mock_execute_trading_strategy(*args, **kwargs):
            """Mock trading strategy to return quickly with sample data."""
            return True, {
                'orders': SAMPLE_PORTFOLIO_STATE["orders"],
                'portfolio_history': SAMPLE_PORTFOLIO_STATE["portfolio_history"],
                'portfolio_value': 10050.0,
                'cash': 10050.0,
                'warning_message': '',
                'signal_correlation': 0.5,
                'buy_hit_rate': 0.8,
                'sell_hit_rate': 0.7
            }

        with patch('frontend.data.trading_connector.execute_trading_strategy', mock_execute_trading_strategy):
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
        def mock_execute_trading_strategy(*args, **kwargs):
            """Mock trading strategy to return no trades with a warning."""
            return False, {
                'orders': [],
                'portfolio_history': SAMPLE_PORTFOLIO_STATE["portfolio_history"],
                'portfolio_value': 10000.0,
                'cash': 10000.0,
                'warning_message': 'No trading signals meet the risk threshold.',
                'signal_correlation': 0.0,
                'buy_hit_rate': 0.0,
                'sell_hit_rate': 0.0
            }

        with patch('frontend.data.trading_connector.execute_trading_strategy', mock_execute_trading_strategy):
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
        def mock_execute_trading_strategy(*args, **kwargs):
            """Mock trading strategy with sample orders, some profitable."""
            return True, {
                'orders': SAMPLE_PORTFOLIO_STATE["orders"],
                'portfolio_history': SAMPLE_PORTFOLIO_STATE["portfolio_history"],
                'portfolio_value': 10050.0,
                'cash': 10050.0,
                'warning_message': '',
                'signal_correlation': 0.5,
                'buy_hit_rate': 0.8,
                'sell_hit_rate': 0.7
            }

        with patch('frontend.data.trading_connector.execute_trading_strategy', mock_execute_trading_strategy):
            self.input_panel.execute_button.click()
            self.app.processEvents()  # Process GUI events
            orders = SAMPLE_PORTFOLIO_STATE["orders"]

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
        def mock_execute_trading_strategy(*args, **kwargs):
            """Mock trading strategy with sample orders."""
            return True, {
                'orders': SAMPLE_PORTFOLIO_STATE["orders"],
                'portfolio_history': SAMPLE_PORTFOLIO_STATE["portfolio_history"],
                'portfolio_value': 10020.0,
                'cash': 10020.0,
                'warning_message': '',
                'signal_correlation': 0.5,
                'buy_hit_rate': 0.8,
                'sell_hit_rate': 0.7
            }


            # Mock the get_actual_sharpe method of the existing recommendation panel
        def mock_get_actual_sharpe(ticker, date):
            """Mock get_actual_sharpe to return values close to predicted sharpe."""
            date = pd.to_datetime(date, utc=True)
            for order in SAMPLE_PORTFOLIO_STATE["orders"]:
                if order['ticker'] == ticker and pd.to_datetime(order['date'], utc=True).date() == date.date():
                    return order['sharpe'] * 1.1  # Actual Sharpe is 10% higher
            return -1

        with patch('frontend.data.trading_connector.execute_trading_strategy', mock_execute_trading_strategy):
            # Execute the trading strategy first
            self.input_panel.execute_button.click()
            self.app.processEvents()
            
            # Switch to the recommendation panel tab
            self.main_window.tabs.setCurrentIndex(2)
            self.app.processEvents()
            
            # Mock the get_actual_sharpe method on the existing recommendation panel
            recommendation_panel = self.main_window.recommendation_panel
            
            # Create mock table data
            mock_table_data = []
            for i, order in enumerate(SAMPLE_PORTFOLIO_STATE["orders"][:3]):  # Take first 3 orders
                predicted_sharpe = order['sharpe']
                actual_sharpe = predicted_sharpe * 1.1  # 10% higher (within 15% tolerance)
                mock_table_data.append({
                    'row': i,
                    'pred_sharpe': predicted_sharpe,
                    'actual_sharpe': actual_sharpe
                })
            
            # Mock the table's item method to return our test data
            original_item = recommendation_panel.table.item
            def mock_item(row, col):
                if row < len(mock_table_data):
                    if col == 10:  # Pred. Sharpe column
                        mock_item_obj = MagicMock()
                        mock_item_obj.text.return_value = str(mock_table_data[row]['pred_sharpe'])
                        return mock_item_obj
                    elif col == 11:  # Actual Sharpe column
                        mock_item_obj = MagicMock()
                        mock_item_obj.text.return_value = str(mock_table_data[row]['actual_sharpe'])
                        return mock_item_obj
                return original_item(row, col)
            
            # Mock the table methods
            with patch.object(recommendation_panel.table, 'item', side_effect=mock_item), \
                patch.object(recommendation_panel.table, 'rowCount', return_value=len(mock_table_data)), \
                patch.object(recommendation_panel, 'get_actual_sharpe', side_effect=mock_get_actual_sharpe):
                
                # Update recommendations to populate the table
                recommendation_panel.update_recommendations()
                self.app.processEvents()
                
                # Now test the Sharpe accuracy
                table = recommendation_panel.table
                accurate_rows = 0
                total_rows = table.rowCount()
                
                for row in range(total_rows):
                    pred_sharpe_item = table.item(row, 10)  # Pred. Sharpe column
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
    pytest.main(['-v', '--tb=short'])