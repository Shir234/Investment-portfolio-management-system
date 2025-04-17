from trading_logic import run_trading_strategy, orders
import pandas as pd

def execute_trading_strategy(investment_amount, risk_level, start_date, end_date, merged_data=None):
    """
    Execute the trading strategy with user inputs from the frontend.
    
    Args:
        investment_amount (float): Initial amount to invest.
        risk_level (int): Risk level from the slider (scaled to 0-100).
        start_date (date): Start date of the investment window.
        end_date (date): End date of the investment window.
        merged_data (pd.DataFrame, optional): Merged ticker data. If None, it will be loaded.
    
    Returns:
        bool: True if successful, False if an error occurs.
    """
    try:
        # Validate date range
        if end_date < start_date:
            raise ValueError("End date cannot be earlier than start date.")

        # If merged_data is not provided, load it using DataManager
        if merged_data is None:
            from data.data_manager import DataManager
            data_manager = DataManager('20250415_all_tickers_results.csv')  # Updated path
            merged_data = data_manager.data
        
        # Call the trading strategy function with the provided inputs
        run_trading_strategy(
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            merged_data=merged_data
        )
        return True
    except Exception as e:
        print(f"Error executing trading strategy: {e}")
        return False

def get_order_history_df():
    global orders
    if not orders:
        return pd.DataFrame()
    
    # Create DataFrame from orders
    order_df = pd.DataFrame(orders)
    # Ensure columns match what the UI expects
    expected_columns = ['date', 'ticker', 'action', 'shares_amount', 'price', 'investment_amount', 'previous_shares', 'new_total_shares', 'sharpe']
    if not all(col in order_df.columns for col in expected_columns):
        print(f"Missing columns in order DataFrame: {set(expected_columns) - set(order_df.columns)}")
        return pd.DataFrame()
    
    return order_df