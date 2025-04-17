from trading_logic import run_trading_strategy, orders
import pandas as pd
import logging 
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_trading_strategy(investment_amount, risk_level, start_date, end_date, data_manager):
    """
    Execute the trading strategy with user inputs from the frontend.
    
    Args:
        investment_amount (float): Initial amount to invest.
        risk_level (int): Risk level from the slider (scaled to 0-100).
        start_date (date): Start date of the investment window.
        end_date (date): End date of the investment window.
        data_manager (DataManager): Data manager instance with loaded data.
    
    Returns:
        bool: True if successful, False if an error occurs.
    """
    logging.debug("Starting execute_trading_strategy")
    try:
        # Validate date range
        if end_date < start_date:
            logging.warning("End date is earlier than start date")
            raise ValueError("End date cannot be earlier than start date.")
        
        # Get merged data from data_manager
        merged_data = data_manager.data
        logging.debug(f"Merged data shape: {merged_data.shape}")
        
        # Call the trading strategy function with the provided inputs
        logging.debug("Calling run_trading_strategy")
        run_trading_strategy(
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            merged_data=merged_data,
            data_manager=data_manager
        )
        logging.debug("run_trading_strategy completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error in execute_trading_strategy: {e}", exc_info=True)
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