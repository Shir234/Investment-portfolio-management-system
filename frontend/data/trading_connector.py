import pandas as pd
import logging 
from trading_logic import run_trading_strategy, run_portfolio_simulation, get_orders, get_portfolio_history

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger('trading_connector')
logger.setLevel(logging.DEBUG)

def execute_trading_strategy(investment_amount, risk_level, start_date, end_date, data_manager, mode="automatic", reset_state=False):
    logger.debug("Starting execute_trading_strategy")
    try:
        if end_date < start_date:
            logger.warning("End date is earlier than start date")
            raise ValueError("End date cannot be earlier than start date.")
        
        if data_manager.data is None or data_manager.data.empty:
            logger.error("No data available in data_manager")
            raise ValueError("No data loaded. Please ensure a valid CSV file is selected.")
        
        merged_data = data_manager.data
        logger.debug(f"Merged data shape: {merged_data.shape}")
        
        logger.debug("Calling run_trading_strategy")
        result = run_trading_strategy(
            merged_data=merged_data,
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            data_manager=data_manager,
            mode=mode,
            reset_state=reset_state
        )
        
        if mode == "semi-automatic":
            suggestions = result
            portfolio_history = get_portfolio_history()  # Fetch history for UI
        else:
            orders, portfolio_history, _ = result  # Include portfolio_history
            suggestions = orders
        
        # Extract final portfolio value
        portfolio_value = portfolio_history[-1]['value'] if portfolio_history else 0.0
        
        logger.debug("run_trading_strategy completed successfully")
        return True, {
            'orders': suggestions,
            'portfolio_history': portfolio_history,
            'portfolio_value': portfolio_value
        }
    except Exception as e:
        logger.error(f"Error in execute_trading_strategy: {e}", exc_info=True)
        return False, {
            'orders': [],
            'portfolio_history': [],
            'portfolio_value': 0.0
        }

def execute_portfolio_simulation(investment_amount, risk_level, start_date, end_date, data_manager):
    logger.debug("Starting execute_portfolio_simulation")
    try:
        if end_date < start_date:
            logger.warning("End date is earlier than start date")
            raise ValueError("End date cannot be earlier than start date.")
        
        if data_manager.data is None or data_manager.data.empty:
            logger.error("No data available in data_manager")
            raise ValueError("No data loaded. Please ensure a valid CSV file is selected.")
        
        merged_data = data_manager.data
        logger.debug(f"Merged data shape: {merged_data.shape}")
        
        from trading_logic import map_risk_threshold_to_sharpe
        buy_threshold, _, _ = map_risk_threshold_to_sharpe(risk_level, merged_data)
        
        logger.debug("Calling run_portfolio_simulation")
        orders, portfolio_history, _ = run_portfolio_simulation(
            merged_data=merged_data,
            min_acceptable_sharpe=buy_threshold,
            investment_amount=investment_amount,
            investment_period_days=(end_date - start_date).days,
            start_date=start_date,
            end_date=end_date,
            risk_level=risk_level
        )
        
        # Extract final portfolio value
        portfolio_value = portfolio_history['portfolio_value'].iloc[-1] if not portfolio_history.empty else 0.0
        
        logger.debug("run_portfolio_simulation completed successfully")
        return True, {
            'orders': orders,
            'portfolio_history': portfolio_history,
            'portfolio_value': portfolio_value
        }
    except Exception as e:
        logger.error(f"Error in execute_portfolio_simulation: {e}", exc_info=True)
        return False, {
            'orders': [],
            'portfolio_history': [],
            'portfolio_value': 0.0
        }

def get_order_history_df():
    orders = get_orders()
    if not orders:
        return pd.DataFrame()
    
    order_df = pd.DataFrame(orders)
    expected_columns = ['date', 'ticker', 'action', 'shares_amount', 'price', 'investment_amount', 'previous_shares', 'new_total_shares', 'sharpe']
    if not all(col in order_df.columns for col in expected_columns):
        missing_cols = set(expected_columns) - set(order_df.columns)
        logger.warning(f"Missing columns in order DataFrame: {missing_cols}")
        return pd.DataFrame(columns=expected_columns)
    
    return order_df