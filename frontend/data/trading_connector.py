import pandas as pd
import logging 
from backend.trading_logic import run_integrated_trading_strategy, get_orders, get_portfolio_history, validate_prediction_quality

# Import centralized logging (remove the old logging.basicConfig)
from logging_config import get_logger

# Use regular logger for GUI/connector logging (goes to main app log)
logger = get_logger('trading_connector')
logger.setLevel(logging.INFO)

def execute_trading_strategy(investment_amount, risk_level, start_date, end_date, data_manager, mode="automatic", reset_state=True):
    """Execute the trading strategy with user inputs from the frontend."""
    logger.info("="*50)
    logger.info("EXECUTE TRADING STRATEGY CALLED")
    logger.info("="*50)
    logger.info(f"Parameters: Investment=${investment_amount}, Risk={risk_level}, Mode={mode}, Reset={reset_state}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        # Validate inputs
        if end_date < start_date:
            error_msg = "End date is earlier than start date"
            logger.error(error_msg)
            raise ValueError("End date cannot be earlier than start date.")
        
        if data_manager.data is None or data_manager.data.empty:
            error_msg = "No data available in data_manager"
            logger.error(error_msg)
            raise ValueError("No data loaded. Please ensure a valid CSV file is selected.")
        
        merged_data = data_manager.data
        logger.info(f"Data validation passed - Shape: {merged_data.shape}")
        logger.info(f"Data columns: {list(merged_data.columns)}")
        
        # Validate signal quality
        logger.info("Validating prediction quality...")
        correlation, buy_hit_rate, sell_hit_rate, sharpe_min, sharpe_max = validate_prediction_quality(merged_data)
        
        logger.info(f"Signal Quality Results:")
        logger.info(f"  Correlation: {correlation:.3f}")
        logger.info(f"  Buy Hit Rate: {buy_hit_rate:.1%}")
        logger.info(f"  Sell Hit Rate: {sell_hit_rate:.1%}")
        logger.info(f"  Sharpe Range: [{sharpe_min:.2f}, {sharpe_max:.2f}]")
        
        warning_message = ""
        if correlation < 0.1:
            warning_message = f"Low signal-return correlation ({correlation:.3f}). Strategy may be unreliable."
            logger.warning(warning_message)

        logger.info(f"Calling integrated trading strategy...")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Reset State: {reset_state}")
        
        # NOTE: All the detailed trading logic will be logged to logs/trading_only_YYYYMMDD.log
        # This connector logging appears in logs/app_YYYYMMDD.log
        result = run_integrated_trading_strategy(
            merged_data=merged_data,
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            reset_state=reset_state
        )
        
        # Process results based on mode
        if mode == "semi-automatic":
            orders, warning_message = result
            portfolio_history = get_portfolio_history()
            portfolio_value = portfolio_history[-1]['value'] if portfolio_history else investment_amount
            logger.info(f"Semi-automatic mode completed - {len(orders)} orders suggested")
        else:
            orders, portfolio_history, portfolio_value, warning_message = result
            logger.info(f"Automatic mode completed:")
            logger.info(f"  Orders executed: {len(orders)}")
            logger.info(f"  Portfolio history entries: {len(portfolio_history)}")
            logger.info(f"  Final portfolio value: ${portfolio_value:,.2f}")
        
        # Log performance summary
        if portfolio_history:
            initial_value = investment_amount
            final_value = portfolio_value
            total_return = ((final_value / initial_value) - 1) * 100
            logger.info(f"Performance Summary:")
            logger.info(f"  Initial Value: ${initial_value:,.2f}")
            logger.info(f"  Final Value: ${final_value:,.2f}")
            logger.info(f"  Total Return: {total_return:.2f}%")
        
        logger.info("Trading strategy execution completed successfully")
        
        return True, {
            'orders': orders,
            'portfolio_history': portfolio_history,
            'portfolio_value': portfolio_value,
            'warning_message': warning_message,
            'signal_correlation': correlation,
            'buy_hit_rate': buy_hit_rate,
            'sell_hit_rate': sell_hit_rate
        }
        
    except ValueError as ve:
        # User input errors
        logger.error(f"Input validation error: {ve}")
        return False, {
            'orders': [],
            'portfolio_history': [],
            'portfolio_value': 0.0,
            'warning_message': str(ve),
            'signal_correlation': 0.0,
            'buy_hit_rate': 0.0,
            'sell_hit_rate': 0.0
        }
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in execute_trading_strategy: {e}", exc_info=True)
        return False, {
            'orders': [],
            'portfolio_history': [],
            'portfolio_value': 0.0,
            'warning_message': f"Error executing strategy: {e}",
            'signal_correlation': 0.0,
            'buy_hit_rate': 0.0,
            'sell_hit_rate': 0.0
        }

def get_order_history_df():
    """Return the order history as a DataFrame."""
    logger.debug("Retrieving order history...")
    
    try:
        orders = get_orders()
        if not orders:
            logger.info("No orders found in history")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(orders)} orders from history")
        
        order_df = pd.DataFrame(orders)
        logger.debug(f"Order DataFrame shape: {order_df.shape}")
        logger.debug(f"Order DataFrame columns: {list(order_df.columns)}")
        
        # Expected columns for validation
        expected_columns = [
            'date', 'ticker', 'action', 'shares_amount', 'price', 
            'investment_amount', 'transaction_cost', 'previous_shares', 
            'new_total_shares', 'sharpe', 'ticker_weight', 'weighted_allocation'
        ]
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(order_df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in order DataFrame: {missing_cols}")
            # Return DataFrame with expected columns (empty)
            return pd.DataFrame(columns=expected_columns)
        
        logger.info("Order history DataFrame created successfully")
        return order_df
        
    except Exception as e:
        logger.error(f"Error creating order history DataFrame: {e}", exc_info=True)
        return pd.DataFrame()

def log_trading_summary():
    """Log a summary of current trading state."""
    try:
        logger.info("="*40)
        logger.info("TRADING STATE SUMMARY")
        logger.info("="*40)
        
        # Get current orders
        orders = get_orders()
        logger.info(f"Total orders in history: {len(orders)}")
        
        if orders:
            # Count by action type
            buy_orders = [o for o in orders if o.get('action') == 'buy']
            sell_orders = [o for o in orders if o.get('action') == 'sell']
            
            logger.info(f"  Buy orders: {len(buy_orders)}")
            logger.info(f"  Sell orders: {len(sell_orders)}")
            
            # Get unique tickers
            tickers = set(o.get('ticker', 'Unknown') for o in orders)
            logger.info(f"  Unique tickers traded: {len(tickers)}")
            logger.info(f"  Tickers: {sorted(list(tickers))}")
            
            # Calculate total investment
            total_investment = sum(o.get('investment_amount', 0) for o in buy_orders)
            total_sales = sum(o.get('investment_amount', 0) for o in sell_orders)
            
            logger.info(f"  Total invested: ${total_investment:,.2f}")
            logger.info(f"  Total sales: ${total_sales:,.2f}")
        
        # Get portfolio history
        portfolio_history = get_portfolio_history()
        logger.info(f"Portfolio history entries: {len(portfolio_history)}")
        
        if portfolio_history:
            latest_entry = portfolio_history[-1]
            logger.info(f"  Latest portfolio value: ${latest_entry.get('value', 0):,.2f}")
            logger.info(f"  Latest cash: ${latest_entry.get('cash', 0):,.2f}")
            logger.info(f"  Active positions: {len(latest_entry.get('holdings', {}))}")
        
        logger.info("="*40)
        
    except Exception as e:
        logger.error(f"Error generating trading summary: {e}", exc_info=True)