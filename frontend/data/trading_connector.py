import sys
import os
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.append(backend_path)
print(f"Added to sys.path: {backend_path}")  # Debug print
import pandas as pd
import logging

from backend.trading_logic_new import run_trading_strategy, get_orders, get_portfolio_history, validate_prediction_quality
from frontend.logging_config import get_logger
logger = get_logger(__name__)

# Use centralized logger
logger = get_logger('trading_logic')  # Align with logging_config.py logger name
logger.setLevel(logging.INFO)

def execute_trading_strategy(investment_amount, risk_level, start_date, end_date, data_manager, mode="automatic", reset_state=True, selected_orders=None):
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

        warning_message = ""

        logger.info(f"Calling trading strategy...")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Reset State: {reset_state}")
        logger.info(f"  Selected Orders: {len(selected_orders) if selected_orders else 'None'}")

        result = run_trading_strategy(
            merged_data=merged_data,
            investment_amount=investment_amount,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            reset_state=reset_state,
            selected_orders=selected_orders
        )

        portfolio_history = get_portfolio_history()
        cash = portfolio_history[-1].get('cash', investment_amount) if portfolio_history else investment_amount
        portfolio_value = portfolio_history[-1].get('value', investment_amount) if portfolio_history else investment_amount

        if mode == "semi-automatic":
            orders, warning_message = result
            logger.info(f"Semi-automatic mode completed - {len(orders)} orders suggested")
        else:
            orders, portfolio_history, portfolio_value, warning_message = result
            cash = portfolio_history[-1].get('cash', investment_amount) if portfolio_history else investment_amount
            logger.info(f"Automatic mode completed:")
            logger.info(f"  Orders executed: {len(orders)}")
            logger.info(f"  Portfolio history entries: {len(portfolio_history)}")
            logger.info(f"  Final portfolio value: ${portfolio_value:,.2f}")

        logger.info(f"Orders returned: {len(orders)}")

        if portfolio_history:
            initial_value = investment_amount
            final_value = portfolio_value
            total_return = ((final_value - initial_value) / initial_value) * 100
            logger.info(f"Performance Summary:")
            logger.info(f"  Initial Value: ${initial_value:,.2f}")
            logger.info(f"  Final Value: ${final_value:,.2f}")
            logger.info(f"  Total Return: {total_return:.2f}%")

        logger.info("Trading strategy execution completed successfully")

        return True, {
            'orders': orders,
            'portfolio_history': portfolio_history,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'warning_message': warning_message,
            'signal_correlation': correlation,
            'buy_hit_rate': buy_hit_rate,
            'sell_hit_rate': sell_hit_rate
        }

    except ValueError as ve:
        logger.error(f"Input validation error: {ve}")
        portfolio_history = get_portfolio_history()
        cash = portfolio_history[-1].get('cash', investment_amount) if portfolio_history else investment_amount
        return False, {
            'orders': [],
            'portfolio_history': portfolio_history,
            'portfolio_value': 0.0,
            'cash': cash,
            'warning_message': str(ve),
            'signal_correlation': 0.0,
            'buy_hit_rate': 0.0,
            'sell_hit_rate': 0.0
        }

    except Exception as e:
        logger.error(f"Unexpected error in execute_trading_strategy: {e}", exc_info=True)
        portfolio_history = get_portfolio_history()
        cash = portfolio_history[-1].get('cash', investment_amount) if portfolio_history else investment_amount
        return False, {
            'orders': [],
            'portfolio_history': portfolio_history,
            'portfolio_value': 0.0,
            'cash': cash,
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

        logger.info(f"Retrieved {len(orders)} orders from orders history")
        
        order_df = pd.DataFrame(orders)
        logger.debug(f"Order DataFrame shape: {order_df.shape}")
        logger.debug(f"Order DataFrame columns: {list(order_df.columns)}")

        expected_columns = [
            'date', 'ticker', 'action', 'shares_amount', 'price',
            'investment_amount', 'transaction_cost', 'previous_shares',
            'new_total_shares', 'sharpe', 'ticker_weight', 'weighted_allocation'
        ]

        missing_cols = set(expected_columns) - set(order_df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in order DataFrame: {missing_cols}")
            for col in missing_cols:
                order_df[col] = None  # Add missing columns for consistency

        logger.info("Order history DataFrame created successfully")
        return order_df

    except Exception as e:
        logger.error(f"Error generating order history DataFrame: {e}", exc_info=True)
        return pd.DataFrame()

def log_trading_orders():
    """Log a summary of current trading orders."""
    try:
        logger.info("="*50)
        logger.info("TRADING ORDERS SUMMARY")
        logger.info("="*50)
        
        orders = get_orders()
        logger.info(f"Total orders in history: {len(orders)}")
        
        if orders:
            buy_orders = [o for o in orders if o.get('action') == 'buy']
            sell_orders = [o for o in orders if o.get('action') == 'sell']
            
            logger.info(f"  Buy orders: {len(buy_orders)}")
            logger.info(f"  Sell orders: {len(sell_orders)}")
            
            tickers = sorted(set(o.get('ticker', 'Unknown') for o in orders))
            logger.info(f"  Unique tickers traded: {len(tickers)}")
            logger.info(f"  Tickers: {tickers}")
            
            total_investment = sum(o.get('investment_amount', 0) for o in buy_orders)
            total_sales = sum(o.get('investment_amount', 0) for o in sell_orders)
            
            logger.info(f"  Total invested: ${total_investment:,.2f}")
            logger.info(f"  Total sales: ${total_sales:,.2f}")
        
        portfolio_history = get_portfolio_history()
        logger.info(f"Portfolio history entries: {len(portfolio_history)}")
        
        if portfolio_history:
            latest_entry = portfolio_history[-1]
            logger.info(f"  Latest portfolio value: ${latest_entry.get('value', 0):,.2f}")
            logger.info(f"  Latest cash: ${latest_entry.get('cash', 0):,.2f}")
            logger.info(f"  Active positions: {len(latest_entry.get('holdings', {}))}")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error generating trading orders summary: {e}", exc_info=True)