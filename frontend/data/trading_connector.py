import sys
import os
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.append(backend_path)
print(f"Added to sys.path: {backend_path}")  # Debug print
import pandas as pd
import logging
from backend.trading_logic_new import load_ticker_weights
from datetime import timedelta

from backend.trading_logic_new import run_trading_strategy, get_orders, get_portfolio_history, validate_prediction_quality
from frontend.logging_config import get_logger
logger = get_logger(__name__)

# Use centralized logger
logger = get_logger('trading_logic')  # Align with logging_config.py logger name
logger.setLevel(logging.INFO)

def execute_trading_strategy(investment_amount, risk_level, start_date, end_date, data_manager, mode, reset_state, selected_orders):
    # Define project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    weights_file = os.path.join(project_root, 'Investment-portfolio-management-system', 'backend', 'resources', 'final_tickers_score.csv')
    
    # Log debugging information
    logger.info(f"EXECUTE TRADING STRATEGY CALLED")
    logger.info(f"==================================================")
    logger.info(f"Parameters: Investment=${investment_amount}, Risk={risk_level}, Mode={mode}, Reset={reset_state}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Using weights file: {weights_file}")
    
    # Load weights
    weights = load_ticker_weights(weights_file=weights_file)
    if not weights:
        logger.warning(f"No valid weights loaded, using default weight of 0.02")
        weights = {ticker: 0.02 for ticker in ['ABBV', 'CTSH', 'FICO', 'ATO', 'LLY']}  # Fallback tickers

    # Validate data
    filtered_data = data_manager.get_filtered_data()
    if filtered_data is None:
        logger.error("Filtered data is None, likely due to unset date range")
        return False, {
            'orders': [],
            'portfolio_value': investment_amount,
            'warning_message': "Filtered data is None, likely due to unset date range",
            'portfolio_history': []
        }
    
    logger.info(f"Data validation passed - Shape: {filtered_data.shape}")
    logger.info(f"Data columns: {list(filtered_data.columns)}")
    
    if filtered_data.empty:
        logger.warning("No data available for the specified date range")
        return True, {
            'orders': [],
            'portfolio_value': investment_amount,
            'warning_message': "No data available for the specified date range",
            'portfolio_history': []
        }

    # Log prediction quality
    logger.info("Validating prediction quality...")
    sharpe_return_corr = filtered_data[['Actual_Sharpe', 'Best_Prediction']].corr().iloc[0, 1]
    logger.info(f"Sharpe-Return Correlation: {sharpe_return_corr:.3f}")
    logger.info(f"Sharpe Range: [{filtered_data['Actual_Sharpe'].min():.2f}, {filtered_data['Actual_Sharpe'].max():.2f}]")
    strong_buy_hit_rate = len(filtered_data[filtered_data['Actual_Sharpe'] >= 1.51]) / len(filtered_data) * 100
    strong_sell_hit_rate = len(filtered_data[filtered_data['Actual_Sharpe'] <= -1.24]) / len(filtered_data) * 100
    logger.info(f"Strong Buy Hit Rate (Sharpe ≥ 1.51): {strong_buy_hit_rate:.1f}%")
    logger.info(f"Strong Sell Hit Rate (Sharpe ≤ -1.24): {strong_sell_hit_rate:.1f}%")

    logger.info(f"Calling trading strategy...")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Reset State: {reset_state}")
    logger.info(f"  Selected Orders: {selected_orders}")
    logger.info(f"Starting integrated trading strategy - Risk Level: {risk_level}")

    # Threshold logic
    base_buy_threshold = 4.221
    base_sell_threshold = -3.343
    scaling_factor = 2.0
    buy_threshold = base_buy_threshold + risk_level * scaling_factor
    sell_threshold = base_sell_threshold - risk_level * scaling_factor
    logger.info(f"Risk Level {risk_level}: Buy threshold = {buy_threshold:.3f}, Sell threshold = {sell_threshold:.3f}")
    logger.info(f"Max Best_Prediction: {filtered_data['Best_Prediction'].max()}, Min Best_Prediction: {filtered_data['Best_Prediction'].min()}")

    if filtered_data['Best_Prediction'].max() < buy_threshold and filtered_data['Best_Prediction'].min() > sell_threshold:
        logger.warning(f"No trades possible: Max Best_Prediction {filtered_data['Best_Prediction'].max()} < Buy threshold {buy_threshold}, "
                       f"Min Best_Prediction {filtered_data['Best_Prediction'].min()} > Sell threshold {sell_threshold}")
        return True, {
            'orders': [],
            'portfolio_value': investment_amount,
            'warning_message': "No trades available due to high thresholds",
            'portfolio_history': []
        }

    # Trading logic
    orders = []
    cash = investment_amount
    portfolio_value = investment_amount
    portfolio_history = []
    current_date = start_date
    max_orders_per_day = 26
    allocation_per_ticker = investment_amount / max_orders_per_day if max_orders_per_day > 0 else investment_amount

    # Iterate over date range
    while current_date <= end_date:
        daily_data = filtered_data[filtered_data['date'].dt.date == current_date.date()]
        if not daily_data.empty:
            buy_candidates = daily_data[daily_data['Best_Prediction'] >= buy_threshold]
            sell_candidates = daily_data[daily_data['Best_Prediction'] <= sell_threshold]
            
            logger.info(f"Date {current_date.strftime('%Y-%m-%d')}: Found {len(buy_candidates)} buy candidates, {len(sell_candidates)} sell candidates")
            
            # Process buy candidates
            if not buy_candidates.empty:
                buy_candidates = buy_candidates.copy()
                buy_candidates['ticker_weight'] = buy_candidates['Ticker'].map(weights)
                buy_candidates['signal_strength'] = buy_candidates['Best_Prediction']
                buy_candidates = buy_candidates.sort_values(by=['ticker_weight', 'signal_strength', 'Ticker'], ascending=[False, False, True])
                top_candidates = buy_candidates.head(min(max_orders_per_day, len(buy_candidates)))
                logger.info(f"Date {current_date.strftime('%Y-%m-%d')}: Taking top {len(top_candidates)} buy candidates")
                
                for _, row in top_candidates.iterrows():
                    cost = allocation_per_ticker
                    shares = int(cost / row['Close'])
                    if shares > 0 and cost <= cash:
                        orders.append({
                            'type': 'buy',
                            'ticker': row['Ticker'],
                            'shares': shares,
                            'price': row['Close'],
                            'date': row['date']
                        })
                        logger.info(f"NEW BUY {row['Ticker']}: {shares} shares @ ${row['Close']:.2f}")
                        cash -= shares * row['Close']
                    else:
                        logger.warning(f"Skipping {row['Ticker']} buy: Insufficient cash ${cash:.2f} < ${cost:.2f}")

            # Process sell candidates (placeholder)
            if not sell_candidates.empty:
                logger.info(f"Date {current_date.strftime('%Y-%m-%d')}: Sell logic not implemented")
        
        portfolio_value = cash  # Update with actual holdings if sell logic is implemented
        portfolio_history.append({'date': current_date, 'value': portfolio_value})
        current_date += timedelta(days=1)

    logger.info(f"Buy logic completed: Generated {len(orders)} orders")
    logger.info(f"Strategy completed: Buy orders={len([o for o in orders if o['type'] == 'buy'])}, Sell orders={len([o for o in orders if o['type'] == 'sell'])}")
    logger.info(f"Final value: ${portfolio_value:.2f} ({(portfolio_value/investment_amount-1)*100:.1f}% return)")
    logger.info(f"Automatic mode completed:")
    logger.info(f"  Orders executed: {len(orders)}")
    logger.info(f"  Portfolio history entries: {len(portfolio_history)}")
    logger.info(f"  Final portfolio value: ${portfolio_value:.2f}")
    
    return True, {
        'orders': orders,
        'portfolio_value': portfolio_value,
        'warning_message': "",
        'portfolio_history': portfolio_history
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