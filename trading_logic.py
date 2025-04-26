import pandas as pd
import logging
from datetime import timedelta
import json
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

orders = []
portfolio_history = []
portfolio_state_file = 'data/portfolio_state.json'

def map_risk_threshold_to_sharpe(risk_level, merged_data):
    """Map risk level (0-100) to buy/sell thresholds and holding period."""
    risk_level = max(1, min(100, risk_level))
    buy_threshold = 3.0 - (risk_level / 100) * 2.5  # Scales from 3.0 (conservative) to 0.5 (aggressive)
    sell_threshold = -buy_threshold
    min_holding_days = max(5, int(30 - (risk_level / 100) * 25))  # Scales from 30 to 5 days
    logger.info(f"Risk level: {risk_level}, buy_threshold: {buy_threshold}, sell_threshold: {sell_threshold}, min_holding_days: {min_holding_days}")
    return buy_threshold, sell_threshold, min_holding_days

def save_portfolio_state():
    """Save orders and portfolio history to portfolio_state.json."""
    state = {
        'orders': orders,
        'portfolio_history': portfolio_history
    }
    try:
        with open(portfolio_state_file, 'w') as f:
            json.dump(state, f, default=str)
        logger.debug("Portfolio state saved successfully")
    except Exception as e:
        logger.error(f"Error saving portfolio state: {e}")

def load_portfolio_state():
    """Load orders and portfolio history from portfolio_state.json."""
    global orders, portfolio_history
    if os.path.exists(portfolio_state_file):
        try:
            with open(portfolio_state_file, 'r') as f:
                state = json.load(f)
            orders = state.get('orders', [])
            portfolio_history = state.get('portfolio_history', [])
            logger.debug("Portfolio state loaded successfully")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            orders = []
            portfolio_history = []

def get_orders():
    """Return a copy of the orders list."""
    logger.debug(f"Returning orders: {len(orders)}")
    return orders.copy()

def get_portfolio_history():
    """Return a copy of the portfolio history list."""
    logger.debug(f"Returning portfolio history: {len(portfolio_history)}")
    return portfolio_history.copy()

def run_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, data_manager, mode="automatic", reset_state=False):
    """Execute the trading strategy over the specified date range."""
    logger.debug("Starting run_trading_strategy")
    global orders, portfolio_history
    
    try:
        # Reset state if requested
        if reset_state:
            logger.debug("Resetting portfolio state")
            orders = []
            portfolio_history = []
            if os.path.exists(portfolio_state_file):
                os.remove(portfolio_state_file)
        
        # Load existing state
        load_portfolio_state()
        
        # Map risk level to thresholds
        buy_threshold, sell_threshold, min_holding_days = map_risk_threshold_to_sharpe(risk_level, merged_data)
        
        # Initialize portfolio
        current_date = start_date
        portfolio_value = investment_amount
        holdings = {}
        
        # Ensure dates are timezone-aware
        merged_data['date'] = pd.to_datetime(merged_data['date'], utc=True)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for current_date in date_range:
            current_date = current_date.replace(tzinfo=pd.Timestamp.utcnow().tz)
            daily_data = merged_data[merged_data['date'].dt.date == current_date.date()]
            
            if daily_data.empty:
                logger.debug(f"No data for {current_date.date()}")
                portfolio_history.append({
                    'date': current_date,
                    'value': portfolio_value,
                    'holdings': holdings.copy()
                })
                continue
            
            # Buy signals
            buy_signals = daily_data[daily_data['Best_Prediction'] >= buy_threshold]
            logger.info(f"Day: {current_date.date()}, Buy signals: {len(buy_signals)}")
            
            if mode == "automatic" and not buy_signals.empty:
                for _, row in buy_signals.iterrows():
                    ticker = row['Ticker']
                    price = row['Close']
                    prediction = row['Best_Prediction']
                    sharpe = row.get('Actual_Sharpe', 0.0)
                    
                    # Allocate 10% of initial investment per trade
                    shares_to_buy = int((investment_amount * 0.1) / price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        if cost <= portfolio_value:
                            holdings[ticker] = holdings.get(ticker, {'shares': 0, 'purchase_date': None})
                            holdings[ticker]['shares'] += shares_to_buy
                            holdings[ticker]['purchase_date'] = current_date
                            portfolio_value -= cost
                            
                            order = {
                                'date': current_date,
                                'ticker': ticker,
                                'action': 'buy',
                                'shares_amount': shares_to_buy,
                                'price': price,
                                'investment_amount': cost,
                                'previous_shares': holdings[ticker]['shares'] - shares_to_buy,
                                'new_total_shares': holdings[ticker]['shares'],
                                'sharpe': sharpe
                            }
                            orders.append(order)
                            logger.info(f"Buying {ticker} on {current_date.date()}: Best_Prediction = {prediction}, Shares = {shares_to_buy}, Price = {price}")
            
            # Sell signals
            for ticker, holding in list(holdings.items()):
                holding_data = daily_data[daily_data['Ticker'] == ticker]
                if holding_data.empty:
                    continue
                
                row = holding_data.iloc[0]
                current_price = row['Close']
                prediction = row.get('Best_Prediction', -1.0)
                
                days_held = (current_date - holding['purchase_date']).days if holding['purchase_date'] else 0
                
                if days_held >= min_holding_days and prediction < sell_threshold:
                    shares = holding['shares']
                    sale_value = shares * current_price
                    portfolio_value += sale_value
                    
                    order = {
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'sell',
                        'shares_amount': shares,
                        'price': current_price,
                        'investment_amount': sale_value,
                        'previous_shares': shares,
                        'new_total_shares': 0,
                        'sharpe': row.get('Actual_Sharpe', 0.0)
                    }
                    orders.append(order)
                    logger.info(f"Selling {ticker} on {current_date.date()}: Best_Prediction ({prediction}) < Sell threshold ({sell_threshold})")
                    
                    del holdings[ticker]
            
            # Calculate current portfolio value
            current_value = portfolio_value
            for ticker, holding in holdings.items():
                ticker_data = daily_data[daily_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    current_value += holding['shares'] * ticker_data.iloc[0]['Close']
            
            portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'holdings': holdings.copy()
            })
        
        # Save final state
        save_portfolio_state()
        logger.info(f"Final Portfolio Value: {portfolio_value}")
        logger.debug("run_trading_strategy completed successfully")
        
        # Return based on mode
        return orders, portfolio_history, portfolio_value if mode == "automatic" else orders
    
    except Exception as e:
        logger.error(f"Error in run_trading_strategy: {e}", exc_info=True)
        return [], [], 0.0 if mode == "automatic" else []

def get_order_history():
    """Return the order history as a DataFrame."""
    return pd.DataFrame(orders)