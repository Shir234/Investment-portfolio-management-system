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

def map_risk_threshold_to_sharpe(risk_level):
    """
    Maps risk level (0-100) to buy/sell thresholds for Sharpe ratios.
    Risk level 0 (most conservative): Buy ≥ 2.0, Sell ≤ -2.0.
    Risk level 100 (most aggressive): Buy ≥ 0.5, Sell ≤ -0.5.
    """
    base_buy_threshold = 2.0  # Very good Sharpe for conservative strategy
    base_sell_threshold = -2.0
    threshold_range = 1.5  # Difference between conservative and aggressive
    
    buy_threshold = base_buy_threshold - (risk_level / 100) * threshold_range
    sell_threshold = base_sell_threshold + (risk_level / 100) * threshold_range
    min_holding_days = 0  # No holding period for flexibility
    return buy_threshold, sell_threshold, min_holding_days

def load_ticker_weights(weights_file='final_tickers_score.csv'):
    """Load and normalize ticker weights from a CSV file."""
    try:
        if not os.path.exists(weights_file):
            logger.warning(f"Weights file {weights_file} not found. Using default weight of 1.0.")
            return {}
        
        df = pd.read_csv(weights_file)
        if 'Ticker' not in df.columns or 'Weight' not in df.columns:
            logger.error(f"Invalid weights file format. Expected columns: Ticker, Weight.")
            return {}
        
        # Ensure weights are positive and finite
        df = df[df['Weight'].notnull() & (df['Weight'] > 0)]
        if df.empty:
            logger.warning("No valid weights found in weights file. Using default weight of 1.0.")
            return {}
        
        # Normalize weights to sum to 1
        total_weight = df['Weight'].sum()
        df['Weight'] = df['Weight'] / total_weight
        weights = dict(zip(df['Ticker'], df['Weight']))
        logger.debug(f"Loaded {len(weights)} ticker weights from {weights_file}")
        return weights
    
    except Exception as e:
        logger.error(f"Error loading ticker weights: {e}")
        return {}
def save_portfolio_state():
    """Save orders and portfolio history to portfolio_state.json."""
    state = {
        'orders': orders,
        'portfolio_history': portfolio_history
    }
    try:
        with open(portfolio_state_file, 'w') as f:
            json.dump(state, f, default=str)
        logger.debug(f"Portfolio state saved: {len(orders)} orders, {len(portfolio_history)} history entries")
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
            logger.debug(f"Portfolio state loaded: {len(orders)} orders, {len(portfolio_history)} history entries")
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
        ticker_weights = load_ticker_weights()
        defualt_weight = 1.0
        # Reset state if requested
        if reset_state:
            logger.debug("Resetting portfolio state")
            orders = []
            portfolio_history = []
            if os.path.exists(portfolio_state_file):
                os.remove(portfolio_state_file)
                logger.debug(f"Deleted {portfolio_state_file}")
        
        # Load existing state
        load_portfolio_state()
        
        # Validate required columns
        required_columns = ['date', 'Ticker', 'Close', 'Best_Prediction']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        if missing_columns:
            logger.error(f"Missing columns in merged_data: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Validate Close prices
        if merged_data['Close'].isnull().any() or (merged_data['Close'] <= 0).any():
            logger.error("Invalid Close prices detected: null or negative values")
            raise ValueError("Close prices contain null or negative values")
        if merged_data['Close'].max() > 10000:
            logger.warning(f"Extreme Close price detected: {merged_data['Close'].max()}. Verify data scaling.")

        # Map risk level to thresholds
        buy_threshold, sell_threshold, min_holding_days = map_risk_threshold_to_sharpe(risk_level)
        
        # Log potential buy and sell signals
        potential_buys = merged_data[merged_data['Best_Prediction'] >= buy_threshold][['date', 'Ticker', 'Best_Prediction', 'Close']]
        potential_sells = merged_data[merged_data['Best_Prediction'] <= sell_threshold][['date', 'Ticker', 'Best_Prediction', 'Close']]
        logger.info(f"Potential buy signals: {len(potential_buys)}")
        for _, row in potential_buys.iterrows():
            logger.info(f"Potential buy: {row['date'].date()}, {row['Ticker']}, Best_Prediction={row['Best_Prediction']:.4f}, Close={row['Close']:.2f}")
        logger.info(f"Potential sell signals: {len(potential_sells)}")
        for _, row in potential_sells.iterrows():
            logger.info(f"Potential sell: {row['date'].date()}, {row['Ticker']}, Best_Prediction={row['Best_Prediction']:.4f}, Close={row['Close']:.2f}")

        # Initialize portfolio
        cash = investment_amount
        holdings = {}
        total_buy_signals = 0
        total_sell_signals = 0
        warning_message = ""
        suggested_orders = [] if mode == "semi-automatic" else orders
        max_positions = 10  # Limit for diversification
        
        # Ensure dates are timezone-aware
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        for current_date in date_range:
            daily_data = merged_data[merged_data['date'].dt.date == current_date.date()]
            
            if daily_data.empty:
                portfolio_history.append({
                    'date': current_date,
                    'value': cash,
                    'holdings': holdings.copy()
                })
                continue
            
            # Buy signals
            buy_signals = daily_data[daily_data['Best_Prediction'] >= buy_threshold]
            total_buy_signals += len(buy_signals)
            if buy_signals.empty and not daily_data.empty:
                logger.debug(f"No buy signals for {current_date.date()}: Max Best_Prediction={daily_data['Best_Prediction'].max():.4f}")

            if not buy_signals.empty and len(holdings) < max_positions:
                total_sharpe_strength = sum(abs(buy_signals['Best_Prediction']))
                daily_target = cash / max(1, (end_date - current_date).days + 1)
                
                for _, row in buy_signals.iterrows():
                    ticker = row['Ticker']
                    price = row['Close']
                    prediction = row['Best_Prediction']
                    sharpe = row.get('Actual_Sharpe', 0.0)
                    
                    if price > 10000:
                        logger.warning(f"Skipping buy for {ticker} on {current_date.date()}: Unrealistic price {price:.2f}")
                        continue
                    
                    allocation = (abs(prediction) / total_sharpe_strength) * daily_target if total_sharpe_strength > 0 else daily_target / len(buy_signals)
                    shares_to_buy = int(allocation / price)
                    if shares_to_buy < 1:
                        continue
                    cost = shares_to_buy * price
                    if cost > cash:
                        logger.warning(f"Insufficient cash ({cash:.2f}) for {ticker} buy on {current_date.date()}: Cost = {cost:.2f}")
                        continue
                    order = {
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'buy',
                        'shares_amount': shares_to_buy,
                        'price': price,
                        'investment_amount': cost,
                        'previous_shares': holdings.get(ticker, {'shares': 0})['shares'],
                        'new_total_shares': holdings.get(ticker, {'shares': 0})['shares'] + shares_to_buy,
                        'sharpe': sharpe
                    }
                    if mode == "automatic":
                        holdings[ticker] = holdings.get(ticker, {'shares': 0, 'purchase_date': None, 'purchase_price': 0.0})
                        holdings[ticker]['shares'] += shares_to_buy
                        holdings[ticker]['purchase_date'] = current_date
                        holdings[ticker]['purchase_price'] = price
                        holdings[ticker]['position_type'] = 'LONG'
                        cash -= cost
                        orders.append(order)
                        logger.info(f"Buying {ticker} on {current_date.date()}: Best_Prediction={prediction:.4f}, Shares={shares_to_buy}, Price={price:.2f}, Cost={cost:.2f}, Cash={cash:.2f}")
                    elif mode == "semi-automatic":
                        suggested_orders.append(order)
                        logger.info(f"Suggesting buy {ticker} on {current_date.date()}: Best_Prediction={prediction:.4f}, Shares={shares_to_buy}, Price={price:.2f}, Cost={cost:.2f}")
            
            # Sell signals (for LONG) and SHORT positions
            sell_signals = []
            for ticker, holding in list(holdings.items()):
                holding_data = daily_data[daily_data['Ticker'] == ticker]
                if holding_data.empty:
                    continue
                
                row = holding_data.iloc[0]
                current_price = row['Close']
                prediction = row.get('Best_Prediction', -1.0)
                
                if current_price > 10000:
                    logger.warning(f"Skipping sell for {ticker} on {current_date.date()}: Unrealistic price {current_price:.2f}")
                    continue
                
                position_type = holding.get('position_type', 'LONG')
                should_sell = False
                if position_type == 'LONG' and (prediction < 1.0 or current_price < holding['purchase_price'] * 0.95):
                    should_sell = True
                elif position_type == 'SHORT' and (prediction > -1.0 or current_price > holding['purchase_price'] * 1.05):
                    should_sell = True
                
                if should_sell:
                    shares = holding['shares']
                    sale_value = shares * current_price
                    order = {
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'sell' if position_type == 'LONG' else 'cover',
                        'shares_amount': shares,
                        'price': current_price,
                        'investment_amount': sale_value,
                        'previous_shares': shares,
                        'new_total_shares': 0,
                        'sharpe': row.get('Actual_Sharpe', 0.0)
                    }
                    if mode == "automatic":
                        cash += sale_value
                        orders.append(order)
                        sell_signals.append(order)
                        total_sell_signals += 1
                        logger.info(f"{'Selling' if position_type == 'LONG' else 'Covering'} {ticker} on {current_date.date()}: Best_Prediction={prediction:.4f}, Shares={shares}, Price={current_price:.2f}, Value={sale_value:.2f}, Cash={cash:.2f}")
                        del holdings[ticker]
                    elif mode == "semi-automatic":
                        suggested_orders.append(order)
                        logger.info(f"Suggesting {'sell' if position_type == 'LONG' else 'cover'} {ticker} on {current_date.date()}: Best_Prediction={prediction:.4f}, Shares={shares}, Price={current_price:.2f}, Value={sale_value:.2f}")
            
            # SHORT positions
            short_signals = daily_data[daily_data['Best_Prediction'] <= sell_threshold]
            if not short_signals.empty and len(holdings) < max_positions:
                total_sharpe_strength = sum(abs(short_signals['Best_Prediction']))
                daily_target = cash / max(1, (end_date - current_date).days + 1)
                
                for _, row in short_signals.iterrows():
                    ticker = row['Ticker']
                    price = row['Close']
                    prediction = row['Best_Prediction']
                    sharpe = row.get('Actual_Sharpe', 0.0)
                    
                    if price > 10000:
                        logger.warning(f"Skipping short for {ticker} on {current_date.date()}: Unrealistic price {price:.2f}")
                        continue
                    
                    allocation = (abs(prediction) / total_sharpe_strength) * daily_target if total_sharpe_strength > 0 else daily_target / len(short_signals)
                    shares_to_short = int(allocation / price)
                    if shares_to_short < 1:
                        continue
                    cost = shares_to_short * price
                    if cost > cash:
                        logger.warning(f"Insufficient cash ({cash:.2f}) for {ticker} short on {current_date.date()}: Cost = {cost:.2f}")
                        continue
                    order = {
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'short',
                        'shares_amount': shares_to_short,
                        'price': price,
                        'investment_amount': cost,
                        'previous_shares': holdings.get(ticker, {'shares': 0})['shares'],
                        'new_total_shares': holdings.get(ticker, {'shares': 0})['shares'] + shares_to_short,
                        'sharpe': sharpe
                    }
                    if mode == "automatic":
                        holdings[ticker] = holdings.get(ticker, {'shares': 0, 'purchase_date': None, 'purchase_price': 0.0})
                        holdings[ticker]['shares'] += shares_to_short
                        holdings[ticker]['purchase_date'] = current_date
                        holdings[ticker]['purchase_price'] = price
                        holdings[ticker]['position_type'] = 'SHORT'
                        cash -= cost
                        orders.append(order)
                        logger.info(f"Shorting {ticker} on {current_date.date()}: Best_Prediction={prediction:.4f}, Shares={shares_to_short}, Price={price:.2f}, Cost={cost:.2f}, Cash={cash:.2f}")
                    elif mode == "semi-automatic":
                        suggested_orders.append(order)
                        logger.info(f"Suggesting short {ticker} on {current_date.date()}: Best_Prediction={prediction:.4f}, Shares={shares_to_short}, Price={price:.2f}, Cost={cost:.2f}")
            
            # Calculate current portfolio value
            current_value = cash
            for ticker, holding in holdings.items():
                ticker_data = daily_data[daily_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    holding_value = holding['shares'] * ticker_data.iloc[0]['Close']
                    if holding_value > investment_amount * 10:
                        logger.warning(f"Capping holding value for {ticker} on {current_date.date()}: {holding_value:.2f} exceeds 10x initial investment")
                        holding_value = investment_amount * 10
                    current_value += holding_value
                    logger.debug(f"Holding {ticker}: {holding['shares']} shares, Price={ticker_data.iloc[0]['Close']:.2f}, Value={holding_value:.2f}")
            
            if current_value > investment_amount * 10:
                logger.warning(f"Capping portfolio value on {current_date.date()}: {current_value:.2f} exceeds 10x initial investment")
                current_value = investment_amount * 10
            
            portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'holdings': holdings.copy()
            })
        
        # Log summary if no signals detected
        if total_buy_signals == 0 and total_sell_signals == 0:
            warning_message = f"No signals detected for the period. Buy threshold ({buy_threshold:.4f}) exceeds max Best_Prediction ({merged_data['Best_Prediction'].max():.4f})."
            logger.warning(warning_message)
        
        # Log final orders
        logger.info(f"Final orders: {len(orders)} orders")
        for order in orders:
            logger.info(f"Order: {order}")
        
        # Log portfolio history summary
        logger.debug(f"Portfolio history length: {len(portfolio_history)}")
        if portfolio_history:
            logger.debug(f"Portfolio history sample: {portfolio_history[:5] + portfolio_history[-5:]}")
        
        # Save final state only in automatic mode
        if mode == "automatic":
            save_portfolio_state()
        logger.info(f"Final Portfolio Value: {current_value:.2f}")
        logger.debug("run_trading_strategy completed")
        
        # Return based on mode
        return_value = (orders, portfolio_history, current_value, warning_message) if mode == "automatic" else (suggested_orders, warning_message)
        return return_value
    
    except Exception as e:
        logger.error(f"Error in run_trading_strategy: {e}", exc_info=True)
        warning_message = f"Error running strategy: {e}"
        return ([], [], 0.0, warning_message) if mode == "automatic" else ([], warning_message)

def get_order_history():
    """Return the order history as a DataFrame."""
    return pd.DataFrame(orders)