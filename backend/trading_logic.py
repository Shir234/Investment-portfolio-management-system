import pandas as pd
import numpy as np
import logging
import datetime
import json
import os

from logging_config import get_isolated_logger

# Create isolated trading logger - ONLY trading logic will appear in this log file
logger = get_isolated_logger("trading_logic", "trading_only", logging.INFO)

# Test the isolated logger
logger.info("="*60)
logger.info("TRADING LOGIC SYSTEM INITIALIZED")
logger.info("="*60)

# Global variables for portfolio state
orders = []
portfolio_history = []
portfolio_state_file = 'data/portfolio_state.json'


def map_risk_to_sharpe_thresholds(risk_level, sharpe_min, sharpe_max):
    """
    Calculate dynamic buy/sell thresholds based on sharpe range in data and user chosen risk level.
    
    Args:
        risk_level: 0-10 (0=conservative, 10=aggressive)
        sharpe_min, sharpe_max: sharpe ratio limits in data
    
    Returns:
        tuple: (buy_threshold, sell_threshold)
    """
    
    # Calculate sharpe ratio range
    sharpe_range = sharpe_max - sharpe_min
    buy_percentile = 85 - (risk_level * 7)  # Risk 0: 85%, Risk 10: 15%
    sell_percentile = 15 + (risk_level * 7)  # Risk 0: 15%, Risk 10: 85%
    
    # Get buy & sell threshold in sharpe range      
    buy_threshold = sharpe_max - (sharpe_range * (100 - buy_percentile) / 100)
    sell_threshold = sharpe_min + (sharpe_range * sell_percentile / 100)
    
    # Make sure the thresholds in appropriate range
    buy_threshold = min(max(buy_threshold, sharpe_min), sharpe_max)
    sell_threshold = min(max(sell_threshold, sharpe_min), sharpe_max)
    
    logger.info(f"Risk Level {risk_level}: Buy threshold = {buy_threshold:.3f}, Sell threshold = {sell_threshold:.3f}")
    return buy_threshold, sell_threshold


def load_ticker_weights(weights_file='final_tickers_score.csv'):
    """Load and normalize ticker weights from a CSV file."""
    try:
        # Check that the file exists
        if not os.path.exists(weights_file):
            logger.warning(f"Weights file {weights_file} not found. Using default weight of 1.0.")
            return {}
        
        df = pd.read_csv(weights_file)
        weight_col = 'Weight' if 'Weight' in df.columns else 'Transaction_Score'

        if 'Ticker' not in df.columns or 'weight_col' not in df.columns:
            logger.error(f"Invalid weights file format.")
            return {}
        
        # Clean and normalize weights - ensure weights are positive and finite
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        df = df[df[weight_col].notnull() & (df[weight_col] > 0)]

        if df.empty:
            logger.warning("No valid weights found in weights file. Using default weight of 1.0.")
            return {}
        
        # Normalize weights to sum to 1
        total_weight = df['Weight'].sum()
        df['Weight'] = df[weight_col] / total_weight

        weights = dict(zip(df['Ticker'], df['Weight']))
        logger.debug(f"Loaded {len(weights)} ticker weights from {weights_file}")

        return weights
    
    except Exception as e:
        logger.error(f"Error loading ticker weights: {e}")
        return {}
    

def calculate_position_size(sharpe_value, ticker_weight, available_cash, sharpe_threshold, max_position_pct=0.3):
    """
    Calculate position size based on signal strength and ticker weight.
    Position size = amount of money to invest in specific stock.
    """
    # Signal strength (how far above/below threshold)
    sharpe_strength = abs(sharpe_value - sharpe_threshold) / (sharpe_threshold + 0.1)
    sharpe_strength = min(sharpe_strength, 3.0)  # Cap extreme signals
    
    # Combine with ticker weight
    combined_score = sharpe_strength * ticker_weight

    # Base allocation (5-15% per position based on signal strength)
    base_allocation = min(combined_score * 0.25, max_position_pct)

    # Calculate position size (Signal strength × Ticker weight × Available cash)
    position_size = available_cash * base_allocation

    return max(position_size, 200)  # Increased minimum


def add_technical_indicators(data):
    """
    Add technical indicators to the data.

    Extra filter - Adds momentum confirmation to Sharpe predictions
    """

    data = data.sort_values(['Ticker', 'date'])
    
    # Moving averages
    data['MA10'] = data.groupby('Ticker')['Close'].rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
    data['MA50'] = data.groupby('Ticker')['Close'].rolling(50, min_periods=25).mean().reset_index(level=0, drop=True)
    
    # Volatility (20-day rolling standard deviation)
    data['Volatility'] = data.groupby('Ticker')['Close'].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    
    # Price momentum (5-day return)
    data['Momentum'] = data.groupby('Ticker')['Close'].pct_change(5)
    
    return data


def validate_prediction_quality(merged_data, forward_days=5):
    """
    Validate that predictions are reliable enough for trading.
    Test if predicted Sharpe predicts future returns.
    """

    merged_data = merged_data.sort_values(['Ticker', 'date']).copy()
    merged_data['forward_return'] = merged_data.groupby('Ticker')['Close'].pct_change(forward_days).shift(-forward_days)
    
    correlation = merged_data['Best_Prediction'].corr(merged_data['forward_return'])
    sharpe_min = merged_data['Best_Prediction'].min()
    sharpe_max = merged_data['Best_Prediction'].max()
    
    # Calculate hit rates for compatibility with calling code
    buy_threshold = sharpe_max * 0.25  # Dynamic based on range
    sell_threshold = sharpe_min * 0.25
    
    strong_buy = merged_data[merged_data['Best_Prediction'] >= buy_threshold]
    strong_sell = merged_data[merged_data['Best_Prediction'] <= sell_threshold]
    
    buy_hit_rate = (strong_buy['forward_return'] > 0).mean() if not strong_buy.empty else 0
    sell_hit_rate = (strong_sell['forward_return'] < 0).mean() if not strong_sell.empty else 0
    
    logger.info(f"Sharpe-Return Correlation: {correlation:.3f}")
    logger.info(f"Sharpe Range: [{sharpe_min:.2f}, {sharpe_max:.2f}]")
    logger.info(f"Strong Buy Hit Rate (Sharpe ≥ {buy_threshold:.2f}): {buy_hit_rate:.1%}")
    logger.info(f"Strong Sell Hit Rate (Sharpe ≤ {sell_threshold:.2f}): {sell_hit_rate:.1%}")
    
    return correlation, buy_hit_rate, sell_hit_rate, sharpe_min, sharpe_max


"""
def validate_prediction_quality(merged_data, min_correlation=0.6):


    if 'Actual_Sharpe' not in merged_data.columns:
        logger.warning("No Actual_Sharpe column for validation")
        return True  # Proceed if no validation data
    
    valid_data = merged_data.dropna(subset=['Best_Prediction', 'Actual_Sharpe'])
    
    if len(valid_data) < 100:
        logger.warning(f"Insufficient data for validation: {len(valid_data)} samples")
        return True
    
    correlation = valid_data['Best_Prediction'].corr(valid_data['Actual_Sharpe'])
    logger.info(f"Prediction-Actual correlation: {correlation:.4f}")
    
    if correlation < min_correlation:
        logger.warning(f"Low correlation {correlation:.4f} < {min_correlation}")
        return False
    
    return True
"""


def save_portfolio_state():
    """
    Save orders and portfolio history to JSON file.
        - portfolio_state.json.
    """
    
    state = {
        'orders': orders,
        'portfolio_history': portfolio_history
    }

    try:
        os.makedirs(os.path.dirname(portfolio_state_file), exist_ok=True)
        with open(portfolio_state_file, 'w') as f:
            json.dump(state, f, default=str, indent=2)
        logger.debug(f"Portfolio state saved: {len(orders)} orders, {len(portfolio_history)} history entries")
    except Exception as e:
        logger.error(f"Error saving portfolio state: {e}")


def load_portfolio_state():
    """
    Load orders and portfolio history from portfolio_state.json.
    """

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
    """
    Return a copy of the orders list.
    """
    logger.debug(f"Returning orders: {len(orders)}")
    return orders.copy()


def get_portfolio_history():
    """
    Return a copy of the portfolio history list.
    """
    logger.debug(f"Returning portfolio history: {len(portfolio_history)}")
    return portfolio_history.copy()

"""
def run_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, data_manager, mode="automatic", reset_state=False):
    
    Execute Sharpe-based trading strategy with dynamic thresholds and risk management.

    Args:
        merged_data: DataFrame with predictions and price data
        investment_amount: Initial cash amount
        risk_level: Risk level 0-10 (0=conservative, 10=aggressive)
        start_date: Start date for trading
        end_date: End date for trading
        mode: "automatic" or "semi-automatic"
        reset_state: Whether to reset portfolio state
    
    Returns:
        tuple: (orders, portfolio_history, final_value, warning_message) or 
               (suggested_orders, warning_message) for semi-automatic

    # Entry Rules
    - Buy when Sharpe >= buy_threshold AND MA10 > MA50 (momentum)
    - Position size scales with signal strength and ticker weight
    - Max 15 positions, max 20% daily cash deployment

    # Exit Rules  
    - Automatic sell after 5-7 trading days (based on your data timeframe)
    - Stop loss at 8% for conservative, 12% for aggressive
    - Take profit at 15% for any position
    - Sell if Sharpe prediction drops below neutral (0)

    # Risk Controls
    - Maximum 15% per position
    - Maximum 20% of cash deployed per day
    - Rebalance weekly based on new predictions
    - Emergency stop if portfolio drops >20% in 5 days

    # Diversification
    - Use your ticker weights to prefer high-scoring stocks
    - Limit sector concentration (max 30% in any sector)
    - Maintain 10-20% cash buffer
    

    logger.info(f"Starting trading strategy - Risk Level: {risk_level}")
    global orders, portfolio_history
    
    try:
        # Initialize- reset state if requested
        if reset_state:
            logger.info("Resetting portfolio state")
            orders = []
            portfolio_history = []

            if os.path.exists(portfolio_state_file):
                os.remove(portfolio_state_file)
                logger.info(f"Deleted {portfolio_state_file}")

        # Load existing state
        load_portfolio_state()

        # Validate required columns
        required_columns = ['date', 'Ticker', 'Close', 'Best_Prediction']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        if missing_columns:
            logger.error(f"Missing columns in merged_data: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Validate prediction quality
        if not validate_prediction_quality(merged_data):
            return [], [], investment_amount, "Prediction quality too low for trading"
        

        # Load ticker weights and calculate thresholds
        ticker_weights = load_ticker_weights()
        default_weight = 0.02
        
        sharpe_min = merged_data['Best_Prediction'].min()
        sharpe_max = merged_data['Best_Prediction'].max()

        # Map risk level to thresholds
        buy_threshold, sell_threshold = map_risk_to_sharpe_thresholds(risk_level, sharpe_min, sharpe_max)

        # TODO : ? 
        # Add technical indicators
        #merged_data = add_technical_indicators(merged_data.copy())

        # Initialize portfolio
        cash = investment_amount
        holdings = {}
        suggested_orders = [] if mode == "semi-automatic" else orders
        warning_message = ""

        # Portfolio parameters based on risk level
        max_positions =  12 + (risk_level // 3)  # 12-15 positions
        # TODO : deployment? 
        max_daily_deployment = 0.15 + (risk_level * 0.01)  # 15-25% daily deployment
        transaction_cost_bps = 5  # 0.05% transaction cost
        min_holding_days = max(3, 7 - risk_level // 2)  # 3-7 days based on risk


        # Date processing
        merged_data['date'] = pd.to_datetime(merged_data['date'], utc=True)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        logger.info(f"Trading parameters: Max positions={max_positions}, "
                   f"Daily deployment={max_daily_deployment:.1%}, "
                   f"Min holding={min_holding_days} days")
        

        for current_date in date_range:
            # Get daily data
            daily_data = merged_data[
                merged_data['date'].dt.floor('D') == current_date
            ].copy()
            
            if daily_data.empty:
                # Calculate portfolio value using last known prices
                current_value = cash
                for ticker, holding in holdings.items():
                    last_price_data = merged_data[
                        (merged_data['Ticker'] == ticker) & 
                        (merged_data['date'] <= current_date)
                    ].tail(1)
                    if not last_price_data.empty:
                        current_value += holding['shares'] * last_price_data.iloc[0]['Close']
                
                portfolio_history.append({
                    'date': current_date,
                    'value': current_value,
                    'holdings': holdings.copy(),
                    'cash': cash
                })
                continue

            logger.debug(f"Processing {current_date.date()}: {len(daily_data)} tickers")
            
            # Calculate available cash for new positions
            max_daily_cash = cash * max_daily_deployment




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
"""


def get_order_history():
    """Return the order history as a DataFrame."""
    return pd.DataFrame(orders)


def process_sell_signals(daily_data, holdings, current_date, risk_level, mode="automatic", orders=None, suggested_orders=None):
    """
    Process sell signals for existing LONG and SHORT positions.
    
    Args:
        daily_data: DataFrame with current day's data
        holdings: Dict of current holdings
        current_date: Current trading date
        risk_level: Risk level 0-10
        mode: "automatic" or "semi-automatic"
        orders: List to append executed orders (for automatic mode)
        suggested_orders: List to append suggested orders (for semi-automatic mode)
    
    Returns:
        tuple: (updated_cash, sell_orders_count)
    """
   
    cash_from_sales = 0
    sell_orders_count = 0
    transaction_cost_bps = 5  # 0.05% transaction cost
    min_holding_days = max(3, 7 - risk_level // 2)  # 3-7 days based on risk
    
    # Calculate risk-based thresholds
    stop_loss_pct = 0.08 + (risk_level * 0.004)  # 8-12% based on risk
    take_profit_pct = 0.15 + (risk_level * 0.01)  # 15-25% based on risk
    
    for ticker, holding in list(holdings.items()):
        ticker_data = daily_data[daily_data['Ticker'] == ticker]
        if ticker_data.empty:
            continue
        
        row = ticker_data.iloc[0]
        current_price = row['Close']
        current_sharpe = row['Best_Prediction']
        position_type = holding.get('position_type', 'LONG')
        
        # Calculate holding period
        days_held = (current_date - holding['purchase_date']).days
        
        # Exit conditions
        should_sell = False
        sell_reason = ""
        
        # # 1. Minimum holding period reached
        # if days_held >= min_holding_days:
        #     should_sell = True
        #     sell_reason = f"Minimum holding period ({min_holding_days} days)"
        
        # 2. Stop loss conditions
        if position_type == 'LONG':
            if current_price <= holding['purchase_price'] * (1 - stop_loss_pct):
                should_sell = True
                sell_reason = f"LONG stop loss ({stop_loss_pct:.1%})"
        # else:  # SHORT position
        #     if current_price >= holding['purchase_price'] * (1 + stop_loss_pct):
        #         should_sell = True
        #         sell_reason = f"SHORT stop loss ({stop_loss_pct:.1%})"
        
        # 3. Take profit conditions
        if position_type == 'LONG':
            if current_price >= holding['purchase_price'] * (1 + take_profit_pct):
                should_sell = True
                sell_reason = f"LONG take profit ({take_profit_pct:.1%})"
        # else:  # SHORT position
        #     if current_price <= holding['purchase_price'] * (1 - take_profit_pct):
        #         should_sell = True
        #         sell_reason = f"SHORT take profit ({take_profit_pct:.1%})"
        
        # 4. Signal deterioration
        if position_type == 'LONG' and current_sharpe < 0 and days_held >= 2:
            should_sell = True
            sell_reason = "LONG signal turned negative"
        # elif position_type == 'SHORT' and current_sharpe > 0 and days_held >= 2:
        #     should_sell = True
        #     sell_reason = "SHORT signal turned positive"
        
        if should_sell:
            shares = holding['shares']
            
            # Calculate P&L differently for LONG vs SHORT
            if position_type == 'LONG':
                sale_value = shares * current_price
                profit_loss = sale_value - (holding['purchase_price'] * shares)
            # else:  # SHORT position
            #     # For short: profit when price goes down
            #     sale_value = shares * holding['purchase_price']  # We borrowed at this price
            #     buy_back_cost = shares * current_price  # Cost to buy back
            #     profit_loss = sale_value - buy_back_cost
            #     sale_value = buy_back_cost  # Cash needed to close position
            
            transaction_cost = sale_value * (transaction_cost_bps / 10000)
            net_proceeds = sale_value - transaction_cost
            
            # # For SHORT positions, we get back the collateral minus the buyback cost
            # if position_type == 'SHORT':
            #     net_proceeds = holding.get('collateral', sale_value) - sale_value - transaction_cost
            
            order = {
                'date': current_date,
                'ticker': ticker,
                'action': 'sell' if position_type == 'LONG' else 'cover',
                'position_type': position_type,
                'shares_amount': shares,
                'price': current_price,
                'investment_amount': sale_value,
                'transaction_cost': transaction_cost,
                'total_cost': net_proceeds,
                'sharpe': current_sharpe,
                'profit_loss': profit_loss,
                'days_held': days_held,
                'sell_reason': sell_reason,
                'purchase_price': holding['purchase_price'],
                'previous_shares': shares,
                'new_total_shares': 0
            }
            
            if mode == "automatic":
                cash_from_sales += net_proceeds
                del holdings[ticker]
                if orders is not None:
                    orders.append(order)
                sell_orders_count += 1
                
                action_word = "SELL" if position_type == 'LONG' else "COVER"
                logger.info(f"{action_word} {ticker}: {sell_reason}, Days held={days_held}, "
                           f"P&L=${profit_loss:.2f}, Price=${current_price:.2f}")
            else:
                if suggested_orders is not None:
                    suggested_orders.append(order)
    
    return cash_from_sales, sell_orders_count


def process_buy_signals(daily_data, buy_threshold, sell_threshold, holdings, cash, current_date, risk_level, ticker_weights, default_weight, max_positions, max_daily_deployment, mode="automatic", orders=None, suggested_orders=None):
    """
    Process buy signals for both LONG and SHORT positions.
    
    Args:
        daily_data: DataFrame with current day's data
        buy_threshold: Threshold for LONG positions
        sell_threshold: Threshold for SHORT positions  
        holdings: Dict of current holdings
        cash: Available cash
        current_date: Current trading date
        risk_level: Risk level 0-10
        ticker_weights: Dict of ticker weights
        default_weight: Default weight for tickers not in weights
        max_positions: Maximum number of positions
        max_daily_deployment: Maximum daily cash deployment ratio
        mode: "automatic" or "semi-automatic"
        orders: List to append executed orders (for automatic mode)
        suggested_orders: List to append suggested orders (for semi-automatic mode)
    
    Returns:
        tuple: (cash_used, buy_orders_count)
    """

    risk_level = int(risk_level)

    cash_used = 0
    buy_orders_count = 0
    transaction_cost_bps = 5  # 0.05% transaction cost
    max_position_size_pct = 0.15 + (risk_level * 0.005)  # 15-20% max per position based on risk

    if len(holdings) >= max_positions or cash <= 1000:
        return cash_used, buy_orders_count
    
    # Calculate available cash for new positions
    max_daily_cash = cash * max_daily_deployment
    available_cash = min(cash * 0.8, max_daily_cash)
    
    # STEP 1: Filter for buy signal candidates  --> LONG POSITIONS - Buy when Sharpe >= buy_threshold
    buy_signal_candidates = daily_data[
        daily_data['Best_Prediction'] >= buy_threshold
    ].copy()
    
    if buy_signal_candidates.empty:
        return cash_used, buy_orders_count

    # STEP 2: Add weight information and prioritize
    buy_signal_candidates['ticker_weight'] = buy_signal_candidates['Ticker'].map(
        lambda x: ticker_weights.get(x, default_weight)
    )

    # STEP 3: Sort by weight (highest first), then by signal strength
    buy_signal_candidates['signal_strength'] = (
        buy_signal_candidates['Best_Prediction'] - buy_threshold
    )

    # Sort by: Weight (desc) -> Signal Strength (desc) -> Ticker (for consistency)
    buy_signal_candidates = buy_signal_candidates.sort_values([
        'ticker_weight', 'signal_strength', 'Ticker'
    ], ascending=[False, False, True])

    # STEP 4: Take only top N candidates based on available positions
    max_new_positions = int(max_positions - len(holdings))
    top_candidates = buy_signal_candidates.head(max_new_positions * 2)  # 2x buffer for filtering
    
    logger.info(f"Buy candidates: {len(buy_signal_candidates)} total, "
               f"considering top {len(top_candidates)} by weight")
    
    # STEP 5: Apply momentum filter if available
    if 'MA10' in top_candidates.columns and 'MA50' in top_candidates.columns:
        momentum_filtered = top_candidates[
            (top_candidates['MA10'] > top_candidates['MA50']) |
            (top_candidates['MA10'].isna())
        ]
        if not momentum_filtered.empty:
            top_candidates = momentum_filtered
            logger.info(f"After momentum filter: {len(top_candidates)} candidates")
    
    # STEP 6: Process candidates in weight order
    candidates_processed = 0
    max_candidates_to_process = min(len(top_candidates), max_new_positions + 3)
    
    for idx, row in top_candidates.head(max_candidates_to_process).iterrows():
        if available_cash < 500 or len(holdings) >= max_positions:
            break
            
        ticker = row['Ticker']
        price = row['Close']
        sharpe = row['Best_Prediction']
        ticker_weight = row['ticker_weight']
        signal_strength = row['signal_strength']
        
        # Validate price
        if price <= 0 or price > 5000:
            logger.warning(f"Skipping {ticker}: Invalid price ${price:.2f}")
            continue
        
        # Check if we already own this (for position sizing)
        current_position_value = 0
        if ticker in holdings:
            current_shares = holdings[ticker]['shares']
            current_position_value = current_shares * price
        
        # Calculate current position as percentage of total portfolio
        total_portfolio_value = cash + sum(
            holding['shares'] * daily_data[daily_data['Ticker'] == t]['Close'].iloc[0] 
            for t, holding in holdings.items() 
            if t in daily_data['Ticker'].values
        )
        
        current_position_pct = (current_position_value / total_portfolio_value) if total_portfolio_value > 0 else 0
        
        # Skip if we're already over-concentrated in this stock
        if current_position_pct >= max_position_size_pct:
            logger.info(f"Skipping {ticker}: Position size {current_position_pct:.1%} >= {max_position_size_pct:.1%}")
            continue
            
        # Calculate position size based on weight and signal strength
        base_position_size = calculate_position_size(sharpe, ticker_weight, available_cash, buy_threshold)
        
        # Adjust for existing positions
        if ticker in holdings:
            position_size = base_position_size * 0.5  # Reduce for additional purchases
            logger.info(f"Additional purchase for {ticker}: reducing size by 50%")
        else:
            position_size = base_position_size
        
        # Adjust for volatility if available
        if not pd.isna(row.get('Volatility', np.nan)):
            volatility_adj = max(0.5, min(1.5, 1.0 / (row['Volatility'] + 0.1)))
            position_size *= volatility_adj
        
        # Calculate shares and costs
        shares = int(position_size / price)
        if shares < 1:
            logger.debug(f"Skipping {ticker}: Position too small ({shares} shares)")
            continue
            
        actual_cost = shares * price
        transaction_cost = actual_cost * (transaction_cost_bps / 10000)
        total_cost = actual_cost + transaction_cost
        
        if total_cost > available_cash:
            # Try smaller position that fits available cash
            max_affordable_shares = int((available_cash - 50) / price)  # Leave buffer for transaction costs
            if max_affordable_shares >= 1:
                shares = max_affordable_shares
                actual_cost = shares * price
                transaction_cost = actual_cost * (transaction_cost_bps / 10000)
                total_cost = actual_cost + transaction_cost
            else:
                logger.debug(f"Skipping {ticker}: Can't afford minimum position")
                continue
        
        # Get current holding info
        current_shares = holdings.get(ticker, {}).get('shares', 0)
        new_total_shares = current_shares + shares
        
        # Create order
        order = {
            'date': current_date,
            'ticker': ticker,
            'action': 'buy',
            'position_type': 'LONG',
            'shares_amount': shares,
            'price': price,
            'investment_amount': actual_cost,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost,
            'sharpe': sharpe,
            'ticker_weight': ticker_weight,
            'signal_strength': signal_strength,
            'position_size_pct': (total_cost / cash) * 100,
            'previous_shares': current_shares,
            'new_total_shares': new_total_shares,
            'is_additional_purchase': ticker in holdings,
            'weight_rank': candidates_processed + 1
        }
        
        if mode == "automatic":
            # Update holdings
            if ticker in holdings:
                # Update existing position with weighted average price
                old_avg_price = holdings[ticker]['purchase_price']
                old_shares = holdings[ticker]['shares']
                
                total_old_cost = old_shares * old_avg_price
                total_new_cost = shares * price
                new_avg_price = (total_old_cost + total_new_cost) / (old_shares + shares)
                
                holdings[ticker]['shares'] = new_total_shares
                holdings[ticker]['purchase_price'] = new_avg_price
                holdings[ticker]['last_purchase_date'] = pd.Timestamp(current_date)
                
                logger.info(f"ADDING to {ticker} (Weight: {ticker_weight:.3f}, Rank: {candidates_processed + 1}): "
                           f"Old={old_shares}@${old_avg_price:.2f}, New={shares}@${price:.2f}, "
                           f"Total={new_total_shares}@${new_avg_price:.2f}")
            else:
                # Create new position
                holdings[ticker] = {
                    'shares': shares,
                    'purchase_date': pd.Timestamp(current_date),
                    'purchase_price': price,
                    'position_type': 'LONG',
                    'last_purchase_date': pd.Timestamp(current_date)
                }
                
                logger.info(f"NEW BUY {ticker} (Weight: {ticker_weight:.3f}, Rank: {candidates_processed + 1}): "
                           f"Sharpe={sharpe:.3f}, Shares={shares}, Price=${price:.2f}, Cost=${total_cost:.2f}")
            
            cash_used += total_cost
            available_cash -= total_cost
            
            if orders is not None:
                orders.append(order)
            buy_orders_count += 1
            
        else:  # semi-automatic mode
            if suggested_orders is not None:
                suggested_orders.append(order)
        
        candidates_processed += 1
        
        # Log top weighted candidates for transparency
        if candidates_processed <= 5:
            logger.info(f"Weight Rank {candidates_processed}: {ticker} "
                       f"(Weight: {ticker_weight:.3f}, Sharpe: {sharpe:.3f})")
    
    if candidates_processed == 0:
        logger.info("No suitable buy candidates after weight prioritization")
    else:
        logger.info(f"Processed {candidates_processed} weight-prioritized buy candidates")


    """
    # # LONG POSITIONS - Buy when Sharpe >= buy_threshold
    # long_candidates = daily_data[
    #     (daily_data['Best_Prediction'] >= buy_threshold) 
    #     #& (~daily_data['Ticker'].isin(holdings.keys()))
    # ].copy()

    # # SHORT POSITIONS - Sell when Sharpe <= sell_threshold  
    # short_candidates = daily_data[
    #     (daily_data['Best_Prediction'] <= sell_threshold) &
    #     (~daily_data['Ticker'].isin(holdings.keys()))
    # ].copy()
    
    # # Apply momentum filter for LONG positions (uptrend)
    # if 'MA10' in long_candidates.columns and 'MA50' in long_candidates.columns:
    #     long_candidates = long_candidates[
    #         (long_candidates['MA10'] > long_candidates['MA50']) |
    #         (long_candidates['MA10'].isna())  # Allow if MA not available
    #     ]
    
    # # Apply momentum filter for SHORT positions (downtrend)
    # if 'MA10' in short_candidates.columns and 'MA50' in short_candidates.columns:
    #     short_candidates = short_candidates[
    #         (short_candidates['MA10'] < short_candidates['MA50']) |
    #         (short_candidates['MA10'].isna())  # Allow if MA not available
    #     ]
    
    # Prefer tickers with weights (insider analysis)
    def filter_weighted_candidates(candidates):
        weighted_candidates = candidates[
            candidates['Ticker'].isin(ticker_weights.keys())
        ]
        return weighted_candidates if not weighted_candidates.empty else candidates
    
    long_candidates = filter_weighted_candidates(long_candidates)
    #short_candidates = filter_weighted_candidates(short_candidates)
    
    # Combine and sort by signal strength
    all_candidates = []
    
    # Add LONG candidates
    for _, row in long_candidates.iterrows():
        ticker = row['Ticker']
        
        # Check if we already own this stock and calculate current position size
        current_position_value = 0
        if ticker in holdings:
            current_shares = holdings[ticker]['shares']
            current_position_value = current_shares * row['Close']
        
        # Calculate current position as percentage of total portfolio value
        total_portfolio_value = cash + sum(
            holding['shares'] * daily_data[daily_data['Ticker'] == t]['Close'].iloc[0] 
            for t, holding in holdings.items() 
            if t in daily_data['Ticker'].values
        )
        
        current_position_pct = (current_position_value / total_portfolio_value) if total_portfolio_value > 0 else 0
        
        # Only add to candidates if we haven't exceeded max position size
        if current_position_pct < max_position_size_pct:
            all_candidates.append({
                'data': row,
                'position_type': 'LONG',
                'signal_strength': row['Best_Prediction'] - buy_threshold,
                'current_position_pct': current_position_pct,
                'is_new_position': ticker not in holdings
            })
    
    # Add SHORT candidates  
    # for _, row in short_candidates.iterrows():
    #     all_candidates.append({
    #         'data': row,
    #         'position_type': 'SHORT',
    #         'signal_strength': abs(row['Best_Prediction'] - sell_threshold)
    #     })
    
    # Sort by signal strength (strongest first), with preference for new positions if signal strength is similar
    all_candidates.sort(key=lambda x: (x['signal_strength'], -x['current_position_pct']), reverse=True)
    
    # Process candidates
    for candidate in all_candidates:
        if available_cash < 500 or len(holdings) >= max_positions:
            break


        # For existing positions, be more selective about max positions limit
        if not candidate['is_new_position'] or len(holdings) < max_positions:
            row = candidate['data']
            position_type = candidate['position_type']
            ticker = row['Ticker']
            price = row['Close']
            sharpe = row['Best_Prediction']
        
            # Validate price
            if price <= 0 or price > 5000:
                logger.warning(f"Skipping {ticker}: Invalid price ${price:.2f}")
                continue
        
            # Get ticker weight
            ticker_weight = ticker_weights.get(ticker, default_weight)
            
            # Calculate position size - reduce for existing positions to avoid over-concentration
            threshold = buy_threshold if position_type == 'LONG' else sell_threshold
            base_position_size = calculate_position_size(
                sharpe, ticker_weight, available_cash, threshold
            )

            # Adjust position size based on current holdings
            if ticker in holdings:
                # Reduce position size for additional purchases (e.g., 50% of normal size)
                position_size = base_position_size * 0.5
                logger.info(f"Reducing position size for existing holding {ticker}: {base_position_size:.2f} -> {position_size:.2f}")
            else:
                position_size = base_position_size
        
            # Adjust for volatility if available
            if not pd.isna(row.get('Volatility', np.nan)):
                volatility_adj = max(0.5, min(1.5, 1.0 / (row['Volatility'] + 0.1)))
                position_size *= volatility_adj
            
            # Calculate shares
            shares = int(position_size / price)
            if shares < 1:
                continue
            
            actual_cost = shares * price
            transaction_cost = actual_cost * (transaction_cost_bps / 10000)
            total_cost = actual_cost + transaction_cost
        
        # # For SHORT positions, we need collateral (typically 150% of position value)
        # if position_type == 'SHORT':
        #     collateral_required = actual_cost * 1.5
        #     if collateral_required > available_cash:
        #         # Adjust position size to fit available cash
        #         max_shares = int((available_cash / 1.5 - 50) / price)
        #         if max_shares < 1:
        #             continue
        #         shares = max_shares
        #         actual_cost = shares * price
        #         transaction_cost = actual_cost * (transaction_cost_bps / 10000)
        #         total_cost = actual_cost + transaction_cost
        #         collateral_required = actual_cost * 1.5
        
            # Final cash check
            # required_cash = collateral_required if position_type == 'SHORT' else total_cost
            required_cash = total_cost

            if required_cash > available_cash:
                continue

            # Get current holding info for order tracking
            current_shares = holdings.get(ticker, {}).get('shares', 0)
            new_total_shares = current_shares + shares

            order = {
                'date': current_date,
                'ticker': ticker,
                'action': 'buy' if position_type == 'LONG' else 'short',
                'position_type': position_type,
                'shares_amount': shares,
                'price': price,
                'investment_amount': actual_cost,
                'transaction_cost': transaction_cost,
                'total_cost': required_cash,
                'sharpe': sharpe,
                'ticker_weight': ticker_weight,
                'position_size_pct': (required_cash / cash) * 100,
                'previous_shares': current_shares,
                'new_total_shares': new_total_shares,
                'is_additional_purchase': ticker in holdings
            }
        
       
            if mode == "automatic":
                # Update or create holding info
                if ticker in holdings:
                    # Update existing position
                    old_avg_price = holdings[ticker]['purchase_price']
                    old_shares = holdings[ticker]['shares']
                    
                    # Calculate new weighted average price
                    total_old_cost = old_shares * old_avg_price
                    total_new_cost = shares * price
                    new_avg_price = (total_old_cost + total_new_cost) / (old_shares + shares)
                    
                    holdings[ticker]['shares'] = new_total_shares
                    holdings[ticker]['purchase_price'] = new_avg_price
                    holdings[ticker]['last_purchase_date'] = pd.Timestamp(current_date)
                    
                    logger.info(f"ADDING to {ticker}: Old shares={old_shares}, New shares={shares}, "
                               f"Total shares={new_total_shares}, Old avg price=${old_avg_price:.2f}, "
                               f"New avg price=${new_avg_price:.2f}")
                    
                else:
                    # Create new position
                    holdings[ticker] = {
                        'shares': shares,
                        'purchase_date': pd.Timestamp(current_date),
                        'purchase_price': price,
                        'position_type': position_type,
                        'last_purchase_date': pd.Timestamp(current_date)
                    }
                    
            # # For SHORT positions, store collateral info
            # if position_type == 'SHORT':
            #     holding_info['collateral'] = collateral_required
            
                #holdings[ticker] = holding_info
                cash_used += required_cash
                available_cash -= required_cash
            
                if orders is not None:
                    orders.append(order)
                buy_orders_count += 1

                action_word = "BUY" if position_type == 'LONG' else "SHORT"
                additional_text = " (ADDITIONAL)" if ticker in holdings else " (NEW)"
                logger.info(f"{action_word}{additional_text} {ticker}: Sharpe={sharpe:.3f}, Weight={ticker_weight:.3f}, "
                           f"Shares={shares}, Price=${price:.2f}, Cost=${required_cash:.2f}")
            

        else:
            if suggested_orders is not None:
                suggested_orders.append(order)

        """
    
    return cash_used, buy_orders_count


def run_integrated_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, mode="automatic", reset_state=False):
    """
    Execute integrated trading strategy combining Buy/Sell functions with LONG/SHORT support.
    
    Args:
        merged_data: DataFrame with predictions and price data
        investment_amount: Initial cash amount
        risk_level: Risk level 0-10 (0=conservative, 10=aggressive) 
        start_date: Start date for trading
        end_date: End date for trading
        mode: "automatic" or "semi-automatic"
        reset_state: Whether to reset portfolio state
    
    Returns:
        tuple: (orders, portfolio_history, final_value, warning_message) or 
               (suggested_orders, warning_message) for semi-automatic
    """

    logger.info(f"Starting integrated trading strategy - Risk Level: {risk_level}")
    global orders, portfolio_history
    
    try:
        # Initialize
        if reset_state:
            orders = []
            portfolio_history = []
            if os.path.exists(portfolio_state_file):
                os.remove(portfolio_state_file)
        
        load_portfolio_state()
        
        # Validate data
        required_columns = ['date', 'Ticker', 'Close', 'Best_Prediction']
        missing_columns = [col for col in required_columns if col not in merged_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Validate prediction quality using YOUR function
        correlation, buy_hit_rate, sell_hit_rate, sharpe_min, sharpe_max = validate_prediction_quality(merged_data)
        if correlation < 0.3:  # More reasonable threshold than my 0.7
            logger.warning(f"Low correlation {correlation:.3f} detected. Proceeding with caution.")
        
        # Load ticker weights and calculate thresholds
        ticker_weights = load_ticker_weights()
        default_weight = 0.02  # 2% default weight
        
        buy_threshold, sell_threshold = map_risk_to_sharpe_thresholds(risk_level, sharpe_min, sharpe_max)
        
        # Add technical indicators for momentum confirmation
        merged_data = add_technical_indicators(merged_data.copy())
        
        # Initialize portfolio
        cash = investment_amount
        holdings = {}
        suggested_orders = [] if mode == "semi-automatic" else orders
        warning_message = ""
        
        # Portfolio parameters based on risk level
        max_positions = 12 + (risk_level // 3)  # 12-15 positions - maximum number of different stocks (tickers) can be held simultaneously
        max_daily_deployment = 0.15 + (risk_level * 0.01)  # 15-25% daily deployment - how much of available cash we allow to spend on new stock purchases in a single day
        
        # Date processing
        merged_data['date'] = pd.to_datetime(merged_data['date'], utc=True)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        
        logger.info(f"Trading parameters: Max positions={max_positions}, "
                   f"Daily deployment={max_daily_deployment:.1%}")
        
        total_buy_orders = 0
        total_sell_orders = 0
        
        for current_date in date_range:
            # Get daily data
            daily_data = merged_data[
                merged_data['date'].dt.floor('D') == current_date
            ].copy()
            
            if daily_data.empty:
                # Calculate portfolio value using last known prices
                current_value = cash
                for ticker, holding in holdings.items():
                    last_price_data = merged_data[
                        (merged_data['Ticker'] == ticker) & 
                        (merged_data['date'] <= current_date)
                    ].tail(1)
                    if not last_price_data.empty:
                        current_value += holding['shares'] * last_price_data.iloc[0]['Close']
                
                portfolio_history.append({
                    'date': current_date,
                    'value': current_value,
                    'holdings': holdings.copy(),
                    'cash': cash,
                    'num_positions': len(holdings)
                })
                continue
            
            logger.debug(f"Processing {current_date.date()}: {len(daily_data)} tickers")
            
            # SELL LOGIC - Process exits first
            cash_from_sales, sell_count = process_sell_signals(
                daily_data, holdings, current_date, risk_level, 
                mode, orders if mode == "automatic" else None, suggested_orders
            )
            cash += cash_from_sales
            total_sell_orders += sell_count
            
            # BUY LOGIC - Process entries
            cash_used, buy_count = process_buy_signals(
                daily_data, buy_threshold, sell_threshold, holdings, cash,
                current_date, risk_level, ticker_weights, default_weight,
                max_positions, max_daily_deployment, mode,
                orders if mode == "automatic" else None, suggested_orders
            )
            cash -= cash_used
            total_buy_orders += buy_count
            
            # Calculate current portfolio value
            current_value = cash
            for ticker, holding in holdings.items():
                ticker_data = daily_data[daily_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    current_value += holding['shares'] * ticker_data.iloc[0]['Close']
                else:
                    # Use last known price
                    last_price_data = merged_data[
                        (merged_data['Ticker'] == ticker) & 
                        (merged_data['date'] <= current_date)
                    ].tail(1)
                    if not last_price_data.empty:
                        current_value += holding['shares'] * last_price_data.iloc[0]['Close']
            
            # Store portfolio history
            portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'holdings': holdings.copy(),
                'cash': cash,
                'num_positions': len(holdings)
            })
            
            logger.debug(f"Portfolio: Value=${current_value:.2f}, Cash=${cash:.2f}, "
                        f"Positions={len(holdings)}")
        
        # Calculate final results
        final_value = portfolio_history[-1]['value'] if portfolio_history else investment_amount
        total_return = (final_value / investment_amount - 1) * 100
        
        logger.info(f"Strategy completed:")
        logger.info(f"  Buy orders: {total_buy_orders}")
        logger.info(f"  Sell orders: {total_sell_orders}")
        logger.info(f"  Final value: ${final_value:.2f}")
        logger.info(f"  Total return: {total_return:.1f}%")
        logger.info(f"  Active positions: {len(holdings)}")
        
        # Generate warnings
        if total_buy_orders == 0:
            warning_message = f"No buy signals detected. Threshold {buy_threshold:.2f} may be too high."
        elif total_return < -20:
            warning_message = f"Large loss detected: {total_return:.1f}%. Check risk parameters."
        
        return_value = (orders, portfolio_history, current_value, warning_message) if mode == "automatic" else (suggested_orders, warning_message)
        if mode == "automatic":
            save_portfolio_state()
            return return_value
        else:
            return return_value
    
    except Exception as e:
        logger.error(f"Error in integrated trading strategy: {e}", exc_info=True)
        warning_message = f"Strategy error: {e}"
        if mode == "automatic":
            return [], [], investment_amount, warning_message
        else:
            return [], warning_message



# ===============================================================================
# PERFORMANCE ANALYSIS 
# ===============================================================================
def analyze_portfolio_performance(portfolio_history, initial_investment, orders_list=None):
    """
    Comprehensive portfolio performance analysis.
    """
    if not portfolio_history:
        return {"error": "No portfolio history available"}
    
    df = pd.DataFrame(portfolio_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Returns calculation
    df['daily_return'] = df['value'].pct_change()
    df['cumulative_return'] = (df['value'] / initial_investment) - 1
    
    # Basic metrics
    total_return = (df['value'].iloc[-1] / initial_investment) - 1
    days_total = len(df)
    annualized_return = (1 + total_return) ** (365 / days_total) - 1 if days_total > 0 else 0
    
    # Risk metrics
    daily_returns = df['daily_return'].dropna()
    volatility = daily_returns.std() * (252 ** 0.5) if len(daily_returns) > 1 else 0
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0
    
    # Ratios
    risk_free_rate = 0.02  # 2% risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Drawdown analysis
    df['peak'] = df['value'].expanding().max()
    df['drawdown'] = (df['value'] - df['peak']) / df['peak']
    max_drawdown = df['drawdown'].min()
    
    # Win rate
    positive_days = (daily_returns > 0).sum()
    total_trading_days = len(daily_returns)
    win_rate = positive_days / total_trading_days if total_trading_days > 0 else 0
    
    # Trading metrics
    if orders_list:
        buy_orders = [o for o in orders_list if o['action'] in ['buy', 'short']]
        sell_orders = [o for o in orders_list if o['action'] in ['sell', 'cover']]
        
        total_trades = len(sell_orders)
        winning_trades = len([o for o in sell_orders if o.get('profit_loss', 0) > 0])
        trade_win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        if sell_orders:
            profits = [o.get('profit_loss', 0) for o in sell_orders]
            avg_profit_per_trade = np.mean(profits)
            best_trade = max(profits)
            worst_trade = min(profits)
            
            # Separate LONG and SHORT performance
            long_trades = [o for o in sell_orders if o.get('position_type') == 'LONG']
            short_trades = [o for o in sell_orders if o.get('position_type') == 'SHORT']
            
            long_profits = [o.get('profit_loss', 0) for o in long_trades] if long_trades else [0]
            short_profits = [o.get('profit_loss', 0) for o in short_trades] if short_trades else [0]
            
            long_win_rate = len([p for p in long_profits if p > 0]) / len(long_profits) if long_profits else 0
            short_win_rate = len([p for p in short_profits if p > 0]) / len(short_profits) if short_profits else 0
        else:
            avg_profit_per_trade = best_trade = worst_trade = 0
            long_win_rate = short_win_rate = 0
            
        avg_holding_days = np.mean([o.get('days_held', 0) for o in sell_orders]) if sell_orders else 0
    else:
        total_trades = trade_win_rate = avg_profit_per_trade = 0
        best_trade = worst_trade = avg_holding_days = 0
        long_win_rate = short_win_rate = 0
    
    performance_metrics = {
        # Returns
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'annualized_return': annualized_return,
        'annualized_return_pct': annualized_return * 100,
        
        # Risk
        'volatility': volatility,
        'volatility_pct': volatility * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        
        # Ratios
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        
        # Win rates
        'daily_win_rate': win_rate,
        'daily_win_rate_pct': win_rate * 100,
        'trade_win_rate': trade_win_rate,
        'trade_win_rate_pct': trade_win_rate * 100,
        'long_win_rate_pct': long_win_rate * 100,
        'short_win_rate_pct': short_win_rate * 100,
        
        # Trading
        'total_trades': total_trades,
        'avg_profit_per_trade': avg_profit_per_trade,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'avg_holding_days': avg_holding_days,
        
        # Portfolio
        'final_value': df['value'].iloc[-1],
        'initial_value': initial_investment,
        'days_traded': len(df),
        'max_positions': df['num_positions'].max() if 'num_positions' in df.columns else 0
    }
    
    logger.info(f"Performance Summary:")
    logger.info(f"  Total Return: {performance_metrics['total_return_pct']:.1f}%")
    logger.info(f"  Annualized Return: {performance_metrics['annualized_return_pct']:.1f}%")
    logger.info(f"  Volatility: {performance_metrics['volatility_pct']:.1f}%")
    logger.info(f"  Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {performance_metrics['max_drawdown_pct']:.1f}%")
    logger.info(f"  Trade Win Rate: {performance_metrics['trade_win_rate_pct']:.1f}%")
    logger.info(f"  LONG Win Rate: {performance_metrics['long_win_rate_pct']:.1f}%")
    logger.info(f"  SHORT Win Rate: {performance_metrics['short_win_rate_pct']:.1f}%")
    
    return performance_metrics


# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================
def backtest_strategy(merged_data, investment_amount, risk_levels, start_date, end_date):
    """
    Backtest the strategy across multiple risk levels.
    
    Args:
        merged_data: DataFrame with predictions and price data
        investment_amount: Initial cash amount
        risk_levels: List of risk levels to test
        start_date: Start date for backtesting
        end_date: End date for backtesting
    
    Returns:
        dict: Results for each risk level
    """
    results = {}
    
    for risk_level in risk_levels:
        logger.info(f"Backtesting risk level {risk_level}")
        
        orders_result, portfolio_history_result, final_value, warning = run_integrated_trading_strategy(
            merged_data.copy(), investment_amount, risk_level, 
            start_date, end_date, mode="automatic", reset_state=True
        )
        
        performance = analyze_portfolio_performance(portfolio_history_result, investment_amount, orders_result)
        
        results[risk_level] = {
            'performance': performance,
            'orders': orders_result,
            'portfolio_history': portfolio_history_result,
            'final_value': final_value,
            'warning': warning
        }
    
    return results

def optimize_risk_level(merged_data, investment_amount, start_date, end_date, target_metric='sharpe_ratio'):
    """
    Find optimal risk level based on target metric.
    
    Args:
        merged_data: DataFrame with predictions and price data
        investment_amount: Initial cash amount
        start_date: Start date for optimization
        end_date: End date for optimization
        target_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
    
    Returns:
        dict: Optimal risk level and results
    """
    risk_levels = list(range(0, 11))  # 0 to 10
    results = backtest_strategy(merged_data, investment_amount, risk_levels, start_date, end_date)
    
    best_risk_level = None
    best_score = float('-inf')
    
    for risk_level, result in results.items():
        score = result['performance'].get(target_metric, float('-inf'))
        if score > best_score:
            best_score = score
            best_risk_level = risk_level
    
    logger.info(f"Optimal risk level: {best_risk_level} ({target_metric}: {best_score:.4f})")
    
    return {
        'optimal_risk_level': best_risk_level,
        'optimal_score': best_score,
        'all_results': results
    }

# ===============================================================================
# USAGE EXAMPLES AND TESTING
# ===============================================================================

def run_example_strategy():
    """
    Example usage of the integrated trading strategy.
    """
    # Example with your data
    logger.info("Running example trading strategy...")
    
    # Load your prediction data (replace with actual file path)
    try:
        merged_data = pd.read_csv('20250527_all_tickers_results.csv')
        merged_data['date'] = pd.to_datetime(merged_data['Date'])
        
        # Run strategy
        orders_result, portfolio_history_result, final_value, warning = run_integrated_trading_strategy(
            merged_data=merged_data,
            investment_amount=10000,  # $10,000 starting capital
            risk_level=5,             # Moderate risk
            start_date='2022-04-01',
            end_date='2022-09-28',
            mode="automatic",
            reset_state=True
        )
        
        # Analyze performance
        performance = analyze_portfolio_performance(portfolio_history_result, 10000, orders_result)
        
        print(f"\n{'='*50}")
        print(f"EXAMPLE STRATEGY RESULTS")
        print(f"{'='*50}")
        print(f"Initial Investment: $10,000")
        print(f"Final Value: ${performance['final_value']:,.2f}")
        print(f"Total Return: {performance['total_return_pct']:.1f}%")
        print(f"Annualized Return: {performance['annualized_return_pct']:.1f}%")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Trade Win Rate: {performance['trade_win_rate_pct']:.1f}%")
        print(f"LONG Win Rate: {performance['long_win_rate_pct']:.1f}%")
        print(f"SHORT Win Rate: {performance['short_win_rate_pct']:.1f}%")
        print(f"Warning: {warning}")
        print(f"{'='*50}")
        
        return orders_result, portfolio_history_result, performance
        
    except FileNotFoundError:
        logger.error("Data file not found. Please ensure '20250527_all_tickers_results.csv' exists.")
        return None, None, None
    except Exception as e:
        logger.error(f"Error running example: {e}")
        return None, None, None

def run_risk_level_optimization():
    """
    Example of optimizing risk levels.
    """
    logger.info("Running risk level optimization...")
    
    try:
        merged_data = pd.read_csv('20250527_all_tickers_results.csv')
        merged_data['date'] = pd.to_datetime(merged_data['Date'])
        
        # Optimize for Sharpe ratio
        optimization_result = optimize_risk_level(
            merged_data=merged_data,
            investment_amount=10000,
            start_date='2022-04-01',
            end_date='2022-09-28',
            target_metric='sharpe_ratio'
        )
        
        print(f"\n{'='*50}")
        print(f"RISK LEVEL OPTIMIZATION RESULTS")
        print(f"{'='*50}")
        print(f"Target Metric: Sharpe Ratio")
        print(f"Optimal Risk Level: {optimization_result['optimal_risk_level']}")
        print(f"Optimal Sharpe Ratio: {optimization_result['optimal_score']:.4f}")
        print(f"\nAll Risk Level Results:")
        
        for risk_level, result in optimization_result['all_results'].items():
            perf = result['performance']
            print(f"Risk {risk_level}: Return={perf['total_return_pct']:.1f}%, "
                  f"Sharpe={perf['sharpe_ratio']:.2f}, "
                  f"Drawdown={perf['max_drawdown_pct']:.1f}%")
        
        print(f"{'='*50}")
        
        return optimization_result
        
    except FileNotFoundError:
        logger.error("Data file not found. Please ensure '20250527_all_tickers_results.csv' exists.")
        return None
    except Exception as e:
        logger.error(f"Error running optimization: {e}")
        return None

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

# if __name__ == "__main__":
#     logger.info("Starting Integrated Trading Strategy System")
    
#     # Run example
#     orders_result, portfolio_history_result, performance = run_example_strategy()
    
#     if performance:
#         # Run optimization
#         optimization_result = run_risk_level_optimization()
        
#         logger.info("Trading strategy system initialization complete!")
#     else:
#         logger.error("Failed to run example strategy. Check data file and configuration.")