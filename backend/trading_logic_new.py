# trading_logic_new.py
import os
import pandas as pd
import json

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
    - risk_level: 0-10 (0=conservative, 10=aggressive)
    - sharpe_min, sharpe_max: sharpe ratio limits in data
    
    Returns:
    - tuple: (buy_threshold, sell_threshold)
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
    """
    Load and normalize ticker weights from a CSV file.
    Preserve relative ordering - Highest score still becomes highest normalized value 
    """

    try:
        if not os.path.exists(weights_file):
            logger.warning(f"Weights file {weights_file} not found. Using default weight of 1.0.")
            return {}
        
        df = pd.read_csv(weights_file)
        weight_col = 'Transaction_Score'

        if 'Ticker' not in df.columns or weight_col not in df.columns:
            logger.error(f"Invalid weights file format.")
            return {}
        
        # Clean but KEEP negative values
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        df = df[df[weight_col].notnull()]

        # Normalize to -1 to +1 range (preserves meaning)
        min_score = df[weight_col].min()
        max_score = df[weight_col].max()
        df['Trading_Signal'] = 2 * (df[weight_col] - min_score) / (max_score - min_score) - 1
        
        weights = dict(zip(df['Ticker'], df['Trading_Signal']))

        return weights
    
    except Exception as e:
        logger.error(f"Error loading ticker weights: {e}")
        return {}


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


def calculate_position_size(sharpe_value, ticker_weight, available_cash, signal_strength, max_position_pct=0.20):
    """
    Calculate position size based on signal strength and ticker weight.
    Position size = amount of money to invest in specific stock.

    Args:
    - sharpe_value: Predicted Sharpe ratio for this stock
    - ticker_weight: Weight from scoring system (-1 to +1 range)
    - available_cash: Total cash available for trading
    - signal_strength: How much above threshold (calculated in buy_logic)
    - max_position_pct: Maximum percentage of cash for one position (default 20%)
    
    Returns:
    - position_size: Dollar amount to invest in this stock
    """

    # 1 Normalize ticker weight to usable range
    # Convert from (-1, +1) to (0.2, 1.0) so even negative weights get some allocation
    if ticker_weight >= 0:
        # Positive weights: 0.6 to 1.0
        normalized_weight = 0.6 + (ticker_weight * 0.4)
    else:
        # Negative weights: 0.2 to 0.6  
        normalized_weight = 0.6 + (ticker_weight * 0.4)  # ticker_weight is negative
    
    # Ensure weight is in valid range
    normalized_weight = max(0.2, min(normalized_weight, 1.0))

    # 2 Combine factors to get allocation percentage
    # Base allocation: 3% to 12% of available cash
    # base_pct = 0.03  # 3% minimum
    # variable_pct = 0.09  # Up to 9% additional (total max 12%)
    
    # Combined score
    combined_score = signal_strength * normalized_weight
    # allocation_pct = base_pct + (combined_score * variable_pct)
    
    # # 4. Apply maximum position limit
    # allocation_pct = min(allocation_pct, max_position_pct)
    
    # 5. Calculate dollar amount
    position_size = available_cash * combined_score
    
    # # 6. Apply minimum position size
    # min_position = 300  # Minimum $300 position
    # position_size = max(position_size, min_position)
    
    # 7. Don't exceed available cash (safety check)
    position_size = min(position_size, available_cash * 0.95)  # Leave 5% buffer
    
    return position_size 


# def process_sell_signals(daily_data, holdings, current_date, risk_level, mode="automatic", orders=None, suggested_orders=None):
def sell_logic(daily_data, holdings, current_date, mode="automatic", orders=None, suggested_orders=None):
    """
    Process sell signals for existing LONG and SHORT positions.
    
    Args:
    - daily_data: DataFrame with current day's data
    - holdings: Dict of current holdings
    - current_date: Current trading date
    - mode: "automatic" or "semi-automatic"
    - orders: List to append executed orders (for automatic mode)
    - suggested_orders: List to append suggested orders (for semi-automatic mode)
    
    Returns:
        tuple: (updated_cash, sell_orders_count)
    """

    """
    TODO : WHEN DO WE SELL ? 
        1 when the trading period is over? 
        2 when we got above 20% profit from the buying 
        3 when the predicted sharpe is under the threshold sharpe- more then 15 days ? 
        4 stop loss -> sell when we lost too much

    """

    cash_from_sales = 0
    sell_orders_count = 0
    transaction_cost_bps = 5  # 0.05% transaction cost (bps=basis point)

    for ticker, holding in list (holding.items()):
        ticker_data = daily_data[daily_data['Ticker'] == ticker]
        if ticker_data.empty:
            continue
    
    
# def process_buy_signals(daily_data, buy_threshold, sell_threshold, holdings, cash, current_date, risk_level, ticker_weights, default_weight, max_positions, max_daily_deployment, mode="automatic", orders=None, suggested_orders=None):
def buy_logic(daily_data, buy_threshold, holdings, cash, current_date, ticker_weights, default_weight, max_positions, mode="automatic", orders=None, suggested_orders=None):
    """
    Process buy signals - buy when predicted sharpe > threshold.
    Only buy top half of signals using signal strength and weights.
    
    Args:
    - daily_data: DataFrame with current day's data
    - buy_threshold: Threshold for LONG positions
    - holdings: Dict of current holdings
    - cash: Available cash
    - current_date: Current trading date
    - ticker_weights: Dict of ticker weights
    - default_weight: Default weight for tickers not in weights
    - max_positions: Maximum number of positions
    - mode: "automatic" or "semi-automatic"
    - orders: List to append executed orders (for automatic mode)
    - suggested_orders: List to append suggested orders (for semi-automatic mode)
    
    Returns:
    - tuple: (cash_used, buy_orders_count)

    TODO: when do we buy?
    1 find buying options according to the sharpe threshold (risk) 
    - only buy the upper half options using the weights 

    """

    cash_used = 0
    buy_orders_count = 0
    transaction_cost_bps = 5  # 0.05% transaction cost

    # Check if we have money and room for more positions
    # TODO : Check we have money to do buying 
    if cash <= 1000 or len(holdings) >= max_positions:
        logger.info(f"No buying: Cash={cash:.2f}, Positions={len(holdings)}/{max_positions}")
        return cash_used, buy_orders_count

    # STEP 1: Filter for buy signal (Sharpe >= threshold)
    buy_candidates = daily_data[
        daily_data['Best_Prediction'] >= buy_threshold
    ].copy()

    if buy_candidates.empty:
        logger.info(f"No buy signals: Max Sharpe={daily_data['Best_Prediction'].max():.3f} < Threshold={buy_threshold:.3f}")
        return cash_used, buy_orders_count
    
    logger.info(f"Found {len(buy_candidates)} buy candidates above threshold {buy_threshold:.3f}")
    
    # STEP 2: Add weight and signal strength information
    buy_candidates['ticker_weight'] = buy_candidates['Ticker'].map(
        lambda x: ticker_weights.get(x, default_weight)
    )
    
    # Signal strength = how much above threshold
    buy_candidates['signal_strength'] = (
        buy_candidates['Best_Prediction'] - buy_threshold
    )

    # STEP 3: Sort by weight (highest first), then by signal strength
    buy_candidates = buy_candidates.sort_values([
        'ticker_weight', 'signal_strength', 'Ticker'
    ], ascending=[False, False, True])

    # STEP 4: Take only TOP HALF of candidates
    top_half_count = max(1, len(buy_candidates) // 2)
    top_candidates = buy_candidates.head(top_half_count)

    logger.info(f"Taking top {top_half_count} candidates from {len(buy_candidates)} total")
    
    # STEP 5: Process each candidate 
    available_positions = max_positions - len(holdings)
    candidates_to_process = min(len(top_candidates), available_positions)
    
    if candidates_to_process == 0:
        logger.info("No available positions for new stocks")
        return cash_used, buy_orders_count
    
    logger.info(f"Will process up to {candidates_to_process} candidates")

    
    # STEP 5: Apply momentum filter if available

    # STEP 6: Process candidates in weight order
    for idx, row in top_candidates.head(candidates_to_process).iterrows():
        ticker = row['Ticker']
        price = row['Close']
        sharpe = row['Best_Prediction']
        ticker_weight = row['ticker_weight']
        signal_strength = row['signal_strength']

        # Validate price
        if price <= 0 or price > 5000:
            logger.warning(f"Skipping {ticker}: Invalid price ${price:.2f}")
            continue

        # Calculate position size using your function
        position_size = calculate_position_size(
            sharpe_value=sharpe,
            ticker_weight=ticker_weight, 
            available_cash=cash,
            sharpe_threshold=buy_threshold
        )

        # if position_size < 200:  # Minimum position size
        #     logger.info(f"Skipping {ticker}: Position too small ${position_size:.2f}")
        #     continue

        # Calculate shares
        shares = int(position_size / price)
        if shares < 1:
            logger.info(f"Skipping {ticker}: Can't afford 1 share at ${price:.2f}")
            continue

        # Calculate actual costs
        actual_cost = shares * price
        transaction_cost = actual_cost * (transaction_cost_bps / 10000)
        total_cost = actual_cost + transaction_cost
        
        # Final cash check
        if total_cost > cash:
            logger.info(f"Skipping {ticker}: Total cost ${total_cost:.2f} > Available cash ${cash:.2f}")
            continue

        # Get current holding info (for tracking)
        current_shares = holdings.get(ticker, {}).get('shares', 0)
        new_total_shares = current_shares + shares

        # Create order record
        order = {
            'date': current_date,
            'ticker': ticker,
            'action': 'buy',
            'shares_amount': shares,
            'price': price,
            'investment_amount': actual_cost,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost,
            'sharpe': sharpe,
            'ticker_weight': ticker_weight,
            'signal_strength': signal_strength,
            'previous_shares': current_shares,
            'new_total_shares': new_total_shares,
            'is_additional_purchase': ticker in holdings
        }

        if mode == "automatic":
            # Update holdings
            if ticker in holdings:
                # EXISTING STOCK - Update weighted average price
                old_shares = holdings[ticker]['shares']
                old_avg_price = holdings[ticker]['purchase_price']
                
                # Calculate new weighted average price
                total_old_cost = old_shares * old_avg_price
                total_new_cost = shares * price
                new_avg_price = (total_old_cost + total_new_cost) / (old_shares + shares)
                
                # Update holdings
                holdings[ticker]['shares'] = new_total_shares
                holdings[ticker]['purchase_price'] = new_avg_price
                holdings[ticker]['last_purchase_date'] = pd.Timestamp(current_date)
                
                logger.info(f"ADDING to {ticker}: {old_shares}→{new_total_shares} shares, "
                           f"Avg price: ${old_avg_price:.2f}→${new_avg_price:.2f}")
            else:
                # NEW STOCK - Create new position
                holdings[ticker] = {
                    'shares': shares,
                    'purchase_date': pd.Timestamp(current_date),
                    'purchase_price': price,
                    'position_type': 'LONG',
                    'last_purchase_date': pd.Timestamp(current_date)
                }
                
                logger.info(f"NEW BUY {ticker}: Weight={ticker_weight:.3f}, "
                           f"Sharpe={sharpe:.3f}, {shares} shares @ ${price:.2f}")
            
            # Update cash and add order
            cash_used += total_cost
            cash -= total_cost
            
            if orders is not None:
                orders.append(order)
            buy_orders_count += 1
            
        else:  # semi-automatic mode
            if suggested_orders is not None:
                suggested_orders.append(order)
                logger.info(f"SUGGEST BUY {ticker}: {shares} shares @ ${price:.2f}")
    
    logger.info(f"Buy logic completed: Used ${cash_used:.2f}, {buy_orders_count} orders")
    return cash_used, buy_orders_count


#def run_integrated_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, mode="automatic", reset_state=False):
def execute_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, mode="automatic", reset_state=False):
    """
    Execute trading strategy combining Buy/Sell functions with LONG/SHORT support.
    
    Args:
    - merged_data: DataFrame with predictions and price data
    - investment_amount: Initial cash amount
    - risk_level: Risk level 0-10 (0=conservative, 10=aggressive) 
    - start_date: Start date for trading
    - end_date: End date for trading
    - mode: "automatic" or "semi-automatic"
    - reset_state: Whether to reset portfolio state
        
    Returns:
    - tuple: (orders, portfolio_history, final_value, warning_message) or 
                (suggested_orders, warning_message) for semi-automatic

    orders - Transaction History : a list of dictionaries where each order represents a single transaction (buy/sell)
    portfolio_history - Daily Portfolio Snapshots:  tracks the portfolio value over time
    holdings - Current Stock Positions: tracks what we currently own
    
    """

    logger.info(f"Starting integrated trading strategy - Risk Level: {risk_level}")
    global orders, portfolio_history

    try:
        # Initialize the portfolio history and transactions if needed
        if reset_state:
            # TODO : what exactly are orders and portfolio_history (structure and all)
            orders = []
            portfolio_history = []
            if os.path.exists(portfolio_state_file):
                os.remove(portfolio_state_file)
    
        # Load the portfolio state (if exists)
        load_portfolio_state()

        # Initialize portfolio variables
        cash = investment_amount
        # TODO : what exactly are holdings
        holdings = {}
        suggested_orders = [] if mode == "semi-automatic" else orders
        warning_message = ""

        total_buy_orders = 0
        total_sell_orders = 0

        # Date processing
        merged_data['date'] = pd.to_datetime(merged_data['date'], utc=True)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        

        # Validate data

        # Validate prediction quality

        # Load ticker weights and calculate thresholds
        ticker_weights = load_ticker_weights()
        default_weight = 0.02  # 2% default weight

        sharpe_min = merged_data['Best_Prediction'].min()
        sharpe_max = merged_data['Best_Prediction'].max()
        
        # Get the risk threshold as a sharpe ratio value - to match the predictions scale
        buy_threshold, sell_threshold = map_risk_to_sharpe_thresholds(risk_level, sharpe_min, sharpe_max)

        # Add technical indicators for momentum confirmation

        # Portfolio parameters based on risk level
        
        # Get Buy / Sell transaction for each dat - based on trading logic 
        for current_date in date_range:
            # Get daily data
            daily_data = merged_data[
                merged_data['date'].dt.floor('D') == current_date
            ].copy()

            # TODO: ADD SOMETHING IF THE DAILY DATA IS EMPTY

            logger.debug(f"Processing {current_date.date()}: {len(daily_data)} tickers")

            # First - Sell, so the money from the sell transaction could be used in the Buy later






    
        
    except Exception as e:
        logger.error(f"Error in integrated trading strategy: {e}", exc_info=True)
        warning_message = f"Strategy error: {e}"
        if mode == "automatic":
            return [], [], investment_amount, warning_message
        else:
            return [], warning_message




