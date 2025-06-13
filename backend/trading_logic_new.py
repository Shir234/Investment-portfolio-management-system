# trading_logic_new.py
import os
import sys
import pandas as pd
import json
import logging

from frontend.logging_config import get_isolated_logger

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


def save_portfolio_state():
    """Save orders and portfolio history to portfolio_state.json."""
    try:
        state = {
            'orders': orders,
            'portfolio_history': portfolio_history
        }
        with open(portfolio_state_file, 'w') as f:
            json.dump(state, f, default=str)
        logger.debug(f"Portfolio state saved: {len(orders)} orders, {len(portfolio_history)} history entries")
    except Exception as e:
        logger.error(f"Error saving portfolio state: {e}")


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
    normalized_weight = 0.6 + (ticker_weight * 0.4)

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
#def sell_logic(daily_data, sell_threshold, holdings, current_date, mode="automatic", orders=None, suggested_orders=None):
def sell_logic(daily_data, sell_threshold, holdings, current_date):
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

    # cash_from_sales = 0
    # sell_orders_count = 0

    sell_orders = []
    transaction_cost_bps = 5  # 0.05% transaction cost (bps=basis point)

    # Simple rules
    MIN_HOLDING_DAYS = 5        # Hold at least 5 days
    PROFIT_TARGET = 0.15        # Sell when 15% profit
    STOP_LOSS = -0.10           # Sell when 10% loss

    for ticker, holding in list (holdings.items()):
        ticker_data = daily_data[daily_data['Ticker'] == ticker]
        if ticker_data.empty:
            continue

        current_price = ticker_data.iloc[0]['Close']
        purchase_price = holding['purchase_price']
        purchase_date = holding['purchase_date']
        shares = holding['shares']

        # Calculate days held and profit/loss
        days_held = (current_date - purchase_date).days
        profit_pct = (current_price - purchase_price) / purchase_price
        current_sharpe = ticker_data.iloc[0]['Best_Prediction']

        # Decision: Should we sell?
        should_sell = False
        sell_reason = ""
        
        if days_held >= MIN_HOLDING_DAYS:
            if profit_pct >= PROFIT_TARGET:
                should_sell = True
                sell_reason = f"Profit target hit: {profit_pct:.1%}"
            elif current_sharpe < sell_threshold:  # Same threshold you used to buy
                should_sell = True
                sell_reason = f"Signal weakened: Sharpe={current_sharpe:.3f} < {sell_threshold:.3f}"
            # elif profit_pct <= STOP_LOSS:
            #     should_sell = True
            #     sell_reason = f"Stop loss hit: {profit_pct:.1%}"
        
        if should_sell:
            # Execute the sell
            sale_value = shares * current_price
            transaction_cost = sale_value * (transaction_cost_bps / 10000)
            net_proceeds = sale_value - transaction_cost
            profit_loss = net_proceeds - (shares * purchase_price)
            
            # Create sell order
            order = {
                'date': current_date,
                'ticker': ticker,
                'action': 'sell',
                'shares_amount': shares,
                'price': current_price,
                'investment_amount': sale_value,
                'transaction_cost': transaction_cost,
                'total_proceeds': net_proceeds,
                'profit_loss': profit_loss,
                'profit_pct': profit_pct,
                'days_held': days_held,
                'sell_reason': sell_reason,
                'purchase_price': purchase_price
            }
            sell_orders.append(order)
            
    #         if mode == "automatic":
    #             cash_from_sales += net_proceeds
    #             del holdings[ticker]  # Remove from holdings
    #             if orders is not None:
    #                 orders.append(order)
    #             sell_orders_count += 1
                
    #             logger.info(f"SELL {ticker}: {sell_reason}, Days={days_held}, "
    #                        f"Profit=${profit_loss:.2f} ({profit_pct:.1%})")
    #         else:
    #             if suggested_orders is not None:
    #                 suggested_orders.append(order)
    
    # return cash_from_sales, sell_orders_count
    return sell_orders


def sell(order, holdings, cash):
    """
    Execute a sell orders, updating holdings and cash.
    """
    ticker = order['ticker']
    shares = order['shares_amount']
    price = order['price']
    net_proceeds = order['total_proceeds']

    if ticker not in holdings or holdings[ticker]['shares'] < shares:
        logger.warning(f"Skipping {ticker} sell: Insufficient shares")
        return cash, False
    
    holdings[ticker]['shares'] -= shares
    if holdings[ticker]['shares'] == 0:
        del holdings[ticker]
    
    cash += net_proceeds
    orders.append(order)
    logger.info(f"EXECUTED SELL {ticker}: {shares} shares @ ${price:.2f}")
    return cash, True

  
# def process_buy_signals(daily_data, buy_threshold, sell_threshold, holdings, cash, current_date, risk_level, ticker_weights, default_weight, max_positions, max_daily_deployment, mode="automatic", orders=None, suggested_orders=None):
#def buy_logic(daily_data, buy_threshold, holdings, cash, current_date, ticker_weights, default_weight, max_positions, mode="automatic", orders=None, suggested_orders=None, use_weights=True, use_signal_strength=True):
def buy_logic(daily_data, buy_threshold, holdings, cash, current_date, ticker_weights, default_weight, max_positions, use_weights=True, use_signal_strength=True):
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

    """
    Determine buy opportunities based on Sharpe thresholds and weights.
    """

    # cash_used = 0
    # buy_orders_count = 0
    buy_orders = []
    transaction_cost_bps = 5  # 0.05% transaction cost

    # Check if we have money and room for more positions
    if cash <= 1000 or len(holdings) >= max_positions:
        logger.info(f"No buying: Cash={cash:.2f}, Positions={len(holdings)}/{max_positions}")
        return buy_orders

    # STEP 1: Filter for buy signal (Sharpe >= threshold)
    buy_candidates = daily_data[
        daily_data['Best_Prediction'] >= buy_threshold
    ].copy()

    if buy_candidates.empty:
        logger.info(f"No buy signals: Max Sharpe={daily_data['Best_Prediction'].max():.3f} < Threshold={buy_threshold:.3f}")
        return buy_orders
    
    logger.info(f"Found {len(buy_candidates)} buy candidates above threshold {buy_threshold:.3f}")
    
    # STEP 2: Add weight and signal strength information
    if use_weights:
        buy_candidates['ticker_weight'] = buy_candidates['Ticker'].map(
            lambda x: ticker_weights.get(x, default_weight)
        )
    else:
        # Set uniform weight for all candidates
        buy_candidates['ticker_weight'] = default_weight
        logger.info("NOT using ticker weights - all candidates have equal weight")
    
    if use_signal_strength:
        # Signal strength = how much above threshold
        buy_candidates['signal_strength'] = (
            buy_candidates['Best_Prediction'] - buy_threshold
        )
    else:
        # Set uniform signal strength for all candidates
        buy_candidates['signal_strength'] = 0.0  # Neutral value
        logger.info("NOT using signal strength - all candidates have equal signal strength")

    # STEP 3: Sort by weight (highest first), then by signal strength
    sort_columns = []
    sort_ascending = []
    
    if use_weights:
        sort_columns.append('ticker_weight')
        sort_ascending.append(False)  # Highest weight first
    
    if use_signal_strength:
        sort_columns.append('signal_strength')
        sort_ascending.append(False)  # Highest signal strength first
    
    # Always add ticker for consistent sorting
    sort_columns.append('Ticker')
    sort_ascending.append(True)  # Alphabetical order
    
    # If neither weights nor signal strength are used, just sort by ticker
    if not use_weights and not use_signal_strength:
        logger.info("Sorting candidates alphabetically by ticker (no weights or signal strength)")
        buy_candidates = buy_candidates.sort_values(['Ticker'], ascending=[True])
    else:
        logger.info(f"Sorting candidates by: {sort_columns}")
        buy_candidates = buy_candidates.sort_values(sort_columns, ascending=sort_ascending)


    # STEP 4: Take only TOP HALF of candidates
    top_half_count = max(1, len(buy_candidates) // 2)
    top_candidates = buy_candidates.head(top_half_count)
    logger.info(f"Taking top {top_half_count} candidates from {len(buy_candidates)} total")
    available_positions = max_positions - len(holdings)
    candidates_to_process = min(len(top_candidates), available_positions)
    
    if candidates_to_process == 0:
        logger.info("No available positions for new stocks")
        return buy_orders
    
    logger.info(f"Will process up to {candidates_to_process} candidates")

    # STEP 5: Process candidates in weight order
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
        if use_weights or use_signal_strength:
            position_size = calculate_position_size(
                sharpe_value = sharpe,
                ticker_weight = ticker_weight, 
                available_cash = cash,
                #sharpe_threshold=buy_threshold
                signal_strength = signal_strength
            )
        else:
            # Simple equal allocation when not using weights/signal strength
            base_allocation_pct = 0.05  # 5% of available cash per position
            position_size = cash * base_allocation_pct
            logger.debug(f"Using simple allocation: {base_allocation_pct*100}% of cash = ${position_size:.2f}")

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
        buy_orders.append(order)

        """
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
    """
    
    logger.info(f"Buy logic completed: Generated {len(buy_orders)} orders")
    return buy_orders


def buy(order, holdings, cash):
    """
    Execute a buy orders, updating holdings and cash.
    """

    ticker = order['ticker']
    shares = order['shares_amount']
    price = order['price']
    total_cost = order['total_cost']
    order_date = order['date']
    if isinstance(order_date, str):
        current_date = pd.to_datetime(order_date, utc=True)
    elif isinstance(order_date, pd.Timestamp):
        current_date = order_date if order_date.tz else order_date.tz_localize('UTC')
    else:
        current_date = pd.Timestamp(order_date).tz_localize('UTC')

    if total_cost > cash:
        logger.warning(f"Skipping {ticker} buy: Insufficient cash ${cash:.2f} < ${total_cost:.2f}")
        return cash, False

    current_shares = holdings.get(ticker, {}).get('shares', 0)
    new_total_shares = current_shares + shares

    if ticker in holdings:
        old_shares = holdings[ticker]['shares']
        old_avg_price = holdings[ticker]['purchase_price']
        total_old_cost = old_shares * old_avg_price
        total_new_cost = shares * price
        new_avg_price = (total_old_cost + total_new_cost) / (old_shares + shares)
        
        holdings[ticker]['shares'] = new_total_shares
        holdings[ticker]['purchase_price'] = new_avg_price
        holdings[ticker]['last_purchase_date'] = current_date
        
        logger.info(f"ADDING to {ticker}: {old_shares}→{new_total_shares} shares, "
                   f"Avg price: ${old_avg_price:.2f}→${new_avg_price:.2f}")
    else:
        holdings[ticker] = {
            'shares': shares,
            'purchase_date': current_date,
            'purchase_price': price,
            'position_type': 'LONG',
            'last_purchase_date': current_date
        }
        
        logger.info(f"NEW BUY {ticker}: {shares} shares @ ${price:.2f}")
    
    cash -= total_cost
    orders.append(order)
    return cash, True


def execute_orders(orders_to_execute, holdings, cash, mode="automatic"):
    """
    Execute a list of orders, updating holdings and cash.
    """
    executed_count = 0
    for order in orders_to_execute:
        if mode == "semi-automatic" and order not in orders_to_execute:  # Ensure only selected orders
            continue
        if order['action'] == 'buy':
            cash, success = buy(order, holdings, cash)
            if success:
                executed_count += 1
        elif order['action'] == 'sell':
            cash, success = sell(order, holdings, cash)
            if success:
                executed_count += 1
    logger.info(f"Executed {executed_count} orders")
    return cash, executed_count


#def run_integrated_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, mode="automatic", reset_state=False):
def run_trading_strategy(merged_data, investment_amount, risk_level, start_date, end_date, mode="automatic", reset_state=False, use_weights=True, use_signal_strength=True, selected_orders=None):

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

    """
    Execute trading strategy combining Buy/Sell functions with LONG/SHORT support.
    """

    logger.info(f"Starting integrated trading strategy - Risk Level: {risk_level}")
    global orders, portfolio_history

    try:
        # Initialize the portfolio history and transactions if needed
        if reset_state:
            orders = []
            portfolio_history = []
            if os.path.exists(portfolio_state_file):
                os.remove(portfolio_state_file)
    
        # Load the portfolio state (if exists)
        load_portfolio_state()

        # Initialize portfolio variables
        cash = investment_amount
        holdings = {}
        #suggested_orders = [] if mode == "semi-automatic" else orders
        warning_message = ""

        total_buy_orders = 0
        total_sell_orders = 0

        # TODO -> the new logic for selected orders
        if selected_orders and mode == "semi-automatic":
            if selected_orders:
                order_date = selected_orders[0]['date']
                if isinstance(order_date, str):
                    current_date = pd.to_datetime(order_date, utc=True)
                elif isinstance(order_date, pd.Timestamp):
                    current_date = order_date if order_date.tz else order_date.tz_localize('UTC')
                else:
                    current_date = pd.Timestamp(order_date).tz_localize('UTC')
            else:
                current_date = pd.Timestamp.now(tz='UTC')
            # Execute only selected orders
            #current_date = pd.Timestamp(selected_orders[0]['date'], tz='UTC') if selected_orders else pd.Timestamp.now(tz='UTC')
            cash, executed_count = execute_orders(selected_orders, holdings, cash, mode="semi-automatic")
            total_buy_orders = sum(1 for o in selected_orders if o['action'] == 'buy' and o in selected_orders)
            total_sell_orders = sum(1 for o in selected_orders if o['action'] == 'sell' and o in selected_orders)

            current_value = cash
            for ticker, holding in holdings.items():
                # Find the current price for this ticker from selected orders, or use purchase price
                ticker_price = holding['purchase_price']  # fallback
                for order in selected_orders:
                    if order['ticker'] == ticker:
                        ticker_price = order['price']
                        break
                current_value += holding['shares'] * ticker_price

            portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'holdings': holdings.copy(),
                'cash': cash,
                'num_positions': len(holdings)
            })

            save_portfolio_state()
            return selected_orders, warning_message

        # Standard trading logic
        # Date processing
        merged_data['date'] = pd.to_datetime(merged_data['date'], utc=True)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')

        # Load ticker weights and calculate thresholds
        if use_weights:
            ticker_weights = load_ticker_weights()
        else:
            ticker_weights = {}
        default_weight = 0.02  # 2% default weight

        sharpe_min = merged_data['Best_Prediction'].min()
        sharpe_max = merged_data['Best_Prediction'].max()
        # Get the risk threshold as a sharpe ratio value - to match the predictions scale
        buy_threshold, sell_threshold = map_risk_to_sharpe_thresholds(risk_level, sharpe_min, sharpe_max)
        # Portfolio parameters
        max_positions = 70  # Maximum number of different stocks

        # Get Buy / Sell transaction for each dat - based on trading logic 
        for current_date in date_range:
            # Get daily data
            daily_data = merged_data[
                merged_data['date'].dt.floor('D') == current_date
            ].copy()

            if daily_data.empty:
                # No data today - just record portfolio value
                current_value = cash
                for ticker, holding in holdings.items():
                    # Use last known price or skip
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

            # STEP 1: SELL FIRST (to free up cash for buying)
            sell_orders = sell_logic(
                daily_data = daily_data,
                sell_threshold = sell_threshold,
                holdings = holdings,
                current_date = current_date
                # mode = mode,
                # orders = orders if mode == "automatic" else None,
                # suggested_orders = suggested_orders
            )
            cash, sell_count = execute_orders(sell_orders, holdings, cash, mode=mode)
            total_sell_orders += sell_count

            # STEP 2: BUY (with updated cash amount)
            buy_orders = buy_logic(
                daily_data = daily_data,
                buy_threshold = buy_threshold,
                holdings=holdings,
                cash = cash,
                current_date = current_date,
                ticker_weights = ticker_weights,
                default_weight = default_weight,
                max_positions = max_positions,
                use_weights=use_weights,
                use_signal_strength=use_signal_strength
            )
            cash, buy_count = execute_orders(buy_orders, holdings, cash, mode=mode)
            total_buy_orders += buy_count

            # STEP 3: Calculate current portfolio value
            current_value = cash
            for ticker, holding in holdings.items():
                ticker_data = daily_data[daily_data['Ticker'] == ticker]
                if not ticker_data.empty:
                    current_value += holding['shares'] * ticker_data.iloc[0]['Close']

            # STEP 4: Record daily portfolio snapshot
            portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'holdings': holdings.copy(),
                'cash': cash,
                'num_positions': len(holdings)
            })

            logger.debug(f"Day summary: Sells={sell_count}, Buys={buy_count}, "
                        f"Cash=${cash:.0f}, Value=${current_value:.0f}, Positions={len(holdings)}")

        # After the loop - calculate final results
        final_value = portfolio_history[-1]['value'] if portfolio_history else investment_amount
        total_return = (final_value / investment_amount - 1) * 100

        logger.info(f"Strategy completed: Buy orders={total_buy_orders}, Sell orders={total_sell_orders}")
        logger.info(f"Final value: ${final_value:.2f} ({total_return:.1f}% return)")

        save_portfolio_state()

        # Return results based on mode
        if mode == "automatic":
            return orders, portfolio_history, final_value, warning_message
        else:
            return orders, warning_message

    except Exception as e:
        logger.error(f"Error in integrated trading strategy: {e}", exc_info=True)
        warning_message = f"Strategy error: {e}"
        save_portfolio_state()
        if mode == "automatic":
            return [], [], investment_amount, warning_message
        else:
            return [], warning_message


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