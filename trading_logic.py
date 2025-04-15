"""
1 merge the date
2 get input from the user:
    risk threshold in pct (0-100)
    investment amount - how mush money we have to invest
    trading window - investment period --> convert it to number of days
3 convert the risk pct to sharpe threshold
4 start trading algorithm
"""

def map_risk_threshold_to_sharpe(threshold):
    if threshold >= 80:
        min_acceptable_sharpe = 0.2
    elif threshold >= 60:
        min_acceptable_sharpe = 0.5
    elif threshold >= 40:
        min_acceptable_sharpe = 1
    elif threshold >= 20:
        min_acceptable_sharpe = 1.5
    else: 
        min_acceptable_sharpe = 2

    return min_acceptable_sharpe


def close_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, available_capital):
    """
    Close transactions based on threshold
    - sell shares 
    - update available capital
    - return updated current_positions
    """
    positions_to_close = []

    for ticker, position in current_positions.items():
        # Get today's prediction for this ticker
        ticker_today = today_data[today_data['ticker'] == ticker]
        if ticker_today.empty:
            continue # Skip

        current_sharpe = ticker_today['Best_Prediction'].values[0]
        position_type = position['position_type']

        # Evaluate if position should be closed
        should_close = False

        # Close if Sharpe drops below threshold
        if position_type == 'LONG' and current_sharpe < min_acceptable_sharpe:
            should_close = True
        elif position_type == 'SHORT' and current_sharpe > -min_acceptable_sharpe:
            should_close = True
        
        if should_close:
            positions_to_close.append(ticker)
            close_price = ticker_today['Close'].values[0]
            # Add to orders
            orders.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'SELL' if position_type == 'LONG' else 'COVER',
                'shares_amount': position['shares_amount'],
                'price': close_price,
                'reason': 'Below Sharpe threshold'
            })

    # Make the transaction: remove closed positions from current_positions
    for ticker in positions_to_close:
        # Get ticker data and price
        ticker_today = today_data[today_data['ticker'] == ticker]
        close_price = ticker_today['Close'].values(0)
        # Calculate current value + profit / loss
        current_value = current_positions[ticker]['shares_amount'] * close_price
        profit_loss = current_value - current_positions[ticker]['investment_amount']
        # Update available capital and remove from portfolio
        available_capital += current_value
        del current_positions[ticker]
    
    return current_positions, orders, available_capital


def open_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, daily_target, transaction_cost_pct, available_capital):
    """
    Open transactions based on threshold
    - buy shares 
    - update available capital
    - return updated current_positions
    """

    # Filter for acceptable trades today
    potential_trades = today_data[abs(today_data['Best_Prediction'] >= min_acceptable_sharpe)]

    if not potential_trades.empty:
        # Calculate total potential sharpe strength 
        total_sharpe_strength = sum(abs(potential_trades['Best_Prediction']))

        # Distribute daily target based on sharpe strength 
        for _, row in potential_trades.iterrows():
            ticker = row['ticker']
            current_sharpe = row['Best_Prediction']
            position_type = "LONG" if current_sharpe > 0 else "SHORT"

            # TODO - After results, maybe try different allocation ? 
            # Calculate allocation based on Sharpe strength - how much money to invest
            allocation = (abs(current_sharpe) / total_sharpe_strength) * daily_target
            # Adjust for transaction costs - include tax
            allocation_after_costs = allocation * (1 - transaction_cost_pct)

            # Check if we have enough capital
            if allocation_after_costs <=0 or allocation_after_costs > available_capital:
                continue

            entry_price = row['Close'] 
            # Calculate number of shares to buy/short
            shares_to_buy = allocation_after_costs / entry_price

            # Check if the position exists
            if ticker in current_positions and current_positions[ticker]['position_type'] == position_type:
                # Get the existing data
                existing_shares = current_positions[ticker]['shares_amount']
                existing_investment = current_positions[ticker]['investment_amount']

                # Calculate new position details
                new_shares_total = existing_shares + shares_to_buy
                new_investment_total = existing_investment + allocation_after_costs

                # Calculate average entry price
                old_entry_price = existing_investment / existing_shares if existing_shares > 0 else 0
                avg_entry_price = ((old_entry_price * existing_shares) + (entry_price * shares_to_buy)) / new_shares_total

                # Add to orders
                orders.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'ADD_TO_' + position_type,  # Special action to indicate adding to position
                    'shares_amount': shares_to_buy,
                    'price': entry_price,
                    'investment_amount': allocation_after_costs,
                    'previous_shares': existing_shares,
                    'new_total_shares': new_shares_total,
                    'sharpe': current_sharpe
                })

                # Update the existing position
                current_positions[ticker]['shares_amount'] = new_shares_total
                current_positions[ticker]['investment_amount'] = new_investment_total

            else:
                # Create new position
                orders.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'BUY' if position_type == 'LONG' else 'SHORT',
                    'shares_amount': shares_to_buy,
                    'price': entry_price,
                    'investment_amount': allocation_after_costs,
                    'sharpe': current_sharpe
                })

                # Add to current positions
                current_positions[ticker] = {
                    'entry_date': current_date,
                    'shares_amount': shares_to_buy,
                    'investment_amount': allocation_after_costs,
                    'position_type': position_type
                }

            # Update available capital
            available_capital -= allocation_after_costs

    return current_positions, orders, available_capital


def daily_trading_algorithm(merged_data, min_acceptable_sharpe, investment_amount,
                            investment_period_days, current_date=None,
                            current_positions=None, transaction_cost_pct=0.001):
    """
    Daily trading algorithm that decides which positions to enter or exit.
    Parameters:
    - merged_data: DataFrame with predictions for all tickers
    - min_acceptable_sharpe: Minimum acceptable Sharpe ratio
    - investment_amount: Total amount to invest
    - investment_period_days: Period over which to invest all money
    - current_date: Current trading date (defaults to earliest date in data)
    - current_positions: Dictionary of current positions {ticker: {amount, entry_date, position_type, entry_price}}
    - transaction_cost_pct: Cost of transaction as percentage (Trading fee percentage)
    """

    # Initialize variables
    if current_positions is None:
        current_positions = {}
    # Set current date if not provided
    if current_date is None:
        current_date = merged_data['Date'].min()
    # Get today's data
    today_data = merged_data[merged_data['Date'] == current_date]

    # Calculate available capital 
    """
    {ticker}
    entry date -> day of opening position
    shares_amount -> number of shares i own 
    investment_amount -> amount of money invested in this stock
    position_type -> LONG / SHORT
    """
    invested_capital = sum(position['investment_amount'] for position in current_positions.values())
    available_capital = investment_amount - invested_capital

    # Initialize orders list
    orders = []
    
    # Part 1: find positions to close (according to portfolio threshold)
    current_positions, orders, available_capital = close_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, available_capital)

    # Calculate daily investment target (how much to invest today)
    days_left = investment_period_days - len(set(merged_data[merged_data['Date'] < current_date]['Date']))
    days_left = max(1, days_left) # Avoid division by zero
    daily_target = available_capital / days_left

    # Part 2: find new opportunities (according to portfolio threshold)
    current_positions, orders, available_capital = open_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, daily_target, transaction_cost_pct, available_capital)

    # TODO - not sure, maybe implement later
    # PART 3: REBALANCE PORTFOLIO (optional)
    # This section would reallocate capital from lower Sharpe positions to higher ones

    return orders, current_positions, available_capital

def run_portfolio_simulation(merged_data, min_acceptable_sharpe, investment_amount, 
                            investment_period_days, start_date=None, end_date=None,
                            transaction_cost_pct=0.001):
    """
    Run a full portfolio simulation over a date range.
    
    Parameters:
    - merged_data: DataFrame with predictions for all tickers
    - min_acceptable_sharpe: Minimum acceptable Sharpe ratio
    - investment_amount: Total amount to invest
    - investment_period_days: Period over which to invest all money
    - start_date: Start date for simulation (default: earliest date in data)
    - end_date: End date for simulation (default: latest date in data)
    - transaction_cost_pct: Cost of transaction as percentage
    
    Returns:
    - all_orders: List of all orders executed
    - portfolio_value_history: DataFrame with daily portfolio value
    - final_positions: Dictionary of positions at end of simulation
    """