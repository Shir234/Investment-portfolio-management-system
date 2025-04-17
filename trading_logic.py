import pandas as pd
from datetime import datetime, timedelta

# Define global orders list at module level
orders = []
portfolio_history = []

def map_risk_threshold_to_sharpe(threshold):
    """Map risk threshold (0-100) to minimum acceptable Sharpe ratio."""
    threshold = max(0, min(100, threshold))  # Clamp to 0-100
    if threshold >= 80:
        return 0.5  # Aggressive: lower Sharpe threshold
    elif threshold >= 60:
        return 1.0
    elif threshold >= 40:
        return 1.5
    elif threshold >= 20:
        return 2.0
    else:
        return 3.0  # Conservative: high Sharpe threshold

def close_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, available_capital, risk_level):
    """Close positions where Sharpe ratio falls below threshold or stop-loss/take-profit is triggered."""
    positions_to_close = []
    
    for ticker, position in list(current_positions.items()):
        ticker_data = today_data[today_data['Ticker'] == ticker]
        if ticker_data.empty:
            continue

        current_sharpe = ticker_data['Best_Prediction'].values[0]
        position_type = position['position_type']
        entry_price = position['entry_price']
        current_price = ticker_data['Close'].values[0]
        shares = position['shares_amount']

        # Calculate current profit/loss percentage
        if position_type == 'LONG':
            profit_loss_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            profit_loss_pct = (entry_price - current_price) / entry_price

        # Define stop-loss and take-profit based on risk level
        stop_loss_pct = -0.05 if risk_level < 40 else -0.10  # 5% loss for conservative
        take_profit_pct = 0.10 if risk_level < 40 else 0.20  # 10% gain for conservative

        # Close if Sharpe is below threshold or stop-loss/take-profit is triggered
        should_close = False
        if position_type == 'LONG' and (current_sharpe < min_acceptable_sharpe or profit_loss_pct <= stop_loss_pct or profit_loss_pct >= take_profit_pct):
            should_close = True
        elif position_type == 'SHORT' and (current_sharpe > -min_acceptable_sharpe or profit_loss_pct <= stop_loss_pct or profit_loss_pct >= take_profit_pct):
            should_close = True

        if should_close:
            positions_to_close.append(ticker)
            proceeds = shares * current_price
            available_capital += proceeds

            orders.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'SELL' if position_type == 'LONG' else 'COVER',
                'shares_amount': shares,
                'price': current_price,
                'investment_amount': proceeds,
                'previous_shares': shares,
                'new_total_shares': 0,
                'sharpe': current_sharpe
            })

    for ticker in positions_to_close:
        del current_positions[ticker]
    
    return current_positions, orders, available_capital

def open_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, daily_target, transaction_cost_pct, available_capital, risk_level):
    """Open new positions based on Sharpe ratio and risk-adjusted allocation."""
    potential_trades = today_data[abs(today_data['Best_Prediction']) >= min_acceptable_sharpe]
    if potential_trades.empty:
        return current_positions, orders, available_capital

    investment_fraction = min(risk_level / 100.0, 1.0)  # Scale 0-100 to 0-1
    adjusted_daily_target = daily_target * investment_fraction

    max_stocks = 3 if risk_level < 40 else 5
    potential_trades = potential_trades.sort_values(by='Best_Prediction', ascending=False).head(max_stocks)
    total_sharpe_strength = sum(abs(potential_trades['Best_Prediction']))
    if total_sharpe_strength == 0:
        return current_positions, orders, available_capital

    for _, row in potential_trades.iterrows():
        ticker = row['Ticker']
        current_sharpe = row['Best_Prediction']
        position_type = 'LONG' if current_sharpe > 0 else 'SHORT'
        entry_price = row['Close']

        allocation = (abs(current_sharpe) / total_sharpe_strength) * adjusted_daily_target
        allocation_after_costs = allocation * (1 - transaction_cost_pct)

        if allocation_after_costs <= 0 or allocation_after_costs > available_capital:
            continue

        shares_to_buy = int(allocation_after_costs / entry_price)
        if shares_to_buy < 1:
            continue

        investment = shares_to_buy * entry_price
        transaction_cost = investment * transaction_cost_pct
        total_cost = investment + transaction_cost

        if total_cost > available_capital:
            continue

        available_capital -= total_cost

        if ticker in current_positions and current_positions[ticker]['position_type'] == position_type:
            existing_position = current_positions[ticker]
            previous_shares = existing_position['shares_amount']
            existing_investment = existing_position['investment_amount']
            old_entry_price = existing_position['entry_price']

            new_shares_total = previous_shares + shares_to_buy
            new_investment_total = existing_investment + total_cost
            avg_entry_price = ((old_entry_price * previous_shares) + (entry_price * shares_to_buy)) / new_shares_total

            orders.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'ADD_TO_' + position_type,
                'shares_amount': shares_to_buy,
                'price': entry_price,
                'investment_amount': total_cost,
                'previous_shares': previous_shares,
                'new_total_shares': new_shares_total,
                'sharpe': current_sharpe
            })

            current_positions[ticker].update({
                'shares_amount': new_shares_total,
                'investment_amount': new_investment_total,
                'entry_price': avg_entry_price
            })
        else:
            orders.append({
                'date': current_date,
                'ticker': ticker,
                'action': 'BUY' if position_type == 'LONG' else 'SHORT',
                'shares_amount': shares_to_buy,
                'price': entry_price,
                'investment_amount': total_cost,
                'previous_shares': 0,
                'new_total_shares': shares_to_buy,
                'sharpe': current_sharpe
            })

            current_positions[ticker] = {
                'entry_date': current_date,
                'shares_amount': shares_to_buy,
                'investment_amount': total_cost,
                'position_type': position_type,
                'entry_price': entry_price
            }

    return current_positions, orders, available_capital

def daily_trading_algorithm(merged_data, min_acceptable_sharpe, investment_amount, investment_period_days, current_date=None, current_positions=None, transaction_cost_pct=0.001, risk_level=50):
    """Execute daily trading decisions."""
    if current_positions is None:
        current_positions = {}
    if current_date is None:
        current_date = merged_data['date'].min()

    today_data = merged_data[merged_data['date'].dt.date == current_date.date()]
    print(f"Date: {current_date}, Data available: {len(today_data)} rows")

    invested_capital = sum(position['investment_amount'] for position in current_positions.values())
    available_capital = investment_amount - invested_capital

    orders = []
    current_positions, orders, available_capital = close_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, available_capital, risk_level)
    print(f"After closing positions: {len(orders)} orders, Available capital: {available_capital}")

    trading_days_left = len(pd.bdate_range(current_date, merged_data['date'].max()))
    daily_target = available_capital / max(1, trading_days_left)
    print(f"Daily target: {daily_target}, Days left: {trading_days_left}")

    current_positions, orders, available_capital = open_positions(today_data, min_acceptable_sharpe, current_date, current_positions, orders, daily_target, transaction_cost_pct, available_capital, risk_level)
    print(f"After opening positions: {len(orders)} orders, Final capital: {available_capital}")

    return orders, current_positions, available_capital

def run_portfolio_simulation(merged_data, min_acceptable_sharpe, investment_amount, investment_period_days, start_date, end_date, risk_level):
    """Run portfolio simulation over the specified date range."""
    current_positions = {}
    portfolio_value_history = []
    all_orders = []

    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only

    for current_date in trading_days:
        orders, current_positions, available_capital = daily_trading_algorithm(
            merged_data=merged_data,
            min_acceptable_sharpe=min_acceptable_sharpe,
            investment_amount=investment_amount,
            investment_period_days=investment_period_days,
            current_date=current_date,
            current_positions=current_positions,
            transaction_cost_pct=0.001,
            risk_level=risk_level
        )
        all_orders.extend(orders)

        portfolio_value = available_capital
        for ticker, position in current_positions.items():
            stock_data = merged_data[(merged_data['Ticker'] == ticker) & (merged_data['date'].dt.date <= current_date.date())]
            if not stock_data.empty:
                latest_price = stock_data['Close'].iloc[-1]
                portfolio_value += position['shares_amount'] * latest_price

        portfolio_value_history.append({
            'date': current_date,
            'portfolio_value': portfolio_value
        })

    return all_orders, pd.DataFrame(portfolio_value_history), current_positions

def run_trading_strategy(investment_amount, risk_level, start_date, end_date, merged_data, data_manager):
    global orders, portfolio_history
    orders.clear()
    portfolio_history.clear()
    
    # Initialize capital and holdings
    capital = investment_amount
    holdings = {}
    
    # Get the minimum Sharpe threshold based on risk level
    min_sharpe_threshold = map_risk_threshold_to_sharpe(risk_level)
    
    # Get filtered stocks based on risk level
    filtered_data = data_manager.get_stocks_by_risk_level(risk_level)
    trading_data = merged_data[merged_data['Ticker'].isin(filtered_data['Ticker'])]
    
    # Get unique trading days within the date range
    trading_days = trading_data['date'].dt.date.unique()
    trading_days = [day for day in trading_days if start_date <= day <= end_date]
    
    for day in trading_days:
        day_data = trading_data[trading_data['date'].dt.date == day]
        
        # Perform opening positions (buy signals)
        # Buy only if 'Buy' signal exists and Sharpe ratio >= min_sharpe_threshold
        buy_signals = day_data[(day_data['Buy'] != -1) & (day_data['Best_Prediction'] >= min_sharpe_threshold)]
        for _, row in buy_signals.iterrows():
            ticker = row['Ticker']
            price = row['Close']
            shares_to_buy = int(min(capital / price, 10))  # Buy up to 10 shares or available capital
            cost = shares_to_buy * price
            if shares_to_buy > 0 and cost <= capital:
                capital -= cost
                holdings[ticker] = holdings.get(ticker, 0) + shares_to_buy
                orders.append({
                    'date': day,
                    'ticker': ticker,
                    'action': 'Buy',
                    'shares_amount': shares_to_buy,
                    'price': price,
                    'investment_amount': cost,
                    'previous_shares': holdings.get(ticker, 0) - shares_to_buy,
                    'new_total_shares': holdings[ticker],
                    'sharpe': row['Best_Prediction'],
                    'capital_after': capital
                })
        
        # Perform closing positions (sell signals)
        # Sell only if 'Sell' signal exists and Sharpe ratio <= -min_sharpe_threshold
        sell_signals = day_data[(day_data['Sell'] != -1) & (day_data['Best_Prediction'] <= -min_sharpe_threshold)]
        for _, row in sell_signals.iterrows():
            ticker = row['Ticker']
            price = row['Close']
            if ticker in holdings and holdings[ticker] > 0:
                shares_to_sell = holdings[ticker]  # Sell all
                revenue = shares_to_sell * price
                capital += revenue
                del holdings[ticker]
                orders.append({
                    'date': day,
                    'ticker': ticker,
                    'action': 'Sell',
                    'shares_amount': shares_to_sell,
                    'price': price,
                    'investment_amount': revenue,
                    'previous_shares': shares_to_sell,
                    'new_total_shares': 0,
                    'sharpe': row['Best_Prediction'],
                    'capital_after': capital
                })
        
        # Calculate portfolio value at the end of the day
        portfolio_value = capital
        for ticker in holdings:
            close_price = day_data[day_data['Ticker'] == ticker]['Close'].values
            if len(close_price) > 0:
                portfolio_value += holdings[ticker] * close_price[0]
        portfolio_history.append({'date': day, 'portfolio_value': portfolio_value})

def get_orders():
    print("Orders:", orders)
    return orders.copy()

def get_portfolio_history():
    return portfolio_history.copy()

def get_order_history():
    """Return the order history as a DataFrame."""
    global orders
    return pd.DataFrame(orders)

# Main execution with user input
if __name__ == "__main__":
    # Step 1: Assume merged_data is provided or loaded (modify as needed)
    # Here, you'd load your merged_data; for now, it's a placeholder
    merged_data = pd.DataFrame()  # Replace with actual data loading logic

    # Step 2: Get input from the user
    risk_threshold = float(input("Enter risk threshold (0-100%): "))
    investment_amount = float(input("Enter investment amount: "))
    trading_window = input("Enter trading window (e.g., '30 days', '1 year'): ")

    # Convert trading window to number of days
    trading_window = trading_window.lower().strip()
    if 'year' in trading_window:
        num = float(trading_window.split()[0])
        investment_period_days = int(num * 365)
    elif 'month' in trading_window:
        num = float(trading_window.split()[0])
        investment_period_days = int(num * 30)
    elif 'day' in trading_window:
        num = float(trading_window.split()[0])
        investment_period_days = int(num)
    else:
        raise ValueError("Invalid trading window format. Use 'X days', 'X months', or 'X years'.")

    # Step 3: Convert risk percentage to Sharpe threshold
    sharpe_threshold = map_risk_threshold_to_sharpe(risk_threshold)

    # Step 4: Start trading algorithm
    start_date = datetime.now()  # Or get from user/data
    end_date = start_date + timedelta(days=investment_period_days - 1)
    run_trading_strategy(investment_amount, risk_threshold, start_date, end_date, merged_data, None)  # Pass None as data_manager for now

    print("Trading strategy executed successfully.")
    print("Order history:", get_order_history())