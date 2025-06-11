# standalone_trading_runner.py

#!/usr/bin/env python3
"""
Standalone Trading Logic Runner - FINAL FIXED VERSION
====================================================

Run trading strategy directly without frontend.
Tracks portfolio state, trade history, and profitability.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
import sys

# Fix import paths - similar to trading_connector.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
frontend_dir = os.path.join(parent_dir, 'frontend')

# Add paths to sys.path
sys.path.append(current_dir)
sys.path.append(parent_dir) 
sys.path.append(frontend_dir)

print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"Frontend dir: {frontend_dir}")

# Try to import logging config
try:
    from frontend.logging_config import setup_logging, get_logger
    print("Successfully imported from frontend.logging_config")
except ImportError:
    try:
        from logging_config import setup_logging, get_logger
        print("Successfully imported from logging_config")
    except ImportError:
        print("Could not import logging_config. Creating basic logging setup...")
        
        # Fallback logging setup
        def setup_logging():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(f'trading_standalone_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
                ]
            )
            print("Basic logging setup completed")
        
        def get_logger(name):
            return logging.getLogger(name)

# Now try to import trading logic
try:
    from trading_logic_new import run_trading_strategy, get_orders, get_portfolio_history, validate_prediction_quality
    print("  Successfully imported trading logic")
except ImportError as e:
    print(f"Error importing trading logic: {e}")
    print("This usually means the import path in trading_logic_new.py is wrong")
    print("Let's try to fix it dynamically...")
    
    # Read and fix the trading logic file
    trading_logic_path = os.path.join(current_dir, 'trading_logic_new.py')
    
    if os.path.exists(trading_logic_path):
        with open(trading_logic_path, 'r') as f:
            content = f.read()
        
        # Fix the problematic import line
        original_import = 'from frontend.logging_config import get_isolated_logger'
        
        if original_import in content:
            print("Found problematic import. Trying to fix it...")
            
            # Try different import options
            fixed_imports = [
                'from frontend.logging_config import get_isolated_logger',
                'from logging_config import get_isolated_logger',
                '''# Import fix for standalone execution
try:
    from frontend.logging_config import get_isolated_logger
except ImportError:
    try:
        from logging_config import get_isolated_logger
    except ImportError:
        # Fallback logger
        import logging
        def get_isolated_logger(name, prefix, level):
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(level)
            return logger'''
            ]
            
            # Replace with the robust import
            fixed_content = content.replace(original_import, fixed_imports[2])
            
            # Write to a temporary file
            temp_file = os.path.join(current_dir, 'trading_logic_fixed.py')
            with open(temp_file, 'w') as f:
                f.write(fixed_content)
            
            print(f"Created fixed version: {temp_file}")
            
            # Import the fixed version
            import importlib.util
            spec = importlib.util.spec_from_file_location("trading_logic_fixed", temp_file)
            trading_logic_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(trading_logic_module)
            
            # Extract the functions we need
            run_trading_strategy = trading_logic_module.run_trading_strategy
            get_orders = trading_logic_module.get_orders
            get_portfolio_history = trading_logic_module.get_portfolio_history
            validate_prediction_quality = trading_logic_module.validate_prediction_quality
            
            print("Successfully imported fixed trading logic")
            
        else:
            print("Could not find the problematic import line")
            sys.exit(1)
    else:
        print(f"Could not find trading_logic_new.py at {trading_logic_path}")
        sys.exit(1)

class TradingRunner:
    def __init__(self, csv_path, output_dir="trading_results"):
        """
        Initialize the trading runner.
        
        Args:
            csv_path: Path to your data CSV file
            output_dir: Directory to save results
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.data = None
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Setup logging
        setup_logging()
        self.logger = get_logger('trading_runner')
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and validate data from CSV."""
        try:
            self.logger.info(f"Loading data from {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)
            
            # Validate required columns
            required_columns = ['Date', 'Ticker', 'Close', 'Best_Prediction']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Add Actual_Sharpe if missing
            if 'Actual_Sharpe' not in self.data.columns:
                self.data['Actual_Sharpe'] = -1.0
                self.logger.warning("Actual_Sharpe column missing, filled with -1.0")
            
            # Convert dates
            self.data['date'] = pd.to_datetime(self.data['Date'], utc=True)
            
            # Data info
            self.logger.info(f"Data loaded: {len(self.data)} rows")
            self.logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
            self.logger.info(f"Tickers: {len(self.data['Ticker'].unique())}")
            self.logger.info(f"Columns: {list(self.data.columns)}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def run_strategy(self, 
                    investment_amount=10000,
                    risk_level=5,
                    start_date=None,
                    end_date=None,
                    mode="automatic",
                    reset_state=True,
                    use_weights=True,
                    use_signal_strength=True):
        """
        Run the trading strategy.
        
        Args:
            investment_amount: Starting cash
            risk_level: 0-10 (0=conservative, 10=aggressive)
            start_date: Start date (auto-detect if None)
            end_date: End date (auto-detect if None)
            mode: "automatic" or "semi-automatic"
            reset_state: Whether to reset portfolio state
        """
        try:
            # Auto-detect date range if not provided
            if start_date is None:
                start_date = self.data['date'].min()
            if end_date is None:
                end_date = self.data['date'].max()
                
            # Ensure timezone-aware dates
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date, utc=True)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date, utc=True)
                
            self.logger.info("="*80)
            self.logger.info("STARTING TRADING STRATEGY")
            self.logger.info("="*80)
            self.logger.info(f"Investment Amount: ${investment_amount:,}")
            self.logger.info(f"Risk Level: {risk_level}")
            self.logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
            self.logger.info(f"Mode: {mode}")
            self.logger.info(f"Reset State: {reset_state}")
            self.logger.info(f"Use Weights: {use_weights}")
            self.logger.info(f"Use Signal Strength: {use_signal_strength}")
            
            # Validate prediction quality first
            correlation, buy_hit_rate, sell_hit_rate, sharpe_min, sharpe_max = validate_prediction_quality(self.data)
            
            # Run the strategy
            result = run_trading_strategy(
                merged_data=self.data,
                investment_amount=investment_amount,
                risk_level=risk_level,
                start_date=start_date,
                end_date=end_date,
                mode=mode,
                reset_state=reset_state,
                use_weights=use_weights,
                use_signal_strength=use_signal_strength
            )
            
            if mode == "automatic":
                orders, portfolio_history, final_value, warning_message = result
            else:
                orders, warning_message = result
                portfolio_history = get_portfolio_history()
                final_value = portfolio_history[-1]['value'] if portfolio_history else investment_amount
            
            # Store results
            self.results = {
                'orders': orders,
                'portfolio_history': portfolio_history,
                'final_value': final_value,
                'warning_message': warning_message,
                'signal_correlation': correlation,
                'buy_hit_rate': buy_hit_rate,
                'sell_hit_rate': sell_hit_rate,
                'investment_amount': investment_amount,
                'total_return_pct': ((final_value / investment_amount) - 1) * 100,
                'use_weights': use_weights,
                'use_signal_strength': use_signal_strength
            }
            
            self.logger.info("="*80)
            self.logger.info("STRATEGY COMPLETED")
            self.logger.info("="*80)
            self.logger.info(f"Total Orders: {len(orders)}")
            self.logger.info(f"Final Value: ${final_value:,.2f}")
            self.logger.info(f"Total Return: {self.results['total_return_pct']:.2f}%")
            self.logger.info(f"Features Used: Weights={use_weights}, Signal Strength={use_signal_strength}")
            self.logger.info(f"Warning: {warning_message}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {e}", exc_info=True)
            raise
    
    def analyze_trade_profitability(self):
        """
        Enhanced profitability analysis including both completed and open positions.
        
        Returns a DataFrame with detailed trade analysis covering:
        1. Completed trades (buy → sell)
        2. Open positions (still holding shares)
        """

        try:
            orders = self.results.get('orders', [])
            portfolio_history = self.results.get('portfolio_history', [])
            
            if not orders:
                self.logger.warning("No orders found for profitability analysis")
                return pd.DataFrame()
            
            # Convert to DataFrame for easier analysis
            orders_df = pd.DataFrame(orders)
            
            # Separate buy and sell orders
            buy_orders = orders_df[orders_df['action'] == 'buy'].copy()
            sell_orders = orders_df[orders_df['action'] == 'sell'].copy()
            
            # Get current holdings and latest prices
            current_holdings = {}
            latest_prices = {}
            
            if portfolio_history:
                current_holdings = portfolio_history[-1].get('holdings', {})
                
                # Get latest prices from the data for current valuation
                latest_date = self.data['date'].max()
                latest_data = self.data[self.data['date'] == latest_date]
                for _, row in latest_data.iterrows():
                    latest_prices[row['Ticker']] = row['Close']
            
            # ===========================================
            # CASE 1: COMPLETED TRADES (buy -> sell)
            # ===========================================
            completed_trades = []
            
            if not sell_orders.empty:
                for _, sell_order in sell_orders.iterrows():
                    ticker = sell_order['ticker']
                    sell_date = sell_order['date']
                    sell_price = sell_order['price']
                    shares_sold = sell_order['shares_amount']
                    
                    # Get profit/loss info from sell order
                    if 'profit_loss' in sell_order and 'purchase_price' in sell_order:
                        # Direct profit/loss calculation from sell order
                        profit_loss = sell_order['profit_loss']
                        purchase_price = sell_order['purchase_price']
                        days_held = sell_order.get('days_held', 0)
                        profit_pct = sell_order.get('profit_pct', 0)
                        
                        trade_result = {
                            'ticker': ticker,
                            'trade_type': 'COMPLETED',
                            'shares': shares_sold,
                            'buy_price': purchase_price,
                            'sell_price': sell_price,
                            'current_price': sell_price,  # For completed trades, current = sell price
                            'trade_date': sell_date,
                            'days_held': days_held,
                            'profit_loss_dollar': profit_loss,
                            'profit_loss_pct': profit_pct * 100,
                            'trade_value': shares_sold * sell_price,
                            'is_profitable': profit_loss > 0
                        }
                        completed_trades.append(trade_result)
                    else:
                        # Fallback: find matching buy orders
                        ticker_buys = buy_orders[buy_orders['ticker'] == ticker]
                        if not ticker_buys.empty:
                            # Use the most recent buy order as approximation
                            recent_buy = ticker_buys.iloc[-1]
                            buy_price = recent_buy['price']
                            profit_loss = shares_sold * (sell_price - buy_price)
                            profit_pct = ((sell_price / buy_price) - 1) * 100
                            
                            trade_result = {
                                'ticker': ticker,
                                'trade_type': 'COMPLETED',
                                'shares': shares_sold,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'current_price': sell_price,
                                'trade_date': sell_date,
                                'days_held': 0,  # Unknown
                                'profit_loss_dollar': profit_loss,
                                'profit_loss_pct': profit_pct,
                                'trade_value': shares_sold * sell_price,
                                'is_profitable': profit_loss > 0
                            }
                            completed_trades.append(trade_result)
            
            # ===========================================
            # CASE 2: OPEN POSITIONS (still holding)
            # ===========================================
            open_positions = []
            
            for ticker, holding in current_holdings.items():
                shares = holding.get('shares', 0)
                purchase_price = holding.get('purchase_price', 0)
                purchase_date = holding.get('purchase_date')
                
                # Get current price
                current_price = latest_prices.get(ticker, purchase_price)  # Fallback to purchase price
                
                # Calculate current profit/loss
                current_value = shares * current_price
                invested_value = shares * purchase_price
                profit_loss = current_value - invested_value
                profit_pct = ((current_price / purchase_price) - 1) * 100 if purchase_price > 0 else 0
                
                # Calculate days held
                days_held = 0
            if purchase_date:
                try:
                    # Convert purchase_date to timezone-aware if needed
                    if isinstance(purchase_date, str):
                        purchase_date = pd.to_datetime(purchase_date, utc=True)
                    elif isinstance(purchase_date, pd.Timestamp) and purchase_date.tz is None:
                        purchase_date = purchase_date.tz_localize('UTC')
                    
                    # Get current time with UTC timezone
                    current_time = pd.Timestamp.now(tz='UTC')
                    
                    # Calculate days held
                    days_held = (current_time - purchase_date).days
                    
                except Exception as e:
                    self.logger.warning(f"Could not calculate days held for {ticker}: {e}")
                    days_held = 0
            
            position_result = {
                'ticker': ticker,
                'trade_type': 'OPEN',
                'shares': shares,
                'buy_price': purchase_price,
                'sell_price': None,  # Not sold yet
                'current_price': current_price,
                'trade_date': purchase_date,
                'days_held': days_held,
                'profit_loss_dollar': profit_loss,
                'profit_loss_pct': profit_pct,
                'trade_value': current_value,
                'is_profitable': profit_loss > 0
            }
            open_positions.append(position_result)

            # ===========================================
            # COMBINE ALL TRADES
            # ===========================================
            all_trades = completed_trades + open_positions
            
            if not all_trades:
                self.logger.warning("No trades (completed or open) found for analysis")
                return pd.DataFrame()
            
            # Create results DataFrame
            trades_df = pd.DataFrame(all_trades)
            
            # ===========================================
            # ENHANCED STATISTICS
            # ===========================================
            
            # Overall statistics
            total_trades = len(trades_df)
            profitable_trades = len(trades_df[trades_df['is_profitable']])
            overall_win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Completed trades statistics
            completed_df = trades_df[trades_df['trade_type'] == 'COMPLETED']
            completed_count = len(completed_df)
            completed_profitable = len(completed_df[completed_df['is_profitable']])
            completed_win_rate = (completed_profitable / completed_count) * 100 if completed_count > 0 else 0
            
            # Open positions statistics
            open_df = trades_df[trades_df['trade_type'] == 'OPEN']
            open_count = len(open_df)
            open_profitable = len(open_df[open_df['is_profitable']])
            open_win_rate = (open_profitable / open_count) * 100 if open_count > 0 else 0
            
            # Financial metrics
            total_profit_loss = trades_df['profit_loss_dollar'].sum()
            avg_profit_loss = trades_df['profit_loss_dollar'].mean()
            avg_return_pct = trades_df['profit_loss_pct'].mean()
            
            # Completed trades financial metrics
            completed_profit_loss = completed_df['profit_loss_dollar'].sum() if not completed_df.empty else 0
            completed_avg_return = completed_df['profit_loss_pct'].mean() if not completed_df.empty else 0
            
            # Open positions financial metrics
            open_profit_loss = open_df['profit_loss_dollar'].sum() if not open_df.empty else 0
            open_avg_return = open_df['profit_loss_pct'].mean() if not open_df.empty else 0
            
            # ===========================================
            # 70% PROFITABILITY CHECK (Your Request!)
            # ===========================================
            profitability_threshold = 70.0
            meets_70_percent = overall_win_rate >= profitability_threshold
            
            # ===========================================
            # DETAILED LOGGING
            # ===========================================
            self.logger.info("="*80)
            self.logger.info("ENHANCED TRADE PROFITABILITY ANALYSIS")
            self.logger.info("="*80)
            
            # Overall Summary
            self.logger.info("OVERALL SUMMARY:")
            self.logger.info(f"  Total Trades (Completed + Open): {total_trades}")
            self.logger.info(f"  Profitable Trades: {profitable_trades}")
            self.logger.info(f"  Overall Win Rate: {overall_win_rate:.1f}%")
            self.logger.info(f"  Total Profit/Loss: ${total_profit_loss:,.2f}")
            self.logger.info(f"  Average Return per Trade: {avg_return_pct:.2f}%")
            
            # 70% Profitability Check
            self.logger.info("")
            self.logger.info("70% PROFITABILITY TARGET:")
            if meets_70_percent:
                self.logger.info(f"SUCCESS! Win rate ({overall_win_rate:.1f}%) exceeds 70% target")
            else:
                self.logger.info(f"BELOW TARGET: Win rate ({overall_win_rate:.1f}%) is below 70% target")
                self.logger.info(f"Need {profitability_threshold - overall_win_rate:.1f}% improvement to reach target")
            
            # Completed Trades Breakdown
            self.logger.info("")
            self.logger.info("COMPLETED TRADES (Buy → Sell):")
            self.logger.info(f"  Count: {completed_count}")
            if completed_count > 0:
                self.logger.info(f"  Profitable: {completed_profitable}")
                self.logger.info(f"  Win Rate: {completed_win_rate:.1f}%")
                self.logger.info(f"  Total Realized P&L: ${completed_profit_loss:,.2f}")
                self.logger.info(f"  Average Return: {completed_avg_return:.2f}%")
            else:
                self.logger.info("  No completed trades yet")
            
            # Open Positions Breakdown
            self.logger.info("")
            self.logger.info("OPEN POSITIONS (Still Holding):")
            self.logger.info(f"  Count: {open_count}")
            if open_count > 0:
                self.logger.info(f"  Currently Profitable: {open_profitable}")
                self.logger.info(f"  Current Win Rate: {open_win_rate:.1f}%")
                self.logger.info(f"  Unrealized P&L: ${open_profit_loss:,.2f}")
                self.logger.info(f"  Average Unrealized Return: {open_avg_return:.2f}%")
            else:
                self.logger.info("  No open positions")
            
            # Best and Worst Trades
            if len(trades_df) > 0:
                best_trade = trades_df.loc[trades_df['profit_loss_dollar'].idxmax()]
                worst_trade = trades_df.loc[trades_df['profit_loss_dollar'].idxmin()]
                
                self.logger.info("")
                self.logger.info("BEST & WORST TRADES:")
                self.logger.info(f"  Best: {best_trade['ticker']} ({best_trade['trade_type']}) "
                            f"+${best_trade['profit_loss_dollar']:.2f} ({best_trade['profit_loss_pct']:.1f}%)")
                self.logger.info(f"  Worst: {worst_trade['ticker']} ({worst_trade['trade_type']}) "
                            f"${worst_trade['profit_loss_dollar']:.2f} ({worst_trade['profit_loss_pct']:.1f}%)")
            
            # Add summary metrics to the DataFrame for easy access
            trades_df.attrs['summary'] = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'overall_win_rate': overall_win_rate,
                'completed_trades': completed_count,
                'completed_win_rate': completed_win_rate,
                'open_positions': open_count,
                'open_win_rate': open_win_rate,
                'meets_70_percent_target': meets_70_percent,
                'total_profit_loss': total_profit_loss,
                'avg_return_pct': avg_return_pct
            }
            
            return trades_df
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trade profitability analysis: {e}", exc_info=True)
            return pd.DataFrame()
    # def analyze_trade_profitability(self):
    #     """
    #     Analyze profitability of individual trades.
    #     Returns a DataFrame with detailed trade analysis.
    #     """
    #     try:
    #         orders = self.results.get('orders', [])
    #         if not orders:
    #             self.logger.warning("No orders found for profitability analysis")
    #             return pd.DataFrame()
            
    #         # Convert to DataFrame for easier analysis
    #         orders_df = pd.DataFrame(orders)
            
    #         # Separate buy and sell orders
    #         buy_orders = orders_df[orders_df['action'] == 'buy'].copy()
    #         sell_orders = orders_df[orders_df['action'] == 'sell'].copy()
            
    #         if sell_orders.empty:
    #             self.logger.warning("No sell orders found - cannot calculate trade profitability")
    #             return pd.DataFrame()
            
    #         # Track trade profitability
    #         trade_results = []
            
    #         for _, sell_order in sell_orders.iterrows():
    #             ticker = sell_order['ticker']
    #             sell_date = sell_order['date']
    #             sell_price = sell_order['price']
    #             shares_sold = sell_order['shares_amount']
                
    #             # Get profit/loss info from sell order
    #             if 'profit_loss' in sell_order and 'purchase_price' in sell_order:
    #                 # Direct profit/loss calculation from sell order
    #                 profit_loss = sell_order['profit_loss']
    #                 purchase_price = sell_order['purchase_price']
    #                 days_held = sell_order.get('days_held', 0)
    #                 profit_pct = sell_order.get('profit_pct', 0)
                    
    #                 trade_result = {
    #                     'ticker': ticker,
    #                     'shares': shares_sold,
    #                     'buy_price': purchase_price,
    #                     'sell_price': sell_price,
    #                     'sell_date': sell_date,
    #                     'days_held': days_held,
    #                     'profit_loss_dollar': profit_loss,
    #                     'profit_loss_pct': profit_pct * 100,
    #                     'trade_value': shares_sold * sell_price,
    #                     'is_profitable': profit_loss > 0
    #                 }
    #                 trade_results.append(trade_result)
    #             else:
    #                 # Fallback: find matching buy orders
    #                 ticker_buys = buy_orders[buy_orders['ticker'] == ticker]
    #                 if not ticker_buys.empty:
    #                     # Use the most recent buy order as approximation
    #                     recent_buy = ticker_buys.iloc[-1]
    #                     buy_price = recent_buy['price']
    #                     profit_loss = shares_sold * (sell_price - buy_price)
    #                     profit_pct = ((sell_price / buy_price) - 1) * 100
                        
    #                     trade_result = {
    #                         'ticker': ticker,
    #                         'shares': shares_sold,
    #                         'buy_price': buy_price,
    #                         'sell_price': sell_price,
    #                         'sell_date': sell_date,
    #                         'days_held': 0,  # Unknown
    #                         'profit_loss_dollar': profit_loss,
    #                         'profit_loss_pct': profit_pct,
    #                         'trade_value': shares_sold * sell_price,
    #                         'is_profitable': profit_loss > 0
    #                     }
    #                     trade_results.append(trade_result)
            
    #         if not trade_results:
    #             self.logger.warning("Could not match any buy/sell pairs for profitability analysis")
    #             return pd.DataFrame()
            
    #         # Create results DataFrame
    #         trades_df = pd.DataFrame(trade_results)
            
    #         # Summary statistics
    #         total_trades = len(trades_df)
    #         profitable_trades = len(trades_df[trades_df['is_profitable']])
    #         win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    #         total_profit_loss = trades_df['profit_loss_dollar'].sum()
    #         avg_profit_loss = trades_df['profit_loss_dollar'].mean()
    #         avg_return_pct = trades_df['profit_loss_pct'].mean()
            
    #         self.logger.info("="*60)
    #         self.logger.info("TRADE PROFITABILITY ANALYSIS")
    #         self.logger.info("="*60)
    #         self.logger.info(f"Total Completed Trades: {total_trades}")
    #         self.logger.info(f"Profitable Trades: {profitable_trades}")
    #         self.logger.info(f"Win Rate: {win_rate:.1f}%")
    #         self.logger.info(f"Total Profit/Loss: ${total_profit_loss:,.2f}")
    #         self.logger.info(f"Average Profit/Loss per Trade: ${avg_profit_loss:,.2f}")
    #         self.logger.info(f"Average Return per Trade: {avg_return_pct:.2f}%")
            
    #         # Top profitable and losing trades
    #         if len(trades_df) > 0:
    #             best_trade = trades_df.loc[trades_df['profit_loss_dollar'].idxmax()]
    #             worst_trade = trades_df.loc[trades_df['profit_loss_dollar'].idxmin()]
                
    #             self.logger.info(f"Best Trade: {best_trade['ticker']} (+${best_trade['profit_loss_dollar']:.2f}, {best_trade['profit_loss_pct']:.1f}%)")
    #             self.logger.info(f"Worst Trade: {worst_trade['ticker']} (${worst_trade['profit_loss_dollar']:.2f}, {worst_trade['profit_loss_pct']:.1f}%)")
            
    #         return trades_df
            
    #     except Exception as e:
    #         self.logger.error(f"Error in trade profitability analysis: {e}", exc_info=True)
    #         return pd.DataFrame()
    
    def save_results(self, filename_prefix="trading_results"):
        """Save all results to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            orders_file = None
            portfolio_file = None
            trades_file = None
            
            # Save orders
            if self.results.get('orders'):
                orders_file = f"{self.output_dir}/{filename_prefix}_orders_{timestamp}.csv"
                orders_df = pd.DataFrame(self.results['orders'])
                orders_df.to_csv(orders_file, index=False)
                self.logger.info(f"Orders saved to: {orders_file}")
            
            # Save portfolio history
            if self.results.get('portfolio_history'):
                portfolio_file = f"{self.output_dir}/{filename_prefix}_portfolio_{timestamp}.csv"
                portfolio_df = pd.DataFrame(self.results['portfolio_history'])
                # Convert holdings dict to string for CSV
                portfolio_df['holdings_str'] = portfolio_df['holdings'].astype(str)
                portfolio_df.drop('holdings', axis=1, inplace=True)
                portfolio_df.to_csv(portfolio_file, index=False)
                self.logger.info(f"Portfolio history saved to: {portfolio_file}")
            
            # Save trade profitability analysis
            trades_df = self.analyze_trade_profitability()
            if not trades_df.empty:
                trades_file = f"{self.output_dir}/{filename_prefix}_trades_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False)
                self.logger.info(f"Trade analysis saved to: {trades_file}")
            
            # Save summary
            summary_file = f"{self.output_dir}/{filename_prefix}_summary_{timestamp}.json"
            summary = {
                'timestamp': timestamp,
                'investment_amount': self.results.get('investment_amount', 0),
                'final_value': self.results.get('final_value', 0),
                'total_return_pct': self.results.get('total_return_pct', 0),
                'total_orders': len(self.results.get('orders', [])),
                'signal_correlation': self.results.get('signal_correlation', 0),
                'buy_hit_rate': self.results.get('buy_hit_rate', 0),
                'sell_hit_rate': self.results.get('sell_hit_rate', 0),
                'warning_message': self.results.get('warning_message', '')
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Summary saved to: {summary_file}")
            
            return {
                'orders_file': orders_file,
                'portfolio_file': portfolio_file,
                'trades_file': trades_file,
                'summary_file': summary_file
            }
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}", exc_info=True)
            return {}
    
    def print_current_holdings(self):
        """Print current portfolio holdings."""
        try:
            portfolio_history = self.results.get('portfolio_history', [])
            if not portfolio_history:
                self.logger.warning("No portfolio history available")
                return
            
            latest_portfolio = portfolio_history[-1]
            holdings = latest_portfolio.get('holdings', {})
            cash = latest_portfolio.get('cash', 0)
            total_value = latest_portfolio.get('value', 0)
            
            self.logger.info("="*60)
            self.logger.info("CURRENT PORTFOLIO HOLDINGS")
            self.logger.info("="*60)
            self.logger.info(f"Cash: ${cash:,.2f}")
            self.logger.info(f"Total Portfolio Value: ${total_value:,.2f}")
            self.logger.info(f"Number of Positions: {len(holdings)}")
            
            if holdings:
                self.logger.info("\nStock Holdings:")
                for ticker, holding in holdings.items():
                    shares = holding.get('shares', 0)
                    purchase_price = holding.get('purchase_price', 0)
                    position_value = shares * purchase_price  # Approximate
                    self.logger.info(f"  {ticker}: {shares} shares @ ${purchase_price:.2f} = ${position_value:,.2f}")
            else:
                self.logger.info("No stock holdings")
                
        except Exception as e:
            self.logger.error(f"Error printing holdings: {e}", exc_info=True)


def cleanup_temp_files():
    """Clean up any temporary files created during execution."""
    temp_files = ['trading_logic_fixed.py']
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up: {temp_file}")
        except:
            pass


def main():
    """Main function for standalone execution."""
    # Configuration - UPDATE THESE VALUES
    CSV_PATH = "20250527_all_tickers_results.csv"  # UPDATE THIS PATH
    INVESTMENT_AMOUNT = 10000
    RISK_LEVEL = 3  # 0-10
    
    # # Optional: specify date range
    # START_DATE = "2022-01-01"
    # END_DATE = "2022-04-01"   

    # START_DATE = "2022-01-01" 
    # END_DATE = "2022-07-01" 

    START_DATE = "2022-01-01" 
    END_DATE = "2023-01-01" 
    
    try:
        print("="*80)
        print("STANDALONE TRADING LOGIC RUNNER")
        print("="*80)
        
        # Check if data file exists
        if not os.path.exists(CSV_PATH):
            print(f"  ERROR: Data file not found: {CSV_PATH}")
            print("Please update the CSV_PATH variable in the main() function")
            return
        
        # Create runner
        print(f"Initializing with data file: {CSV_PATH}")
        runner = TradingRunner(CSV_PATH)
        
        # Run strategy
        print(f"Running strategy...")
        results = runner.run_strategy(
            investment_amount=INVESTMENT_AMOUNT,
            risk_level=RISK_LEVEL,
            start_date=START_DATE,
            end_date=END_DATE,
            mode="automatic",
            reset_state=True,
            use_weights=True,
            use_signal_strength=True
        )
        
        # Analyze trades
        print(f"Analyzing trade profitability...")
        trades_df = runner.analyze_trade_profitability()
        
        # Print current holdings
        runner.print_current_holdings()
        
        # Save all results
        print(f"Saving results...")
        saved_files = runner.save_results()
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Check the 'logs/' directory for detailed execution logs")
        print(f"Check the 'trading_results/' directory for output files")
        
        if saved_files:
            print("\nSaved files:")
            for file_type, file_path in saved_files.items():
                if file_path:
                    print(f"  {file_type}: {file_path}")
        
        # Quick summary
        print(f"\n  QUICK SUMMARY:")
        print(f"  Initial Investment: ${INVESTMENT_AMOUNT:,}")
        print(f"  Final Value: ${results['final_value']:,.2f}")
        print(f"  Total Return: {results['total_return_pct']:.2f}%")
        print(f"  Total Orders: {len(results['orders'])}")
        
    except FileNotFoundError:
        print(f"  ERROR: Data file not found: {CSV_PATH}")
        print("Please update the CSV_PATH variable in the main() function")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary files
        cleanup_temp_files()


if __name__ == "__main__":
    main()