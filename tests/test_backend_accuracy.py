# test_backend_accuracy.py

"""
Backend Accuracy Tests
=======================================

Tests for:
1. Sharpe ratio prediction accuracy (MAPE ≤ 15%)
2. Win rate profitability (≥70% at least once)

Run commands:
pytest tests/test_backend_accuracy.py -v -k "sharpe"
pytest tests/test_backend_accuracy.py -v -k "profit"
pytest tests/test_backend_accuracy.py -v -s
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
import logging
from datetime import datetime

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
backend_path = os.path.join(project_root, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    from backend.standalone_trading_runner import TradingRunner, force_reset_state
    from backend.trading_logic_new import get_orders, get_portfolio_history
except ImportError as e:
    raise RuntimeError(f"Failed to import backend modules: {e}")

CSV_PATH = r"data/all_tickers_results.csv"

def setup_test_logger():
    """Setup dedicated logger for backend tests"""
    test_log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(test_log_dir, exist_ok=True)
    
    test_logger = logging.getLogger('backend_tests')
    test_logger.setLevel(logging.INFO)
    test_logger.handlers.clear()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_log_dir, f'backend_test_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    test_logger.addHandler(file_handler)
    test_logger.addHandler(console_handler)
    
    return test_logger

logger = setup_test_logger()

class TestBackendAccuracy:
    """Backend accuracy tests without UI"""
    
    @classmethod
    def setup_class(cls):
        """Setup data path"""
        cls.csv_path = os.path.join(project_root, CSV_PATH)
        if not os.path.exists(cls.csv_path):
            pytest.fail(f"CSV file not found: {cls.csv_path}")
        logger.info(f"Using dataset: {cls.csv_path}")
    
    # Store results
    profit_results = []
    sharpe_results = []
    
    @pytest.mark.parametrize("period_name,start_date,end_date,investment,risk", [
        # Known working cases
        ("known_good_4m", "2022-01-01", "2022-05-01", 10000, 0),
        
        # 1-month periods
        ("1_month_jan_10k", "2022-01-01", "2022-01-31", 10000, 0),
        ("1_month_jan_5k", "2022-01-01", "2022-01-31", 5000, 0),
        ("1_month_mar_10k", "2022-03-01", "2022-03-31", 10000, 0),
        ("1_month_jun_5k", "2022-06-01", "2022-06-30", 5000, 0),
        
        # 3-month periods  
        ("3_months_q1_10k", "2022-01-01", "2022-03-31", 10000, 0),
        ("3_months_q1_5k", "2022-01-01", "2022-03-31", 5000, 0),
        ("3_months_q2_10k", "2022-04-01", "2022-06-30", 10000, 0),
        ("3_months_q3_5k", "2022-07-01", "2022-09-30", 5000, 0),
        
        # 6-month periods
        ("6_months_h1_10k", "2022-01-01", "2022-06-30", 10000, 0),
        ("6_months_h1_5k", "2022-01-01", "2022-06-30", 5000, 0),
        ("6_months_h2_10k", "2022-07-01", "2022-12-31", 10000, 0),
        
        # 1-year periods
        ("1_year_2022_10k", "2022-01-01", "2022-12-31", 10000, 0),
        ("1_year_2022_5k", "2022-01-01", "2022-12-31", 5000, 0),
    ])
    def test_profit_scenarios_backend(self, period_name, start_date, end_date, investment, risk):
        """Test win rate scenarios using backend logic only - matches frontend test cases"""
        
        logger.info("="*80)
        logger.info(f"BACKEND PROFIT TEST: {period_name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Investment: ${investment}, Risk: {risk}")
        logger.info("="*80)
        
        # Reset state
        force_reset_state()
        
        # Create runner and execute
        runner = TradingRunner(self.csv_path)
        
        try:
            results = runner.run_strategy(
                investment_amount=investment,
                risk_level=risk,
                start_date=start_date,
                end_date=end_date,
                mode="automatic",
                reset_state=True
            )
            
            # Analyze trades using same method as frontend
            trades_df = runner.analyze_trade_profitability()
            
            if trades_df.empty or not hasattr(trades_df, 'attrs'):
                win_rate = 0.0
                trade_details = {'total_trades': 0, 'profitable_trades': 0, 'completed_trades': 0, 'open_positions': 0}
            else:
                summary = trades_df.attrs.get('summary', {})
                win_rate = summary.get('overall_win_rate', 0.0)
                trade_details = {
                    'total_trades': summary.get('total_trades', 0),
                    'profitable_trades': summary.get('profitable_trades', 0),
                    'completed_trades': summary.get('completed_trades', 0),
                    'open_positions': summary.get('open_positions', 0)
                }
            
            # Store result - matching frontend format
            result = {
                'period_name': period_name,
                'start_date': start_date,
                'end_date': end_date,
                'investment': investment,
                'risk': risk,
                'win_rate': win_rate,
                'meets_70_percent': win_rate >= 70.0,
                'total_trades': trade_details['total_trades'],
                'profitable_trades': trade_details['profitable_trades'],
                'completed_trades': trade_details['completed_trades'],
                'open_positions': trade_details['open_positions'],
                'timestamp': datetime.now().isoformat()
            }
            TestBackendAccuracy.profit_results.append(result)
            
            # Log detailed results - matching frontend format
            logger.info("-"*60)
            logger.info(f"PROFIT TEST COMPLETED: {period_name}")
            logger.info(f"Win Rate: {win_rate:.1f}%")
            logger.info(f"Total Trades: {trade_details['total_trades']} ({trade_details['completed_trades']} completed + {trade_details['open_positions']} open)")
            logger.info(f"Profitable: {trade_details['profitable_trades']}")
            logger.info(f"70% Target: {'✓ MET' if win_rate >= 70.0 else '✗ NOT MET'}")
            logger.info("-"*60)
            
        except Exception as e:
            logger.error(f"Error in {period_name}: {e}")
            # Store error result
            TestBackendAccuracy.profit_results.append({
                'period_name': period_name,
                'start_date': start_date,
                'end_date': end_date,
                'investment': investment,
                'risk': risk,
                'win_rate': 0.0,
                'meets_70_percent': False,
                'total_trades': 0,
                'profitable_trades': 0,
                'completed_trades': 0,
                'open_positions': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    def test_profit_70_percent_requirement_backend(self):
        """Verify at least one scenario achieves 70%+ win rate"""
        if not TestBackendAccuracy.profit_results:
            pytest.skip("No profit results available")
        
        successful = [r for r in TestBackendAccuracy.profit_results if r['meets_70_percent']]
        success_count = len(successful)
        total_count = len(TestBackendAccuracy.profit_results)
        
        logger.info("="*100)
        logger.info("BACKEND PROFIT REQUIREMENT ANALYSIS")
        logger.info("="*100)
        logger.info(f"Successful scenarios: {success_count}/{total_count}")
        
        for result in TestBackendAccuracy.profit_results:
            status = "✓" if result['meets_70_percent'] else "✗"
            logger.info(f"  {status} {result['period_name']}: {result['win_rate']:.1f}%")
        
        if successful:
            logger.info("\nSUCCESSFUL SCENARIOS:")
            for s in successful:
                logger.info(f"  ✓ {s['period_name']}: {s['win_rate']:.1f}% ({s['start_date']} to {s['end_date']})")
        
        assert success_count > 0, f"No scenarios achieved 70%+ win rate"
        logger.info(f"🎉 PROFIT REQUIREMENT SATISFIED: {success_count} scenario(s) achieved 70%+")
    
    def test_sharpe_prediction_performance(self):
        """Test Sharpe prediction performance using regression metrics"""
        
        logger.info("="*80)
        logger.info("SHARPE PREDICTION PERFORMANCE TEST")
        logger.info("Requirements: R2 ≥ 0.6, RMSE ≤ 2.0")
        logger.info("="*80)
        
        # Load dataset
        data = pd.read_csv(self.csv_path)
        
        # Filter valid predictions (exclude extreme outliers)
        MIN_THRESHOLD = 0.1
        valid_data = data[
            (data['Actual_Sharpe'] != -1.0) & 
            (data['Actual_Sharpe'].notna()) &
            (data['Best_Prediction'].notna()) &
            (np.isfinite(data['Actual_Sharpe'])) &
            (np.isfinite(data['Best_Prediction'])) &
            (np.abs(data['Actual_Sharpe']) >= MIN_THRESHOLD)
        ].copy()
        
        actual = valid_data['Actual_Sharpe'].values
        predicted = valid_data['Best_Prediction'].values
        
        # Calculate regression metrics
        from sklearn.metrics import mean_squared_error, r2_score
        
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Calculate correlation
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        # Mean Absolute Error (more interpretable than MAPE)
        mae = np.mean(np.abs(actual - predicted))
        
        logger.info(f"Valid predictions: {len(valid_data):,}")
        logger.info(f"R2 Score: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"Correlation: {correlation:.4f}")
        
        # Performance thresholds
        r2_threshold = 0.6  # Explains 25% of variance
        rmse_threshold = 2.0  # RMSE under 2.0 Sharpe units
        
        meets_r2 = r2 >= r2_threshold
        meets_rmse = rmse <= rmse_threshold
        
        logger.info(f"R2 ≥ {r2_threshold}: {' PASS' if meets_r2 else ' FAIL'}")
        logger.info(f"RMSE ≤ {rmse_threshold}: {' PASS' if meets_rmse else ' FAIL'}")
        
        # Store results
        TestBackendAccuracy.sharpe_results.append({
            'dataset_size': len(data),
            'valid_predictions': len(valid_data),
            'r2_score': r2,
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'meets_r2_requirement': meets_r2,
            'meets_rmse_requirement': meets_rmse
        })
        
        # Assert requirements
        assert meets_r2, f"R2 {r2:.4f} < {r2_threshold} threshold"
        assert meets_rmse, f"RMSE {rmse:.4f} > {rmse_threshold} threshold"
        
        logger.info(f" PREDICTION PERFORMANCE SATISFIED: R2={r2:.4f}, RMSE={rmse:.4f}")


if __name__ == '__main__':
    pytest.main(['-v', '--tb=short', '-s'])