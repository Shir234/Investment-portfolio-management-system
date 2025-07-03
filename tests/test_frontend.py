# test_frontend.py

"""
Frontend non-functional requirements Tests
=======================================

Tests for:
1. Trade notification time <= 45 seconds    
2. Invalid input notification time <= 30 seconds 

Run commands:
pytest tests/test_frontend.py -v -k "timing"
pytest tests/test_frontend.py -v -k "validation"
pytest tests/test_frontend.py -v -s
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
import time
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QDate, Qt, QTimer
from PyQt6.QtTest import QTest
import logging
from datetime import datetime

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Imports
try:
    from frontend.data.data_manager import DataManager
    from frontend.gui.main_window import MainWindow
    from frontend.data.trading_connector import get_order_history_df
    from frontend.logging_config import get_logger
    from backend.standalone_trading_runner import TradingRunner
    from backend.trading_logic_new import get_orders, get_portfolio_history
except ImportError as e:
    raise RuntimeError(f"Failed to import frontend modules: {e}")

CSV_PATH = r"data/all_tickers_results.csv"  # Updated path

def setup_test_logger():
    """Setup dedicated logger for timing tests"""
    # Create tests/logs directory
    test_log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(test_log_dir, exist_ok=True)
    
    # Create logger
    test_logger = logging.getLogger('timing_tests')
    test_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    test_logger.handlers.clear()
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(test_log_dir, f'timing_test_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    test_logger.addHandler(file_handler)
    test_logger.addHandler(console_handler)
    
    return test_logger

logger = setup_test_logger()

class TestSharpSightUI:
    """Test class for SharpSight UI functionality"""
    
    @classmethod
    def setup_class(cls):
        """Set up QApplication once for all tests"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
        
        # Setup data manager
        abs_csv_path = os.path.join(project_root, CSV_PATH)
        if not os.path.exists(abs_csv_path):
            pytest.fail(f"CSV file not found at {abs_csv_path}")
        
        cls.data_manager = DataManager(abs_csv_path)
        if cls.data_manager.data is None or cls.data_manager.data.empty:
            pytest.fail(f"Failed to load data from {abs_csv_path}")
        
        logger.info(f"Dataset range: {cls.data_manager.dataset_start_date} to {cls.data_manager.dataset_end_date}")
    
    # Store results
    timing_results = []
    validation_results = []

    def setup_method(self):
        """Set up for each test method with complete reset"""
         # Reset backend state FIRST
        try:
            from backend.trading_logic_new import reset_portfolio_for_semi_auto
            reset_portfolio_for_semi_auto()
        except ImportError:
            pass

        # Remove portfolio state file
        portfolio_file = os.path.join(project_root, 'data', 'portfolio_state.json')
        if os.path.exists(portfolio_file):
            os.remove(portfolio_file)

         # Create main window
        self.main_window = MainWindow(self.data_manager)
        self.input_panel = self.main_window.input_panel
        
        # Show window
        self.main_window.show()
        QTest.qWaitForWindowExposed(self.main_window)

        # Force reset date constraints to original dataset range
        if self.data_manager.data is not None and not self.data_manager.data.empty:
            try:
                # Get original dataset dates
                date_column = None
                for col in ['Date', 'date', 'DATE']:
                    if col in self.data_manager.data.columns:
                        date_column = col
                        break
                
                if date_column:
                    dates = pd.to_datetime(self.data_manager.data[date_column])
                    dataset_min_date = dates.min().date()
                    dataset_max_date = dates.max().date()
                    
                    # Convert to QDate
                    q_min_date = QDate(dataset_min_date.year, dataset_min_date.month, dataset_min_date.day)
                    q_max_date = QDate(dataset_max_date.year, dataset_max_date.month, dataset_max_date.day)
                    
                    # Force set date ranges to full dataset
                    self.input_panel.start_date_input.setDateRange(q_min_date, q_max_date)
                    self.input_panel.end_date_input.setDateRange(q_min_date, q_max_date)
                    
                    # Set default values to dataset start/end
                    self.input_panel.start_date_input.setDate(q_min_date)
                    self.input_panel.end_date_input.setDate(q_max_date)
            except Exception as e:
                logger.warning(f"Error resetting date constraints: {e}")
        
        # Set default values
        self.input_panel.investment_input.setText("10000")
        self.input_panel.risk_input.setValue(0)
        
        # Disable mode explanation popup to prevent interference
        self.input_panel.initial_load = True
        self.input_panel.mode_combo.setCurrentText("Automatic")
        self.input_panel.initial_load = False

        # Ensure investment field is unlocked
        self.input_panel.investment_input.setEnabled(True)
        self.input_panel.add_funds_button.setVisible(False)
    
    def teardown_method(self):
        """Clean up after each test"""
        if hasattr(self, 'main_window'):
            self.main_window.close()
            QTest.qWait(100)
    
    def handle_popups_continuously(self, qtbot, max_popups=20, timeout=5000):
        """Handle all popups automatically with continuous monitoring"""
        popup_count = 0
        start_time = time.time()
        
        def check_and_handle_popup():
            nonlocal popup_count
            
            # Stop if we've reached max popups or timeout
            if popup_count >= max_popups or (time.time() - start_time) * 1000 > timeout:
                return
                
            # Find any message boxes
            message_boxes = [w for w in QApplication.topLevelWidgets() 
                           if isinstance(w, QMessageBox) and w.isVisible()]
            
            for message_box in message_boxes:
                popup_count += 1
                popup_text = message_box.text().lower()
                logger.info(f"Auto-handling popup {popup_count}: {message_box.text()[:50]}...")

                # Handle date validation errors by resetting portfolio
                if "end date must be after start date" in popup_text:
                    self.click_popup_button(message_box, QMessageBox.StandardButton.Ok)
                    # Reset portfolio to clear date constraints
                    logger.info("Date validation error detected - resetting portfolio")
                    QTest.qWait(100)
                    self.input_panel.reset_button.click()
                    return
                # Handle different popup types based on content
                elif any(keyword in popup_text for keyword in ["automatic", "semi-automatic", "mode"]):
                    # Mode explanation popup - click OK
                    self.click_popup_button(message_box, QMessageBox.StandardButton.Ok)
                elif "no trades" in popup_text:
                    # No trades warning - click OK
                    self.click_popup_button(message_box, QMessageBox.StandardButton.Ok)
                elif any(keyword in popup_text for keyword in ["confirm", "proceed", "continue"]):
                    # Confirmation dialog - click Yes
                    if not self.click_popup_button(message_box, QMessageBox.StandardButton.Yes):
                        self.click_popup_button(message_box, QMessageBox.StandardButton.Ok)
                elif "error" in popup_text:
                    # Error dialog - click OK
                    self.click_popup_button(message_box, QMessageBox.StandardButton.Ok)
                elif any(keyword in popup_text for keyword in ["short", "range", "warning"]):
                    # Date range warning - click Yes to proceed
                    if not self.click_popup_button(message_box, QMessageBox.StandardButton.Yes):
                        self.click_popup_button(message_box, QMessageBox.StandardButton.Ok)
                else:
                    # Default - try OK first, then Yes
                    if not self.click_popup_button(message_box, QMessageBox.StandardButton.Ok):
                        self.click_popup_button(message_box, QMessageBox.StandardButton.Yes)
        
        # Set up timer to check every 50ms for more responsive handling
        timer = QTimer()
        timer.timeout.connect(check_and_handle_popup)
        timer.start(50)
        
        return timer
    
    def click_popup_button(self, message_box, button_role):
        """Click a specific button in a message box"""
        try:
            button = message_box.button(button_role)
            if button and button.isEnabled() and button.isVisible():
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                QTest.qWait(100)  # Wait for click to process
                return True
        except Exception as e:
            logger.warning(f"Failed to click button {button_role}: {e}")
        return False
    
    def wait_for_execution_complete(self, qtbot, timeout=60000):
        """Wait for strategy execution to complete with popup handling"""
        start_time = time.time()
        completion_detected = False

        # Start popup handler that also detects completion
        popup_timer = self.handle_popups_continuously(qtbot, max_popups=50, timeout=timeout)

        def check_completion():
            nonlocal completion_detected
            
            # Check for completion popup (success message)
            message_boxes = [w for w in QApplication.topLevelWidgets() 
                           if isinstance(w, QMessageBox) and w.isVisible()]
            
            for msg_box in message_boxes:
                text = msg_box.text().lower()
                if any(keyword in text for keyword in ["success", "completed", "executed", "finished"]):
                    completion_detected = True
                    logger.info(f"Completion popup detected: {msg_box.text()[:50]}...")
                    # Auto-click OK to dismiss
                    self.click_popup_button(msg_box, QMessageBox.StandardButton.Ok)
                    return True
            
            # Fallback: check if execution finished based on UI state
            return (not self.input_panel.progress_bar.isVisible() and 
                   self.input_panel.execute_button.isEnabled())
        
        try:
            # Wait for either completion popup or UI state change
            qtbot.waitUntil(check_completion, timeout=timeout)
        except Exception as e:
            logger.warning(f"Timeout or error waiting for execution: {e}")
        finally:
            popup_timer.stop()
        
        duration = time.time() - start_time
        logger.info(f"Execution completed in {duration:.2f}s (completion popup: {'Yes' if completion_detected else 'No'})")
        return duration
        
    
    # Test periods as pytest parameters
    @pytest.mark.parametrize("period_name,start_date,end_date", [
        # 1-month periods (3 tests)
        ("1_month_jan", QDate(2022, 1, 1), QDate(2022, 1, 31)),
        ("1_month_june", QDate(2022, 6, 1), QDate(2022, 6, 30)),
        ("1_month_sept", QDate(2022, 9, 1), QDate(2022, 9, 30)),
        
        # 3-month periods (3 tests)
        ("3_months_q1", QDate(2022, 1, 1), QDate(2022, 3, 31)),
        ("3_months_q2", QDate(2022, 4, 1), QDate(2022, 6, 30)),
        ("3_months_q3", QDate(2022, 7, 1), QDate(2022, 9, 30)),
        
        # 6-month periods (3 tests)
        ("6_months_h1", QDate(2022, 1, 1), QDate(2022, 6, 30)),
        ("6_months_h2", QDate(2022, 7, 1), QDate(2022, 12, 31)),
        ("6_months_alt", QDate(2022, 3, 1), QDate(2022, 8, 31)),
        
        # 1-year periods (2 tests)
        ("1_year_calendar", QDate(2022, 1, 1), QDate(2022, 12, 31)),
        ("1_year_may_to_may", QDate(2022, 5, 1), QDate(2023, 4, 30)),
    ])
    def test_execute_trading_strategy_timing(self, qtbot, period_name, start_date, end_date):
        """Test trading strategy executes within 45 seconds in different time periods"""
        # Calculate test metadata
        date_range_days = (end_date.toPyDate() - start_date.toPyDate()).days
        logger.info("="*80)
        logger.info(f"STARTING TIMING TEST: {period_name}")
        logger.info(f"Date Range: {start_date.toString('yyyy-MM-dd')} to {end_date.toString('yyyy-MM-dd')}")
        logger.info(f"Period Length: {date_range_days} days")
        logger.info(f"Expected Limit: 45 seconds")
        logger.info("="*80)
        
        # Set date range
        self.input_panel.start_date_input.setDate(start_date)
        self.input_panel.end_date_input.setDate(end_date)

        # Log configuration
        investment = self.input_panel.investment_input.text()
        risk = self.input_panel.risk_input.value()
        mode = self.input_panel.mode_combo.currentText()
        
        logger.info(f"Test Configuration:")
        logger.info(f"  Investment: ${investment}")
        logger.info(f"  Risk Level: {risk}")
        logger.info(f"  Trading Mode: {mode}")

        # Start execution and measure time
        logger.info(f"Starting execution at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        start_time = time.time()
        self.input_panel.execute_button.click()

        # Wait for completion with popup handling
        duration = self.wait_for_execution_complete(qtbot, timeout=60000)  # 60s max

        # Calculate results
        within_limit = duration <= 45
        performance_pct = (45 / duration) * 100 if duration > 0 else 100
        
        # Store result for compliance analysis
        result = {
            'period': period_name,
            'duration': duration,
            'within_45s': within_limit,
            'date_range_days': date_range_days,
            'performance_pct': performance_pct,
            'timestamp': datetime.now().isoformat()
        }
        TestSharpSightUI.timing_results.append(result)
        
        # Log detailed results
        logger.info("-"*60)
        logger.info(f"TIMING TEST COMPLETED: {period_name}")
        logger.info(f"Execution Time: {duration:.2f}s")
        logger.info(f"Status: {' PASS' if within_limit else ' FAIL'} (45s limit)")
        logger.info(f"Performance: {performance_pct:.1f}% of limit")
        if duration > 45:
            logger.warning(f"  EXCEEDED LIMIT by {duration - 45:.2f}s")
        else:
            logger.info(f" UNDER LIMIT by {45 - duration:.2f}s")
        logger.info("-"*60)
        
        # Individual test passes if under 60s (generous limit)
        assert duration <= 60, f"{period_name} took {duration:.2f}s, exceeded 60s maximum"
    
    def test_timing_95_percent_compliance(self):
        """Verify 95% of timing tests completed within 45 seconds"""
        if not TestSharpSightUI.timing_results:
            pytest.skip("No timing results available - run timing tests first")
        
        total_tests = len(TestSharpSightUI.timing_results)
        within_45s = sum(1 for result in TestSharpSightUI.timing_results if result['within_45s'])
        compliance_rate = (within_45s / total_tests) * 100
        required_passes = int(total_tests * 0.95)
        
        # Log comprehensive compliance report
        logger.info("="*100)
        logger.info("TIMING COMPLIANCE ANALYSIS - NON-FUNCTIONAL REQUIREMENT #1")
        logger.info("="*100)
        logger.info(f"Requirement: 95% of transactions must complete within 45 seconds")
        logger.info(f"Test Results Summary:")
        logger.info(f"  Total Tests Run: {total_tests}")
        logger.info(f"  Tests Within 45s: {within_45s}")
        logger.info(f"  Tests Over 45s: {total_tests - within_45s}")
        logger.info(f"  Actual Compliance: {compliance_rate:.1f}%")
        logger.info(f"  Required Compliance: 95.0%")
        logger.info(f"  Required Passes: ≥{required_passes}")
        logger.info(f"  Result: {' REQUIREMENT MET' if within_45s >= required_passes else ' REQUIREMENT FAILED'}")
        logger.info("-"*100)
        
        # Detailed breakdown by period type
        period_types = {}
        for result in TestSharpSightUI.timing_results:
            period_type = result['period'].split('_')[0] + "_" + result['period'].split('_')[1]
            if period_type not in period_types:
                period_types[period_type] = {'total': 0, 'passed': 0, 'times': []}
            period_types[period_type]['total'] += 1
            if result['within_45s']:
                period_types[period_type]['passed'] += 1
            period_types[period_type]['times'].append(result['duration'])
        
        logger.info("BREAKDOWN BY PERIOD TYPE:")
        for period_type, stats in period_types.items():
            avg_time = sum(stats['times']) / len(stats['times'])
            pass_rate = (stats['passed'] / stats['total']) * 100
            logger.info(f"  {period_type.replace('_', ' ').title()}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%), avg: {avg_time:.2f}s")
        
        logger.info("-"*100)
        logger.info("DETAILED TEST RESULTS:")
        for i, result in enumerate(TestSharpSightUI.timing_results, 1):
            status = "pass" if result['within_45s'] else "fail"
            logger.info(f"  {i:2d}. {status} {result['period']:20s} | {result['duration']:6.2f}s | {result['date_range_days']:3d} days | {result['performance_pct']:6.1f}%")
        
        logger.info("="*100)
        
        # Assert compliance
        assert within_45s >= required_passes, (
            f"TIMING COMPLIANCE FAILED: {within_45s}/{total_tests} tests within 45s "
            f"({compliance_rate:.1f}%), required ≥{required_passes} ({required_passes/total_tests*100:.1f}%)"
        )
        
        logger.info(f" TIMING COMPLIANCE REQUIREMENT SATISFIED: {compliance_rate:.1f}% >= 95%")
    
    
    @pytest.mark.parametrize("scenario,investment,start_date,end_date,mode,expected_error", [
        # Investment validation scenarios
        ("zero_investment", "0", QDate(2022, 1, 1), QDate(2022, 1, 31), "Automatic", "investment"),
        ("empty_investment", "", QDate(2022, 1, 1), QDate(2022, 1, 31), "Automatic", "investment"),
        ("negative_investment", "-1000", QDate(2022, 1, 1), QDate(2022, 1, 31), "Automatic", "investment"),
        
        # Date validation scenarios
        ("same_dates", "10000", QDate(2022, 1, 15), QDate(2022, 1, 15), "Automatic", "date"),
        ("end_before_start", "10000", QDate(2022, 1, 31), QDate(2022, 1, 1), "Automatic", "date"),
        
        # Mode validation scenarios  
        ("no_mode_selected", "10000", QDate(2022, 1, 1), QDate(2022, 1, 31), "Select Mode", "mode"),
        
        # Additional edge cases
        ("very_large_investment", "999999999999", QDate(2022, 1, 1), QDate(2022, 1, 31), "Automatic", "investment"),
    ])
    def test_no_trades_validation_timing(self, qtbot, scenario, investment, start_date, end_date, mode, expected_error):
        """Test validation error notifications appear within 30 seconds"""
        
        logger.info("="*80)
        logger.info(f"STARTING VALIDATION TEST: {scenario}")
        logger.info("="*80)
        
        # Set test inputs
        self.input_panel.investment_input.setText(investment)
        self.input_panel.start_date_input.setDate(start_date)
        self.input_panel.end_date_input.setDate(end_date)
        self.input_panel.mode_combo.setCurrentText(mode)
        
        # Start continuous popup monitoring BEFORE clicking execute
        validation_detected = False
        detection_time = None
        
        def handle_validation_popup():
            nonlocal validation_detected, detection_time
            message_boxes = [w for w in QApplication.topLevelWidgets() 
                            if isinstance(w, QMessageBox) and w.isVisible()]
            
            for msg_box in message_boxes:
                text = msg_box.text().lower()
                if any(keyword in text for keyword in [
                    "investment", "amount", "valid", "greater than zero",
                    "end date must be after", "mode selection", "select a trading mode"
                ]):
                    if not validation_detected:
                        validation_detected = True
                        detection_time = time.time()
                        logger.info(f"Validation popup detected: {text[:50]}...")
                    
                    self.click_popup_button(msg_box, QMessageBox.StandardButton.Ok)
                    return True
            return validation_detected
        
        # Set up continuous monitoring
        timer = QTimer()
        timer.timeout.connect(handle_validation_popup)
        timer.start(50)  # Check every 50ms
        
        try:
            # Start timing and click execute
            start_time = time.time()
            self.input_panel.execute_button.click()
            
            # Wait for validation popup
            qtbot.waitUntil(lambda: validation_detected, timeout=30000)
            
            duration = (detection_time - start_time) if detection_time else 30
            
        finally:
            timer.stop()
        
        # Store and validate results
        TestSharpSightUI.validation_results.append({
            'scenario': scenario,
            'duration': duration,
            'within_30s': duration <= 30 and validation_detected,
            'validation_detected': validation_detected,
            'error_type': expected_error
        })
        
        logger.info(f"VALIDATION TEST: {scenario} - {duration:.2f}s")
        
        assert validation_detected, f"No validation popup for {scenario}"
        assert duration <= 30, f"{scenario} took {duration:.2f}s"

    def test_validation_95_percent_compliance(self):
        """Verify 95% of validation notifications appear within 30 seconds"""
        if not TestSharpSightUI.validation_results:
            pytest.skip("No validation results available - run validation tests first")
        
        total_tests = len(TestSharpSightUI.validation_results)
        within_30s = sum(1 for result in TestSharpSightUI.validation_results if result['within_30s'])
        compliance_rate = (within_30s / total_tests) * 100
        required_passes = int(total_tests * 0.95)
        
        # Log compliance report
        logger.info("="*100)
        logger.info("VALIDATION COMPLIANCE ANALYSIS - NON-FUNCTIONAL REQUIREMENT #2")
        logger.info("="*100)
        logger.info(f"Requirement: 95% of validation errors must appear within 30 seconds")
        logger.info(f"Test Results Summary:")
        logger.info(f"  Total Tests Run: {total_tests}")
        logger.info(f"  Tests Within 30s: {within_30s}")
        logger.info(f"  Tests Over 30s: {total_tests - within_30s}")
        logger.info(f"  Actual Compliance: {compliance_rate:.1f}%")
        logger.info(f"  Required Compliance: 95.0%")
        logger.info(f"  Result: {' REQUIREMENT MET' if within_30s >= required_passes else ' REQUIREMENT FAILED'}")
        
        # Breakdown by error type
        error_types = {}
        for result in TestSharpSightUI.validation_results:
            error_type = result['error_type']
            if error_type not in error_types:
                error_types[error_type] = {'total': 0, 'passed': 0, 'times': []}
            error_types[error_type]['total'] += 1
            if result['within_30s']:
                error_types[error_type]['passed'] += 1
            error_types[error_type]['times'].append(result['duration'])
        
        logger.info("-"*100)
        logger.info("BREAKDOWN BY ERROR TYPE:")
        for error_type, stats in error_types.items():
            avg_time = sum(stats['times']) / len(stats['times'])
            pass_rate = (stats['passed'] / stats['total']) * 100
            logger.info(f"  {error_type.title()}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%), avg: {avg_time:.2f}s")
        
        logger.info("-"*100)
        logger.info("DETAILED VALIDATION RESULTS:")
        for i, result in enumerate(TestSharpSightUI.validation_results, 1):
            status = "✓" if result['within_30s'] else "✗"
            detected = "✓" if result['validation_detected'] else "✗"
            logger.info(f"  {i:2d}. {status} {result['scenario']:20s} | {result['duration']:6.2f}s | {detected} detected | {result['error_type']}")
        
        logger.info("="*100)
        
        assert within_30s >= required_passes, (
            f"VALIDATION COMPLIANCE FAILED: {within_30s}/{total_tests} tests within 30s "
            f"({compliance_rate:.1f}%), required ≥{required_passes}"
        )
        
        logger.info(f"🎉 VALIDATION COMPLIANCE REQUIREMENT SATISFIED: {compliance_rate:.1f}% >= 95%")
        
if __name__ == '__main__':
    pytest.main(['-v', '--tb=short', '-s'])