from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QDoubleSpinBox, QCalendarWidget, QPushButton,
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QDate
from trading_logic import get_portfolio_history, get_orders
from data.trading_connector import execute_trading_strategy
import logging
from datetime import date

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class InputPanel(QWidget):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.data_manager = data_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        risk_group = QGroupBox("Risk Level")
        risk_group.setStyleSheet("color: #ffffff;")
        risk_layout = QVBoxLayout()
        self.risk_slider = QSlider(Qt.Horizontal)
        self.risk_slider.setMinimum(1)
        self.risk_slider.setMaximum(10)
        self.risk_slider.setValue(1)  # Default to conservative for testing
        self.risk_slider.setSingleStep(1)
        self.risk_slider.setPageStep(1)
        self.risk_slider.setTickPosition(QSlider.TicksBelow)
        self.risk_slider.setTickInterval(1)
        self.risk_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3c3f41;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff4500;
                border: 1px solid #ffffff;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #555555;
            }
        """)
        self.risk_label = QLabel("Risk Level: 1 (Very Conservative)")
        self.risk_label.setStyleSheet("color: #ffffff;")
        self.risk_slider.valueChanged.connect(self.update_risk_label)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_slider)
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        amount_group = QGroupBox("Investment Amount")
        amount_group.setStyleSheet("color: #ffffff;")
        amount_layout = QHBoxLayout()
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(100, 1000000)
        self.amount_spin.setValue(10000)
        self.amount_spin.setPrefix("$")
        self.amount_spin.setSingleStep(1000)
        self.amount_spin.setStyleSheet("QDoubleSpinBox { background-color: #3c3f41; color: #ffffff; border: 1px solid #555555; }")
        amount_layout.addWidget(QLabel("Amount:"))
        amount_layout.addWidget(self.amount_spin)
        amount_group.setLayout(amount_layout)
        layout.addWidget(amount_group)
        window_group = QGroupBox("Investment Window")
        window_group.setStyleSheet("color: #ffffff;")
        window_layout = QVBoxLayout()
        if self.data_manager and self.data_manager.data is not None and not self.data_manager.data.empty:
            try:
                min_date, max_date = self.data_manager.get_date_range()
                if min_date and max_date:
                    date_range_label = QLabel(f"Available trading dates: {min_date.strftime('%d/%m/%Y')} to {max_date.strftime('%d/%m/%Y')}")
                    date_range_label.setStyleSheet("color: #ffffff; font-style: italic;")
                    window_layout.addWidget(date_range_label)
                else:
                    date_range_label = QLabel("No data available to determine trading dates.")
                    date_range_label.setStyleSheet("color: red; font-style: italic;")
                    window_layout.addWidget(date_range_label)
            except Exception as e:
                logging.error(f"Error getting date range: {e}")
                date_range_label = QLabel(f"Error loading date range: {str(e)}")
                date_range_label.setStyleSheet("color: red; font-style: italic;")
                window_layout.addWidget(date_range_label)
                min_date, max_date = None, None
        else:
            date_range_label = QLabel("Data manager not initialized or data is empty.")
            date_range_label.setStyleSheet("color: red; font-style: italic;")
            window_layout.addWidget(date_range_label)
            min_date, max_date = None, None
        start_label = QLabel("Start Date:")
        self.start_calendar = QCalendarWidget()
        self.start_calendar.setStyleSheet("QCalendarWidget { background-color: #2b2b2b; color: #ffffff; } QCalendarWidget QToolButton { color: #ffffff; }")
        end_label = QLabel("End Date:")
        self.end_calendar = QCalendarWidget()
        self.end_calendar.setStyleSheet("QCalendarWidget { background-color: #2b2b2b; color: #ffffff; } QCalendarWidget QToolButton { color: #ffffff; }")
        if min_date and max_date:
            min_qdate = QDate(min_date.year, min_date.month, min_date.day)
            max_qdate = QDate(max_date.year, max_date.month, max_date.day)
            self.start_calendar.setMinimumDate(min_qdate)
            self.start_calendar.setMaximumDate(max_qdate)
            self.start_calendar.setSelectedDate(QDate(2021, 10, 18))  # Default to user-preferred start
            self.end_calendar.setMinimumDate(min_qdate)
            self.end_calendar.setMaximumDate(max_qdate)
            self.end_calendar.setSelectedDate(QDate(2021, 12, 31))  # Default to user-preferred end
        else:
            default_min_date = QDate(2021, 1, 1)
            default_max_date = QDate(2023, 12, 31)
            self.start_calendar.setMinimumDate(default_min_date)
            self.start_calendar.setMaximumDate(default_max_date)
            self.start_calendar.setSelectedDate(QDate(2021, 10, 18))
            self.end_calendar.setMinimumDate(default_min_date)
            self.end_calendar.setMaximumDate(default_max_date)
            self.end_calendar.setSelectedDate(QDate(2021, 12, 31))
        window_layout.addWidget(start_label)
        window_layout.addWidget(self.start_calendar)
        window_layout.addWidget(end_label)
        window_layout.addWidget(self.end_calendar)
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        self.portfolio_value_label = QLabel("Current Portfolio Value: N/A")
        self.portfolio_value_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        layout.addWidget(self.portfolio_value_label)
        self.update_button = QPushButton("Update Portfolio")
        self.update_button.clicked.connect(self.update_portfolio)
        layout.addWidget(self.update_button)
        
    def update_risk_label(self, value):
        risk_levels = {
            1: "Very Conservative", 2: "Conservative", 3: "Moderately Conservative",
            4: "Moderate", 5: "Moderately Aggressive", 6: "Aggressive",
            7: "Moderately Aggressive", 8: "Very Aggressive", 9: "Extremely Aggressive",
            10: "Maximum Risk"
        }
        self.risk_label.setText(f"Risk Level: {value} ({risk_levels[value]})")
        
    def update_portfolio(self):
        try:
            if self.data_manager.data is None or self.data_manager.data.empty:
                raise ValueError("No data loaded. Please ensure a valid CSV file is selected.")
            investment_amount = self.amount_spin.value()
            risk_level = self.risk_slider.value()
            start_qdate = self.start_calendar.selectedDate()
            end_qdate = self.end_calendar.selectedDate()
            start_date = date(start_qdate.year(), start_qdate.month(), start_qdate.day())
            end_date = date(end_qdate.year(), end_qdate.month(), end_qdate.day())
            logging.info(f"Selected UI date range: {start_date} to {end_date}")
            min_date, max_date = self.data_manager.get_date_range()
            # Warn if signals are excluded
            signals_after_end = len(self.data_manager.data[
                (self.data_manager.data['date'].dt.date > end_date) & 
                (self.data_manager.data['Best_Prediction'] >= 0.5)
            ])
            if signals_after_end > 0:
                QMessageBox.warning(self, "Narrow Date Range",
                                   f"The selected end date ({end_date}) excludes {signals_after_end} "
                                   f"signals with Best_Prediction >= 0.5. Consider extending to {max_date.date()}.")
            # Warn if date range is short
            if (end_date - start_date).days < 365:
                logging.warning(f"Date range {start_date} to {end_date} is less than 1 year.")
                QMessageBox.warning(self, "Short Date Range",
                                   "The selected date range is less than 1 year. Some strategies may require a longer period.")
            mode = "automatic"
            success, suggestions = execute_trading_strategy(
                investment_amount=investment_amount,
                risk_level=risk_level,
                start_date=start_date,
                end_date=end_date,
                data_manager=self.data_manager,
                mode=mode,
                reset_state=True
            )
            if not success:
                raise ValueError("Trading strategy execution failed. Check risk level or date range.")
            portfolio_history = get_portfolio_history()
            orders = get_orders()
            if portfolio_history:
                latest_value = portfolio_history[-1]['portfolio_value']
                self.portfolio_value_label.setText(f"Current Portfolio Value: ${latest_value:,.2f}")
            else:
                self.portfolio_value_label.setText("Current Portfolio Value: $0.00")
            if orders and len(orders) < 5 and risk_level <= 3:
                QMessageBox.warning(self, "Few Trades Generated",
                                   f"Only {len(orders)} trades generated for conservative strategy. "
                                   "The prediction model may have insufficient positive signals. "
                                   "Consider increasing risk level or retraining the model.")
            elif not orders:
                QMessageBox.warning(self, "No Trades Generated",
                                   "No trades were generated. Try increasing the risk level, "
                                   "extending the date range, or retraining the prediction model.")
            if mode == "semi-automatic":
                self.main_window.suggestions = suggestions
            self.main_window.update_dashboard()
            self.main_window.update_recommendations()
        except Exception as e:
            logging.error(f"Error in update_portfolio: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update portfolio: {str(e)}")