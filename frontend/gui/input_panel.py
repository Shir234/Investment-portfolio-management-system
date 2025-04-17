from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QSpinBox, QCalendarWidget, QPushButton,
                             QGroupBox, QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QDate
from datetime import datetime

from data.trading_connector import execute_trading_strategy

class InputPanel(QWidget):
    def __init__(self, parent=None, data_manager=None):
        super().__init__(parent)
        self.main_window = parent  # Store reference to MainWindow
        self.data_manager = data_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Risk Level Input
        risk_group = QGroupBox("Risk Level")
        risk_group.setStyleSheet("color: #ffffff;")
        risk_layout = QVBoxLayout()
        
        self.risk_slider = QSlider(Qt.Horizontal)
        self.risk_slider.setMinimum(1)
        self.risk_slider.setMaximum(10)
        self.risk_slider.setValue(5)
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
        
        self.risk_label = QLabel("Risk Level: 5 (Moderate)")
        self.risk_label.setStyleSheet("color: #ffffff;")
        self.risk_slider.valueChanged.connect(self.update_risk_label)
        
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_slider)
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # Investment Amount
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
        
        # Investment Window Selection
        window_group = QGroupBox("Investment Window")
        window_group.setStyleSheet("color: #ffffff;")
        window_layout = QVBoxLayout()
        
        # Get the date range from data_manager
        min_date, max_date = self.data_manager.get_date_range()
        
        # Add a label showing the available date range
        if min_date and max_date:
            date_range_label = QLabel(f"Available trading dates: {min_date.strftime('%d/%m/%Y')} to {max_date.strftime('%d/%m/%Y')}")
            date_range_label.setStyleSheet("color: #ffffff; font-style: italic;")
            window_layout.addWidget(date_range_label)
        else:
            date_range_label = QLabel("No data available to determine trading dates.")
            date_range_label.setStyleSheet("color: red; font-style: italic;")
            window_layout.addWidget(date_range_label)
            
        # Start Date Calendar
        start_label = QLabel("Start Date:")
        self.start_calendar = QCalendarWidget()
        self.start_calendar.setStyleSheet("QCalendarWidget { background-color: #2b2b2b; color: #ffffff; } QCalendarWidget QToolButton { color: #ffffff; }")
        
        # End Date Calendar
        end_label = QLabel("End Date:")
        self.end_calendar = QCalendarWidget()
        self.end_calendar.setStyleSheet("QCalendarWidget { background-color: #2b2b2b; color: #ffffff; } QCalendarWidget QToolButton { color: #ffffff; }")
        
        # Set date limits and defaults
        if min_date and max_date:
            min_qdate = QDate(min_date.year, min_date.month, min_date.day)
            max_qdate = QDate(max_date.year, max_date.month, max_date.day)
            self.start_calendar.setMinimumDate(min_qdate)
            self.start_calendar.setMaximumDate(max_qdate)
            self.start_calendar.setSelectedDate(min_qdate)  # Default to earliest date
            self.end_calendar.setMinimumDate(min_qdate)
            self.end_calendar.setMaximumDate(max_qdate)
            self.end_calendar.setSelectedDate(max_qdate)  # Default to latest date
        
        window_layout.addWidget(start_label)
        window_layout.addWidget(self.start_calendar)
        window_layout.addWidget(end_label)
        window_layout.addWidget(self.end_calendar)
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        # Update Button (unchanged)
        self.update_button = QPushButton("Update Portfolio")
        self.update_button.clicked.connect(self.update_portfolio)
        layout.addWidget(self.update_button)
        
    def update_risk_label(self, value):
        risk_levels = {
            1: "Very Conservative",
            2: "Conservative",
            3: "Moderately Conservative",
            4: "Moderate",
            5: "Moderately Aggressive",
            6: "Aggressive",
            7: "Moderately Aggressive",
            8: "Very Aggressive",
            9: "Extremely Aggressive",
            10: "Maximum Risk"
        }
        self.risk_label.setText(f"Risk Level: {value} ({risk_levels[value]})")
        
    def update_portfolio(self):
        investment_amount = self.amount_spin.value()
        risk_level = self.risk_slider.value() * 10
        start_date = self.start_calendar.selectedDate().toPyDate()
        end_date = self.end_calendar.selectedDate().toPyDate()
        
        if end_date < start_date:
            QMessageBox.critical(self, "Invalid Date Range", "End date cannot be earlier than start date.")
            return
        
        success = execute_trading_strategy(
            investment_amount,
            risk_level,
            start_date,
            end_date,
            merged_data=self.data_manager.data
        )
        if success:
            self.main_window.update_dashboard()
        else:
            QMessageBox.critical(self, "Error", 
                "Failed to execute trading strategy. Possible reasons:\n"
                "- No data available for the selected date range.\n"
                "- Risk level might be too restrictive.\n"
                "- Investment amount might be too low.\n"
                "Check the console for more details and adjust your parameters.")