from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QSpinBox, QCalendarWidget, QPushButton,
                             QGroupBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QDate
from datetime import datetime

class InputPanel(QWidget):
    def __init__(self, data_manager):
        super().__init__()
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
        self.risk_slider.setTickPosition(QSlider.TicksBelow)
        self.risk_slider.setTickInterval(1)
        self.risk_slider.setStyleSheet("QSlider::groove:horizontal { background: #3c3f41; } QSlider::handle:horizontal { background: #ffffff; }")
        
        self.risk_label = QLabel("Risk Level: 5 (Moderate)")
        self.risk_label.setStyleSheet("color: #ffffff;")
        self.risk_slider.valueChanged.connect(self.update_risk_label)
        
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_slider)
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # Time Window Selection
        time_group = QGroupBox("Investment Time Window")
        time_group.setStyleSheet("color: #ffffff;")
        time_layout = QVBoxLayout()
        
        self.calendar = QCalendarWidget()
        self.calendar.setStyleSheet("QCalendarWidget { background-color: #2b2b2b; color: #ffffff; } QCalendarWidget QToolButton { color: #ffffff; }")
        min_date, max_date = self.data_manager.get_date_range()
        if min_date and max_date:
            self.calendar.setMinimumDate(QDate(min_date.year, min_date.month, min_date.day))
            self.calendar.setMaximumDate(QDate(max_date.year, max_date.month, max_date.day))
        
        time_layout.addWidget(self.calendar)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
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
        
        # Update Button
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
        risk_level = self.risk_slider.value()
        selected_date = self.calendar.selectedDate().toPyDate()
        investment_amount = self.amount_spin.value()
        self.parent().update_dashboard()