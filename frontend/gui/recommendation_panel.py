from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QMessageBox)
from PyQt5.QtCore import Qt

class RecommendationPanel(QWidget):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Trading Recommendations")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Recommendations table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Stock", "Action", "Price", "Sharpe Ratio", "Allocation"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("QTableWidget { background-color: #2b2b2b; color: #ffffff; gridline-color: #555555; } QHeaderView::section { background-color: #3c3f41; color: #ffffff; }")
        layout.addWidget(self.table)
        
        # Alert message
        self.alert_label = QLabel()
        self.alert_label.setStyleSheet("color: red;")
        self.alert_label.setVisible(False)
        layout.addWidget(self.alert_label)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Recommendations")
        refresh_button.clicked.connect(self.update_recommendations)
        layout.addWidget(refresh_button)
        
    def update_recommendations(self):
        risk_level = self.parent().input_panel.risk_slider.value()
        recommendations = self.data_manager.get_recommendations(
            risk_level=risk_level,
            investment_amount=self.parent().input_panel.amount_spin.value()
        )
        
        if recommendations.empty:
            self.show_no_recommendations_alert(risk_level)
            return
            
        self.table.setRowCount(0)
        for idx, row in recommendations.iterrows():
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(row['stock'])))
            self.table.setItem(row_position, 1, QTableWidgetItem(row['action']))
            self.table.setItem(row_position, 2, QTableWidgetItem(f"${row['price']:,.2f}"))
            self.table.setItem(row_position, 3, QTableWidgetItem(f"{row['daily_sharpe']:.2f}"))
            self.table.setItem(row_position, 4, QTableWidgetItem(f"{row['allocation']:.1%}"))
        self.table.resizeColumnsToContents()
        
    def show_no_recommendations_alert(self, current_risk_level):
        self.alert_label.setText(f"No recommendations available at risk level {current_risk_level}. Consider adjusting your risk tolerance.")
        self.alert_label.setVisible(True)
        for risk in range(1, current_risk_level):
            if not self.data_manager.get_recommendations(risk_level=risk).empty:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Alternative Risk Level Available")
                msg.setText(f"Recommendations are available at risk level {risk}. Would you like to view those instead?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                
                # Apply dark style to message box
                msg.setStyleSheet("""
                    QMessageBox {
                        background-color: #353535;
                        color: white;
                    }
                    QLabel {
                        color: white;
                    }
                    QPushButton {
                        background-color: #444;
                        color: white;
                        border: 1px solid #666;
                        padding: 5px 15px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #555;
                    }
                """)
                
                if msg.exec_() == QMessageBox.Yes:
                    self.parent().input_panel.risk_slider.setValue(risk)
                    self.update_recommendations()
                break