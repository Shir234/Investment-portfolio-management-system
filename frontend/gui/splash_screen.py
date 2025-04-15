from PyQt5.QtWidgets import QSplashScreen, QLabel, QVBoxLayout, QWidget, QFrame
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QPoint
from PyQt5.QtWidgets import QApplication, QGraphicsOpacityEffect


class SplashScreen(QWidget):  # Changed from QSplashScreen to QWidget
    def __init__(self):
        super().__init__()
        # Remove window frame and stay on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create frame for the window effect
        container = QFrame()
        container.setObjectName("splash_container")
        container.setStyleSheet("""
            QFrame#splash_container {
                background-color: #353535;
                border-radius: 10px;
                border: 2px solid #555;
            }
        """)
        
        # Create layout for the frame content
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(20)
        container_layout.setContentsMargins(30, 30, 30, 30)
        
        # Load and scale the logo
        logo_label = QLabel()
        pixmap = QPixmap("logo.JPG")
        scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        
        # Create welcome message
        self.label = QLabel("Welcome to SharpSight")
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        self.label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layouts
        container_layout.addWidget(logo_label)
        container_layout.addWidget(self.label)
        main_layout.addWidget(container)
        
        # Set fixed size for the window
        self.setFixedSize(400, 500)
        
        # Center the splash screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
        
    def fade_out(self, next_window):
        """Create a fade out animation"""
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(1000)  # 1 second
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(lambda: self.finish_fade(next_window))
        self.animation.start()
        
    def finish_fade(self, next_window):
        """Complete the transition by showing the next window directly"""
        self.close()
        
        # Simply show the window without fade effect
        # This will ensure the window displays properly
        next_window.show()