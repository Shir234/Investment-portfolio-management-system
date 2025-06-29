from PyQt6.QtWidgets import QSplashScreen, QLabel, QVBoxLayout, QWidget, QFrame
from PyQt6.QtGui import QPixmap, QFont, QColor
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QPoint
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect

class SplashScreen(QWidget):  # Changed from QSplashScreen to QWidget
    def __init__(self):
        super().__init__()
        # Remove window frame and stay on top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
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
        scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create welcome message
        self.label = QLabel("Welcome to SharpSight")
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
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
        self.animation.setDuration(2500)  # 1 second
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(lambda: self.finish_fade(next_window))
        self.animation.start()
        
    def finish_fade(self, next_window):
        """Complete the transition by showing the next window directly"""
        self.close()
        next_window.show()

    def finish(self, window):
        """Initiate fade-out transition to the main window"""
        self.fade_out(window)

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the splash screen."""
        self.is_dark_mode = is_dark_mode
        if is_dark_mode:
            container_style = """
                QFrame#splash_container {
                    background-color: #353535;
                    border-radius: 10px;
                    border: 2px solid #555;
                }
            """
            label_style = """
                QLabel {
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                }
            """
        else:
            container_style = """
                QFrame#splash_container {
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    border: 2px solid #ccc;
                }
            """
            label_style = """
                QLabel {
                    color: black;
                    font-size: 24px;
                    font-weight: bold;
                }
            """
        self.findChild(QFrame, "splash_container").setStyleSheet(container_style)
        self.label.setStyleSheet(label_style)