from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QApplication, QGraphicsOpacityEffect
from PyQt6.QtGui import QPixmap, QFont, QColor
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QPoint
from frontend.gui.styles import ModernStyles

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        # Remove window frame and stay on top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create modern container frame
        self.container = QFrame()
        self.container.setObjectName("splash_container")
        
        # Create layout for the frame content
        container_layout = QVBoxLayout(self.container)
        container_layout.setSpacing(30)
        container_layout.setContentsMargins(40, 40, 40, 40)
        
        # Load and scale the logo
        logo_label = QLabel()
        try:
            pixmap = QPixmap("logo.JPG")
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                # Fallback if logo not found
                logo_label.setText("📊")
                logo_label.setStyleSheet("font-size: 80px;")
        except Exception:
            # Fallback emoji logo
            logo_label.setText("📊")
            logo_label.setStyleSheet("font-size: 80px;")
        
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create welcome message
        self.title_label = QLabel("SharpSight")
        self.title_label.setProperty("class", "title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create subtitle
        self.subtitle_label = QLabel("Investment Portfolio Management System")
        self.subtitle_label.setProperty("class", "subtitle")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Loading message
        self.loading_label = QLabel("Loading...")
        self.loading_label.setProperty("class", "caption")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add widgets to layouts
        container_layout.addWidget(logo_label)
        container_layout.addWidget(self.title_label)
        container_layout.addWidget(self.subtitle_label)
        container_layout.addWidget(self.loading_label)
        
        main_layout.addWidget(self.container)
        
        # Set fixed size for the window
        self.setFixedSize(450, 550)
        
        # Apply modern styling
        self.apply_modern_style()
        
        # Center the splash screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

        self.raise_()
        self.activateWindow()
        
    def apply_modern_style(self, is_dark=True):
        """Apply modern styling to the splash screen."""
        colors = ModernStyles.COLORS['dark'] if is_dark else ModernStyles.COLORS['light']
        
        # Container styling with modern design
        container_style = f"""
            QFrame#splash_container {{
                background-color: {colors['primary']};
                border: 1px solid {colors['border']};
                border-radius: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {colors['primary']}, 
                    stop:1 {colors['secondary']});
            }}
        """
        
        # Apply complete modern styles
        complete_style = (
            ModernStyles.get_complete_style(is_dark) + 
            container_style
        )
        self.setStyleSheet(complete_style)
        
        # Additional specific styling for labels
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['text_primary']};
                font-size: 32px;
                font-weight: 700;
                margin: 10px 0;
            }}
        """)
        
        self.subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['text_secondary']};
                font-size: 16px;
                font-weight: 500;
                margin: 5px 0;
            }}
        """)
        
        self.loading_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['accent']};
                font-size: 14px;
                font-weight: 600;
                margin: 15px 0;
            }}
        """)
        
    def fade_out(self, next_window):
        """Create a fade out animation with modern easing"""
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(2000)  # Smooth 1.5 second fade
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(lambda: self.finish_fade(next_window))
        self.animation.start()
        
    def finish_fade(self, next_window):
        """Complete the transition by showing the next window"""
        self.close()
        next_window.show()

    def finish(self, window):
        """Initiate fade-out transition to the main window"""
        self.fade_out(window)

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the splash screen."""
        self.apply_modern_style(is_dark_mode)