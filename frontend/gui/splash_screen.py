from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QProgressBar, QGraphicsDropShadowEffect
from PyQt6.QtGui import QPixmap, QFont, QColor
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QPoint, pyqtSignal, QEasingCurve
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect
from frontend.logging_config import get_logger
from frontend.utils import resource_path

# Configure logging
logger = get_logger(__name__)

class SplashScreen(QWidget):
    loading_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.is_dark_mode = True
        self.next_window = None
        self.setup_ui()
        logger.info("SplashScreen initialized")

    def setup_ui(self):
        """Set up the UI for the splash screen with vibrant, modern design."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Container frame with vibrant gradient
        container = QFrame()
        container.setObjectName("splash_container")
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(20)
        container_layout.setContentsMargins(30, 30, 30, 30)

        # Background logo (semi-transparent)
        self.bg_logo = QLabel(self)
        bg_pixmap = None
        for logo_path in ["frontend/gui/icons/logo.JPG", "frontend/gui/icons/logo.ico"]:
            full_path = resource_path(logo_path)
            bg_pixmap = QPixmap(full_path)
            if not bg_pixmap.isNull():
                logger.info(f"Background logo loaded from: {full_path}")
                break
        if bg_pixmap and not bg_pixmap.isNull():
            scaled_bg = bg_pixmap.scaled(440, 440, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.bg_logo.setPixmap(scaled_bg)
            self.bg_logo.setStyleSheet("opacity: 0.2;")
            self.bg_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bg_logo.move(30, 30)
        else:
            logger.debug("No background logo loaded")

        # Main logo
        self.logo_label = QLabel()
        pixmap = None
        for logo_path in ["frontend/gui/icons/logo.JPG", "frontend/gui/icons/logo.ico"]:
            full_path = resource_path(logo_path)
            pixmap = QPixmap(full_path)
            if not pixmap.isNull():
                logger.info(f"Main logo loaded from: {full_path}")
                break
        if pixmap is None or pixmap.isNull():
            logger.warning("Logo file not found, using placeholder text")
            self.logo_label.setText("SharpSight")
            self.logo_label.setStyleSheet("""
                font-size: 48px; 
                font-weight: bold; 
                font-family: 'Segoe UI'; 
                color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #BAE6FD, stop:1 #3B82F6);
                padding: 10px;
            """)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(40)
            shadow.setColor(QColor(0, 120, 212, 240))
            shadow.setOffset(0, 0)
            self.logo_label.setGraphicsEffect(shadow)
        else:
            scaled_pixmap = pixmap.scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(40)
            shadow.setColor(QColor(0, 120, 212, 240))
            shadow.setOffset(0, 0)
            self.logo_label.setGraphicsEffect(shadow)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.logo_label)

        # Scale animation for logo
        self.scale_animation = QPropertyAnimation(self.logo_label, b"geometry")
        self.scale_animation.setDuration(900)
        start_rect = self.logo_label.geometry()
        self.scale_animation.setStartValue(start_rect)
        self.scale_animation.setEndValue(start_rect.adjusted(-30, -30, 30, 30))
        self.scale_animation.setLoopCount(-1)
        self.scale_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.scale_animation.start()

        # Welcome message with fade-in
        self.label = QLabel("Welcome to SharpSight")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.opacity_effect = QGraphicsOpacityEffect(self.label)
        self.label.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(2000)
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.start()
        container_layout.addWidget(self.label)

        # Progress bar with pulse animation
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(18)
        self.progress_opacity = QGraphicsOpacityEffect(self.progress_bar)
        self.progress_bar.setGraphicsEffect(self.progress_opacity)
        self.pulse_animation = QPropertyAnimation(self.progress_opacity, b"opacity")
        self.pulse_animation.setDuration(600)
        self.pulse_animation.setStartValue(0.7)
        self.pulse_animation.setEndValue(1.0)
        self.pulse_animation.setLoopCount(-1)
        self.pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse_animation.start()
        container_layout.addWidget(self.progress_bar)

        main_layout.addWidget(container)

        # Set fixed size
        self.setFixedSize(500, 540)

        # Center the splash screen
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        # Apply initial theme
        self.set_theme(self.is_dark_mode)

        # Start loading animation
        self.progress = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)

    def update_progress(self):
        """Simulate loading progress and emit signal when complete."""
        self.progress += 1
        self.progress_bar.setValue(self.progress)
        if self.progress >= 100:
            self.timer.stop()
            self.scale_animation.stop()
            self.pulse_animation.stop()
            self.loading_completed.emit()
            logger.debug("Loading animation completed")
        else:
            logger.debug(f"Progress updated: {self.progress}%")

    def set_theme(self, is_dark_mode):
        """Apply light or dark theme to the splash screen."""
        self.is_dark_mode = is_dark_mode
        container_style = f"""
            QFrame#splash_container {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 {'#1E40AF' if is_dark_mode else '#93C5FD'},
                    stop:1 {'#2563EB' if is_dark_mode else '#DBEAFE'}
                );
                border-radius: 20px;
                border: 2px solid {'#2563EB' if is_dark_mode else '#93C5FD'};
            }}
        """
        label_style = f"""
            QLabel {{
                color: {'#BAE6FD' if is_dark_mode else '#1E40AF'};
                font-size: 28px;
                font-weight: bold;
                font-family: 'Segoe UI';
                padding: 10px;
            }}
        """
        progress_style = f"""
            QProgressBar {{
                border: 1px solid {'#2563EB' if is_dark_mode else '#93C5FD'};
                border-radius: 9px;
                background-color: {'#1E3A8A' if is_dark_mode else '#E0E7FF'};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563EB,
                    stop:1 #60A5FA
                );
                border-radius: 8px;
                margin: 0.5px;
            }}
        """
        self.findChild(QFrame, "splash_container").setStyleSheet(container_style)
        self.label.setStyleSheet(label_style)
        self.progress_bar.setStyleSheet(progress_style)
        logger.debug(f"Applied theme: {'dark' if is_dark_mode else 'light'}")

    def fade_out(self, next_window):
        """Create a fade-out animation for transitioning to the next window."""
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(lambda: self.finish_fade(next_window))
        self.animation.start()
        logger.debug("Started fade-out animation")

    def finish_fade(self, next_window):
        """Complete the transition by closing the splash screen and showing the next window."""
        self.close()
        next_window.show()
        logger.info("Splash screen closed, main window shown")

    def finish(self, window):
        """Store the next window and wait for loading completion."""
        self.next_window = window
        if self.progress >= 100:
            self.fade_out(window)
        else:
            self.loading_completed.connect(lambda: self.fade_out(window))