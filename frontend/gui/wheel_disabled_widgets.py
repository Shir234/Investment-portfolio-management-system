# wheel_disabled_widgets.py
"""
Custom widgets that ignore mouse wheel events to prevent accidental value changes
"""

from PyQt6.QtWidgets import QSpinBox, QDateEdit, QComboBox
from PyQt6.QtGui import QWheelEvent

class WheelDisabledSpinBox(QSpinBox):
    """QSpinBox that ignores mouse wheel events."""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class WheelDisabledDateEdit(QDateEdit):
    """QDateEdit that ignores mouse wheel events."""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class WheelDisabledComboBox(QComboBox):
    """QComboBox that ignores mouse wheel events."""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental value changes."""
        event.ignore()


class WheelEventFilter:
    """Event filter that can be installed on any widget to block wheel events."""
    
    def __init__(self, widget):
        self.widget = widget
        self.widget.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """
        Filter out wheel events from monitored widgets while preserving other interactions.
        Returns True to block wheel events, False to allow all other event types.
        """
        if event.type() == event.Type.Wheel:
            return True  # Block the event
        return False  # Allow other events