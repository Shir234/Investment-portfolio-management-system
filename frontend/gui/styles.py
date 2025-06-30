# styles.py - Optimized UI styling for compact, no-scroll layout

class ModernStyles:
    """Modern, compact styling optimized for single-window display"""
    
    # Color Palette (unchanged)
    COLORS = {
        'dark': {
            'primary': '#1E1E2E',           # Deep dark background
            'secondary': '#2A2A3E',        # Slightly lighter panels
            'surface': '#363650',          # Card/widget surfaces
            'accent': '#6366F1',           # Modern indigo accent
            'accent_hover': '#7C3AED',     # Purple hover state
            'accent_pressed': '#5B21B6',   # Darker pressed state
            'success': '#10B981',          # Green for positive actions
            'warning': '#F59E0B',          # Amber for warnings
            'danger': '#EF4444',           # Red for destructive actions
            'text_primary': '#F8FAFC',     # Main text
            'text_secondary': '#94A3B8',   # Secondary text
            'text_muted': '#64748B',       # Muted text
            'border': '#475569',           # Border color
            'border_light': '#334155',     # Lighter borders
            'hover': '#4338CA',            # Hover overlay
            'selected': '#6366F1',         # Selection color
        },
        'light': {
            'primary': '#D1D5DB',          # Even darker background - medium-dark gray
            'secondary': '#B0B7C3',        # Much darker panels 
            'surface': '#C4C7CF',          # Much darker card surfaces
            'accent': '#6366F1',           # Same indigo accent for consistency
            'accent_hover': '#7C3AED',     # Purple hover
            'accent_pressed': '#5B21B6',   # Darker pressed
            'success': '#10B981',          # Green
            'warning': '#F59E0B',          # Amber
            'danger': '#EF4444',           # Red
            'text_primary': '#111827',     # Very dark text for great contrast
            'text_secondary': '#374151',   # Dark gray text
            'text_muted': '#6B7280',       # Medium muted text
            'border': '#9CA3AF',           # Darker, more defined borders
            'border_light': '#B0B7C3',     # Medium border
            'hover': '#A8ACB8',            # Much darker hover state
            'selected': '#DBEAFE',         # Subtle blue selection
        }
    }
    
    @classmethod
    def get_main_window_style(cls, is_dark=True):
        """Compact main window styling optimized for no-scroll layout"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        # For light mode, keep some elements darker for contrast and visual hierarchy
        if not is_dark:
            tab_bg = '#374151'  # Even darker tabs in light mode
            tab_text = '#F9FAFB'  # Light text on dark tabs
            tab_selected_bg = colors['surface']  # Light background for selected tab
            tab_selected_text = colors['text_primary']  # Dark text on light background
        else:
            tab_bg = 'transparent'
            tab_text = colors['text_secondary']
            tab_selected_bg = colors['surface']
            tab_selected_text = colors['accent']
        
        return f"""
        QMainWindow {{
            background-color: {colors['primary']};
            color: {colors['text_primary']};
            font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            font-size: 12px;  /* Reduced base font size */
            line-height: 1.4;  /* Tighter line height */
        }}
        
        /* Central Widget and Containers */
        QWidget {{
            background-color: transparent;
            color: {colors['text_primary']};
        }}
        
        QMainWindow > QWidget {{
            background-color: {colors['primary']};
        }}
        
        /* Compact Tab Widget */
        QTabWidget::pane {{
            border: 1px solid {colors['border_light']};
            background-color: {colors['surface']};
            border-radius: 8px;  /* Reduced radius */
            margin-top: 6px;  /* Reduced margin */
        }}
        
        QTabBar::tab {{
            background-color: {tab_bg};
            color: {tab_text};
            padding: 8px 16px;  /* Reduced padding */
            margin-right: 2px;  /* Reduced margin */
            border-radius: 6px 6px 0 0;  /* Smaller radius */
            font-weight: 500;
            font-size: 12px;  /* Smaller font */
            {"border: 1px solid #4A5568;" if not is_dark else ""}
            min-height: 12px;  /* Compact height */
        }}
        
        QTabBar::tab:selected {{
            background-color: {tab_selected_bg};
            color: {tab_selected_text};
            font-weight: 600;
            border-bottom: 2px solid {colors['accent']};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {colors['secondary'] if is_dark else '#4A5568'};
            color: {colors['text_primary'] if is_dark else '#F7FAFC'};
        }}
        """
    
    @classmethod
    def get_button_styles(cls, is_dark=True):
        """Compact button styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        /* Primary Button */
        QPushButton {{
            background-color: {colors['accent']};
            color: white;
            border: none;
            border-radius: 6px;  /* Smaller radius */
            padding: 8px 16px;  /* Reduced padding */
            font-weight: 600;
            font-size: 12px;  /* Smaller font */
            min-height: 16px;  /* Reduced height */
        }}
        
        QPushButton:hover {{
            background-color: {colors['accent_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {colors['accent_pressed']};
        }}
        
        QPushButton:disabled {{
            background-color: {colors['text_muted']};
            color: {colors['text_secondary']};
        }}
        
        /* Secondary Button */
        QPushButton[class="secondary"] {{
            background-color: {colors['secondary']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
        }}
        
        QPushButton[class="secondary"]:hover {{
            background-color: {colors['hover']};
            border-color: {colors['accent']};
        }}
        
        /* Danger Button */
        QPushButton[class="danger"] {{
            background-color: {colors['danger']};
            color: white;
        }}
        
        QPushButton[class="danger"]:hover {{
            background-color: #DC2626;
        }}
        
        /* Success Button */
        QPushButton[class="success"] {{
            background-color: {colors['success']};
            color: white;
        }}
        
        QPushButton[class="success"]:hover {{
            background-color: #059669;
        }}
        
        /* Compact Theme Toggle Button */
        QPushButton[class="theme-toggle"] {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 16px;  /* Smaller radius */
            padding: 6px 12px;  /* Reduced padding */
            font-size: 11px;  /* Smaller font */
            font-weight: 500;
            min-width: 60px;  /* Reduced width */
        }}
        
        QPushButton[class="theme-toggle"]:hover {{
            background-color: {colors['hover']};
            border-color: {colors['accent']};
        }}
        """
    
    @classmethod
    def get_input_styles(cls, is_dark=True):
        """Compact input field styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        /* Compact Text Inputs */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;  /* Smaller radius */
            padding: 6px 10px;  /* Reduced padding */
            font-size: 12px;  /* Smaller font */
            min-width: 120px;  /* Reduced width */
            min-height: 14px;  /* Reduced height */
            selection-background-color: {colors['accent']};
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {colors['accent']};
            outline: none;
        }}
        
        QLineEdit::placeholder {{
            color: {colors['text_muted']};
        }}
        
        /* Compact Date Inputs */
        QDateEdit {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;
            padding: 6px 10px;  /* Reduced padding */
            font-size: 12px;  /* Smaller font */
            min-width: 120px;  /* Reduced width */
            min-height: 14px;  /* Reduced height */
        }}
        
        QDateEdit:focus {{
            border: 2px solid {colors['accent']};
        }}
        
        QDateEdit::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 16px;  /* Smaller dropdown */
            border-left: 1px solid {colors['border']};
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
            background-color: {colors['secondary']};
        }}
        
        QDateEdit::drop-down:hover {{
            background-color: {colors['accent']};
        }}
        
        QDateEdit::down-arrow {{
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-top: 5px solid {colors['text_primary']};
            width: 0px;
            height: 0px;
        }}
        
        /* Compact Combo Boxes */
        QComboBox {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;
            padding: 6px 10px;  /* Reduced padding */
            font-size: 12px;  /* Smaller font */
            min-width: 120px;  /* Reduced width */
            min-height: 14px;  /* Reduced height */
        }}
        
        QComboBox:focus {{
            border: 2px solid {colors['accent']};
        }}
        
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 16px;  /* Smaller dropdown */
            border-left: 1px solid {colors['border']};
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
            background-color: {colors['secondary']};
        }}
        
        QComboBox::drop-down:hover {{
            background-color: {colors['accent']};
        }}
        
        QComboBox::down-arrow {{
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-top: 5px solid {colors['text_primary']};
            width: 0px;
            height: 0px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
            selection-background-color: {colors['accent']};
            outline: none;
        }}
        
        /* SpinBox compact buttons */
        QSpinBox::up-button, QSpinBox::down-button {{
            border: none;
            background-color: {colors['secondary']};
            width: 14px;  /* Smaller buttons */
        }}
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background-color: {colors['accent']};
        }}
        
        QSpinBox::up-arrow {{
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-bottom: 5px solid {colors['text_primary']};
            width: 0px;
            height: 0px;
        }}
        
        QSpinBox::down-arrow {{
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-top: 5px solid {colors['text_primary']};
            width: 0px;
            height: 0px;
        }}
        """
    
    @classmethod
    def get_table_styles(cls, is_dark=True):
        """Compact table styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QTableWidget {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;  /* Smaller radius */
            gridline-color: {colors['border_light']};
            font-size: 11px;  /* Smaller font */
            selection-background-color: {colors['selected']};
            alternate-background-color: {'#2A2A3E' if is_dark else '#E5E7EB'};
        }}
        
        QTableWidget::item {{
            border: none;
            padding: 8px 6px;  /* Reduced padding */
            border-bottom: 1px solid {colors['border_light']};
        }}
        
        QTableWidget::item:selected {{
            background-color: {colors['selected']};
            color: {colors['text_primary']};
        }}
        
        QTableWidget::item:hover {{
            background-color: {colors['hover']};
        }}
        
        QHeaderView::section {{
            background-color: {colors['secondary']};
            color: {colors['text_primary']};
            border: none;
            border-bottom: 2px solid {colors['border']};
            padding: 10px 6px;  /* Reduced padding */
            font-weight: 600;
            font-size: 11px;  /* Smaller font */
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        QHeaderView::section:hover {{
            background-color: {colors['hover']};
        }}
        """
    
    @classmethod
    def get_card_styles(cls, is_dark=True):
        """Compact card/panel styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        /* Compact Group Boxes */
        QGroupBox {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;  /* Smaller radius */
            margin: 4px 0;  /* Reduced margin */
            padding-top: 16px;  /* Reduced padding */
            font-weight: 600;
            font-size: 12px;  /* Smaller font */
            color: {colors['text_primary']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 10px;  /* Reduced padding */
            background-color: {colors['surface']};
            border: 1px solid {colors['border_light']};
            border-radius: 4px;  /* Smaller radius */
            color: {colors['accent']};
            font-weight: 600;
            font-size: 11px;  /* Smaller font */
        }}
        
        /* Compact List Widgets */
        QListWidget {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;
            padding: 2px;  /* Reduced padding */
            font-size: 11px;  /* Smaller font */
        }}
        
        QListWidget::item {{
            border-radius: 4px;
            padding: 6px 8px;  /* Reduced padding */
            margin: 1px;
        }}
        
        QListWidget::item:selected {{
            background-color: {colors['accent']};
            color: white;
        }}
        
        QListWidget::item:hover {{
            background-color: {colors['hover']};
        }}
        """
    
    @classmethod
    def get_label_styles(cls, is_dark=True):
        """Compact label and text styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QLabel {{
            color: {colors['text_primary']};
            font-size: 12px;  /* Smaller font */
            font-weight: 500;
            background-color: transparent;
        }}
        
        QLabel[class="label"] {{
            color: {colors['text_primary']};
            font-size: 13px;  /* Slightly bigger but still compact */
            font-weight: 600;
            margin-bottom: 4px;  /* Reduced margin */
            background-color: transparent;
            border: none;
        }}
        
        QLabel[class="title"] {{
            font-size: 20px;  /* Reduced from 24px */
            font-weight: 700;
            color: {colors['text_primary']};
            margin: 8px 0;  /* Reduced margin */
            background-color: transparent;
        }}
        
        QLabel[class="subtitle"] {{
            font-size: 14px;  /* Reduced from 18px */
            font-weight: 600;
            color: {colors['text_secondary']};
            margin: 6px 0;  /* Reduced margin */
            background-color: transparent;
        }}
        
        QLabel[class="caption"] {{
            font-size: 10px;  /* Smaller font */
            font-weight: 400;
            color: {colors['text_muted']};
            margin: 2px 0;  /* Reduced margin */
            background-color: transparent;
        }}
        
        QLabel[class="metric"] {{
            font-size: 14px;  /* Slightly reduced */
            font-weight: 600;
            padding: 8px 12px;  /* Reduced padding */
            background-color: {colors['surface']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;  /* Smaller radius */
            color: {colors['text_primary']};
        }}
        
        QLabel[class="metric-success"] {{
            color: {colors['success']};
            border-left: 3px solid {colors['success']};  /* Thinner border */
        }}
        
        QLabel[class="metric-warning"] {{
            color: {colors['warning']};
            border-left: 3px solid {colors['warning']};
        }}
        
        QLabel[class="metric-danger"] {{
            color: {colors['danger']};
            border-left: 3px solid {colors['danger']};
        }}
        """
    
    @classmethod
    def get_dialog_styles(cls, is_dark=True):
        """Compact dialog styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QDialog {{
            background-color: {colors['primary']};
            color: {colors['text_primary']};
            border-radius: 12px;
        }}
        
        QMessageBox {{
            background-color: {colors['primary']};
            color: {colors['text_primary']};
            border-radius: 8px;
        }}
        
        QMessageBox QLabel {{
            color: {colors['text_primary']};
            font-size: 12px;  /* Smaller font */
            padding: 6px;  /* Reduced padding */
        }}
        
        QMessageBox QPushButton {{
            min-width: 60px;  /* Reduced width */
            margin: 2px;  /* Reduced margin */
            padding: 6px 12px;  /* Reduced padding */
            font-size: 11px;  /* Smaller font */
        }}
        """
    
    @classmethod
    def get_progress_styles(cls, is_dark=True):
        """Compact progress bar styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QProgressBar {{
            background-color: {colors['secondary']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;  /* Smaller radius */
            text-align: center;
            font-weight: 600;
            font-size: 10px;  /* Smaller font */
            color: {colors['text_primary']};
            min-height: 16px;  /* Reduced height */
        }}
        
        QProgressBar::chunk {{
            background-color: {colors['accent']};
            border-radius: 4px;  /* Smaller radius */
            margin: 1px;
        }}
        """
    
    @classmethod
    def get_scrollbar_styles(cls, is_dark=True):
        """Compact scrollbar styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QScrollBar:vertical {{
            background-color: {colors['secondary']};
            width: 8px;  /* Thinner scrollbar */
            border-radius: 4px;
            margin: 0;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors['text_muted']};
            border-radius: 4px;
            min-height: 16px;  /* Smaller handle */
            margin: 1px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {colors['text_secondary']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background-color: {colors['secondary']};
            height: 8px;  /* Thinner scrollbar */
            border-radius: 4px;
            margin: 0;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {colors['text_muted']};
            border-radius: 4px;
            min-width: 16px;  /* Smaller handle */
            margin: 1px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {colors['text_secondary']};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
            width: 0px;
        }}
        """
    
    @classmethod
    def get_complete_style(cls, is_dark=True):
        """Get complete compact styling optimized for no-scroll layout"""
        return (
            cls.get_main_window_style(is_dark) +
            cls.get_button_styles(is_dark) +
            cls.get_input_styles(is_dark) +
            cls.get_table_styles(is_dark) +
            cls.get_card_styles(is_dark) +
            cls.get_label_styles(is_dark) +
            cls.get_dialog_styles(is_dark) +
            cls.get_progress_styles(is_dark) +
            cls.get_scrollbar_styles(is_dark)
        )