# styles.py - Modern UI styling for SharpSight Investment System

class ModernStyles:
    """Modern, comfortable styling following UI design principles"""
    
    # Color Palette
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
            # Even darker "light" mode - almost medium-dark theme
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
        """Main window styling with modern design principles"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        # For light mode, keep some elements darker for contrast and visual hierarchy
        if not is_dark:
            # Override certain colors to maintain contrast in light mode
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
            font-size: 13px;
            line-height: 1.5;
        }}
        
        /* Central Widget and Scroll Areas */
        QWidget {{
            background-color: transparent;  /* Make all QWidget containers transparent */
            color: {colors['text_primary']};
        }}
        
        /* Specific styling for main containers that need background */
        QMainWindow > QWidget {{
            background-color: {colors['primary']};
        }}
        
        QScrollArea {{
            background-color: {colors['primary']};
            border: none;
        }}
        
        QScrollArea > QWidget > QWidget {{
            background-color: transparent;  /* Keep scroll content transparent */
        }}
        
        /* Modern Tab Widget */
        QTabWidget::pane {{
            border: 1px solid {colors['border_light']};
            background-color: {colors['surface']};
            border-radius: 12px;
            margin-top: 8px;
        }}
        
        QTabBar::tab {{
            background-color: {tab_bg};
            color: {tab_text};
            padding: 12px 24px;
            margin-right: 4px;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
            font-size: 14px;
            {"border: 1px solid #4A5568;" if not is_dark else ""}
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
        """Modern button styling with proper hierarchy"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        /* Primary Button */
        QPushButton {{
            background-color: {colors['accent']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 14px;
            min-height: 20px;
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
        
        /* Theme Toggle Button */
        QPushButton[class="theme-toggle"] {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 500;
            min-width: 80px;
        }}
        
        QPushButton[class="theme-toggle"]:hover {{
            background-color: {colors['hover']};
            border-color: {colors['accent']};
        }}
        """
    

    @classmethod
    def get_input_styles(cls, is_dark=True):
        """Modern input field styling with text-based dropdown arrows"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        /* Text Inputs */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 14px;
            min-width: 200px;
            min-height: 20px;
            selection-background-color: {colors['accent']};
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {colors['accent']};
            outline: none;
        }}
        
        QLineEdit::placeholder {{
            color: {colors['text_muted']};
        }}
        
        /* Date Inputs */
        QDateEdit {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 14px;
            min-width: 200px;
            min-height: 20px;
        }}
        
        QDateEdit:focus {{
            border: 2px solid {colors['accent']};
        }}
        
        QDateEdit::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {colors['border']};
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
            background-color: {colors['secondary']};
        }}
        
        QDateEdit::drop-down:hover {{
            background-color: {colors['accent']};
        }}
        
        QDateEdit::down-arrow {{
            width: 12px;
            height: 12px;
            font-size: 12px;
            color: {colors['text_primary']};
            background-color: transparent;
            border: none;
        }}
        
        QDateEdit::down-arrow:after {{
            content: "▼";
        }}
        
        /* Combo Boxes */
        QComboBox {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 14px;
            min-width: 200px;
            min-height: 20px;
        }}
        
        QComboBox:focus {{
            border: 2px solid {colors['accent']};
        }}
        
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {colors['border']};
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
            background-color: {colors['secondary']};
        }}
        
        QComboBox::drop-down:hover {{
            background-color: {colors['accent']};
        }}
        
        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
            font-size: 12px;
            color: {colors['text_primary']};
            background-color: transparent;
            border: none;
        }}
        
        QComboBox::down-arrow:after {{
            content: "▼";
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            selection-background-color: {colors['accent']};
            outline: none;
        }}
        """
    
    @classmethod
    def get_table_styles(cls, is_dark=True):
        """Modern table styling with better readability"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QTableWidget {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 12px;
            gridline-color: {colors['border_light']};
            font-size: 13px;
            selection-background-color: {colors['selected']};
        }}
        
        QTableWidget::item {{
            border: none;
            padding: 12px 8px;
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
            padding: 16px 8px;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        QHeaderView::section:hover {{
            background-color: {colors['hover']};
        }}
        """
    
    @classmethod
    def get_card_styles(cls, is_dark=True):
        """Modern card/panel styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        /* Group Boxes as Cards */
        QGroupBox {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border_light']};
            border-radius: 12px;
            margin: 8px 0;
            padding-top: 24px;
            font-weight: 600;
            font-size: 14px;
            color: {colors['text_primary']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 8px 16px;
            background-color: {colors['surface']};
            border: 1px solid {colors['border_light']};
            border-radius: 6px;
            color: {colors['accent']};
            font-weight: 600;
        }}
        
        /* List Widgets */
        QListWidget {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;
            padding: 4px;
            font-size: 13px;
        }}
        
        QListWidget::item {{
            border-radius: 6px;
            padding: 8px 12px;
            margin: 2px;
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
        """Modern label and text styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QLabel {{
            color: {colors['text_primary']};
            font-size: 14px;
            font-weight: 500;
            background-color: transparent;  /* Ensure transparent background */
        }}
        
        QLabel[class="label"] {{
            color: {colors['text_primary']};
            font-size: 15px;  /* Slightly bigger */
            font-weight: 700;  /* Bold */
            margin-bottom: 8px;
            background-color: transparent;  /* Make sure labels have transparent background */
            border: none;  /* Remove any borders */
        }}
        
        QLabel[class="title"] {{
            font-size: 24px;
            font-weight: 700;
            color: {colors['text_primary']};
            margin: 16px 0;
            background-color: transparent;
        }}
        
        QLabel[class="subtitle"] {{
            font-size: 18px;
            font-weight: 600;
            color: {colors['text_secondary']};
            margin: 12px 0;
            background-color: transparent;
        }}
        
        QLabel[class="caption"] {{
            font-size: 12px;
            font-weight: 400;
            color: {colors['text_muted']};
            margin: 4px 0;
            background-color: transparent;
        }}
        
        QLabel[class="metric"] {{
            font-size: 16px;
            font-weight: 600;
            padding: 12px 16px;
            background-color: {colors['surface']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;
            color: {colors['text_primary']};
        }}
        
        QLabel[class="metric-success"] {{
            color: {colors['success']};
            border-left: 4px solid {colors['success']};
        }}
        
        QLabel[class="metric-warning"] {{
            color: {colors['warning']};
            border-left: 4px solid {colors['warning']};
        }}
        
        QLabel[class="metric-danger"] {{
            color: {colors['danger']};
            border-left: 4px solid {colors['danger']};
        }}
        """
    
    @classmethod
    def get_dialog_styles(cls, is_dark=True):
        """Modern dialog styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QDialog {{
            background-color: {colors['primary']};
            color: {colors['text_primary']};
            border-radius: 16px;
        }}
        
        QMessageBox {{
            background-color: {colors['primary']};
            color: {colors['text_primary']};
            border-radius: 12px;
        }}
        
        QMessageBox QLabel {{
            color: {colors['text_primary']};
            font-size: 14px;
            padding: 8px;
        }}
        
        QMessageBox QPushButton {{
            min-width: 80px;
            margin: 4px;
        }}
        """
    
    @classmethod
    def get_progress_styles(cls, is_dark=True):
        """Modern progress bar styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QProgressBar {{
            background-color: {colors['secondary']};
            border: 1px solid {colors['border_light']};
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            color: {colors['text_primary']};
            min-height: 20px;
        }}
        
        QProgressBar::chunk {{
            background-color: {colors['accent']};
            border-radius: 6px;
            margin: 2px;
        }}
        """
    
    @classmethod
    def get_scrollbar_styles(cls, is_dark=True):
        """Modern scrollbar styling"""
        colors = cls.COLORS['dark'] if is_dark else cls.COLORS['light']
        
        return f"""
        QScrollBar:vertical {{
            background-color: {colors['secondary']};
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors['text_muted']};
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
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
            height: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {colors['text_muted']};
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
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
        """Get complete modern styling"""
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