import logging
import logging.handlers
import os
import sys
import atexit
from datetime import datetime

# Global list to track all handlers for cleanup
_all_handlers = []

def setup_logging(log_level=logging.INFO):
    """
    Centralized logging configuration for the entire application.
    """
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    _all_handlers.append(console_handler)
    
    # Main application log file
    app_log_file = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    app_file_handler = logging.handlers.RotatingFileHandler(
        app_log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    app_file_handler.setLevel(log_level)
    app_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(app_file_handler)
    _all_handlers.append(app_file_handler)
    
    # Trading logic specific log - ISOLATED (no propagation to root logger)
    trading_log_file = f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
    trading_file_handler = logging.handlers.RotatingFileHandler(
        trading_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    trading_file_handler.setLevel(log_level)
    trading_file_handler.setFormatter(detailed_formatter)
    
    # Create isolated trading logger (doesn't propagate to root)
    trading_logger = logging.getLogger('trading_logic')
    trading_logger.handlers.clear()  # Clear any existing handlers
    trading_logger.addHandler(trading_file_handler)
    trading_logger.setLevel(log_level)
    trading_logger.propagate = False  # CRITICAL: Don't propagate to root logger
    _all_handlers.append(trading_file_handler)
    
    # Pipeline log - ISOLATED (no propagation to root logger)
    pipeline_log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    pipeline_file_handler = logging.handlers.RotatingFileHandler(
        pipeline_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    pipeline_file_handler.setLevel(log_level)
    pipeline_file_handler.setFormatter(detailed_formatter)
    
    # Create isolated pipeline logger (doesn't propagate to root)
    pipeline_logger = logging.getLogger('stock_pipeline')
    pipeline_logger.handlers.clear()  # Clear any existing handlers
    pipeline_logger.addHandler(pipeline_file_handler)
    pipeline_logger.setLevel(log_level)
    pipeline_logger.propagate = False  # CRITICAL: Don't propagate to root logger
    _all_handlers.append(pipeline_file_handler)
    
    # Suppress matplotlib font manager logs
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Register cleanup function
    atexit.register(cleanup_logging)
    
    print(f"Logging initialized. Logs will be written to:")
    print(f"  Main: {app_log_file}")
    print(f"  Trading: {trading_log_file}")
    print(f"  Pipeline: {pipeline_log_file}")

def cleanup_logging():
    """
    Properly close and flush all logging handlers.
    """
    print("Cleaning up logging handlers...")
    
    # Flush and close all handlers
    for handler in _all_handlers:
        try:
            handler.flush()
            handler.close()
        except Exception as e:
            print(f"Error closing handler {handler}: {e}")
    
    # Clear the list
    _all_handlers.clear()
    
    # Force flush all loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            try:
                handler.flush()
                handler.close()
            except:
                pass

def force_log_flush():
    """
    Force flush all log handlers immediately.
    """
    for handler in _all_handlers:
        try:
            handler.flush()
        except Exception as e:
            print(f"Error flushing handler {handler}: {e}")
    
    # Also flush all other loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            try:
                handler.flush()
            except:
                pass

def get_logger(name):
    """
    Get a logger with the specified name.
    """
    return logging.getLogger(name)