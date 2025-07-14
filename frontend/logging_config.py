# logging_config.py
"""
Centralized logging configuration for the SharpSight application.
Provides main application logging plus isolated loggers for trading and pipeline operations.
"""
import logging
import logging.handlers
import os
import sys
import atexit
from datetime import datetime

# Track all handlers for proper cleanup
_all_handlers = []


def setup_logging(log_level=logging.INFO):
    """Configure application-wide logging with multiple specialized log files."""

    # Ensure logs directory exists
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
    
    # Console output handler
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
    
    # Trading logic isolated logger
    trading_log_file = f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
    trading_file_handler = logging.handlers.RotatingFileHandler(
        trading_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    trading_file_handler.setLevel(log_level)
    trading_file_handler.setFormatter(detailed_formatter)
    
    trading_logger = logging.getLogger('trading_logic')
    trading_logger.handlers.clear()  # Clear any existing handlers
    trading_logger.addHandler(trading_file_handler)
    trading_logger.setLevel(log_level)
    trading_logger.propagate = False  # Prevent propagation to root logger
    _all_handlers.append(trading_file_handler)
    
    # Pipeline operations isolated logger
    pipeline_log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    pipeline_file_handler = logging.handlers.RotatingFileHandler(
        pipeline_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    pipeline_file_handler.setLevel(log_level)
    pipeline_file_handler.setFormatter(detailed_formatter)
    
    pipeline_logger = logging.getLogger('stock_pipeline')
    pipeline_logger.handlers.clear()
    pipeline_logger.addHandler(pipeline_file_handler)
    pipeline_logger.setLevel(log_level)
    pipeline_logger.propagate = False  # Prevent propagation to root logger
    _all_handlers.append(pipeline_file_handler)
    
    # Suppress matplotlib font manager logs
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Register cleanup for application exit
    atexit.register(cleanup_logging)
    
    print(f"Logging initialized. Logs will be written to:")
    print(f"  Main: {app_log_file}")
    print(f"  Trading: {trading_log_file}")
    print(f"  Pipeline: {pipeline_log_file}")


def cleanup_logging():
    """Properly close and flush all logging handlers during application shutdown."""

    print("Cleaning up logging handlers...")
    
    # Flush and close all tracked handlers
    for handler in _all_handlers:
        try:
            handler.flush()
            handler.close()
        except Exception as e:
            print(f"Error closing handler {handler}: {e}")
    
    _all_handlers.clear()
    
    # Force cleanup of any remaining handlers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            try:
                handler.flush()
                handler.close()
            except:
                pass


def force_log_flush():
    """Force flush all log handlers immediately."""
    
    for handler in _all_handlers:
        try:
            handler.flush()
        except Exception as e:
            print(f"Error flushing handler {handler}: {e}")
    
    # Flush all other active loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            try:
                handler.flush()
            except:
                pass


def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def get_isolated_logger(name, log_file_prefix=None, level=logging.INFO):
    """
    Create an isolated logger that writes only to its dedicated log file.
    
    Args:
    - name: Logger name/identifier
    - log_file_prefix: Custom prefix for log file (defaults to logger name)
    - level: Logging level for this logger
    
    Returns:
    - Isolated logger instance that doesn't propagate to root logger
    """

    if log_file_prefix is None:
        log_file_prefix = name
    
    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create and configure logger
    logger = logging.getLogger(name)
    logger.handlers.clear()
    
    # Create dedicated file handler
    log_file = f"logs/{log_file_prefix}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False  # CRITICAL: Isolate from root logger
    
    # Track for cleanup
    _all_handlers.append(file_handler)
    
    return logger