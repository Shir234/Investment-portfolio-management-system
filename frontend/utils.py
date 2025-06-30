import os
import sys
from frontend.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

def resource_path(relative_path):
    """Get absolute path to resources, works for dev and PyInstaller."""
    # Base path is project root (Investment-portfolio-management-system)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        # Resolve project root from frontend directory
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_path = os.path.normpath(os.path.join(base_path, relative_path))
    logger.debug(f"Resolved resource path: {relative_path} -> {full_path}")
    if not os.path.exists(full_path):
        logger.warning(f"Resource not found: {full_path}")
    return full_path