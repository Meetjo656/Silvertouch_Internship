import logging
import sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a centralized logger for the project.
    
    Args:
        name (str): The name of the logger, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers are set to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(ch)
        
    return logger
