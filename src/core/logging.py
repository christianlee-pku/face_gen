import logging
import sys

def get_logger(name="face_gen", log_file=None, level=logging.INFO):
    """
    Get a logger instance.
    
    Args:
        name (str): Logger name.
        log_file (str, optional): Path to log file.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Prevent double logging if attached to root

    if not logger.handlers:
        # Console Handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
