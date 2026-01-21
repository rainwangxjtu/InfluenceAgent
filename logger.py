import logging
from datetime import datetime

# Configure logging to file
logging.basicConfig(
    filename="influenceagent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_event(stage, message):
    """Log an event with the pipeline stage and message"""
    logging.info(f"[{stage}] {message}")
