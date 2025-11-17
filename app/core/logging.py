# Legal Chatbot - Logging Configuration

import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from datetime import datetime
from app.core.config import settings


def serialize_log(record: Dict[str, Any]) -> str:
    """Serialize log record to JSON format"""
    # Handle file path - it's a RecordFile object, not a dict
    file_path = ""
    if "file" in record:
        file_obj = record["file"]
        if hasattr(file_obj, "path"):
            file_path = file_obj.path
        elif isinstance(file_obj, dict):
            file_path = file_obj.get("path", "")
    
    log_data = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record.get("name", ""),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "file": file_path,
    }
    
    # Add exception info if present
    if "exception" in record:
        log_data["exception"] = {
            "type": record["exception"]["type"],
            "value": str(record["exception"]["value"]),
            "traceback": record["exception"]["traceback"],
        }
    
    # Add extra fields if present
    if "extra" in record:
        log_data.update(record["extra"])
    
    return json.dumps(log_data, default=str)


def setup_logging():
    """Setup application logging with structured JSON format"""
    # Remove default logger
    logger.remove()
    
    # Ensure logs directory exists
    log_file_path = Path(settings.LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine log format based on settings
    use_json = settings.LOG_FORMAT.lower() == "json"
    
    if use_json:
        # JSON format for file logging (machine-readable)
        logger.add(
            settings.LOG_FILE,
            level=settings.LOG_LEVEL,
            format=serialize_log,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,  # Thread-safe
        )
        
        # Human-readable format for console (development)
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
    else:
        # Standard format for both console and file
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
        
        logger.add(
            settings.LOG_FILE,
            level=settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )
    
    # Replace standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logger


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward loguru"""
    
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name"""
    return logger.bind(name=name)
