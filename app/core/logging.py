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
    try:
        # Handle file path - it's a RecordFile object, not a dict
        file_path = ""
        if "file" in record:
            file_obj = record["file"]
            if hasattr(file_obj, "path"):
                file_path = file_obj.path
            elif isinstance(file_obj, dict):
                file_path = file_obj.get("path", "")
        
        # Safely get time
        timestamp = record.get("time")
        if hasattr(timestamp, "isoformat"):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp) if timestamp else ""
        
        # Safely get level
        level = record.get("level")
        level_name = level.name if hasattr(level, "name") else str(level) if level else "UNKNOWN"
        
        log_data = {
            "timestamp": timestamp_str,
            "level": level_name,
            "message": record.get("message", ""),
            "module": record.get("name", ""),
            "function": record.get("function", ""),
            "line": record.get("line", 0),
            "file": file_path,
        }
        
        # Add exception info if present
        if "exception" in record and record["exception"] is not None and isinstance(record["exception"], dict):
            log_data["exception"] = {
                "type": record["exception"].get("type", "Unknown"),
                "value": str(record["exception"].get("value", "")),
                "traceback": record["exception"].get("traceback", ""),
            }
        
        # Add extra fields if present (handle nested extra)
        if "extra" in record and record["extra"]:
            extra_data = record["extra"]
            # Handle nested "extra" key
            if isinstance(extra_data, dict) and "extra" in extra_data:
                log_data.update(extra_data["extra"])
            elif isinstance(extra_data, dict):
                log_data.update(extra_data)
        
        out = json.dumps(log_data, default=str)
        # Loguru treats callable format return values as templates and runs format_map(record).
        # JSON braces { } must be escaped ({{ }}) to avoid KeyError; output stays valid JSON.
        return out.replace("{", "{{").replace("}", "}}")
    except Exception as e:
        fallback = json.dumps({"message": str(record.get("message", "Unknown")), "error": str(e)})
        return fallback.replace("{", "{{").replace("}", "}}")


def setup_logging():
    """Setup application logging. Production-safe: stdout always used; file optional."""
    logger.remove()
    use_json = settings.LOG_FORMAT.lower() == "json"
    is_production = (settings.ENVIRONMENT or "development").lower() == "production"

    # Console: always add. In production prefer JSON for Cloud Run / log aggregation.
    if use_json or is_production:
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            format=serialize_log,
        )
    else:
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

    # File: only when not production and log directory is writable (avoid read-only filesystem)
    if not is_production:
        try:
            log_file_path = Path(settings.LOG_FILE)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = serialize_log if use_json else "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            logger.add(
                settings.LOG_FILE,
                level=settings.LOG_LEVEL,
                format=fmt,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                enqueue=True,
            )
        except OSError:
            pass  # Log directory not writable (e.g. Cloud Run); stdout only
    
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
