import sys
import threading
from datetime import datetime
from typing import Optional

from basic_utils.config import settings


class Logger:
    """Simple synchronous stdout logger."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._log_level = getattr(settings, "log_level", "INFO").upper()
        self._write_lock = threading.Lock()

    def _format_line(self, level: str, message: str, extra: Optional[dict] = None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {self.name} - {level} - {message}"
        if extra:
            log_message += f" - {extra}"
        return f"{log_message}\n"

    def _write_line_sync(self, line: str) -> None:
        try:
            with self._write_lock:
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception as exc:
            print(f"Error writing log line: {exc}")

    def _log(self, level: str, message: str, extra: Optional[dict] = None) -> None:
        line = self._format_line(level, message, extra)
        self._write_line_sync(line)

    def debug(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("DEBUG", message, extra)

    def info(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("INFO", message, extra)

    def warning(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("WARNING", message, extra)

    def error(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("ERROR", message, extra)

    def critical(self, message: str, extra: Optional[dict] = None) -> None:
        self._log("CRITICAL", message, extra)


_loggers: dict[str, Logger] = {}


def get_logger(name: str) -> Logger:
    """Get or create a stdout logger."""
    if name not in _loggers:
        _loggers[name] = Logger(name=name)
    return _loggers[name]