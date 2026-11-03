"""Log format and handlers shared by the GUIs, the command line, the run log and the session log.

Source:   https://github.com/maxsitt/insect-detect-post
License:  GNU AGPLv3 (https://choosealicense.com/licenses/agpl-3.0/)
Author:   Maximilian Sittinger (https://github.com/maxsitt)
AI use:   Developed with AI coding assistance, reviewed by the author
Docs:     https://maxsitt.github.io/insect-detect-docs/

All outputs write one layout, 'time LEVEL name: message': the terminal and the log viewer with
the time of day, log files with date, milliseconds and UTC offset. Modules of this package are
named without the package prefix, other loggers by their full name.

Classes:
    LogFormatter: Formatter for the shared log layout.

Functions:
    configure_logging(): Configure logging for an entry point.
    adopt_library_logger(): Route a library logger with its own terminal handler through the shared handlers.
    open_log_file(): Attach a log file that records this package and warnings of other packages.
    close_log_file(): Detach and close a handler from open_log_file().
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Logger of this package (parent of all module loggers)
_PACKAGE_LOGGER = "insectdetect_post"

# Loggers of other packages whose INFO records are noise and are set to log level WARNING
_QUIET_LOGGERS = ("httpx2", "httpcore2")


class LogFormatter(logging.Formatter):
    """Formatter for the shared layout 'time LEVEL name: message'."""

    def __init__(self, timestamp: Literal["time", "full"] = "time", color_level: bool = False) -> None:
        """Initialize the formatter.

        Args:
            timestamp: 'time' for the time of day (14:57:41), 'full' for date, milliseconds and
                UTC offset (2026-10-22 14:57:41.998+02:00).
            color_level: Mark the level name with '[html]'/'[/html]' for the qt-logging viewer to color.
        """
        level = "[html]{levelname:<8}[/html]" if color_level else "{levelname:<8}"
        super().__init__(f"{{asctime}} {level} {{name}}: {{message}}", style="{")
        self._timestamp = timestamp

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Return the record time in the configured precision (local time)."""
        if self._timestamp == "full":
            created = datetime.fromtimestamp(record.created).astimezone()
            return created.isoformat(sep=" ", timespec="milliseconds")
        return time.strftime("%H:%M:%S", time.localtime(record.created))

    def format(self, record: logging.LogRecord) -> str:
        """Format a copy of the record, so other handlers keep the full logger name."""
        record = logging.makeLogRecord(record.__dict__)
        record.name = record.name.removeprefix(f"{_PACKAGE_LOGGER}.")
        return super().format(record)


def configure_logging(console_level: int = logging.INFO, stream: Any = None) -> None:
    """Configure logging for an entry point.

    Terminal output with the shared layout; this package at DEBUG, the noisy loggers of other
    packages (_QUIET_LOGGERS) at WARNING, everything else at INFO. The terminal only shows
    records from console_level, while other handlers (log viewer, log files) still receive
    the DEBUG records of this package.

    Args:
        console_level: Lowest level shown in the terminal.
        stream: Object with write() and flush() the terminal output goes to (default: sys.stderr).
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(LogFormatter())
    handler.setLevel(console_level)
    # 'pygbif' calls logging.basicConfig() on import: 'force=True' replaces the handler it installed
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger(_PACKAGE_LOGGER).setLevel(logging.DEBUG)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def adopt_library_logger(name: str) -> None:
    """Route a library logger with its own terminal handler through the shared handlers.

    Some libraries add their own handler on import, which prints their records a second time, in
    another layout or to stdout. Call this after the import: their records are then shown once,
    in the shared layout, and also reach the log viewer and (from WARNING) the log files.

    Args:
        name: Name of the library's top-level logger.
    """
    library_logger = logging.getLogger(name)
    library_logger.handlers.clear()
    library_logger.propagate = True


class _LogFileFilter(logging.Filter):
    """Pass every record of this package and records of other packages from WARNING."""

    def filter(self, record: logging.LogRecord) -> bool:
        own = record.name == _PACKAGE_LOGGER or record.name.startswith(f"{_PACKAGE_LOGGER}.")
        return own or record.levelno >= logging.WARNING


def open_log_file(path: Path, mode: Literal["w", "a"]) -> logging.Handler:
    """Attach a log file that records this package from DEBUG and other packages from WARNING.

    Args:
        path: Log file path (missing parent directories are created).
        mode: 'w' to replace an existing file, 'a' to append to it.

    Returns:
        The added handler (close it with close_log_file()).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LogFormatter(timestamp="full"))
    handler.addFilter(_LogFileFilter())
    logging.getLogger().addHandler(handler)
    return handler


def close_log_file(handler: logging.Handler | None) -> None:
    """Detach and close a handler from open_log_file()."""
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    handler.close()
