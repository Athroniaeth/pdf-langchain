"""Logging utilities for the application."""

import functools
import os
import sys
from enum import StrEnum
from pathlib import Path
from typing import Union

from loguru import logger

from src import LOGGING_PATH, PROJECT_PATH


class Level(StrEnum):
    """Log levels for the application."""

    # TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EXCEPTION = "EXCEPTION"


@functools.partial(logger.catch(message="Setup security failed."))
def setup_security(path: Union[str, os.PathLike], desired_permissions: int = 0o750):
    path = Path(path).absolute()
    short_path = path.relative_to(PROJECT_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Path '{path}' does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(f"Path '{path}' is not a directory.")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"Path '{path}' is not readable.")

    if not os.access(path, os.W_OK):
        raise PermissionError(f"Path '{path}' is not writable.")

    # Vérification des permissions actuelles du répertoire
    current_permissions = os.stat(path).st_mode & 0o777

    if current_permissions != desired_permissions:
        logger.debug(f"Current permissions for folder '{short_path}': {oct(current_permissions)}")
        os.chmod(path, desired_permissions)

    logger.debug(f"Permissions for folder '{short_path}' set to {oct(desired_permissions)}.")
    logger.info(f"Security setup for folder '{short_path}' has been completed.")


def setup_logger(
    name: str = "app",
    level: Level = Level.DEBUG,
    rotation: str = "06:00",
    retention: str = "30 days",
    format_: str = "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> | "
    "<level>{level: <6}</level> | <red>"
    "<cyan>{file}</cyan>:"
    "<cyan>{function}</cyan>:"
    "<cyan>{line}</cyan> - "
    "</red><level>{message}</level>",
):
    """
    Setup the logger for the application.

    Args:
        name (str): The name of the logger.
        rotation (str): The rotation time for the log file.
        retention (str): The retention time for the log file.
        format_ (str): The format of the log message.
    """
    log_file = LOGGING_PATH / f"{name}.log"

    # Refresh format loguru's native handler
    logger.remove()
    logger.add(
        sink=sys.stdout,
        format=format_,
        colorize=True,
        level=level,
    )

    # Add a new handler for the log file
    logger.add(log_file, rotation=rotation, retention=retention, compression="xz", format=format_, level=level)

    # Short the path by PROJECT_PATH (security ask to don't log full path)
    short_path = log_file.relative_to(PROJECT_PATH)
    logger.info(f"Server started with log file '{short_path}'.")


if __name__ == "__main__":
    setup_logger(name="app", rotation="06:00", retention="30 days", level=Level.DEBUG)
    setup_security(path=LOGGING_PATH, desired_permissions=0o750)
