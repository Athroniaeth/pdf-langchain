"""Logging utilities for the application."""

import functools
import logging
import os
import sys
from enum import StrEnum

from loguru import logger

from src import LOGGING_PATH, PROJECT_PATH

DEFAULT_FORMAT = (
    "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> | "
    "<level>{level: <6}</level> | <red>"
    "<cyan>{file}</cyan>:"
    "<cyan>{function}</cyan>:"
    "<cyan>{line}</cyan> - "
    "</red><level>{message}</level>"
)


class Level(StrEnum):
    """Log levels for the application."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EXCEPTION = "EXCEPTION"


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Appliquer le prétraitement au message
        message = self.preprocess_message(record.getMessage())

        # Récupérer le niveau de log correspondant dans Loguru
        level = logger.level(record.levelname).name if logger.level(record.levelname).name else "INFO"

        # Trouver l'origine de l'appel du message loggué
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, message)

    @staticmethod
    def preprocess_message(message):
        # Remplacer les retours à la ligne par des espaces
        message = message.replace("\n", " ").replace("\r", " ")

        # Supprimer les espaces en trop
        message = " ".join(message.split())

        return message


def set_level(level: Level):
    """Set the logging level for all handlers."""
    logger.info(f"Application change the logging level to '{level}'")
    loguru_level = logger._core.levels.get(level)  # noqa

    if loguru_level is None:
        raise ValueError(f"Loguru level does not exist: '{level}'")

    for index, handler in logger._core.handlers.items():  # noqa
        logger._core.handlers[index]._levelno = loguru_level.no  # noqa


def set_level_logging(custom_logger: logging.Logger, logging_level_loguru: Level):
    """Allow to loguru to intercept stdout of any library that use logging module."""
    logger.info(f"Application change the logging level of '{custom_logger.name}' to '{logging_level_loguru}'")

    # This loguru_level don't have a corresponding logging_level
    switcher = {
        Level.TRACE: Level.DEBUG,
        Level.SUCCESS: Level.INFO,
        Level.EXCEPTION: Level.CRITICAL,
    }

    logging_level_loguru = switcher.get(logging_level_loguru, logging_level_loguru)

    # Found the corresponding logging level
    logging_level = logging.getLevelName(logging_level_loguru)

    # Intercept stdout of the library with the corresponding level
    custom_logger.addHandler(InterceptHandler())
    custom_logger.setLevel(logging_level)


@functools.partial(logger.catch(message="Setup security failed."))
def setup_security(desired_permissions: int = 0o750):
    """
    Setup the security for the logging folder.

    Args:
        desired_permissions (int): The desired permissions for the folder.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
        PermissionError: If the path is not readable or writable
    """
    path = LOGGING_PATH.absolute()
    short_path = path.relative_to(PROJECT_PATH)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

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
    rotation: str = "06:00",
    retention: str = "30 days",
    level: Level = Level.DEBUG,
    format_: str = DEFAULT_FORMAT,
):
    """
    Setup the logger for the application.

    Args:
        name (str): The name of the logger.
        rotation (str): The rotation time for the logging file.
        retention (str): The retention time for the logging file.
        level (Level): The logging level for the application.
        format_ (str): The format of the logging message.
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

    # Add a new handler for the logging file
    logger.add(log_file, rotation=rotation, retention=retention, compression="xz", format=format_, level=level)

    # Short the path by PROJECT_PATH (security ask to don't logging full path)
    short_path = log_file.relative_to(PROJECT_PATH)
    logger.info(f"Application start to logging in '{short_path}'")


if __name__ == "__main__":
    setup_logger()
    setup_security(desired_permissions=0o750)

    set_level(Level.ERROR)

    logger.info("Hello world")

    set_level(Level.INFO)

    logger.info("Hello world")
