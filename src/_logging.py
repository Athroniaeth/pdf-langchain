"""Logging utilities for the application."""

import functools
import inspect
import logging
import os
import sys
from enum import StrEnum

from loguru import logger

from src import LOGGING_PATH, PROJECT_PATH

SEPARATOR_RETURN = "\n" + " " * 20 + "|"
DEFAULT_FORMAT = (
    "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> | "
    "<level>{level}</level> | <red>"
    "<cyan>{extra[short_name]}</cyan>."
    "<cyan>{file}</cyan>:"
    "<cyan>{function}</cyan>:"
    f"</red><cyan>{{line}}</cyan>{SEPARATOR_RETURN} "
    "<level>{message}</level>\n"
)


def custom_filter(record):
    """Custom filter for loguru, allow to transform module name to logger name."""
    record["extra"]["short_name"] = record["name"].split(".")[0]
    return True


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


class StreamToLogger:
    def __init__(self, level_name="PRINT"):
        self.level_name = level_name

    def write(self, message):
        message = message.strip()

        if message:
            depth = self.find_depth()
            message = InterceptHandler.preprocess_message(message)
            level_name = "WARNING" if "warning" in message.lower() else self.level_name
            logger.opt(depth=depth).log(level_name, f"{message}")

    @staticmethod
    def find_depth():
        """
        Find the depth of the caller of the application.

        Notes:
            We ignore the first depth because it's related to the StreamToLogger class.
            All callers of the application are in the "src" directory.
        """
        depth = 0
        frame = inspect.currentframe().f_back

        while frame:
            # Get the filename of the caller
            filename = frame.f_code.co_filename

            conditions = (
                depth != 0,
                "src" in filename.replace("\\", "/").split("/"),
            )

            if all(conditions):
                return depth

            frame = frame.f_back
            depth += 1

        # If not found, use the default depth
        return 1

    @staticmethod
    def flush():
        # StreamToLogger must have flush method
        pass

    @staticmethod
    def isatty():
        """Return True if the stream is a tty (terminal)."""
        return False


class InterceptHandler(logging.Handler):
    def emit(self, record):
        """
        Intercept the logging message and send it to loguru.

        Notes:
            depth = 1: logging:handle:1028
            depth = 2: logging:callHandlers:1762
            depth = 3: logging:handle:1700
            depth = 4: logging:_log:1684
            depth = 5: logging:LEVEL:15XX
            depth = 6: is the real caller
        """
        # Apply preprocessing to the message
        message = self.preprocess_message(record.getMessage())

        # Get the corresponding log level in Loguru
        level = logger.level(record.levelname).name if logger.level(record.levelname).name else "CRITICAL"

        # Find the origin of the message call and the logger
        logger.opt(depth=6, exception=record.exc_info).log(level, message)

    @staticmethod
    def preprocess_message(message):
        # Remove duplicate spaces
        list_words = message.split(" ")

        # Remove empty strings (spaces)
        list_words = filter(lambda x: x != "", list_words)

        # Join the words with a space
        new_message = " ".join(list_words)

        # Replace newlines with spaces (have one line per log)
        new_message = new_message.replace("\n", SEPARATOR_RETURN)  # .replace("\r", " ")

        return new_message


def set_level(level: Level):
    """Set the logging level for all handlers."""
    loguru_level = logger._core.levels.get(level)  # noqa

    if loguru_level is None:
        raise ValueError(f"Loguru level does not exist: '{level}'")

    for index, handler in logger._core.handlers.items():  # noqa
        logger._core.handlers[index]._levelno = loguru_level.no  # noqa

    # On augmente le niveau de log pour l'afficher même en production
    logger.success(f"Application change the logging level to '{level}'")


def set_level_logging(custom_logger: logging.Logger, logging_level_loguru: Level):
    """Allow to loguru to intercept stdout of any library that use logging module."""
    logger.success(f"Application change the logging level of '{custom_logger.name}' to '{logging_level_loguru}'")

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

    # Remove default loguru's handler
    logger.remove()

    # Add a new handler for file (will be use for stdout)
    logger.add(
        log_file,
        rotation=rotation,
        retention=retention,
        compression="xz",
        format=format_,
        level=level,
        filter=custom_filter,
    )

    logger.add(
        sink=sys.stdout,
        format=format_,
        colorize=True,
        level=Level.ERROR,
        filter=custom_filter,
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=level)

    # Get the configuration of the INFO level
    info_level = logger.level("INFO")

    # Create a new level "PRINT" equivalent to INFO (level 20)
    logger.level("PRINT", no=info_level.no, color=info_level.color, icon=info_level.icon)

    # Redirect stdout and stderr to loguru
    sys.stdout = StreamToLogger()
    sys.stderr = StreamToLogger(level_name="ERROR")

    # Change the level of the logger (else not working)
    set_level(level)

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
