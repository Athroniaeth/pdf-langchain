""" Logging utilities for the application. """
import logging
from enum import StrEnum


class Level(StrEnum):
    """Log levels for the application."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def clean_logging(logging_level: Level) -> None:
    """ Clean all logging handlers and set the logging level. """
    list_name = [name for name in logging.root.manager.loggerDict]
    list_name = filter(lambda name: not name == "root", list_name)
    loggers = [logging.getLogger(name) for name in list_name]

    for logger in loggers:
        set_logging_level(logger.name, logging_level)


def set_logging_level(logging_name: str, logging_level: Level) -> None:
    """ Set the logging level for a specific logger. """
    _logging_level = logging.getLevelName(logging_level)
    logging.getLogger(logging_name).setLevel(_logging_level)
    logging.info(f"Logging level of '{logging_name}': {logging_level}")
