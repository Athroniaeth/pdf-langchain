import logging
import os
import tomllib
from enum import StrEnum
from types import SimpleNamespace
from typing import List, Optional, Tuple

import typer
from loguru import logger
from typer import Typer

from src._logging import Level, set_level, set_level_logging
from src.app import app

cli = Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=False, no_args_is_help=True)

LOGGER_TO_VERBOSE = ["httpcore", "urllib3", "httpx", "multipart.multipart", "PIL.PngImagePlugin", "chromadb.config"]


@cli.callback()
def callback(
    ctx: typer.Context,
    hf_token: str = typer.Option(None, envvar="HF_TOKEN", help="Access token for Hugging Face API."),
    logging_level: Level = typer.Option(Level.INFO, help="Log level for application logs."),
    ignore_logging: List[str] = typer.Option(LOGGER_TO_VERBOSE, help="Ignore logs from the given loggers."),
):
    """
    Initialize the CLI application context.

    Args:
        ctx (typer.Context): Command context.
        hf_token (str): Access token for Hugging Face API.
        logging_level (Level): Log level for application logs.
        ignore_logging (List[str]): Ignore logs from the given loggers.

    Raises:
        typer.Exit: If Hugging Face token is missing.

    Returns:
        SimpleNamespace: Object containing application parameters.
    """
    # Set the logging level for the application
    set_level(logging_level)

    # Set WARNING for ignored loggers
    for logger_name in ignore_logging:
        logger_ignore = logging.getLogger(logger_name)
        set_level_logging(logger_ignore, Level.WARNING)

    if hf_token is None:
        logger.exception("Missing Hugging Face token; pass --hf-token or set env[HF_TOKEN]")
        raise typer.Exit(1)

    ctx.obj = SimpleNamespace(hf_token=hf_token, logging_level=logging_level)


def conf_callback(ctx: typer.Context, param: typer.CallbackParam, filepath: str) -> str:
    """
    Load a configuration file and update the default map.

    References:
         https://github.com/tiangolo/typer/issues/86

    Args:
        ctx (typer.Context): Command context.
        param (typer.CallbackParam): Callback parameter.
        filepath (str): Path to the configuration file.

    Returns:
        str: Path to the configuration file.
    """
    if filepath:
        try:
            # Load the configuration file
            with open(filepath, "rb") as file:
                conf = tomllib.load(file)

            # Init the dictionary and update the default map
            ctx.default_map = ctx.default_map or {}
            ctx.default_map.update(conf)
        except Exception as ex:
            raise typer.BadParameter(str(ex))

    return filepath


class Environment(StrEnum):
    """Environment to use (local or ovh cloud)."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_info_environment(
    environment: Environment = Environment.DEVELOPMENT,
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
) -> Tuple[str, int, str, str, str, bool]:
    """
    Get the information of the environment.

    Args:
        environment (Environment): The environment to use.
        ssl_keyfile (Optional[str]): The SSL key file path.
        ssl_certfile (Optional[str]): The SSL certificate file path.

    Returns:
        str: The host to use for the server.
        int: The port to use for the server.
        str: The SSL key file path.
        str: The SSL certificate file path.
        str: The maximum file size for upload.
        bool: Enable monitoring for the application.
    """
    match environment:
        case Environment.DEVELOPMENT:
            host = "127.0.0.1"
            port = 7860
            ssl_keyfile = None
            ssl_certfile = None
            max_file_size = "100mb"
            enable_monitoring = False

        case Environment.PRODUCTION:
            list_raises = []
            host = "0.0.0.0"
            port = 7860
            max_file_size = "1mb"
            enable_monitoring = True

            if not ssl_keyfile:
                list_raises.append(EnvironmentError("'SSL_KEYFILE' environment variable not set"))

            if not ssl_certfile:
                list_raises.append(EnvironmentError("'SSL_CERTFILE' environment variable not set"))

            if ssl_keyfile and not os.path.exists(ssl_keyfile):
                list_raises.append(FileNotFoundError(f"'{ssl_keyfile}' not found"))

            if ssl_certfile and not os.path.exists(ssl_certfile):
                list_raises.append(FileNotFoundError(f"'{ssl_certfile}' not found"))

            if list_raises:
                raise ExceptionGroup("Failed to start the server", list_raises)

        case _:
            raise ValueError(f"Environment '{environment}' not supported")

    return host, port, ssl_keyfile, ssl_certfile, max_file_size, enable_monitoring


@cli.command()
def start(
    ctx: typer.Context = typer.Option(None, callback=callback, is_eager=True),
    config: str = typer.Option("", callback=conf_callback, is_eager=True),
    environment: Environment = typer.Option(Environment.DEVELOPMENT, envvar="ENVIRONMENT", help="Environnement d'exécution."),
    ssl_keyfile: str = typer.Option(None, envvar="SSL_KEYFILE", help="Fichier de clé SSL."),
    ssl_certfile: str = typer.Option(None, envvar="SSL_CERTFILE", help="Fichier de certificat SSL."),
    model_id: str = typer.Option("meta-llama/Meta-Llama-3.1-8B-Instruct", help="Identifiant HuggingFace du modèle LLM."),
):
    """
    Start the server with the given environment.

    Args:
        ctx (typer.Context): Command context.
        config (str): Path to the configuration file.
        environment (Environment): The environment to use.
        ssl_keyfile (str): The SSL key file path.
        ssl_certfile (str): The SSL certificate file path.
        model_id (str): The model ID of Hugging Face LLM.

    Raises:
        FileNotFoundError: If the SSL key or certificate file is not found.
        ValueError: If the SSL key or certificate file is missing.
    """

    # Get the environment information
    logger.info(f"Environment: {environment}")
    host, port, ssl_keyfile, ssl_certfile, max_file_size, enable_monitoring = get_info_environment(environment, ssl_keyfile, ssl_certfile)

    # Use 'run' command to start the server
    run(
        ctx=ctx,
        config=config,
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        model_id=model_id,
        max_file_size=max_file_size,
        enable_monitoring=enable_monitoring,
    )


@cli.command()
def run(
    ctx: typer.Context = typer.Option(None, callback=callback, is_eager=True),
    config: str = typer.Option("", callback=conf_callback, is_eager=True),
    host: str = typer.Option("127.0.0.1", envvar="HOST", help="Adresse sur laquelle le serveur doit écouter."),
    port: int = typer.Option(7860, envvar="PORT", help="Port sur lequel le serveur doit écouter."),
    ssl_keyfile: str = typer.Option(None, envvar="SSL_KEYFILE", help="Fichier de clé SSL."),
    ssl_certfile: str = typer.Option(None, envvar="SSL_CERTFILE", help="Fichier de certificat SSL."),
    model_id: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.3", help="Identifiant HuggingFace du modèle LLM."),
    max_file_size: str = typer.Option("1mb", envvar="MAX_FILE_SIZE", help="Taille maximale du fichier à télécharger."),
    enable_monitoring: bool = typer.Option(True, envvar="ENABLE_MONITORING", help="Activer la surveillance de l'application."),
):
    """
    Start the server with the given environment.

    Args:
        ctx (typer.Context): Command context.
        config (str): Path to the configuration file.
        host (str): Host IP address of the server.
        port (int): Port number of host server.
        ssl_keyfile (str): The SSL key file path.
        ssl_certfile (str): The SSL certificate file path.
        model_id (str): The model ID of Hugging Face LLM.
        max_file_size (str): The maximum file size for upload.
        enable_monitoring (bool): Enable monitoring for the application.

    Raises:
        FileNotFoundError: If the SSL key or certificate file is not found.
        ValueError: If the SSL key or certificate file is missing.
    """

    # Check if SSL key and certificate files exist
    if ssl_keyfile and not os.path.exists(ssl_keyfile):
        raise FileNotFoundError(f"SSL Key file '{ssl_keyfile}' not found")

    if ssl_certfile and not os.path.exists(ssl_certfile):
        raise FileNotFoundError(f"SSL Certificate file '{ssl_certfile}' not found")

    if (not ssl_keyfile and ssl_certfile) or (ssl_keyfile and not ssl_certfile):
        raise ValueError(f"Both SSL Key and Certificate files must be provided ({ssl_keyfile=}, {ssl_certfile=})")

    # Log the environment information
    ssl = bool(ssl_keyfile and ssl_certfile)
    logger.info(f"{host=}, {port=}, {ssl=}")

    # Run the Gradio application with the given environment
    app(
        model_id=model_id,
        hf_token=ctx.obj.hf_token,
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        max_file_size=max_file_size,
        enable_monitoring=enable_monitoring,
    )


if __name__ == "__main__":
    cli()
