import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Automatically add the working directory
path = Path(__file__).parents[1].absolute()
sys.path.append(f"{path}")

from _logging import Level, setup_logger, setup_security  # noqa: E402

from src.cli import cli  # noqa: E402


def main():
    """
    Main function to run the application.

    Raises:
        SystemExit: If the program is exited.

    Returns:
        int: The return code of the program.
    """
    # Load the environment variables
    load_dotenv()

    # Load the logging system
    setup_logger(level=Level.DEBUG)

    # Setup the security for the logging folder
    setup_security(desired_permissions=0o750)

    # Add the user command to the logs (first is src path)
    logger.info(f"Command for launch application is \"{' '.join(sys.argv[1:])}\"")

    try:
        cli()
        # Typer have his own exception handling
    except KeyboardInterrupt as exception:
        logger.debug(f"Exiting the program: '{exception}'")


if __name__ == "__main__":
    main()
