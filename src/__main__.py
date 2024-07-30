import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Automatically add the working directory
path = Path(__file__).parents[1].absolute()
sys.path.append(f"{path}")

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

    # Add the user command to the logs
    logging.debug(f"Command use \"{' '.join(sys.argv)}\"")

    try:
        cli()
        # Typer have his own exception handling
    except KeyboardInterrupt as exception:
        logging.debug(f"Exiting the program: '{exception}'")


if __name__ == "__main__":
    main()
