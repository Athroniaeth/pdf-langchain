from pathlib import Path

# Classic global variables
PROJECT_PATH = Path(__file__).absolute().parents[1]
SOURCE_PATH = Path(__file__).absolute().parents[0]
ENV_PATH = PROJECT_PATH / ".env"

# Project specific global variables
DATABASE_PATH = PROJECT_PATH / "db"
