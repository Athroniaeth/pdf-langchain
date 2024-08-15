import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Folder to monitor
DIRECTORY_TO_MONITOR = Path('../db/').absolute()


def check_and_delete_old_files():
    """Check and delete old files in the folder."""
    logging.info(f"Check and deleting old files in folder : '{DIRECTORY_TO_MONITOR}'")

    if not DIRECTORY_TO_MONITOR.exists():
        raise FileExistsError(f"Database vector path don't exist : '{DIRECTORY_TO_MONITOR}'")

    # Get the current time
    current_time = time.time()

    # Generator for all folders (no files) in the directory (no subdirectories)
    generator = DIRECTORY_TO_MONITOR.glob('**/*')
    generator = (file_path for file_path in generator if file_path.is_dir())

    for folder_path in generator:
        # Get the last modified time of the folder
        current_time_folder = os.path.getmtime(folder_path)
        time_elapsed = round(current_time - current_time_folder)

        logging.info(f"Folder: {folder_path.stem}, Last modified: {time_elapsed} seconds ago")

        # If the folder was modified more than 1 hour ago, delete it
        if time_elapsed > 3600:
            logging.info(f"\tThis folder is deleting !")
            shutil.rmtree(folder_path)


# Définir le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'delete_old_files_dag',
    default_args=default_args,
    description='DAG qui supprime les fichiers modifiés il y a plus d\'une heure',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Python Task to check and delete old files
check_and_delete_task = PythonOperator(
    task_id='check_and_delete_old_files',
    python_callable=check_and_delete_old_files,
    dag=dag,
)

check_and_delete_task  # noqa

if __name__ == "__main__":
    check_and_delete_old_files()
