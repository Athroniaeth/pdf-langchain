# Use the official Airflow image
FROM apache/airflow:2.9.3-python3.12

# Change the user to root
USER root

# Set write permission on the 'db' volume directory for the 'airflow' user
RUN chown -R airflow:root /db

# Change the user to 'airflow'
USER airflow

# Set the Airflow environment
ENV AIRFLOW_HOME=/opt/airflow

# Optional: Set a specific timezone
ENV AIRFLOW__CORE__DEFAULT_TIMEZONE="Europe/Paris"

# Copy local DAGs files to the container
COPY ./dags ${AIRFLOW_HOME}/dags

# Enable authentication (optional for dev)
ENV AIRFLOW__WEBSERVER__AUTHENTICATE=True
ENV AIRFLOW__WEBSERVER__AUTH_BACKEND=airflow.contrib.auth.backends.password_auth

# Initialize the SQLite database (useful for a development environment)
RUN airflow db init

# Create an admin user
RUN airflow users create \
    --username airflow \
    --firstname Air \
    --lastname Flow \
    --role Admin \
    --email admin@example.com \
    --password airflow

# Start Airflow Webserver and Scheduler when the container starts
CMD ["bash", "-c", "airflow scheduler & airflow webserver --port 8080"]