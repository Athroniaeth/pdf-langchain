# Use the official Airflow image
FROM apache/airflow:slim-latest

# Default environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__WEBSERVER__AUTHENTICATE=True
ENV AIRFLOW__WEBSERVER__AUTH_BACKEND=airflow.contrib.auth.backends.password_auth
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db

# Change the user to root
USER root

# Set write permission on the 'db' volume directory for the 'airflow' user
RUN mkdir /db

# Copy local DAGs files to the container
COPY ./dags ${AIRFLOW_HOME}/dags


# Initialize the SQLite database (useful for a development environment)
RUN airflow db init

# Expose by default port 8080
EXPOSE 8080

# Start Airflow Webserver and Scheduler when the container starts
CMD "bash -c airflow scheduler & airflow webserver --port 8080 --ssl-key /etc/letsencrypt/live/pierrechaumont.fr/privkey.pem --ssl-cert /etc/letsencrypt/live/pierrechaumont.fr/fullchain.pem"
