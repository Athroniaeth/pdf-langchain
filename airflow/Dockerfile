# Utiliser l'image officielle d'Airflow
FROM apache/airflow:2.9.3-python3.12

# Change l'utilisateur à root pour installer des packages supplémentaires si nécessaire
# USER root

# Change l'utilisateur pour Airflow
# USER user

# Initialisation de l'environnement Airflow
ENV AIRFLOW_HOME=/opt/airflow

# Optionnel : Configuration d'une timezone spécifique
ENV AIRFLOW__CORE__DEFAULT_TIMEZONE="Europe/Paris"

# Copie des fichiers DAGs locaux dans le conteneur
COPY ./dags ${AIRFLOW_HOME}/dags

# Activer l'authentification (optionnel pour dev)
ENV AIRFLOW__WEBSERVER__AUTHENTICATE=True
ENV AIRFLOW__WEBSERVER__AUTH_BACKEND=airflow.contrib.auth.backends.password_auth

# Initialisation de la base de données SQLite (utile pour un environnement de développement)
RUN airflow db init

# Création d'un utilisateur admin
RUN airflow users create \
    --username airflow \
    --firstname Air \
    --lastname Flow \
    --role Admin \
    --email admin@example.com \
    --password airflow

# Lancement d'Airflow Webserver et Scheduler au démarrage du conteneur
CMD ["bash", "-c", "airflow scheduler & airflow webserver --port 8080"]