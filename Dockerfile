# Utiliser l'image de base de Miniconda
# Use the Miniconda base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the .env file to the working directory
COPY .env .

# Copy the contents of the src folder to the working directory
COPY src/ src/

# Copy the environment.yml file to install the conda dependencies
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Activate the conda environment
SHELL ["conda", "run", "-n", "chat_pdf", "/bin/bash", "-c"]

# Expose port 7860 (80, 433 already in use)
EXPOSE 7860

# Set the command to run the application
# (--no-capture-output to return stdout to the console)
CMD ["conda", "run", "--no-capture-output", "-n", "chat_pdf", "python", "src", "run", "--host", "0.0.0.0"]
