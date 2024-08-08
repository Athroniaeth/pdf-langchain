<div align="center">
    <h1 style="font-size: large; font-weight: bold;">Chat PDF</h1>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.12-0">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-f">
    </a>
  <br>
</div>

## Presentation

This project is example Chat-PDF, a web application that lets you chat with any PDF. This project uses LangChain for the
prompt engineering, Gradio for the web user interface and Gradio to create the CLI for production.
![presentation.png](static/presentation.png)

## Installation

This project use poetry, to install the dependencies, you can run the following command:

```bash
poetry install
```

## Usage

To run the project, you can use the following command:

```bash
poetry run python src
```

## Structure

```bash
├── src               # Project source code
├── docs              # Project documentation
│   └── static        # README.md static files
├── tests             # Folder containing software tests
│   ├── units         # Unit tests
│   └── integrations  # Integration tests
├── scripts           # Useful scripts for the project (no CI/CD)
├── ruff.toml         # Ruff configuration file
├── environment.yml   # Conda environment configuration file
```

## Installation

This project requires **conda** to be installed. To install the dependencies, simply run the following command:

```bash
conda env create -f environment.yml
```

You can update the environment with the following command:

```bash
conda env update -f environment.yml
```

## Usage

This project uses `typer` to create a command-line interface. To launch command help, simply issue the following command
the following command:

```bash
python src
```

## Docker

To build the Docker image, you can run the following command:

```bash
docker build -t chat_pdf .
```

To run the Docker container, you can use the following command:

```bash
docker run -p 7860:7860 chat_pdf
```

For debugging purposes, you can run the following command:

```bash
docker run -it -p 7860:7860 chat_pdf /bin/bash
```

For deleting all Docker containers, you can use the following command:

```bash
docker rm -f $(docker ps -a -q)
```

For deleting all Docker images, you can use the following command:

```bash
docker rmi -f $(docker images -q)
```