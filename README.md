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

This project is example Chat-PDF, a web application that lets you chat with any PDF. This project uses LangChain for the prompt engineering, Gradio for the web user interface and Gradio to create the CLI for production.
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