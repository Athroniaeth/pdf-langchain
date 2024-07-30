import platform
import re
import subprocess

import yaml


def run_command(command: str) -> str:
    """
    Exécute une commande shell en supprimant les séquences ANSI des sorties

    Args:
        command (str): Commande shell à exécuter.

    Returns:
        str: La sortie de la commande après nettoyage des séquences ANSI.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Command '{command}' returned non-zero exit status.\n{e.stderr}")

    return re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', result.stdout)


def export_conda_environment(env_name: str, output_path: str = "environment.yml"):
    """
    Exporte un fichier de l'environnement Conda.

    Args:
        env_name (str): Nom de l'environnement Conda.
        output_path (str): Chemin de sortie du fichier d'export.
    """
    # Initialisation des commandes pour l'environnement Conda
    conda_init_command = ". ~/miniconda3/etc/profile.d/conda.sh && " if platform.system() == "Linux" else ""

    def conda_command(cmd):
        return f"{conda_init_command}conda run -n {env_name} {cmd}"

    # Crée l'export de l'environnement Conda
    env_command = conda_command("conda env export --from-history")

    # Crée l'export des canaux Conda (non fournit par `conda env export`)
    channels_command = conda_command("conda env export -c conda-forge")

    # Crée l'export des dépendances de pip
    pip_command = conda_command("pip list --format=freeze")

    # Stocke les canaux Conda
    channels_result = run_command(channels_command)
    channels_data = yaml.safe_load(channels_result)
    channels = channels_data.get('channels', [])

    # Stocke les dépendances de l'environnement Conda
    env_result = run_command(env_command)
    env_data = yaml.safe_load(env_result)
    env_data['prefix'] = '~'  # Permet de rendre l'environnement portable
    env_data['channels'] = channels

    # Stocke les dépendances de pip
    pip_result = run_command(pip_command)
    pip_dependencies = list(filter(None, pip_result.split('\n')))

    if not env_data.get('dependencies'):
        env_data['dependencies'] = []

    env_data['dependencies'].append({'pip': pip_dependencies})

    # Enregistre les dépendances le fichier YAML
    try:
        with open(output_path, 'w') as file:
            yaml.dump(env_data, file)
    except PermissionError as e:
        raise Exception(f"Vous n'avez pas les permissions d'écriture sur le dossier.\n{e}")


if __name__ == "__main__":
    export_conda_environment('chat_pdf')
