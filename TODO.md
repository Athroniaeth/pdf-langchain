(optional)

- Crée des fichiers de configurations pour les interfaces
- Refactoriser les composants gradio, simplifier les fonctions, rendre les composants modulaires
- Permettre de mettre tout le document en contexte


- ~~Crée un système de log complet (savoir qui, quand se connecte d'utiliser le service, stockage des logs, etc.)~~
- Ajouter les outils de BP (~~lint~~, CI/CD, ~~formatage~~, coverage, etc.)
- Crée un système de test complet (unitaire, intégration, fonctionnel, etc.)
- Crée un système de monitoring (nombre de requêtes, temps de réponse, etc.)
- ~~Use transformers chat template (else LLM got prediction user/assistant simulation)~~
- Adapting image size if need (some documents are too big)
- Crée une automatisation (avec Airflow ou un Thread au lancement du serveur) pour supprimer les db vector
    - Spécifier dans le .env le temps de vie des db vector
    - Spécifier dans le .env le chemin des db vector (a la place de l'avoir en variable globale, cela permet de viser
      des stockages attachés par exemple)
