"""Point d'entrée direct, sans installation du package.

Utilisation depuis la racine du projet :
    python run.py --type Fire --output fire_card.png --seed 42

Ce fichier ne contient aucun chemin absolu : il ajoute simplement le dossier
`src/` voisin au sys.path, puis délègue au CLI du package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pokecardgen.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
