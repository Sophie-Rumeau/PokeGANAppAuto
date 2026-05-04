from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PokemonCardPipeline
from .schema import GenerationRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère une carte Pokémon custom selon un type.")
    parser.add_argument("--type", required=True, help="Ex: Fire, Water, Grass, Electric, Feu, Eau...")
    parser.add_argument("--config", default=None, help="YAML relatif à la racine du projet. Défaut: configs/default.yaml")
    parser.add_argument("--output", default=None, help="Fichier de sortie. Relatif à outputs/ si non absolu.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="Nom imposé. Sinon générateur de nom.")
    args = parser.parse_args()

    request = GenerationRequest.from_type(
        args.type,
        output_path=Path(args.output) if args.output else None,
        seed=args.seed,
        name_hint=args.name,
    )

    pipeline = PokemonCardPipeline(config_path=args.config)
    path = pipeline.generate(request)
    print(path)


if __name__ == "__main__":
    main()
