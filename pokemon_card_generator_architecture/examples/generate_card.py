from pathlib import Path

from pokecardgen.pipeline import PokemonCardPipeline
from pokecardgen.schema import GenerationRequest


# Aucun chemin absolu requis. Modifier project_root si les checkpoints/data sont ailleurs,
# ou définir POKECARDGEN_HOME dans l'environnement.
pipeline = PokemonCardPipeline(project_root=Path(__file__).resolve().parents[1])
request = GenerationRequest.from_type("Fire", seed=42)
output = pipeline.generate(request)
print(output)
