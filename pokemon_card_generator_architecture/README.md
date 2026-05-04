# Pokémon Card Generator

Architecture modulaire pour générer des cartes Pokémon custom à partir d'un type.

## Lancement direct

Depuis la racine du projet :

```powershell
python run.py --type Fire --output fire_card.png --seed 42
python run.py --type Eau --output water_card.png
python run.py --type Electric --name Voltrix
```

Aucune variable d'environnement n'est nécessaire. Aucun chemin absolu n'est codé dans le projet.

## Règle de résolution des chemins

La racine du projet est détectée automatiquement à partir de l'emplacement du code source :

```text
pokemon_card_generator_architecture/
  run.py
  configs/default.yaml
  checkpoints/
  data/
  outputs/
  src/pokecardgen/
```

Les chemins du YAML sont relatifs à cette racine :

```yaml
paths:
  checkpoints_dir: "checkpoints"
  data_dir: "data"
  output_dir: "outputs"

models:
  ctgan_full: "ctgan_model_full.pt"
  vae: "vae_final.pt"
  latent_diffusion: "latent_diffusion_last.pt"
```

Donc :

```text
ctgan_model_full.pt       -> checkpoints/ctgan_model_full.pt
vae_final.pt              -> checkpoints/vae_final.pt
pokemon_data.csv          -> data/pokemon_data.csv
fire_card.png             -> outputs/fire_card.png
```

## Fichiers attendus

```text
checkpoints/
  ctgan_model_full.pt
  generator_state.pt
  latent_diffusion_last.pt
  vae_final.pt

data/
  pokemon_data.csv
  pokemon-tcg-data-master 1999-2023.csv
```

Si un modèle ou un CSV est absent, le générateur utilise un fallback quand c'est possible.

## Ajouter un générateur

Créer une classe qui hérite de l'interface voulue, puis l'enregistrer :

```python
from pokecardgen.generators.base import NameGenerator
from pokecardgen.registry import register_generator

@register_generator("name:mon_generateur")
class MonGenerateurNom(NameGenerator):
    def generate_name(self) -> str:
        return "NomGenere"
```

Importer ce module dans `plugins` du YAML :

```yaml
plugins:
  - pokecardgen.generators.mon_generateur_nom

generators:
  name: "name:mon_generateur"
```
