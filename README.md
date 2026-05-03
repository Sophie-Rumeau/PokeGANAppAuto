# PokeGANAppAuto

## Ce que j'ai fait dans ce dossier

Ce dossier correspond a ma phase d'experimentation autour de la creation de donnees et des essais GAN sur des sprites Pokemon.

## Travail realise

1. Constitution d'une base de donnees de travail

- J'ai prepare un fichier tabulaire Pokemon pour centraliser les informations utiles aux tests:
	- `spriteTestDataSet/pokemonDB_dataset.csv`
- Ce CSV m'a servi de point d'entree commun pour eviter de repartir de zero entre les essais.

2. Iterations sur plusieurs versions de notebooks StyleGAN

- J'ai produit une suite de notebooks numerotes pour garder une trace des variantes testees:
	- `spriteTestDataSet/styleGAN001.ipynb` a `spriteTestDataSet/styleGAN009.ipynb`
- La numerotation represente l'evolution des essais (preparation des donnees, choix de parametres, structure du notebook, observations sur la qualite des sorties).

## Modifications entre chaque iteration

- styleGAN001 -> styleGAN002
	- Changement de source d'images: `../testMohamed/archive32/PokemonImagesDB` vers `../testMohamed/archive32/PokemonImagesDBNew`.

- styleGAN002 -> styleGAN003
	- Changement de source d'images: `../testMohamed/archive32/PokemonImagesDBNew` vers `./pokemons/sprites/pokemon-gen8/regular`.

- styleGAN003 -> styleGAN004
	- Changement de source d'images: `./pokemons/sprites/pokemon-gen8/regular` vers `../testCreateDataSet/pokemonResize`.

- styleGAN004 -> styleGAN005
	- Passage explicite en RGBA:
		- normalisation de 1 canal vers 4 canaux,
		- ajout d'un loader RGBA (`Image.open(...).convert("RGBA")`),
		- adaptation des couches d'entree/sortie (3 canaux -> 4 canaux) pour G et D.
	- Ajustement du DataLoader (`num_workers=2` -> `num_workers=0`).
	- Amelioration de la visualisation finale: affichage RGBA, masque alpha et grille RGB.

- styleGAN005 -> styleGAN006
	- Passage de `image_size=64` a `image_size=128`.
	- Preprocessing rendu dynamique (`Resize(image_size)`, `CenterCrop(image_size)`).
	- Introduction d'un mapping `dataset_roots` (128/256) avec validation de la taille.
	- Refonte de l'architecture G/D en version dynamique selon la resolution cible (construction des blocs en boucle + validation puissance de 2).

- styleGAN006 -> styleGAN007
	- Augmentation de la duree d'entrainement: `epochs=50` -> `epochs=200`.

- styleGAN007 -> styleGAN008
	- Reduction de la duree d'entrainement: `epochs=200` -> `epochs=10`.
	- Abandon du mapping `dataset_roots` et retour a un chemin fixe: `../testCreateDataSet/pokemon`.
	- Ajout d'une sauvegarde explicite du modele (`stylegan_rgba.pth`).

- styleGAN008 -> styleGAN009
	- Passage d'un dataset image simple a un dataset conditionnel image+type (`PokemonTypeDataset`) base sur `pokemonDB_dataset.csv`.
	- Ajout d'une normalisation des noms Pokemon (accents, caracteres speciaux, suffixe `_new`) pour faire correspondre CSV et images.
	- Retour a `epochs=50`.
	- Conditionnement du modele par type:
		- embedding de type dans le Mapping Network,
		- embedding de type concatene dans le Discriminator,
		- entrainement de G/D avec `type_idx`.
	- Generation d'echantillons avec types aleatoires et affichage des types utilises.
	- Remplacement du bloc de sauvegarde final par un affichage des types echantillonnes.

3. Mise en place d'une logique d'experimentation

- Au lieu d'un seul notebook, j'ai volontairement separe les tentatives pour:
	- comparer les approches,
	- identifier ce qui ameliore ou degrade les resultats,
	- conserver l'historique technique du travail.

## Resultat de cette etape

- Un dataset de reference exploitable pour les tests.
- Un historique d'essais StyleGAN documente par versions.
- Une base solide pour selectionner/industrialiser la meilleure variante dans une etape suivante du projet.