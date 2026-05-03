# PokeGANAppAuto

## AEGAN: Adversarial Encoder-Generator Autoencoder

### Ce que j'ai fait dans cette section

J'ai expérimenté un modèle **AEGAN** (Adversarial Encoder-Generator Autoencoder) sur les sprites Pokémon, en testant plusieurs configurations pour optimiser la qualité et la stabilité de l'encodage/décodage.

### Livrables

- 9 notebooks d'expérimentation: `aegan001.ipynb` à `aegan009.ipynb`
- 8 checkpoints sauvegardés dans les répertoires `checkpoints_aegan*`
- Architecture dynamique supportant plusieurs résolutions (64, 128)

### Modifications entre chaque itération

#### aegan001 → aegan002 → aegan003
- **Configuration de base** : image_size=128, batch_size=64, epochs=20
- Paramètres de reconstruction : lambda_rx=10.0, lambda_rz=1.0
- Premiers essais du pipeline AEGAN sur 128x128
- Variations mineures dans les données ou l'ordre des cellules

#### aegan003 → aegan004
- Ajout du **Gradient Penalty (GP)** pour améliorer la stabilité du discriminateur:
  - lambda_gp_x = 10.0 (pénalité gradient entrée réelle)
  - lambda_gp_z = 2.0 (pénalité gradient latent)
- Reste : image_size=128, batch_size=64, epochs=20

#### aegan004 → aegan005
- Augmentation drastique de la durée d'entrainement: **epochs=20 → epochs=200**
- Objectif : laisser converger plus lentement avec les gradients pénalisés
- Paramètres : image_size=128, batch_size=64, lambda_rx=10.0, lambda_rz=1.0

#### aegan005 → aegan006
- Maintien de la configuration longue durée (200 epochs)
- Variations expérimentales dans les hyperparamètres de la loss ou du réseau

#### aegan006 → aegan007
- **Réduction drastique en résolution** : image_size=128 → image_size=64
- Réduction du batch_size : batch_size=64 → batch_size=16
- Réinitialisation des epochs : epochs=200 → epochs=20
- Rationale: tester un modèle plus léger et plus rapide

#### aegan007 → aegan008
- Même configuration résolution/batch (64x64, batch=16)
- Augmentation epochs : epochs=20 → epochs=200
- Permettre la convergence du modèle plus léger

#### aegan008 → aegan009
- Même résolution et batch (64x64, batch=16)
- **Entrainement très long** : epochs=200 → **epochs=1000**
- Objectif : obtenir une convergence maximale et étudier le plateau de performance

### Architecture AEGAN

Le modèle suit le schéma:
- **Encoder**: compresse les images (128 ou 64) vers un vecteur latent
- **Generator (Decoder)**: reconstruit les images à partir du latent
- **Discriminator X**: clasifie images réelles vs générées
- **Discriminator Z**: clasifie encodages réels vs générés
- **Loss** : combinaison de reconstruction (L1 ou L2) + adversarial + (optionnellement) gradient penalty

### Résultats de cette étape

- Compréhension du comportement AEGAN selon la résolution et la durée d'entraînement
- Identification de la stabilité: 64x64 avec GP et 200+ epochs semble plus stable
- Baseline solide pour futures optimisations (normalization, architecture améliorée)