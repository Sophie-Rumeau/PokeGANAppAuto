# Stats module structure

This folder is now organized to keep notebooks light and move core logic into reusable Python modules.

## Structure

- pokemon_data.csv: source dataset (Dataset 2)
- ctGanStats.ipynb: light notebook for preprocessing + CTGAN orchestration
- umapCarteInteractive.ipynb: light notebook for UMAP interactive map orchestration
- requirements.txt: quick dependency install
- src/pokestats/: reusable Python functions
- scripts/: command line runners
- processed/: generated clean datasets
- artifacts/: generated model and visual outputs

## Quick start

Run from the Stats folder.

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Preprocess data

```bash
python scripts/run_preprocessing.py
```

1. (Optional) Benchmark CTGAN epochs to find optimal training duration

```bash
python scripts/benchmark_ctgan_epochs.py
```

This tests epochs 600, 800, 1000 and compares metrics. Results saved to `artifacts/ctgan_benchmark_epochs.csv`.

1. Train CTGAN and generate synthetic data

```bash
python scripts/train_ctgan.py
```

1. Build UMAP interactive map

```bash
python scripts/build_umap_map.py
```

## Python modules

- src/pokestats/preprocessing.py
  - preprocess_dataset(input_csv, out_dir): clean data + export train set for CTGAN

- src/pokestats/ctgan_pipeline.py
  - train_and_sample_ctgan(...): train CTGAN, sample synthetic data, export quick eval

- src/pokestats/umap_map.py
  - build_umap_artifacts(...): build UMAP projection + export interactive HTML maps

## Outputs

### Preprocessing

- processed/pokemon_preprocessed_full.csv
- processed/pokemon_ctgan_train.csv
- processed/preprocessing_metadata.json

### CTGAN

- artifacts/ctgan_pokemon.pkl
- artifacts/pokemon_synthetic.csv
- artifacts/pokemon_synthetic_fire_gen1.csv
- artifacts/ctgan_eval_summary.csv (if available)
- artifacts/ctgan_benchmark_epochs.csv (from benchmark script; compares 600/800/1000 epochs)

### UMAP

- artifacts/umap_pokemon_map.html
- artifacts/umap_pokemon_map_by_gen.html
- artifacts/pokemon_umap_projection.csv
- artifacts/umap_pokemon_map_publication.html (from notebook publication section)

## Notes

- The notebooks are intentionally minimal and call functions from src/pokestats.
- For reproducibility, random_state/seed is set inside the module logic where relevant.
- You can tune epochs and model params in src/pokestats/ctgan_pipeline.py or from notebook cells.
