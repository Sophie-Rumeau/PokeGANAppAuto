from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pokestats.umap_map import build_umap_artifacts


if __name__ == "__main__":
    result = build_umap_artifacts(
        real_candidates=[
            ROOT / "processed/pokemon_preprocessed_full.csv",
            ROOT / "pokemon_data.csv",
        ],
        synthetic_csv=ROOT / "artifacts/pokemon_synthetic.csv",
        artifacts_dir=ROOT / "artifacts",
    )

    print("UMAP map OK")
    print("-", result["html_type"])
    print("-", result["html_gen"])
    print("-", result["csv_projection"])
