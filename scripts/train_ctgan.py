from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pokestats.ctgan_pipeline import train_and_sample_ctgan


if __name__ == "__main__":
    train_csv = ROOT / "processed/pokemon_ctgan_train.csv"
    artifacts = ROOT / "artifacts"

    result = train_and_sample_ctgan(
        train_csv=train_csv,
        artifacts_dir=artifacts,
        epochs=300,
    )

    print("CTGAN OK")
    print("-", result["model_path"])
    print("-", result["synth_path"])
    print("-", result["synth_cond_path"])
    if not result["eval_df"].empty:
        print("-", result["eval_path"])
