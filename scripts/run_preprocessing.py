from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pokestats.preprocessing import preprocess_dataset


if __name__ == "__main__":
    input_csv = ROOT / "pokemon_data.csv"
    out_dir = ROOT / "processed"

    result = preprocess_dataset(input_csv=input_csv, out_dir=out_dir)
    print("Preprocessing OK")
    print("-", result["full_out"])
    print("-", result["ctgan_out"])
    print("-", result["meta_out"])
