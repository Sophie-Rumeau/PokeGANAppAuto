from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pokestats.ctgan_pipeline import train_and_sample_ctgan


def summarize_eval(eval_df: pd.DataFrame) -> dict:
    if eval_df.empty:
        return {
            "avg_ks": None,
            "max_ks": None,
            "features_good": 0,
            "features_accept": 0,
            "features_bad": 0,
        }

    avg_ks = float(eval_df["ks_stat"].mean())
    max_ks = float(eval_df["ks_stat"].max())
    good = int((eval_df["ks_stat"] < 0.10).sum())
    accept = int(((eval_df["ks_stat"] >= 0.10) & (eval_df["ks_stat"] < 0.20)).sum())
    bad = int((eval_df["ks_stat"] >= 0.20).sum())

    return {
        "avg_ks": avg_ks,
        "max_ks": max_ks,
        "features_good": good,
        "features_accept": accept,
        "features_bad": bad,
    }


if __name__ == "__main__":
    train_csv = ROOT / "processed/pokemon_ctgan_train.csv"
    artifacts = ROOT / "artifacts"
    epochs_list = [600, 800, 1000]

    rows = []
    for epochs in epochs_list:
        run_dir = artifacts / f"epochs_{epochs}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== CTGAN run: epochs={epochs} ===")
        result = train_and_sample_ctgan(
            train_csv=train_csv,
            artifacts_dir=run_dir,
            epochs=epochs,
        )

        summary = summarize_eval(result["eval_df"])
        row = {
            "epochs": epochs,
            **summary,
            "eval_path": str(result["eval_path"]),
            "model_path": str(result["model_path"]),
            "synth_path": str(result["synth_path"]),
        }
        rows.append(row)

        if result["eval_df"].empty:
            print("No evaluation rows generated.")
        else:
            print(result["eval_df"][['feature', 'ks_stat']].sort_values('ks_stat').to_string(index=False))
            print(
                f"Summary | avg_ks={row['avg_ks']:.4f} | max_ks={row['max_ks']:.4f} | "
                f"good={row['features_good']} accept={row['features_accept']} bad={row['features_bad']}"
            )

    benchmark_df = pd.DataFrame(rows).sort_values("avg_ks", na_position="last")
    out_path = artifacts / "ctgan_benchmark_epochs.csv"
    benchmark_df.to_csv(out_path, index=False)

    print("\n=== Benchmark summary ===")
    print(benchmark_df.to_string(index=False))
    print("\nSaved:", out_path)
