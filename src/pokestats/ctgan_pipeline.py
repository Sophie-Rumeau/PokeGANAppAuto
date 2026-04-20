from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from .io_utils import ensure_dir


def _safe_conditional_sample(model: Any, n: int, condition_column: str, condition_value: str) -> pd.DataFrame:
    try:
        return model.sample(n, condition_column=condition_column, condition_value=condition_value)
    except TypeError:
        fallback = model.sample(max(n * 5, 2000))
        if condition_column in fallback.columns:
            fallback = fallback[fallback[condition_column].astype(str) == str(condition_value)]
        return fallback.head(n)


def train_and_sample_ctgan(
    train_csv: Path | str,
    artifacts_dir: Path | str = "artifacts",
    epochs: int = 300,
    batch_size: int = 128,
    pac: int = 8,
    generator_dim: tuple[int, int] = (256, 256),
    discriminator_dim: tuple[int, int] = (256, 256),
    conditional_type: str = "Fire",
    conditional_gen: int = 1,
    conditional_n: int = 200,
) -> dict[str, Any]:
    from ctgan import CTGAN

    train_path = Path(train_csv)
    artifacts_path = ensure_dir(Path(artifacts_dir))

    model_path = artifacts_path / "ctgan_pokemon.pkl"
    synth_path = artifacts_path / "pokemon_synthetic.csv"
    synth_cond_path = artifacts_path / "pokemon_synthetic_fire_gen1.csv"
    eval_path = artifacts_path / "ctgan_eval_summary.csv"

    train_df = pd.read_csv(train_path)

    preferred_discrete = ["type_1", "type_2", "gen", "is_legendary", "is_mythical", "is_ultra_beast"]
    discrete_columns = [c for c in preferred_discrete if c in train_df.columns]

    for col in discrete_columns:
        train_df[col] = train_df[col].astype(str)

    # CTGAN discriminator requires batch_size % pac == 0.
    if pac <= 0:
        raise ValueError("pac must be a positive integer")
    if batch_size < pac:
        batch_size = pac
    if batch_size % pac != 0:
        adjusted_batch_size = (batch_size // pac) * pac
        if adjusted_batch_size == 0:
            adjusted_batch_size = pac
        print(
            f"[CTGAN] batch_size={batch_size} is not divisible by pac={pac}. "
            f"Using batch_size={adjusted_batch_size}."
        )
        batch_size = adjusted_batch_size

    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        pac=pac,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        verbose=True,
    )
    model.fit(train_df, discrete_columns=discrete_columns)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    synth_df = model.sample(len(train_df))
    for col in ["gen", "is_legendary", "is_mythical", "is_ultra_beast"]:
        if col in synth_df.columns:
            synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").round().fillna(0).astype(int)
            if col.startswith("is_"):
                synth_df[col] = synth_df[col].clip(0, 1)
    synth_df.to_csv(synth_path, index=False)

    cond_df = _safe_conditional_sample(model, conditional_n, "type_1", conditional_type)
    if "gen" in cond_df.columns:
        cond_df = cond_df[
            pd.to_numeric(cond_df["gen"], errors="coerce").round().astype("Int64") == conditional_gen
        ]
    cond_df.to_csv(synth_cond_path, index=False)

    stats_to_check = [
        c for c in ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "base_stats"]
        if c in train_df.columns and c in synth_df.columns
    ]

    eval_rows = []
    for col in stats_to_check:
        real = pd.to_numeric(train_df[col], errors="coerce").dropna()
        fake = pd.to_numeric(synth_df[col], errors="coerce").dropna()
        if len(real) == 0 or len(fake) == 0:
            continue
        ks_stat, ks_pvalue = ks_2samp(real, fake)
        eval_rows.append(
            {
                "feature": col,
                "real_mean": real.mean(),
                "fake_mean": fake.mean(),
                "real_std": real.std(),
                "fake_std": fake.std(),
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
            }
        )

    eval_df = pd.DataFrame(eval_rows).sort_values("ks_stat") if eval_rows else pd.DataFrame()
    if not eval_df.empty:
        eval_df.to_csv(eval_path, index=False)

    return {
        "train_df": train_df,
        "synth_df": synth_df,
        "cond_df": cond_df,
        "eval_df": eval_df,
        "model_path": model_path,
        "synth_path": synth_path,
        "synth_cond_path": synth_cond_path,
        "eval_path": eval_path,
        "discrete_columns": discrete_columns,
    }
