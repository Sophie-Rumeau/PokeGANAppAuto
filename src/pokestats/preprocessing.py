from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import ensure_dir


def to_snake(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("%", "pct")
    name = re.sub(r"[\s\./-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def preprocess_dataset(input_csv: Path | str, out_dir: Path | str = "processed") -> dict[str, Any]:
    input_path = Path(input_csv)
    out_path = ensure_dir(Path(out_dir))

    full_out = out_path / "pokemon_preprocessed_full.csv"
    ctgan_out = out_path / "pokemon_ctgan_train.csv"
    meta_out = out_path / "preprocessing_metadata.json"

    df_raw = pd.read_csv(input_path)
    rename_map = {c: to_snake(c) for c in df_raw.columns}
    df = df_raw.rename(columns=rename_map).copy()

    stats_cols = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    weakness_cols = [c for c in df.columns if c.endswith("_weakness")]
    count_cols = [
        "number_immune",
        "number_not_effective",
        "number_normal",
        "number_super_effective",
    ]

    to_numeric_cols = [
        "id",
        "gen",
        *stats_cols,
        "base_stats",
        *weakness_cols,
        "height_inches",
        "height_meters",
        "weight_pounds",
        "weight_kilograms",
        "capturing_rate",
        "gender_male_ratio",
        "egg_steps",
        "egg_cycles",
        *count_cols,
    ]

    for col in to_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["is_legendary", "is_mythical", "is_ultra_beast"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int).clip(0, 1).astype("int8")

    if "type_2" in df.columns:
        df["type_2"] = df["type_2"].fillna("None").replace("", "None")

    if "forms" in df.columns:
        df["forms"] = df["forms"].fillna("")
        df["has_forms"] = (df["forms"].str.strip() != "").astype("int8")

    if "abilities" in df.columns:
        df["abilities"] = df["abilities"].fillna("")
        df["abilities_count"] = (
            df["abilities"]
            .str.split(";")
            .apply(lambda values: len([a for a in values if str(a).strip()]))
            .astype("int16")
        )

    if "gender_male_ratio" in df.columns:
        median_gender = df["gender_male_ratio"].median()
        df["gender_male_ratio"] = df["gender_male_ratio"].fillna(median_gender)

    if all(c in df.columns for c in stats_cols):
        df["base_stats_calc"] = df[stats_cols].sum(axis=1)

    if "base_stats" in df.columns and "base_stats_calc" in df.columns:
        df["base_stats_diff"] = df["base_stats_calc"] - df["base_stats"]

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in ["type_1", "type_2", "classification_info", "name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    ctgan_categorical = [
        c
        for c in ["type_1", "type_2", "gen", "is_legendary", "is_mythical", "is_ultra_beast"]
        if c in df.columns
    ]

    ctgan_numerical = [
        c
        for c in [
            "hp",
            "attack",
            "defense",
            "sp_attack",
            "sp_defense",
            "speed",
            "base_stats",
            "height_meters",
            "weight_kilograms",
            "capturing_rate",
            "gender_male_ratio",
            "egg_steps",
            "egg_cycles",
            "abilities_count",
            "number_immune",
            "number_not_effective",
            "number_normal",
            "number_super_effective",
        ]
        if c in df.columns
    ]

    ctgan_df = df[ctgan_categorical + ctgan_numerical].copy()
    for col in ctgan_categorical:
        ctgan_df[col] = ctgan_df[col].astype(str)

    df.to_csv(full_out, index=False)
    ctgan_df.to_csv(ctgan_out, index=False)

    metadata = {
        "input_file": str(input_path),
        "full_output_file": str(full_out),
        "ctgan_output_file": str(ctgan_out),
        "n_rows": int(df.shape[0]),
        "n_cols_full": int(df.shape[1]),
        "n_cols_ctgan": int(ctgan_df.shape[1]),
        "ctgan_categorical": ctgan_categorical,
        "ctgan_numerical": ctgan_numerical,
        "rename_map": rename_map,
    }

    with open(meta_out, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=True)

    return {
        "full_df": df,
        "ctgan_df": ctgan_df,
        "full_out": full_out,
        "ctgan_out": ctgan_out,
        "meta_out": meta_out,
        "metadata": metadata,
    }
