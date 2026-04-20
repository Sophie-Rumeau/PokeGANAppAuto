from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler

from .io_utils import ensure_dir, first_existing_path


def to_snake(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\s\./-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_features = [
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
        "number_immune",
        "number_not_effective",
        "number_normal",
        "number_super_effective",
    ]

    for col in ["name", "type_1", "type_2", "gen", "is_legendary", "is_mythical", "is_ultra_beast", "source"]:
        if col not in out.columns:
            out[col] = "Unknown"

    if "base_stats" not in out.columns and all(c in out.columns for c in ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]):
        out["base_stats"] = out[["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]].sum(axis=1)

    for col in numeric_features:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)

    out["name"] = out["name"].fillna("Unknown").astype(str)
    out["type_1"] = out["type_1"].fillna("Unknown").astype(str)
    out["type_2"] = out["type_2"].fillna("None").replace("", "None").astype(str)
    out["source"] = out["source"].fillna("Real").astype(str)

    for col in ["gen", "is_legendary", "is_mythical", "is_ultra_beast"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round().astype(int)

    return out


def build_umap_artifacts(
    real_candidates: list[Path | str],
    synthetic_csv: Path | str | None = None,
    artifacts_dir: Path | str = "artifacts",
    max_points: int = 5000,
) -> dict[str, Any]:
    import umap

    real_paths = [Path(path) for path in real_candidates]
    real_path = first_existing_path(real_paths)

    real_df = pd.read_csv(real_path).rename(columns=lambda c: to_snake(c))
    real_df["source"] = "Real"

    frames = [_prepare_frame(real_df)]

    if synthetic_csv is not None:
        synth_path = Path(synthetic_csv)
        if synth_path.exists():
            synth_df = pd.read_csv(synth_path).rename(columns=lambda c: to_snake(c))
            synth_df["source"] = "Synthetic"
            frames.append(_prepare_frame(synth_df))

    df_map = pd.concat(frames, ignore_index=True)

    if len(df_map) > max_points:
        source_n = max(df_map["source"].nunique(), 1)
        per_source = max_points // source_n
        df_map = (
            df_map.groupby("source", group_keys=False)
            .apply(lambda frame: frame.sample(min(len(frame), per_source), random_state=42))
            .reset_index(drop=True)
        )

    numeric_features = [
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
        "number_immune",
        "number_not_effective",
        "number_normal",
        "number_super_effective",
    ]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_map[numeric_features].to_numpy())

    umap_model = umap.UMAP(
        n_neighbors=25,
        min_dist=0.12,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    embedding = umap_model.fit_transform(x_scaled)

    df_map["umap_1"] = embedding[:, 0]
    df_map["umap_2"] = embedding[:, 1]

    hover_cols = [
        c
        for c in ["name", "type_1", "type_2", "gen", "source", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "base_stats"]
        if c in df_map.columns
    ]

    fig_type = px.scatter(
        df_map,
        x="umap_1",
        y="umap_2",
        color="type_1",
        symbol="source",
        hover_data=hover_cols,
        opacity=0.8,
        title="Carte UMAP Pokemon - couleur par Type 1",
    )
    fig_type.update_traces(marker={"size": 7})
    fig_type.update_layout(template="plotly_white", legend_title_text="Type / Source")

    fig_gen = px.scatter(
        df_map,
        x="umap_1",
        y="umap_2",
        color=df_map["gen"].astype(str),
        symbol="source",
        hover_data=hover_cols,
        opacity=0.8,
        title="Carte UMAP Pokemon - couleur par generation",
    )
    fig_gen.update_traces(marker={"size": 7})
    fig_gen.update_layout(template="plotly_white", legend_title_text="Generation / Source")

    artifacts_path = ensure_dir(Path(artifacts_dir))
    html_type = artifacts_path / "umap_pokemon_map.html"
    html_gen = artifacts_path / "umap_pokemon_map_by_gen.html"
    csv_projection = artifacts_path / "pokemon_umap_projection.csv"

    fig_type.write_html(html_type, include_plotlyjs="cdn")
    fig_gen.write_html(html_gen, include_plotlyjs="cdn")
    df_map.to_csv(csv_projection, index=False)

    return {
        "df_map": df_map,
        "fig_type": fig_type,
        "fig_gen": fig_gen,
        "real_path": real_path,
        "html_type": html_type,
        "html_gen": html_gen,
        "csv_projection": csv_projection,
    }
