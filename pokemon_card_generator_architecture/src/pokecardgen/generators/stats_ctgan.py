from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import StatsGenerator
from ..registry import register_generator
from ..schema import CardStats
from ..utils.torch_loading import torch_load_any


_STAT_COLS = ["HP", "Attack", "Defense", "Sp. Attack", "Sp. Defense", "Speed"]


@register_generator("stats:ctgan")
class CTGANStatsGenerator(StatsGenerator):
    """
    Génère les stats via un CTGAN sauvegardé complet.

    Important: le fichier ctgan_model_full.pt est préférable à generator_state.pt,
    car sample() dépend aussi du transformer et des métadonnées internes CTGAN.
    """

    def setup(self) -> None:
        cfg = self.context.config
        model_path = self.context.paths.model(cfg.models.ctgan_full)
        self.model_path = model_path
        self.ctgan: Any | None = None

        if model_path.exists():
            self.ctgan = torch_load_any(model_path, map_location="cpu")

    def generate_stats(self) -> CardStats:
        if self.ctgan is None:
            return self._fallback_stats()

        target_type = self.context.request.pokemon_type.value
        max_attempts = int(self.context.config.stats_generator.get("max_attempts_for_type", 64))

        best_row = None
        for _ in range(max_attempts):
            sample = self.ctgan.sample(1)
            row = sample.iloc[0]
            best_row = row
            if str(row.get("Type 1", "")).lower() == target_type.lower():
                break

        if best_row is None:
            return self._fallback_stats()

        row = self._clean_row(best_row)
        hp_game = int(row.get("HP", 60))
        hp_card = int(np.clip(hp_game * 1.5, 40, 180))

        defense = int(row.get("Defense", 60))
        sp_defense = int(row.get("Sp. Defense", 60))
        retreat = int(np.clip((defense + sp_defense) / 100, 0, 4))

        legendary_raw = row.get("Is_Legendary", False)
        legendary = str(legendary_raw).lower() in {"true", "1", "yes"} or legendary_raw is True

        return CardStats(hp=hp_card, retreat=retreat, legendary=legendary)

    @staticmethod
    def _clean_row(row):
        for col in _STAT_COLS:
            try:
                row[col] = int(np.clip(float(row[col]), 10, 255))
            except Exception:
                row[col] = 60
        return row

    def _fallback_stats(self) -> CardStats:
        return CardStats(
            hp=int(np.random.choice([60, 70, 80, 90, 100, 110, 120])),
            retreat=int(np.random.choice([0, 1, 1, 2, 2, 3])),
            legendary=False,
        )
