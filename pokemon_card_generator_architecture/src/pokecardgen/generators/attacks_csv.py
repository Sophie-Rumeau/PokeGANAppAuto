from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base import AttackGenerator
from ..registry import register_generator
from ..schema import Attack


def _safe_literal_eval(value: Any) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _parse_damage(value: Any) -> int | str:
    if value is None:
        return 0
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return 0
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[0])
    return text


@register_generator("attacks:csv")
class CSVAttackGenerator(AttackGenerator):
    """Réutilise les attaques du CSV TCG et filtre en priorité par type."""

    def setup(self) -> None:
        csv_path = self.context.paths.data_file(self.context.config.paths.tcg_cards_csv)
        self.attacks_df = None

        if not csv_path.exists():
            return

        df_cards = pd.read_csv(csv_path)
        if "attacks" not in df_cards.columns:
            return

        df = df_cards.dropna(subset=["attacks"]).copy()
        df["attacks"] = df["attacks"].apply(_safe_literal_eval)
        df = df.explode("attacks").dropna(subset=["attacks"]).reset_index(drop=True)

        rows = []
        for _, row in df.iterrows():
            attack = row["attacks"]
            if not isinstance(attack, dict):
                continue
            rows.append({
                "type": row.get("types", row.get("type", "")),
                "name": attack.get("name", "Attaque"),
                "damage": attack.get("damage", 0),
                "text": attack.get("text", ""),
                "cost": attack.get("cost", []),
            })
        self.attacks_df = pd.DataFrame(rows)

    def generate_attacks(self) -> list[Attack]:
        count = int(self.context.config.attacks_generator.get("count", 2))
        if self.attacks_df is None or self.attacks_df.empty:
            return self._fallback_attacks(count)

        target_type = self.context.request.pokemon_type.value
        type_mask = self.attacks_df["type"].astype(str).str.contains(target_type, case=False, na=False)
        subset = self.attacks_df[type_mask]
        if subset.empty:
            subset = self.attacks_df

        sample = subset.sample(min(count, len(subset)), replace=len(subset) < count)
        attacks: list[Attack] = []
        for _, row in sample.iterrows():
            cost = row.get("cost", [])
            if isinstance(cost, str):
                parsed = _safe_literal_eval(cost)
                cost = parsed if isinstance(parsed, list) else [cost]
            attacks.append(
                Attack(
                    name=str(row.get("name", "Attaque")),
                    damage=_parse_damage(row.get("damage", 0)),
                    effect=str(row.get("text", "") or ""),
                    cost=list(cost) if isinstance(cost, list) else [],
                )
            )
        return attacks

    def _fallback_attacks(self, count: int) -> list[Attack]:
        type_name = self.context.request.pokemon_type.value
        pool = [
            Attack(name=f"Élan {type_name}", damage=20, effect=""),
            Attack(name=f"Impact {type_name}", damage=40, effect="Lance une pièce. Si face, +20 dégâts."),
            Attack(name=f"Rafale {type_name}", damage=60, effect=""),
        ]
        random.shuffle(pool)
        return pool[:count]
