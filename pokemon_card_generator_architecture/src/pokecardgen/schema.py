from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import PokemonType, TYPE_COLORS, normalize_type


@dataclass(frozen=True)
class GenerationRequest:
    pokemon_type: PokemonType
    output_path: Path | None = None
    seed: int | None = None
    name_hint: str | None = None

    @classmethod
    def from_type(cls, pokemon_type: str | PokemonType, **kwargs: Any) -> "GenerationRequest":
        return cls(pokemon_type=normalize_type(pokemon_type), **kwargs)


@dataclass
class CardStats:
    hp: int
    retreat: int = 1
    legendary: bool = False


@dataclass
class Attack:
    name: str
    damage: int | str
    effect: str = ""
    cost: list[str] = field(default_factory=list)


@dataclass
class PokemonCard:
    name: str
    pokemon_type: PokemonType
    stats: CardStats
    attacks: list[Attack]
    image_path: Path | None = None
    background_color: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bg_color(self) -> str:
        return self.background_color or TYPE_COLORS[self.pokemon_type]
