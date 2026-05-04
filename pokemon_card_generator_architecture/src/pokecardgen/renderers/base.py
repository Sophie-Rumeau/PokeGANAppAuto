from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schema import PokemonCard


class CardRenderer(ABC):
    @abstractmethod
    def render(self, card: PokemonCard, output_path: Path) -> Path:
        raise NotImplementedError
