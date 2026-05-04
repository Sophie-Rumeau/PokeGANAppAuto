from __future__ import annotations

import random

from .base import NameGenerator
from ..registry import register_generator


_PREFIXES = {
    "Fire": ["Pyro", "Flam", "Cendra", "Ignis"],
    "Water": ["Aqua", "Nauti", "Bruma", "Ond"],
    "Grass": ["Flora", "Moss", "Liana", "Bud"],
    "Electric": ["Volt", "Zap", "Spark", "Tesla"],
}


_SUFFIXES = ["mon", "rix", "chu", "dile", "zor", "leaf", "bolt", "ling"]


@register_generator("name:random")
class RandomNameGenerator(NameGenerator):
    """Fallback. Remplaçable par un générateur de noms entraîné."""

    def generate_name(self) -> str:
        hint = self.context.request.name_hint
        if hint:
            return hint

        type_name = self.context.request.pokemon_type.value
        prefixes = _PREFIXES.get(type_name, [type_name[:4]])
        return random.choice(prefixes) + random.choice(_SUFFIXES)
