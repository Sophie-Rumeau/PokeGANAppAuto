from __future__ import annotations

from enum import Enum


class PokemonType(str, Enum):
    NORMAL = "Normal"
    FIRE = "Fire"
    WATER = "Water"
    GRASS = "Grass"
    ELECTRIC = "Electric"
    ICE = "Ice"
    FIGHTING = "Fighting"
    POISON = "Poison"
    GROUND = "Ground"
    FLYING = "Flying"
    PSYCHIC = "Psychic"
    BUG = "Bug"
    ROCK = "Rock"
    GHOST = "Ghost"
    DRAGON = "Dragon"
    DARK = "Dark"
    STEEL = "Steel"
    FAIRY = "Fairy"


TYPE_ALIASES = {
    "feu": PokemonType.FIRE,
    "fire": PokemonType.FIRE,
    "eau": PokemonType.WATER,
    "water": PokemonType.WATER,
    "plante": PokemonType.GRASS,
    "grass": PokemonType.GRASS,
    "électrique": PokemonType.ELECTRIC,
    "electrique": PokemonType.ELECTRIC,
    "electric": PokemonType.ELECTRIC,
    "normal": PokemonType.NORMAL,
    "glace": PokemonType.ICE,
    "ice": PokemonType.ICE,
    "combat": PokemonType.FIGHTING,
    "fighting": PokemonType.FIGHTING,
    "poison": PokemonType.POISON,
    "sol": PokemonType.GROUND,
    "ground": PokemonType.GROUND,
    "vol": PokemonType.FLYING,
    "flying": PokemonType.FLYING,
    "psy": PokemonType.PSYCHIC,
    "psychic": PokemonType.PSYCHIC,
    "insecte": PokemonType.BUG,
    "bug": PokemonType.BUG,
    "roche": PokemonType.ROCK,
    "rock": PokemonType.ROCK,
    "spectre": PokemonType.GHOST,
    "ghost": PokemonType.GHOST,
    "dragon": PokemonType.DRAGON,
    "ténèbres": PokemonType.DARK,
    "tenebres": PokemonType.DARK,
    "dark": PokemonType.DARK,
    "acier": PokemonType.STEEL,
    "steel": PokemonType.STEEL,
    "fée": PokemonType.FAIRY,
    "fee": PokemonType.FAIRY,
    "fairy": PokemonType.FAIRY,
}


TYPE_COLORS = {
    PokemonType.NORMAL: "#D9D6C7",
    PokemonType.FIRE: "#F4A261",
    PokemonType.WATER: "#73A9E6",
    PokemonType.GRASS: "#7BC96F",
    PokemonType.ELECTRIC: "#F7D560",
    PokemonType.ICE: "#A8DADC",
    PokemonType.FIGHTING: "#C97A6B",
    PokemonType.POISON: "#B58AD9",
    PokemonType.GROUND: "#D2B48C",
    PokemonType.FLYING: "#B8C7FF",
    PokemonType.PSYCHIC: "#F48FB1",
    PokemonType.BUG: "#B7C65A",
    PokemonType.ROCK: "#B8A16A",
    PokemonType.GHOST: "#8E7CC3",
    PokemonType.DRAGON: "#8A7FF0",
    PokemonType.DARK: "#6B5B53",
    PokemonType.STEEL: "#B0BEC5",
    PokemonType.FAIRY: "#F3B5D8",
}


def normalize_type(value: str | PokemonType) -> PokemonType:
    if isinstance(value, PokemonType):
        return value
    key = str(value).strip().lower()
    if key not in TYPE_ALIASES:
        valid = ", ".join(t.value for t in PokemonType)
        raise ValueError(f"Type inconnu: {value!r}. Types supportés: {valid}")
    return TYPE_ALIASES[key]
