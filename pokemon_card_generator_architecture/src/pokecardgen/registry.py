from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_GENERATORS: dict[str, type] = {}


def register_generator(name: str) -> Callable[[type[T]], type[T]]:
    """Décorateur pour brancher un nouveau générateur sans modifier le pipeline."""

    def decorator(cls: type[T]) -> type[T]:
        if name in _GENERATORS:
            raise ValueError(f"Un générateur nommé {name!r} existe déjà.")
        _GENERATORS[name] = cls
        return cls

    return decorator


def get_generator_class(name: str) -> type:
    try:
        return _GENERATORS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_GENERATORS)) or "<aucun>"
        raise KeyError(f"Générateur inconnu: {name!r}. Générateurs connus: {known}") from exc


def list_generators() -> list[str]:
    return sorted(_GENERATORS)
