from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import RuntimePaths
from ..schema import Attack, CardStats, GenerationRequest


@dataclass
class GenerationContext:
    request: GenerationRequest
    config: Any
    paths: RuntimePaths

    @property
    def project_root(self) -> Path:
        # Compatibilité avec les plugins qui utilisaient context.project_root.
        return self.paths.project_root


class BaseGenerator(ABC):
    """Contrat minimal commun à tous les générateurs."""

    def __init__(self, context: GenerationContext):
        self.context = context

    def setup(self) -> None:
        """Chargement des poids / CSV. Appelé une fois au démarrage du pipeline."""


class NameGenerator(BaseGenerator):
    @abstractmethod
    def generate_name(self) -> str:
        raise NotImplementedError


class StatsGenerator(BaseGenerator):
    @abstractmethod
    def generate_stats(self) -> CardStats:
        raise NotImplementedError


class AttackGenerator(BaseGenerator):
    @abstractmethod
    def generate_attacks(self) -> list[Attack]:
        raise NotImplementedError


class ImageGenerator(BaseGenerator):
    @abstractmethod
    def generate_image(self) -> Path | None:
        raise NotImplementedError
