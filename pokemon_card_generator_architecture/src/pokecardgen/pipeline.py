from __future__ import annotations

import importlib
from pathlib import Path

from .config import build_runtime_paths, load_config
from .generators.base import GenerationContext
# Imports nécessaires pour enregistrer les générateurs built-in via décorateurs.
from .generators import name as _builtin_name  # noqa: F401
from .generators import stats_ctgan as _builtin_stats  # noqa: F401
from .generators import attacks_csv as _builtin_attacks  # noqa: F401
from .generators import image_latent_diffusion as _builtin_image  # noqa: F401
from .registry import get_generator_class
from .renderers.pil_card import PILCardRenderer
from .schema import GenerationRequest, PokemonCard
from .utils.random import seed_everything


class PokemonCardPipeline:
    """Façade principale: orchestre nom, stats, attaques, image et rendu."""

    def __init__(self, config_path: str | Path | None = None):
        self.config = load_config(config_path)
        self.paths = build_runtime_paths(self.config)
        self.project_root = self.paths.project_root  # compatibilité ancienne API
        self._load_plugins()

    def generate(self, request: GenerationRequest) -> Path:
        seed_everything(request.seed or self.config.runtime.get("seed", None))

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        context = GenerationContext(request=request, config=self.config, paths=self.paths)

        name_gen = self._make_generator("name", context)
        stats_gen = self._make_generator("stats", context)
        attacks_gen = self._make_generator("attacks", context)
        image_gen = self._make_generator("image", context)

        for gen in (name_gen, stats_gen, attacks_gen, image_gen):
            gen.setup()

        card = PokemonCard(
            name=name_gen.generate_name(),
            pokemon_type=request.pokemon_type,
            stats=stats_gen.generate_stats(),
            attacks=attacks_gen.generate_attacks(),
            image_path=image_gen.generate_image(),
        )

        output_path = request.output_path or self.paths.output_dir / f"{card.name}_{card.pokemon_type.value}.png"
        output_path = self.paths.resolve(output_path) if output_path.is_absolute() else self.paths.output_file(output_path)

        renderer_cfg = self.config.get("renderer", {})
        font_path = renderer_cfg.get("font_path", None)
        resolved_font_path = self.paths.resolve(font_path) if font_path else None
        renderer = PILCardRenderer(
            width=int(renderer_cfg.get("width", 744)),
            height=int(renderer_cfg.get("height", 1038)),
            font_path=str(resolved_font_path) if resolved_font_path else None,
        )
        return renderer.render(card, output_path)

    def _load_plugins(self) -> None:
        for module_path in self.config.get("plugins", []):
            importlib.import_module(module_path)

    def _make_generator(self, role: str, context: GenerationContext):
        generator_name = self.config.generators[role]
        cls = get_generator_class(generator_name)
        return cls(context)
