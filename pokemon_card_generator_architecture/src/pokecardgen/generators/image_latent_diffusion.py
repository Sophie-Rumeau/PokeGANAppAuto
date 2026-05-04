from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from .base import ImageGenerator
from ..registry import register_generator
from ..types import TYPE_COLORS
from ..utils.torch_loading import torch_load_any
from ..models.latent_diffusion_64 import (
    ConvVAE,
    LatentUNetConditional,
    LatentDiffusion,
    cond_vector_from_types,
    denorm,
)


@register_generator("image:latent_diffusion")
class LatentDiffusionImageGenerator(ImageGenerator):
    """
    Générateur de sprite 64x64 conditionnel par type.

    Compatible avec les checkpoints du notebook:
    - vae_final.pt / vae_last.pt: dict avec clé 'model' et idéalement 'latent_scale'
    - latent_diffusion_last.pt: dict avec clé 'model' et optionnellement 'ema_model'
    """

    def setup(self) -> None:
        cfg = self.context.config
        runtime_device = cfg.runtime.get("device", "auto")
        if runtime_device == "auto":
            runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(runtime_device)

        self.enabled = bool(cfg.image_generator.get("enabled", True))
        self.allowed_types = set(cfg.image_generator.get("allowed_types", ["Fire", "Water", "Grass", "Electric"]))

        self.vae_path = self.context.paths.model(cfg.models.vae)
        self.diff_path = self.context.paths.model(cfg.models.latent_diffusion)

        self.vae = None
        self.diff_model = None
        self.diffusion = None
        self.latent_scale = 1.0
        self.latent_channels = int(cfg.image_generator.get("latent_channels", 4))

        if not self.enabled:
            return
        if not self.vae_path.exists() or not self.diff_path.exists():
            return

        self._load_models()

    def generate_image(self) -> Path | None:
        out_dir = self.context.paths.output_dir / "sprites"
        out_dir.mkdir(parents=True, exist_ok=True)

        type_name = self.context.request.pokemon_type.value
        out_path = out_dir / f"{type_name.lower()}_sprite.png"

        if self.vae is None or self.diff_model is None or self.diffusion is None:
            self._placeholder_sprite(out_path)
            return out_path

        if type_name not in self.allowed_types:
            self._placeholder_sprite(out_path)
            return out_path

        img = self._sample_sprite(type_name)
        img.save(out_path)
        return out_path

    def _load_models(self) -> None:
        cfg = self.context.config.image_generator

        vae_ckpt = torch_load_any(self.vae_path, map_location=self.device)
        vae_cfg = vae_ckpt.get("config", {}) if isinstance(vae_ckpt, dict) else {}

        vae_base_channels = int(vae_cfg.get("vae_base_channels", cfg.get("vae_base_channels", 64)))
        latent_channels = int(vae_cfg.get("vae_latent_channels", cfg.get("latent_channels", 4)))
        self.latent_channels = latent_channels

        self.vae = ConvVAE(
            in_channels=3,
            base_channels=vae_base_channels,
            latent_channels=latent_channels,
        ).to(self.device)

        vae_state = vae_ckpt["model"] if isinstance(vae_ckpt, dict) and "model" in vae_ckpt else vae_ckpt
        self.vae.load_state_dict(vae_state, strict=True)
        self.vae.eval().requires_grad_(False)

        self.latent_scale = float(vae_ckpt.get("latent_scale", 1.0)) if isinstance(vae_ckpt, dict) else 1.0

        diff_ckpt = torch_load_any(self.diff_path, map_location=self.device)
        diff_cfg = diff_ckpt.get("config", {}) if isinstance(diff_ckpt, dict) else {}

        diff_base_channels = int(diff_cfg.get("diff_base_channels", cfg.get("diff_base_channels", 128)))

        self.diff_model = LatentUNetConditional(
            latent_channels=latent_channels,
            base_channels=diff_base_channels,
            cond_dim=4,
        ).to(self.device)

        state_key = "ema_model" if isinstance(diff_ckpt, dict) and diff_ckpt.get("ema_model") is not None else "model"
        diff_state = diff_ckpt[state_key] if isinstance(diff_ckpt, dict) and state_key in diff_ckpt else diff_ckpt
        self.diff_model.load_state_dict(diff_state, strict=True)
        self.diff_model.eval().requires_grad_(False)

        self.diffusion = LatentDiffusion(
            timesteps=int(diff_cfg.get("diff_timesteps", cfg.get("diff_timesteps", 300))),
            objective=str(diff_cfg.get("diff_objective", cfg.get("diff_objective", "v"))),
            device=str(self.device),
        )

    @torch.no_grad()
    def _sample_sprite(self, type_name: str) -> Image.Image:
        cfg = self.context.config.image_generator
        image_size = int(cfg.get("image_size", 64))
        latent_h = image_size // 4
        latent_w = image_size // 4
        latent_channels = int(self.latent_channels)

        cond = cond_vector_from_types([type_name]).to(self.device)

        latents = self.diffusion.ddim_sample(
            model=self.diff_model,
            shape=(1, latent_channels, latent_h, latent_w),
            steps=int(cfg.get("ddim_steps", 100)),
            eta=float(cfg.get("eta", 0.20)),
            temperature=float(cfg.get("temperature", 1.12)),
            cond=cond,
            guidance_scale=float(cfg.get("guidance_scale", 0.0)),
        )

        decoded = self.vae.decode(latents / self.latent_scale)
        decoded = denorm(decoded[0]).clamp(0, 1)
        arr = (decoded.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(arr, mode="RGB").resize((320, 320), Image.Resampling.NEAREST)

    def _placeholder_sprite(self, out_path: Path) -> None:
        color = TYPE_COLORS[self.context.request.pokemon_type]
        img = Image.new("RGB", (320, 320), color)
        draw = ImageDraw.Draw(img)
        label = self.context.request.pokemon_type.value
        draw.ellipse((70, 50, 250, 230), fill="white")
        draw.text((110, 245), label, fill="black")
        img.save(out_path)
