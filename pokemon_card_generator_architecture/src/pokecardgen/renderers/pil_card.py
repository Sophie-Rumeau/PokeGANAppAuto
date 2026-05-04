from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .base import CardRenderer
from ..schema import PokemonCard
from ..types import TYPE_COLORS


def _font(size: int, font_path: str | None = None):
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


class PILCardRenderer(CardRenderer):
    """Rendu simple et custom, volontairement indépendant des assets officiels."""

    def __init__(self, width: int = 744, height: int = 1038, font_path: str | None = None):
        self.width = width
        self.height = height
        self.font_path = font_path

    def render(self, card: PokemonCard, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bg = card.bg_color
        img = Image.new("RGB", (self.width, self.height), bg)
        draw = ImageDraw.Draw(img)

        margin = 42
        card_box = (margin, margin, self.width - margin, self.height - margin)
        draw.rounded_rectangle(card_box, radius=36, fill=bg, outline="#2C2C2C", width=6)

        # Header
        title_font = _font(46, self.font_path)
        hp_font = _font(34, self.font_path)
        type_font = _font(24, self.font_path)

        draw.text((70, 72), card.name, fill="#1F1F1F", font=title_font)
        draw.text((self.width - 210, 80), f"HP {card.stats.hp}", fill="#1F1F1F", font=hp_font)
        draw.rounded_rectangle((70, 135, 230, 178), radius=18, fill="#FFFFFF", outline="#333333", width=2)
        draw.text((92, 142), card.pokemon_type.value, fill="#1F1F1F", font=type_font)

        if card.stats.legendary:
            draw.rounded_rectangle((245, 135, 390, 178), radius=18, fill="#FFF7C2", outline="#333333", width=2)
            draw.text((266, 142), "Legendary", fill="#1F1F1F", font=type_font)

        # Sprite area
        art_box = (82, 205, self.width - 82, 615)
        draw.rounded_rectangle(art_box, radius=28, fill="#F8F8F8", outline="#333333", width=4)
        if card.image_path and Path(card.image_path).exists():
            sprite = Image.open(card.image_path).convert("RGBA")
            sprite.thumbnail((360, 360), Image.Resampling.LANCZOS)
            x = (self.width - sprite.width) // 2
            y = 235 + (340 - sprite.height) // 2
            img.paste(sprite, (x, y), sprite if sprite.mode == "RGBA" else None)

        # Attacks
        attack_font = _font(30, self.font_path)
        effect_font = _font(20, self.font_path)
        y = 655
        for attack in card.attacks[:2]:
            draw.line((90, y - 18, self.width - 90, y - 18), fill="#333333", width=3)
            draw.text((100, y), attack.name, fill="#111111", font=attack_font)
            draw.text((self.width - 170, y), str(attack.damage), fill="#111111", font=attack_font)
            effect = attack.effect or ""
            wrapped = self._wrap(effect, max_chars=58)
            draw.text((100, y + 42), wrapped, fill="#202020", font=effect_font)
            y += 130

        # Footer
        footer_font = _font(20, self.font_path)
        draw.line((90, self.height - 150, self.width - 90, self.height - 150), fill="#333333", width=2)
        draw.text((100, self.height - 125), f"Retraite: {card.stats.retreat}", fill="#111111", font=footer_font)
        draw.text((self.width - 250, self.height - 125), "Carte générée", fill="#111111", font=footer_font)

        img.save(output_path)
        return output_path

    @staticmethod
    def _wrap(text: str, max_chars: int) -> str:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > max_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        return "\n".join(lines[:3])
