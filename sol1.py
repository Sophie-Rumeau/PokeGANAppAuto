''' OPTION 1 — SD img2img + Pokémon LoRA '''

import argparse
import random
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ─── Device ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32
print(f"  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════════════════════
# 1. BACKGROUND REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def remove_bg(path: Path, size: int = 512) -> Image.Image:
    """Remove background with rembg, return RGB on white."""
    try:
        from rembg import remove
        import io
        print(f"    removing background...")
        with open(path, "rb") as f:
            out = remove(f.read())
        img = Image.open(io.BytesIO(out)).convert("RGBA")
    except ImportError:
        print("    [warn] rembg not installed — keeping background")
        img = Image.open(path).convert("RGBA")
    except Exception as e:
        print(f"    [warn] rembg failed ({e})")
        img = Image.open(path).convert("RGBA")

    # Compose on white
    bg = Image.new("RGB", img.size, (255, 255, 255))
    if img.mode == "RGBA":
        bg.paste(img, mask=img.split()[3])
    else:
        bg = img.convert("RGB")

    # Resize to square
    bg = bg.resize((size, size), Image.LANCZOS)
    return bg


# ══════════════════════════════════════════════════════════════════════════════
# 2. PIPELINE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(lora_path: str = None, use_hf_pokemon: bool = True):
    """
    Load the SD img2img pipeline.

    Two modes:
      A) use_hf_pokemon=True  → loads lambdalabs/sd-pokemon-diffusers directly
         (already fine-tuned on Pokémon, no LoRA needed)
      B) use_hf_pokemon=False → loads SD 1.5 base + applies your LoRA weights
    """
    from diffusers import StableDiffusionImg2ImgPipeline

    if use_hf_pokemon:
        # lambdalabs/sd-pokemon-diffusers is SD 1.4 fine-tuned on 833 Pokémon images
        # with BLIP captions — generates Pokémon-style images natively
        print("  Loading lambdalabs/sd-pokemon-diffusers...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "lambdalabs/sd-pokemon-diffusers",
            torch_dtype=DTYPE,
            safety_checker=None,
        )
    else:
        print("  Loading SD 1.5 base...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=DTYPE,
            safety_checker=None,
        )

    if lora_path and Path(lora_path).exists():
        print(f"  Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=0.9)

    pipe = pipe.to(DEVICE)

    # Memory optimizations
    if DEVICE.type == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers enabled")
        except Exception:
            pass

    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# Prompts tuned for Pokémon style
POSITIVE_PROMPTS = [
    "a cute Pokemon creature, Ken Sugimori art style, official Pokemon artwork, "
    "white background, clean cartoon illustration, vibrant colors, "
    "sharp outlines, game freak style",

    "new Pokemon design, anime style, official Ken Sugimori illustration, "
    "colorful cartoon creature, white background, clean linework",
]

NEGATIVE_PROMPT = (
    "photo, realistic, photograph, 3d render, blurry, noisy, dark, "
    "human, person, text, watermark, ugly, deformed, low quality, "
    "grainy, photorealistic, real animal fur"
)


def generate(
    pipe,
    animal_img  : Image.Image,
    prompt      : str,
    strength    : float = 0.65,
    guidance    : float = 9.0,
    steps       : int   = 40,
    seed        : int   = None,
) -> Image.Image:
    """
    Run img2img.

    strength: how much to deviate from input image
        0.4 = stays very close to animal shape (safer)
        0.65 = good balance shape vs Pokémon style
        0.8 = very creative, may lose animal shape
    """
    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)

    result = pipe(
        prompt          = prompt,
        negative_prompt = NEGATIVE_PROMPT,
        image           = animal_img,
        strength        = strength,
        guidance_scale  = guidance,
        num_inference_steps = steps,
        generator       = generator,
    ).images[0]

    return result


def generate_variations(
    pipe,
    animal_img: Image.Image,
    n_variations: int = 3,
    strengths: list = None,
    seed: int = 42,
) -> list[tuple[float, Image.Image]]:
    """Generate multiple variations with different strengths."""
    if strengths is None:
        strengths = [0.5, 0.65, 0.8]

    results = []
    prompt  = random.choice(POSITIVE_PROMPTS)
    for i, s in enumerate(strengths[:n_variations]):
        img = generate(pipe, animal_img, prompt, strength=s,
                       seed=seed + i)
        results.append((s, img))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def save_grid(results: list[dict], out_path: Path, strengths: list):
    """Grid: each row = one animal | cols = original + one per strength."""
    n_rows  = len(results)
    n_cols  = 1 + len(strengths)
    fig     = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.2))
    fig.patch.set_facecolor("#0a0a14")

    # Header row
    for col, title in enumerate(["Animal (no bg)"] + [f"strength={s}" for s in strengths]):
        ax = fig.add_subplot(n_rows + 1, n_cols, col + 1)
        ax.set_facecolor("#0a0a14")
        ax.text(0.5, 0.5, title, ha="center", va="center",
                color="#a78bfa", fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")

    for row, r in enumerate(results):
        base = (row + 1) * n_cols + 1

        # Original animal
        ax = fig.add_subplot(n_rows + 1, n_cols, base)
        ax.imshow(r["animal"])
        ax.set_ylabel(r["name"][:16], color="#888", fontsize=7,
                      rotation=0, labelpad=55, va="center")
        ax.axis("off")

        # Variations
        for col, (s, img) in enumerate(r["variations"]):
            ax = fig.add_subplot(n_rows + 1, n_cols, base + col + 1)
            ax.imshow(img)
            ax.axis("off")

    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="#0a0a14", edgecolor="none")
    plt.close()
    print(f"  ✓ Grid saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(directory: str) -> list[Path]:
    d = Path(directory)
    return sorted({p for ext in IMG_EXTS for p in d.rglob(f"*{ext}")})


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    use_hf  = args.lora_path is None
    pipe    = load_pipeline(
        lora_path      = args.lora_path,
        use_hf_pokemon = use_hf,
    )

    strengths = args.strengths

    # ── Single image mode ───────────────────────────────────────────────────
    if args.content:
        path       = Path(args.content)
        animal_img = remove_bg(path, args.size) if not args.no_rembg \
                     else Image.open(path).convert("RGB").resize((args.size, args.size))

        print(f"\n  Generating {len(strengths)} variations for {path.name}...")
        variations = generate_variations(pipe, animal_img, strengths=strengths,
                                         seed=args.seed)

        # Save individual
        for s, img in variations:
            img.save(out_dir / f"{path.stem}_s{str(s).replace('.','')}.png")

        # Side by side
        n     = len(variations)
        fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3.5, 3.5))
        fig.patch.set_facecolor("#0a0a14")
        axes[0].imshow(animal_img)
        axes[0].set_title("Animal", color="white", fontsize=9)
        axes[0].axis("off")
        for ax, (s, img) in zip(axes[1:], variations):
            ax.imshow(img)
            ax.set_title(f"strength={s}", color="#a78bfa", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{path.stem}_comparison.png", dpi=130,
                    facecolor="#0a0a14", bbox_inches="tight")
        plt.show()
        print(f"  ✓ Saved to {out_dir}")
        return

    # ── Multi-test mode ─────────────────────────────────────────────────────
    all_animals = collect(args.animals)
    assert all_animals, f"No images found in {args.animals}"

    random.seed(args.seed)
    animals = random.sample(all_animals, min(args.n_tests, len(all_animals)))

    print(f"\n  {len(animals)} animals | strengths: {strengths}")
    print(f"  Steps: {args.steps} | Guidance: {args.guidance}\n")

    results = []
    for i, path in enumerate(animals):
        print(f"\n[{i+1}/{len(animals)}] {path.name}")
        t0 = time.time()

        animal_img = remove_bg(path, args.size) if not args.no_rembg \
                     else Image.open(path).convert("RGB").resize((args.size, args.size))

        variations = generate_variations(
            pipe, animal_img,
            strengths = strengths,
            seed      = args.seed + i,
        )

        # Save individual outputs
        for s, img in variations:
            stem = f"{path.stem}_s{str(s).replace('.','')}"
            img.save(out_dir / f"{stem}.png")

        results.append({
            "name"      : path.stem,
            "animal"    : animal_img,
            "variations": variations,
        })
        print(f"  ✓ {time.time() - t0:.1f}s")

    save_grid(results, out_dir / "grid_sd_img2img.png", strengths)
    print(f"\n  ✓ All done — {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pokémon SD img2img")

    p.add_argument("--content",    type=str,   default=None,
                   help="Single animal image path")
    p.add_argument("--animals",    type=str,   default=None,
                   help="Folder of animal images")
    p.add_argument("--n_tests",    type=int,   default=6)
    p.add_argument("--output",     type=str,   default="./output_sd_img2img")
    p.add_argument("--lora_path",  type=str,   default=None,
                   help="Path to a .safetensors LoRA file (optional)")
    p.add_argument("--strengths",  type=float, nargs="+", default=[0.5, 0.65, 0.8],
                   help="img2img strength values to test (0=keep animal, 1=ignore animal)")
    p.add_argument("--steps",      type=int,   default=40,
                   help="Diffusion steps (30=fast, 40=balanced, 60=quality)")
    p.add_argument("--guidance",   type=float, default=9.0,
                   help="Classifier-free guidance scale (7-12 recommended)")
    p.add_argument("--size",       type=int,   default=512)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no_rembg",   action="store_true")

    args = p.parse_args()

    if not args.content and not args.animals:
        p.error("Provide --content or --animals")

    run(args)