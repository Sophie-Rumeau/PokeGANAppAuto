'''POKEMON NST — High Quality Style Transfer'''

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageFilter

# ─── Device ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── Config ───────────────────────────────────────────────────────────────────

CFG = {
    # Resolution — 512 gives good results, 256 is faster
    "size"           : 512,
    # Optimization steps — 600 is solid quality
    "steps"          : 600,
    # Style weight — very high to really inject Pokémon colors/texture
    "style_weight"   : 5e6,
    # Content weight — lower = more stylized, less animal-looking
    "content_weight" : 1.0,
    # TV weight — smoothing, avoids noise
    "tv_weight"      : 0.5,
    # Content layer — conv3_2 preserves shape without locking in animal colors
    "content_layer"  : "conv3_2",
    # Style layers with per-layer weights
    # conv1_1 heaviest = raw color/pixel-level style injection
    "style_layers"   : {
        "conv1_1": 1.0,   # color palette, pixel-level texture (eyes, fur edges)
        "conv2_1": 0.8,   # fine outlines, cartoon contours
        "conv3_1": 0.5,   # mid-level patterns
        "conv4_1": 0.3,   # shape grammar
        "conv5_1": 0.1,   # high-level semantics (barely used)
    },
    # Init: mix of content + noise — escapes the animal color basin
    # 0.0 = pure noise, 1.0 = pure content
    "content_init_ratio": 0.1,
    "log_every"      : 50,
}

# ─── VGG19 layer index map ────────────────────────────────────────────────────

VGG_LAYERS = {
    "0":"conv1_1","2":"conv1_2",
    "5":"conv2_1","7":"conv2_2",
    "10":"conv3_1","12":"conv3_2","14":"conv3_3","16":"conv3_4",
    "19":"conv4_1","21":"conv4_2","23":"conv4_3","25":"conv4_4",
    "28":"conv5_1","30":"conv5_2","32":"conv5_3","34":"conv5_4",
}
NAME_TO_IDX = {v: k for k, v in VGG_LAYERS.items()}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# 1. IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def remove_bg(path: Path, size: int) -> Image.Image:
    """
    Remove background with rembg (U2-Net).
    Returns RGBA image with transparent background.
    Falls back gracefully if rembg isn't installed.
    """
    try:
        from rembg import remove
        print(f"    removing background...")
        with open(path, "rb") as f:
            data = remove(f.read())
        import io
        img = Image.open(io.BytesIO(data)).convert("RGBA")
    except ImportError:
        print("    [warn] rembg not installed — skipping bg removal (pip install rembg)")
        img = Image.open(path).convert("RGBA")
    except Exception as e:
        print(f"    [warn] rembg failed ({e}) — skipping")
        img = Image.open(path).convert("RGBA")
    return letterbox_rgba(img, size)


def letterbox_rgba(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping aspect ratio, pad with transparency."""
    img.thumbnail((size, size), Image.LANCZOS)
    result = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    result.paste(img, offset, img if img.mode == "RGBA" else None)
    return result


def load_pokemon(path: Path, size: int) -> Image.Image:
    """Load a Pokémon PNG, compose on white, letterbox."""
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return letterbox_rgba(bg.convert("RGBA"), size)


def rgba_to_rgb_white(img: Image.Image) -> Image.Image:
    """Compose RGBA on white background."""
    bg = Image.new("RGB", img.size, (255, 255, 255))
    if img.mode == "RGBA":
        bg.paste(img, mask=img.split()[3])
    else:
        bg.paste(img.convert("RGB"))
    return bg


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → normalized tensor (1, 3, H, W)."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def to_pil(t: torch.Tensor) -> Image.Image:
    """Normalized tensor → PIL RGB."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = (t.cpu().squeeze(0) * std + mean).clamp(0, 1)
    return transforms.ToPILImage()(img)


# ══════════════════════════════════════════════════════════════════════════════
# 2. VGG19 FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class VGG19(nn.Module):
    """VGG19 feature extractor — frozen, cut at the deepest needed layer."""

    def __init__(self, content_layer: str, style_layers: dict):
        super().__init__()
        self.content_layer = content_layer
        self.style_layers  = list(style_layers.keys())

        vgg     = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        max_idx = max(int(NAME_TO_IDX[l]) for l in self.style_layers + [content_layer])
        self.slices = nn.Sequential(*list(vgg.children())[:max_idx + 1])

        for p in self.slices.parameters():
            p.requires_grad_(False)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> dict:
        feats = {}
        for idx, layer in enumerate(self.slices):
            x    = layer(x)
            name = VGG_LAYERS.get(str(idx))
            if name in self.style_layers or name == self.content_layer:
                feats[name] = x
        return feats


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def gram(feat: torch.Tensor) -> torch.Tensor:
    """Gram matrix — captures style correlations between channels."""
    _, c, h, w = feat.shape
    f = feat.view(c, h * w)
    return torch.mm(f, f.t()) / (c * h * w)


def content_loss(gen_f: torch.Tensor, target_f: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(gen_f, target_f)


def style_loss(
    gen_feats   : dict,
    style_grams : dict,
    layer_weights: dict,
) -> torch.Tensor:
    """Weighted sum of gram matrix MSE across style layers."""
    loss    = torch.zeros(1, device=DEVICE)
    total_w = sum(layer_weights.values())
    for layer, w in layer_weights.items():
        g_gen   = gram(gen_feats[layer])
        g_style = style_grams[layer]
        loss    = loss + (w / total_w) * nn.functional.mse_loss(g_gen, g_style)
    return loss


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Total variation — penalizes noisy pixel transitions."""
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN NST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_nst(
    content_pil : Image.Image,   # RGB, already bg-removed & white-bg
    style_pil   : Image.Image,   # RGB Pokémon
    cfg         : dict = CFG,
) -> tuple[Image.Image, list]:
    """
    Run Neural Style Transfer.
    Returns (best_generated_pil, loss_history).
    """
    content_t = to_tensor(content_pil)
    style_t   = to_tensor(style_pil)

    # Build VGG once
    vgg = VGG19(cfg["content_layer"], cfg["style_layers"])

    # Pre-compute fixed targets
    with torch.no_grad():
        c_feats = vgg(content_t)
        s_feats = vgg(style_t)

    content_target = c_feats[cfg["content_layer"]].detach()
    style_grams    = {l: gram(s_feats[l]).detach() for l in cfg["style_layers"]}

    # ── Initialization ──────────────────────────────────────────────────────
    # Low content ratio = break free from animal color palette
    noise   = torch.randn_like(content_t) * 0.15
    r       = cfg["content_init_ratio"]
    gen_img = (r * content_t + (1 - r) * noise).detach().requires_grad_(True)

    optimizer = optim.LBFGS([gen_img], max_iter=20, line_search_fn="strong_wolfe")

    step      = [0]
    best_img  = [gen_img.detach().clone()]
    best_loss = [float("inf")]
    history   = []

    def closure():
        optimizer.zero_grad()
        with torch.no_grad():
            gen_img.clamp_(-3.0, 3.0)

        feats = vgg(gen_img)

        c_loss = content_loss(feats[cfg["content_layer"]], content_target)
        s_loss = style_loss(feats, style_grams, cfg["style_layers"])
        t_loss = tv_loss(gen_img)

        total = (
            cfg["content_weight"] * c_loss
            + cfg["style_weight"]  * s_loss
            + cfg["tv_weight"]     * t_loss
        )
        total.backward()

        step[0] += 1
        val = total.item()

        if step[0] % cfg["log_every"] == 0 or step[0] == 1:
            print(
                f"    step {step[0]:>4}/{cfg['steps']} | "
                f"total {val:.1f} | "
                f"content {c_loss.item():.4f} | "
                f"style {s_loss.item():.6f}"
            )
            history.append(val)

        if val < best_loss[0]:
            best_loss[0] = val
            best_img[0]  = gen_img.detach().clone()

        return total

    while step[0] < cfg["steps"]:
        optimizer.step(closure)

    print(f"    ✓ done — best loss: {best_loss[0]:.1f}")
    return to_pil(best_img[0]), history


# ══════════════════════════════════════════════════════════════════════════════
# 5. POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def postprocess(img: Image.Image, quantize: bool = True, n_colors: int = 24) -> Image.Image:
    """
    Light sharpening + palette quantization for cartoon look.
    n_colors=24 keeps more detail than 16 while still looking stylized.
    """
    # Subtle unsharp to crisp up edges
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
    # Palette reduction
    if quantize:
        img = img.quantize(colors=n_colors, method=Image.Quantize.FASTOCTREE).convert("RGB")
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRID VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def save_grid(results: list[dict], output_path: Path):
    """
    Save a dark-themed grid:
    Each row = one test | cols: animal | pokémon style | raw NST | post-processed
    """
    n    = len(results)
    cols = 4
    fig  = plt.figure(figsize=(cols * 3.5, n * 3.5))
    fig.patch.set_facecolor("#0f0f1a")

    headers = ["Animal (no bg)", "Style Pokémon", "NST Output", "Post-processed"]
    for col, h in enumerate(headers):
        ax = fig.add_subplot(n + 1, cols, col + 1)
        ax.set_facecolor("#0f0f1a")
        ax.text(0.5, 0.5, h, ha="center", va="center",
                color="#a78bfa", fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")

    for row, r in enumerate(results):
        row_offset = (row + 1) * cols + 1
        imgs = [r["animal_rgb"], r["pokemon_rgb"], r["nst_raw"], r["nst_post"]]
        for col, img in enumerate(imgs):
            ax = fig.add_subplot(n + 1, cols, row_offset + col)
            ax.imshow(img)
            ax.axis("off")
            ax.set_facecolor("#0f0f1a")
            if col == 0:
                ax.set_ylabel(
                    f"{r['animal_name'][:14]}\n× {r['pokemon_name'][:14]}",
                    color="#888", fontsize=7, rotation=0,
                    labelpad=60, va="center"
                )
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a4a")

    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")
    plt.close()
    print(f"\n  ✓ Grid saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POKEMON_EXTS = {".png"}


def collect(directory: str, exts: set) -> list[Path]:
    d = Path(directory)
    return sorted({p for ext in exts for p in d.rglob(f"*{ext}")})


def run_pipeline(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Single pair mode ────────────────────────────────────────────────────
    if args.content and args.style:
        print(f"\nSingle pair mode")
        animal_path  = Path(args.content)
        pokemon_path = Path(args.style)

        size  = args.size
        print(f"  Loading animal  : {animal_path.name}")
        if args.no_rembg:
            animal_rgba = letterbox_rgba(Image.open(animal_path).convert("RGBA"), size)
        else:
            animal_rgba = remove_bg(animal_path, size)

        animal_rgb  = rgba_to_rgb_white(animal_rgba)
        pokemon_pil = load_pokemon(pokemon_path, size)
        pokemon_rgb = rgba_to_rgb_white(pokemon_pil)

        print(f"  Running NST...")
        cfg          = {**CFG, "size": size, "steps": args.steps}
        raw, history = run_nst(animal_rgb, pokemon_rgb, cfg)
        post         = postprocess(raw, quantize=not args.no_quantize)

        raw.save(out_dir / "result_raw.png")
        post.save(out_dir / "result_post.png")

        # Quick side-by-side
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor("#0f0f1a")
        for ax, img, title in zip(axes,
            [animal_rgb, pokemon_rgb, raw, post],
            ["Animal", "Style", "NST Raw", "Post-processed"]):
            ax.imshow(img)
            ax.set_title(title, color="white", fontsize=9)
            ax.axis("off")
            ax.set_facecolor("#0f0f1a")
        plt.tight_layout()
        plt.savefig(out_dir / "result_comparison.png", dpi=130,
                    facecolor="#0f0f1a", bbox_inches="tight")
        plt.show()
        print(f"  ✓ Saved to {out_dir}")
        return

    # ── Multi-test mode ─────────────────────────────────────────────────────
    all_animals = collect(args.animals, IMG_EXTS)
    all_pokemon = collect(args.pokemon, POKEMON_EXTS)
    assert all_animals, f"No animal images found in {args.animals}"
    assert all_pokemon, f"No Pokémon PNGs found in {args.pokemon}"

    random.seed(args.seed)
    animals = random.sample(all_animals, min(args.n_tests, len(all_animals)))
    pokemon = random.sample(all_pokemon, min(args.n_tests, len(all_pokemon)))
    pairs   = list(zip(animals, pokemon))

    print(f"\n  Running {len(pairs)} NST combinations")
    print(f"  Size: {args.size}px | Steps: {args.steps} | Style weight: {CFG['style_weight']:.0e}")
    print(f"  BG removal: {not args.no_rembg} | Quantize: {not args.no_quantize}\n")

    cfg     = {**CFG, "size": args.size, "steps": args.steps}
    results = []

    for i, (animal_path, pokemon_path) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {animal_path.name} × {pokemon_path.name}")
        t0 = time.time()

        if args.no_rembg:
            animal_rgba = letterbox_rgba(Image.open(animal_path).convert("RGBA"), args.size)
        else:
            animal_rgba = remove_bg(animal_path, args.size)

        animal_rgb  = rgba_to_rgb_white(animal_rgba)
        pokemon_pil = load_pokemon(pokemon_path, args.size)
        pokemon_rgb = rgba_to_rgb_white(pokemon_pil)

        raw, history = run_nst(animal_rgb, pokemon_rgb, cfg)
        post         = postprocess(raw, quantize=not args.no_quantize)

        # Save individual outputs
        stem = f"{animal_path.stem}__{pokemon_path.stem}"
        raw.save(out_dir / f"{stem}_raw.png")
        post.save(out_dir / f"{stem}_post.png")

        results.append({
            "animal_name" : animal_path.stem,
            "pokemon_name": pokemon_path.stem,
            "animal_rgb"  : animal_rgb,
            "pokemon_rgb" : pokemon_rgb,
            "nst_raw"     : raw,
            "nst_post"    : post,
            "history"     : history,
            "time_s"      : time.time() - t0,
        })
        print(f"    ✓ {time.time() - t0:.0f}s")

    # Save grid
    save_grid(results, out_dir / "grid_results.png")

    # Loss curves
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3))
    fig.patch.set_facecolor("#0f0f1a")
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        steps = [j * cfg["log_every"] for j in range(len(r["history"]))]
        ax.plot(steps, r["history"], color="#a78bfa", linewidth=2)
        ax.set_title(f"{r['animal_name'][:10]}×{r['pokemon_name'][:10]}",
                     color="white", fontsize=7)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=100,
                facecolor="#0f0f1a", bbox_inches="tight")
    plt.close()

    total = sum(r["time_s"] for r in results)
    print(f"\n  ✓ All done in {total:.0f}s ({total/len(results):.0f}s/img)")
    print(f"  ✓ Results in {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pokémon NST — High Quality")

    # Single pair
    p.add_argument("--content",     type=str, default=None, help="Single animal image")
    p.add_argument("--style",       type=str, default=None, help="Single Pokémon image")

    # Multi-test
    p.add_argument("--animals",     type=str, default=None, help="Folder of animal images")
    p.add_argument("--pokemon",     type=str, default=None, help="Folder of Pokémon PNGs")
    p.add_argument("--n_tests",     type=int, default=6,   help="Number of random pairs to test")
    p.add_argument("--seed",        type=int, default=42)

    # Shared
    p.add_argument("--output",      type=str, default="./output_nst")
    p.add_argument("--size",        type=int, default=512,
                   help="Image size in px (256=fast, 512=quality, 768=slow but crisp)")
    p.add_argument("--steps",       type=int, default=600,
                   help="Optimization steps (400=fast, 600=balanced, 800=best)")
    p.add_argument("--no_rembg",    action="store_true", help="Skip background removal")
    p.add_argument("--no_quantize", action="store_true", help="Skip palette quantization")

    args = p.parse_args()

    if not args.content and not args.animals:
        p.error("Provide either --content + --style, or --animals + --pokemon")

    run_pipeline(args)