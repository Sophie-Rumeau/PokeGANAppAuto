'''OPTION 2 — ControlNet + SD'''

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# ─── Device ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32
print(f"  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── ControlNet model IDs ─────────────────────────────────────────────────────

CONTROLNET_IDS = {
    "edges"  : "lllyasviel/sd-controlnet-canny",
    "depth"  : "lllyasviel/sd-controlnet-depth",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
}

# ─── SD base — pokemon fine-tuned ────────────────────────────────────────────
SD_MODEL = "lambdalabs/sd-pokemon-diffusers"

# ══════════════════════════════════════════════════════════════════════════════
# 1. BACKGROUND REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def remove_bg(path: Path, size: int = 512) -> Image.Image:
    """Remove background, return RGB on white."""
    try:
        from rembg import remove
        import io
        print(f"    removing background...")
        with open(path, "rb") as f:
            out = remove(f.read())
        img = Image.open(io.BytesIO(out)).convert("RGBA")
    except Exception as e:
        print(f"    [warn] rembg: {e} — keeping bg")
        img = Image.open(path).convert("RGBA")

    bg = Image.new("RGB", img.size, (255, 255, 255))
    if img.mode == "RGBA":
        bg.paste(img, mask=img.split()[3])
    else:
        bg = img.convert("RGB")
    return bg.resize((size, size), Image.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTROL SIGNAL EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def extract_canny(img: Image.Image,
                  low: int = 50, high: int = 200) -> Image.Image:
    """
    Canny edge detection.
    Captures silhouette + internal edges. Good for preserving outline shape.
    """
    arr   = np.array(img.convert("L"))
    edges = cv2.Canny(arr, low, high)
    # ControlNet expects 3-channel image
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def extract_depth(img: Image.Image) -> Image.Image:
    """
    MiDaS monocular depth estimation.
    Captures 3D structure — good for animals with complex poses.
    Requires: pip install transformers
    """
    from transformers import pipeline as hf_pipeline

    print("    extracting depth map (MiDaS)...")
    estimator = hf_pipeline(
        "depth-estimation",
        model  = "Intel/dpt-large",
        device = 0 if DEVICE.type == "cuda" else -1,
    )
    result    = estimator(img)
    depth_pil = result["depth"]

    # Normalize to 0-255 and convert to RGB
    depth_arr = np.array(depth_pil).astype(np.float32)
    depth_arr = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-6)
    depth_u8  = (depth_arr * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb).resize(img.size, Image.LANCZOS)


def extract_lineart(img: Image.Image) -> Image.Image:
    """
    HED soft edge detection via controlnet_aux.
    Produces clean cartoon-like lineart — recommended for Pokémon style.
    Requires: pip install controlnet-aux
    """
    try:
        from controlnet_aux import LineartDetector
        print("    extracting lineart (HED)...")
        detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        return detector(img)
    except ImportError:
        print("    [warn] controlnet-aux not installed — falling back to Canny")
        return extract_canny(img)
    except Exception as e:
        print(f"    [warn] lineart failed ({e}) — falling back to Canny")
        return extract_canny(img)


def extract_control(img: Image.Image, mode: str) -> Image.Image:
    """Dispatch to the right extractor."""
    if mode == "edges":
        return extract_canny(img)
    elif mode == "depth":
        return extract_depth(img)
    elif mode == "lineart":
        return extract_lineart(img)
    else:
        raise ValueError(f"Unknown control mode: {mode}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE SETUP
# ══════════════════════════════════════════════════════════════════════════════

_pipe_cache = {}  # cache loaded pipelines by mode


def load_pipeline(control_mode: str):
    """Load ControlNet + SD pipeline for the given mode."""
    if control_mode in _pipe_cache:
        return _pipe_cache[control_mode]

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    from diffusers import UniPCMultistepScheduler

    print(f"  Loading ControlNet ({control_mode})...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_IDS[control_mode],
        torch_dtype=DTYPE,
    )

    print(f"  Loading SD base ({SD_MODEL})...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL,
        controlnet    = controlnet,
        torch_dtype   = DTYPE,
        safety_checker= None,
    )

    # UniPC scheduler — fast, high quality
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)

    if DEVICE.type == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers enabled")
        except Exception:
            pass

    _pipe_cache[control_mode] = pipe
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# 4. GENERATION
# ══════════════════════════════════════════════════════════════════════════════

POSITIVE_PROMPTS = [
    "a new Pokemon creature, Ken Sugimori official artwork style, "
    "white background, clean cartoon illustration, vibrant colors, "
    "sharp black outlines, game freak style, high quality",

    "original Pokemon design, anime illustration, colorful cartoon creature, "
    "white background, clean linework, official Pokemon art style",

    "cute Pokemon character, watercolor illustration, Ken Sugimori style, "
    "vibrant palette, white background, sharp outlines, professional artwork",
]

NEGATIVE_PROMPT = (
    "photo, realistic, photograph, 3d render, blurry, noisy, dark, "
    "human, person, text, watermark, ugly, deformed, low quality, "
    "grainy, photorealistic, real animal, fur texture, natural background"
)


def generate(
    pipe,
    control_img      : Image.Image,
    prompt           : str    = None,
    control_strength : float  = 0.8,
    guidance         : float  = 9.0,
    steps            : int    = 40,
    seed             : int    = None,
) -> Image.Image:
    """
    Generate a Pokémon conditioned on the control signal.

    control_strength: how strictly to follow the animal's structure
        0.5 = loose — more creative freedom, less shape fidelity
        0.8 = balanced — recommended
        1.0 = strict — very close to input structure
    """
    if prompt is None:
        prompt = random.choice(POSITIVE_PROMPTS)

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)

    result = pipe(
        prompt                  = prompt,
        negative_prompt         = NEGATIVE_PROMPT,
        image                   = control_img,
        controlnet_conditioning_scale = control_strength,
        guidance_scale          = guidance,
        num_inference_steps     = steps,
        generator               = generator,
    ).images[0]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def save_grid(results: list[dict], out_path: Path):
    """
    Grid: animal | control signal | generated Pokémon (x3 seeds)
    """
    n_rows = len(results)
    n_cols = 5  # animal | control | gen1 | gen2 | gen3
    fig    = plt.figure(figsize=(n_cols * 3.0, n_rows * 3.0))
    fig.patch.set_facecolor("#0a0a14")

    headers = ["Animal (no bg)", "Control Signal",
               "Generated #1", "Generated #2", "Generated #3"]
    for col, h in enumerate(headers):
        ax = fig.add_subplot(n_rows + 1, n_cols, col + 1)
        ax.set_facecolor("#0a0a14")
        ax.text(0.5, 0.5, h, ha="center", va="center",
                color="#a78bfa", fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")

    for row, r in enumerate(results):
        base = (row + 1) * n_cols + 1
        imgs = [r["animal"], r["control"]] + r["generated"]

        for col, img in enumerate(imgs[:n_cols]):
            ax = fig.add_subplot(n_rows + 1, n_cols, base + col)
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(r["name"][:16], color="#888", fontsize=7,
                              rotation=0, labelpad=55, va="center")

    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="#0a0a14", edgecolor="none")
    plt.close()
    print(f"  ✓ Grid saved: {out_path}")


def save_mode_comparison(animal: Image.Image, results_by_mode: dict,
                         name: str, out_path: Path):
    """
    Compare all three control modes for a single animal.
    Rows: edges | depth | lineart
    Cols: animal | control | gen1 | gen2 | gen3
    """
    modes  = list(results_by_mode.keys())
    n_rows = len(modes)
    n_cols = 5
    fig    = plt.figure(figsize=(n_cols * 3.0, n_rows * 3.5))
    fig.patch.set_facecolor("#0a0a14")

    headers = ["Animal", "Control", "Gen #1", "Gen #2", "Gen #3"]
    for col, h in enumerate(headers):
        ax = fig.add_subplot(n_rows + 1, n_cols, col + 1)
        ax.text(0.5, 0.5, h, ha="center", va="center",
                color="#a78bfa", fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        ax.set_facecolor("#0a0a14")
        ax.axis("off")

    for row, mode in enumerate(modes):
        r    = results_by_mode[mode]
        base = (row + 1) * n_cols + 1
        imgs = [animal, r["control"]] + r["generated"]
        for col, img in enumerate(imgs[:n_cols]):
            ax = fig.add_subplot(n_rows + 1, n_cols, base + col)
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(mode, color="#a78bfa", fontsize=9,
                              rotation=0, labelpad=45, va="center", fontweight="bold")

    plt.suptitle(f"ControlNet mode comparison — {name}",
                 color="white", fontsize=11, fontweight="bold")
    plt.subplots_adjust(wspace=0.03, hspace=0.06)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="#0a0a14", edgecolor="none")
    plt.close()
    print(f"  ✓ Mode comparison saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(directory: str) -> list[Path]:
    d = Path(directory)
    return sorted({p for ext in IMG_EXTS for p in d.rglob(f"*{ext}")})


def process_one(pipe, animal_img, control_mode, args, n_generations=3, base_seed=42):
    """Extract control signal + generate N Pokémon from one animal."""
    control_img = extract_control(animal_img, control_mode)
    generated   = []
    prompt      = random.choice(POSITIVE_PROMPTS)
    for i in range(n_generations):
        print(f"    generating #{i+1}...")
        img = generate(
            pipe,
            control_img,
            prompt           = prompt,
            control_strength = args.control_strength,
            guidance         = args.guidance,
            steps            = args.steps,
            seed             = base_seed + i * 7,
        )
        generated.append(img)
    return control_img, generated


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mode comparison on single image ────────────────────────────────────
    if args.content and args.compare_modes:
        path       = Path(args.content)
        animal_img = remove_bg(path, args.size) if not args.no_rembg \
                     else Image.open(path).convert("RGB").resize((args.size, args.size))

        results_by_mode = {}
        for mode in ["edges", "depth", "lineart"]:
            print(f"\n  Mode: {mode}")
            pipe = load_pipeline(mode)
            ctrl, gens = process_one(pipe, animal_img, mode, args,
                                     n_generations=3, base_seed=args.seed)
            results_by_mode[mode] = {"control": ctrl, "generated": gens}
            for i, g in enumerate(gens):
                g.save(out_dir / f"{path.stem}_{mode}_gen{i+1}.png")

        save_mode_comparison(animal_img, results_by_mode, path.stem,
                             out_dir / f"{path.stem}_mode_comparison.png")
        print(f"\n  ✓ Saved to {out_dir}/")
        return

    # ── Single image, single mode ───────────────────────────────────────────
    if args.content:
        path       = Path(args.content)
        animal_img = remove_bg(path, args.size) if not args.no_rembg \
                     else Image.open(path).convert("RGB").resize((args.size, args.size))

        pipe       = load_pipeline(args.control_mode)
        ctrl, gens = process_one(pipe, animal_img, args.control_mode, args,
                                 n_generations=3, base_seed=args.seed)

        for i, g in enumerate(gens):
            g.save(out_dir / f"{path.stem}_gen{i+1}.png")
        ctrl.save(out_dir / f"{path.stem}_control.png")

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.patch.set_facecolor("#0a0a14")
        for ax, img, title in zip(
            axes,
            [animal_img, ctrl] + gens,
            ["Animal", f"Control ({args.control_mode})", "Gen #1", "Gen #2", "Gen #3"]
        ):
            ax.imshow(img)
            ax.set_title(title, color="white" if "Gen" not in title else "#a78bfa",
                         fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{path.stem}_result.png", dpi=130,
                    facecolor="#0a0a14", bbox_inches="tight")
        plt.show()
        return

    # ── Multi-test mode ─────────────────────────────────────────────────────
    all_animals = collect(args.animals)
    assert all_animals, f"No images found in {args.animals}"

    random.seed(args.seed)
    animals = random.sample(all_animals, min(args.n_tests, len(all_animals)))
    pipe    = load_pipeline(args.control_mode)

    print(f"\n  {len(animals)} animals | mode: {args.control_mode}")
    print(f"  Control strength: {args.control_strength} | Steps: {args.steps}\n")

    results = []
    for i, path in enumerate(animals):
        print(f"\n[{i+1}/{len(animals)}] {path.name}")
        t0 = time.time()

        animal_img = remove_bg(path, args.size) if not args.no_rembg \
                     else Image.open(path).convert("RGB").resize((args.size, args.size))

        ctrl, gens = process_one(pipe, animal_img, args.control_mode, args,
                                 n_generations=3, base_seed=args.seed + i * 13)

        # Save individual outputs
        ctrl.save(out_dir / f"{path.stem}_control.png")
        for j, g in enumerate(gens):
            g.save(out_dir / f"{path.stem}_gen{j+1}.png")

        results.append({
            "name"     : path.stem,
            "animal"   : animal_img,
            "control"  : ctrl,
            "generated": gens,
        })
        print(f"  ✓ {time.time() - t0:.1f}s")

    save_grid(results, out_dir / f"grid_controlnet_{args.control_mode}.png")
    print(f"\n  ✓ All done — {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pokémon ControlNet")

    p.add_argument("--content",          type=str,   default=None)
    p.add_argument("--animals",          type=str,   default=None)
    p.add_argument("--n_tests",          type=int,   default=6)
    p.add_argument("--output",           type=str,   default="./output_controlnet")
    p.add_argument("--control_mode",     type=str,   default="lineart",
                   choices=["edges", "depth", "lineart"],
                   help="lineart=recommended, edges=fast, depth=best 3D shape")
    p.add_argument("--control_strength", type=float, default=0.8,
                   help="0.5=loose/creative, 0.8=balanced, 1.0=strict shape")
    p.add_argument("--steps",            type=int,   default=40)
    p.add_argument("--guidance",         type=float, default=9.0)
    p.add_argument("--size",             type=int,   default=512)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no_rembg",         action="store_true")
    p.add_argument("--compare_modes",    action="store_true",
                   help="With --content: compare all 3 control modes side by side")

    args = p.parse_args()

    if not args.content and not args.animals:
        p.error("Provide --content or --animals")

    run(args)