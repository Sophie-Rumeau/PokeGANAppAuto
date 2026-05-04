''' Stable Diffusion LoRA '''

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

from torch.cuda.amp import autocast, GradScaler


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# DATASET (WITH AUGMENTATION)
# ─────────────────────────────────────────────
class PokemonDataset(Dataset):
    def __init__(self, root, size=512):

        self.paths = list(Path(root).rglob("*.png"))

        # 🔥 AUGMENTATION PIPELINE
        self.transform = transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),

            # mild color jitter (important for sprites but subtle)
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02
            ),

            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.transform(img)

        caption = "a pokemon sprite creature, pixel art"
        return img, caption


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(args):

    print("Loading Stable Diffusion...")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        safety_checker=None,
    ).to(DEVICE)

    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )

    # 🔥 IMPORTANT FIX: stable VAE dtype
    vae.to(dtype=torch.float32)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    unet.add_adapter(lora_config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    dataset = PokemonDataset(args.data, args.size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    scaler = GradScaler()

    print(f"Dataset size: {len(dataset)}")

    for epoch in range(args.epochs):
        for i, (images, captions) in enumerate(loader):

            images = images.to(DEVICE, dtype=torch.float32)

            # ───── VAE ENCODING (SAFE FP32) ─────
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215

            latents = latents.to(torch.float16 if DEVICE.type == "cuda" else torch.float32)

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE
            ).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # ───── TEXT ENCODING ─────
            inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = inputs.input_ids.to(DEVICE)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # ───── UNET FORWARD ─────
            with autocast():
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample

                loss = F.mse_loss(noise_pred, noise)

            # NaN guard
            if torch.isnan(loss):
                print("NaN detected — skipping batch")
                continue

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if i % 20 == 0:
                print(f"[Epoch {epoch}] [Step {i}] Loss: {loss.item():.4f}")

        Path(args.output).mkdir(parents=True, exist_ok=True)
        unet.save_attn_procs(args.output)

        print(f"✓ Saved epoch {epoch}")

    print("✓ Training complete")


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def generate(args):

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        safety_checker=None,
    ).to(DEVICE)

    pipe.load_lora_weights(args.model)

    image = pipe(
        args.prompt,
        num_inference_steps=40,
        guidance_scale=8.0,
    ).images[0]

    Path(args.output).mkdir(exist_ok=True)
    image.save(Path(args.output) / "generated.png")

    print("✓ Generated")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "generate"], required=True)
    parser.add_argument("--data", type=str, default="./images/images")
    parser.add_argument("--output", type=str, default="./sd_lora_out")
    parser.add_argument("--model", type=str, default="./sd_lora_out")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)

    parser.add_argument("--prompt", type=str,
                        default="a pokemon sprite creature, pixel art")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        generate(args)