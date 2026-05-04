from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

TYPE_LIST = ["Fire", "Water", "Grass", "Electric"]
TYPE_TO_IDX = {t.lower(): i for i, t in enumerate(TYPE_LIST)}

def num_groups(channels: int, max_groups: int = 8) -> int:
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1

def denorm(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1, 1).add(1).div(2)

def cond_vector_from_types(type_names: List[str]) -> torch.Tensor:
    vec = torch.zeros(len(TYPE_LIST), dtype=torch.float32)
    for name in type_names:
        idx = TYPE_TO_IDX.get(name.lower())
        if idx is not None:
            vec[idx] = 1.0
    return vec

def make_sample_conditions(type_names: List[str], device: torch.device) -> torch.Tensor:
    conds = [cond_vector_from_types([t]) for t in type_names]
    return torch.stack(conds, dim=0).to(device)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: Optional[int] = None):
        super().__init__()
        self.time_dim = time_dim
        self.norm1 = nn.GroupNorm(num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.time_proj = nn.Linear(time_dim, out_channels) if time_dim is not None else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x).reshape(b, c, h * w).permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = self.proj(attn_out)
        attn_out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)
        return residual + attn_out

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, latent_channels: int = 4):
        super().__init__()
        ch = base_channels

        self.enc_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        self.enc_block1 = ResidualBlock(ch, ch)
        self.down1 = Downsample(ch)
        self.enc_block2 = ResidualBlock(ch, ch * 2)
        self.down2 = Downsample(ch * 2)
        self.enc_block3 = ResidualBlock(ch * 2, ch * 4)
        self.enc_attn = AttentionBlock(ch * 4)

        self.to_mu = nn.Conv2d(ch * 4, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(ch * 4, latent_channels, kernel_size=1)

        self.dec_in = nn.Conv2d(latent_channels, ch * 4, kernel_size=3, padding=1)
        self.dec_block1 = ResidualBlock(ch * 4, ch * 4)
        self.dec_attn = AttentionBlock(ch * 4)
        self.up1 = Upsample(ch * 4)
        self.dec_block2 = ResidualBlock(ch * 4, ch * 2)
        self.up2 = Upsample(ch * 2)
        self.dec_block3 = ResidualBlock(ch * 2, ch)
        self.out_norm = nn.GroupNorm(num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc_in(x)
        x = self.enc_block1(x)
        x = self.down1(x)
        x = self.enc_block2(x)
        x = self.down2(x)
        x = self.enc_block3(x)
        x = self.enc_attn(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x).clamp(-10, 5)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * temperature
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_in(z)
        x = self.dec_block1(x)
        x = self.dec_attn(x)
        x = self.up1(x)
        x = self.dec_block2(x)
        x = self.up2(x)
        x = self.dec_block3(x)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class LatentUNetConditional(nn.Module):
    def __init__(self, latent_channels: int = 4, base_channels: int = 128, time_dim: int = 512, cond_dim: int = 18):
        super().__init__()
        self.cond_dim = cond_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        ch = base_channels
        self.in_conv = nn.Conv2d(latent_channels, ch, kernel_size=3, padding=1)

        self.down1a = ResidualBlock(ch, ch, time_dim=time_dim)
        self.down1b = ResidualBlock(ch, ch, time_dim=time_dim)
        self.attn1 = AttentionBlock(ch)
        self.downsample1 = Downsample(ch)

        self.down2a = ResidualBlock(ch, ch * 2, time_dim=time_dim)
        self.down2b = ResidualBlock(ch * 2, ch * 2, time_dim=time_dim)
        self.attn2 = AttentionBlock(ch * 2)
        self.downsample2 = Downsample(ch * 2)

        self.mid1 = ResidualBlock(ch * 2, ch * 4, time_dim=time_dim)
        self.mid_attn = AttentionBlock(ch * 4)
        self.mid2 = ResidualBlock(ch * 4, ch * 2, time_dim=time_dim)

        self.upsample2 = Upsample(ch * 2)
        self.up2a = ResidualBlock(ch * 4, ch * 2, time_dim=time_dim)
        self.up2b = ResidualBlock(ch * 2, ch, time_dim=time_dim)
        self.upattn2 = AttentionBlock(ch)

        self.upsample1 = Upsample(ch)
        self.up1a = ResidualBlock(ch * 2, ch, time_dim=time_dim)
        self.up1b = ResidualBlock(ch, ch, time_dim=time_dim)
        self.upattn1 = AttentionBlock(ch)

        self.out_norm = nn.GroupNorm(num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is None:
            cond = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=x.dtype)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)
        emb = t_emb + c_emb

        x0 = self.in_conv(x)

        x1 = self.down1a(x0, emb)
        x1 = self.down1b(x1, emb)
        x1 = self.attn1(x1)
        x = self.downsample1(x1)

        x2 = self.down2a(x, emb)
        x2 = self.down2b(x2, emb)
        x2 = self.attn2(x2)
        x = self.downsample2(x2)

        x = self.mid1(x, emb)
        x = self.mid_attn(x)
        x = self.mid2(x, emb)

        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2a(x, emb)
        x = self.up2b(x, emb)
        x = self.upattn2(x)

        x = self.upsample1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1a(x, emb)
        x = self.up1b(x, emb)
        x = self.upattn1(x)

        return self.out_conv(F.silu(self.out_norm(x)))

class LatentDiffusion:
    def __init__(
        self,
        timesteps: int = 300,
        objective: str = "v",
        min_snr_gamma: float = 5.0,
        device: str = "cuda",
    ):
        if objective not in {"v", "eps"}:
            raise ValueError("objective must be 'v' or 'eps'")
        self.timesteps = timesteps
        self.objective = objective
        self.min_snr_gamma = min_snr_gamma
        self.device = torch.device(device)

        betas = self.cosine_beta_schedule(timesteps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-4, 0.999)

    def extract(self, arr: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = arr.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sigma = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return alpha * x0 + sigma * noise

    def predict_v(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sigma = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return alpha * noise - sigma * x0

    def predict_x0_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self.extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sigma = self.extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return alpha * xt - sigma * v

    def predict_eps_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self.extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sigma = self.extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return sigma * xt + alpha * v

    def model_predictions(self, model_out: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.objective == "v":
            x0 = self.predict_x0_from_v(xt, t, model_out)
            eps = self.predict_eps_from_v(xt, t, model_out)
        else:
            eps = model_out
            alpha = self.extract(self.sqrt_alphas_cumprod, t, xt.shape)
            sigma = self.extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
            x0 = (xt - sigma * eps) / alpha.clamp(min=1e-8)
        return x0, eps

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t]).clamp(min=1e-8)
        clipped = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma))
        if self.objective == "v":
            return clipped / (snr + 1.0)
        return clipped / snr.clamp(min=1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        steps: int = 50,
        eta: float = 0.0,
        temperature: float = 1.0,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        b = shape[0]
        x = torch.randn(shape, device=self.device) * temperature
        times = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        next_times = torch.cat([times[1:], torch.tensor([-1], device=self.device, dtype=torch.long)], dim=0)

        for t_curr, t_next in zip(times, next_times):
            t = torch.full((b,), int(t_curr.item()), device=self.device, dtype=torch.long)

            if cond is None or guidance_scale == 0.0:
                model_out = model(x, t, cond)
            else:
                uncond = torch.zeros_like(cond)
                model_out_cond = model(x, t, cond)
                model_out_uncond = model(x, t, uncond)
                model_out = model_out_uncond + guidance_scale * (model_out_cond - model_out_uncond)

            x0, eps = self.model_predictions(model_out, x, t)
            x0 = x0.clamp(-4.0, 4.0)

            if t_next < 0:
                x = x0
                continue

            at = self.alphas_cumprod[t_curr]
            an = self.alphas_cumprod[t_next]
            sigma = eta * torch.sqrt(((1.0 - an) / (1.0 - at)).clamp(min=1e-8) * (1.0 - at / an).clamp(min=0.0))
            c = torch.sqrt((1.0 - an - sigma ** 2).clamp(min=0.0))
            noise = torch.randn_like(x) * temperature
            x = torch.sqrt(an) * x0 + c * eps + sigma * noise
        return x

