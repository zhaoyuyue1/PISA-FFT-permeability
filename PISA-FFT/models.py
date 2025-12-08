from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Stochastic Depth / DropPath ----------------------


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.

    This is a standard implementation that randomly drops entire residual
    paths during training to regularize deep networks.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob) if drop_prob is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------- Multi-Head Spectral Attention ----------------------


class MultiHeadSpectralAttention(nn.Module):
    """
    Multi-head spectral attention operating in the token sequence domain.

    Input: x in shape [B, N, D], where
        B = batch size
        N = sequence length (1 + number of patches)
        D = embedding dimension

    We first apply LayerNorm, then reshape to [B, H, Dh, N] where H is the
    number of heads and Dh is the per-head channel dimension. A 1D FFT is
    performed along the sequence dimension, followed by a learnable
    frequency-domain filter and complex-valued non-linearity. Finally, we
    transform back via inverse FFT.
    """

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive

        # Number of frequency bins after rfft along the sequence axis
        self.freq_bins = self.seq_len // 2 + 1

        # Base frequency filter and bias (shared across samples)
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins))
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins), -0.1))

        # Optional adaptive modulation conditioned on the global token context
        if adaptive:
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2),
            )
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def complex_activation(x: torch.Tensor) -> torch.Tensor:
        """Apply a simple complex-valued activation."""
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, N, D = x.shape
        if N != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got N={N}")
        if D != self.num_heads * self.head_dim:
            raise ValueError("Embedding dim D must equal num_heads * head_dim.")

        # Pre-normalization
        x_norm = self.pre_norm(x)

        # [B, N, D] -> [B, H, Dh, N]
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1).contiguous()

        # FFT along sequence dimension
        F_fft = torch.fft.rfft(x_heads, dim=-1, norm="ortho")  # [B, H, Dh, F_seq]

        if self.adaptive:
            # Global context from mean pooling over tokens
            context = x_norm.mean(dim=1)  # [B, D]
            adapt_params = self.adaptive_mlp(context).view(
                B, self.num_heads, self.freq_bins, 2
            )
            adaptive_scale = adapt_params[..., 0]  # [B, H, F_seq]
            adaptive_bias = adapt_params[..., 1]   # [B, H, F_seq]
        else:
            adaptive_scale = torch.zeros_like(self.base_filter).unsqueeze(0)
            adaptive_bias = torch.zeros_like(self.base_bias).unsqueeze(0)

        # Broadcast to [B, H, 1, F_seq]
        adaptive_scale = adaptive_scale[:, :, None, :]
        adaptive_bias = adaptive_bias[:, :, None, :]

        effective_filter = (self.base_filter[None, :, None, :] * (1.0 + adaptive_scale)).to(F_fft.dtype)
        effective_bias = (self.base_bias[None, :, None, :] + adaptive_bias).to(F_fft.dtype)

        F_fft_mod = F_fft * effective_filter + effective_bias
        F_fft_nl = self.complex_activation(F_fft_mod)

        # Inverse FFT along sequence dimension, back to [B, H, Dh, N]
        x_filtered_heads = torch.fft.irfft(F_fft_nl, n=self.seq_len, dim=-1, norm="ortho")

        # [B, H, Dh, N] -> [B, N, D]
        x_filtered = x_filtered_heads.permute(0, 3, 1, 2).contiguous().view(B, N, D)

        if return_attention:
            # Optional simple "attention score" per patch token
            att_score = x_filtered[:, 1:, :].norm(dim=-1)  # [B, N_patches]
            return x_filtered, att_score
        return x_filtered


# ---------------------- FFT Transformer Encoder Block ----------------------


class FFTTransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_module: Optional[nn.Module] = None,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if attention_module is None:
            raise ValueError("attention_module must be provided.")

        self.attention = attention_module
        self.norm1 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------- Patch Embedding with Physics Features ----------------------


class PatchEmbedWithPhysics(nn.Module):
    """
    Patch embedding for RGB images with additional per-patch physics features.

    - Three independent Conv2d layers are used for R/G/B channels.
    - Patch-level physics features (e.g. porosity and morphology descriptors)
      are projected by a linear layer.
    - A learnable gate fuses image and physics embeddings.
    """

    def __init__(
        self,
        patch_size: int = 56,
        in_chans: int = 3,
        embed_dim: int = 64,
        physics_dim: int = 6,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        if embed_dim % in_chans != 0:
            raise ValueError("embed_dim must be divisible by in_chans.")

        ch_dim = embed_dim // in_chans
        self.patch_embed_r = nn.Conv2d(1, ch_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_g = nn.Conv2d(1, ch_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_b = nn.Conv2d(1, ch_dim, kernel_size=patch_size, stride=patch_size)

        self.ln = nn.LayerNorm(embed_dim)
        self.physics_projection = nn.Linear(physics_dim, embed_dim)
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor, patch_feats: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size

        # R/G/B channel-specific patch embeddings
        x_r = self.patch_embed_r(x[:, 0:1, :, :]).flatten(2).transpose(1, 2)
        x_g = self.patch_embed_g(x[:, 1:2, :, :]).flatten(2).transpose(1, 2)
        x_b = self.patch_embed_b(x[:, 2:3, :, :]).flatten(2).transpose(1, 2)

        x_img = torch.cat([x_r, x_g, x_b], dim=-1)  # [B, N_patches, embed_dim]
        x_img = self.ln(x_img)

        x_phys = self.physics_projection(patch_feats)    # [B, N_patches, embed_dim]
        gate_in = torch.cat([x_img, x_phys], dim=-1)     # [B, N_patches, 2*embed_dim]
        gate = torch.sigmoid(self.gate_linear(gate_in))  # [B, N_patches, embed_dim]
        fused = gate * x_img + (1.0 - gate) * x_phys
        return fused


# ---------------------- Main Model ----------------------


class FFTPermeabilityPredictorPatchPhysics(nn.Module):
    """
    FFT-based permeability predictor with physics-aware patch embeddings.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        depth: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.seq_len = num_patches + 1  # +1 for CLS token

        self.patch_embed = PatchEmbedWithPhysics(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            physics_dim=6,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [
                FFTTransformerEncoderBlock(
                    embed_dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    attention_module=MultiHeadSpectralAttention(
                        embed_dim=embed_dim,
                        seq_len=self.seq_len,
                        num_heads=num_heads,
                    ),
                    drop_path_prob=0.0,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, patch_feats: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(x, patch_feats)          # [B, N_patches, C]
        cls_token = self.cls_token.expand(feat.size(0), -1, -1)
        x = torch.cat((cls_token, feat), dim=1)          # [B, 1 + N_patches, C]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        out = self.head(cls).squeeze(-1)
        return out
