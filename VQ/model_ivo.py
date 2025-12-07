import torch
import torch.nn as nn


LATENT_W = LATENT_H = 16     # latent grid 8x8 → 64 tokens
IMG_W = LATENT_W * 4        # 64×64 images
IMG_H = LATENT_H * 4

EMBEDDING_DIM = LATENT_W * LATENT_H   # 128-dimensional embedding
NUM_EMBEDDINGS = 512
HIDDEN_CHANNELS = 256



class ResidualBlock(nn.Module):
    """
    Standard VQ-VAE ResBlock:
    ReLU → Conv3×3 → ReLU → Conv1×1 → skip connection.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(x)


class SelfAttention2d(nn.Module):
    """
    Self-attention over spatial tokens at low resolution.
    Uses MultiheadAttention on flattened (H*W) positions.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # (B, HW, C)

        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)

        out = attn_out + x_flat  # residual
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out



class Encoder(nn.Module):
    """
    3→256→256→EmbeddingDim
    Downsample ×3 → (8×8) latents
    """
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),  # 64→32
            ResidualBlock(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),  # 32→16
            ResidualBlock(),
            nn.Conv2d(HIDDEN_CHANNELS, EMBEDDING_DIM, kernel_size=3, stride=1, padding=1),     # 16→16
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    """
    Strong Decoder:
    - Conv1x1
    - ResBlock → Attention → ResBlock
    - 3× Upsample(Nearest + Conv + ResBlocks)
    - Final Conv → Sigmoid
    """
    def __init__(self):
        super().__init__()

        body = []

        # 16×16 resolution
        body.append(nn.Conv2d(EMBEDDING_DIM, HIDDEN_CHANNELS, kernel_size=1))
        body.append(ResidualBlock())
        body.append(SelfAttention2d(HIDDEN_CHANNELS, num_heads=4))
        body.append(ResidualBlock())
 
        # 16 → 16
        body.append(nn.Upsample(scale_factor=1, mode="nearest"))
        body.append(nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1))
        body.append(ResidualBlock())

        # 16 → 32
        body.append(nn.Upsample(scale_factor=2, mode="nearest"))
        body.append(nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1))
        body.append(ResidualBlock())

        # 32 → 64
        body.append(nn.Upsample(scale_factor=2, mode="nearest"))
        body.append(nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1))
        body.append(ResidualBlock())

        self.body = nn.Sequential(*body)

        self.to_img = nn.Conv2d(HIDDEN_CHANNELS, 3, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()  # match your training code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)           # (B, 256, 64, 64)
        x = self.to_img(x)         # (B, 3, 64, 64)
        return self.act(x)



class Quantizer(nn.Module):
    """
    Codebook with nearest neighbor lookup + optional EMA updates.
    """
    def __init__(self, use_EMA=False, batch_size=0, decay=0.99):
        super().__init__()
        self.use_EMA = use_EMA

        if not use_EMA:
            self.register_parameter("e", nn.Parameter(torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM)))
        else:
            self.register_buffer("e", torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM))
            self.decay = decay
            expected_count = batch_size * LATENT_W * LATENT_H / NUM_EMBEDDINGS
            self.register_buffer("N", torch.full((NUM_EMBEDDINGS,), expected_count))
            self.register_buffer("m", self.e.clone() * expected_count) # type: ignore

    def nearest_neighbor_indices(self, x):
        B, _, H, W = x.shape
        z = x.permute(0,2,3,1).reshape(-1, EMBEDDING_DIM)

        with torch.no_grad():
            dist = (
                z.pow(2).sum(1, keepdim=True) +
                self.e.pow(2).sum(1)[None, :] - # type: ignore
                2 * z @ self.e.T # type: ignore
            )
        idx = dist.argmin(1)
        return idx.view(B, H, W)

    def get_latent_tensor_from_indices(self, idx):
        z_q = nn.functional.embedding(idx, self.e)    # (B,H,W,EmbDim) # type: ignore
        return z_q.permute(0,3,1,2).contiguous()

    
    
    def forward(self, x):
        B, _, H, W = x.shape
        z = x.permute(0,2,3,1).reshape(-1, EMBEDDING_DIM)

        with torch.no_grad():
            dist = (
                z.pow(2).sum(1, keepdim=True) +
                self.e.pow(2).sum(1)[None, :] - # type: ignore
                2 * z @ self.e.T
            )
        idx = dist.argmin(1)

        # ----- FIXED EMA BLOCK -----
        if self.use_EMA and self.training:
            with torch.no_grad():
                # use detached z so no grad is tracked
                z_flat = z.detach()
                n_i = torch.bincount(idx, minlength=NUM_EMBEDDINGS).float()
                m_i = torch.zeros_like(self.e) # type: ignore
                m_i.index_add_(0, idx, z_flat)

                # in-place EMA updates, no new graph
                self.N.mul_(self.decay).add_((1.0 - self.decay) * n_i) # type: ignore
                self.m.mul_(self.decay).add_((1.0 - self.decay) * m_i) # type: ignore
                self.e.copy_(self.m / (self.N[:, None] + 1e-8)) # type: ignore
        # ---------------------------

        z_q = nn.functional.embedding(idx, self.e).view(B, H, W, EMBEDDING_DIM) # type: ignore
        return z_q.permute(0,3,1,2).contiguous()

class VQ_VAE(nn.Module):
    def __init__(self, encoder, decoder, quantizer, use_EMA=False, beta=0.25):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.use_EMA = use_EMA
        self.beta = beta

    @torch.no_grad()
    def compute_indices(self, x):
        return self.quantizer.nearest_neighbor_indices(self.encoder(x))

    @torch.no_grad()
    def reconstruct_from_indices(self, idx):
        z_q = self.quantizer.get_latent_tensor_from_indices(idx)
        return self.decoder(z_q)

    @torch.no_grad()
    def reconstruct(self, x):
        idx = self.compute_indices(x)
        return self.reconstruct_from_indices(idx)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        recon = self.decoder(z_q_st)

        recon_loss = nn.functional.l1_loss(recon, x)
        commit_loss = nn.functional.mse_loss(z_e, z_q.detach())

        if self.use_EMA:
            codebook_loss = nn.functional.mse_loss(z_e.detach(), z_q)
            return recon_loss, commit_loss, codebook_loss

        return recon_loss, commit_loss, None