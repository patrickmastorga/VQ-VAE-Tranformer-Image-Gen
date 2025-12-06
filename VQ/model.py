import torch
import torch.nn as nn

# model hyperparemters
LATENT_W = 16
LATENT_H = 16
IMG_W = LATENT_W * 4
IMG_H = LATENT_H * 4

EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512
HIDDEN_CHANNELS = 256

class ResidualBlock(nn.Module):
    """
    Convolutional residual block for 2D data of shape (B, C, H, W)\\
    GN -> SiLU -> 3x3 conv -> GN -> SiLU -> 1x1 conv -> skip connection
    """
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),

            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
    
class SelfAttentionBlock2d(nn.Module):
    """
    Self-attention block over 2D data of shape (B, C, H, W)\\
    Uses MultiheadAttention on flattened (H*W) positions.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # (B, HW, C)

        # x = x + attn(norm(x))
        x_norm = self.norm1(x_flat)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_flat = x_flat + x_attn

        # x = x + ff(norm(x))
        x_flat = x_flat + self.ff(self.norm2(x_flat))
        
        return x_flat.permute(0, 2, 1).view(B, C, H, W)

class Encoder(nn.Module):
    """
    Simple ResNet style encoder
    maps an (3, IMG_H, IMG_W) image tensor to a (EMBEDDING_DIM, LATENT_H, LATENT_W) latent tensor\\
    downsample -> residual block -> downsample -> residual block -> 1x1 conv
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # (3, IMG_H, IMG_W) image
            nn.Conv2d(in_channels=3, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/2, IMG_W/2) hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/4, IMG_W/4) = (HIDDEN_CHANNELS, LATENT_H, LATENT_W) hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=EMBEDDING_DIM, kernel_size=1),
            # (EMBEDDING_DIM, LATENT_H, LATENT_W) latents
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        Returns:
            z_e (torch.Tensor): encoder output FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        """
        return self.network(image)

class Decoder(nn.Module):
    """
    Complex decoder with attention layer at lowest resolution and nearest neighbor upsampling with residual blocks
    maps a (EMBEDDING_DIM, LATENT_H, LATENT_W) quantized latent tensor to an (3, IMG_H, IMG_W) image tensor (scaled to [0, 1])\\
    1x1 conv -> resblocks + attn -> upsample + conv -> resblocks -> upsample + conv -> resblocks -> 1x1 conv + sigmoid
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # (EMBEDDING_DIM, LATENT_H, LATENT_W) latents
            nn.Conv2d(in_channels=EMBEDDING_DIM, out_channels=HIDDEN_CHANNELS, kernel_size=1),
            # (HIDDEN_CHANNELS, LATENT_H, LATENT_W) = (HIDDEN_CHANNELS, IMG_H/4, IMG_W/4) hidden

            ResidualBlock(HIDDEN_CHANNELS),
            SelfAttentionBlock2d(HIDDEN_CHANNELS),
            ResidualBlock(HIDDEN_CHANNELS),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/2, IMG_W/2) hidden

            ResidualBlock(HIDDEN_CHANNELS),
            ResidualBlock(HIDDEN_CHANNELS),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            # (HIDDEN_CHANNELS, IMG_H, IMG_W) hidden

            ResidualBlock(HIDDEN_CHANNELS),
            ResidualBlock(HIDDEN_CHANNELS),

            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=3, kernel_size=1),
            nn.Sigmoid()
            # (3, IMG_H, IMG_W) image
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q (torch.Tensor): quantized latent FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        Returns:
            image (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        """
        return self.network(z_q)

class Quantizer(nn.Module):
    """
    implementation of the codebook with nearnest neighbor lookup\\
    Embeddings are learnt automatically as exponential moving averages of the cluster assignments over minibatches (see Appendix A.1 of "Neural Discrete Representational Learning")\\
    """
    def __init__(self, batch_size=0, decay=0.99):
        """
        Args:
            batch_size (int): used to initialize the EMA running cluster counts/sums
            decay (float): EMA decay parameter
        """
        super().__init__()
        self.decay = decay
        self.batch_size = batch_size

        # codebook dictionary
        self.register_buffer('e', torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM))

        # EMA running cluster counts/sums
        expected_count = self.batch_size * LATENT_W * LATENT_H / NUM_EMBEDDINGS
        self.register_buffer('N', torch.full((NUM_EMBEDDINGS,), expected_count))
        self.register_buffer('m', self.e.clone() * expected_count) # type: ignore

    def initialize_codebook(self, e: torch.Tensor, p: torch.Tensor) -> None:
        """
        Initializes the codebook from a Tensor
        Args:
            e (torch.Tensor): a FloatTensor of shape (NUM_EMBEDDINGS, EMBEDDING_DIM) to initialize the codebook with
            p (torch.Tensor): a FLoatTensor of shape (NUM_EMBEDDINGS,) containing relative cluster proportions (they sum to 1)
        """
        with torch.no_grad():
            self.e.data.copy_(e) # type: ignore
            self.N.data.copy_(self.batch_size * LATENT_W * LATENT_H * p) # type: ignore
            self.m.data.copy_(e * self.N.unsqueeze(1)) # type: ignore

    def get_indices_from_latent_tensor(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_e (torch.Tensor): encoder output FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        Returns:
            indices (torch.Tensor): index LongTensor of shape (B, LATENT_H, LATENT_W)
        """
        # flatten the embeddings along batch size, height, and width (B, embedding_dim, H, W) -> (BHW, embedding_dim)
        B, _, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True) # ||z||^2 (BHW, 1)
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING) # type: ignore
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING) # type: ignore
            )
        indices_flat = dist.argmin(1)                                   # (BHW,)
        return indices_flat.view(B, H, W).permute(0, 1, 2).contiguous() # (B, H, W)
    
    def get_latent_tensor_from_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): index LongTensor of shape (B, LATENT_H, LATENT_W)
        Returns:
            z_q (torch.Tensor): quantized latent FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        """
        x = nn.functional.embedding(x, self.e)    # (B, H, W, EMBEDDING_DIM) # type: ignore
        return x.permute(0, 3, 1, 2).contiguous() # (B, EMBEDDING_DIM, H, W)

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_e (torch.Tensor): encoder output FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        Returns:
            z_q (torch.Tensor): quantized latent FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        """
        # flatten the embeddings along batch size, height, and width (B, embedding_dim, H, W) -> (BHW, embedding_dim)
        B, _, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True) # ||z||^2 (BHW, 1)
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING) # type: ignore
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING) # type: ignore
            )
        indices_flat = dist.argmin(1)

        # lookup z_q from embedding codebook
        z_q = nn.functional.embedding(indices_flat, self.e).view(B, H, W, EMBEDDING_DIM) # (B, H, W, embedding_dim) # type: ignore
        z_q = z_q.permute(0, 3, 1, 2).contiguous()                                       # (B, embedding_dim, H, W)

        # EMA update
        if self.training:
            with torch.no_grad():
                n_i = torch.bincount(indices_flat, minlength=NUM_EMBEDDINGS).float()
                m_i = torch.zeros_like(self.e) # type: ignore
                m_i.index_add_(0, indices_flat, z_e_flat)

                # in-place EMA updates
                self.N.mul_(self.decay).add_((1.0 - self.decay) * n_i) # type: ignore
                self.m.mul_(self.decay).add_((1.0 - self.decay) * m_i) # type: ignore
                self.e.copy_(self.m / (self.N.unsqueeze(1) + 1e-8)) # type: ignore

                # dead codebook refresh
                p = self.N / self.N.sum() * NUM_EMBEDDINGS # type: ignore
                dead_idx = torch.where(p < 0.001)[0]
                if len(dead_idx) > 0:
                    choice = torch.randint(0, z_e_flat.shape[0], (len(dead_idx),), device=z_e_flat.device)
                    self.e[dead_idx] = z_e_flat[choice] # type: ignore
                    self.N[dead_idx] = self.N.sum() / NUM_EMBEDDINGS # give a small initialization to the cluster count # type: ignore
                    print(f'Reassigned {len(dead_idx)} codebooks!')

        return z_q

class VQ_VAE(nn.Module):
    """
        implements the encoder, decoder, and quantizer into a single model for training
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, quantizer: Quantizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    @torch.no_grad()
    def compute_indices(self, image: torch.Tensor) -> torch.Tensor:
        """
        with torch.no_grad()
        Args:
            image (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        Returns:
            indices (torch.Tensor): index LongTensor of shape (B, LATENT_H, LATENT_W)
        """
        x = self.encoder(image)
        return self.quantizer.get_indices_from_latent_tensor(x)

    @torch.no_grad()
    def reconstruct_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        with torch.no_grad()
        Args:
            indices (torch.Tensor): index LongTensor of shape (B, LATENT_H, LATENT_W)
        Returns:
            reconstructed (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        """
        x = self.quantizer.get_latent_tensor_from_indices(indices)
        return self.decoder(x)

    @torch.no_grad()
    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        """
        with torch.no_grad()
        Args:
            image (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        Returns:
            reconstructed (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        """
        x = self.compute_indices(image)
        return self.reconstruct_from_indices(x)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        Returns:
            losses (tuple[torch.Tensor, torch.Tensor]): reconstruction_loss, commitment_loss
        """
        z_e = self.encoder(input)
        z_q = self.quantizer(z_e)

        # straight through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        reconstructed = self.decoder(z_q_st)

        # compute losses
        reconstruction_loss = nn.functional.l1_loss(reconstructed, input)
        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
        return reconstruction_loss, commitment_loss