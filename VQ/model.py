import torch
import torch.nn as nn

# model hyperparemters
LATENT_W = 8
LATENT_H = 8
IMG_W = LATENT_W * 8
IMG_H = LATENT_H * 8

EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512
HIDDEN_CHANNELS = 256

class ResidualBlock(nn.Module):
    """
    implementation of the residual block as described in section 4.1 of the original VQ-VAE paper\\
    ReLU -> 3x3 conv -> ReLU -> 1x1 conv -> skip connection
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(x)

class Encoder(nn.Module):
    """
    maps an (3, IMG_H, IMG_W) image tensor to a (EMBEDDING_DIM, LATENT_H, LATENT_W) latent tensor\\
    downsample -> residual block -> downsample -> residual block -> downsample -> residual block -> 1x1 conv
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # (3, IMG_H, IMG_W) image
            nn.Conv2d(in_channels=3, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/2, IMG_W/2) hidden
            ResidualBlock(),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/4, IMG_W/4) hidden
            ResidualBlock(),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/8, IMG_W/8) = (HIDDEN_CHANNELS, LATENT_H, LATENT_W) hidden
            ResidualBlock(),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=EMBEDDING_DIM, kernel_size=1),
            # (EMBEDDING_DIM, LATENT_H, LATENT_W) latents
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Decoder(nn.Module):
    """
    maps a (EMBEDDING_DIM, LATENT_H, LATENT_W) quantized latent tensor to an (3, IMG_H, IMG_W) image tensor (scaled to [0, 1])\\
    1x1 conv -> residual block -> upsample -> residual block -> upsample -> residual block -> upsample
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # (EMBEDDING_DIM, LATENT_H, LATENT_W) latents
            nn.Conv2d(in_channels=EMBEDDING_DIM, out_channels=HIDDEN_CHANNELS, kernel_size=1),
            # (HIDDEN_CHANNELS, LATENT_H, LATENT_W) = (HIDDEN_CHANNELS, IMG_H/8, IMG_W/8) hidden
            ResidualBlock(),
            nn.ConvTranspose2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/4, IMG_W/4) hidden
            ResidualBlock(),
            nn.ConvTranspose2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # (HIDDEN_CHANNELS, IMG_H/2, IMG_W/2) hidden
            ResidualBlock(),
            nn.ConvTranspose2d(in_channels=HIDDEN_CHANNELS, out_channels=3, kernel_size=4, stride=2, padding=1),
            # (3, IMG_H, IMG_W) image
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Quantizer(nn.Module):
    """
    implementation of the codebook with nearnest neighbor lookup\\
    if use_EMA=True, embeddings are learnt automatically as exponential moving averages of the encoder outputs assigned to them over minibatches (see Appendix A.1 of the original VQ-VAE paper)\\
    otherwise, the embeddings are parameters to be learnt with gradient descent on the codebook loss (see section 3.2 of the original VQ-VAE paper)
    """
    def __init__(self, use_EMA=False, batch_size=0, decay=0.99):
        """
        Args:
            use_EMA (bool): if True, use EMA updates to learn the codebook during training
            batch_size (int): used to initialize the EMA running cluster counts/sums
            decay (float): EMA decay parameter
        """
        super().__init__()
        self.use_EMA = use_EMA

        # codebook dictionary
        if not self.use_EMA:
            self.register_parameter('e', nn.Parameter(torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM)))
        else:
            self.register_buffer('e', torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM))

            # EMA running cluster counts and sums
            self.decay = decay
            expected_count = batch_size * LATENT_W * LATENT_H / NUM_EMBEDDINGS
            self.register_buffer('N', torch.full((NUM_EMBEDDINGS,), expected_count))
            self.register_buffer('m', self.e.clone() * expected_count)

    def nearest_neighbor_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        (EMBEDDING_DIM, LATENT_H, LATENT_W) latent tensor -> (LATENT_H, LATENT_W) index tensor
        """
        # flatten the embeddings along batch size, height, and width (B, embedding_dim, H, W) -> (BHW, embedding_dim)
        B, _, H, W = x.shape
        z_e_flat = x.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True) # ||z||^2 (BHW, 1)
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING)
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING)
            )
        indices_flat = dist.argmin(1)                                   # (BHW,)
        return indices_flat.view(B, H, W).permute(0, 1, 2).contiguous() # (B, H, W)
    
    def get_latent_tensor_from_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        (LATENT_H, LATENT_W) index tensor -> (EMBEDDING_DIM, LATENT_H, LATENT_W) latent tensor
        """
        x = nn.functional.embedding(x, self.e)    # (B, H, W, EMBEDDING_DIM)
        return x.permute(0, 3, 1, 2).contiguous() # (B, EMBEDDING_DIM, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the embeddings along batch size, height, and width (B, embedding_dim, H, W) -> (BHW, embedding_dim)
        B, _, H, W = x.shape
        z_e_flat = x.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True) # ||z||^2 (BHW, 1)
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING)
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING)
            )
        indices_flat = dist.argmin(1)

        # EMA codebook update
        if self.use_EMA and self.training:
            # current minibatch cluster counts
            n_i = torch.bincount(indices_flat, minlength=NUM_EMBEDDINGS).float()

            with torch.no_grad():
                # current minibatch cluster sums
                m_i = torch.zeros_like(self.e)
                m_i.index_add_(0, indices_flat, z_e_flat)

                # EMA updates
                self.N = self.decay * self.N + (1 - self.decay) * n_i
                self.m = self.decay * self.m + (1 - self.decay) * m_i
                self.e = self.m / (self.N.unsqueeze(1) + 1e-8)

        z_q = nn.functional.embedding(indices_flat, self.e).view(B, H, W, EMBEDDING_DIM) # (B, H, W, embedding_dim)
        return z_q.permute(0, 3, 1, 2).contiguous()                                      # (B, embedding_dim, H, W)

class VQ_VAE(nn.Module):
    """
        implements the encoder, decoder, and quantizer into a single model for training
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, quantizer: Quantizer, use_EMA=False, beta=0.25):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.use_EMA = use_EMA
        self.beta = beta

    @torch.no_grad()
    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        (3, IMG_H, IMG_W) image tensor -> (LATENT_H, LATENT_W) index tensor (without computing gradients)
        """
        x = self.encoder(x)
        return self.quantizer.nearest_neighbor_indices(x)

    @torch.no_grad()
    def reconstruct_from_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        (LATENT_H, LATENT_W) index tensor -> (3, IMG_H, IMG_W) reconstructed image tensor (without computing gradients)
        """
        x = self.quantizer.get_latent_tensor_from_indices(x)
        return self.decoder(x)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        (3, IMG_H, IMG_W) image tensor -> (3, IMG_H, IMG_W) reconstructed image tensor (without computing gradients)
        """
        x = self.compute_indices(x)
        return self.reconstruct_from_indices(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        (3, IMG_H, IMG_W) image tensor -> reconstruction_loss, commitment_loss, codebook_loss\\
        if use_EMA=True, codebook_loss is None
        """
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)

        # straight through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        reconstructed = self.decoder(z_q_st)

        # compute loss
        reconstruction_loss = nn.functional.mse_loss(reconstructed, x)
        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
        if self.use_EMA:
            codebook_loss = nn.functional.mse_loss(z_e.detach(), z_q)
            return reconstruction_loss, commitment_loss, codebook_loss
        else:
            return reconstruction_loss, commitment_loss, None