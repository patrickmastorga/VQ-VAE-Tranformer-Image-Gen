import torch
import torch.nn as nn

# these are the hyperparameters used in the original VQ-VAE paper (see section 4.1)
HIDDEN_CHANNELS = 256
LATENT_DIM = 8 * 8
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512

class ResidualBlock(nn.Module):
    """
    implementation of the residual block as described in section 4.1 of the original VQ-VAE paper\\
    ReLU -> 3x3 conv -> ReLU -> 1x1 conv -> skip connection
    """
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.network(x)

class Encoder(nn.Module):
    """
    maps an 64x64 image tensor to a 8x8 latent tensor\\
    downsample -> residual block -> downsample -> residual block -> downsample -> residual block -> 1x1 conv
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 64x64 image
            nn.Conv2d(in_channels=3, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # 32x32 hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # 16x16 hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # 8x8 hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.Conv2d(in_channels=HIDDEN_CHANNELS, out_channels=EMBEDDING_DIM, kernel_size=1),
            # 8x8 latents
        )

    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    """
    maps a 8x8 latent tensor to an 64x64 image tensor\\
    1x1 conv -> residual block -> upsample -> residual block -> upsample -> residual block -> upsample
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 8x8 latents
            nn.Conv2d(in_channels=EMBEDDING_DIM, out_channels=HIDDEN_CHANNELS, kernel_size=1),   
            # 8x8 hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.ConvTranspose2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # 16x16 hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.ConvTranspose2d(in_channels=HIDDEN_CHANNELS, out_channels=HIDDEN_CHANNELS, kernel_size=4, stride=2, padding=1),
            # 32x32 hidden
            ResidualBlock(HIDDEN_CHANNELS),
            nn.ConvTranspose2d(in_channels=HIDDEN_CHANNELS, out_channels=3, kernel_size=4, stride=2, padding=1),
            # 64x64 image
        )

    def forward(self, x):
        return self.network(x)

class Quantizer(nn.Module):
    """
        implementation of the codebook with nearnest neighbor lookup\\
        the embeddings are parameters to be learnt with gradient descent on the codebook loss (see section 3.2 of the original VQ-VAE paper)
    """
    def __init__(self):
        super().__init__()
        # codebook dictionary
        self.e = nn.Embedding(NUM_EMBEDDINGS, EMBEDDING_DIM)

    def foward(self, z_e):
        B, _, H, W = z_e.shape

        # flatten the embeddings along batch size, height, and width (B, EMBEDDING_DIM, H, W) -> (BHW, EMBEDDING_DIM)
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True)        # ||z||^2 (BHW, 1)
                + self.e.weight.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING)
                - 2 * z_e_flat @ self.e.weight.T                # -2z*e   (BHW, NUM_EMBEDDING)
            )
        indices_flat = dist.argmin(1)

        # lookup embeddings from codebook
        z_q = self.e(indices_flat).view(B, H, W, EMBEDDING_DIM) # (B, H, W, embedding_dim)
        return z_q.permute(0, 3, 1, 2).contiguous()             # (B, embedding_dim, H, W)

class QuantizerEMA(nn.Module):
    """
        implementation of the codebook with nearnest neighbor lookup\\
        the embeddings are learnt automatically as exponential movig averages of the encoder outputs assigned to them over minibatches (see Appendix A.1 of the original VQ-VAE paper)
    """
    def __init__(self, batch_size, decay=0.99):
        super().__init__()
        self.decay = decay

        # codebook dictionary
        self.e = torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM)
        self.register_buffer('e', self.e)

        # EMA running cluster counts and sums
        expected_count = batch_size * LATENT_DIM / NUM_EMBEDDINGS
        self.register_buffer('N', torch.full((NUM_EMBEDDINGS,), expected_count))
        self.register_buffer('m', self.e.clone() * expected_count)
    
    def foward(self, z_e):
        B, _, H, W = z_e.shape

        # flatten the embeddings along batch size, height, and width (B, embedding_dim, H, W) -> (BHW, embedding_dim)
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True) # ||z||^2 (BHW, 1)
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING)
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING)
            )
        indices_flat = dist.argmin(1)

        if self.training:
            with torch.no_grad():
                # current minibatch cluster counts
                n_i = torch.bincount(indices_flat, minlength=NUM_EMBEDDINGS).float()

                # current minibatch cluster sums
                m_i = torch.zeros_like(self.e)
                m_i.index_add_(0, indices_flat, z_e_flat)

                # EMA updates
                self.N = self.decay * self.N + (1 - self.decay) * n_i
                self.m = self.decay * self.m + (1 - self.decay) * m_i
                self.e = self.m / (self.N.unsqueeze(1) + 1e8)
        
        z_q = nn.functional.embedding(indices_flat, self.e).view(B, H, W, EMBEDDING_DIM) # (B, H, W, embedding_dim)
        return z_q.permute(0, 3, 1, 2).contiguous()                                      # (B, embedding_dim, H, W)

class VQ_VAE(nn.Module):
    """
        implements the encoder, decoder, and quantizer into a single model for training (with codebook loss) (see section 3.2 of the original VQ-VAE paper)
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, quantizer: Quantizer):
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
    
    def foward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)

        # straight through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        codebook_loss = nn.functional.mse_loss(z_e.detach(), z_q)
        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
        reconstructed = self.decoder(z_q_st)

        return reconstructed, codebook_loss, commitment_loss
    
class VQ_VAE_EMA(nn.Module):
    """
        implements the encoder, decoder, and quantizer into a single model for training (with EMA dictionalty learning) (see section 3.2/Appendix A.1 of the original VQ-VAE paper)
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, quantizer: QuantizerEMA):
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
    
    def foward(self, x):
        z_e = self.encoder(x)
        z_q = self.quantizer(z_e)

        # straight through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
        reconstructed = self.decoder(z_q_st)
        
        return reconstructed, commitment_loss