
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
        self.decay = decay

        # EMA running cluster counts
        expected_count = batch_size * LATENT_W * LATENT_H / NUM_EMBEDDINGS
        self.register_buffer('N', torch.full((NUM_EMBEDDINGS,), expected_count))

        # codebook dictionary
        if not self.use_EMA:
            self.register_parameter('e', nn.Parameter(torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM)))
        else:
            self.register_buffer('e', torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM))
            # EMA running cluster sums
            self.register_buffer('m', self.e.clone() * expected_count)

    def nearest_neighbor_indices(self, z_e: torch.Tensor) -> torch.Tensor:
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
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING)
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING)
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
        x = nn.functional.embedding(x, self.e)    # (B, H, W, EMBEDDING_DIM)
        return x.permute(0, 3, 1, 2).contiguous() # (B, EMBEDDING_DIM, H, W)

    def forward(self, z_e: torch.Tensor, refresh_dead: bool = False) -> torch.Tensor:
        """
        If use_EMA = True, updates the codebook dictionary
        Args:
            z_e (torch.Tensor): encoder output FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        Returns:
            z_q (torch.Tensor): quantized latent FloatTensor of shape (B, EMBEDDING_DIM, LATENT_H, LATENT_W)
        """
        # Dead codebook refresh
        if self.refresh_dead:
            p = self.N / self.N.sum() * NUM_EMBEDDINGS

            # find codes with p < threshold
            dead_idx = torch.where(p < 0.1)[0]

            if len(dead_idx) > 0:
                choice = torch.randint(0, z_e_flat.shape[0], (len(dead_idx),), device=z_e_flat.device)
                self.e[dead_idx] = z_e_flat[choice]

        # flatten the embeddings along batch size, height, and width (B, embedding_dim, H, W) -> (BHW, embedding_dim)
        B, _, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, EMBEDDING_DIM)

        # to calculate pairwise distance, use ||z - e||^2 = ||z||^2 - 2z*e + ||e||^2
        with torch.no_grad():
            dist = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True) # ||z||^2 (BHW, 1)
                + self.e.pow(2).sum(dim=1).unsqueeze(0)  # ||e||^2 (1, NUM_EMBEDDING)
                - 2 * z_e_flat @ self.e.T                # -2z*e   (BHW, NUM_EMBEDDING)
            )
        indices_flat = dist.argmin(1)

        z_q = nn.functional.embedding(indices_flat, self.e).view(B, H, W, EMBEDDING_DIM) # (B, H, W, embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()                                       # (B, embedding_dim, H, W)

        if self.training:
            # current minibatch cluster counts
            n_i = torch.bincount(indices_flat, minlength=NUM_EMBEDDINGS).float()
            self.N = self.decay * self.N + (1 - self.decay) * n_i

            # EMA codebook update
            if self.use_EMA:
                with torch.no_grad():
                    # current minibatch cluster sums
                    m_i = torch.zeros_like(self.e)
                    m_i.index_add_(0, indices_flat, z_e_flat)
                    self.m = self.decay * self.m + (1 - self.decay) * m_i

                    # EMA updates
                    self.e = self.m / (self.N.unsqueeze(1) + 1e-8)

        return z_q

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
    def compute_indices(self, image: torch.Tensor) -> torch.Tensor:
        """
        with torch.no_grad()
        Args:
            image (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        Returns:
            indices (torch.Tensor): index LongTensor of shape (B, LATENT_H, LATENT_W)
        """
        x = self.encoder(image)
        return self.quantizer.nearest_neighbor_indices(x)

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

    def forward(self, input: torch.Tensor, refresh_dead: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input (torch.Tensor): image FloatTensor of shape (B, 3, IMG_H, IMG_W) in range [0, 1]
        Returns:
            losses (tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]): reconstruction_loss, commitment_loss, codebook_loss; if use_EMA=True, codebook_loss is None
        """
        B = input.shape[0]
        z_e = self.encoder(input)
        z_q = self.quantizer(z_e, refresh_dead)

        # straight through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        reconstructed = self.decoder(z_q_st)

        # compute loss
        reconstruction_loss = nn.functional.mse_loss(reconstructed, input, reduction='sum') / B
        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
        if self.use_EMA:
            return reconstruction_loss, commitment_loss, None
        else:
            codebook_loss = nn.functional.mse_loss(z_e.detach(), z_q)
            return reconstruction_loss, commitment_loss, codebook_loss