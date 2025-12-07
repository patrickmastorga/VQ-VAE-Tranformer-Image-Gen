import torch
import torch.nn as nn

# hyperparameters for model
VOCAB_SIZE = 512 + 1 # 512 codebooks and 1 BOS token
SEQ_LEN = 256
BOS_ID = 512

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 12
D_FF = 4 * D_MODEL
DROPOUT = 0.1

class DecoderBlock(nn.Module):
    """
    Standard decoder self-attention block with causal masking
    """
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=N_HEADS, dropout=DROPOUT, batch_first=True)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT)
        )
        self.register_buffer("causal_mask", torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), 1).bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor) A FloatTensor of shape (B, S, D_MODEL) where S is the sequence we are attending over
        Returns:
            transformed (torch.Tensor) A FloatTensor of shape (B, S, D_MODEL) of transformed embeddings
        """
        B, S, D = x.shape
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=self.causal_mask[:S, :S]) # type: ignore
        x = x + x_attn
        return x + self.ff(self.norm2(x))


class TransformerPrior(nn.Module):
    """
    Autoregressive Prior for generating unconditional samples of CelebA embeddings
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.positional_embedding = nn.Embedding(SEQ_LEN, D_MODEL)
        self.decoder_stack = nn.Sequential(*[DecoderBlock() for _ in range(N_LAYERS)])
        self.to_logits = nn.Linear(D_MODEL, VOCAB_SIZE)

    @torch.no_grad()
    def generate(self, N: int, temp=1.0) -> torch.Tensor:
        device = next(self.parameters()).device
        seq = torch.full((N, 1), BOS_ID, dtype=torch.long, device=device)

        for i in range(SEQ_LEN):
            logits = self(seq)               # (N, i + 1, VOCAB_SIZE)
            logits = logits[:, -1, :] / temp # (N, VOCAB_SIZE)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (N, 1)
            seq = torch.cat([seq, next_token], dim=1) # (N, i + 2)

        return seq[:, 1:] # (N, SEQ_LEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor) a LongTensor of shape (B, S) of integer ids
        Returns:
            logits (torch.Tensor) a FloatTensor of shape (B, S, VOCAB_SIZE) of predicted next token logits
        """
        B, S = x.shape
        pos = torch.arange(S, device=x.device)
        x = self.token_embedding(x) + self.positional_embedding(pos) # (B, S, D) + (S, D)
        x = self.decoder_stack(x)
        return self.to_logits(x)