import torch
import torch.nn as nn

# hyperparameters for model
VOCAB_SIZE = 512
SEQ_LEN = 256
BOS_ID = VOCAB_SIZE + 1

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
DROPOUT = 0.1

class DecoderBlock(nn.Module):
    """
    Standard decoder self-attention block with causal masking
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=DROPOUT, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor) A FloatTensor of shape (B, S, D_MODEL) where S is the sequence we are attending over
            attn_mask (torch.Tensor) A BoolTensor of shape (S, S) where attn_mask[i, j] = True means the ith token cannot attend to the jth token
        Returns:
            transformed (torch.Tensor) A FloatTensor of shape (B, S, D_MODEL) of transformed embeddings
        """
        B, S, D = x.shape
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + x_attn
        return x + self.ff(self.norm2(x))


class TransformerPrior(nn.Module):
    """
    Autoregressive Prior for generating unconditional samples of CelebA embeddings
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE + 1, D_MODEL) # VOCAB_SIZE codebooks and 1 BOS token
        self.positional_embedding = nn.Embedding(SEQ_LEN, D_MODEL)
        self.decoder_stack = nn.ModuleList([DecoderBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.to_logits = nn.Linear(D_MODEL, VOCAB_SIZE + 1) # REMOVE THIS LATER!

        self.register_buffer("causal_mask", torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool())

    @torch.no_grad()
    def generate(self, N: int, temp: float = 1.0) -> torch.Tensor:
        """
        Args:
            N (int): The number of sequences to generate
            temp (float): divides the logits before sampling
        Returns:
            sequences (torch.Tensor): A LongTensor of shape (N, SEQ_LEN) of generates sequences
        """
        seq = torch.full((N, 1), BOS_ID, dtype=torch.long, device=next(self.parameters()).device)

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

        mask = self.causal_mask[:S, :S] # type: ignore
        for i in range(N_LAYERS):
            x = self.decoder_stack[i](x, attn_mask=mask)
        return self.to_logits(x)