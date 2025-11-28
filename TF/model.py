import torch
import torch.nn as nn

# hyperparameters for model
VOCAB_SIZE = 512 + 1 # 512 codebooks and 1 BOS token
SEQ_LEN = 64
BOS_ID = 512

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
D_FF = 4 * D_MODEL
DROPOUT = 0.1

class DecoderBlock(nn.Module):
    """
    Implementation of the GPT style decoder block
    """
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=N_HEADS, dropout=DROPOUT, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(D_MODEL)
        self.feed_foward = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x_attn, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + x_attn
        x = self.layer_norm2(x)
        return x + self.feed_foward(x)


class TransformerPrior(nn.Module):
    """
    Autoregressive Prior for generating unconditional samples of CelebA embeddings
    """
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.positional_embedding = nn.Embedding(SEQ_LEN, D_MODEL)

        self.decoder_stack = nn.ModuleList([DecoderBlock() for _ in range(N_LAYERS)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, VOCAB_SIZE)
        )

        self.register_buffer("mask", torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), 1).bool())

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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, seq_len = idx.shape

        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.positional_embedding(torch.arange(seq_len, device=idx.device))

        # slice the causal mask to match input length
        attn_mask = self.mask[:seq_len, :seq_len]

        for block in self.decoder_stack:
            x = block(x, mask=attn_mask)

        return self.to_logits(x)