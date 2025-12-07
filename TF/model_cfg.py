import torch
import torch.nn as nn

# hyperparameters for model
# CELEBA_ATTRIBUTES = ['Attractive', 'Smiling', 'Young', 'Male', 'Blonde_Hair', 'Black_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald', 'Glasses', 'No_Beard']
K = 12
VOCAB_SIZE = 512
SEQ_LEN = 256
BOS_ID = VOCAB_SIZE + 1

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 6
DROPOUT = 0.1

from TF.model import DecoderBlock

class CFGTransformerPrior(nn.Module):
    """
    Autoregressive Prior for generating conditional samples of CelebA embeddings
    Has a fixed knowledge of K fixed binary attribute signals
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE + 3 * K + 1, D_MODEL) # VOCAB_SIZE codebooks, 3 tokens for each attribute (true, false, masked) and 1 BOS token
        self.positional_embedding = nn.Embedding(SEQ_LEN, D_MODEL)
        self.decoder_stack = nn.ModuleList([DecoderBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.to_logits = nn.Linear(D_MODEL, VOCAB_SIZE)

        self.register_buffer("attn_mask", torch.triu(torch.ones(SEQ_LEN + K, SEQ_LEN + K), diagonal=1).bool())
        self.attn_mask[:K, :K] = False # attribute tokens can always attend to each other # type: ignore

    @torch.no_grad()
    def generate(self, N: int, attrs: torch.Tensor, w: float, temp: float = 1.0) -> torch.Tensor:
        """
        Args:
            N (int): The number of sequences to generate
            attrs (torch.Tensor): a LongTensor of shape (B, K) where 0: False, 1: True, 2: MASKED
            w (float): the guidance weight for the conditional generation
            temp (float): divides the logits before sampling
        Returns:
            sequences (torch.Tensor): A LongTensor of shape (N, SEQ_LEN) of generates sequences
        """
        attrs_masked = torch.full_like(attrs, 2)
        seq = torch.full((N, 1), BOS_ID, dtype=torch.long, device=attrs.device)

        for i in range(SEQ_LEN):
            logits_uncond = self(seq, attrs_masked) # (N, i + 1, VOCAB_SIZE)
            logits_cond = self(seq, attrs)   # (N, i + 1, VOCAB_SIZE)
            logits = (1 - w) * logits_uncond[:, -1, :] + w * logits_cond[:, -1, :] # (N, VOCAB_SIZE)
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (N, 1)
            seq = torch.cat([seq, next_token], dim=1) # (N, i + 2)

        return seq[:, 1:] # (N, SEQ_LEN)

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor) a LongTensor of shape (B, S) of integer ids
            attrs (torch.Tensor) a LongTensor of shape (B, K) where 0: False, 1: True, 2: MASKED
        Returns:
            logits (torch.Tensor) a FloatTensor of shape (B, S, VOCAB_SIZE) of predicted next token logits
        """
        B, S = x.shape

        attr_ids = attrs + torch.arange(0, 3 * K, 3).unsqueeze(0) + VOCAB_SIZE + 2 # (B, K)
    
        pos = torch.arange(S, device=x.device)
        x = self.token_embedding(x) + self.positional_embedding(pos) # (B, S, D) + (S, D)
        x = torch.cat([self.token_embedding(attr_ids), x], dim=1) # (B, K + S, D)

        mask = self.attn_mask[:K + S, :K + S] # type: ignore
        for i in range(N_LAYERS):
            x = self.decoder_stack[i](x, attn_mask=mask)
        return self.to_logits(x[:, K:, :]) # ignore the logits corresponding to the attribute embeddings