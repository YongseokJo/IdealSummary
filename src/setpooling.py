import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def safe_softmax(logits, mask, dim=-1, eps=1e-9):
    # logits: (..., N)
    # mask:   (..., N) in {0,1}
    logits = logits.masked_fill(mask == 0, float("-inf"))
    w = F.softmax(logits, dim=dim)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    w = w / (w.sum(dim=dim, keepdim=True).clamp_min(eps))
    return w

class SlotSetPool(nn.Module):
    """
    Set-transformer-style pooling via cross-attention from K learnable slots.

    Inputs:
      x:    (B, N, D)
      mask: (B, N) 1 real, 0 pad

    Returns:
      y:    (B, out_dim)
      attn: (B, K, N) attention weights (for diagnostics)
    """
    def __init__(self, input_dim, logm_idx=0, H=128, K=8, out_dim=1,
                 phi_hidden=(256, 256), head_hidden=(256, 256), dropout=0.0):
        super().__init__()
        self.logm_idx = logm_idx
        self.H = H
        self.K = K

        # Per-galaxy embedding
        self.phi = MLP(input_dim, list(phi_hidden), H, dropout=dropout)

        # Learnable slot queries
        self.queries = nn.Parameter(torch.randn(K, H) * 0.02)

        # Keys get logM* appended to help mass specialization
        self.key = nn.Linear(H + 1, H)
        self.val = nn.Linear(H, H)

        # Optional: post-attention refinement (tiny MLP per slot)
        self.slot_ff = MLP(H, [H], H, dropout=dropout)

        # Prediction head
        self.head = MLP(K * H, list(head_hidden), out_dim, dropout=dropout)

    def forward(self, x, mask):
        B, N, D = x.shape
        m = mask.float()

        h = self.phi(x)  # (B,N,H)
        logm = x[..., self.logm_idx:self.logm_idx+1]  # (B,N,1)

        k = self.key(torch.cat([h, logm], dim=-1))  # (B,N,H)
        v = self.val(h)                              # (B,N,H)

        # Expand queries to batch: (B,K,H)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: (B,K,N)
        scale = self.H ** 0.5
        logits = (q @ k.transpose(1, 2)) / scale  # (B,K,N)

        # Mask padding galaxies
        logits = logits.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))

        # Pool into slots: (B,K,H)
        slots = attn @ v

        # Small per-slot refinement (residual-ish)
        slots = slots + self.slot_ff(slots)

        y = self.head(slots.reshape(B, self.K * self.H))
        return y, attn
