"""A small GPT, written out in full: it reads the shapes drawn so far and guesses the next token."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from tokens import VOCAB, BLOCK


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = drop

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)
        h = self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x + F.dropout(h, self.drop, self.training)


class SketchGPT(nn.Module):
    def __init__(self, d=256, layers=6, heads=4, drop=0.0):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, drop=drop)
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(BLOCK, d)
        self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))
