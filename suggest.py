"""Ask the model for its best few guesses at the next shape (beam search that only allows valid shapes)."""
import torch
from tokens import *


def allowed_mask(dev):
    """mask[kind] -> which tokens may come next. kind 0 = a shape type (or 'finished'), kind 1 = a coordinate."""
    m = torch.full((2, VOCAB), float("-inf"), device=dev)
    m[0, FIRST_TYPE:LAST_TYPE + 1] = 0
    m[0, EOS] = 0
    m[1, COORD0:] = 0
    return m


@torch.no_grad()
def next_shapes(model, prefix, width=5):
    """prefix: LongTensor [B, T] (BOS + whole shapes, every row the same length).
    Returns, per row, a list of (probability, [tokens of the suggested shape]) best first.
    A suggestion of [EOS] means "the sketch looks finished"."""
    dev = prefix.device
    B, T = prefix.shape
    mask = allowed_mask(dev)
    need = torch.zeros(VOCAB, dtype=torch.long, device=dev)          # coordinates that follow each type token
    for t in range(FIRST_TYPE, LAST_TYPE + 1):
        need[t] = n_coords(t)
    seq = prefix[:, None, :]                                         # [B, beams, T]
    logp = torch.zeros(B, 1, device=dev)
    left = torch.full((B, 1), -1, device=dev)                        # -1 = type not chosen yet, 0 = shape complete
    for step in range(7):
        K = seq.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(seq.reshape(B * K, -1)[:, -BLOCK:])[:, -1].float()
        lp = torch.log_softmax(logits + mask[(left.reshape(-1) > 0).long()], -1).view(B, K, VOCAB)
        done = left == 0
        lp = torch.where(done[..., None], torch.full_like(lp, float("-inf")), lp)
        lp[..., PAD] = torch.where(done, torch.zeros_like(logp), lp[..., PAD])   # finished beams wait, at no cost
        total = (logp[..., None] + lp).view(B, K * VOCAB)
        logp, pick = total.topk(min(width, K * VOCAB), -1)
        beam, tok = pick // VOCAB, pick % VOCAB
        seq = torch.cat([torch.gather(seq, 1, beam[..., None].expand(-1, -1, seq.shape[-1])), tok[..., None]], -1)
        old = torch.gather(left, 1, beam)
        left = torch.where(old == -1, need[tok], torch.where(old > 0, old - 1, old))   # EOS has need 0 -> complete
        if (left == 0).all():
            break
    out = []
    for b in range(B):
        rows = []
        for k in range(seq.shape[1]):
            if torch.isfinite(logp[b, k]):
                rows.append((float(logp[b, k].exp()), [int(t) for t in seq[b, k, T:] if int(t) != PAD]))
        out.append(rows)
    return out
