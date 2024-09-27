import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v):
        """
        q - (batch_size, seq_len, d_q)
        k - (batch_size, seq_len, d_k)
        v - (batch_size, seq_len, d_v)

        d_q == d_k
        """
        d_k = k.shape[-1]
        attn_weights = q @ k.transpose(1, 2) / d_k ** 0.5  # (batch_size, seq_len, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights[:, :, :, torch.newaxis]  # (batch_size, seq_len, seq_len, 1)
        v = v[:, :, torch.newaxis, :]  # (batch_size, seq_len, 1, d_v)
        attn_weights = attn_weights * v  # (batch_size, seq_len, seq_len, d_v)
        return attn_weights.sum(dim=-2)
