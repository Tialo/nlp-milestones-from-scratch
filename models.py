import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v):
    """
    q - (batch_size, seq_len, d_q)
    k - (batch_size, seq_len, d_k)
    v - (batch_size, seq_len, d_v)

    d_q == d_k
    """
    d_k = k.size(-1)
    attn_weights = q @ k.transpose(1, 2) / d_k ** 0.5  # (batch_size, seq_len, seq_len)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return attn_weights @ v  # (batch_size, seq_len, d_v)


class AttentionBlock(nn.Module):
    def __init__(self, d_x, d_qk, d_v):
        super().__init__()
        self.q_weights = nn.Linear(d_x, d_qk)
        self.k_weights = nn.Linear(d_x, d_qk)
        self.v_weights = nn.Linear(d_x, d_v)

    def forward(self, x):
        # x - (batch_size, seq_len, d_x)
        q = self.q_weights(x)
        k = self.k_weights(x)
        v = self.v_weights(x)
        return scaled_dot_product_attention(q, k, v)  # (batch_size, seq_len, d_v)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, n_heads, d_x, d_qk, d_v, d_out):
        super().__init__()
        self.concat_head = nn.Linear(n_heads * d_v, d_out)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_x, d_qk, d_v) for _ in range(n_heads)
        ])

    def forward(self, x):
        # x - (batch_size, seq_len, d_x)
        batch_size = x.size(0)
        seq_len = x.size(1)
        attentions = [  # (n_heads, batch_size, seq_len, d_v)
            attention_block(x) for attention_block in self.attention_blocks
        ]
        attentions = torch.concat(attentions).reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, n_heads * d_v)
        return self.concat_head(attentions)  # (batch_size, seq_len, d_out)


class FeedForward(nn.Module):
    def __init__(self, d_out, d_ff):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_out, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_out)
        )

    def forward(self, x):
        return self.ff(x)


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, d_x, d_qk, d_v, d_out, d_ff):
        super().__init__()
        self.ff = FeedForward(d_out, d_ff)
        self.mha = MultiHeadAttentionBlock(n_heads, d_x, d_qk, d_v, d_out)
        self.ln1 = nn.LayerNorm(d_out)
        self.ln2 = nn.LayerNorm(d_out)

    def forward(self, x):
        attention = self.ln1(self.mha(x) + x)  # (batch_size, seq_len, d_out)
        return self.ln2(self.ff(attention) + attention)  # (batch_size, seq_len, d_out)
