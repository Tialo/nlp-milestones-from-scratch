"""
https://arxiv.org/pdf/1706.03762
"""
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, q, k, v):
        """
        q - (batch_size, seq_len_q, d_qk)
        k - (batch_size, seq_len_kv, d_qk)
        v - (batch_size, seq_len_kv, d_v)
        """
        d_k = k.size(-1)
        attn_weights = q @ k.transpose(1, 2) / d_k ** 0.5  # (batch_size, seq_len_q, seq_len_kv)
        if self.mask:
            # seq_len_q == seq_len_kv
            mask = torch.triu(torch.ones(q.size(-2), q.size(-2)), diagonal=1).type(torch.bool)  #  (seq_len_q, seq_len_q)
            neg_inf = torch.tensor(float('-inf'))
            attn_weights = torch.where(mask, neg_inf, attn_weights)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights @ v  # (batch_size, seq_len_kv, d_v)


class AttentionBlock(nn.Module):
    def __init__(self, d_x, d_qk, d_v, mask=False):
        super().__init__()
        self.q_weights = nn.Linear(d_x, d_qk)
        self.k_weights = nn.Linear(d_x, d_qk)
        self.v_weights = nn.Linear(d_x, d_v)
        self.sdpa = ScaledDotProductAttention(mask)

    def forward(self, x):
        # x - (batch_size, seq_len, d_x)
        q = self.q_weights(x)
        k = self.k_weights(x)
        v = self.v_weights(x)
        return self.sdpa(q, k, v)  # (batch_size, seq_len, d_v)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, n_heads, d_x, d_qk, d_v, d_out, mask=False):
        super().__init__()
        self.concat_head = nn.Linear(n_heads * d_v, d_out)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_x, d_qk, d_v, mask) for _ in range(n_heads)
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


class Encoder(nn.Module):
    def __init__(self, n_blocks, n_heads, d_x, d_qk, d_v, d_out, d_ff):
        super().__init__()
        self.encoder_blocks = nn.Sequential(*[
            EncoderBlock(n_heads, d_x, d_qk, d_v, d_out, d_ff) for _ in range(n_blocks)
        ])

    def forward(self, x):
        return self.encoder_blocks(x)
