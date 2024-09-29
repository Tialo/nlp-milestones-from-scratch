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
        q - (batch_size, n_heads, seq_len_q, d_qk)
        k - (batch_size, n_heads, seq_len_kv, d_qk)
        v - (batch_size, n_heads, seq_len_kv, d_v)
        """
        d_k = k.size(-1)
        attn_weights = q @ k.transpose(-1, -2) / d_k ** 0.5  # (batch_size, n_heads, seq_len_q, seq_len_kv)
        if self.mask:
            # seq_len_q == seq_len_kv
            mask = torch.triu(torch.ones(q.size(-2), q.size(-2)), diagonal=1).type(torch.bool)  #  (seq_len_q, seq_len_q)
            neg_inf = torch.tensor(float('-inf'))
            attn_weights = torch.where(mask, neg_inf, attn_weights)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights @ v  # (batch_size, n_heads, seq_len_kv, d_v)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_size, mask=False):
        super().__init__()
        assert embed_size % n_heads == 0

        self.n_heads = n_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // n_heads

        self.proj = nn.Linear(embed_size, embed_size)
        self.q_weights = nn.Linear(embed_size, embed_size)
        self.k_weights = nn.Linear(embed_size, embed_size)
        self.v_weights = nn.Linear(embed_size, embed_size)

        self.sdpa = ScaledDotProductAttention(mask)

    def forward(self, q, k, v):
        """
        q - (batch_size, seq_len_q, embed_size)
        k - (batch_size, seq_len_k, embed_size)
        v - (batch_size, seq_len_v, embed_size)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)

        q = self.q_weights(q)
        k = self.k_weights(k)
        v = self.v_weights(v)

        q = q.view(batch_size, self.n_heads, q.size(1), self.head_dim)
        k = k.view(batch_size, self.n_heads, k.size(1), self.head_dim)
        v = v.view(batch_size, self.n_heads, v.size(1), self.head_dim)

        attentions = self.sdpa(q, k, v)  # (batch_size, n_heads, seq_len_v, head_dim)
        # .contiguous() often used here
        return self.proj(attentions.view(batch_size, seq_len_q, self.embed_size))  # (batch_size, seq_len_q, embed_size)


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
    def __init__(self, n_heads, embed_size, d_ff):
        super().__init__()
        self.ff = FeedForward(embed_size, d_ff)
        self.mha = MultiHeadAttention(n_heads, embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attention = self.ln1(self.mha(x, x, x) + x)  # (batch_size, seq_len, embed_size)
        return self.ln2(self.ff(attention) + attention)  # (batch_size, seq_len, embed_size)


class Encoder(nn.Module):
    def __init__(self, n_blocks, n_heads, embed_size, d_ff):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_heads, embed_size, d_ff) for _ in range(n_blocks)
        ])

    def forward(self, x):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, embed_size, d_ff):
        super().__init__()
        self.ff = FeedForward(embed_size, d_ff)
        self.mha = MultiHeadAttention(n_heads, embed_size)
        self.mmha = MultiHeadAttention(n_heads, embed_size, mask=True)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ln3 = nn.LayerNorm(embed_size)

    def forward(self, x, memory):
        attention = self.ln1(self.mmha(x, x, x) + x)
        attention = self.ln2(self.mha(attention, memory, memory) + attention)
        return self.ln3(self.ff(attention) + attention)


class Decoder(nn.Module):
    def __init__(self, n_blocks, n_heads, embed_size, d_ff):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_heads, embed_size, d_ff) for _ in range(n_blocks)
        ])

    def forward(self, x, memory):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, memory)
        return x
