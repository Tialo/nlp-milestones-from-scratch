"""
https://arxiv.org/pdf/1706.03762
"""
import math

import torch
import torch.nn as nn


def positional_encoding(seq_len, embed_size):
    pos_vec = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    i_times_two_vec = torch.arange(0, embed_size, 2)  # (embed_size // 2)
    pos_encoding = torch.empty(seq_len, embed_size)
    div_term = torch.exp(-math.log(10000) * i_times_two_vec / embed_size)
    pos_encoding[:, ::2] = torch.sin(pos_vec * div_term)
    pos_encoding[:, 1::2] = torch.cos(pos_vec * div_term)
    return pos_encoding


class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
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
        attn_weights = self.dropout(attn_weights)
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
            nn.Dropout(p=0.1),
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
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attention = self.ln1(self.dropout(self.mha(x, x, x)) + x)  # (batch_size, seq_len, embed_size)
        return self.ln2(self.dropout(self.ff(attention)) + attention)  # (batch_size, seq_len, embed_size)


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
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, memory):
        attention = self.ln1(self.dropout(self.mmha(x, x, x)) + x)
        attention = self.ln2(self.dropout(self.mha(attention, memory, memory)) + attention)
        return self.ln3(self.dropout(self.ff(attention)) + attention)


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


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_encoder_blocks=6,
        n_decoder_block=6,
        n_encoder_heads=8,
        n_decoder_heads=8,
        embed_size=512,
        d_ff=2048,
        max_len=4096,
    ):
        super().__init__()
        self.pos_enc = positional_encoding(max_len, embed_size)
        self.encoder = Encoder(n_encoder_blocks, n_encoder_heads, embed_size, d_ff)
        self.decoder = Decoder(n_decoder_block, n_decoder_heads, embed_size, d_ff)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.proj = nn.Linear(embed_size, vocab_size)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, tgt):
        """
        src - (batch_size, seq_len_src)
        tgt - (batch_size, seq_len_tgt)
        """
        src_embed = self.embedding(src) + self.pos_enc[:src.size(1)]  # (batch_size, seq_len_src, embed_size)
        src_embed = self.dropout(src_embed)
        memory = self.encoder(src_embed)
        tgt_embed = self.embedding(tgt) + self.pos_enc[:tgt.size(1)]  # (batch_size, seq_len_tgt, embed_size)
        tgt_embed = self.dropout(tgt_embed)
        attention = self.decoder(tgt_embed, memory)
        return self.proj(attention)

    @torch.no_grad
    def generate(self, src, start_token, eos_token, max_tokens=20):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        memory = self.encoder(self.embedding(src))
        generated = [start_token]

        for _ in range(max_tokens):
            embeds = self.embedding(torch.tensor([generated]))
            attention = self.decoder(embeds, memory)
            logits = self.proj(attention)
            generated_token = logits[:, -1].argmax()

            if generated_token == eos_token:
                break

            generated.append(generated_token.item())
        return torch.tensor(generated)
