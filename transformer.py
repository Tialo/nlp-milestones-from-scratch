"""
https://arxiv.org/pdf/1706.03762
"""
import math

import torch
import torch.nn as nn


def create_causal_mask(seq_len: int):
    """
    Example:
        >>> create_causal_mask(4)
        tensor([[1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1]], dtype=torch.uint8)
    """
    return torch.tril(torch.ones(seq_len, seq_len), diagonal=0).type(torch.uint8)  # (seq_len, seq_len)


def positional_encoding(seq_len: int, embed_size: int):
    pos_vec = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    i_times_two_vec = torch.arange(0, embed_size, 2)  # (embed_size // 2)
    pos_encoding = torch.empty(seq_len, embed_size)
    div_term = torch.exp(-math.log(10000) * i_times_two_vec / embed_size)
    pos_encoding[:, ::2] = torch.sin(pos_vec * div_term)
    pos_encoding[:, 1::2] = torch.cos(pos_vec * div_term)
    return pos_encoding  # (seq_len, embed_size)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, use_dropout: bool = False):
        super().__init__()
        # was not mentioned in original paper
        p = 0.1 if use_dropout else 0.0
        self.dropout = nn.Dropout(p=p)

    def forward(self, q, k, v, mask=None):
        """
        q - (batch_size, n_heads, seq_len_q, head_dim)
        k - (batch_size, n_heads, seq_len_kv, head_dim)
        v - (batch_size, n_heads, seq_len_kv, head_dim)
        mask - (batch_size, 1, seq_len_q)  in case of mask for encoder self-attention
        or     (batch_size, 1, seq_len_kv) in case of padding mask for encoder-decoder attention
        or     (1, seq_len_kv, seq_len_kv)  in case of decoder causal self-attention

        seq_len_q == seq_len_kv == seq_len_qkv in case of any self-attention
        """
        d_k = k.size(-1)
        k = k.transpose(-1, -2)  # (batch_size, n_heads, head_dim, seq_len_kv)
        attn_weights = q @ k / d_k ** 0.5  # (batch_size, n_heads, seq_len_q, seq_len_kv)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (1, 1, seq_len_qkv, seq_len_qkv) in case of any self-attention
                                      # (batch_size, 1, 1, seq_len) otherwise
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ v  # (batch_size, n_heads, seq_len_q, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, use_dropout_in_spda: bool = False):
        super().__init__()
        assert embed_size % n_heads == 0

        self.n_heads = n_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // n_heads

        self.q_weights = nn.Linear(embed_size, embed_size)
        self.k_weights = nn.Linear(embed_size, embed_size)
        self.v_weights = nn.Linear(embed_size, embed_size)
        self.proj = nn.Linear(embed_size, embed_size)

        self.sdpa = ScaledDotProductAttention(use_dropout=use_dropout_in_spda)

    def forward(self, q, k, v, mask=None):
        """
        q - (batch_size, seq_len_q, embed_size)
        k - (batch_size, seq_len_kv, embed_size)
        v - (batch_size, seq_len_kv, embed_size)
        mask - refer to ScaledDotProductAttention.forward() for details

        seq_len_q == seq_len_kv in case of self-attention
        """
        batch_size = q.size(0)

        q = self.q_weights(q)
        k = self.k_weights(k)
        v = self.v_weights(v)

        # THIS WON'T WORK.
        # q = q.view(batch_size, self.n_heads, q.size(1), self.head_dim)
        # k = k.view(batch_size, self.n_heads, k.size(1), self.head_dim)
        # v = v.view(batch_size, self.n_heads, v.size(1), self.head_dim)

        q = q.view(batch_size, q.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len_q, head_dim)
        k = k.view(batch_size, k.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len_kv, head_dim)
        v = v.view(batch_size, v.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len_kv, head_dim)

        attentions = self.sdpa(q, k, v, mask=mask)  # (batch_size, n_heads, seq_len_q, head_dim)
        # this won't work without .transpose(1, 2).contiguous()
        attentions = attentions.transpose(1, 2).contiguous().view(batch_size, q.size(2), self.embed_size)
        return self.proj(attentions)  # (batch_size, seq_len_q, embed_size)


class FeedForward(nn.Module):
    def __init__(self, d_out: int, d_ff: int, use_dropout: bool = False):
        super().__init__()
        p = 0.1 if use_dropout else 0.0
        self.ff = nn.Sequential(
            nn.Linear(d_out, d_ff),
            nn.ReLU(),
            nn.Dropout(p=p),  # was not mentioned in original paper
            nn.Linear(d_ff, d_out)
        )

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(nn.Module):
    def __init__(self,
        n_heads: int,
        embed_size: int,
        d_ff: int,
        post_ln: bool = True,
        use_dropout_in_ff: bool = False,
        use_dropout_in_sdpa: bool = False,
    ):
        super().__init__()
        self.ff = FeedForward(embed_size, d_ff, use_dropout=use_dropout_in_ff)
        self.mha = MultiHeadAttention(n_heads, embed_size, use_dropout_in_spda=use_dropout_in_sdpa)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        # 5.4 We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.post_ln = post_ln

    def forward(self, x, mask=None):
        """
        x - (batch_size, seq_len_src, embed_size)
        mask - (batch_size, 1, seq_len_src)
        """
        if self.post_ln:
            attention = self.ln1(x + self.dropout1(self.mha(x, x, x, mask)))
            return self.ln2(attention + self.dropout2(self.ff(attention)))  # (batch_size, seq_len_src, embed_size)
        else:
            x_ln = self.ln1(x)
            attention = x + self.dropout1(self.mha(x_ln, x_ln, x_ln, mask))
            attention_ln = self.ln2(attention)
            return attention + self.dropout2(self.ff(attention_ln))  # (batch_size, seq_len_src, embed_size)


class Encoder(nn.Module):
    def __init__(self,
        n_layers: int,
        n_heads: int,
        embed_size: int,
        d_ff: int,
        post_ln: bool = True,
        final_ln: bool = False,
        use_dropout_in_ff: bool = False,
        use_dropout_in_sdpa: bool = False,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(n_heads, embed_size, d_ff, post_ln=post_ln, use_dropout_in_ff=use_dropout_in_ff, use_dropout_in_sdpa=use_dropout_in_sdpa) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_size) if final_ln else None

    def forward(self, x, mask=None):
        """
        x - (batch_size, seq_len_src, embed_size)
        mask - (batch_size, 1, seq_len_src)
        """
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)  # (batch_size, seq_len_src, embed_size)
        if self.final_ln is not None:
            x = self.final_ln(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,
        n_heads: int,
        embed_size: int,
        d_ff: int,
        post_ln: bool = True,
        use_dropout_in_ff: bool = False,
        use_dropout_in_sdpa: bool = False,
    ):
        super().__init__()
        self.ff = FeedForward(embed_size, d_ff, use_dropout=use_dropout_in_ff)
        self.mha = MultiHeadAttention(n_heads, embed_size, use_dropout_in_spda=use_dropout_in_sdpa)
        self.mmha = MultiHeadAttention(n_heads, embed_size, use_dropout_in_spda=use_dropout_in_sdpa)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ln3 = nn.LayerNorm(embed_size)
        # 5.4 We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.post_ln = post_ln

    def forward(self, x, memory, causal_mask, encoder_mask=None):
        """
        x - (batch_size, seq_len_tgt, embed_size)
        memory - (batch_size, seq_len_src, embed_size)
        causal_mask - (1, seq_len_tgt, seq_len_tgt)
        encoder_mask - (batch_size, 1, seq_len_src)
        """
        if self.post_ln:
            attention = self.ln1(x + self.dropout1(self.mmha(x, x, x, mask=causal_mask)))  # (batch_size, seq_len_tgt, embed_size)
            attention = self.ln2(attention + self.dropout2(self.mha(attention, memory, memory, mask=encoder_mask)))
            return self.ln3(attention + self.dropout3(self.ff(attention)))  # (batch_size, seq_len_tgt, embed_size)
        else:
            x_ln = self.ln1(x)
            attention = x + self.dropout1(self.mmha(x_ln, x_ln, x_ln, mask=causal_mask))
            attention_ln = self.ln2(attention)
            attention = attention + self.dropout2(self.mha(attention_ln, memory, memory, mask=encoder_mask))
            attention_ln = self.ln3(attention)
            return attention + self.dropout3(self.ff(attention_ln))  # (batch_size, seq_len_tgt, embed_size)


class Decoder(nn.Module):
    def __init__(self,
        n_layers: int,
        n_heads: int,
        embed_size: int,
        d_ff: int,
        post_ln: bool = True,
        final_ln: bool = False,
        use_dropout_in_ff: bool = False,
        use_dropout_in_sdpa: bool = False,
    ):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(n_heads, embed_size, d_ff, post_ln=post_ln, use_dropout_in_ff=use_dropout_in_ff, use_dropout_in_sdpa=use_dropout_in_sdpa) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_size) if final_ln else None

    def forward(self, x, memory, encoder_mask=None):
        """
        x - (batch_size, seq_len_tgt, embed_size)
        memory - (batch_size, seq_len_src, embed_size)
        encoder_mask - (batch_size, 1, seq_len_src)
        """
        tgt_mask = create_causal_mask(
            x.size(1),
        ).unsqueeze(0).to(x.device)  # (1, seq_len_tgt, seq_len_tgt)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, tgt_mask, encoder_mask=encoder_mask)  # (batch_size, seq_len_tgt, embed_size)
        if self.final_ln is not None:
            x = self.final_ln(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_encoder_heads: int = 8,
        n_decoder_heads: int = 8,
        embed_size: int = 512,
        d_ff: int = 2048,
        max_len: int = 4096,
        tie_embeddings: bool = True,
        post_ln: bool = True,
        add_two_layer_norms: bool = False,
        use_dropout_in_ff: bool = False,
        use_dropout_in_sdpa: bool = False,
        xavier_initialization: bool = False,
    ):
        super().__init__()
        # compute positional encoding for max_len once to save time for each forward pass
        self.pos_enc = positional_encoding(max_len, embed_size)
        self.encoder = Encoder(
            n_encoder_layers,
            n_encoder_heads,
            embed_size,
            d_ff,
            post_ln=post_ln,
            final_ln=add_two_layer_norms,
            use_dropout_in_ff=use_dropout_in_ff,
            use_dropout_in_sdpa=use_dropout_in_sdpa,
        )
        self.decoder = Decoder(
            n_decoder_layers,
            n_decoder_heads,
            embed_size,
            d_ff,
            post_ln=post_ln,
            final_ln=add_two_layer_norms,
            use_dropout_in_ff=use_dropout_in_ff,
            use_dropout_in_sdpa=use_dropout_in_sdpa,
        )

        self.sqrt_dmodel = embed_size ** 0.5
        # original paper used shared embedding layer for source and target
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, tgt_vocab_size)

        # 3.4 In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation
        # see also https://paperswithcode.com/method/weight-tying
        if tie_embeddings:
            self.tgt_embedding.weight = self.fc.weight

        # 5.4 we apply dropout to the sums of the embeddings and the positional encodings
        self.dropout = nn.Dropout(p=0.1)
        if xavier_initialization:
            self._init_params()

    def _init_params(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, src, src_mask=None):
        """
        src - (batch_size, seq_len_src)
        src_mask - (batch_size, seq_len_src)
        """
        if src_mask is not None:
            src_mask = src_mask[:, torch.newaxis]  # (batch_size, 1, seq_len_src)
        src_embed = self.src_embedding(src)  # (batch_size, seq_len_src, embed_size)
        # 3.4 In the embedding layers, we multiply those weights by √dmodel
        src_embed *= self.sqrt_dmodel
        src_embed += self.pos_enc[:src.size(1)].to(src_embed.device)  # (batch_size, seq_len_src, embed_size)
        src_embed = self.dropout(src_embed)  # (batch_size, seq_len_src, embed_size)
        memory = self.encoder(src_embed, mask=src_mask)  # (batch_size, seq_len_src, embed_size)
        return memory, src_mask

    def _decode(self, memory, tgt, src_mask=None):
        """
        memory - (batch_size, seq_len_src, embed_size)
        tgt - (batch_size, seq_len_tgt)
        src_mask - (batch_size, seq_len_src)
        """
        tgt_embed = self.tgt_embedding(tgt)  # (batch_size, seq_len_tgt, embed_size)
        # 3.4 In the embedding layers, we multiply those weights by √dmodel
        tgt_embed *= self.sqrt_dmodel
        tgt_embed += self.pos_enc[:tgt.size(1)].to(tgt_embed.device)
        tgt_embed = self.dropout(tgt_embed)
        attention = self.decoder(tgt_embed, memory, encoder_mask=src_mask)  # (batch_size, seq_len_tgt, embed_size)
        return self.fc(attention)  # (batch_size, seq_len_tgt, vocab_size)

    def forward(self, src, tgt, src_mask=None):
        """
        src - (batch_size, seq_len_src)
        tgt - (batch_size, seq_len_tgt)
        src_mask - (batch_size, seq_len_src)
        """
        memory, src_mask = self._encode(src, src_mask)
        return self._decode(memory, tgt, src_mask)  # (batch_size, seq_len_tgt, vocab_size)

    @torch.no_grad
    def generate(self, src, start_token: int, eos_token: int, max_tokens: int = 20):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        elif src.size(0) != 1:
            raise ValueError("batch_size > 1 is not supported...")
        memory, _ = self._encode(src)
        generated = [start_token]

        for _ in range(max_tokens):
            logits = self._decode(memory, torch.tensor([generated]).to(src.device))
            # original paper used beam search
            generated_token = logits[:, -1].argmax()

            if generated_token == eos_token:
                break

            generated.append(generated_token.item())

        return torch.tensor(generated)
