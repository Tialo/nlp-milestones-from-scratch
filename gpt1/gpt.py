import os
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_causal_mask(n: int):
    return torch.tril(torch.ones(n, n), diagonal=0).type(torch.uint8)


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_len: int = 512
    dropout: float = 0.1
    use_flash_attn: bool = True


class ScaledDotProductAttention(nn.Module):
    def __init__(self, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create mask inside sdpa because attention is always
        # used with causal mask in decoder-only transformer
        self.register_buffer("causal_mask", create_causal_mask(max_len))

    def forward(self, q, k, v):
        """
        q - (batch_size, n_head, seq_len, head_dim)
        k - (batch_size, n_head, seq_len, head_dim)
        v - (batch_size, n_head, seq_len, head_dim)
        """
        d_k = k.size(-1)
        seq_len = k.size(-2)
        attn_weights = q @ k.transpose(-1, -2) / d_k ** 0.5  # (batch_size, n_head, seq_len, seq_len)

        mask = self.causal_mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        attn_weights = torch.masked_fill(attn_weights, mask == 0, float('-inf'))

        attn_probabilities = attn_weights.softmax(dim=-1)
        return attn_probabilities @ v  # (batch_size, n_head, seq_len, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_len, dropout, use_flash_attn):
        super().__init__()

        assert hidden_size % num_attention_heads == 0
        self.n_head = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
    
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.sdpa = ScaledDotProductAttention(max_len, dropout)
        else:
            self.dropout_p = dropout

    def forward(self, x):
        """
        x - (batch_size, seq_len, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_flash_attn:
            sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout_p)  # (batch_size, n_head, seq_len, head_dim)
        else:
            sdpa = self.sdpa(q, k, v)  # (batch_size, n_head, seq_len, head_dim)
        sdpa = sdpa.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.proj(sdpa)  # (batch_size, seq_len, hidden_size)


# https://arxiv.org/pdf/1606.08415
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                (2 / torch.pi) ** 0.5 *
                (x + 0.044715 * x ** 3)
            )
        )


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):
        return self.ffn(x)  # (batch_size, seq_len, hidden_size)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, max_len, dropout, use_flash_attn):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_attention_heads, max_len, dropout, use_flash_attn)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        # used in original implementations
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x - (batch_size, seq_len, hidden_size)
        """
        attention = self.layer_norm1(x + self.dropout1(self.mha(x)))  # (batch_size, seq_len, hidden_size)
        return self.layer_norm2(attention + self.dropout2(self.ffn(attention)))  # (batch_size, seq_len, hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, intermediate_size, max_len, dropout, use_flash_attn):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_attention_heads, intermediate_size, max_len, dropout, use_flash_attn)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        return x  # (batch_size, seq_len, hidden_size)


class GPT(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
    ):
        super().__init__()
        self.config = config
        self.decoder = Decoder(config.hidden_size, config.num_layers, config.num_attention_heads, config.intermediate_size, config.max_len, config.dropout, config.use_flash_attn)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embeddings = nn.Embedding(config.max_len, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(p=config.dropout)

        self.proj.weight = self.embeddings.weight

        self._init_params()

    # matches original implementation
    # https://github.com/openai/finetune-transformer-lm/blob/master/train.py
    def _init_params(self):
        how_many_initialized = 0
        # note: loop only for decoder parameters
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                # (4.1) a simple weight initialization of N (0, 0.02) was sufficient
                nn.init.normal_(module.weight, mean=0, std=0.02)
                how_many_initialized += module.weight.numel()
                # not mentioned in the paper, but used in original implementation
                nn.init.zeros_(module.bias)
                how_many_initialized += module.bias.numel()
            elif isinstance(module, nn.LayerNorm):
                # not mentioned in the paper, but used in original implementation
                nn.init.ones_(module.weight)
                how_many_initialized += module.weight.numel()
                nn.init.zeros_(module.bias)
                how_many_initialized += module.bias.numel()
            # skip high-level modules e.g. DecoderLayer, MultiHeadAttention etc
            # also skip nn.Embedding, because Decoder doesn't have any
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
        how_many_initialized += self.embeddings.weight.numel()
        nn.init.normal_(self.positional_embeddings.weight, mean=0, std=0.02)
        how_many_initialized += self.positional_embeddings.weight.numel()
        # note: don't initialize self.proj.weight
        # because they are tied to self.embeddings.weight, so 
        # they are already initialized.
        nn.init.zeros_(self.proj.bias)
        how_many_initialized += self.proj.bias.numel()

        model_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert how_many_initialized == model_parameters

    def forward(self, x, return_hidden_states: bool = False):
        """
        x - (batch_size, seq_len)
        """
        seq_len = x.size(1)
        embeddings = self.embeddings(x)
        pos_index = torch.arange(seq_len, device=x.device)
        pos_embeddings = self.positional_embeddings(pos_index)  # (seq_len, hidden_size)
        embeddings += pos_embeddings.unsqueeze(0)
        embeddings = self.dropout(embeddings)
        embeddings = self.decoder(embeddings)  # (batch_size, seq_len, hidden_size)
        if return_hidden_states:
            return embeddings
        return self.proj(embeddings)  # (batch_size, seq_len, vocab_size)

    def save_pretrained(self, save_path: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "config.json")) as f:
            config = json.load(f)
        model = cls(GPTConfig(**config))
        state_dict = torch.load(os.path.join(pretrained_path, "model.pt"))
        model.load_state_dict(state_dict)
        return model

    def get_splitted_params_for_opt(self):
        """
        Splits parameters of model into 2 groups:
            * Those which will be influenced by weight_decay
            * Those which won't
        
        "We also employed a modified version of L2 regularization
        proposed in https://arxiv.org/abs/1711.05101
        with w = 0.01 on all non bias or gain weights."
        (gain is layer_norm.weight)
        """
        decay_parameters = []
        no_decay_parameters = []
        # note: same logic as in _init_params
        for module in self.decoder.modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_parameters.append(module.weight)
                no_decay_parameters.append(module.bias)
            elif isinstance(module, nn.Linear):
                decay_parameters.append(module.weight)
                no_decay_parameters.append(module.bias)

        decay_parameters.append(self.embeddings.weight)
        decay_parameters.append(self.positional_embeddings.weight)
        # note: same logic as in _init_params
        no_decay_parameters.append(self.proj.bias)

        n_decay_parameters = sum(p.numel() for p in decay_parameters)
        n_no_decay_parameters = sum(p.numel() for p in no_decay_parameters)
        model_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        assert n_decay_parameters + n_no_decay_parameters == model_parameters
        return no_decay_parameters, decay_parameters


class GPTClassificator(nn.Module):
    def __init__(self, gpt: GPT, pad_token_id: int, n_targets: int = 1, classification_dropout: float = 0.1):
        super().__init__()
        self.gpt = gpt
        self.dropout = nn.Dropout(p=classification_dropout)
        self.classification_head = nn.Linear(gpt.config.hidden_size, n_targets)
        self.pad_token_id = pad_token_id
        self._init_weights()

    def _init_weights(self):
        # self.gpt weights should be already initialized
        how_many_initialized = sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)
        nn.init.normal_(self.classification_head.weight, mean=0, std=0.02)
        nn.init.zeros_(self.classification_head.bias)
        how_many_initialized += self.classification_head.weight.numel()
        how_many_initialized += self.classification_head.bias.numel()

        model_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert how_many_initialized == model_parameters

    def forward(self, x):
        hidden_states = self.gpt(x, return_hidden_states=True)  # (batch_size, seq_len, hidden_size)
        last_content_hidden_state = self._get_last_content_hidden_state(x, hidden_states)  # (batch_size, hidden_size)
        last_content_hidden_state = self.dropout(last_content_hidden_state)
        return self.classification_head(last_content_hidden_state)  # (batch_size, n_targets)

    def _get_last_content_hidden_state(self, inputs, hidden_state):
        mask = (inputs != self.pad_token_id).type(torch.int)  # (batch_size, seq_len)
        content_lengths = mask.sum(dim=1)  # (batch_size,)
        batch_index = torch.arange(inputs.size(0), device=inputs.device)
        seq_index = content_lengths - 1  # otherwise it will get first padding hidden_state
        return hidden_state[batch_index, seq_index]

    def get_splitted_params_for_opt(self):
        no_decay_parameters, decay_parameters = self.gpt.get_splitted_params_for_opt()
        decay_parameters.append(self.classification_head.weight)
        no_decay_parameters.append(self.classification_head.bias)

        n_decay_parameters = sum(p.numel() for p in decay_parameters)
        n_no_decay_parameters = sum(p.numel() for p in no_decay_parameters)
        model_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        assert n_decay_parameters + n_no_decay_parameters == model_parameters
        return no_decay_parameters, decay_parameters


class GPTSimilarity(GPTClassificator):
    def forward(self, x):
        hidden_states = self.gpt(x, return_hidden_states=True)  # (batch_size * 2, seq_len, hidden_size)
        hidden_size = hidden_states.size(-1)

        last_content_hidden_state = self._get_last_content_hidden_state(x, hidden_states)  # (batch_size * 2, hidden_size)
        last_content_hidden_state = last_content_hidden_state.view(-1, 2, hidden_size)  # (batch_size, 2, hidden_state)

        average_hidden_state = last_content_hidden_state.sum(dim=1)  # (batch_size, hidden_state)
        average_hidden_state = self.dropout(average_hidden_state)
        return self.classification_head(average_hidden_state)  # (batch_size, 1)
