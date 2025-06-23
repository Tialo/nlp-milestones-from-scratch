from typing import TYPE_CHECKING
from contextlib import nullcontext

import torch


if TYPE_CHECKING:
    from gpt import GPT


class Generator:
    def __init__(
        self, gpt: "GPT", sos_token_id: int, eos_token_id: int
    ):
        self.gpt = gpt
        self.max_len = gpt.config.max_len
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    @torch.no_grad
    def _get_top_tokens(
        self, generated: torch.Tensor, k: int, autocast: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        generated - (current_seq_len,)
        """
        context_kwargs = {}
        context = nullcontext
        if autocast:
            context_kwargs = {"device_type": generated.device.type, "dtype": torch.bfloat16}
            context = torch.autocast
        with context(**context_kwargs):
            logits = self.gpt(generated.unsqueeze(0))
        logits = logits[0, -1]  # (vocab_size,)
        token_probs = logits.softmax(dim=-1)
        top_tokens = torch.topk(token_probs, k=k, sorted=False, dim=-1)
        return top_tokens.indices, top_tokens.values  # each (k,)

    def generate(
        self, tokens: torch.Tensor, max_tokens: int = 20, top_k: int = 50, autocast: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 2:
            if tokens.size(0) != 1:
                raise ValueError("batch_size > 1 is not supported...")
            tokens = tokens.squeeze(0)

        tokens_until_max_len = self.max_len - len(tokens)
        max_tokens = min(tokens_until_max_len, max_tokens)

        for _ in range(max_tokens):
            topk_indices, topk_probabilities = self._get_top_tokens(tokens, k=top_k, autocast=autocast)
            top_sampled_token_index = torch.multinomial(topk_probabilities, 1)
            top_sampled_token = topk_indices[top_sampled_token_index]
            tokens = torch.cat([tokens, top_sampled_token])
            if top_sampled_token == self.eos_token_id:
                break

        return tokens
