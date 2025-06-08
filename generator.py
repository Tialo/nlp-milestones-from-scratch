import heapq
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from transformer import Transformer


class Generator:
    def __init__(self, transformer: "Transformer", sos_token_id: int, eos_token_id: int):
        self.transformer = transformer
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
    
    @torch.no_grad
    def _get_top_tokens(self, src: torch.Tensor, memory: torch.Tensor, generated: list[int], n_beams: int) -> list[tuple[int, float]]:
        logits = self.transformer.decode(memory, torch.tensor([generated]).to(src.device))  # (1, len(generated), vocab_size)
        logits = logits[0, -1]  # (vocab_size, )
        token_probs = logits.log_softmax(dim=0)
        top_tokens = torch.topk(token_probs, k=n_beams, sorted=False)
        return list(zip(top_tokens.indices.tolist(), top_tokens.values.tolist()))

    @staticmethod
    def _get_normalized_score(beam: tuple[list[int], float], alpha: float = 0.6) -> float:
        # 6.1 We used beam search with a beam size of 4 and length penalty α = 0.6
        return beam[1] * 6  ** alpha / (5 + len(beam[0])) ** alpha

    def generate(self, src, max_tokens: int = 20, n_beams: int = 5) -> torch.Tensor:
        if src.dim() == 1:
            src = src.unsqueeze(0)
        elif src.size(0) != 1:
            raise ValueError("batch_size > 1 is not supported...")
        with torch.no_grad():
            memory, _ = self.transformer.encode(src)
        beams: list[tuple[list[int], float]] = [([self.sos_token_id], 0.0)]

        for _ in range(max_tokens):
            candidate_list = []
            if all([generated[-1] == self.eos_token_id for generated, _ in beams]):
                break
            for generated, probability in beams:
                if generated[-1] == self.eos_token_id:
                    candidate_list.append((generated, probability))
                    continue
                for top_token, proba in self._get_top_tokens(src, memory, generated, n_beams):
                    candidate_list.append((
                        generated + [top_token],
                        probability + proba,
                    ))
            beams = heapq.nlargest(n_beams, candidate_list, key=self._get_normalized_score)

        [best_generation, _] = max(beams, key=self._get_normalized_score)
        return torch.tensor(best_generation)
